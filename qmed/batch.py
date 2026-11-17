"""Batch molecular screening, CSV validation, capped active-space VQE execution, and SQLite batch persistence."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from qmed.active_space import estimate_qubits_from_smiles, prepare_active_space_problem
from qmed.benchmark import classify_advantage_verdict, classify_reliability
from qmed.classical import run_classical_reference
from qmed.config import (
    BATCH_CAPPED_ACTIVE_SPACE,
    BATCH_CUSTOM_GEOMETRY_CAPTION,
    BATCH_DEFAULT_MAX_SIMULATED,
    BATCH_DEFAULT_MOLECULE_TIMEOUT_SEC,
    BATCH_DEFAULT_TOTAL_BUDGET_SEC,
    BATCH_GEOMETRY_SEED,
    BATCH_MAX_ROWS,
    DEFAULT_BASIS,
    DEFAULT_BENCHMARK_SEEDS,
    DEFAULT_DB_PATH,
    DEFAULT_SHOTS,
    SUPPORTED_ELEMENTS,
)
from qmed.molecules import smiles_to_3d_coordinates, validate_and_describe
from qmed.qee import evaluate_eligibility
from qmed.storage import get_connection, init_db
from qmed.vqe_runner import run_vqe_from_prepared

SAMPLE_BATCH_CSV_CONTENT: str = """name,smiles,charge,spin
Hydrogen,[H][H],0,0
Water,O,0,0
Helium,[He],0,0
Ammonia,N,0,0
Aspirin,CC(=O)OC1=CC=CC=C1C(=O)O,0,0
Water_Duplicate,O,0,0
Methyl_Radical,[CH3],0,1
Hydroxide_Ion,[OH-],-1,0
Chloroethane,CCCl,0,0
Malformed_Input,not_a_valid_smiles,0,0
"""


@dataclass(frozen=True)
class ValidatedCSVRow:
    """Single validated and deduplicated input row from a batch CSV."""

    row_index: int
    name: str
    input_smiles: str
    canonical_smiles: str
    charge: int
    spin: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the validated row."""
        return asdict(self)


@dataclass(frozen=True)
class InvalidCSVRow:
    """Single rejected/invalid row discovered during CSV parsing."""

    row_index: int
    name: str
    input_smiles: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the invalid row."""
        return asdict(self)


@dataclass(frozen=True)
class BatchCSVValidationResult:
    """Result of validating a batch CSV upload."""

    is_valid: bool
    total_rows: int
    valid_rows: List[ValidatedCSVRow] = field(default_factory=list)
    invalid_rows: List[InvalidCSVRow] = field(default_factory=list)
    duplicates_removed: int = 0
    duplicate_rows: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the CSV validation summary."""
        return asdict(self)


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for a batch screening run."""

    seeds: Tuple[int, ...] = (42, 43, 44)
    geometry_seed: int = BATCH_GEOMETRY_SEED
    capped_active_space: Tuple[int, int] = BATCH_CAPPED_ACTIVE_SPACE
    run_noisy: bool = False
    noisy_mode: str = "depolarizing_readout"
    shots: int = DEFAULT_SHOTS
    ideal_maxiter: int = 25
    noisy_maxiter: int = 10
    max_simulated_molecules: int = BATCH_DEFAULT_MAX_SIMULATED
    per_molecule_timeout_sec: float = BATCH_DEFAULT_MOLECULE_TIMEOUT_SEC
    total_time_budget_sec: float = BATCH_DEFAULT_TOTAL_BUDGET_SEC
    resume: bool = True

    def to_canonical_json(self) -> str:
        """Serialize simulation-affecting parameters to a canonical JSON string for resume matching."""
        sim_dict = {
            "seeds": list(self.seeds),
            "geometry_seed": int(self.geometry_seed),
            "capped_active_space": list(self.capped_active_space),
            "run_noisy": bool(self.run_noisy),
            "noisy_mode": str(self.noisy_mode),
            "shots": int(self.shots) if self.run_noisy else None,
            "ideal_maxiter": int(self.ideal_maxiter),
            "noisy_maxiter": int(self.noisy_maxiter) if self.run_noisy else None,
        }
        return json.dumps(sim_dict, sort_keys=True)


@dataclass(frozen=True)
class BatchItemResult:
    """Processed result for a single molecule row in a batch run."""

    row_index: int
    name: str
    input_smiles: str
    canonical_smiles: Optional[str]
    charge: int
    spin: int
    seed: int
    descriptors: Optional[Dict[str, Any]]
    qubits: Optional[int]
    depth: Optional[int]
    gate_count: Optional[int]
    active_space: Optional[str]
    qee_decision: str
    reasons: List[str]
    warnings: List[str]
    outcome: str
    hf_energy: Optional[float] = None
    fci_energy: Optional[float] = None
    ideal_vqe_energy: Optional[float] = None
    ideal_error_mha: Optional[float] = None
    ideal_uncertainty_mha: Optional[float] = None
    noisy_vqe_energy: Optional[float] = None
    noisy_error_mha: Optional[float] = None
    noisy_uncertainty_mha: Optional[float] = None
    active_space_error_mha: Optional[float] = None
    reliability: Optional[str] = None
    verdict: Optional[str] = None
    geometry_caption: Optional[str] = None
    runtime_sec: float = 0.0
    was_resumed: bool = False
    db_item_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the batch item result."""
        return asdict(self)


@dataclass(frozen=True)
class BatchRunResult:
    """Complete result of a batch screening run."""

    batch_id: int
    status: str
    config: Dict[str, Any]
    validation: BatchCSVValidationResult
    items: List[BatchItemResult]
    simulated_count: int
    resumed_count: int
    total_runtime_sec: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert processed batch items into a flat pandas DataFrame for UI display and CSV export."""
        rows: List[Dict[str, Any]] = []
        for item in self.items:
            desc = item.descriptors or {}
            rows.append(
                {
                    "Row": item.row_index,
                    "Name": item.name,
                    "Input SMILES": item.input_smiles,
                    "Canonical SMILES": item.canonical_smiles or "—",
                    "Formula": desc.get("formula", "—"),
                    "MW": desc.get("mw"),
                    "QED": desc.get("qed"),
                    "QEE Decision": item.qee_decision,
                    "Outcome": item.outcome,
                    "Active Space": item.active_space or "—",
                    "Qubits": item.qubits,
                    "Est. Depth": item.depth,
                    "Est. Gates": item.gate_count,
                    "HF Energy (Ha)": round(item.hf_energy, 6) if item.hf_energy is not None else None,
                    "FCI/CASCI Energy (Ha)": round(item.fci_energy, 6) if item.fci_energy is not None else None,
                    "Ideal VQE (Ha)": round(item.ideal_vqe_energy, 6) if item.ideal_vqe_energy is not None else None,
                    "Ideal |Error| (mHa)": round(item.ideal_error_mha, 4) if item.ideal_error_mha is not None else None,
                    "Ideal ±σ (mHa)": round(item.ideal_uncertainty_mha, 4) if item.ideal_uncertainty_mha is not None else None,
                    "Noisy VQE (Ha)": round(item.noisy_vqe_energy, 6) if item.noisy_vqe_energy is not None else None,
                    "Noisy |Error| (mHa)": round(item.noisy_error_mha, 4) if item.noisy_error_mha is not None else None,
                    "AS Error (mHa)": round(item.active_space_error_mha, 4) if item.active_space_error_mha is not None else None,
                    "Reliability": item.reliability or "—",
                    "Verdict": item.verdict or "—",
                    "Runtime (s)": round(item.runtime_sec, 3),
                    "Reasons": "; ".join(item.reasons),
                    "Geometry Note": item.geometry_caption or "—",
                }
            )
        return pd.DataFrame(rows)


def get_sample_batch_csv() -> str:
    """Return the sample CSV string for batch screening download/testing."""
    return SAMPLE_BATCH_CSV_CONTENT


def validate_batch_csv(
    csv_source: Union[str, bytes, Path, pd.DataFrame],
    max_rows: int = BATCH_MAX_ROWS,
) -> BatchCSVValidationResult:
    """Pure function to validate a batch CSV, enforce row limits, deduplicate SMILES, and list invalid rows."""
    try:
        if isinstance(csv_source, pd.DataFrame):
            raw_rows = csv_source.to_dict(orient="records")
            columns = [str(c).strip() for c in csv_source.columns]
        else:
            if isinstance(csv_source, Path):
                text = csv_source.read_text(encoding="utf-8")
            elif isinstance(csv_source, bytes):
                text = csv_source.decode("utf-8", errors="replace")
            elif isinstance(csv_source, str):
                text = csv_source
            else:
                return BatchCSVValidationResult(
                    is_valid=False,
                    total_rows=0,
                    error="Unsupported CSV input type.",
                )

            if not text.strip():
                return BatchCSVValidationResult(
                    is_valid=False,
                    total_rows=0,
                    error="CSV input is empty.",
                )

            reader = csv.DictReader(io.StringIO(text.strip()))
            columns = [str(c).strip() for c in (reader.fieldnames or []) if c is not None]
            raw_rows = list(reader)
    except Exception as exc:
        return BatchCSVValidationResult(
            is_valid=False,
            total_rows=0,
            error=f"Failed to parse CSV: {exc}",
        )

    col_map = {c.lower(): c for c in columns}
    if "smiles" not in col_map:
        return BatchCSVValidationResult(
            is_valid=False,
            total_rows=len(raw_rows),
            error="Missing required 'smiles' column in CSV header.",
        )

    total_rows = len(raw_rows)
    if total_rows == 0:
        return BatchCSVValidationResult(
            is_valid=False,
            total_rows=0,
            error="CSV contains header only (0 data rows).",
        )

    if total_rows > max_rows:
        return BatchCSVValidationResult(
            is_valid=False,
            total_rows=total_rows,
            error=f"CSV contains {total_rows} rows, which exceeds the maximum limit of {max_rows} rows.",
        )

    smiles_col = col_map["smiles"]
    name_col = col_map.get("name")
    charge_col = col_map.get("charge")
    spin_col = col_map.get("spin")

    valid_rows: List[ValidatedCSVRow] = []
    invalid_rows: List[InvalidCSVRow] = []
    duplicate_rows: List[Dict[str, Any]] = []
    seen_canonical: set[str] = set()

    for idx, row in enumerate(raw_rows, start=1):
        raw_smiles = str(row.get(smiles_col) or "").strip()
        raw_name = str(row.get(name_col) or "").strip() if name_col else ""
        display_name = raw_name if raw_name else (raw_smiles or f"Row_{idx}")

        if not raw_smiles:
            invalid_rows.append(
                InvalidCSVRow(
                    row_index=idx,
                    name=display_name,
                    input_smiles="",
                    reason="Empty SMILES value.",
                )
            )
            continue

        # Parse optional charge and spin
        raw_charge = row.get(charge_col) if charge_col else None
        raw_spin = row.get(spin_col) if spin_col else None
        try:
            charge_val = (
                int(str(raw_charge).strip())
                if raw_charge is not None and str(raw_charge).strip() != ""
                else 0
            )
        except ValueError:
            invalid_rows.append(
                InvalidCSVRow(
                    row_index=idx,
                    name=display_name,
                    input_smiles=raw_smiles,
                    reason=f"Invalid integer charge '{raw_charge}'.",
                )
            )
            continue

        try:
            spin_val = (
                int(str(raw_spin).strip())
                if raw_spin is not None and str(raw_spin).strip() != ""
                else 0
            )
        except ValueError:
            invalid_rows.append(
                InvalidCSVRow(
                    row_index=idx,
                    name=display_name,
                    input_smiles=raw_smiles,
                    reason=f"Invalid integer spin '{raw_spin}'.",
                )
            )
            continue

        mol = Chem.MolFromSmiles(raw_smiles, sanitize=True)
        if mol is None or mol.GetNumAtoms() == 0:
            invalid_rows.append(
                InvalidCSVRow(
                    row_index=idx,
                    name=display_name,
                    input_smiles=raw_smiles,
                    reason=f"Invalid SMILES: '{raw_smiles}' could not be parsed by RDKit.",
                )
            )
            continue

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        if canonical_smiles in seen_canonical:
            duplicate_rows.append(
                {
                    "row_index": idx,
                    "name": display_name,
                    "input_smiles": raw_smiles,
                    "canonical_smiles": canonical_smiles,
                }
            )
            continue

        seen_canonical.add(canonical_smiles)
        valid_rows.append(
            ValidatedCSVRow(
                row_index=idx,
                name=display_name,
                input_smiles=raw_smiles,
                canonical_smiles=canonical_smiles,
                charge=charge_val,
                spin=spin_val,
            )
        )

    return BatchCSVValidationResult(
        is_valid=True,
        total_rows=total_rows,
        valid_rows=valid_rows,
        invalid_rows=invalid_rows,
        duplicates_removed=len(duplicate_rows),
        duplicate_rows=duplicate_rows,
        error=None,
    )


def _simulate_capped_vqe_on_rdkit_geometry(
    name: str,
    canonical_smiles: str,
    act_e: int,
    act_o: int,
    config: BatchConfig,
    stage_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run PySCF HF/CASCI and Ideal (+ optional Noisy) VQE on the same seeded RDKit 3D geometry."""
    if stage_cb:
        stage_cb("Generating seeded RDKit 3D geometry")
    atoms, _, _, _ = smiles_to_3d_coordinates(canonical_smiles, seed=config.geometry_seed)

    if stage_cb:
        stage_cb(f"Computing PySCF HF & CASCI({act_e}e, {act_o}o) reference")
    matched_ref = run_classical_reference(
        atoms=atoms,
        basis=DEFAULT_BASIS,
        charge=0,
        spin=0,
        frozen_core=True,
        active_space=(act_e, act_o),
    )
    full_fc_ref = run_classical_reference(
        atoms=atoms,
        basis=DEFAULT_BASIS,
        charge=0,
        spin=0,
        frozen_core=True,
        active_space=None,
    )
    active_space_error_mha = (matched_ref.fci_energy - full_fc_ref.fci_energy) * 1000.0

    if stage_cb:
        stage_cb(f"Preparing qubit Hamiltonian for ({act_e}e, {act_o}o)")
    prepared = prepare_active_space_problem(
        atoms=atoms,
        basis=DEFAULT_BASIS,
        charge=0,
        spin=0,
        frozen_core=True,
        active_space=(act_e, act_o),
        molecule_id=name,
    )

    if stage_cb:
        stage_cb(f"Running Ideal VQE across {len(config.seeds)} seed(s)")
    ideal_runs = []
    for s in config.seeds:
        r_ideal = run_vqe_from_prepared(
            prepared=prepared,
            ansatz_type="UCCSD",
            optimizer_name="SLSQP",
            seed=int(s),
            maxiter=int(config.ideal_maxiter),
            shots=None,
            noise_model="ideal",
            noise_model_name="ideal",
            matched_reference_energy=matched_ref.fci_energy,
            full_frozen_core_energy=full_fc_ref.fci_energy,
        )
        ideal_runs.append(r_ideal)

    ideal_energies = np.array([r.vqe_energy for r in ideal_runs], dtype=float)
    mean_ideal = float(np.mean(ideal_energies))
    std_ideal_ha = float(np.std(ideal_energies, ddof=1)) if len(ideal_energies) > 1 else 0.0
    abs_ideal_err_ha = abs(mean_ideal - matched_ref.fci_energy)
    ideal_conv = all(r.converged for r in ideal_runs)
    mean_ideal_time = float(np.mean([r.runtime_sec for r in ideal_runs]))

    _, reliability, rel_warns = classify_reliability(
        abs_error_ha=abs_ideal_err_ha,
        uncertainty_ha=std_ideal_ha,
        converged=ideal_conv,
        active_space_error_mha=active_space_error_mha,
        vqe_energy=mean_ideal,
        matched_fci_energy=matched_ref.fci_energy,
        is_noisy=False,
    )

    verdict = classify_advantage_verdict(
        converged=ideal_conv,
        num_seeds=len(config.seeds),
        reliability=reliability,
        abs_error_ha=abs_ideal_err_ha,
        vqe_runtime_sec=mean_ideal_time,
        classical_runtime_sec=matched_ref.total_runtime_sec,
        classical_exact_available=True,
        classical_reference_error_ha=0.0,
    )

    mean_noisy: Optional[float] = None
    noisy_err_mha: Optional[float] = None
    noisy_std_mha: Optional[float] = None

    if config.run_noisy:
        if stage_cb:
            stage_cb(f"Running Noisy VQE [{config.noisy_mode}] across {len(config.seeds)} seed(s)")
        noisy_runs = []
        for s in config.seeds:
            r_noisy = run_vqe_from_prepared(
                prepared=prepared,
                ansatz_type="UCCSD",
                optimizer_name="COBYLA",
                seed=int(s),
                maxiter=int(config.noisy_maxiter),
                shots=int(config.shots),
                noise_model=config.noisy_mode,
                noise_model_name=config.noisy_mode,
                matched_reference_energy=matched_ref.fci_energy,
                full_frozen_core_energy=full_fc_ref.fci_energy,
            )
            noisy_runs.append(r_noisy)

        noisy_energies = np.array([r.vqe_energy for r in noisy_runs], dtype=float)
        mean_noisy = float(np.mean(noisy_energies))
        noisy_std_ha = float(np.std(noisy_energies, ddof=1)) if len(noisy_energies) > 1 else 0.0
        noisy_err_mha = abs(mean_noisy - matched_ref.fci_energy) * 1000.0
        noisy_std_mha = noisy_std_ha * 1000.0

    first_run = ideal_runs[0]
    return {
        "qubits": first_run.num_qubits,
        "depth": first_run.circuit_depth,
        "gate_count": first_run.gate_count,
        "active_space": f"({act_e}e,{act_o}o)",
        "hf_energy": matched_ref.hf_energy,
        "fci_energy": matched_ref.fci_energy,
        "ideal_vqe_energy": mean_ideal,
        "ideal_error_mha": abs_ideal_err_ha * 1000.0,
        "ideal_uncertainty_mha": std_ideal_ha * 1000.0,
        "noisy_vqe_energy": mean_noisy,
        "noisy_error_mha": noisy_err_mha,
        "noisy_uncertainty_mha": noisy_std_mha,
        "active_space_error_mha": active_space_error_mha,
        "reliability": reliability,
        "verdict": verdict,
        "extra_warnings": rel_warns,
    }


def process_batch_row(
    smiles: str,
    name: Optional[str] = None,
    charge: int = 0,
    spin: int = 0,
    row_index: int = 1,
    config: Optional[BatchConfig] = None,
    run_simulation: bool = True,
    stage_callback: Optional[Callable[[str], None]] = None,
    simulation_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> BatchItemResult:
    """Pure function to validate, screen with QEE, and (if eligible) simulate a single batch row."""
    t0 = time.perf_counter()
    cfg = config or BatchConfig()
    primary_seed = int(cfg.seeds[0]) if cfg.seeds else 42
    display_name = name.strip() if name and name.strip() else smiles.strip()

    if stage_callback:
        stage_callback("Validating SMILES & computing RDKit descriptors")

    val = validate_and_describe(smiles)
    if not val.is_valid or val.descriptors is None:
        return BatchItemResult(
            row_index=row_index,
            name=display_name,
            input_smiles=smiles,
            canonical_smiles=None,
            charge=charge,
            spin=spin,
            seed=primary_seed,
            descriptors=None,
            qubits=None,
            depth=None,
            gate_count=None,
            active_space=None,
            qee_decision="REJECTED",
            reasons=[val.error or f"Invalid SMILES '{smiles}'."],
            warnings=[],
            outcome="REJECTED_INVALID_SMILES",
            runtime_sec=time.perf_counter() - t0,
        )

    desc = val.descriptors
    mol = Chem.MolFromSmiles(desc.canonical_smiles, sanitize=True)
    mol_h = Chem.AddHs(mol)

    formal_charge = int(charge) if int(charge) != 0 else int(Chem.GetFormalCharge(mol))
    total_electrons = sum(a.GetAtomicNum() for a in mol_h.GetAtoms()) - formal_charge
    num_radicals = int(Descriptors.NumRadicalElectrons(mol))

    # 1. Reject charged species
    if formal_charge != 0:
        return BatchItemResult(
            row_index=row_index,
            name=display_name,
            input_smiles=smiles,
            canonical_smiles=desc.canonical_smiles,
            charge=formal_charge,
            spin=spin,
            seed=primary_seed,
            descriptors=desc.to_dict(),
            qubits=None,
            depth=None,
            gate_count=None,
            active_space=None,
            qee_decision="REJECTED",
            reasons=[
                f"Charged molecule (charge={formal_charge}) rejected: batch screening requires neutral (charge=0) species."
            ],
            warnings=[],
            outcome="REJECTED_CHARGED",
            runtime_sec=time.perf_counter() - t0,
        )

    # 2. Reject open-shell / radical species
    if int(spin) != 0 or num_radicals > 0 or (total_electrons % 2 != 0):
        return BatchItemResult(
            row_index=row_index,
            name=display_name,
            input_smiles=smiles,
            canonical_smiles=desc.canonical_smiles,
            charge=formal_charge,
            spin=spin,
            seed=primary_seed,
            descriptors=desc.to_dict(),
            qubits=None,
            depth=None,
            gate_count=None,
            active_space=None,
            qee_decision="REJECTED",
            reasons=[
                f"Open-shell molecule (spin={spin}, radical_electrons={num_radicals}, total_electrons={total_electrons}) "
                "rejected: batch screening requires closed-shell singlet (spin=0) species."
            ],
            warnings=[],
            outcome="REJECTED_OPEN_SHELL",
            runtime_sec=time.perf_counter() - t0,
        )

    # 3. Reject unsupported elements
    unsupported = [el for el in desc.elements if el not in SUPPORTED_ELEMENTS]
    if unsupported:
        return BatchItemResult(
            row_index=row_index,
            name=display_name,
            input_smiles=smiles,
            canonical_smiles=desc.canonical_smiles,
            charge=formal_charge,
            spin=spin,
            seed=primary_seed,
            descriptors=desc.to_dict(),
            qubits=None,
            depth=None,
            gate_count=None,
            active_space=None,
            qee_decision="REJECTED",
            reasons=[
                f"Unsupported element(s) for STO-3G workflow: {', '.join(unsupported)}. "
                f"Supported elements: {', '.join(sorted(SUPPORTED_ELEMENTS))}."
            ],
            warnings=[],
            outcome="REJECTED_UNSUPPORTED_ELEMENT",
            runtime_sec=time.perf_counter() - t0,
        )

    # 4. Estimate qubits, depth, and gates & run QEE
    if stage_callback:
        stage_callback("Estimating qubits/depth and running QEE v1")
    est = estimate_qubits_from_smiles(desc.canonical_smiles, frozen_core=True)
    est_gates = int(round(est.estimated_circuit_depth * 1.75)) if est.estimated_circuit_depth > 0 else 0
    qee: QEEResult = evaluate_eligibility(desc.canonical_smiles)
    qee_decision = qee.decision or "DEFERRED"
    act_str = f"({est.active_electrons}e,{est.active_spatial_orbitals}o)"

    if qee_decision not in {"SIMULATED_QUANTUM", "HYBRID"} or not run_simulation:
        verdict = (
            "NO_DEMONSTRATED_ADVANTAGE"
            if qee_decision == "CLASSICAL_ONLY"
            else "INSUFFICIENT_EVIDENCE"
        )
        return BatchItemResult(
            row_index=row_index,
            name=display_name,
            input_smiles=smiles,
            canonical_smiles=desc.canonical_smiles,
            charge=formal_charge,
            spin=spin,
            seed=primary_seed,
            descriptors=desc.to_dict(),
            qubits=est.num_qubits,
            depth=est.estimated_circuit_depth,
            gate_count=est_gates,
            active_space=act_str,
            qee_decision=qee_decision,
            reasons=list(qee.reasons),
            warnings=list(qee.warnings),
            outcome="SCREENED_ONLY" if qee_decision in {"CLASSICAL_ONLY", "DEFERRED"} else "SKIPPED_SIMULATION_LIMIT",
            verdict=verdict,
            runtime_sec=time.perf_counter() - t0,
        )

    # 5. Eligible for quantum simulation (SIMULATED_QUANTUM or HYBRID): apply capped active-space policy
    cap_e, cap_o = int(cfg.capped_active_space[0]), int(cfg.capped_active_space[1])
    eff_e = min(est.active_electrons, cap_e)
    eff_o = min(est.active_spatial_orbitals, cap_o)
    # Ensure at least 1 occupied and 1 virtual orbital in the active window
    if eff_o <= eff_e // 2:
        eff_o = (eff_e // 2) + 1

    sim_runner = simulation_fn or _simulate_capped_vqe_on_rdkit_geometry

    try:
        if cfg.per_molecule_timeout_sec and cfg.per_molecule_timeout_sec > 0:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    sim_runner,
                    display_name,
                    desc.canonical_smiles,
                    eff_e,
                    eff_o,
                    cfg,
                    stage_callback,
                )
                sim_out = future.result(timeout=float(cfg.per_molecule_timeout_sec))
        else:
            sim_out = sim_runner(
                display_name,
                desc.canonical_smiles,
                eff_e,
                eff_o,
                cfg,
                stage_callback,
            )
    except FuturesTimeoutError:
        return BatchItemResult(
            row_index=row_index,
            name=display_name,
            input_smiles=smiles,
            canonical_smiles=desc.canonical_smiles,
            charge=formal_charge,
            spin=spin,
            seed=primary_seed,
            descriptors=desc.to_dict(),
            qubits=est.num_qubits,
            depth=est.estimated_circuit_depth,
            gate_count=est_gates,
            active_space=f"({eff_e}e,{eff_o}o)",
            qee_decision=qee_decision,
            reasons=list(qee.reasons)
            + [f"Simulation exceeded per-molecule timeout ({cfg.per_molecule_timeout_sec:.2f}s)."],
            warnings=list(qee.warnings),
            outcome="TIMEOUT",
            verdict="INSUFFICIENT_EVIDENCE",
            geometry_caption=BATCH_CUSTOM_GEOMETRY_CAPTION,
            runtime_sec=time.perf_counter() - t0,
        )
    except Exception as exc:
        return BatchItemResult(
            row_index=row_index,
            name=display_name,
            input_smiles=smiles,
            canonical_smiles=desc.canonical_smiles,
            charge=formal_charge,
            spin=spin,
            seed=primary_seed,
            descriptors=desc.to_dict(),
            qubits=est.num_qubits,
            depth=est.estimated_circuit_depth,
            gate_count=est_gates,
            active_space=f"({eff_e}e,{eff_o}o)",
            qee_decision=qee_decision,
            reasons=list(qee.reasons) + [f"Simulation error: {exc}"],
            warnings=list(qee.warnings),
            outcome="SIMULATION_ERROR",
            verdict="INSUFFICIENT_EVIDENCE",
            geometry_caption=BATCH_CUSTOM_GEOMETRY_CAPTION,
            runtime_sec=time.perf_counter() - t0,
        )

    combined_warnings = list(dict.fromkeys(list(qee.warnings) + list(sim_out.get("extra_warnings", []))))
    return BatchItemResult(
        row_index=row_index,
        name=display_name,
        input_smiles=smiles,
        canonical_smiles=desc.canonical_smiles,
        charge=formal_charge,
        spin=spin,
        seed=primary_seed,
        descriptors=desc.to_dict(),
        qubits=sim_out["qubits"],
        depth=sim_out["depth"],
        gate_count=sim_out["gate_count"],
        active_space=sim_out["active_space"],
        qee_decision=qee_decision,
        reasons=list(qee.reasons),
        warnings=combined_warnings,
        outcome="SIMULATED",
        hf_energy=sim_out["hf_energy"],
        fci_energy=sim_out["fci_energy"],
        ideal_vqe_energy=sim_out["ideal_vqe_energy"],
        ideal_error_mha=sim_out["ideal_error_mha"],
        ideal_uncertainty_mha=sim_out["ideal_uncertainty_mha"],
        noisy_vqe_energy=sim_out["noisy_vqe_energy"],
        noisy_error_mha=sim_out["noisy_error_mha"],
        noisy_uncertainty_mha=sim_out["noisy_uncertainty_mha"],
        active_space_error_mha=sim_out["active_space_error_mha"],
        reliability=sim_out["reliability"],
        verdict=sim_out["verdict"],
        geometry_caption=BATCH_CUSTOM_GEOMETRY_CAPTION,
        runtime_sec=time.perf_counter() - t0,
    )


def _find_resumable_batch_item(
    canonical_smiles: str,
    config_json: str,
    seed: int,
    db_path: Union[Path, str],
) -> Optional[Dict[str, Any]]:
    """Look up a previously stored batch_items record with identical canonical_smiles, config_json, and seed."""
    init_db(db_path)
    sql = """
    SELECT * FROM batch_items
    WHERE canonical_smiles = ? AND config_json = ? AND seed = ?
      AND outcome IN ('SIMULATED', 'SCREENED_ONLY', 'RESUMED')
    ORDER BY id DESC LIMIT 1;
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute(sql, (canonical_smiles, config_json, int(seed)))
        row = cursor.fetchone()
        return dict(row) if row is not None else None


def _save_batch_item(
    batch_id: int,
    config_json: str,
    item: BatchItemResult,
    db_path: Union[Path, str],
) -> int:
    """Insert a BatchItemResult row into the SQLite batch_items table."""
    sql = """
    INSERT INTO batch_items (
        batch_id, row_index, name, input_smiles, canonical_smiles,
        charge, spin, config_json, seed, descriptors_json,
        qubits, depth, gate_count, active_space, qee_decision,
        reasons_json, warnings_json, outcome,
        hf_energy, fci_energy, ideal_vqe_energy, ideal_error_mha, ideal_uncertainty_mha,
        noisy_vqe_energy, noisy_error_mha, noisy_uncertainty_mha,
        active_space_error_mha, reliability, verdict, geometry_caption, runtime_sec
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    );
    """
    params = (
        int(batch_id),
        int(item.row_index),
        item.name,
        item.input_smiles,
        item.canonical_smiles,
        int(item.charge),
        int(item.spin),
        config_json,
        int(item.seed),
        json.dumps(item.descriptors, sort_keys=True) if item.descriptors else None,
        item.qubits,
        item.depth,
        item.gate_count,
        item.active_space,
        item.qee_decision,
        json.dumps(item.reasons),
        json.dumps(item.warnings),
        item.outcome,
        item.hf_energy,
        item.fci_energy,
        item.ideal_vqe_energy,
        item.ideal_error_mha,
        item.ideal_uncertainty_mha,
        item.noisy_vqe_energy,
        item.noisy_error_mha,
        item.noisy_uncertainty_mha,
        item.active_space_error_mha,
        item.reliability,
        item.verdict,
        item.geometry_caption,
        float(item.runtime_sec),
    )
    with get_connection(db_path) as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        return int(cursor.lastrowid)


def run_batch_screening(
    csv_source: Union[str, bytes, Path, pd.DataFrame],
    config: Optional[BatchConfig] = None,
    max_rows: int = BATCH_MAX_ROWS,
    db_path: Union[Path, str] = DEFAULT_DB_PATH,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    stop_requested_fn: Optional[Callable[[], bool]] = None,
    simulation_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> BatchRunResult:
    """Validate a CSV and execute batch screening with timeouts, resume, and SQLite persistence."""
    batch_start = time.perf_counter()
    cfg = config or BatchConfig()
    config_json = cfg.to_canonical_json()
    primary_seed = int(cfg.seeds[0]) if cfg.seeds else 42

    validation = validate_batch_csv(csv_source, max_rows=max_rows)
    if not validation.is_valid:
        raise ValueError(validation.error or "Batch CSV validation failed.")

    init_db(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO batches (
                created_at, config_json, seed, total_rows, valid_rows,
                duplicates_removed, simulated_count, status, total_runtime_sec
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 'RUNNING', 0.0);
            """,
            (
                created_at,
                config_json,
                primary_seed,
                validation.total_rows,
                len(validation.valid_rows),
                validation.duplicates_removed,
            ),
        )
        conn.commit()
        batch_id = int(cursor.lastrowid)

    items: List[BatchItemResult] = []
    simulated_count = 0
    resumed_count = 0
    batch_status = "COMPLETED"
    total_valid = len(validation.valid_rows)

    for idx_zero, vrow in enumerate(validation.valid_rows):
        if stop_requested_fn is not None and stop_requested_fn():
            batch_status = "STOPPED"
            break

        elapsed = time.perf_counter() - batch_start
        within_budget = (
            cfg.total_time_budget_sec <= 0 or elapsed < float(cfg.total_time_budget_sec)
        )

        def _stage_cb(stage_msg: str, _i: int = idx_zero + 1, _name: str = vrow.name) -> None:
            if progress_callback is not None:
                progress_callback(_i, total_valid, _name, stage_msg)

        # 1. Check Resume Cache if enabled and charge==0, spin==0
        if cfg.resume and vrow.charge == 0 and vrow.spin == 0:
            cached = _find_resumable_batch_item(
                canonical_smiles=vrow.canonical_smiles,
                config_json=config_json,
                seed=primary_seed,
                db_path=db_path,
            )
            if cached is not None:
                _stage_cb("Resumed from stored SQLite result")
                resumed_item = BatchItemResult(
                    row_index=vrow.row_index,
                    name=vrow.name,
                    input_smiles=vrow.input_smiles,
                    canonical_smiles=vrow.canonical_smiles,
                    charge=vrow.charge,
                    spin=vrow.spin,
                    seed=primary_seed,
                    descriptors=json.loads(cached["descriptors_json"])
                    if cached.get("descriptors_json")
                    else None,
                    qubits=cached.get("qubits"),
                    depth=cached.get("depth"),
                    gate_count=cached.get("gate_count"),
                    active_space=cached.get("active_space"),
                    qee_decision=str(cached["qee_decision"]),
                    reasons=json.loads(cached["reasons_json"])
                    if cached.get("reasons_json")
                    else [],
                    warnings=json.loads(cached["warnings_json"])
                    if cached.get("warnings_json")
                    else [],
                    outcome="RESUMED",
                    hf_energy=cached.get("hf_energy"),
                    fci_energy=cached.get("fci_energy"),
                    ideal_vqe_energy=cached.get("ideal_vqe_energy"),
                    ideal_error_mha=cached.get("ideal_error_mha"),
                    ideal_uncertainty_mha=cached.get("ideal_uncertainty_mha"),
                    noisy_vqe_energy=cached.get("noisy_vqe_energy"),
                    noisy_error_mha=cached.get("noisy_error_mha"),
                    noisy_uncertainty_mha=cached.get("noisy_uncertainty_mha"),
                    active_space_error_mha=cached.get("active_space_error_mha"),
                    reliability=cached.get("reliability"),
                    verdict=cached.get("verdict"),
                    geometry_caption=cached.get("geometry_caption"),
                    runtime_sec=0.0,
                    was_resumed=True,
                )
                item_id = _save_batch_item(batch_id, config_json, resumed_item, db_path)
                items.append(
                    BatchItemResult(**{**resumed_item.to_dict(), "db_item_id": item_id})
                )
                resumed_count += 1
                continue

        allow_sim = within_budget and (simulated_count < int(cfg.max_simulated_molecules))
        processed = process_batch_row(
            smiles=vrow.input_smiles,
            name=vrow.name,
            charge=vrow.charge,
            spin=vrow.spin,
            row_index=vrow.row_index,
            config=cfg,
            run_simulation=allow_sim,
            stage_callback=_stage_cb,
            simulation_fn=simulation_fn,
        )

        if not within_budget and processed.qee_decision in {"SIMULATED_QUANTUM", "HYBRID"}:
            processed = BatchItemResult(
                **{
                    **processed.to_dict(),
                    "outcome": "SKIPPED_TIME_BUDGET",
                    "reasons": processed.reasons
                    + [f"Skipped quantum simulation: total batch time budget ({cfg.total_time_budget_sec:.1f}s) exhausted."],
                }
            )

        if processed.outcome == "SIMULATED":
            simulated_count += 1

        item_id = _save_batch_item(batch_id, config_json, processed, db_path)
        items.append(BatchItemResult(**{**processed.to_dict(), "db_item_id": item_id}))

    total_runtime = time.perf_counter() - batch_start
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE batches
            SET simulated_count = ?, status = ?, total_runtime_sec = ?
            WHERE id = ?;
            """,
            (simulated_count, batch_status, total_runtime, batch_id),
        )
        conn.commit()

    return BatchRunResult(
        batch_id=batch_id,
        status=batch_status,
        config=json.loads(config_json),
        validation=validation,
        items=items,
        simulated_count=simulated_count,
        resumed_count=resumed_count,
        total_runtime_sec=total_runtime,
    )


def list_batches(
    limit: int = 50,
    db_path: Union[Path, str] = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """Return recent batch runs stored in SQLite ordered from newest to oldest."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM batches ORDER BY id DESC LIMIT ?;", (int(limit),)
        )
        return [dict(r) for r in cursor.fetchall()]


def get_batch_items(
    batch_id: int,
    db_path: Union[Path, str] = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """Return all batch_items rows for a given batch_id ordered by row_index."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM batch_items WHERE batch_id = ? ORDER BY row_index ASC;",
            (int(batch_id),),
        )
        rows = []
        for r in cursor.fetchall():
            d = dict(r)
            d["reasons"] = json.loads(d["reasons_json"]) if d.get("reasons_json") else []
            d["warnings"] = json.loads(d["warnings_json"]) if d.get("warnings_json") else []
            d["descriptors"] = (
                json.loads(d["descriptors_json"]) if d.get("descriptors_json") else None
            )
            rows.append(d)
        return rows
