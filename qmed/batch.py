"""Batch molecular screening, CSV validation, shared capped active-space VQE execution, and SQLite batch persistence."""

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from rdkit import Chem

from qmed.config import (
    BATCH_CAPPED_ACTIVE_SPACE,
    BATCH_CUSTOM_GEOMETRY_CAPTION,
    BATCH_DEFAULT_MAX_SIMULATED,
    BATCH_DEFAULT_MOLECULE_TIMEOUT_SEC,
    BATCH_DEFAULT_TOTAL_BUDGET_SEC,
    BATCH_GEOMETRY_SEED,
    BATCH_MAX_ROWS,
    DEFAULT_DB_PATH,
    DEFAULT_SHOTS,
)
from qmed.storage import (
    MoleculePipelineConfig,
    PipelineExecutionResult,
    cap_active_space,
    get_connection,
    init_db,
    run_molecule_pipeline,
)

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
    ideal_maxiter: int = 8
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
    error_stage: Optional[str] = None
    error_message: Optional[str] = None
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
    failed_count: int = 0

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
                    "Error Stage": item.error_stage or "—",
                    "Error Message": item.error_message or "—",
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
    """Validate, screen with QEE, and (if eligible) simulate a single batch row via the shared `run_molecule_pipeline`."""
    cfg = config or BatchConfig()
    primary_seed = int(cfg.seeds[0]) if cfg.seeds else 42
    display_name = name.strip() if name and name.strip() else str(smiles).strip()

    mol_cfg = MoleculePipelineConfig(
        seeds=cfg.seeds,
        geometry_seed=cfg.geometry_seed,
        capped_active_space=cfg.capped_active_space,
        use_benchmark_geometry=False,
        charge=int(charge),
        spin=int(spin),
        run_simulation=bool(run_simulation),
        run_noisy=bool(cfg.run_noisy),
        noisy_mode=str(cfg.noisy_mode),
        shots=int(cfg.shots),
        ideal_maxiter=int(cfg.ideal_maxiter),
        noisy_maxiter=int(cfg.noisy_maxiter),
        per_molecule_timeout_sec=float(cfg.per_molecule_timeout_sec),
        persist_to_sqlite=False,
        name=display_name,
    )

    pipe_res: PipelineExecutionResult = run_molecule_pipeline(
        smiles_or_preset=smiles,
        config=mol_cfg,
        stage_callback=stage_callback,
        simulation_fn=simulation_fn,
    )

    ideal_bm = pipe_res.ideal_benchmark
    noisy_bm = pipe_res.noisy_benchmark
    summary = pipe_res.active_space_summary

    if ideal_bm is not None:
        qubits = ideal_bm.num_qubits
        depth = ideal_bm.circuit_depth
        gates = ideal_bm.gate_count
        act_str: Optional[str] = f"({ideal_bm.active_electrons}e,{ideal_bm.active_orbitals}o)"
        combined_warnings = list(dict.fromkeys(list(pipe_res.qee.warnings) + list(ideal_bm.warnings)))
    elif summary is not None:
        qubits = summary.num_qubits
        depth = summary.estimated_circuit_depth
        gates = int(round(depth * 1.75)) if depth > 0 else 0
        if pipe_res.effective_active_space is not None:
            act_str = f"({pipe_res.effective_active_space[0]}e,{pipe_res.effective_active_space[1]}o)"
        else:
            act_str = f"({summary.active_electrons}e,{summary.active_spatial_orbitals}o)"
        combined_warnings = list(pipe_res.qee.warnings)
    else:
        qubits = None
        depth = None
        gates = None
        act_str = None
        combined_warnings = list(pipe_res.qee.warnings)

    reasons_list = list(pipe_res.qee.reasons)
    if pipe_res.outcome in {"SIMULATION_ERROR", "TIMEOUT"} and pipe_res.error_message:
        err_entry = (
            f"Stage '{pipe_res.error_stage}': {pipe_res.error_message}"
            if pipe_res.error_stage
            else pipe_res.error_message
        )
        if err_entry not in reasons_list:
            reasons_list.append(err_entry)

    return BatchItemResult(
        row_index=row_index,
        name=display_name,
        input_smiles=smiles,
        canonical_smiles=pipe_res.canonical_smiles,
        charge=pipe_res.charge,
        spin=pipe_res.spin,
        seed=primary_seed,
        descriptors=pipe_res.descriptors,
        qubits=qubits,
        depth=depth,
        gate_count=gates,
        active_space=act_str,
        qee_decision=pipe_res.qee.decision or "REJECTED",
        reasons=reasons_list,
        warnings=combined_warnings,
        outcome=pipe_res.outcome,
        error_stage=pipe_res.error_stage,
        error_message=pipe_res.error_message,
        hf_energy=ideal_bm.hf_energy if ideal_bm else None,
        fci_energy=ideal_bm.matched_fci_energy if ideal_bm else None,
        ideal_vqe_energy=ideal_bm.mean_vqe_energy if ideal_bm else None,
        ideal_error_mha=ideal_bm.abs_mean_error_mha if ideal_bm else None,
        ideal_uncertainty_mha=ideal_bm.uncertainty_mha if ideal_bm else None,
        noisy_vqe_energy=noisy_bm.mean_vqe_energy if noisy_bm else None,
        noisy_error_mha=noisy_bm.abs_mean_error_mha if noisy_bm else None,
        noisy_uncertainty_mha=noisy_bm.uncertainty_mha if noisy_bm else None,
        active_space_error_mha=ideal_bm.active_space_error_mha if ideal_bm else None,
        reliability=ideal_bm.reliability if ideal_bm else None,
        verdict=pipe_res.verdict if not pipe_res.outcome.startswith("REJECTED") else None,
        geometry_caption=pipe_res.geometry_caption,
        runtime_sec=pipe_res.runtime_sec,
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
        reasons_json, warnings_json, outcome, error_stage, error_message,
        hf_energy, fci_energy, ideal_vqe_energy, ideal_error_mha, ideal_uncertainty_mha,
        noisy_vqe_energy, noisy_error_mha, noisy_uncertainty_mha,
        active_space_error_mha, reliability, verdict, geometry_caption, runtime_sec
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
        item.error_stage,
        item.error_message,
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
                duplicates_removed, simulated_count, failed_count, status, total_runtime_sec
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, 'RUNNING', 0.0);
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
    failed_count = 0
    resumed_count = 0
    was_stopped = False
    total_valid = len(validation.valid_rows)

    for idx_zero, vrow in enumerate(validation.valid_rows):
        if stop_requested_fn is not None and stop_requested_fn():
            was_stopped = True
            break

        elapsed = time.perf_counter() - batch_start
        within_budget = (
            cfg.total_time_budget_sec <= 0 or elapsed < float(cfg.total_time_budget_sec)
        )

        def _stage_cb(stage_msg: str, _i: int = idx_zero + 1, _name: str = vrow.name) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(_i, total_valid, _name, stage_msg)
                except Exception:
                    pass

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
                    error_stage=cached.get("error_stage"),
                    error_message=cached.get("error_message"),
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
                    + [
                        f"Skipped quantum simulation: total batch time budget ({cfg.total_time_budget_sec:.1f}s) exhausted."
                    ],
                }
            )

        if processed.outcome == "SIMULATED":
            simulated_count += 1
        elif processed.outcome in {"SIMULATION_ERROR", "TIMEOUT"}:
            failed_count += 1

        item_id = _save_batch_item(batch_id, config_json, processed, db_path)
        items.append(BatchItemResult(**{**processed.to_dict(), "db_item_id": item_id}))

    if was_stopped:
        batch_status = "STOPPED"
    elif failed_count > 0:
        batch_status = "COMPLETED_WITH_ERRORS"
    else:
        batch_status = "COMPLETED"

    total_runtime = time.perf_counter() - batch_start
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE batches
            SET simulated_count = ?, failed_count = ?, status = ?, total_runtime_sec = ?
            WHERE id = ?;
            """,
            (simulated_count, failed_count, batch_status, total_runtime, batch_id),
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
        failed_count=failed_count,
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
