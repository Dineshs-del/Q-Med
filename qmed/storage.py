"""SQLite persistence layer, shared single/batch molecule pipeline, and deterministic experiment replay."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from qmed.active_space import (
    ActiveSpaceSummary,
    estimate_qubits_from_smiles,
    prepare_active_space_problem,
)
from qmed.benchmark import (
    MultiSeedBenchmarkResult,
    classify_advantage_verdict,
    classify_reliability,
)
from qmed.classical import run_classical_reference
from qmed.config import (
    BATCH_CAPPED_ACTIVE_SPACE,
    BATCH_CUSTOM_GEOMETRY_CAPTION,
    BATCH_GEOMETRY_SEED,
    DEFAULT_BASIS,
    DEFAULT_BENCHMARK_SEEDS,
    DEFAULT_DB_PATH,
    DEFAULT_SHOTS,
    SUPPORTED_ELEMENTS,
)
from qmed.molecules import (
    AtomCoord,
    get_benchmark,
    smiles_to_3d_coordinates,
    validate_and_describe,
)
from qmed.qee import QEEResult, evaluate_eligibility, resolve_input_smiles
from qmed.vqe_runner import VQERunResult, run_vqe_from_prepared

EXPERIMENTS_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_identifier TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    config_json TEXT NOT NULL,
    descriptors_json TEXT,
    qee_decision TEXT NOT NULL,
    qee_explanation_json TEXT NOT NULL,
    basis TEXT,
    active_space TEXT,
    mapping TEXT,
    ansatz TEXT,
    optimizer TEXT,
    backend TEXT,
    shots INTEGER,
    seed INTEGER,
    noise_model TEXT,
    vqe_energy REAL,
    hf_energy REAL,
    fci_energy REAL,
    error_ha REAL,
    uncertainty_ha REAL,
    iterations INTEGER,
    converged INTEGER,
    runtime_sec REAL,
    qubits INTEGER,
    depth INTEGER,
    gate_count INTEGER,
    reliability TEXT,
    verdict TEXT
);
"""

BATCH_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    config_json TEXT NOT NULL,
    seed INTEGER NOT NULL,
    total_rows INTEGER NOT NULL,
    valid_rows INTEGER NOT NULL,
    duplicates_removed INTEGER NOT NULL,
    simulated_count INTEGER NOT NULL,
    failed_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    total_runtime_sec REAL
);

CREATE TABLE IF NOT EXISTS batch_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    row_index INTEGER NOT NULL,
    name TEXT,
    input_smiles TEXT NOT NULL,
    canonical_smiles TEXT,
    charge INTEGER,
    spin INTEGER,
    config_json TEXT NOT NULL,
    seed INTEGER NOT NULL,
    descriptors_json TEXT,
    qubits INTEGER,
    depth INTEGER,
    gate_count INTEGER,
    active_space TEXT,
    qee_decision TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    warnings_json TEXT,
    outcome TEXT NOT NULL,
    error_stage TEXT,
    error_message TEXT,
    hf_energy REAL,
    fci_energy REAL,
    ideal_vqe_energy REAL,
    ideal_error_mha REAL,
    ideal_uncertainty_mha REAL,
    noisy_vqe_energy REAL,
    noisy_error_mha REAL,
    noisy_uncertainty_mha REAL,
    active_space_error_mha REAL,
    reliability TEXT,
    verdict TEXT,
    geometry_caption TEXT,
    runtime_sec REAL,
    FOREIGN KEY (batch_id) REFERENCES batches(id)
);
"""


def cap_active_space(
    requested_active_space: Optional[Tuple[int, int]],
    available_electrons: int,
    available_orbitals: int,
) -> Tuple[int, int]:
    """Compute effective (active_electrons, active_orbitals) using min(requested, available) on occupied and virtual manifolds.

    In a closed-shell molecule with `available_electrons` valence electrons and
    `available_orbitals` valence spatial orbitals:
      - `avail_occ = available_electrons // 2` doubly-occupied valence orbitals
      - `avail_vir = available_orbitals - avail_occ` unoccupied virtual orbitals
    For a requested `(req_e, req_o)` active space:
      - `req_occ = req_e // 2`
      - `req_vir = req_o - req_occ`
      - `eff_occ = min(req_occ, avail_occ)`
      - `eff_vir = min(req_vir, avail_vir)`
      - `eff_e = 2 * eff_occ`, `eff_o = eff_occ + eff_vir`
    This guarantees `nvir = avail_vir - eff_vir >= 0` in PySCF CASCI and Qiskit ActiveSpaceTransformer.
    """
    avail_e = max(0, int(available_electrons))
    avail_o = max(0, int(available_orbitals))
    avail_occ = avail_e // 2
    avail_vir = max(0, avail_o - avail_occ)

    if requested_active_space is None:
        return avail_e, avail_o

    req_e = max(2, int(requested_active_space[0]))
    req_o = max(2, int(requested_active_space[1]))
    req_occ = max(1, req_e // 2)
    req_vir = max(1, req_o - req_occ)

    eff_occ = min(req_occ, avail_occ)
    eff_vir = min(req_vir, avail_vir)
    eff_e = 2 * eff_occ
    eff_o = eff_occ + eff_vir
    return eff_e, eff_o


@dataclass(frozen=True)
class MoleculePipelineConfig:
    """Configuration for the unified single-molecule and batch Q-Med simulation pipeline."""

    seeds: Tuple[int, ...] = DEFAULT_BENCHMARK_SEEDS
    geometry_seed: int = BATCH_GEOMETRY_SEED
    capped_active_space: Optional[Tuple[int, int]] = BATCH_CAPPED_ACTIVE_SPACE
    active_space_override: Optional[Tuple[int, int]] = None
    use_benchmark_geometry: Optional[bool] = None
    charge: int = 0
    spin: int = 0
    run_simulation: bool = True
    run_noisy: bool = False
    noisy_mode: str = "depolarizing_readout"
    shots: int = DEFAULT_SHOTS
    ansatz_type: str = "UCCSD"
    ideal_optimizer: Optional[str] = "SLSQP"
    noisy_optimizer: Optional[str] = "COBYLA"
    ideal_maxiter: Optional[int] = 40
    noisy_maxiter: Optional[int] = 20
    per_molecule_timeout_sec: Optional[float] = None
    classical_sufficient: bool = False
    reference_available: bool = True
    persist_to_sqlite: bool = False
    db_path: Union[Path, str] = DEFAULT_DB_PATH
    name: Optional[str] = None


@dataclass(frozen=True)
class PipelineExecutionResult:
    """Complete end-to-end Q-Med pipeline output for a molecule or benchmark."""

    input_identifier: str
    smiles: str
    benchmark_id: Optional[str]
    descriptors: Optional[Dict[str, Any]]
    qee: QEEResult
    active_space_summary: Optional[ActiveSpaceSummary]
    ideal_benchmark: Optional[MultiSeedBenchmarkResult]
    noisy_benchmark: Optional[MultiSeedBenchmarkResult]
    stored_experiment_ids: List[int] = field(default_factory=list)
    primary_experiment_id: Optional[int] = None
    canonical_smiles: Optional[str] = None
    charge: int = 0
    spin: int = 0
    outcome: str = "SIMULATED"
    status_category: str = "SIMULATED"
    status_message: str = ""
    geometry_source: str = "rdkit_mmff_approx"
    geometry_caption: Optional[str] = None
    effective_active_space: Optional[Tuple[int, int]] = None
    active_space_error_mha: Optional[float] = None
    error_stage: Optional[str] = None
    error_message: Optional[str] = None
    runtime_sec: float = 0.0

    @property
    def hf_energy(self) -> Optional[float]:
        """Return Hartree-Fock reference energy (Ha) if simulated."""
        return self.ideal_benchmark.hf_energy if self.ideal_benchmark else None

    @property
    def fci_energy(self) -> Optional[float]:
        """Return matched CASCI/FCI reference energy (Ha) if simulated."""
        return self.ideal_benchmark.matched_fci_energy if self.ideal_benchmark else None

    @property
    def matched_fci_energy(self) -> Optional[float]:
        """Return matched CASCI/FCI reference energy (Ha) if simulated."""
        return self.ideal_benchmark.matched_fci_energy if self.ideal_benchmark else None

    @property
    def ideal_vqe_energy(self) -> Optional[float]:
        """Return mean ideal VQE energy (Ha) if simulated."""
        return self.ideal_benchmark.mean_vqe_energy if self.ideal_benchmark else None

    @property
    def vqe_energy(self) -> Optional[float]:
        """Return mean ideal VQE energy (Ha) if simulated."""
        return self.ideal_benchmark.mean_vqe_energy if self.ideal_benchmark else None

    @property
    def ideal_error_mha(self) -> Optional[float]:
        """Return mean absolute error of ideal VQE vs matched reference in mHa."""
        return self.ideal_benchmark.abs_mean_error_mha if self.ideal_benchmark else None

    @property
    def error_mha(self) -> Optional[float]:
        """Return mean absolute error of ideal VQE vs matched reference in mHa."""
        return self.ideal_benchmark.abs_mean_error_mha if self.ideal_benchmark else None

    @property
    def ideal_uncertainty_mha(self) -> Optional[float]:
        """Return standard deviation of ideal VQE across seeds in mHa."""
        return self.ideal_benchmark.uncertainty_mha if self.ideal_benchmark else None

    @property
    def noisy_vqe_energy(self) -> Optional[float]:
        """Return mean noisy VQE energy (Ha) if noisy simulation was run."""
        return self.noisy_benchmark.mean_vqe_energy if self.noisy_benchmark else None

    @property
    def noisy_error_mha(self) -> Optional[float]:
        """Return mean absolute error of noisy VQE vs matched reference in mHa."""
        return self.noisy_benchmark.abs_mean_error_mha if self.noisy_benchmark else None

    @property
    def noisy_uncertainty_mha(self) -> Optional[float]:
        """Return standard deviation of noisy VQE across seeds in mHa."""
        return self.noisy_benchmark.uncertainty_mha if self.noisy_benchmark else None

    @property
    def reliability(self) -> Optional[str]:
        """Return reliability label of the ideal VQE benchmark."""
        return self.ideal_benchmark.reliability if self.ideal_benchmark else None

    @property
    def verdict(self) -> Optional[str]:
        """Return quantum advantage verdict."""
        if self.ideal_benchmark is not None:
            return self.ideal_benchmark.verdict
        if self.qee.decision == "CLASSICAL_ONLY":
            return "NO_DEMONSTRATED_ADVANTAGE"
        return "INSUFFICIENT_EVIDENCE"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the pipeline execution result."""
        return asdict(self)


@dataclass(frozen=True)
class ReplayResult:
    """Result of replaying a stored SQLite experiment by ID."""

    experiment_id: int
    input_identifier: str
    run_type: str
    stored_qee_decision: str
    replayed_qee_decision: str
    stored_vqe_energy: Optional[float]
    replayed_vqe_energy: Optional[float]
    energy_diff_ha: Optional[float]
    stored_iterations: Optional[int]
    replayed_iterations: Optional[int]
    stored_reliability: Optional[str]
    replayed_reliability: Optional[str]
    stored_verdict: Optional[str]
    replayed_verdict: Optional[str]
    reproduced: bool
    stored_record: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the replay verification result."""
        return asdict(self)


def build_pipeline_status_message(
    qee_decision: Optional[str],
    outcome: str,
    reasons: Sequence[str],
    ideal_benchmark: Optional[MultiSeedBenchmarkResult] = None,
    error_stage: Optional[str] = None,
    error_message: Optional[str] = None,
    primary_experiment_id: Optional[int] = None,
) -> Tuple[str, str]:
    """Build a non-contradictory (status_category, status_message) pair.

    Distinguishes:
      - SIMULATED: molecule routed to SIMULATED_QUANTUM or HYBRID and VQE completed.
      - DEFERRED: molecule routed to DEFERRED; no VQE simulation run.
      - CLASSICAL_ONLY: molecule routed to CLASSICAL_ONLY; no VQE simulation run.
      - REJECTED: molecule rejected before QEE/VQE (invalid SMILES, charged, open-shell, unsupported element).
      - SIMULATION_FAILED: molecule routed to SIMULATED_QUANTUM or HYBRID, but simulation failed or timed out.
    """
    decision_str = qee_decision or "REJECTED"
    reason_summary = "; ".join(r for r in reasons if r)
    id_suffix = f" (SQLite record `#{primary_experiment_id}`)" if primary_experiment_id is not None else ""

    if outcome in {"SIMULATED", "RESUMED"} and ideal_benchmark is not None:
        msg = (
            f"Molecule routed to **{decision_str}** and **simulated** in active space "
            f"`({ideal_benchmark.active_electrons}e, {ideal_benchmark.active_orbitals}o)` "
            f"({ideal_benchmark.num_qubits} qubits, depth {ideal_benchmark.circuit_depth}). "
            f"Ideal VQE |error| vs. matched reference = **{ideal_benchmark.abs_mean_error_mha:.4f} mHa** "
            f"(reliability: **{ideal_benchmark.reliability}**){id_suffix}."
        )
        return "SIMULATED", msg

    if outcome in {"SIMULATION_ERROR", "TIMEOUT"}:
        stage_label = error_stage or "simulation"
        err_detail = error_message or reason_summary or "Unknown error"
        msg = (
            f"Molecule routed to **{decision_str}**, but **simulation failed** (`{outcome}`) "
            f"at stage **{stage_label}**: `{err_detail}`{id_suffix}."
        )
        return "SIMULATION_FAILED", msg

    if outcome.startswith("REJECTED") or decision_str == "REJECTED":
        msg = (
            f"Molecule **rejected** (`{outcome}`) — quantum simulation was not run. "
            f"Reason: {reason_summary or error_message or 'Failed input validation.'}{id_suffix}"
        )
        return "REJECTED", msg

    if decision_str == "CLASSICAL_ONLY" or outcome == "CLASSICAL_ONLY":
        msg = (
            f"Molecule routed to **CLASSICAL_ONLY** — quantum simulation is unnecessary because "
            f"classical Hartree–Fock / exact diagonalization is already exact in this basis. "
            f"Reason: {reason_summary}{id_suffix}"
        )
        return "CLASSICAL_ONLY", msg

    if decision_str == "DEFERRED" or outcome == "DEFERRED":
        msg = (
            f"Molecule routed to **DEFERRED** — quantum simulation was not run because the molecule "
            f"exceeds current simulator qubit/depth limits. Reason: {reason_summary}{id_suffix}"
        )
        return "DEFERRED", msg

    msg = (
        f"Molecule routed to **{decision_str}** (`{outcome}`) — simulation was not executed "
        f"({reason_summary}){id_suffix}."
    )
    return outcome, msg


def get_connection(db_path: Union[Path, str] = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row factory set to sqlite3.Row."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Union[Path, str] = DEFAULT_DB_PATH) -> Path:
    """Create the SQLite database file and `experiments`, `batches`, and `batch_items` tables."""
    resolved_path = Path(db_path)
    if str(resolved_path) != ":memory:":
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(resolved_path) as conn:
        conn.executescript(EXPERIMENTS_SCHEMA_SQL)
        conn.executescript(BATCH_SCHEMA_SQL)
        # Ensure columns added in schema updates exist on pre-existing database files
        batch_cols = {
            str(r["name"])
            for r in conn.execute("PRAGMA table_info(batches);").fetchall()
        }
        if "failed_count" not in batch_cols:
            conn.execute(
                "ALTER TABLE batches ADD COLUMN failed_count INTEGER NOT NULL DEFAULT 0;"
            )
        item_cols = {
            str(r["name"])
            for r in conn.execute("PRAGMA table_info(batch_items);").fetchall()
        }
        if "error_stage" not in item_cols:
            conn.execute("ALTER TABLE batch_items ADD COLUMN error_stage TEXT;")
        if "error_message" not in item_cols:
            conn.execute("ALTER TABLE batch_items ADD COLUMN error_message TEXT;")
        conn.commit()

    return resolved_path


def get_table_columns(
    table_name: str = "experiments", db_path: Union[Path, str] = DEFAULT_DB_PATH
) -> List[str]:
    """Return column names for a given table (useful for schema verification tests)."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(f"PRAGMA table_info({table_name});")
        return [str(row["name"]) for row in cursor.fetchall()]


def _serialize_json_field(value: Any) -> Optional[str]:
    """Serialize a dict/list to a JSON string or pass through an existing string/None."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def save_experiment_record(
    record: Dict[str, Any],
    db_path: Union[Path, str] = DEFAULT_DB_PATH,
) -> int:
    """Insert a single experiment record into SQLite and return its auto-incremented ID."""
    init_db(db_path)
    timestamp = record.get("timestamp") or datetime.now(timezone.utc).isoformat()
    config_json = _serialize_json_field(record.get("config_json", {})) or "{}"
    descriptors_json = _serialize_json_field(record.get("descriptors_json"))
    qee_explanation_json = (
        _serialize_json_field(record.get("qee_explanation_json", {})) or "{}"
    )

    converged_val = record.get("converged")
    if converged_val is not None:
        converged_int: Optional[int] = 1 if bool(converged_val) else 0
    else:
        converged_int = None

    sql = """
    INSERT INTO experiments (
        input_identifier, timestamp, config_json, descriptors_json,
        qee_decision, qee_explanation_json, basis, active_space, mapping,
        ansatz, optimizer, backend, shots, seed, noise_model,
        vqe_energy, hf_energy, fci_energy, error_ha, uncertainty_ha,
        iterations, converged, runtime_sec, qubits, depth, gate_count,
        reliability, verdict
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    );
    """
    params = (
        str(record["input_identifier"]),
        str(timestamp),
        config_json,
        descriptors_json,
        str(record["qee_decision"]),
        qee_explanation_json,
        record.get("basis"),
        record.get("active_space"),
        record.get("mapping"),
        record.get("ansatz"),
        record.get("optimizer"),
        record.get("backend"),
        record.get("shots"),
        record.get("seed"),
        record.get("noise_model"),
        record.get("vqe_energy"),
        record.get("hf_energy"),
        record.get("fci_energy"),
        record.get("error_ha"),
        record.get("uncertainty_ha"),
        record.get("iterations"),
        converged_int,
        record.get("runtime_sec"),
        record.get("qubits"),
        record.get("depth"),
        record.get("gate_count"),
        record.get("reliability"),
        record.get("verdict"),
    )

    with get_connection(db_path) as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        return int(cursor.lastrowid)


def _decode_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite3.Row into a dictionary with parsed JSON sub-objects."""
    data = dict(row)
    data["config"] = json.loads(data["config_json"]) if data.get("config_json") else {}
    data["descriptors"] = (
        json.loads(data["descriptors_json"]) if data.get("descriptors_json") else None
    )
    data["qee_explanation"] = (
        json.loads(data["qee_explanation_json"])
        if data.get("qee_explanation_json")
        else {}
    )
    if data.get("converged") is not None:
        data["converged"] = bool(data["converged"])
    return data


def get_experiment_by_id(
    experiment_id: int,
    db_path: Union[Path, str] = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """Retrieve a stored experiment record by ID or raise KeyError if not found."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM experiments WHERE id = ?;", (int(experiment_id),)
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(
                f"Experiment with id={experiment_id} not found in {db_path}."
            )
        return _decode_row(row)


def list_experiments(
    limit: int = 100,
    input_identifier: Optional[str] = None,
    db_path: Union[Path, str] = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """List stored experiments ordered from newest to oldest."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        if input_identifier is not None:
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE input_identifier = ? ORDER BY id DESC LIMIT ?;",
                (str(input_identifier), int(limit)),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM experiments ORDER BY id DESC LIMIT ?;",
                (int(limit),),
            )
        return [_decode_row(r) for r in cursor.fetchall()]


def _save_benchmark_runs_to_sqlite(
    input_identifier: str,
    smiles: str,
    benchmark_id: Optional[str],
    descriptors_dict: Optional[Dict[str, Any]],
    qee: QEEResult,
    bm_res: MultiSeedBenchmarkResult,
    frozen_core: bool,
    active_space: Optional[Tuple[int, int]],
    maxiter: Optional[int],
    reps: int,
    db_path: Union[Path, str],
    use_benchmark_geometry: bool = False,
    geometry_seed: int = BATCH_GEOMETRY_SEED,
) -> List[int]:
    """Persist each individual seed run and the multi-seed aggregate record to SQLite."""
    stored_ids: List[int] = []
    qee_exp = {
        "decision": qee.decision,
        "confidence": qee.confidence,
        "reasons": qee.reasons,
        "warnings": qee.warnings + bm_res.warnings,
        "recommended_active_space": list(qee.recommended_active_space)
        if qee.recommended_active_space
        else None,
    }
    act_str = f"({bm_res.active_electrons}e,{bm_res.active_orbitals}o)"

    # 1. Save each individual seed run so any single seed can be replayed rapidly
    for sr in bm_res.seed_runs:
        seed_cfg = {
            "run_type": "single_seed_vqe",
            "input_identifier": input_identifier,
            "smiles": smiles,
            "benchmark_id": benchmark_id,
            "use_benchmark_geometry": bool(use_benchmark_geometry),
            "geometry_seed": int(geometry_seed),
            "basis": bm_res.basis,
            "frozen_core": frozen_core,
            "active_space": list(active_space) if active_space is not None else None,
            "mapping": bm_res.mapping,
            "ansatz": bm_res.ansatz,
            "optimizer": bm_res.optimizer,
            "seed": sr.seed,
            "maxiter": maxiter,
            "shots": bm_res.shots,
            "noise_mode": bm_res.noise_mode,
            "reps": reps,
        }
        row_id = save_experiment_record(
            {
                "input_identifier": input_identifier,
                "config_json": seed_cfg,
                "descriptors_json": descriptors_dict,
                "qee_decision": qee.decision or "UNKNOWN",
                "qee_explanation_json": qee_exp,
                "basis": bm_res.basis,
                "active_space": act_str,
                "mapping": bm_res.mapping,
                "ansatz": bm_res.ansatz,
                "optimizer": bm_res.optimizer,
                "backend": sr.backend,
                "shots": sr.shots,
                "seed": sr.seed,
                "noise_model": bm_res.noise_mode,
                "vqe_energy": sr.vqe_energy,
                "hf_energy": bm_res.hf_energy,
                "fci_energy": bm_res.matched_fci_energy,
                "error_ha": abs(sr.vqe_energy - bm_res.matched_fci_energy),
                "uncertainty_ha": bm_res.uncertainty_ha,
                "iterations": sr.iterations,
                "converged": sr.converged,
                "runtime_sec": sr.runtime_sec,
                "qubits": sr.num_qubits,
                "depth": sr.circuit_depth,
                "gate_count": sr.gate_count,
                "reliability": sr.reliability,
                "verdict": bm_res.verdict,
            },
            db_path=db_path,
        )
        stored_ids.append(row_id)

    # 2. Save the multi-seed aggregate benchmark row
    agg_cfg = {
        "run_type": "multi_seed_benchmark",
        "input_identifier": input_identifier,
        "smiles": smiles,
        "benchmark_id": benchmark_id,
        "use_benchmark_geometry": bool(use_benchmark_geometry),
        "geometry_seed": int(geometry_seed),
        "basis": bm_res.basis,
        "frozen_core": frozen_core,
        "active_space": list(active_space) if active_space is not None else None,
        "mapping": bm_res.mapping,
        "ansatz": bm_res.ansatz,
        "optimizer": bm_res.optimizer,
        "seeds": bm_res.seeds,
        "maxiter": maxiter,
        "shots": bm_res.shots,
        "noise_mode": bm_res.noise_mode,
        "reps": reps,
    }
    agg_id = save_experiment_record(
        {
            "input_identifier": input_identifier,
            "config_json": agg_cfg,
            "descriptors_json": descriptors_dict,
            "qee_decision": qee.decision or "UNKNOWN",
            "qee_explanation_json": qee_exp,
            "basis": bm_res.basis,
            "active_space": act_str,
            "mapping": bm_res.mapping,
            "ansatz": bm_res.ansatz,
            "optimizer": bm_res.optimizer,
            "backend": bm_res.seed_runs[0].backend,
            "shots": bm_res.shots,
            "seed": bm_res.seeds[0],
            "noise_model": bm_res.noise_mode,
            "vqe_energy": bm_res.mean_vqe_energy,
            "hf_energy": bm_res.hf_energy,
            "fci_energy": bm_res.matched_fci_energy,
            "error_ha": bm_res.abs_mean_error_ha,
            "uncertainty_ha": bm_res.uncertainty_ha,
            "iterations": int(round(bm_res.mean_iterations)),
            "converged": bm_res.all_converged,
            "runtime_sec": bm_res.total_vqe_runtime_sec,
            "qubits": bm_res.num_qubits,
            "depth": bm_res.circuit_depth,
            "gate_count": bm_res.gate_count,
            "reliability": bm_res.reliability,
            "verdict": bm_res.verdict,
        },
        db_path=db_path,
    )
    stored_ids.append(agg_id)
    return stored_ids


def _aggregate_seed_runs_to_benchmark(
    molecule_id: str,
    basis: str,
    seeds: Sequence[int],
    seed_runs: List[VQERunResult],
    matched_hf_energy: float,
    matched_fci_energy: float,
    full_fc_fci_energy: float,
    classical_runtime_sec: float,
    is_ideal: bool,
) -> MultiSeedBenchmarkResult:
    """Aggregate a list of VQERunResult seed runs into a MultiSeedBenchmarkResult."""
    energies = np.array([r.vqe_energy for r in seed_runs], dtype=float)
    iters = np.array([r.iterations for r in seed_runs], dtype=float)
    runtimes = np.array([r.runtime_sec for r in seed_runs], dtype=float)
    conv_flags = [bool(r.converged) for r in seed_runs]

    mean_vqe = float(np.mean(energies))
    min_vqe = float(np.min(energies))
    max_vqe = float(np.max(energies))
    uncertainty_ha = float(np.std(energies, ddof=1)) if len(energies) > 1 else 0.0

    mean_error_ha = mean_vqe - matched_fci_energy
    abs_mean_error_ha = abs(mean_error_ha)
    active_space_error_mha = (matched_fci_energy - full_fc_fci_energy) * 1000.0

    all_converged = all(conv_flags)
    converged_fraction = float(sum(conv_flags)) / float(len(conv_flags))
    mean_vqe_runtime = float(np.mean(runtimes))
    total_vqe_runtime = float(np.sum(runtimes))

    status, reliability, agg_warnings = classify_reliability(
        abs_error_ha=abs_mean_error_ha,
        uncertainty_ha=uncertainty_ha,
        converged=all_converged,
        active_space_error_mha=active_space_error_mha,
        vqe_energy=min_vqe if not is_ideal else mean_vqe,
        matched_fci_energy=matched_fci_energy,
        is_noisy=not is_ideal,
    )

    if mean_vqe > matched_hf_energy:
        agg_warnings.append(
            f"NOISE_ABOVE_HF_WARNING: Mean VQE energy ({mean_vqe:.6f} Ha) is higher (worse) than "
            f"classical mean-field Hartree-Fock ({matched_hf_energy:.6f} Ha) by "
            f"{(mean_vqe - matched_hf_energy)*1000:.2f} mHa due to gate/readout decoherence."
        )

    verdict = classify_advantage_verdict(
        converged=all_converged,
        num_seeds=len(seeds),
        reliability=reliability,
        abs_error_ha=abs_mean_error_ha,
        vqe_runtime_sec=mean_vqe_runtime,
        classical_runtime_sec=classical_runtime_sec,
        classical_exact_available=True,
        classical_reference_error_ha=0.0,
    )

    first = seed_runs[0]
    return MultiSeedBenchmarkResult(
        molecule_id=molecule_id,
        basis=basis,
        active_electrons=first.active_electrons,
        active_orbitals=first.active_orbitals,
        mapping=first.mapping,
        ansatz=first.ansatz,
        optimizer=first.optimizer,
        noise_mode=first.noise_model,
        shots=first.shots,
        seeds=[int(s) for s in seeds],
        num_qubits=first.num_qubits,
        circuit_depth=first.circuit_depth,
        gate_count=first.gate_count,
        num_parameters=first.num_parameters,
        hf_energy=matched_hf_energy,
        matched_fci_energy=matched_fci_energy,
        full_frozen_core_fci_energy=full_fc_fci_energy,
        mean_vqe_energy=mean_vqe,
        min_vqe_energy=min_vqe,
        max_vqe_energy=max_vqe,
        mean_error_ha=mean_error_ha,
        abs_mean_error_ha=abs_mean_error_ha,
        abs_mean_error_mha=abs_mean_error_ha * 1000.0,
        uncertainty_ha=uncertainty_ha,
        uncertainty_mha=uncertainty_ha * 1000.0,
        active_space_error_mha=active_space_error_mha,
        mean_iterations=float(np.mean(iters)),
        all_converged=all_converged,
        converged_fraction=converged_fraction,
        mean_vqe_runtime_sec=mean_vqe_runtime,
        total_vqe_runtime_sec=total_vqe_runtime,
        classical_runtime_sec=classical_runtime_sec,
        status=status,
        reliability=reliability,
        verdict=verdict,
        warnings=agg_warnings,
        seed_runs=seed_runs,
    )


def _coerce_pipeline_config(
    config: Optional[Any] = None,
    **overrides: Any,
) -> MoleculePipelineConfig:
    """Normalize None, dict, BatchConfig, or MoleculePipelineConfig into a MoleculePipelineConfig."""
    base_kwargs: Dict[str, Any] = {}
    if config is not None:
        if isinstance(config, MoleculePipelineConfig):
            base_kwargs = asdict(config)
        elif isinstance(config, dict):
            base_kwargs = dict(config)
        else:
            # Duck-type support for BatchConfig or custom config objects
            for attr in (
                "seeds",
                "geometry_seed",
                "capped_active_space",
                "active_space_override",
                "use_benchmark_geometry",
                "charge",
                "spin",
                "run_simulation",
                "run_noisy",
                "noisy_mode",
                "shots",
                "ansatz_type",
                "ideal_optimizer",
                "noisy_optimizer",
                "ideal_maxiter",
                "noisy_maxiter",
                "per_molecule_timeout_sec",
                "classical_sufficient",
                "reference_available",
                "persist_to_sqlite",
                "db_path",
                "name",
            ):
                if hasattr(config, attr):
                    base_kwargs[attr] = getattr(config, attr)
            # If coming from BatchConfig where use_benchmark_geometry is not defined, default to False
            if "use_benchmark_geometry" not in base_kwargs:
                base_kwargs["use_benchmark_geometry"] = False

    for k, v in overrides.items():
        if v is not None or k not in base_kwargs:
            base_kwargs[k] = v

    if "seeds" in base_kwargs and base_kwargs["seeds"] is not None:
        base_kwargs["seeds"] = tuple(int(s) for s in base_kwargs["seeds"])
    if (
        "capped_active_space" in base_kwargs
        and base_kwargs["capped_active_space"] is not None
    ):
        cap = base_kwargs["capped_active_space"]
        base_kwargs["capped_active_space"] = (int(cap[0]), int(cap[1]))
    if (
        "active_space_override" in base_kwargs
        and base_kwargs["active_space_override"] is not None
    ):
        ov = base_kwargs["active_space_override"]
        base_kwargs["active_space_override"] = (int(ov[0]), int(ov[1]))

    valid_fields = {f.name for f in MoleculePipelineConfig.__dataclass_fields__.values()}
    filtered_kwargs = {k: v for k, v in base_kwargs.items() if k in valid_fields}
    return MoleculePipelineConfig(**filtered_kwargs)


def _format_exception(exc: BaseException) -> str:
    """Format an exception with its class name and message, handling empty str(exc) cleanly."""
    msg = str(exc).strip()
    return f"{type(exc).__name__}: {msg}" if msg else f"{type(exc).__name__}"


def run_molecule_pipeline(
    smiles_or_preset: str,
    config: Optional[Any] = None,
    stage_callback: Optional[Callable[[str], None]] = None,
    simulation_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    **kwargs: Any,
) -> PipelineExecutionResult:
    """Shared end-to-end molecule pipeline used by both Tab 1 and Batch Screening.

    Performs:
      1. SMILES / preset resolution and RDKit descriptor validation.
      2. Charge (neutral), spin/radical (closed-shell singlet), and supported-element checks.
      3. Qubit, depth, and gate estimation + QEE v1 routing.
      4. For SIMULATED_QUANTUM or HYBRID:
         - Seeded RDKit 3D geometry (or exact benchmark geometry when preset benchmark mode is active).
         - Capped active space with `min(requested, available)` on occupied and virtual orbitals.
         - Matched PySCF HF & CASCI/FCI reference + full frozen-core CASCI on the exact same geometry.
         - Ideal statevector VQE across `config.seeds` (+ optional noisy Qiskit Aer VQE).
         - Reliability classification, active-space truncation error (`active_space_error_mha`), and verdict.
         - Main-thread stage progress forwarding and per-molecule timeout enforcement.
    """
    t0 = time.perf_counter()
    cfg = _coerce_pipeline_config(config, **kwargs)
    raw_input = str(smiles_or_preset).strip() if isinstance(smiles_or_preset, str) else ""

    def _emit_stage(stage_msg: str) -> None:
        if stage_callback is not None:
            try:
                stage_callback(stage_msg)
            except Exception:
                pass

    current_stage = "Validating SMILES & computing RDKit descriptors"
    _emit_stage(current_stage)

    resolved_smiles, benchmark_id = resolve_input_smiles(raw_input)
    val = validate_and_describe(resolved_smiles)

    if not val.is_valid or val.descriptors is None:
        err_reason = val.error or f"Invalid SMILES: '{raw_input}' could not be parsed by RDKit."
        rej_qee = QEEResult(
            is_valid=False,
            decision="REJECTED",
            confidence=0.0,
            reasons=[err_reason],
            warnings=[],
            features={"input_smiles": raw_input},
            error=err_reason,
        )
        cat, msg = build_pipeline_status_message(
            qee_decision="REJECTED",
            outcome="REJECTED_INVALID_SMILES",
            reasons=[err_reason],
            error_stage=current_stage,
            error_message=err_reason,
        )
        return PipelineExecutionResult(
            input_identifier=raw_input,
            smiles=resolved_smiles,
            canonical_smiles=None,
            benchmark_id=benchmark_id,
            descriptors=None,
            qee=rej_qee,
            active_space_summary=None,
            ideal_benchmark=None,
            noisy_benchmark=None,
            charge=cfg.charge,
            spin=cfg.spin,
            outcome="REJECTED_INVALID_SMILES",
            status_category=cat,
            status_message=msg,
            error_stage=current_stage,
            error_message=err_reason,
            runtime_sec=time.perf_counter() - t0,
        )

    desc = val.descriptors
    descriptors_dict = desc.to_dict()
    mol = Chem.MolFromSmiles(desc.canonical_smiles, sanitize=True)
    mol_h = Chem.AddHs(mol)

    formal_charge = int(cfg.charge) if int(cfg.charge) != 0 else int(Chem.GetFormalCharge(mol))
    total_electrons = sum(a.GetAtomicNum() for a in mol_h.GetAtoms()) - formal_charge
    num_radicals = int(Descriptors.NumRadicalElectrons(mol))

    current_stage = "Checking charge, spin, and element support"

    # 1. Reject charged species
    if formal_charge != 0:
        reason = (
            f"Charged molecule (charge={formal_charge}) rejected: "
            "workflow requires neutral (charge=0) closed-shell species."
        )
        rej_qee = QEEResult(
            is_valid=False,
            decision="REJECTED",
            confidence=0.0,
            reasons=[reason],
            warnings=[],
            features={"input_smiles": raw_input, "canonical_smiles": desc.canonical_smiles, "charge": formal_charge},
            error=reason,
        )
        cat, msg = build_pipeline_status_message(
            qee_decision="REJECTED",
            outcome="REJECTED_CHARGED",
            reasons=[reason],
            error_stage=current_stage,
            error_message=reason,
        )
        return PipelineExecutionResult(
            input_identifier=raw_input,
            smiles=resolved_smiles,
            canonical_smiles=desc.canonical_smiles,
            benchmark_id=benchmark_id,
            descriptors=descriptors_dict,
            qee=rej_qee,
            active_space_summary=None,
            ideal_benchmark=None,
            noisy_benchmark=None,
            charge=formal_charge,
            spin=cfg.spin,
            outcome="REJECTED_CHARGED",
            status_category=cat,
            status_message=msg,
            error_stage=current_stage,
            error_message=reason,
            runtime_sec=time.perf_counter() - t0,
        )

    # 2. Reject open-shell / radical species
    if int(cfg.spin) != 0 or num_radicals > 0 or (total_electrons % 2 != 0):
        reason = (
            f"Open-shell molecule (spin={cfg.spin}, radical_electrons={num_radicals}, "
            f"total_electrons={total_electrons}) rejected: workflow requires closed-shell singlet (spin=0) species."
        )
        rej_qee = QEEResult(
            is_valid=False,
            decision="REJECTED",
            confidence=0.0,
            reasons=[reason],
            warnings=[],
            features={"input_smiles": raw_input, "canonical_smiles": desc.canonical_smiles, "spin": cfg.spin},
            error=reason,
        )
        cat, msg = build_pipeline_status_message(
            qee_decision="REJECTED",
            outcome="REJECTED_OPEN_SHELL",
            reasons=[reason],
            error_stage=current_stage,
            error_message=reason,
        )
        return PipelineExecutionResult(
            input_identifier=raw_input,
            smiles=resolved_smiles,
            canonical_smiles=desc.canonical_smiles,
            benchmark_id=benchmark_id,
            descriptors=descriptors_dict,
            qee=rej_qee,
            active_space_summary=None,
            ideal_benchmark=None,
            noisy_benchmark=None,
            charge=formal_charge,
            spin=cfg.spin,
            outcome="REJECTED_OPEN_SHELL",
            status_category=cat,
            status_message=msg,
            error_stage=current_stage,
            error_message=reason,
            runtime_sec=time.perf_counter() - t0,
        )

    # 3. Reject unsupported elements
    unsupported = [el for el in desc.elements if el not in SUPPORTED_ELEMENTS]
    if unsupported:
        reason = (
            f"Unsupported element(s) for STO-3G workflow: {', '.join(unsupported)}. "
            f"Supported elements: {', '.join(sorted(SUPPORTED_ELEMENTS))}."
        )
        rej_qee = QEEResult(
            is_valid=False,
            decision="REJECTED",
            confidence=0.0,
            reasons=[reason],
            warnings=[],
            features={"input_smiles": raw_input, "elements": desc.elements},
            error=reason,
        )
        cat, msg = build_pipeline_status_message(
            qee_decision="REJECTED",
            outcome="REJECTED_UNSUPPORTED_ELEMENT",
            reasons=[reason],
            error_stage=current_stage,
            error_message=reason,
        )
        return PipelineExecutionResult(
            input_identifier=raw_input,
            smiles=resolved_smiles,
            canonical_smiles=desc.canonical_smiles,
            benchmark_id=benchmark_id,
            descriptors=descriptors_dict,
            qee=rej_qee,
            active_space_summary=None,
            ideal_benchmark=None,
            noisy_benchmark=None,
            charge=formal_charge,
            spin=cfg.spin,
            outcome="REJECTED_UNSUPPORTED_ELEMENT",
            status_category=cat,
            status_message=msg,
            error_stage=current_stage,
            error_message=reason,
            runtime_sec=time.perf_counter() - t0,
        )

    # 4. Estimate resources and run QEE v1
    current_stage = "Estimating qubits/depth and running QEE v1"
    _emit_stage(current_stage)

    mapping_hint = "parity" if benchmark_id in {"H2", "LiH"} else None
    est_summary = estimate_qubits_from_smiles(
        desc.canonical_smiles if benchmark_id is None else resolved_smiles,
        frozen_core=True,
        mapping=mapping_hint,
    )
    qee = evaluate_eligibility(
        raw_input,
        reference_available=cfg.reference_available,
        classical_sufficient=cfg.classical_sufficient,
        recommended_active_space=cfg.active_space_override,
    )

    qee_decision = qee.decision or "DEFERRED"

    # 5. Non-simulated decisions: CLASSICAL_ONLY, DEFERRED, or run_simulation=False
    if qee_decision in {"CLASSICAL_ONLY", "DEFERRED"} or not cfg.run_simulation:
        if qee_decision in {"CLASSICAL_ONLY", "DEFERRED"}:
            outcome = "SCREENED_ONLY"
        else:
            outcome = "SKIPPED_SIMULATION_LIMIT"

        stored_ids: List[int] = []
        primary_id: Optional[int] = None
        if cfg.persist_to_sqlite:
            verdict_str = (
                "NO_DEMONSTRATED_ADVANTAGE"
                if qee_decision == "CLASSICAL_ONLY"
                else "INSUFFICIENT_EVIDENCE"
            )
            qee_cfg = {
                "run_type": "qee_only",
                "input_identifier": raw_input,
                "smiles": resolved_smiles,
                "benchmark_id": benchmark_id,
                "classical_sufficient": cfg.classical_sufficient,
                "reference_available": cfg.reference_available,
            }
            qee_exp = {
                "decision": qee.decision,
                "confidence": qee.confidence,
                "reasons": qee.reasons,
                "warnings": qee.warnings,
                "recommended_active_space": list(qee.recommended_active_space)
                if qee.recommended_active_space
                else None,
            }
            act_e = qee.features.get("active_electrons", est_summary.active_electrons)
            act_o = qee.features.get("active_spatial_orbitals", est_summary.active_spatial_orbitals)
            primary_id = save_experiment_record(
                {
                    "input_identifier": raw_input,
                    "config_json": qee_cfg,
                    "descriptors_json": descriptors_dict,
                    "qee_decision": qee_decision,
                    "qee_explanation_json": qee_exp,
                    "basis": DEFAULT_BASIS,
                    "active_space": f"({act_e}e,{act_o}o)",
                    "mapping": qee.features.get("mapping", est_summary.mapping),
                    "ansatz": None,
                    "optimizer": None,
                    "backend": None,
                    "shots": None,
                    "seed": None,
                    "noise_model": None,
                    "vqe_energy": None,
                    "hf_energy": None,
                    "fci_energy": None,
                    "error_ha": None,
                    "uncertainty_ha": None,
                    "iterations": None,
                    "converged": None,
                    "runtime_sec": 0.0,
                    "qubits": qee.features.get("estimated_qubits", est_summary.num_qubits),
                    "depth": qee.features.get("estimated_circuit_depth", est_summary.estimated_circuit_depth),
                    "gate_count": None,
                    "reliability": None,
                    "verdict": verdict_str,
                },
                db_path=cfg.db_path,
            )
            stored_ids = [primary_id]

        cat, msg = build_pipeline_status_message(
            qee_decision=qee_decision,
            outcome=outcome,
            reasons=qee.reasons,
            primary_experiment_id=primary_id,
        )
        return PipelineExecutionResult(
            input_identifier=raw_input,
            smiles=resolved_smiles,
            canonical_smiles=desc.canonical_smiles,
            benchmark_id=benchmark_id,
            descriptors=descriptors_dict,
            qee=qee,
            active_space_summary=est_summary,
            ideal_benchmark=None,
            noisy_benchmark=None,
            stored_experiment_ids=stored_ids,
            primary_experiment_id=primary_id,
            charge=formal_charge,
            spin=cfg.spin,
            outcome=outcome,
            status_category=cat,
            status_message=msg,
            effective_active_space=(est_summary.active_electrons, est_summary.active_spatial_orbitals),
            runtime_sec=time.perf_counter() - t0,
        )

    # 6. Eligible for Quantum Simulation (SIMULATED_QUANTUM or HYBRID):
    # Determine whether to use preset benchmark coordinates or seeded RDKit 3D coordinates.
    if cfg.use_benchmark_geometry is not None:
        use_bm_geom = bool(cfg.use_benchmark_geometry and benchmark_id in {"H2", "LiH", "H2O"})
    else:
        # Use preset benchmark geometry only when explicitly invoked by preset ID ("H2", "LiH", "H2O")
        use_bm_geom = raw_input.upper() in {"H2", "LIH", "H2O"} and benchmark_id in {"H2", "LiH", "H2O"}

    # Determine requested active space before applying cap_active_space(requested, available)
    if cfg.active_space_override is not None:
        requested_space: Optional[Tuple[int, int]] = cfg.active_space_override
    elif use_bm_geom:
        # For preset benchmarks H2 and LiH in SIMULATED_QUANTUM mode, use full frozen-core unless HYBRID
        requested_space = qee.recommended_active_space if qee_decision == "HYBRID" else None
    else:
        # For custom SMILES and batch screening, apply capped active space policy (default (4, 4))
        requested_space = (
            cfg.capped_active_space
            if cfg.capped_active_space is not None
            else (qee.recommended_active_space if qee_decision == "HYBRID" else BATCH_CAPPED_ACTIVE_SPACE)
        )

    eff_e, eff_o = cap_active_space(
        requested_active_space=requested_space,
        available_electrons=est_summary.active_electrons,
        available_orbitals=est_summary.active_spatial_orbitals,
    )
    effective_space = (eff_e, eff_o)
    geom_caption = (
        f"Exact benchmark geometry from data/benchmarks.json ({get_benchmark(benchmark_id).geometry_note})."
        if use_bm_geom and benchmark_id is not None
        else BATCH_CUSTOM_GEOMETRY_CAPTION
    )
    geom_source = "benchmark_exact" if use_bm_geom else "rdkit_mmff_approx"
    display_id = cfg.name or (benchmark_id if use_bm_geom and benchmark_id else desc.canonical_smiles)

    last_stage_holder: List[str] = ["Initializing quantum simulation"]

    def _core_simulation(worker_stage_cb: Callable[[str], None]) -> Dict[str, Any]:
        if simulation_fn is not None:
            return simulation_fn(
                display_id,
                desc.canonical_smiles,
                eff_e,
                eff_o,
                cfg,
                worker_stage_cb,
            )

        if use_bm_geom and benchmark_id is not None:
            worker_stage_cb(f"Loading preset benchmark 3D geometry ({benchmark_id})")
            bm_obj = get_benchmark(benchmark_id)
            atoms: List[AtomCoord] = list(bm_obj.atoms)
            basis_name = bm_obj.basis
        else:
            worker_stage_cb("Generating seeded RDKit 3D geometry")
            atoms, _, _, _ = smiles_to_3d_coordinates(
                desc.canonical_smiles, seed=int(cfg.geometry_seed)
            )
            basis_name = DEFAULT_BASIS

        worker_stage_cb(f"Computing PySCF HF & CASCI({eff_e}e, {eff_o}o) reference")
        matched_ref = run_classical_reference(
            atoms=atoms,
            basis=basis_name,
            charge=0,
            spin=0,
            frozen_core=True,
            active_space=(eff_e, eff_o),
            molecule_id=display_id,
        )
        full_fc_ref = run_classical_reference(
            atoms=atoms,
            basis=basis_name,
            charge=0,
            spin=0,
            frozen_core=True,
            active_space=None,
            molecule_id=display_id,
        )

        worker_stage_cb(f"Preparing qubit Hamiltonian for ({eff_e}e, {eff_o}o)")
        mol_key_for_mapping = benchmark_id if benchmark_id in {"H2", "LiH"} else desc.canonical_smiles
        prepared = prepare_active_space_problem(
            atoms=atoms,
            basis=basis_name,
            charge=0,
            spin=0,
            frozen_core=True,
            active_space=(eff_e, eff_o),
            mapping=None,
            molecule_id=mol_key_for_mapping,
        )

        worker_stage_cb(f"Running Ideal VQE across {len(cfg.seeds)} seed(s)")
        ideal_opt = cfg.ideal_optimizer or "SLSQP"
        ideal_maxiter = int(cfg.ideal_maxiter) if cfg.ideal_maxiter is not None else 200
        ideal_runs: List[VQERunResult] = []
        for s in cfg.seeds:
            r_ideal = run_vqe_from_prepared(
                prepared=prepared,
                ansatz_type=cfg.ansatz_type,
                optimizer_name=ideal_opt,
                seed=int(s),
                maxiter=ideal_maxiter,
                shots=None,
                noise_model="ideal",
                noise_model_name="ideal",
                matched_reference_energy=matched_ref.fci_energy,
                full_frozen_core_energy=full_fc_ref.fci_energy,
            )
            ideal_runs.append(r_ideal)

        ideal_bm = _aggregate_seed_runs_to_benchmark(
            molecule_id=display_id,
            basis=basis_name,
            seeds=cfg.seeds,
            seed_runs=ideal_runs,
            matched_hf_energy=matched_ref.hf_energy,
            matched_fci_energy=matched_ref.fci_energy,
            full_fc_fci_energy=full_fc_ref.fci_energy,
            classical_runtime_sec=matched_ref.total_runtime_sec,
            is_ideal=True,
        )

        noisy_bm: Optional[MultiSeedBenchmarkResult] = None
        if cfg.run_noisy:
            worker_stage_cb(
                f"Running Noisy VQE [{cfg.noisy_mode}] across {len(cfg.seeds)} seed(s)"
            )
            noisy_opt = cfg.noisy_optimizer or "COBYLA"
            noisy_maxiter = int(cfg.noisy_maxiter) if cfg.noisy_maxiter is not None else 80
            noisy_runs: List[VQERunResult] = []
            for s in cfg.seeds:
                r_noisy = run_vqe_from_prepared(
                    prepared=prepared,
                    ansatz_type=cfg.ansatz_type,
                    optimizer_name=noisy_opt,
                    seed=int(s),
                    maxiter=noisy_maxiter,
                    shots=int(cfg.shots),
                    noise_model=cfg.noisy_mode,
                    noise_model_name=cfg.noisy_mode,
                    matched_reference_energy=matched_ref.fci_energy,
                    full_frozen_core_energy=full_fc_ref.fci_energy,
                )
                noisy_runs.append(r_noisy)

            noisy_bm = _aggregate_seed_runs_to_benchmark(
                molecule_id=display_id,
                basis=basis_name,
                seeds=cfg.seeds,
                seed_runs=noisy_runs,
                matched_hf_energy=matched_ref.hf_energy,
                matched_fci_energy=matched_ref.fci_energy,
                full_fc_fci_energy=full_fc_ref.fci_energy,
                classical_runtime_sec=matched_ref.total_runtime_sec,
                is_ideal=False,
            )

        return {
            "prepared_summary": prepared.summary,
            "ideal_benchmark": ideal_bm,
            "noisy_benchmark": noisy_bm,
        }

    # Execute with optional per-molecule timeout while keeping all stage_callback calls on the main thread
    try:
        if cfg.per_molecule_timeout_sec is not None and float(cfg.per_molecule_timeout_sec) > 0:
            timeout_limit = float(cfg.per_molecule_timeout_sec)
            stage_q: queue.Queue[str] = queue.Queue()

            def _thread_stage_cb(msg: str) -> None:
                last_stage_holder[0] = msg
                stage_q.put(msg)

            executor = ThreadPoolExecutor(max_workers=1)
            try:
                sim_start = time.perf_counter()
                future = executor.submit(_core_simulation, _thread_stage_cb)
                while not future.done():
                    while True:
                        try:
                            queued_msg = stage_q.get_nowait()
                            _emit_stage(queued_msg)
                        except queue.Empty:
                            break
                    if (time.perf_counter() - sim_start) > timeout_limit:
                        raise FuturesTimeoutError(
                            f"Exceeded per-molecule timeout of {timeout_limit:.2f}s"
                        )
                    time.sleep(0.01)

                while True:
                    try:
                        queued_msg = stage_q.get_nowait()
                        _emit_stage(queued_msg)
                    except queue.Empty:
                        break
                sim_out = future.result(timeout=0.1)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        else:
            def _direct_stage_cb(msg: str) -> None:
                last_stage_holder[0] = msg
                _emit_stage(msg)

            sim_out = _core_simulation(_direct_stage_cb)

    except FuturesTimeoutError:
        failed_stage = last_stage_holder[0]
        timeout_val = float(cfg.per_molecule_timeout_sec or 0.0)
        err_msg = (
            f"TimeoutError: Simulation exceeded per-molecule timeout ({timeout_val:.2f}s) "
            f"during stage '{failed_stage}'."
        )
        reasons_with_err = list(qee.reasons) + [err_msg]
        cat, msg = build_pipeline_status_message(
            qee_decision=qee_decision,
            outcome="TIMEOUT",
            reasons=reasons_with_err,
            error_stage=failed_stage,
            error_message=err_msg,
        )
        return PipelineExecutionResult(
            input_identifier=raw_input,
            smiles=resolved_smiles,
            canonical_smiles=desc.canonical_smiles,
            benchmark_id=benchmark_id,
            descriptors=descriptors_dict,
            qee=qee,
            active_space_summary=est_summary,
            ideal_benchmark=None,
            noisy_benchmark=None,
            charge=formal_charge,
            spin=cfg.spin,
            outcome="TIMEOUT",
            status_category=cat,
            status_message=msg,
            geometry_source=geom_source,
            geometry_caption=geom_caption,
            effective_active_space=effective_space,
            error_stage=failed_stage,
            error_message=err_msg,
            runtime_sec=time.perf_counter() - t0,
        )
    except Exception as exc:
        failed_stage = last_stage_holder[0]
        err_msg = _format_exception(exc)
        reasons_with_err = list(qee.reasons) + [
            f"Simulation failed at stage '{failed_stage}': {err_msg}"
        ]
        cat, msg = build_pipeline_status_message(
            qee_decision=qee_decision,
            outcome="SIMULATION_ERROR",
            reasons=reasons_with_err,
            error_stage=failed_stage,
            error_message=err_msg,
        )
        return PipelineExecutionResult(
            input_identifier=raw_input,
            smiles=resolved_smiles,
            canonical_smiles=desc.canonical_smiles,
            benchmark_id=benchmark_id,
            descriptors=descriptors_dict,
            qee=qee,
            active_space_summary=est_summary,
            ideal_benchmark=None,
            noisy_benchmark=None,
            charge=formal_charge,
            spin=cfg.spin,
            outcome="SIMULATION_ERROR",
            status_category=cat,
            status_message=msg,
            geometry_source=geom_source,
            geometry_caption=geom_caption,
            effective_active_space=effective_space,
            error_stage=failed_stage,
            error_message=err_msg,
            runtime_sec=time.perf_counter() - t0,
        )

    # Support custom simulation_fn in unit tests that returns a raw dict of metrics
    if "ideal_benchmark" not in sim_out:
        synth_seed_run = VQERunResult(
            molecule_id=display_id,
            basis=DEFAULT_BASIS,
            frozen_core=True,
            active_electrons=eff_e,
            active_orbitals=eff_o,
            mapping=est_summary.mapping,
            num_qubits=int(sim_out.get("qubits", est_summary.num_qubits)),
            ansatz=cfg.ansatz_type,
            optimizer=cfg.ideal_optimizer or "SLSQP",
            backend="AerSimulator(statevector)",
            noise_model="ideal",
            shots=None,
            seed=int(cfg.seeds[0]) if cfg.seeds else 42,
            vqe_energy=float(sim_out.get("ideal_vqe_energy", 0.0)),
            electronic_energy=float(sim_out.get("ideal_vqe_energy", 0.0)),
            energy_offset=0.0,
            iterations=1,
            converged=True,
            runtime_sec=0.01,
            circuit_depth=int(sim_out.get("depth", est_summary.estimated_circuit_depth)),
            gate_count=int(sim_out.get("gate_count", 0)),
            num_parameters=est_summary.estimated_uccsd_parameters,
            optimal_parameters=[],
            active_space_error_mha=float(sim_out.get("active_space_error_mha", 0.0)),
            status="VALID",
            reliability=str(sim_out.get("reliability", "HIGH")),
            warnings=list(sim_out.get("extra_warnings", [])),
        )
        ideal_bm = MultiSeedBenchmarkResult(
            molecule_id=display_id,
            basis=DEFAULT_BASIS,
            active_electrons=eff_e,
            active_orbitals=eff_o,
            mapping=est_summary.mapping,
            ansatz=cfg.ansatz_type,
            optimizer=cfg.ideal_optimizer or "SLSQP",
            noise_mode="ideal",
            shots=None,
            seeds=list(cfg.seeds),
            num_qubits=synth_seed_run.num_qubits,
            circuit_depth=synth_seed_run.circuit_depth,
            gate_count=synth_seed_run.gate_count,
            num_parameters=synth_seed_run.num_parameters,
            hf_energy=float(sim_out.get("hf_energy", 0.0)),
            matched_fci_energy=float(sim_out.get("fci_energy", 0.0)),
            full_frozen_core_fci_energy=float(sim_out.get("fci_energy", 0.0)),
            mean_vqe_energy=float(sim_out.get("ideal_vqe_energy", 0.0)),
            min_vqe_energy=float(sim_out.get("ideal_vqe_energy", 0.0)),
            max_vqe_energy=float(sim_out.get("ideal_vqe_energy", 0.0)),
            mean_error_ha=float(sim_out.get("ideal_error_mha", 0.0)) / 1000.0,
            abs_mean_error_ha=float(sim_out.get("ideal_error_mha", 0.0)) / 1000.0,
            abs_mean_error_mha=float(sim_out.get("ideal_error_mha", 0.0)),
            uncertainty_ha=float(sim_out.get("ideal_uncertainty_mha", 0.0)) / 1000.0,
            uncertainty_mha=float(sim_out.get("ideal_uncertainty_mha", 0.0)),
            active_space_error_mha=float(sim_out.get("active_space_error_mha", 0.0)),
            mean_iterations=1.0,
            all_converged=True,
            converged_fraction=1.0,
            mean_vqe_runtime_sec=0.01,
            total_vqe_runtime_sec=0.01,
            classical_runtime_sec=0.01,
            status="VALID",
            reliability=str(sim_out.get("reliability", "HIGH")),
            verdict=str(sim_out.get("verdict", "NO_DEMONSTRATED_ADVANTAGE")),
            warnings=list(sim_out.get("extra_warnings", [])),
            seed_runs=[synth_seed_run],
        )
        noisy_bm = None
        prep_summary = est_summary
    else:
        prep_summary = sim_out["prepared_summary"]
        ideal_bm = sim_out["ideal_benchmark"]
        noisy_bm = sim_out["noisy_benchmark"]

    all_ids: List[int] = []
    if cfg.persist_to_sqlite:
        ideal_ids = _save_benchmark_runs_to_sqlite(
            input_identifier=raw_input,
            smiles=resolved_smiles,
            benchmark_id=benchmark_id,
            descriptors_dict=descriptors_dict,
            qee=qee,
            bm_res=ideal_bm,
            frozen_core=True,
            active_space=effective_space,
            maxiter=cfg.ideal_maxiter,
            reps=1,
            db_path=cfg.db_path,
            use_benchmark_geometry=use_bm_geom,
            geometry_seed=cfg.geometry_seed,
        )
        all_ids.extend(ideal_ids)

        if noisy_bm is not None:
            noisy_ids = _save_benchmark_runs_to_sqlite(
                input_identifier=raw_input,
                smiles=resolved_smiles,
                benchmark_id=benchmark_id,
                descriptors_dict=descriptors_dict,
                qee=qee,
                bm_res=noisy_bm,
                frozen_core=True,
                active_space=effective_space,
                maxiter=cfg.noisy_maxiter,
                reps=1,
                db_path=cfg.db_path,
                use_benchmark_geometry=use_bm_geom,
                geometry_seed=cfg.geometry_seed,
            )
            all_ids.extend(noisy_ids)

    primary_exp_id = all_ids[0] if all_ids else None
    cat, msg = build_pipeline_status_message(
        qee_decision=qee_decision,
        outcome="SIMULATED",
        reasons=qee.reasons,
        ideal_benchmark=ideal_bm,
        primary_experiment_id=primary_exp_id,
    )

    return PipelineExecutionResult(
        input_identifier=raw_input,
        smiles=resolved_smiles,
        canonical_smiles=desc.canonical_smiles,
        benchmark_id=benchmark_id,
        descriptors=descriptors_dict,
        qee=qee,
        active_space_summary=prep_summary,
        ideal_benchmark=ideal_bm,
        noisy_benchmark=noisy_bm,
        stored_experiment_ids=all_ids,
        primary_experiment_id=primary_exp_id,
        charge=formal_charge,
        spin=cfg.spin,
        outcome="SIMULATED",
        status_category=cat,
        status_message=msg,
        geometry_source=geom_source,
        geometry_caption=geom_caption,
        effective_active_space=effective_space,
        active_space_error_mha=ideal_bm.active_space_error_mha,
        runtime_sec=time.perf_counter() - t0,
    )


def run_qmed_pipeline(
    input_identifier: str,
    seeds: Sequence[int] = DEFAULT_BENCHMARK_SEEDS,
    run_noisy: bool = True,
    noisy_mode: str = "depolarizing_readout",
    shots: int = DEFAULT_SHOTS,
    ansatz_type: str = "UCCSD",
    ideal_optimizer: Optional[str] = "SLSQP",
    noisy_optimizer: Optional[str] = "COBYLA",
    ideal_maxiter: Optional[int] = None,
    noisy_maxiter: Optional[int] = None,
    active_space_override: Optional[Tuple[int, int]] = None,
    classical_sufficient: bool = False,
    reference_available: bool = True,
    use_benchmark_geometry: Optional[bool] = None,
    geometry_seed: int = BATCH_GEOMETRY_SEED,
    capped_active_space: Optional[Tuple[int, int]] = BATCH_CAPPED_ACTIVE_SPACE,
    db_path: Union[Path, str] = DEFAULT_DB_PATH,
) -> PipelineExecutionResult:
    """Execute the full Q-Med workflow via the shared `run_molecule_pipeline` and persist to SQLite."""
    cfg = MoleculePipelineConfig(
        seeds=tuple(int(s) for s in seeds),
        geometry_seed=int(geometry_seed),
        capped_active_space=capped_active_space,
        active_space_override=active_space_override,
        use_benchmark_geometry=use_benchmark_geometry,
        run_simulation=True,
        run_noisy=bool(run_noisy),
        noisy_mode=str(noisy_mode),
        shots=int(shots),
        ansatz_type=str(ansatz_type),
        ideal_optimizer=ideal_optimizer,
        noisy_optimizer=noisy_optimizer,
        ideal_maxiter=ideal_maxiter,
        noisy_maxiter=noisy_maxiter,
        classical_sufficient=bool(classical_sufficient),
        reference_available=bool(reference_available),
        persist_to_sqlite=True,
        db_path=db_path,
    )
    return run_molecule_pipeline(input_identifier, config=cfg)


def replay_experiment(
    experiment_id: int,
    db_path: Union[Path, str] = DEFAULT_DB_PATH,
    tolerance_ha: float = 1e-7,
) -> ReplayResult:
    """Replay a stored experiment by ID from its saved configuration and confirm reproducibility."""
    stored = get_experiment_by_id(experiment_id, db_path=db_path)
    cfg = stored["config"]
    run_type = str(cfg.get("run_type", "single_seed_vqe"))
    input_id = str(cfg.get("input_identifier", stored["input_identifier"]))

    act_raw = cfg.get("active_space")
    active_space: Optional[Tuple[int, int]] = (
        (int(act_raw[0]), int(act_raw[1]))
        if isinstance(act_raw, list) and len(act_raw) == 2
        else None
    )

    qee = evaluate_eligibility(
        input_id,
        reference_available=bool(cfg.get("reference_available", True)),
        classical_sufficient=bool(cfg.get("classical_sufficient", False)),
        recommended_active_space=active_space,
    )
    replayed_decision = qee.decision or "UNKNOWN"

    if run_type == "qee_only" or stored["vqe_energy"] is None:
        reproduced = replayed_decision == stored["qee_decision"]
        return ReplayResult(
            experiment_id=int(experiment_id),
            input_identifier=input_id,
            run_type=run_type,
            stored_qee_decision=str(stored["qee_decision"]),
            replayed_qee_decision=replayed_decision,
            stored_vqe_energy=None,
            replayed_vqe_energy=None,
            energy_diff_ha=None,
            stored_iterations=None,
            replayed_iterations=None,
            stored_reliability=stored.get("reliability"),
            replayed_reliability=None,
            stored_verdict=stored.get("verdict"),
            replayed_verdict=stored.get("verdict"),
            reproduced=reproduced,
            stored_record=stored,
        )

    ansatz = str(cfg.get("ansatz") or stored.get("ansatz") or "UCCSD")
    optimizer = str(cfg.get("optimizer") or stored.get("optimizer") or "SLSQP")
    maxiter = cfg.get("maxiter")
    shots = cfg.get("shots", stored.get("shots"))
    noise_mode = str(cfg.get("noise_mode") or stored.get("noise_model") or "ideal")
    is_noisy = noise_mode.strip().lower() not in {"ideal", "none", "exact", "statevector"}
    use_bm_geom = cfg.get("use_benchmark_geometry")
    geometry_seed = int(cfg.get("geometry_seed", BATCH_GEOMETRY_SEED))

    if run_type == "multi_seed_benchmark":
        seeds = tuple(int(s) for s in cfg.get("seeds", [stored.get("seed", 42)]))
    else:
        seeds = (int(cfg.get("seed", stored.get("seed", 42))),)

    pipe_cfg = MoleculePipelineConfig(
        seeds=seeds,
        geometry_seed=geometry_seed,
        active_space_override=active_space,
        use_benchmark_geometry=use_bm_geom,
        run_simulation=True,
        run_noisy=is_noisy,
        noisy_mode=noise_mode if is_noisy else "depolarizing_readout",
        shots=int(shots) if shots is not None else DEFAULT_SHOTS,
        ansatz_type=ansatz,
        ideal_optimizer=optimizer if not is_noisy else "SLSQP",
        noisy_optimizer=optimizer if is_noisy else "COBYLA",
        ideal_maxiter=maxiter if not is_noisy else 10,
        noisy_maxiter=maxiter if is_noisy else 10,
        persist_to_sqlite=False,
        db_path=db_path,
    )
    replayed_pipe = run_molecule_pipeline(input_id, config=pipe_cfg)
    target_bm = replayed_pipe.noisy_benchmark if is_noisy else replayed_pipe.ideal_benchmark

    if target_bm is None:
        return ReplayResult(
            experiment_id=int(experiment_id),
            input_identifier=input_id,
            run_type=run_type,
            stored_qee_decision=str(stored["qee_decision"]),
            replayed_qee_decision=replayed_decision,
            stored_vqe_energy=float(stored["vqe_energy"]),
            replayed_vqe_energy=None,
            energy_diff_ha=None,
            stored_iterations=int(stored["iterations"]) if stored.get("iterations") is not None else None,
            replayed_iterations=None,
            stored_reliability=stored.get("reliability"),
            replayed_reliability=None,
            stored_verdict=stored.get("verdict"),
            replayed_verdict=None,
            reproduced=False,
            stored_record=stored,
        )

    if run_type == "multi_seed_benchmark":
        replayed_energy = target_bm.mean_vqe_energy
        replayed_iters = int(round(target_bm.mean_iterations))
        replayed_rel = target_bm.reliability
        replayed_verdict = target_bm.verdict
    else:
        sr = target_bm.seed_runs[0]
        replayed_energy = sr.vqe_energy
        replayed_iters = sr.iterations
        replayed_rel = sr.reliability
        replayed_verdict = stored.get("verdict")

    stored_energy = float(stored["vqe_energy"])
    diff_ha = abs(replayed_energy - stored_energy)
    reproduced = (
        replayed_decision == stored["qee_decision"]
        and diff_ha <= tolerance_ha
        and replayed_rel == stored.get("reliability")
    )

    return ReplayResult(
        experiment_id=int(experiment_id),
        input_identifier=input_id,
        run_type=run_type,
        stored_qee_decision=str(stored["qee_decision"]),
        replayed_qee_decision=replayed_decision,
        stored_vqe_energy=stored_energy,
        replayed_vqe_energy=replayed_energy,
        energy_diff_ha=diff_ha,
        stored_iterations=int(stored["iterations"]) if stored.get("iterations") is not None else None,
        replayed_iterations=replayed_iters,
        stored_reliability=stored.get("reliability"),
        replayed_reliability=replayed_rel,
        stored_verdict=stored.get("verdict"),
        replayed_verdict=replayed_verdict,
        reproduced=reproduced,
        stored_record=stored,
    )
