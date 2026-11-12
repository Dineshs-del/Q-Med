"""SQLite persistence layer, end-to-end Q-Med pipeline, and deterministic experiment replay."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from qmed.active_space import ActiveSpaceSummary, prepare_benchmark_problem
from qmed.benchmark import MultiSeedBenchmarkResult, run_multi_seed_benchmark
from qmed.classical import run_benchmark_reference
from qmed.config import (
    DEFAULT_BASIS,
    DEFAULT_BENCHMARK_SEEDS,
    DEFAULT_DB_PATH,
    DEFAULT_SHOTS,
)
from qmed.molecules import get_benchmark, validate_and_describe
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


def get_connection(db_path: Union[Path, str] = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row factory set to sqlite3.Row."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Union[Path, str] = DEFAULT_DB_PATH) -> Path:
    """Create the SQLite database file and `experiments` table if they do not exist."""
    resolved_path = Path(db_path)
    if str(resolved_path) != ":memory:":
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(resolved_path) as conn:
        conn.executescript(EXPERIMENTS_SCHEMA_SQL)
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
    benchmark_id: str,
    descriptors_dict: Optional[Dict[str, Any]],
    qee: QEEResult,
    bm_res: MultiSeedBenchmarkResult,
    frozen_core: bool,
    active_space: Optional[Tuple[int, int]],
    maxiter: Optional[int],
    reps: int,
    db_path: Union[Path, str],
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
    db_path: Union[Path, str] = DEFAULT_DB_PATH,
) -> PipelineExecutionResult:
    """Execute the full Q-Med workflow: Input -> QEE -> Active Space -> VQE Ideal+Noisy -> Benchmark -> SQLite."""
    resolved_smiles, benchmark_id = resolve_input_smiles(input_identifier)
    val = validate_and_describe(resolved_smiles)
    qee = evaluate_eligibility(
        input_identifier,
        reference_available=reference_available,
        classical_sufficient=classical_sufficient,
        recommended_active_space=active_space_override,
    )

    if not qee.is_valid or val.descriptors is None:
        raise ValueError(qee.error or val.error or f"Invalid molecule input: '{input_identifier}'")

    descriptors_dict = val.descriptors.to_dict()

    # Non-quantum decisions (CLASSICAL_ONLY or DEFERRED): persist QEE record to SQLite and return
    if qee.decision in {"CLASSICAL_ONLY", "DEFERRED"} or benchmark_id is None:
        verdict = (
            "NO_DEMONSTRATED_ADVANTAGE"
            if qee.decision == "CLASSICAL_ONLY"
            else "INSUFFICIENT_EVIDENCE"
        )
        cfg = {
            "run_type": "qee_only",
            "input_identifier": input_identifier,
            "smiles": resolved_smiles,
            "benchmark_id": benchmark_id,
            "classical_sufficient": classical_sufficient,
            "reference_available": reference_available,
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
        act_e = qee.features.get("active_electrons", 0)
        act_o = qee.features.get("active_spatial_orbitals", 0)
        exp_id = save_experiment_record(
            {
                "input_identifier": input_identifier,
                "config_json": cfg,
                "descriptors_json": descriptors_dict,
                "qee_decision": qee.decision or "DEFERRED",
                "qee_explanation_json": qee_exp,
                "basis": DEFAULT_BASIS,
                "active_space": f"({act_e}e,{act_o}o)",
                "mapping": qee.features.get("mapping"),
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
                "qubits": qee.features.get("estimated_qubits"),
                "depth": qee.features.get("estimated_circuit_depth"),
                "gate_count": None,
                "reliability": None,
                "verdict": verdict,
            },
            db_path=db_path,
        )
        return PipelineExecutionResult(
            input_identifier=input_identifier,
            smiles=resolved_smiles,
            benchmark_id=benchmark_id,
            descriptors=descriptors_dict,
            qee=qee,
            active_space_summary=None,
            ideal_benchmark=None,
            noisy_benchmark=None,
            stored_experiment_ids=[exp_id],
            primary_experiment_id=exp_id,
        )

    # Eligible benchmark molecule (SIMULATED_QUANTUM or HYBRID)
    effective_active_space = (
        active_space_override
        if active_space_override is not None
        else (qee.recommended_active_space if qee.decision == "HYBRID" else None)
    )

    prepared = prepare_benchmark_problem(
        benchmark=benchmark_id,
        frozen_core=True,
        active_space=effective_active_space,
    )

    ideal_bm = run_multi_seed_benchmark(
        benchmark=benchmark_id,
        seeds=seeds,
        noise_mode="ideal",
        shots=None,
        frozen_core=True,
        active_space=effective_active_space,
        ansatz_type=ansatz_type,
        optimizer_name=ideal_optimizer,
        maxiter=ideal_maxiter,
    )

    all_ids: List[int] = []
    ideal_ids = _save_benchmark_runs_to_sqlite(
        input_identifier=input_identifier,
        smiles=resolved_smiles,
        benchmark_id=benchmark_id,
        descriptors_dict=descriptors_dict,
        qee=qee,
        bm_res=ideal_bm,
        frozen_core=True,
        active_space=effective_active_space,
        maxiter=ideal_maxiter,
        reps=1,
        db_path=db_path,
    )
    all_ids.extend(ideal_ids)

    noisy_bm: Optional[MultiSeedBenchmarkResult] = None
    if run_noisy:
        noisy_bm = run_multi_seed_benchmark(
            benchmark=benchmark_id,
            seeds=seeds,
            noise_mode=noisy_mode,
            shots=shots,
            frozen_core=True,
            active_space=effective_active_space,
            ansatz_type=ansatz_type,
            optimizer_name=noisy_optimizer,
            maxiter=noisy_maxiter,
        )
        noisy_ids = _save_benchmark_runs_to_sqlite(
            input_identifier=input_identifier,
            smiles=resolved_smiles,
            benchmark_id=benchmark_id,
            descriptors_dict=descriptors_dict,
            qee=qee,
            bm_res=noisy_bm,
            frozen_core=True,
            active_space=effective_active_space,
            maxiter=noisy_maxiter,
            reps=1,
            db_path=db_path,
        )
        all_ids.extend(noisy_ids)

    return PipelineExecutionResult(
        input_identifier=input_identifier,
        smiles=resolved_smiles,
        benchmark_id=benchmark_id,
        descriptors=descriptors_dict,
        qee=qee,
        active_space_summary=prepared.summary,
        ideal_benchmark=ideal_bm,
        noisy_benchmark=noisy_bm,
        stored_experiment_ids=all_ids,
        primary_experiment_id=all_ids[0] if all_ids else None,
    )


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
        (int(act_raw[0]), int(act_raw[1])) if isinstance(act_raw, list) and len(act_raw) == 2 else None
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

    benchmark_id = str(cfg.get("benchmark_id") or input_id)
    frozen_core = bool(cfg.get("frozen_core", True))
    mapping = cfg.get("mapping")
    ansatz = str(cfg.get("ansatz") or stored.get("ansatz") or "UCCSD")
    optimizer = str(cfg.get("optimizer") or stored.get("optimizer") or "SLSQP")
    maxiter = cfg.get("maxiter")
    shots = cfg.get("shots", stored.get("shots"))
    noise_mode = str(cfg.get("noise_mode") or stored.get("noise_model") or "ideal")
    reps = int(cfg.get("reps", 1))

    if run_type == "multi_seed_benchmark":
        seeds = [int(s) for s in cfg.get("seeds", [stored.get("seed", 42)])]
        bm_replay = run_multi_seed_benchmark(
            benchmark=benchmark_id,
            seeds=seeds,
            noise_mode=noise_mode,
            shots=shots,
            frozen_core=frozen_core,
            active_space=active_space,
            mapping=mapping,
            ansatz_type=ansatz,
            optimizer_name=optimizer,
            maxiter=maxiter,
            reps=reps,
        )
        replayed_energy = bm_replay.mean_vqe_energy
        replayed_iters = int(round(bm_replay.mean_iterations))
        replayed_rel = bm_replay.reliability
        replayed_verdict = bm_replay.verdict
    else:
        seed = int(cfg.get("seed", stored.get("seed", 42)))
        eff_maxiter = int(maxiter) if maxiter is not None else (200 if noise_mode == "ideal" else 80)
        bm_obj = get_benchmark(benchmark_id)
        matched_ref = run_benchmark_reference(bm_obj, frozen_core=frozen_core, active_space=active_space)
        full_fc_ref = run_benchmark_reference(bm_obj, frozen_core=True, active_space=None)
        prepared = prepare_benchmark_problem(
            benchmark=bm_obj,
            frozen_core=frozen_core,
            active_space=active_space,
            mapping=mapping,
        )
        sr_replay: VQERunResult = run_vqe_from_prepared(
            prepared=prepared,
            ansatz_type=ansatz,
            optimizer_name=optimizer,
            seed=seed,
            maxiter=eff_maxiter,
            shots=shots,
            noise_model=noise_mode,
            noise_model_name=noise_mode,
            reps=reps,
            matched_reference_energy=matched_ref.fci_energy,
            full_frozen_core_energy=full_fc_ref.fci_energy,
        )
        replayed_energy = sr_replay.vqe_energy
        replayed_iters = sr_replay.iterations
        replayed_rel = sr_replay.reliability
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
