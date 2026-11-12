"""Variational Quantum Eigensolver (VQE) runner using Qiskit Nature, Algorithms, and Aer.

Note: For 2-active-electron systems (H2 in (2e, 2o) and frozen-core LiH in (2e, 5o)),
UCCSD spans the exact active-space FCI Hilbert space and therefore validates the
quantum pipeline rather than testing approximate ansatz expressibility.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SLSQP, Optimizer
from qiskit_algorithms.utils import algorithm_globals
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

from qmed.active_space import (
    PreparedQuantumProblem,
    prepare_active_space_problem,
    prepare_benchmark_problem,
)
from qmed.classical import AtomInput, run_benchmark_reference, run_classical_reference
from qmed.config import (
    DEFAULT_BASIS,
    DEFAULT_DEPOLARIZING_1Q_ERROR,
    DEFAULT_DEPOLARIZING_2Q_ERROR,
    DEFAULT_READOUT_ERROR,
    DEFAULT_SHOTS,
    HIGH_RELIABILITY_ERROR_HA,
    HIGH_RELIABILITY_UNCERTAINTY_HA,
    LOW_RELIABILITY_ERROR_HA,
    MAX_ACTIVE_SPACE_ERROR_FOR_HIGH_MHA,
    MEDIUM_RELIABILITY_ERROR_HA,
    PROJECT_ROOT,
    VARIATIONAL_BOUND_TOLERANCE_HA,
)
from qmed.molecules import BenchmarkMolecule, get_benchmark


@dataclass(frozen=True)
class VQERunResult:
    """Result container for a single seeded VQE execution."""

    molecule_id: str
    basis: str
    frozen_core: bool
    active_electrons: int
    active_orbitals: int
    mapping: str
    ansatz: str
    optimizer: str
    backend: str
    shots: Optional[int]
    seed: int
    noise_model: str
    vqe_energy: float
    electronic_energy: float
    energy_offset: float
    iterations: int
    converged: bool
    num_qubits: int
    circuit_depth: int
    gate_count: int
    num_parameters: int
    runtime_sec: float
    optimal_parameters: List[float]
    matched_reference_energy: Optional[float] = None
    full_frozen_core_energy: Optional[float] = None
    vqe_error_mha: Optional[float] = None
    active_space_error_mha: float = 0.0
    variational_valid: bool = True
    status: str = "VALID"
    reliability: str = "HIGH"
    run_label: str = "standard"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of VQE results."""
        return asdict(self)


@dataclass(frozen=True)
class H2ScanPoint:
    """Single geometry point from an H2 potential energy surface scan."""

    bond_length_angstrom: float
    hf_energy: float
    fci_energy: float
    vqe_energy: float
    abs_error_ha: float
    abs_error_mha: float
    iterations: int
    converged: bool
    runtime_sec: float
    warm_started: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary for this scan point."""
        return asdict(self)


def build_depolarizing_readout_noise_model(
    p_1q: float = DEFAULT_DEPOLARIZING_1Q_ERROR,
    p_2q: float = DEFAULT_DEPOLARIZING_2Q_ERROR,
    p_ro: float = DEFAULT_READOUT_ERROR,
) -> NoiseModel:
    """Construct a Qiskit Aer noise model with 1Q/2Q depolarizing errors and symmetric readout bit-flip."""
    if not (0.0 <= p_1q <= 1.0 and 0.0 <= p_2q <= 1.0 and 0.0 <= p_ro <= 0.5):
        raise ValueError("Noise probabilities must satisfy 0 <= p_1q, p_2q <= 1 and 0 <= p_ro <= 0.5.")

    noise_model = NoiseModel()
    if p_1q > 0.0:
        err_1q = depolarizing_error(p_1q, 1)
        noise_model.add_all_qubit_quantum_error(
            err_1q, ["rx", "ry", "rz", "sx", "x", "h", "u1", "u2", "u3"]
        )
    if p_2q > 0.0:
        err_2q = depolarizing_error(p_2q, 2)
        noise_model.add_all_qubit_quantum_error(err_2q, ["cx", "cz", "ecr"])
    if p_ro > 0.0:
        ro_err = ReadoutError([[1.0 - p_ro, p_ro], [p_ro, 1.0 - p_ro]])
        noise_model.add_all_qubit_readout_error(ro_err)

    return noise_model


def build_fake_backend_noise_model(
    num_qubits: int = 8,
    seed: int = 42,
) -> Tuple[NoiseModel, str]:
    """Construct a calibration-based NoiseModel from Qiskit's GenericBackendV2 fake provider."""
    n_q = max(2, int(num_qubits))
    fake_backend = GenericBackendV2(num_qubits=n_q, seed=int(seed))
    noise_model = NoiseModel.from_backend(fake_backend)
    return noise_model, f"fake_generic_v2_{n_q}q"


def resolve_noise_configuration(
    noise_mode: Union[str, NoiseModel, None],
    num_qubits: int,
    seed: int,
    shots: Optional[int],
) -> Tuple[Optional[NoiseModel], str, Optional[int]]:
    """Resolve a noise_mode identifier or NoiseModel object into (NoiseModel, label, shots)."""
    if noise_mode is None:
        if shots is not None:
            return None, "shot_noise", int(shots)
        return None, "ideal", None

    if isinstance(noise_mode, NoiseModel):
        eff_shots = int(shots) if shots is not None else DEFAULT_SHOTS
        return noise_mode, "custom_noise_model", eff_shots

    mode_str = str(noise_mode).strip().lower()
    if mode_str in {"ideal", "none", "exact", "statevector"}:
        if shots is not None:
            return None, "shot_noise", int(shots)
        return None, "ideal", None

    eff_shots = int(shots) if shots is not None else DEFAULT_SHOTS
    if mode_str in {"depolarizing", "depolarizing_readout", "noisy"}:
        nm = build_depolarizing_readout_noise_model()
        return nm, "depolarizing_readout", eff_shots

    if mode_str in {"fake_backend", "fake", "generic_backend_v2"}:
        nm, label = build_fake_backend_noise_model(num_qubits=num_qubits, seed=seed)
        return nm, label, eff_shots

    raise ValueError(
        f"Unsupported noise_mode '{noise_mode}'. Use 'ideal', 'depolarizing_readout', or 'fake_backend'."
    )


def evaluate_reliability_and_bounds(
    vqe_energy: float,
    matched_fci_energy: float,
    converged: bool,
    uncertainty_ha: float = 0.0,
    full_frozen_core_fci_energy: Optional[float] = None,
    is_noisy_or_shot_based: bool = False,
) -> Tuple[str, str, bool, float, List[str]]:
    """Check variational bound and assign reliability with active-space error capping."""
    warnings_list: List[str] = []

    if full_frozen_core_fci_energy is not None:
        active_space_error_mha = (matched_fci_energy - full_frozen_core_fci_energy) * 1000.0
    else:
        active_space_error_mha = 0.0

    # 1. Variational-bound guard (E_VQE >= E_FCI_matched - 1e-8 Ha)
    if vqe_energy < matched_fci_energy - VARIATIONAL_BOUND_TOLERANCE_HA:
        violation_ha = matched_fci_energy - vqe_energy
        if not is_noisy_or_shot_based:
            warnings_list.append(
                f"VARIATIONAL_BOUND_VIOLATION: VQE energy ({vqe_energy:.8f} Ha) is below matched "
                f"active-space FCI/CASCI reference ({matched_fci_energy:.8f} Ha) by {violation_ha:.2e} Ha "
                f"(exceeds tolerance {VARIATIONAL_BOUND_TOLERANCE_HA:.1e} Ha). Run marked INVALID."
            )
            return "INVALID", "INVALID", False, active_space_error_mha, warnings_list
        else:
            warnings_list.append(
                f"NOISY_BELOW_FCI_WARNING: Noisy/shot-sampled VQE energy ({vqe_energy:.8f} Ha) fell below "
                f"matched N-electron FCI reference ({matched_fci_energy:.8f} Ha) by {violation_ha*1000:.4f} mHa "
                "due to finite-shot variance or depolarizing leakage into non-particle-conserving Fock sectors."
            )

    abs_err_ha = abs(vqe_energy - matched_fci_energy)

    # 2. Raw reliability in matched active space
    if not converged or abs_err_ha > LOW_RELIABILITY_ERROR_HA:
        raw_reliability = "UNRELIABLE"
    elif (
        abs_err_ha <= HIGH_RELIABILITY_ERROR_HA
        and uncertainty_ha <= HIGH_RELIABILITY_UNCERTAINTY_HA
    ):
        raw_reliability = "HIGH"
    elif abs_err_ha <= MEDIUM_RELIABILITY_ERROR_HA:
        raw_reliability = "MEDIUM"
    else:
        raw_reliability = "LOW"

    # 3. Active-space truncation capping rule
    final_reliability = raw_reliability
    if abs(active_space_error_mha) > MAX_ACTIVE_SPACE_ERROR_FOR_HIGH_MHA:
        warnings_list.append(
            f"ACTIVE_SPACE_TRUNCATION: Reduced active space introduces "
            f"{active_space_error_mha:+.4f} mHa error vs. full frozen-core CASCI "
            f"(exceeds {MAX_ACTIVE_SPACE_ERROR_FOR_HIGH_MHA:.2f} mHa threshold)."
        )
        if final_reliability == "HIGH":
            final_reliability = "MEDIUM"
            warnings_list.append(
                "RELIABILITY_CAPPED: Subspace VQE error met HIGH threshold, but overall "
                "reliability is capped at MEDIUM due to active-space truncation error."
            )

    return "VALID", final_reliability, True, active_space_error_mha, warnings_list


def build_ansatz(
    prepared: PreparedQuantumProblem,
    ansatz_type: str = "UCCSD",
    reps: int = 1,
) -> Tuple[QuantumCircuit, str]:
    """Construct either the chemistry-inspired UCCSD ansatz or EfficientSU2 fallback."""
    problem = prepared.problem
    mapper = prepared.mapper
    num_qubits = prepared.summary.num_qubits

    hf_initial_state = HartreeFock(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=mapper,
    )

    normalized = ansatz_type.strip().upper().replace("-", "_")
    if normalized == "UCCSD":
        raw_ansatz = UCCSD(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
            initial_state=hf_initial_state,
            reps=reps,
        )
        decomposed = raw_ansatz.decompose().decompose().decompose()
        return decomposed, "UCCSD"

    if normalized in {"EFFICIENT_SU2", "EFFICIENTSU2", "HEA", "HARDWARE_EFFICIENT"}:
        raw_ansatz = EfficientSU2(
            num_qubits=num_qubits,
            su2_gates=["ry", "rz"],
            entanglement="linear",
            reps=max(1, reps),
            initial_state=hf_initial_state,
        )
        decomposed = raw_ansatz.decompose().decompose()
        return decomposed, "EfficientSU2"

    raise ValueError(
        f"Unsupported ansatz '{ansatz_type}'. Choose 'UCCSD' or 'EfficientSU2'."
    )


def build_optimizer(optimizer_name: str = "SLSQP", maxiter: int = 200) -> Tuple[Optimizer, str]:
    """Instantiate SLSQP or COBYLA optimizer with explicit maximum iterations."""
    normalized = optimizer_name.strip().upper()
    if normalized == "SLSQP":
        return SLSQP(maxiter=maxiter), "SLSQP"
    if normalized == "COBYLA":
        return COBYLA(maxiter=maxiter), "COBYLA"
    raise ValueError(
        f"Unsupported optimizer '{optimizer_name}'. Choose 'SLSQP' or 'COBYLA'."
    )


def compute_circuit_metrics(circuit: QuantumCircuit) -> Tuple[int, int]:
    """Return (depth, gate_count) for a decomposed quantum circuit."""
    return int(circuit.depth()), int(circuit.size())


def run_vqe_from_prepared(
    prepared: PreparedQuantumProblem,
    ansatz_type: str = "UCCSD",
    optimizer_name: str = "SLSQP",
    seed: int = 42,
    maxiter: int = 200,
    shots: Optional[int] = None,
    noise_model: Union[str, NoiseModel, None] = None,
    noise_model_name: str = "ideal",
    reps: int = 1,
    initial_point: Optional[Sequence[float]] = None,
    matched_reference_energy: Optional[float] = None,
    full_frozen_core_energy: Optional[float] = None,
    run_label: str = "standard",
) -> VQERunResult:
    """Execute a seeded VQE simulation on a prepared active-space quantum problem.

    When `shots is None` and `noise_model is None`, Qiskit Aer Estimator runs in
    exact statevector mode (`approximation=True, shots=None`), computing exact
    expectation values without shot-sampling noise.
    """
    t_start = time.perf_counter()

    # Enforce reproducibility across numpy and qiskit_algorithms
    algorithm_globals.random_seed = int(seed)
    rng = np.random.default_rng(int(seed))

    ansatz, canonical_ansatz = build_ansatz(prepared, ansatz_type=ansatz_type, reps=reps)
    optimizer, canonical_opt = build_optimizer(optimizer_name=optimizer_name, maxiter=maxiter)
    depth, gate_count = compute_circuit_metrics(ansatz)
    num_params = int(ansatz.num_parameters)

    if initial_point is not None:
        if len(initial_point) != num_params:
            raise ValueError(
                f"initial_point length ({len(initial_point)}) must match ansatz parameters ({num_params})."
            )
        init_pt = np.asarray(initial_point, dtype=float)
    else:
        # Seeded small perturbation around Hartree-Fock reference (0 parameters = HF state for UCCSD)
        init_pt = rng.uniform(-0.02, 0.02, size=num_params)

    resolved_nm, resolved_nm_label, resolved_shots = resolve_noise_configuration(
        noise_mode=noise_model if noise_model is not None else noise_model_name,
        num_qubits=prepared.summary.num_qubits,
        seed=int(seed),
        shots=shots,
    )

    is_ideal = (resolved_shots is None) and (resolved_nm is None)
    use_fast_noisy_trajectory = not is_ideal

    backend_options: Dict[str, Any] = {}
    if resolved_nm is not None:
        backend_options["noise_model"] = resolved_nm
        backend_options["method"] = "statevector"

    run_options: Dict[str, Any] = {"seed": int(seed)}
    if resolved_shots is not None and not use_fast_noisy_trajectory:
        run_options["shots"] = int(resolved_shots)
    else:
        run_options["shots"] = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        estimator = AerEstimator(
            backend_options=backend_options if backend_options else None,
            run_options=run_options,
            approximation=True,
            skip_transpilation=True,
        )
        if use_fast_noisy_trajectory:
            n_traj = 64 if prepared.summary.num_qubits >= 6 else 256
            estimator._backend.set_options(
                shots=min(int(resolved_shots or DEFAULT_SHOTS), n_traj),
                seed_simulator=int(seed),
            )

        energy_trace: List[float] = []

        def _callback(eval_count: int, params: np.ndarray, mean: float, meta: Dict[str, Any]) -> None:
            energy_trace.append(float(mean))

        vqe = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=init_pt,
            callback=_callback,
        )
        raw_result = vqe.compute_minimum_eigenvalue(prepared.qubit_op)

    interpreted = prepared.problem.interpret(raw_result)
    total_energy = float(interpreted.total_energies[0].real)
    electronic_energy = float(raw_result.eigenvalue.real)
    energy_offset = float(total_energy - electronic_energy)

    iterations = int(raw_result.cost_function_evals or len(energy_trace) or 1)
    if raw_result.optimal_point is not None:
        opt_params = [float(x) for x in raw_result.optimal_point]
    else:
        opt_params = [float(x) for x in init_pt]

    # Determine convergence: did not exhaust maxiter without stabilizing, and finite energy
    if len(energy_trace) >= 2:
        tail_delta = abs(energy_trace[-1] - energy_trace[-2])
        tail_tol = 1e-2 if not is_ideal else 1e-3
        converged = bool(np.isfinite(total_energy) and (iterations < maxiter or tail_delta < tail_tol))
    else:
        converged = bool(np.isfinite(total_energy))

    runtime_sec = time.perf_counter() - t_start
    backend_label = "aer_statevector" if is_ideal else "aer_qasm_simulator"

    status = "VALID"
    reliability = "HIGH"
    variational_valid = True
    vqe_error_mha: Optional[float] = None
    active_space_error_mha = 0.0
    warnings_list: List[str] = []

    if run_label == "ansatz_limitation_demo":
        warnings_list.append(
            "ANSATZ_LIMITATION_DEMO: Shallow hardware-efficient EfficientSU2(reps=1) run "
            "with capped iterations on full frozen-core active space."
        )

    if matched_reference_energy is not None:
        vqe_error_mha = (total_energy - matched_reference_energy) * 1000.0
        status, reliability, variational_valid, active_space_error_mha, bound_warnings = (
            evaluate_reliability_and_bounds(
                vqe_energy=total_energy,
                matched_fci_energy=matched_reference_energy,
                converged=converged,
                uncertainty_ha=0.0,
                full_frozen_core_fci_energy=full_frozen_core_energy,
                is_noisy_or_shot_based=not is_ideal,
            )
        )
        warnings_list.extend(bound_warnings)

    return VQERunResult(
        molecule_id=prepared.summary.molecule_id,
        basis=prepared.summary.basis,
        frozen_core=prepared.summary.frozen_core,
        active_electrons=prepared.summary.active_electrons,
        active_orbitals=prepared.summary.active_spatial_orbitals,
        mapping=prepared.summary.mapping,
        ansatz=canonical_ansatz,
        optimizer=canonical_opt,
        backend=backend_label,
        shots=resolved_shots,
        seed=int(seed),
        noise_model=resolved_nm_label,
        vqe_energy=total_energy,
        electronic_energy=electronic_energy,
        energy_offset=energy_offset,
        iterations=iterations,
        converged=converged,
        num_qubits=prepared.summary.num_qubits,
        circuit_depth=depth,
        gate_count=gate_count,
        num_parameters=num_params,
        runtime_sec=runtime_sec,
        optimal_parameters=opt_params,
        matched_reference_energy=matched_reference_energy,
        full_frozen_core_energy=full_frozen_core_energy,
        vqe_error_mha=vqe_error_mha,
        active_space_error_mha=active_space_error_mha,
        variational_valid=variational_valid,
        status=status,
        reliability=reliability,
        run_label=run_label,
        warnings=warnings_list,
    )


def run_vqe(
    atoms: AtomInput,
    basis: str = DEFAULT_BASIS,
    charge: int = 0,
    spin: int = 0,
    frozen_core: bool = True,
    active_space: Optional[Tuple[int, int]] = None,
    mapping: Optional[str] = None,
    ansatz_type: str = "UCCSD",
    optimizer_name: str = "SLSQP",
    seed: int = 42,
    maxiter: int = 200,
    shots: Optional[int] = None,
    noise_model: Union[str, NoiseModel, None] = None,
    noise_model_name: str = "ideal",
    molecule_id: str = "custom",
    initial_point: Optional[Sequence[float]] = None,
    reps: int = 1,
    run_label: str = "standard",
) -> VQERunResult:
    """Prepare the electronic structure problem, compute matched classical reference, and run VQE."""
    prepared = prepare_active_space_problem(
        atoms=atoms,
        basis=basis,
        charge=charge,
        spin=spin,
        frozen_core=frozen_core,
        active_space=active_space,
        mapping=mapping,
        molecule_id=molecule_id,
    )
    matched_ref = run_classical_reference(
        atoms=atoms,
        basis=basis,
        charge=charge,
        spin=spin,
        frozen_core=frozen_core,
        active_space=active_space,
        molecule_id=molecule_id,
    )
    full_fc_ref = run_classical_reference(
        atoms=atoms,
        basis=basis,
        charge=charge,
        spin=spin,
        frozen_core=True,
        active_space=None,
        molecule_id=molecule_id,
    )
    return run_vqe_from_prepared(
        prepared=prepared,
        ansatz_type=ansatz_type,
        optimizer_name=optimizer_name,
        seed=seed,
        maxiter=maxiter,
        shots=shots,
        noise_model=noise_model,
        noise_model_name=noise_model_name,
        reps=reps,
        initial_point=initial_point,
        matched_reference_energy=matched_ref.fci_energy,
        full_frozen_core_energy=full_fc_ref.fci_energy,
        run_label=run_label,
    )


def run_benchmark_vqe(
    benchmark: Union[str, BenchmarkMolecule],
    frozen_core: bool = True,
    active_space: Optional[Tuple[int, int]] = None,
    mapping: Optional[str] = None,
    ansatz_type: str = "UCCSD",
    optimizer_name: str = "SLSQP",
    seed: int = 42,
    maxiter: int = 200,
    shots: Optional[int] = None,
    noise_model: Union[str, NoiseModel, None] = None,
    noise_model_name: str = "ideal",
    initial_point: Optional[Sequence[float]] = None,
    reps: int = 1,
    run_label: str = "standard",
) -> VQERunResult:
    """Run seeded VQE on a named benchmark molecule from data/benchmarks.json."""
    bm = get_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
    prepared = prepare_benchmark_problem(
        benchmark=bm,
        frozen_core=frozen_core,
        active_space=active_space,
        mapping=mapping,
    )
    matched_ref = run_benchmark_reference(
        bm,
        frozen_core=frozen_core,
        active_space=active_space,
    )
    full_fc_ref = run_benchmark_reference(
        bm,
        frozen_core=True,
        active_space=None,
    )
    return run_vqe_from_prepared(
        prepared=prepared,
        ansatz_type=ansatz_type,
        optimizer_name=optimizer_name,
        seed=seed,
        maxiter=maxiter,
        shots=shots,
        noise_model=noise_model,
        noise_model_name=noise_model_name,
        reps=reps,
        initial_point=initial_point,
        matched_reference_energy=matched_ref.fci_energy,
        full_frozen_core_energy=full_fc_ref.fci_energy,
        run_label=run_label,
    )


def run_water_option_b_demo(
    seeds: Sequence[int] = (42, 43, 44),
    maxiter: int = 120,
    optimizer_name: str = "COBYLA",
) -> List[VQERunResult]:
    """Option B demonstration: Water (8e, 6o) with Parity mapping (10 qubits) and EfficientSU2(reps=1)."""
    prepared = prepare_benchmark_problem(
        "H2O",
        frozen_core=True,
        active_space=None,
        mapping="parity",
    )
    matched_ref = run_benchmark_reference("H2O", frozen_core=True, active_space=None)

    results: List[VQERunResult] = []
    for seed in seeds:
        res = run_vqe_from_prepared(
            prepared=prepared,
            ansatz_type="EfficientSU2",
            optimizer_name=optimizer_name,
            seed=int(seed),
            maxiter=maxiter,
            reps=1,
            matched_reference_energy=matched_ref.fci_energy,
            full_frozen_core_energy=matched_ref.fci_energy,
            run_label="ansatz_limitation_demo",
        )
        results.append(res)
    return results


def run_h2_bond_scan(
    bond_lengths: Optional[Sequence[float]] = None,
    basis: str = DEFAULT_BASIS,
    ansatz_type: str = "UCCSD",
    optimizer_name: str = "SLSQP",
    seed: int = 42,
    maxiter: int = 200,
    warm_start: bool = True,
) -> List[H2ScanPoint]:
    """Scan H2 bond lengths (in Angstroms) comparing warm-started VQE against PySCF FCI."""
    if bond_lengths is None:
        bond_lengths = [
            0.30,
            0.50,
            0.70,
            0.735,
            0.90,
            1.10,
            1.30,
            1.50,
            1.70,
            1.90,
            2.10,
            2.30,
            2.50,
        ]

    scan_results: List[H2ScanPoint] = []
    prev_params: Optional[List[float]] = None

    for r in bond_lengths:
        dist = float(r)
        atom_str = f"H 0.0 0.0 0.0; H 0.0 0.0 {dist:.6f}"
        ref = run_classical_reference(
            atoms=atom_str,
            basis=basis,
            charge=0,
            spin=0,
            frozen_core=False,
            molecule_id=f"H2_{dist:.3f}A",
        )
        vqe_res = run_vqe(
            atoms=atom_str,
            basis=basis,
            charge=0,
            spin=0,
            frozen_core=True,
            mapping="parity",
            ansatz_type=ansatz_type,
            optimizer_name=optimizer_name,
            seed=seed,
            maxiter=maxiter,
            molecule_id="H2",
            initial_point=prev_params if warm_start else None,
        )
        used_warm_start = bool(warm_start and (prev_params is not None))
        if warm_start:
            prev_params = vqe_res.optimal_parameters

        abs_err_ha = abs(vqe_res.vqe_energy - ref.fci_energy)
        scan_results.append(
            H2ScanPoint(
                bond_length_angstrom=dist,
                hf_energy=ref.hf_energy,
                fci_energy=ref.fci_energy,
                vqe_energy=vqe_res.vqe_energy,
                abs_error_ha=abs_err_ha,
                abs_error_mha=abs_err_ha * 1000.0,
                iterations=vqe_res.iterations,
                converged=vqe_res.converged,
                runtime_sec=vqe_res.runtime_sec,
                warm_started=used_warm_start,
            )
        )

    return scan_results


def save_h2_curve_plot(
    scan_points: Sequence[H2ScanPoint],
    output_path: Union[Path, str] = PROJECT_ROOT / "docs" / "h2_curve.png",
) -> Path:
    """Plot H2 VQE points over the continuous FCI curve with HF as a dashed line."""
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    r_vals = [p.bond_length_angstrom for p in scan_points]
    hf_vals = [p.hf_energy for p in scan_points]
    fci_vals = [p.fci_energy for p in scan_points]
    vqe_vals = [p.vqe_energy for p in scan_points]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ax.plot(r_vals, fci_vals, color="#1f77b4", linewidth=2.2, label="PySCF FCI (Exact)")
    ax.plot(r_vals, hf_vals, color="#ff7f0e", linestyle="--", linewidth=1.8, label="PySCF RHF (Mean-Field)")
    ax.scatter(r_vals, vqe_vals, color="#d62728", s=45, zorder=5, label="VQE (UCCSD, 2 Qubits)")
    ax.set_title("H2 Potential Energy Surface (STO-3G): VQE vs. FCI and HF")
    ax.set_xlabel("Bond Length R (Angstrom)")
    ax.set_ylabel("Total Energy (Hartree)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(resolved)
    plt.close(fig)
    return resolved
