"""Variational Quantum Eigensolver (VQE) runner using Qiskit Nature, Algorithms, and Aer."""

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
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
from qmed.classical import AtomInput, run_classical_reference
from qmed.config import DEFAULT_BASIS
from qmed.molecules import BenchmarkMolecule


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
    noise_model: Optional[Any] = None,
    noise_model_name: str = "ideal",
    reps: int = 1,
    initial_point: Optional[Sequence[float]] = None,
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

    # Configure Qiskit Aer Estimator (exact statevector when shots is None and no noise model)
    is_ideal = (shots is None) and (noise_model is None)
    backend_options: Dict[str, Any] = {}
    if noise_model is not None:
        backend_options["noise_model"] = noise_model

    run_options: Dict[str, Any] = {"seed": int(seed)}
    if shots is not None:
        run_options["shots"] = int(shots)
    else:
        run_options["shots"] = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        estimator = AerEstimator(
            backend_options=backend_options if backend_options else None,
            run_options=run_options,
            approximation=is_ideal,
            skip_transpilation=True,
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
        converged = bool(np.isfinite(total_energy) and (iterations < maxiter or tail_delta < 1e-3))
    else:
        converged = bool(np.isfinite(total_energy))

    runtime_sec = time.perf_counter() - t_start
    backend_label = "aer_statevector" if is_ideal else "aer_qasm_simulator"

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
        shots=shots,
        seed=int(seed),
        noise_model=noise_model_name if noise_model is not None else ("shot_noise" if shots else "ideal"),
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
    noise_model: Optional[Any] = None,
    noise_model_name: str = "ideal",
    molecule_id: str = "custom",
    initial_point: Optional[Sequence[float]] = None,
) -> VQERunResult:
    """Prepare the electronic structure problem and execute a seeded VQE simulation."""
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
    return run_vqe_from_prepared(
        prepared=prepared,
        ansatz_type=ansatz_type,
        optimizer_name=optimizer_name,
        seed=seed,
        maxiter=maxiter,
        shots=shots,
        noise_model=noise_model,
        noise_model_name=noise_model_name,
        initial_point=initial_point,
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
    noise_model: Optional[Any] = None,
    noise_model_name: str = "ideal",
    initial_point: Optional[Sequence[float]] = None,
) -> VQERunResult:
    """Run seeded VQE on a named benchmark molecule from data/benchmarks.json."""
    prepared = prepare_benchmark_problem(
        benchmark=benchmark,
        frozen_core=frozen_core,
        active_space=active_space,
        mapping=mapping,
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
        initial_point=initial_point,
    )


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
