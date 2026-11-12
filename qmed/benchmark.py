"""Multi-seed VQE benchmark aggregation, reliability labels, and advantage verdicts."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from qmed.active_space import prepare_benchmark_problem
from qmed.classical import run_benchmark_reference
from qmed.config import (
    DEFAULT_BENCHMARK_SEEDS,
    DEFAULT_SHOTS,
    HIGH_RELIABILITY_ERROR_HA,
    HIGH_RELIABILITY_UNCERTAINTY_HA,
    LOW_RELIABILITY_ERROR_HA,
    MAX_ACTIVE_SPACE_ERROR_FOR_HIGH_MHA,
    MEDIUM_RELIABILITY_ERROR_HA,
    MIN_SEEDS_FOR_VERDICT,
    VARIATIONAL_BOUND_TOLERANCE_HA,
)
from qmed.molecules import BenchmarkMolecule, get_benchmark
from qmed.vqe_runner import VQERunResult, run_vqe_from_prepared


@dataclass(frozen=True)
class MultiSeedBenchmarkResult:
    """Aggregated multi-seed benchmark comparison between VQE and PySCF baselines."""

    molecule_id: str
    basis: str
    active_electrons: int
    active_orbitals: int
    mapping: str
    ansatz: str
    optimizer: str
    noise_mode: str
    shots: Optional[int]
    seeds: List[int]
    num_qubits: int
    circuit_depth: int
    gate_count: int
    num_parameters: int
    hf_energy: float
    matched_fci_energy: float
    full_frozen_core_fci_energy: float
    mean_vqe_energy: float
    min_vqe_energy: float
    max_vqe_energy: float
    mean_error_ha: float
    abs_mean_error_ha: float
    abs_mean_error_mha: float
    uncertainty_ha: float
    uncertainty_mha: float
    active_space_error_mha: float
    mean_iterations: float
    all_converged: bool
    converged_fraction: float
    mean_vqe_runtime_sec: float
    total_vqe_runtime_sec: float
    classical_runtime_sec: float
    status: str
    reliability: str
    verdict: str
    warnings: List[str] = field(default_factory=list)
    seed_runs: List[VQERunResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the multi-seed benchmark result."""
        return asdict(self)


def classify_reliability(
    abs_error_ha: float,
    uncertainty_ha: float,
    converged: bool,
    active_space_error_mha: float = 0.0,
    vqe_energy: Optional[float] = None,
    matched_fci_energy: Optional[float] = None,
    is_noisy: bool = False,
) -> Tuple[str, str, List[str]]:
    """Assign status and reliability label per Q-Med specification.

    Rules:
    - Variational Guard (ideal runs): if vqe_energy < matched_fci_energy - 1e-8 Ha -> INVALID.
    - HIGH: converged, |error| <= 1.6 mHa, uncertainty <= 1.6 mHa (capped at MEDIUM if
      |active_space_error_mha| > 1.6 mHa).
    - MEDIUM: converged, |error| <= 10 mHa (includes runs with |error| <= 1.6 mHa when
      uncertainty > 1.6 mHa or when capped by active-space truncation error).
    - LOW: converged, |error| <= 50 mHa.
    - UNRELIABLE: not converged, or |error| > 50 mHa.
    """
    notes: List[str] = []

    if vqe_energy is not None and matched_fci_energy is not None:
        if vqe_energy < matched_fci_energy - VARIATIONAL_BOUND_TOLERANCE_HA:
            diff_ha = matched_fci_energy - vqe_energy
            if not is_noisy:
                notes.append(
                    f"VARIATIONAL_BOUND_VIOLATION: Ideal VQE energy ({vqe_energy:.8f} Ha) is below "
                    f"matched FCI ({matched_fci_energy:.8f} Ha) by {diff_ha:.2e} Ha."
                )
                return "INVALID", "INVALID", notes
            else:
                notes.append(
                    f"PHYSICALLY_SUSPICIOUS_NOISY_ENERGY: Noisy VQE energy ({vqe_energy:.8f} Ha) "
                    f"is below matched N-electron FCI ({matched_fci_energy:.8f} Ha) by {diff_ha*1000:.4f} mHa "
                    "(caused by finite-shot statistical fluctuation or depolarizing leakage into "
                    "non-particle-conserving Fock sectors)."
                )

    if not converged or abs_error_ha > LOW_RELIABILITY_ERROR_HA:
        label = "UNRELIABLE"
    elif (
        abs_error_ha <= HIGH_RELIABILITY_ERROR_HA
        and uncertainty_ha <= HIGH_RELIABILITY_UNCERTAINTY_HA
    ):
        label = "HIGH"
    elif abs_error_ha <= MEDIUM_RELIABILITY_ERROR_HA:
        label = "MEDIUM"
    else:
        label = "LOW"

    if abs(active_space_error_mha) > MAX_ACTIVE_SPACE_ERROR_FOR_HIGH_MHA:
        notes.append(
            f"ACTIVE_SPACE_TRUNCATION: Reduced active space introduces "
            f"{active_space_error_mha:+.4f} mHa error vs. full frozen-core CASCI."
        )
        if label == "HIGH":
            label = "MEDIUM"
            notes.append(
                "RELIABILITY_CAPPED: Capped from HIGH to MEDIUM because active_space_error_mHa "
                f"exceeds {MAX_ACTIVE_SPACE_ERROR_FOR_HIGH_MHA:.2f} mHa."
            )

    return "VALID", label, notes


def classify_advantage_verdict(
    converged: bool,
    num_seeds: int,
    reliability: str,
    abs_error_ha: float,
    vqe_runtime_sec: float,
    classical_runtime_sec: float,
    classical_exact_available: bool = True,
    classical_reference_error_ha: float = 0.0,
) -> str:
    """Assign an honest quantum-advantage verdict per Q-Med specification.

    Rules:
    - INSUFFICIENT_EVIDENCE: not converged, fewer than 3 seeds, or reliability is INVALID.
    - NO_DEMONSTRATED_ADVANTAGE: classical exact method is available and matches/exceeds
      accuracy at lower or comparable cost (standard outcome for small benchmarks).
    - DEMONSTRATED_ADVANTAGE: converged, >= 3 seeds, HIGH reliability, and VQE achieves
      strictly better accuracy than available classical approximation in shorter runtime.
    - POTENTIAL_ADVANTAGE: converged, >= 3 seeds, HIGH/MEDIUM reliability, and classical
      exact reference is unavailable/intractable while VQE error is within chemical accuracy.
    """
    if not converged or num_seeds < MIN_SEEDS_FOR_VERDICT or reliability == "INVALID":
        return "INSUFFICIENT_EVIDENCE"

    if classical_exact_available and classical_reference_error_ha <= abs_error_ha:
        if classical_runtime_sec <= vqe_runtime_sec or reliability in {"LOW", "UNRELIABLE"}:
            return "NO_DEMONSTRATED_ADVANTAGE"

    if (
        reliability == "HIGH"
        and abs_error_ha <= HIGH_RELIABILITY_ERROR_HA
        and not classical_exact_available
        and vqe_runtime_sec < classical_runtime_sec
    ):
        return "DEMONSTRATED_ADVANTAGE"

    if (
        reliability in {"HIGH", "MEDIUM"}
        and not classical_exact_available
        and abs_error_ha <= MEDIUM_RELIABILITY_ERROR_HA
    ):
        return "POTENTIAL_ADVANTAGE"

    return "NO_DEMONSTRATED_ADVANTAGE"


def run_multi_seed_benchmark(
    benchmark: Union[str, BenchmarkMolecule],
    seeds: Sequence[int] = DEFAULT_BENCHMARK_SEEDS,
    noise_mode: str = "ideal",
    shots: Optional[int] = None,
    frozen_core: bool = True,
    active_space: Optional[Tuple[int, int]] = None,
    mapping: Optional[str] = None,
    ansatz_type: str = "UCCSD",
    optimizer_name: Optional[str] = None,
    maxiter: Optional[int] = None,
    reps: int = 1,
) -> MultiSeedBenchmarkResult:
    """Run ideal or noisy VQE across multiple seeds and compare against PySCF HF and FCI."""
    bm = get_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
    is_ideal = noise_mode.strip().lower() in {"ideal", "none", "exact", "statevector"} and (shots is None)

    # Default optimizer: SLSQP for exact statevector, COBYLA for shot-sampled/noisy landscapes
    if optimizer_name is None:
        opt_choice = "SLSQP" if is_ideal else "COBYLA"
    else:
        opt_choice = optimizer_name

    if maxiter is None:
        eff_maxiter = 200 if is_ideal else 80
    else:
        eff_maxiter = int(maxiter)

    eff_shots = None if is_ideal else (int(shots) if shots is not None else DEFAULT_SHOTS)

    # 1. Classical PySCF references (matched active space and full frozen-core valence space)
    matched_ref = run_benchmark_reference(bm, frozen_core=frozen_core, active_space=active_space)
    full_fc_ref = run_benchmark_reference(bm, frozen_core=True, active_space=None)

    # 2. Prepare quantum problem once and run across all seeds
    prepared = prepare_benchmark_problem(
        benchmark=bm,
        frozen_core=frozen_core,
        active_space=active_space,
        mapping=mapping,
    )

    seed_runs: List[VQERunResult] = []
    for s in seeds:
        run_res = run_vqe_from_prepared(
            prepared=prepared,
            ansatz_type=ansatz_type,
            optimizer_name=opt_choice,
            seed=int(s),
            maxiter=eff_maxiter,
            shots=eff_shots,
            noise_model=noise_mode,
            noise_model_name=noise_mode,
            reps=reps,
            matched_reference_energy=matched_ref.fci_energy,
            full_frozen_core_energy=full_fc_ref.fci_energy,
        )
        seed_runs.append(run_res)

    energies = np.array([r.vqe_energy for r in seed_runs], dtype=float)
    iters = np.array([r.iterations for r in seed_runs], dtype=float)
    runtimes = np.array([r.runtime_sec for r in seed_runs], dtype=float)
    conv_flags = [bool(r.converged) for r in seed_runs]

    mean_vqe = float(np.mean(energies))
    min_vqe = float(np.min(energies))
    max_vqe = float(np.max(energies))
    uncertainty_ha = float(np.std(energies, ddof=1)) if len(energies) > 1 else 0.0

    mean_error_ha = mean_vqe - matched_ref.fci_energy
    abs_mean_error_ha = abs(mean_error_ha)
    active_space_error_mha = (matched_ref.fci_energy - full_fc_ref.fci_energy) * 1000.0

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
        matched_fci_energy=matched_ref.fci_energy,
        is_noisy=not is_ideal,
    )

    # Check if any noisy seed exceeded the Hartree-Fock energy (meaning noise destroyed correlation)
    if mean_vqe > matched_ref.hf_energy:
        agg_warnings.append(
            f"NOISE_ABOVE_HF_WARNING: Mean VQE energy ({mean_vqe:.6f} Ha) is higher (worse) than "
            f"classical mean-field Hartree-Fock ({matched_ref.hf_energy:.6f} Ha) by "
            f"{(mean_vqe - matched_ref.hf_energy)*1000:.2f} mHa due to gate/readout decoherence."
        )

    verdict = classify_advantage_verdict(
        converged=all_converged,
        num_seeds=len(seeds),
        reliability=reliability,
        abs_error_ha=abs_mean_error_ha,
        vqe_runtime_sec=mean_vqe_runtime,
        classical_runtime_sec=matched_ref.total_runtime_sec,
        classical_exact_available=True,
        classical_reference_error_ha=0.0,
    )

    first = seed_runs[0]
    return MultiSeedBenchmarkResult(
        molecule_id=bm.id,
        basis=bm.basis,
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
        hf_energy=matched_ref.hf_energy,
        matched_fci_energy=matched_ref.fci_energy,
        full_frozen_core_fci_energy=full_fc_ref.fci_energy,
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
        classical_runtime_sec=matched_ref.total_runtime_sec,
        status=status,
        reliability=reliability,
        verdict=verdict,
        warnings=agg_warnings,
        seed_runs=seed_runs,
    )
