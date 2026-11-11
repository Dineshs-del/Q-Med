"""Week 3 pytest suite: H2 VQE, H2 warm-started bond scan, and LiH frozen-core VQE."""

import pytest

from qmed.active_space import prepare_benchmark_problem
from qmed.classical import run_benchmark_reference
from qmed.vqe_runner import run_benchmark_vqe, run_h2_bond_scan


def test_h2_parity_two_qubit_reduction() -> None:
    """Verify H2 (STO-3G) maps to 2 qubits under ParityMapper with two-qubit reduction."""
    prepared = prepare_benchmark_problem("H2", frozen_core=True)
    summary = prepared.summary

    assert summary.molecule_id == "H2"
    assert summary.active_electrons == 2
    assert summary.active_spatial_orbitals == 2
    assert summary.mapping == "parity_2q_reduced"
    assert summary.two_qubit_reduction is True
    assert summary.num_qubits == 2


def test_h2_vqe_matches_fci_within_1_mha() -> None:
    """Verify H2 VQE (UCCSD + SLSQP, seed=42) runs in exact no-shot mode and matches FCI within 1 mHa."""
    ref = run_benchmark_reference("H2", frozen_core=False)
    vqe_res = run_benchmark_vqe(
        "H2",
        frozen_core=True,
        ansatz_type="UCCSD",
        optimizer_name="SLSQP",
        seed=42,
    )

    assert vqe_res.converged is True
    assert vqe_res.backend == "aer_statevector"
    assert vqe_res.shots is None
    assert vqe_res.noise_model == "ideal"
    assert vqe_res.num_qubits == 2
    assert vqe_res.circuit_depth > 0
    assert vqe_res.gate_count > 0
    assert vqe_res.iterations > 0

    # Variational principle: ideal VQE energy cannot be below exact FCI energy (up to numerical tol)
    assert vqe_res.vqe_energy >= ref.fci_energy - 1e-6

    # Accuracy check: |E_VQE - E_FCI| <= 1.0 mHa (1e-3 Ha)
    abs_error_ha = abs(vqe_res.vqe_energy - ref.fci_energy)
    assert abs_error_ha < 1e-3


def test_h2_vqe_seed_reproducibility_and_cobyla() -> None:
    """Verify identical seeds produce identical VQE energies and COBYLA also converges on H2."""
    run1 = run_benchmark_vqe("H2", optimizer_name="COBYLA", seed=123)
    run2 = run_benchmark_vqe("H2", optimizer_name="COBYLA", seed=123)
    ref = run_benchmark_reference("H2", frozen_core=False)

    assert run1.vqe_energy == pytest.approx(run2.vqe_energy, abs=1e-12)
    assert run1.iterations == run2.iterations
    assert abs(run1.vqe_energy - ref.fci_energy) < 1e-3


def test_h2_warm_start_bond_scan() -> None:
    """Verify H2 bond scan (0.3 to 2.5 A) warm-starts from previous parameters and tracks FCI."""
    distances = [0.3, 0.5, 0.735, 1.1, 1.5, 2.0, 2.5]
    points = run_h2_bond_scan(bond_lengths=distances, seed=42, warm_start=True)

    assert len(points) == len(distances)
    assert points[0].warm_started is False
    for pt in points[1:]:
        assert pt.warm_started is True

    for pt in points:
        assert pt.converged is True
        assert pt.abs_error_mha < 1.0


def test_lih_frozen_core_vqe_matches_casci() -> None:
    """Verify LiH frozen-core (2e, 5o) maps to 8 qubits and VQE matches CASCI within 1 mHa."""
    ref_fc = run_benchmark_reference("LiH", frozen_core=True)
    vqe_res = run_benchmark_vqe(
        "LiH",
        frozen_core=True,
        ansatz_type="UCCSD",
        optimizer_name="SLSQP",
        seed=42,
    )

    assert vqe_res.molecule_id == "LiH"
    assert vqe_res.frozen_core is True
    assert vqe_res.active_electrons == 2
    assert vqe_res.active_orbitals == 5
    assert vqe_res.mapping == "parity_2q_reduced"
    assert vqe_res.num_qubits == 8
    assert vqe_res.converged is True
    assert vqe_res.circuit_depth > 0
    assert vqe_res.gate_count > 0

    abs_err_ha = abs(vqe_res.vqe_energy - ref_fc.fci_energy)
    assert abs_err_ha < 1e-3
