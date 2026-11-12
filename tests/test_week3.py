"""Week 3 pytest suite: H2 VQE, H2 scan + plot, LiH frozen-core, H2O Option A & B, and guards."""

from pathlib import Path
import pytest

from qmed.active_space import prepare_benchmark_problem
from qmed.classical import run_benchmark_reference
from qmed.config import MAX_CIRCUIT_DEPTH
from qmed.qee import evaluate_eligibility
from qmed.vqe_runner import (
    evaluate_reliability_and_bounds,
    run_benchmark_vqe,
    run_h2_bond_scan,
    run_water_option_b_demo,
    save_h2_curve_plot,
)


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
    assert vqe_res.status == "VALID"
    assert vqe_res.variational_valid is True
    assert vqe_res.reliability == "HIGH"
    assert vqe_res.active_space_error_mha == pytest.approx(0.0, abs=1e-9)

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


def test_h2_warm_start_bond_scan_and_plot(tmp_path: Path) -> None:
    """Verify H2 bond scan warm-starts from previous parameters, tracks FCI, and saves plot."""
    distances = [0.5, 0.735, 1.5, 2.5]
    points = run_h2_bond_scan(bond_lengths=distances, seed=42, warm_start=True)

    assert len(points) == len(distances)
    assert points[0].warm_started is False
    for pt in points[1:]:
        assert pt.warm_started is True

    for pt in points:
        assert pt.converged is True
        assert pt.abs_error_mha < 1.0

    plot_file = tmp_path / "h2_curve.png"
    saved_path = save_h2_curve_plot(points, output_path=plot_file)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 1000


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
    assert vqe_res.status == "VALID"
    assert vqe_res.reliability == "HIGH"

    abs_err_ha = abs(vqe_res.vqe_energy - ref_fc.fci_energy)
    assert abs_err_ha < 1e-3


def test_water_option_a_qee_hybrid_and_active_space_capping() -> None:
    """Verify QEE routes H2O (8e,6o) to HYBRID due to depth and caps (4e,4o) reliability at MEDIUM."""
    qee_res = evaluate_eligibility("O", estimated_qubits=12, estimated_depth=10388)
    assert qee_res.is_valid is True
    assert qee_res.decision == "HYBRID"
    assert qee_res.recommended_active_space == (4, 4)
    assert any(str(MAX_CIRCUIT_DEPTH) in r for r in qee_res.reasons)

    ref_44 = run_benchmark_reference("H2O", active_space=(4, 4))
    ref_fc = run_benchmark_reference("H2O", frozen_core=True)
    vqe_res = run_benchmark_vqe(
        "H2O",
        active_space=qee_res.recommended_active_space,
        ansatz_type="UCCSD",
        optimizer_name="SLSQP",
        seed=42,
    )

    assert vqe_res.active_electrons == 4
    assert vqe_res.active_orbitals == 4
    assert vqe_res.num_qubits == 8
    assert vqe_res.status == "VALID"
    assert vqe_res.variational_valid is True

    # Subspace error is < 1.6 mHa, but active_space_error_mha is ~ +41.98 mHa (> 1.6 mHa),
    # so reliability must be capped at MEDIUM (cannot be HIGH).
    expected_as_err = (ref_44.fci_energy - ref_fc.fci_energy) * 1000.0
    assert vqe_res.active_space_error_mha == pytest.approx(expected_as_err, abs=1e-4)
    assert vqe_res.active_space_error_mha > 1.6
    assert abs(vqe_res.vqe_energy - ref_44.fci_energy) * 1000.0 < 1.6
    assert vqe_res.reliability == "MEDIUM"
    assert any("ACTIVE_SPACE_TRUNCATION" in w for w in vqe_res.warnings)
    assert any("RELIABILITY_CAPPED" in w for w in vqe_res.warnings)


def test_variational_bound_guard_marks_invalid() -> None:
    """Verify that a VQE energy below matched reference by > 1e-8 Ha is marked INVALID."""
    matched_fci = -1.13730604
    # 1. Valid within numerical tolerance (5e-9 Ha below FCI is allowed)
    status_ok, rel_ok, valid_ok, _, warns_ok = evaluate_reliability_and_bounds(
        vqe_energy=matched_fci - 5e-9,
        matched_fci_energy=matched_fci,
        converged=True,
    )
    assert status_ok == "VALID"
    assert valid_ok is True
    assert rel_ok == "HIGH"
    assert len(warns_ok) == 0

    # 2. Invalid violation: 1e-5 Ha below matched FCI (> 1e-8 Ha guard)
    status_bad, rel_bad, valid_bad, _, warns_bad = evaluate_reliability_and_bounds(
        vqe_energy=matched_fci - 1e-5,
        matched_fci_energy=matched_fci,
        converged=True,
    )
    assert status_bad == "INVALID"
    assert rel_bad == "INVALID"
    assert valid_bad is False
    assert any("VARIATIONAL_BOUND_VIOLATION" in w for w in warns_bad)


def test_water_option_b_ansatz_limitation_demo() -> None:
    """Verify Option B: Water (8e,6o), parity (10 qubits), EfficientSU2(reps=1), 3 seeds."""
    runs = run_water_option_b_demo(seeds=(42, 43, 44), maxiter=120)
    assert len(runs) == 3

    for r in runs:
        assert r.molecule_id == "H2O"
        assert r.active_electrons == 8
        assert r.active_orbitals == 6
        assert r.mapping == "parity_2q_reduced"
        assert r.num_qubits == 10
        assert r.ansatz == "EfficientSU2"
        assert r.run_label == "ansatz_limitation_demo"
        assert r.status == "VALID"
        assert r.variational_valid is True
        assert any("ANSATZ_LIMITATION_DEMO" in w for w in r.warnings)
