"""Week 5 pytest suite: QEE v1 decisions, aspirin DEFERRED, resource estimation, full pipeline, and SQLite replay."""

from pathlib import Path
import pytest

from qmed.active_space import estimate_circuit_depth, estimate_qubits_from_smiles
from qmed.config import (
    MAX_CIRCUIT_DEPTH,
    MAX_HYBRID_FULL_QUBITS,
    MAX_SIMULATED_QUBITS,
)
from qmed.qee import evaluate_eligibility
from qmed.storage import (
    get_experiment_by_id,
    list_experiments,
    replay_experiment,
    run_qmed_pipeline,
)


def test_active_space_qubit_and_depth_estimation() -> None:
    """Verify analytical qubit, UCCSD parameter, and circuit depth estimates in active_space.py."""
    h2_est = estimate_qubits_from_smiles("[H][H]", frozen_core=True)
    assert h2_est.active_electrons == 2
    assert h2_est.active_spatial_orbitals == 2
    assert h2_est.num_qubits == 2
    assert h2_est.estimated_uccsd_parameters == 3
    assert h2_est.estimated_circuit_depth == 15

    lih_est = estimate_qubits_from_smiles("[LiH]", frozen_core=True)
    assert lih_est.active_electrons == 2
    assert lih_est.active_spatial_orbitals == 5
    assert lih_est.num_qubits == 8
    assert lih_est.estimated_uccsd_parameters == 24
    assert lih_est.estimated_circuit_depth == 1771

    h2o_est = estimate_qubits_from_smiles("O", frozen_core=True)
    assert h2o_est.active_electrons == 8
    assert h2o_est.active_spatial_orbitals == 6
    assert h2o_est.num_qubits == 12
    assert h2o_est.estimated_uccsd_parameters == 92
    assert h2o_est.estimated_circuit_depth == 10388

    params_44, depth_44 = estimate_circuit_depth(4, 4, 8, ansatz_type="UCCSD")
    assert params_44 == 26
    assert depth_44 == 2005


def test_qee_decision_classical_only() -> None:
    """Verify QEE routes closed-shell 0-virtual-orbital systems ([He]) and classical_sufficient to CLASSICAL_ONLY."""
    he_res = evaluate_eligibility("[He]")
    assert he_res.is_valid is True
    assert he_res.decision == "CLASSICAL_ONLY"
    assert he_res.features["estimated_uccsd_parameters"] == 0
    assert any("0 virtual orbitals" in r or "Hartree-Fock is exact" in r for r in he_res.reasons)

    forced_res = evaluate_eligibility("[H][H]", classical_sufficient=True)
    assert forced_res.is_valid is True
    assert forced_res.decision == "CLASSICAL_ONLY"


def test_qee_decision_simulated_quantum() -> None:
    """Verify QEE routes H2 and LiH (qubits <= 12 and depth <= 5000) to SIMULATED_QUANTUM."""
    h2_res = evaluate_eligibility("H2")
    assert h2_res.is_valid is True
    assert h2_res.decision == "SIMULATED_QUANTUM"
    assert h2_res.features["estimated_qubits"] == 2
    assert h2_res.features["estimated_circuit_depth"] <= MAX_CIRCUIT_DEPTH
    assert any(str(MAX_SIMULATED_QUBITS) in r for r in h2_res.reasons)

    lih_res = evaluate_eligibility("LiH")
    assert lih_res.is_valid is True
    assert lih_res.decision == "SIMULATED_QUANTUM"
    assert lih_res.features["estimated_qubits"] == 8
    assert lih_res.features["estimated_circuit_depth"] == 1771


def test_qee_decision_hybrid() -> None:
    """Verify QEE routes water (depth > 5000) and ammonia (14 qubits <= 24) to HYBRID."""
    water_res = evaluate_eligibility("O")
    assert water_res.is_valid is True
    assert water_res.decision == "HYBRID"
    assert water_res.recommended_active_space == (4, 4)
    assert any(str(MAX_CIRCUIT_DEPTH) in r for r in water_res.reasons)
    assert any("active_space_error_mHa" in w for w in water_res.warnings)

    nh3_res = evaluate_eligibility("N")
    assert nh3_res.is_valid is True
    assert nh3_res.decision == "HYBRID"
    assert nh3_res.features["estimated_qubits"] == 14
    assert nh3_res.recommended_active_space == (4, 4)
    assert any(str(MAX_HYBRID_FULL_QUBITS) in r for r in nh3_res.reasons)


def test_qee_decision_deferred_aspirin() -> None:
    """Verify aspirin (CC(=O)OC1=CC=CC=C1C(=O)O) returns DEFERRED with a clear explanation."""
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    res = evaluate_eligibility(aspirin_smiles)

    assert res.is_valid is True
    assert res.decision == "DEFERRED"
    assert res.features["formula"] == "C9H8O4"
    assert res.features["active_electrons"] == 68
    assert res.features["active_spatial_orbitals"] == 60
    assert res.features["estimated_qubits"] == 120
    assert res.features["estimated_circuit_depth"] > MAX_CIRCUIT_DEPTH
    assert len(res.reasons) >= 2
    assert any("120 qubits" in r for r in res.reasons)
    assert any(str(MAX_SIMULATED_QUBITS) in r for r in res.reasons)
    assert any("discard" in r or "intractable" in r for r in res.reasons)


def test_full_pipeline_and_sqlite_replay_reproducibility(tmp_path: Path) -> None:
    """Wire input -> QEE -> active space -> VQE ideal+noisy -> benchmark -> SQLite -> replay by ID."""
    db_file = tmp_path / "test_week5_pipeline.db"

    # 1. Run full pipeline on H2 across 3 seeds (both ideal and noisy)
    pipe_h2 = run_qmed_pipeline(
        "H2",
        seeds=(42, 43, 44),
        run_noisy=True,
        noisy_mode="depolarizing_readout",
        shots=512,
        noisy_maxiter=25,
        db_path=db_file,
    )

    assert pipe_h2.qee.decision == "SIMULATED_QUANTUM"
    assert pipe_h2.ideal_benchmark is not None
    assert pipe_h2.noisy_benchmark is not None
    assert pipe_h2.ideal_benchmark.reliability == "HIGH"
    assert len(pipe_h2.stored_experiment_ids) == 8  # 3 ideal seeds + 1 ideal agg + 3 noisy seeds + 1 noisy agg
    assert pipe_h2.primary_experiment_id is not None

    # 2. Replay a stored ideal single-seed experiment by ID and confirm exact reproduction
    ideal_seed_id = pipe_h2.stored_experiment_ids[0]
    replay_ideal = replay_experiment(ideal_seed_id, db_path=db_file)
    assert replay_ideal.reproduced is True
    assert replay_ideal.stored_qee_decision == "SIMULATED_QUANTUM"
    assert replay_ideal.replayed_qee_decision == "SIMULATED_QUANTUM"
    assert replay_ideal.energy_diff_ha is not None
    assert replay_ideal.energy_diff_ha < 1e-10
    assert replay_ideal.replayed_iterations == replay_ideal.stored_iterations

    # 3. Replay a stored noisy single-seed experiment by ID and confirm seeded reproduction
    noisy_seed_id = pipe_h2.stored_experiment_ids[4]
    replay_noisy = replay_experiment(noisy_seed_id, db_path=db_file)
    assert replay_noisy.reproduced is True
    assert replay_noisy.energy_diff_ha is not None
    assert replay_noisy.energy_diff_ha < 1e-10

    # 4. Run pipeline on Aspirin (DEFERRED) and verify SQLite storage + replay
    pipe_aspirin = run_qmed_pipeline(
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        db_path=db_file,
    )
    assert pipe_aspirin.qee.decision == "DEFERRED"
    assert pipe_aspirin.ideal_benchmark is None
    assert pipe_aspirin.noisy_benchmark is None
    assert pipe_aspirin.primary_experiment_id is not None

    stored_aspirin = get_experiment_by_id(pipe_aspirin.primary_experiment_id, db_path=db_file)
    assert stored_aspirin["qee_decision"] == "DEFERRED"
    assert stored_aspirin["qubits"] == 120
    assert stored_aspirin["vqe_energy"] is None

    replay_aspirin = replay_experiment(pipe_aspirin.primary_experiment_id, db_path=db_file)
    assert replay_aspirin.reproduced is True
    assert replay_aspirin.replayed_qee_decision == "DEFERRED"

    all_rows = list_experiments(db_path=db_file)
    assert len(all_rows) == 9
