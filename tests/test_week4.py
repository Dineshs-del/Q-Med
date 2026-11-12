"""Week 4 pytest suite: Qiskit Aer noise models, synthetic label/verdict tests, and multi-seed benchmarks."""

import pytest

from qmed.benchmark import (
    classify_advantage_verdict,
    classify_reliability,
    run_multi_seed_benchmark,
)
from qmed.vqe_runner import (
    build_depolarizing_readout_noise_model,
    build_fake_backend_noise_model,
)


def test_synthetic_reliability_labels_all_branches() -> None:
    """Verify HIGH, MEDIUM, LOW, UNRELIABLE, active-space capping, and INVALID bound via synthetic inputs."""
    # 1. HIGH: converged, |err| <= 1.6 mHa (0.0005 Ha), uncertainty <= 1.6 mHa (0.0002 Ha), AS err = 0
    status, label, warns = classify_reliability(
        abs_error_ha=0.0005,
        uncertainty_ha=0.0002,
        converged=True,
        active_space_error_mha=0.0,
    )
    assert status == "VALID"
    assert label == "HIGH"
    assert len(warns) == 0

    # 2. MEDIUM (by error): converged, 1.6 mHa < |err| <= 10 mHa (0.005 Ha)
    _, label_med_err, _ = classify_reliability(
        abs_error_ha=0.0050,
        uncertainty_ha=0.0005,
        converged=True,
    )
    assert label_med_err == "MEDIUM"

    # 3. MEDIUM (by uncertainty): converged, |err| <= 1.6 mHa, but uncertainty > 1.6 mHa (0.003 Ha)
    _, label_med_unc, _ = classify_reliability(
        abs_error_ha=0.0008,
        uncertainty_ha=0.0030,
        converged=True,
    )
    assert label_med_unc == "MEDIUM"

    # 4. MEDIUM (capped from HIGH by active_space_error_mha > 1.6 mHa):
    _, label_capped, warns_capped = classify_reliability(
        abs_error_ha=0.0004,
        uncertainty_ha=0.0001,
        converged=True,
        active_space_error_mha=41.98,
    )
    assert label_capped == "MEDIUM"
    assert any("RELIABILITY_CAPPED" in w for w in warns_capped)

    # 5. LOW: converged, 10 mHa < |err| <= 50 mHa (0.025 Ha)
    _, label_low, _ = classify_reliability(
        abs_error_ha=0.0250,
        uncertainty_ha=0.0040,
        converged=True,
    )
    assert label_low == "LOW"

    # 6. UNRELIABLE (by error > 50 mHa):
    _, label_unrel_err, _ = classify_reliability(
        abs_error_ha=0.0800,
        uncertainty_ha=0.0050,
        converged=True,
    )
    assert label_unrel_err == "UNRELIABLE"

    # 7. UNRELIABLE (not converged even with small error):
    _, label_unrel_conv, _ = classify_reliability(
        abs_error_ha=0.0005,
        uncertainty_ha=0.0001,
        converged=False,
    )
    assert label_unrel_conv == "UNRELIABLE"

    # 8. INVALID (ideal VQE energy below matched FCI by > 1e-8 Ha):
    status_inv, label_inv, warns_inv = classify_reliability(
        abs_error_ha=0.001,
        uncertainty_ha=0.0,
        converged=True,
        vqe_energy=-1.138306,
        matched_fci_energy=-1.137306,
        is_noisy=False,
    )
    assert status_inv == "INVALID"
    assert label_inv == "INVALID"
    assert any("VARIATIONAL_BOUND_VIOLATION" in w for w in warns_inv)


def test_synthetic_advantage_verdicts_all_branches() -> None:
    """Verify INSUFFICIENT_EVIDENCE, NO_DEMONSTRATED_ADVANTAGE, POTENTIAL_ADVANTAGE, DEMONSTRATED_ADVANTAGE."""
    # 1. INSUFFICIENT_EVIDENCE (fewer than 3 seeds)
    assert (
        classify_advantage_verdict(
            converged=True,
            num_seeds=2,
            reliability="HIGH",
            abs_error_ha=0.0001,
            vqe_runtime_sec=0.1,
            classical_runtime_sec=0.2,
        )
        == "INSUFFICIENT_EVIDENCE"
    )

    # 2. INSUFFICIENT_EVIDENCE (not converged)
    assert (
        classify_advantage_verdict(
            converged=False,
            num_seeds=5,
            reliability="UNRELIABLE",
            abs_error_ha=0.1,
            vqe_runtime_sec=1.0,
            classical_runtime_sec=0.1,
        )
        == "INSUFFICIENT_EVIDENCE"
    )

    # 3. NO_DEMONSTRATED_ADVANTAGE (classical exact available & faster)
    assert (
        classify_advantage_verdict(
            converged=True,
            num_seeds=5,
            reliability="HIGH",
            abs_error_ha=0.0001,
            vqe_runtime_sec=1.5,
            classical_runtime_sec=0.05,
            classical_exact_available=True,
        )
        == "NO_DEMONSTRATED_ADVANTAGE"
    )

    # 4. POTENTIAL_ADVANTAGE (classical exact unavailable, MEDIUM accuracy within 10 mHa, slower VQE)
    assert (
        classify_advantage_verdict(
            converged=True,
            num_seeds=5,
            reliability="MEDIUM",
            abs_error_ha=0.0040,
            vqe_runtime_sec=120.0,
            classical_runtime_sec=60.0,
            classical_exact_available=False,
        )
        == "POTENTIAL_ADVANTAGE"
    )

    # 5. DEMONSTRATED_ADVANTAGE (classical exact unavailable, HIGH reliability <= 1.6 mHa, VQE faster)
    assert (
        classify_advantage_verdict(
            converged=True,
            num_seeds=5,
            reliability="HIGH",
            abs_error_ha=0.0008,
            vqe_runtime_sec=10.0,
            classical_runtime_sec=500.0,
            classical_exact_available=False,
        )
        == "DEMONSTRATED_ADVANTAGE"
    )


def test_noise_models_construction() -> None:
    """Verify depolarizing+readout and GenericBackendV2 fake-backend noise models build cleanly."""
    nm_depol = build_depolarizing_readout_noise_model(p_1q=0.001, p_2q=0.01, p_ro=0.02)
    assert "cx" in nm_depol.noise_instructions
    assert "measure" in nm_depol.noise_instructions

    nm_fake, label_fake = build_fake_backend_noise_model(num_qubits=4, seed=42)
    assert label_fake == "fake_generic_v2_4q"
    assert len(nm_fake.noise_instructions) > 0


def test_h2_5_seed_ideal_vs_noisy_benchmark() -> None:
    """Run 5-seed H2 ideal and depolarizing_readout benchmarks and verify metrics."""
    seeds = (42, 43, 44, 45, 46)
    ideal_res = run_multi_seed_benchmark("H2", seeds=seeds, noise_mode="ideal")
    noisy_res = run_multi_seed_benchmark(
        "H2", seeds=seeds, noise_mode="depolarizing_readout", shots=1024, maxiter=40
    )

    assert len(ideal_res.seed_runs) == 5
    assert ideal_res.reliability == "HIGH"
    assert ideal_res.verdict == "NO_DEMONSTRATED_ADVANTAGE"
    assert ideal_res.abs_mean_error_mha < 1.0
    assert ideal_res.uncertainty_mha < 1.0

    assert len(noisy_res.seed_runs) == 5
    assert noisy_res.shots == 1024
    assert noisy_res.noise_mode == "depolarizing_readout"
    # Noisy run has larger error and uncertainty than ideal statevector
    assert noisy_res.abs_mean_error_mha > ideal_res.abs_mean_error_mha
    assert noisy_res.uncertainty_mha > ideal_res.uncertainty_mha
    assert noisy_res.verdict in {"NO_DEMONSTRATED_ADVANTAGE", "INSUFFICIENT_EVIDENCE"}
