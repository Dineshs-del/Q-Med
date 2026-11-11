"""Week 2 pytest suite: Classical PySCF Hartree-Fock and FCI/CASCI reference energies."""

import pytest

from qmed.classical import (
    clear_classical_cache,
    compute_all_benchmark_references,
    run_benchmark_reference,
    run_classical_reference,
)


def test_h2_sto3g_hf_and_fci_energies() -> None:
    """Verify H2 at 0.735 A (STO-3G) yields HF ~ -1.117 Ha and FCI ~ -1.137 Ha."""
    res = run_benchmark_reference("H2", frozen_core=False)

    assert res.molecule_id == "H2"
    assert res.basis == "STO-3G"
    assert res.n_electrons == 2
    assert res.n_orbitals == 2
    assert res.active_electrons == 2
    assert res.active_orbitals == 2
    assert res.n_frozen_core_orbitals == 0
    assert res.hf_converged is True
    assert res.fci_converged is True

    # Known literature values for H2 at 0.735 A in STO-3G:
    # HF  = -1.116999 Ha (~ -1.117 Ha)
    # FCI = -1.137306 Ha (~ -1.137 Ha)
    assert res.hf_energy == pytest.approx(-1.117, abs=1e-3)
    assert res.fci_energy == pytest.approx(-1.137, abs=1e-3)
    assert res.fci_energy < res.hf_energy
    assert res.correlation_energy == pytest.approx(res.fci_energy - res.hf_energy, abs=1e-12)
    assert res.total_runtime_sec > 0.0


def test_lih_all_electron_vs_frozen_core() -> None:
    """Verify LiH (1.595 A, STO-3G) all-electron (4e, 6o) and frozen-core (2e, 5o) FCI."""
    res_full = run_benchmark_reference("LiH", frozen_core=False)
    res_fc = run_benchmark_reference("LiH", frozen_core=True)

    assert res_full.n_electrons == 4
    assert res_full.n_orbitals == 6
    assert res_full.active_electrons == 4
    assert res_full.active_orbitals == 6
    assert res_full.n_frozen_core_orbitals == 0

    assert res_fc.active_electrons == 2
    assert res_fc.active_orbitals == 5
    assert res_fc.n_frozen_core_orbitals == 1

    # HF energy is identical regardless of post-HF active space
    assert res_full.hf_energy == pytest.approx(res_fc.hf_energy, abs=1e-10)
    assert res_full.hf_energy == pytest.approx(-7.8620, abs=1e-3)

    # Variational principle: all-electron FCI <= frozen-core CASCI <= HF
    assert res_full.fci_energy <= res_fc.fci_energy <= res_full.hf_energy
    # Freezing Li 1s core changes correlation energy by less than 0.5 mHa
    assert abs(res_full.fci_energy - res_fc.fci_energy) < 5e-4


def test_water_all_electron_and_frozen_core() -> None:
    """Verify water (H2O, STO-3G) all-electron (10e, 7o), frozen-core (8e, 6o), and CAS(4e, 4o)."""
    res_full = run_benchmark_reference("water", frozen_core=False)
    res_fc = run_benchmark_reference("water", frozen_core=True)
    res_cas44 = run_benchmark_reference("water", active_space=(4, 4))

    assert res_full.active_electrons == 10
    assert res_full.active_orbitals == 7
    assert res_fc.active_electrons == 8
    assert res_fc.active_orbitals == 6
    assert res_fc.n_frozen_core_orbitals == 1
    assert res_cas44.active_electrons == 4
    assert res_cas44.active_orbitals == 4

    assert res_full.hf_energy == pytest.approx(-74.9629, abs=1e-3)
    assert res_full.fci_energy == pytest.approx(-75.0124, abs=1e-3)
    assert res_fc.fci_energy == pytest.approx(-75.0123, abs=1e-3)

    # Variational hierarchy across nested active spaces:
    # Full FCI (10e, 7o) <= Frozen-core FCI (8e, 6o) <= CASCI (4e, 4o) <= HF
    assert res_full.fci_energy <= res_fc.fci_energy <= res_cas44.fci_energy <= res_full.hf_energy


def test_caching_and_all_benchmarks_helper() -> None:
    """Verify compute_all_benchmark_references populates and reuses the in-memory cache."""
    clear_classical_cache()
    first_batch = compute_all_benchmark_references(frozen_core=True)
    second_batch = compute_all_benchmark_references(frozen_core=True)

    assert set(first_batch.keys()) == {"H2", "LiH", "H2O"}
    for key in ("H2", "LiH", "H2O"):
        # Exact same cached object instance is returned
        assert first_batch[key] is second_batch[key]


def test_invalid_active_space_raises() -> None:
    """Ensure impossible active spaces raise a clear ValueError."""
    with pytest.raises(ValueError, match="Invalid active_electrons"):
        run_classical_reference("H 0 0 0; H 0 0 0.735", active_space=(4, 2))
