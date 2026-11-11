"""Week 1 pytest suite: RDKit descriptors, SMILES validation, benchmarks, and SQLite schema."""

import math
from pathlib import Path
import pytest

from qmed.molecules import (
    MoleculeValidationError,
    compute_descriptors,
    get_benchmark,
    load_benchmarks,
    validate_and_describe,
)
from qmed.storage import get_table_columns, init_db


def test_water_smiles_descriptors() -> None:
    """Validate water ('O') SMILES and check RDKit physicochemical descriptors."""
    result = validate_and_describe("O")
    assert result.is_valid is True
    assert result.error is None
    assert result.descriptors is not None

    desc = result.descriptors
    assert desc.canonical_smiles == "O"
    assert desc.formula == "H2O"
    assert desc.elements == ["H", "O"]
    assert desc.mw == pytest.approx(18.015, abs=0.01)
    assert desc.logp == pytest.approx(-0.8247, abs=0.05)
    # Note: RDKit's Lipinski HBD/HBA SMARTS patterns match organic hydroxyls ([O;H1])
    # and explicitly exclude unbonded water ([O;H2]), returning 0 for isolated H2O.
    assert desc.hbd == 0
    assert desc.hba == 0
    assert desc.rotatable_bonds == 0
    assert desc.tpsa == pytest.approx(31.5, abs=0.1)
    assert desc.qed == pytest.approx(0.3277, abs=0.01)


def test_ethanol_smiles_descriptors() -> None:
    """Validate ethanol ('CCO') SMILES and check RDKit physicochemical descriptors."""
    desc = compute_descriptors("CCO")
    assert desc.canonical_smiles == "CCO"
    assert desc.formula == "C2H6O"
    assert desc.elements == ["C", "H", "O"]
    assert desc.mw == pytest.approx(46.069, abs=0.01)
    assert desc.logp == pytest.approx(-0.0014, abs=0.05)
    assert desc.hbd == 1
    assert desc.hba == 1
    assert desc.tpsa == pytest.approx(20.23, abs=0.1)
    assert desc.rotatable_bonds == 0
    assert 0.35 <= desc.qed <= 0.45


@pytest.mark.parametrize(
    "bad_smiles",
    [
        "",
        "   ",
        "not_a_smiles_string",
        "C(C)(C)(C)(C)C",  # Pentavalent carbon (valence error)
        "12345",
    ],
)
def test_invalid_smiles_returns_clear_error(bad_smiles: str) -> None:
    """Ensure invalid SMILES strings raise MoleculeValidationError and return a clear error."""
    with pytest.raises(MoleculeValidationError, match="Invalid SMILES"):
        compute_descriptors(bad_smiles)

    result = validate_and_describe(bad_smiles)
    assert result.is_valid is False
    assert result.descriptors is None
    assert result.error is not None
    assert "Invalid SMILES" in result.error


def test_load_benchmarks_and_geometries() -> None:
    """Verify H2 (0.735 A), LiH (1.595 A), and H2O (equilibrium) in data/benchmarks.json."""
    benchmarks = load_benchmarks()
    assert set(benchmarks.keys()) == {"H2", "LiH", "H2O"}

    # Check H2 bond length = 0.735 A
    h2 = get_benchmark("H2")
    assert h2.basis == "STO-3G"
    assert h2.charge == 0
    assert h2.spin == 0
    assert len(h2.atoms) == 2
    h2_dist = math.dist(h2.atoms[0].coords, h2.atoms[1].coords)
    assert h2_dist == pytest.approx(0.735, abs=1e-6)

    # Check LiH bond length = 1.595 A
    lih = get_benchmark("LiH")
    assert lih.basis == "STO-3G"
    assert lih.charge == 0
    assert lih.spin == 0
    assert [a.symbol for a in lih.atoms] == ["Li", "H"]
    lih_dist = math.dist(lih.atoms[0].coords, lih.atoms[1].coords)
    assert lih_dist == pytest.approx(1.595, abs=1e-6)

    # Check water equilibrium geometry via alias 'water'
    water = get_benchmark("water")
    assert water.id == "H2O"
    assert water.basis == "STO-3G"
    assert water.charge == 0
    assert water.spin == 0
    assert [a.symbol for a in water.atoms] == ["O", "H", "H"]
    oh1 = math.dist(water.atoms[0].coords, water.atoms[1].coords)
    oh2 = math.dist(water.atoms[0].coords, water.atoms[2].coords)
    assert oh1 == pytest.approx(0.9572, abs=1e-3)
    assert oh2 == pytest.approx(0.9572, abs=1e-3)
    assert "O 0.000000 0.000000 0.000000" in water.to_pyscf_atom_string()


def test_sqlite_schema_initialization(tmp_path: Path) -> None:
    """Verify that storage.init_db creates the experiments table with all required columns."""
    db_file = tmp_path / "test_qmed.db"
    created_path = init_db(db_file)
    assert created_path.exists()

    columns = set(get_table_columns("experiments", db_file))
    expected_columns = {
        "id",
        "input_identifier",
        "timestamp",
        "config_json",
        "descriptors_json",
        "qee_decision",
        "qee_explanation_json",
        "basis",
        "active_space",
        "mapping",
        "ansatz",
        "optimizer",
        "backend",
        "shots",
        "seed",
        "noise_model",
        "vqe_energy",
        "hf_energy",
        "fci_energy",
        "error_ha",
        "uncertainty_ha",
        "iterations",
        "converged",
        "runtime_sec",
        "qubits",
        "depth",
        "gate_count",
        "reliability",
        "verdict",
    }
    assert expected_columns.issubset(columns)
