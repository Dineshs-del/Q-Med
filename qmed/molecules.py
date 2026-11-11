"""Molecular validation, RDKit physicochemical descriptors, and benchmark loader."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

from qmed.config import BENCHMARKS_PATH

# Silence verbose RDKit C++ stderr warnings on intentionally invalid SMILES inputs
RDLogger.DisableLog("rdApp.error")


class MoleculeValidationError(ValueError):
    """Raised when a SMILES string is empty, malformed, or chemically invalid."""


@dataclass(frozen=True)
class MolecularDescriptors:
    """RDKit physicochemical descriptors for a validated molecule."""

    input_smiles: str
    canonical_smiles: str
    formula: str
    elements: List[str]
    mw: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rotatable_bonds: int
    qed: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of descriptors."""
        return asdict(self)


@dataclass(frozen=True)
class ValidationResult:
    """Result container for SMILES validation and descriptor calculation."""

    is_valid: bool
    descriptors: Optional[MolecularDescriptors] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the validation outcome."""
        return {
            "is_valid": self.is_valid,
            "descriptors": self.descriptors.to_dict() if self.descriptors else None,
            "error": self.error,
        }


@dataclass(frozen=True)
class AtomCoord:
    """Single atomic symbol and 3D Cartesian coordinate in Angstroms."""

    symbol: str
    coords: List[float]


@dataclass(frozen=True)
class BenchmarkMolecule:
    """Explicit-geometry benchmark molecule specification."""

    id: str
    name: str
    smiles: str
    basis: str
    charge: int
    spin: int
    atoms: List[AtomCoord]
    geometry_note: str = ""

    def to_pyscf_atom_string(self) -> str:
        """Format atomic coordinates into a PySCF-compatible geometry string."""
        lines = [
            f"{atom.symbol} {atom.coords[0]:.6f} {atom.coords[1]:.6f} {atom.coords[2]:.6f}"
            for atom in self.atoms
        ]
        return "; ".join(lines)


def compute_descriptors(smiles: str) -> MolecularDescriptors:
    """Validate a SMILES string and compute RDKit descriptors.

    Raises:
        MoleculeValidationError: If `smiles` is not a non-empty string or cannot
            be parsed and sanitized by RDKit.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise MoleculeValidationError("Invalid SMILES: input must be a non-empty string.")

    cleaned = smiles.strip()
    mol = Chem.MolFromSmiles(cleaned, sanitize=True)
    if mol is None or mol.GetNumAtoms() == 0:
        raise MoleculeValidationError(
            f"Invalid SMILES: '{cleaned}' could not be parsed as a valid chemical structure."
        )

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mol_with_hs = Chem.AddHs(mol)
    elements = sorted({atom.GetSymbol() for atom in mol_with_hs.GetAtoms()})

    return MolecularDescriptors(
        input_smiles=cleaned,
        canonical_smiles=canonical_smiles,
        formula=formula,
        elements=elements,
        mw=round(float(Descriptors.MolWt(mol)), 4),
        logp=round(float(Crippen.MolLogP(mol)), 4),
        hbd=int(Lipinski.NumHDonors(mol)),
        hba=int(Lipinski.NumHAcceptors(mol)),
        tpsa=round(float(rdMolDescriptors.CalcTPSA(mol)), 4),
        rotatable_bonds=int(Lipinski.NumRotatableBonds(mol)),
        qed=round(float(QED.qed(mol)), 4),
    )


def validate_and_describe(smiles: str) -> ValidationResult:
    """Validate a SMILES string and return a structured ValidationResult without raising."""
    try:
        descriptors = compute_descriptors(smiles)
        return ValidationResult(is_valid=True, descriptors=descriptors, error=None)
    except MoleculeValidationError as exc:
        return ValidationResult(is_valid=False, descriptors=None, error=str(exc))


def load_benchmarks(path: Path = BENCHMARKS_PATH) -> Dict[str, BenchmarkMolecule]:
    """Load explicit-geometry benchmark molecules from data/benchmarks.json."""
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found at {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    benchmarks: Dict[str, BenchmarkMolecule] = {}
    for key, entry in raw_data.items():
        atoms = [
            AtomCoord(symbol=str(a["symbol"]), coords=[float(x) for x in a["coords"]])
            for a in entry["atoms"]
        ]
        benchmarks[key] = BenchmarkMolecule(
            id=str(entry.get("id", key)),
            name=str(entry["name"]),
            smiles=str(entry["smiles"]),
            basis=str(entry["basis"]),
            charge=int(entry["charge"]),
            spin=int(entry["spin"]),
            atoms=atoms,
            geometry_note=str(entry.get("geometry_note", "")),
        )
    return benchmarks


def get_benchmark(name: str, path: Path = BENCHMARKS_PATH) -> BenchmarkMolecule:
    """Retrieve a single benchmark molecule by key or alias (e.g. 'H2', 'LiH', 'H2O', 'water')."""
    benchmarks = load_benchmarks(path)
    normalized = name.strip().upper()
    alias_map = {
        "H2": "H2",
        "HYDROGEN": "H2",
        "LIH": "LiH",
        "LITHIUM HYDRIDE": "LiH",
        "H2O": "H2O",
        "WATER": "H2O",
    }
    target_key = alias_map.get(normalized, name.strip())
    if target_key not in benchmarks:
        available = ", ".join(sorted(benchmarks.keys()))
        raise KeyError(f"Unknown benchmark molecule '{name}'. Available: {available}")
    return benchmarks[target_key]
