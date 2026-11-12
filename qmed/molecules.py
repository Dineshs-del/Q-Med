"""Molecular validation, RDKit physicochemical descriptors, and benchmark loader."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

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


# Standard single-bond covalent radii in Angstroms for bond perception from 3D coordinates
COVALENT_RADII_ANGSTROM: Dict[str, float] = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
}

# Standard Jmol / CPK element colors for 3D visualization
CPK_ELEMENT_COLORS: Dict[str, str] = {
    "H": "#E2E8F0",
    "He": "#D9FFFF",
    "Li": "#CC80FF",
    "Be": "#C2FF00",
    "B": "#FFB5B5",
    "C": "#64748B",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Ne": "#B3E3F5",
    "S": "#FFFF30",
    "P": "#FF8000",
    "Cl": "#1FF01F",
}

APPROXIMATE_GEOMETRY_CAPTION: str = (
    "Approximate force-field geometry, not a quantum-calculated structure."
)


@dataclass(frozen=True)
class BondLengthEntry:
    """Single covalent bond length computed from 3D atomic coordinates."""

    atom_i_index: int
    atom_j_index: int
    atom_i_label: str
    atom_j_label: str
    bond_label: str
    distance_angstrom: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the bond length entry."""
        return asdict(self)


@dataclass(frozen=True)
class BondAngleEntry:
    """Single bond angle (in degrees) computed from 3D atomic coordinates."""

    atom_i_index: int
    vertex_j_index: int
    atom_k_index: int
    angle_label: str
    angle_degrees: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the bond angle entry."""
        return asdict(self)


@dataclass(frozen=True)
class Structure3DData:
    """Complete 3D molecular coordinate and geometry package for interactive visualization."""

    input_identifier: str
    canonical_smiles: str
    formula: str
    source: str
    caption: str
    atoms: List[AtomCoord]
    bonds: List[Tuple[int, int]]
    xyz_block: str
    bond_lengths: List[BondLengthEntry]
    bond_angles: List[BondAngleEntry]

    def to_xyz(self, include_hydrogens: bool = True) -> str:
        """Format coordinates as an XYZ block, optionally omitting hydrogen atoms."""
        return atoms_to_xyz_block(
            self.atoms,
            comment=f"{self.formula} ({self.source})",
            include_hydrogens=include_hydrogens,
        )


def compute_distance_angstrom(coord_a: Sequence[float], coord_b: Sequence[float]) -> float:
    """Compute Euclidean distance in Angstroms between two 3D Cartesian points."""
    if len(coord_a) != 3 or len(coord_b) != 3:
        raise ValueError("Coordinates must be 3D [x, y, z] vectors.")
    dx = float(coord_a[0]) - float(coord_b[0])
    dy = float(coord_a[1]) - float(coord_b[1])
    dz = float(coord_a[2]) - float(coord_b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_angle_degrees(
    coord_a: Sequence[float],
    vertex_b: Sequence[float],
    coord_c: Sequence[float],
) -> float:
    """Compute the angle A-B-C in degrees with B as the central vertex."""
    ux = float(coord_a[0]) - float(vertex_b[0])
    uy = float(coord_a[1]) - float(vertex_b[1])
    uz = float(coord_a[2]) - float(vertex_b[2])
    vx = float(coord_c[0]) - float(vertex_b[0])
    vy = float(coord_c[1]) - float(vertex_b[1])
    vz = float(coord_c[2]) - float(vertex_b[2])

    norm_u = math.sqrt(ux * ux + uy * uy + uz * uz)
    norm_v = math.sqrt(vx * vx + vy * vy + vz * vz)
    if norm_u < 1e-12 or norm_v < 1e-12:
        raise ValueError("Cannot compute bond angle for coincident atomic coordinates.")

    cos_theta = (ux * vx + uy * vy + uz * vz) / (norm_u * norm_v)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def atoms_to_xyz_block(
    atoms: Sequence[AtomCoord],
    comment: str = "",
    include_hydrogens: bool = True,
) -> str:
    """Format a sequence of AtomCoord objects into a standard multi-line XYZ string."""
    if include_hydrogens:
        selected = list(atoms)
    else:
        non_h = [a for a in atoms if a.symbol != "H"]
        selected = non_h if non_h else list(atoms)

    lines = [str(len(selected)), comment.replace("\n", " ")]
    for atom in selected:
        x, y, z = float(atom.coords[0]), float(atom.coords[1]), float(atom.coords[2])
        lines.append(f"{atom.symbol:2s} {x:12.6f} {y:12.6f} {z:12.6f}")
    return "\n".join(lines)


def infer_bonds_from_coordinates(
    atoms: Sequence[AtomCoord],
    tolerance_factor: float = 1.30,
) -> List[Tuple[int, int]]:
    """Infer covalent bond index pairs (i, j) from 3D coordinates using covalent radii."""
    bonds: List[Tuple[int, int]] = []
    n_atoms = len(atoms)
    for i in range(n_atoms):
        r_i = COVALENT_RADII_ANGSTROM.get(atoms[i].symbol, 0.77)
        for j in range(i + 1, n_atoms):
            r_j = COVALENT_RADII_ANGSTROM.get(atoms[j].symbol, 0.77)
            dist = compute_distance_angstrom(atoms[i].coords, atoms[j].coords)
            max_bond_dist = tolerance_factor * (r_i + r_j)
            if 0.10 < dist <= max_bond_dist:
                bonds.append((i, j))
    return bonds


def compute_bond_lengths_and_angles(
    atoms: Sequence[AtomCoord],
    bonds: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[List[BondLengthEntry], List[BondAngleEntry]]:
    """Compute bond lengths (Å) and bond angles (°) from 3D atomic coordinates."""
    effective_bonds = (
        list(bonds) if bonds is not None else infer_bonds_from_coordinates(atoms)
    )

    length_entries: List[BondLengthEntry] = []
    adjacency: Dict[int, List[int]] = {idx: [] for idx in range(len(atoms))}

    for i, j in effective_bonds:
        idx_a, idx_b = (i, j) if i < j else (j, i)
        adjacency[idx_a].append(idx_b)
        adjacency[idx_b].append(idx_a)

        label_a = f"{atoms[idx_a].symbol}{idx_a + 1}"
        label_b = f"{atoms[idx_b].symbol}{idx_b + 1}"
        dist = compute_distance_angstrom(atoms[idx_a].coords, atoms[idx_b].coords)
        length_entries.append(
            BondLengthEntry(
                atom_i_index=idx_a + 1,
                atom_j_index=idx_b + 1,
                atom_i_label=label_a,
                atom_j_label=label_b,
                bond_label=f"{label_a}–{label_b}",
                distance_angstrom=round(dist, 4),
            )
        )

    angle_entries: List[BondAngleEntry] = []
    for vertex_j, neighbors in sorted(adjacency.items()):
        sorted_neighbors = sorted(set(neighbors))
        if len(sorted_neighbors) < 2:
            continue
        for pos_a in range(len(sorted_neighbors)):
            for pos_b in range(pos_a + 1, len(sorted_neighbors)):
                i = sorted_neighbors[pos_a]
                k = sorted_neighbors[pos_b]
                deg = compute_angle_degrees(
                    atoms[i].coords,
                    atoms[vertex_j].coords,
                    atoms[k].coords,
                )
                label_i = f"{atoms[i].symbol}{i + 1}"
                label_j = f"{atoms[vertex_j].symbol}{vertex_j + 1}"
                label_k = f"{atoms[k].symbol}{k + 1}"
                angle_entries.append(
                    BondAngleEntry(
                        atom_i_index=i + 1,
                        vertex_j_index=vertex_j + 1,
                        atom_k_index=k + 1,
                        angle_label=f"{label_i}–{label_j}–{label_k}",
                        angle_degrees=round(deg, 2),
                    )
                )

    return length_entries, angle_entries


def smiles_to_3d_coordinates(
    smiles: str,
    seed: int = 42,
) -> Tuple[List[AtomCoord], List[Tuple[int, int]], str, str]:
    """Generate 3D atomic coordinates from SMILES using RDKit AddHs, seeded EmbedMolecule, and MMFF."""
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
    mol_h = Chem.AddHs(mol)

    if mol_h.GetNumAtoms() == 1:
        atom = mol_h.GetAtomWithIdx(0)
        return [AtomCoord(symbol=atom.GetSymbol(), coords=[0.0, 0.0, 0.0])], [], canonical_smiles, formula

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    embed_status = AllChem.EmbedMolecule(mol_h, params)
    if embed_status != 0:
        embed_status = AllChem.EmbedMolecule(mol_h, randomSeed=int(seed), useRandomCoords=True)

    if embed_status == 0 and mol_h.GetNumConformers() > 0:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol_h):
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            else:
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
    else:
        AllChem.Compute2DCoords(mol_h)

    conf = mol_h.GetConformer()
    atoms: List[AtomCoord] = []
    for idx in range(mol_h.GetNumAtoms()):
        atom = mol_h.GetAtomWithIdx(idx)
        pos = conf.GetAtomPosition(idx)
        atoms.append(
            AtomCoord(
                symbol=atom.GetSymbol(),
                coords=[round(float(pos.x), 6), round(float(pos.y), 6), round(float(pos.z), 6)],
            )
        )

    bonds: List[Tuple[int, int]] = []
    for bond in mol_h.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        bonds.append((min(i, j), max(i, j)))

    return atoms, bonds, canonical_smiles, formula


def load_molecule_3d_structure(
    input_identifier: str,
    use_benchmark_coords: bool = True,
    seed: int = 42,
    benchmarks_path: Path = BENCHMARKS_PATH,
) -> Structure3DData:
    """Load exact benchmark 3D coordinates (for preset benchmarks) or RDKit MMFF 3D coordinates (for custom SMILES)."""
    if not isinstance(input_identifier, str) or not input_identifier.strip():
        raise MoleculeValidationError("Invalid molecule input: must be a non-empty string.")

    cleaned = input_identifier.strip()
    if use_benchmark_coords:
        try:
            bm = get_benchmark(cleaned, path=benchmarks_path)
            bonds = infer_bonds_from_coordinates(bm.atoms)
            lengths, angles = compute_bond_lengths_and_angles(bm.atoms, bonds=bonds)
            xyz_block = atoms_to_xyz_block(
                bm.atoms,
                comment=f"{bm.id} ({bm.name}) — {bm.geometry_note}",
                include_hydrogens=True,
            )
            return Structure3DData(
                input_identifier=cleaned,
                canonical_smiles=bm.smiles,
                formula=bm.id,
                source="benchmark_exact",
                caption=(
                    f"Exact benchmark geometry from data/benchmarks.json ({bm.geometry_note}) — "
                    "identical to the coordinates used by PySCF and VQE."
                ),
                atoms=list(bm.atoms),
                bonds=bonds,
                xyz_block=xyz_block,
                bond_lengths=lengths,
                bond_angles=angles,
            )
        except KeyError:
            pass

    # Map common aliases (e.g. "water", "aspirin") if passed in custom mode
    alias_smiles_map = {
        "H2": "[H][H]",
        "HYDROGEN": "[H][H]",
        "LIH": "[LiH]",
        "LITHIUM HYDRIDE": "[LiH]",
        "H2O": "O",
        "WATER": "O",
        "ASPIRIN": "CC(=O)OC1=CC=CC=C1C(=O)O",
    }
    smiles_str = alias_smiles_map.get(cleaned.upper(), cleaned)
    atoms, bonds, canonical_smiles, formula = smiles_to_3d_coordinates(smiles_str, seed=seed)
    lengths, angles = compute_bond_lengths_and_angles(atoms, bonds=bonds)
    xyz_block = atoms_to_xyz_block(
        atoms,
        comment=f"{formula} — RDKit MMFF94 force-field geometry",
        include_hydrogens=True,
    )
    return Structure3DData(
        input_identifier=cleaned,
        canonical_smiles=canonical_smiles,
        formula=formula,
        source="rdkit_mmff_approx",
        caption=APPROXIMATE_GEOMETRY_CAPTION,
        atoms=atoms,
        bonds=bonds,
        xyz_block=xyz_block,
        bond_lengths=lengths,
        bond_angles=angles,
    )

