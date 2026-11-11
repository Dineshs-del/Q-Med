"""Active-space reduction, fermion-to-qubit mapping, and qubit resource estimation."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, FreezeCoreTransformer
from rdkit import Chem

from qmed.classical import AtomInput, format_atoms_for_pyscf
from qmed.config import DEFAULT_BASIS
from qmed.molecules import BenchmarkMolecule, get_benchmark

# STO-3G spatial orbital count per element symbol (H..Ar)
STO3G_ORBITALS_BY_SYMBOL: Dict[str, int] = {
    "H": 1,
    "He": 1,
    "Li": 5,
    "Be": 5,
    "B": 5,
    "C": 5,
    "N": 5,
    "O": 5,
    "F": 5,
    "Ne": 5,
    "Na": 9,
    "Mg": 9,
    "Al": 9,
    "Si": 9,
    "P": 9,
    "S": 9,
    "Cl": 9,
    "Ar": 9,
}

# Frozen core spatial orbitals per element symbol
CORE_ORBITALS_BY_SYMBOL: Dict[str, int] = {
    "H": 0,
    "He": 0,
    "Li": 1,
    "Be": 1,
    "B": 1,
    "C": 1,
    "N": 1,
    "O": 1,
    "F": 1,
    "Ne": 1,
    "Na": 5,
    "Mg": 5,
    "Al": 5,
    "Si": 5,
    "P": 5,
    "S": 5,
    "Cl": 5,
    "Ar": 5,
}


@dataclass(frozen=True)
class ActiveSpaceSummary:
    """Summary of active space dimensions and mapped qubit counts."""

    molecule_id: str
    basis: str
    frozen_core: bool
    total_electrons: int
    total_spatial_orbitals: int
    frozen_core_orbitals: int
    active_electrons: int
    active_spatial_orbitals: int
    mapping: str
    two_qubit_reduction: bool
    num_qubits: int
    energy_offset_ha: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of active-space metadata."""
        return asdict(self)


@dataclass
class PreparedQuantumProblem:
    """Container holding the transformed ElectronicStructureProblem and qubit Hamiltonian."""

    problem: ElectronicStructureProblem
    mapper: QubitMapper
    qubit_op: SparsePauliOp
    summary: ActiveSpaceSummary


def choose_default_mapping(molecule_id: Optional[str] = None) -> str:
    """Select default fermion-to-qubit mapping per Q-Med spec.

    Uses Parity mapping with two-qubit reduction for H2 and LiH;
    uses Jordan-Wigner mapping otherwise.
    """
    if molecule_id and molecule_id.strip().upper() in {"H2", "HYDROGEN", "LIH", "LITHIUM HYDRIDE"}:
        return "parity"
    return "jordan_wigner"


def build_qubit_mapper(
    mapping: str,
    num_particles: Optional[Tuple[int, int]] = None,
) -> Tuple[QubitMapper, str, bool]:
    """Instantiate a Qiskit Nature QubitMapper and report whether 2-qubit reduction is active."""
    normalized = mapping.strip().lower().replace("-", "_")
    if normalized in {"parity", "parity_2q", "parity_2q_reduced"}:
        mapper = ParityMapper(num_particles=num_particles)
        two_q_reduced = num_particles is not None
        canonical_name = "parity_2q_reduced" if two_q_reduced else "parity"
        return mapper, canonical_name, two_q_reduced
    if normalized in {"jordan_wigner", "jw"}:
        return JordanWignerMapper(), "jordan_wigner", False
    raise ValueError(
        f"Unsupported fermion-to-qubit mapping '{mapping}'. Use 'parity' or 'jordan_wigner'."
    )


def prepare_active_space_problem(
    atoms: AtomInput,
    basis: str = DEFAULT_BASIS,
    charge: int = 0,
    spin: int = 0,
    frozen_core: bool = True,
    active_space: Optional[Tuple[int, int]] = None,
    mapping: Optional[str] = None,
    molecule_id: str = "custom",
) -> PreparedQuantumProblem:
    """Build an ElectronicStructureProblem from PySCF, apply active-space reduction, and map to qubits."""
    atom_str = format_atoms_for_pyscf(atoms)
    driver = PySCFDriver(
        atom=atom_str,
        basis=basis,
        charge=int(charge),
        spin=int(spin),
    )
    full_problem = driver.run()

    total_alpha, total_beta = full_problem.num_particles
    total_electrons = int(total_alpha + total_beta)
    total_spatial_orbitals = int(full_problem.num_spatial_orbitals)

    # Apply explicit ActiveSpaceTransformer or FreezeCoreTransformer
    if active_space is not None:
        act_elec, act_orb = int(active_space[0]), int(active_space[1])
        transformer = ActiveSpaceTransformer(
            num_electrons=act_elec,
            num_spatial_orbitals=act_orb,
        )
        problem = transformer.transform(full_problem)
        frozen_core_orbitals = (total_electrons - act_elec) // 2
        is_frozen = frozen_core_orbitals > 0 or act_orb < total_spatial_orbitals
    elif frozen_core:
        fc_transformer = FreezeCoreTransformer(freeze_core=True)
        problem = fc_transformer.transform(full_problem)
        act_alpha, act_beta = problem.num_particles
        act_elec = int(act_alpha + act_beta)
        act_orb = int(problem.num_spatial_orbitals)
        frozen_core_orbitals = (total_electrons - act_elec) // 2
        is_frozen = frozen_core_orbitals > 0
    else:
        problem = full_problem
        act_elec = total_electrons
        act_orb = total_spatial_orbitals
        frozen_core_orbitals = 0
        is_frozen = False

    selected_mapping = mapping if mapping is not None else choose_default_mapping(molecule_id)
    mapper, canonical_mapping, two_q_reduced = build_qubit_mapper(
        selected_mapping,
        num_particles=problem.num_particles,
    )

    second_q_op = problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(second_q_op)

    constants_sum = sum(float(v.real) for v in problem.hamiltonian.constants.values())

    summary = ActiveSpaceSummary(
        molecule_id=molecule_id,
        basis=basis,
        frozen_core=is_frozen,
        total_electrons=total_electrons,
        total_spatial_orbitals=total_spatial_orbitals,
        frozen_core_orbitals=frozen_core_orbitals,
        active_electrons=act_elec,
        active_spatial_orbitals=act_orb,
        mapping=canonical_mapping,
        two_qubit_reduction=two_q_reduced,
        num_qubits=int(qubit_op.num_qubits),
        energy_offset_ha=constants_sum,
    )

    return PreparedQuantumProblem(
        problem=problem,
        mapper=mapper,
        qubit_op=qubit_op,
        summary=summary,
    )


def prepare_benchmark_problem(
    benchmark: Union[str, BenchmarkMolecule],
    frozen_core: bool = True,
    active_space: Optional[Tuple[int, int]] = None,
    mapping: Optional[str] = None,
) -> PreparedQuantumProblem:
    """Build and map the active-space quantum problem for a named benchmark molecule."""
    bm = get_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
    return prepare_active_space_problem(
        atoms=bm.atoms,
        basis=bm.basis,
        charge=bm.charge,
        spin=bm.spin,
        frozen_core=frozen_core,
        active_space=active_space,
        mapping=mapping,
        molecule_id=bm.id,
    )


def estimate_qubits_from_smiles(
    smiles: str,
    frozen_core: bool = True,
    mapping: str = "jordan_wigner",
) -> ActiveSpaceSummary:
    """Analytical STO-3G active-space and qubit estimate from a validated SMILES string."""
    mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
    if mol is None:
        raise ValueError(f"Cannot estimate active space for invalid SMILES: '{smiles}'")
    mol_h = Chem.AddHs(mol)

    total_electrons = 0
    total_orbitals = 0
    frozen_orbitals = 0

    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        z = atom.GetAtomicNum()
        total_electrons += z - atom.GetFormalCharge()
        total_orbitals += STO3G_ORBITALS_BY_SYMBOL.get(sym, 9)
        if frozen_core:
            frozen_orbitals += CORE_ORBITALS_BY_SYMBOL.get(sym, 5)

    active_electrons = total_electrons - 2 * frozen_orbitals
    active_orbitals = total_orbitals - frozen_orbitals
    two_q = mapping.strip().lower().startswith("parity")
    num_qubits = max(2, 2 * active_orbitals - (2 if two_q else 0))

    return ActiveSpaceSummary(
        molecule_id=smiles.strip(),
        basis=DEFAULT_BASIS,
        frozen_core=frozen_core and (frozen_orbitals > 0),
        total_electrons=total_electrons,
        total_spatial_orbitals=total_orbitals,
        frozen_core_orbitals=frozen_orbitals,
        active_electrons=active_electrons,
        active_spatial_orbitals=active_orbitals,
        mapping="parity_2q_reduced" if two_q else "jordan_wigner",
        two_qubit_reduction=two_q,
        num_qubits=num_qubits,
        energy_offset_ha=0.0,
    )
