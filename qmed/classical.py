"""Classical Hartree-Fock (HF) and exact Full CI (FCI / CASCI) baselines via PySCF.

All energies are computed live by PySCF in Hartree (Ha) and cached in-memory
during execution. No energies or runtimes are ever hard-coded.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from pyscf import fci, gto, mcscf, scf

from qmed.config import DEFAULT_BASIS
from qmed.molecules import AtomCoord, BenchmarkMolecule, get_benchmark, load_benchmarks

AtomInput = Union[str, Sequence[Union[AtomCoord, Dict[str, Any]]]]


@dataclass(frozen=True)
class ClassicalResult:
    """Container for Hartree-Fock and exact active-space FCI baseline results."""

    molecule_id: str
    basis: str
    charge: int
    spin: int
    frozen_core: bool
    n_electrons: int
    n_orbitals: int
    n_frozen_core_orbitals: int
    active_electrons: int
    active_orbitals: int
    nuclear_repulsion_energy: float
    hf_energy: float
    fci_energy: float
    correlation_energy: float
    hf_converged: bool
    fci_converged: bool
    hf_runtime_sec: float
    fci_runtime_sec: float
    total_runtime_sec: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of classical baseline results."""
        return asdict(self)


def format_atoms_for_pyscf(atoms: AtomInput) -> str:
    """Normalize atom specifications into a semicolon-separated PySCF geometry string."""
    if isinstance(atoms, str):
        cleaned = atoms.strip()
        if not cleaned:
            raise ValueError("Atom specification string cannot be empty.")
        return cleaned

    if not atoms:
        raise ValueError("Atom coordinate sequence cannot be empty.")

    lines = []
    for item in atoms:
        if isinstance(item, AtomCoord):
            symbol = item.symbol
            coords = item.coords
        elif isinstance(item, dict):
            symbol = str(item["symbol"])
            coords = [float(c) for c in item["coords"]]
        else:
            raise TypeError(f"Unsupported atom specification type: {type(item)}")

        if len(coords) != 3:
            raise ValueError(f"Atom '{symbol}' must have exactly 3 Cartesian coordinates.")
        lines.append(f"{symbol} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}")

    return "; ".join(lines)


def count_core_orbitals(mol: gto.Mole) -> int:
    """Count standard noble-gas core spatial orbitals for a PySCF molecule.

    - Z <= 2 (H, He): 0 core orbitals
    - 3 <= Z <= 10 (Li..Ne): 1 core orbital (1s^2)
    - 11 <= Z <= 18 (Na..Ar): 5 core orbitals (1s^2 2s^2 2p^6)
    """
    core_orbitals = 0
    for atom_idx in range(mol.natm):
        z = int(mol.atom_charge(atom_idx))
        if z <= 2:
            continue
        elif z <= 10:
            core_orbitals += 1
        elif z <= 18:
            core_orbitals += 5
        else:
            core_orbitals += 9
    return core_orbitals


@lru_cache(maxsize=64)
def _compute_cached_pyscf(
    atom_str: str,
    basis: str,
    charge: int,
    spin: int,
    frozen_core: bool,
    active_space: Optional[Tuple[int, int]],
    molecule_id: str,
) -> ClassicalResult:
    """Internal cached runner executing PySCF RHF/ROHF and exact FCI/CASCI."""
    t_start = time.perf_counter()

    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.build()

    n_electrons = int(mol.nelectron)
    n_orbitals = int(mol.nao_nr())
    nuc_energy = float(mol.energy_nuc())

    # 1. Hartree-Fock (RHF for closed-shell singlet, ROHF for open-shell)
    t_hf_start = time.perf_counter()
    mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    hf_energy = float(mf.kernel())
    hf_converged = bool(mf.converged)
    hf_runtime = time.perf_counter() - t_hf_start

    # 2. Determine active space (n_active_electrons, n_active_orbitals)
    if active_space is not None:
        active_electrons, active_orbitals = int(active_space[0]), int(active_space[1])
        if active_electrons <= 0 or active_electrons > n_electrons:
            raise ValueError(
                f"Invalid active_electrons={active_electrons} for n_electrons={n_electrons}."
            )
        if active_orbitals <= 0 or active_orbitals > n_orbitals:
            raise ValueError(
                f"Invalid active_orbitals={active_orbitals} for n_orbitals={n_orbitals}."
            )
        if (n_electrons - active_electrons) % 2 != 0:
            raise ValueError(
                "Number of frozen core electrons (n_electrons - active_electrons) must be even."
            )
        n_frozen_core = (n_electrons - active_electrons) // 2
    elif frozen_core:
        n_frozen_core = count_core_orbitals(mol)
        active_electrons = n_electrons - 2 * n_frozen_core
        active_orbitals = n_orbitals - n_frozen_core
    else:
        n_frozen_core = 0
        active_electrons = n_electrons
        active_orbitals = n_orbitals

    # 3. Exact diagonalization (FCI in full space or CASCI in reduced active space)
    t_fci_start = time.perf_counter()
    if active_electrons == n_electrons and active_orbitals == n_orbitals:
        fci_solver = fci.FCI(mf)
        fci_energy_val, _ = fci_solver.kernel()
        fci_energy = float(fci_energy_val)
        fci_converged = bool( getattr(fci_solver, "converged", True) )
    else:
        cas = mcscf.CASCI(mf, active_orbitals, active_electrons)
        cas_res = cas.kernel()
        fci_energy = float(cas_res[0])
        fci_converged = bool(getattr(cas.fcisolver, "converged", True))
    fci_runtime = time.perf_counter() - t_fci_start

    total_runtime = time.perf_counter() - t_start
    correlation_energy = fci_energy - hf_energy

    return ClassicalResult(
        molecule_id=molecule_id,
        basis=basis,
        charge=charge,
        spin=spin,
        frozen_core=bool(frozen_core or (n_frozen_core > 0)),
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        n_frozen_core_orbitals=n_frozen_core,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        nuclear_repulsion_energy=nuc_energy,
        hf_energy=hf_energy,
        fci_energy=fci_energy,
        correlation_energy=correlation_energy,
        hf_converged=hf_converged,
        fci_converged=fci_converged,
        hf_runtime_sec=hf_runtime,
        fci_runtime_sec=fci_runtime,
        total_runtime_sec=total_runtime,
    )


def run_classical_reference(
    atoms: AtomInput,
    basis: str = DEFAULT_BASIS,
    charge: int = 0,
    spin: int = 0,
    frozen_core: bool = False,
    active_space: Optional[Tuple[int, int]] = None,
    molecule_id: str = "custom",
) -> ClassicalResult:
    """Compute (or fetch from cache) HF and exact FCI energies in the specified active space."""
    atom_str = format_atoms_for_pyscf(atoms)
    return _compute_cached_pyscf(
        atom_str=atom_str,
        basis=basis,
        charge=int(charge),
        spin=int(spin),
        frozen_core=bool(frozen_core),
        active_space=tuple(active_space) if active_space is not None else None,
        molecule_id=molecule_id,
    )


def run_benchmark_reference(
    benchmark: Union[str, BenchmarkMolecule],
    frozen_core: bool = False,
    active_space: Optional[Tuple[int, int]] = None,
) -> ClassicalResult:
    """Run (and cache) HF and exact FCI reference calculation for a benchmark molecule."""
    bm = get_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
    return run_classical_reference(
        atoms=bm.atoms,
        basis=bm.basis,
        charge=bm.charge,
        spin=bm.spin,
        frozen_core=frozen_core,
        active_space=active_space,
        molecule_id=bm.id,
    )


def compute_all_benchmark_references(
    frozen_core: bool = False,
) -> Dict[str, ClassicalResult]:
    """Compute and cache HF and FCI reference energies for H2, LiH, and H2O."""
    benchmarks = load_benchmarks()
    results: Dict[str, ClassicalResult] = {}
    for key, bm in benchmarks.items():
        results[key] = run_benchmark_reference(bm, frozen_core=frozen_core)
    return results


def clear_classical_cache() -> None:
    """Clear the in-memory LRU cache of classical PySCF reference calculations."""
    _compute_cached_pyscf.cache_clear()
