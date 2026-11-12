"""Central configuration and thresholds for Q-Med.

All numeric thresholds, supported chemical elements, and file paths live here
to avoid magic numbers inside module logic.
"""

from pathlib import Path
from typing import FrozenSet, Tuple

# Repository paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
BENCHMARKS_PATH: Path = DATA_DIR / "benchmarks.json"
DEFAULT_DB_PATH: Path = PROJECT_ROOT / "qmed_experiments.db"

# Default quantum chemistry basis set
DEFAULT_BASIS: str = "STO-3G"

# Supported elements for STO-3G quantum simulation in Q-Med v1
SUPPORTED_ELEMENTS: FrozenSet[str] = frozenset(
    {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"}
)

# Quantum Eligibility Engine (QEE v1) thresholds
MAX_SIMULATED_QUBITS: int = 12
# Documented heuristic: decomposed UCCSD circuits exceeding 5,000 primitive gate depth
# require >1s per statevector evaluation (~45+ min for gradient-based optimization),
# triggering HYBRID routing to a reduced active space.
MAX_CIRCUIT_DEPTH: int = 5000
# Maximum full frozen-core qubits that can still be meaningfully reduced to a frontier
# active space (e.g. (4e, 4o), 8 qubits) in HYBRID mode without discarding the bulk of
# the valence shell. Molecules exceeding this limit (such as aspirin at 120 qubits) are DEFERRED.
MAX_HYBRID_FULL_QUBITS: int = 24
MIN_ACTIVE_ELECTRONS_FOR_QUANTUM: int = 2
DEFAULT_HYBRID_ACTIVE_SPACE: Tuple[int, int] = (4, 4)
MIN_SEEDS_FOR_VERDICT: int = 3
DEFAULT_BENCHMARK_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)

# Default Qiskit Aer noisy simulation parameters
DEFAULT_SHOTS: int = 2048
DEFAULT_DEPOLARIZING_1Q_ERROR: float = 0.001
DEFAULT_DEPOLARIZING_2Q_ERROR: float = 0.010
DEFAULT_READOUT_ERROR: float = 0.015

# Variational principle guard tolerance in Hartree (10 nano-Hartree)
# If E_VQE < E_CASCI(matched) - 1e-8 Ha (on ideal statevector runs), the run is marked INVALID.
VARIATIONAL_BOUND_TOLERANCE_HA: float = 1e-8

# Maximum allowed active-space truncation error (E_CASCI(reduced) - E_CASCI(frozen-core))
# in mHa before capping the reliability label below HIGH.
MAX_ACTIVE_SPACE_ERROR_FOR_HIGH_MHA: float = 1.6

# Reliability thresholds in Hartree (1 mHa = 1e-3 Ha; chemical accuracy ~ 1.6 mHa)
HIGH_RELIABILITY_ERROR_HA: float = 0.0016
HIGH_RELIABILITY_UNCERTAINTY_HA: float = 0.0016
MEDIUM_RELIABILITY_ERROR_HA: float = 0.0100
LOW_RELIABILITY_ERROR_HA: float = 0.0500
