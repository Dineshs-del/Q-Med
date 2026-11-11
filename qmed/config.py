"""Central configuration and thresholds for Q-Med.

All numeric thresholds, supported chemical elements, and file paths live here
to avoid magic numbers inside module logic.
"""

from pathlib import Path
from typing import FrozenSet

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
MAX_CIRCUIT_DEPTH: int = 500
MIN_SEEDS_FOR_VERDICT: int = 3

# Reliability thresholds in Hartree (1 mHa = 1e-3 Ha; chemical accuracy ~ 1.6 mHa)
HIGH_RELIABILITY_ERROR_HA: float = 0.0016
HIGH_RELIABILITY_UNCERTAINTY_HA: float = 0.0016
MEDIUM_RELIABILITY_ERROR_HA: float = 0.0100
LOW_RELIABILITY_ERROR_HA: float = 0.0500
