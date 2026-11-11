"""SQLite persistence layer for Q-Med experiments (schema initialization for Week 1)."""

from pathlib import Path
import sqlite3
from typing import List, Union

from qmed.config import DEFAULT_DB_PATH

EXPERIMENTS_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_identifier TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    config_json TEXT NOT NULL,
    descriptors_json TEXT,
    qee_decision TEXT NOT NULL,
    qee_explanation_json TEXT NOT NULL,
    basis TEXT,
    active_space TEXT,
    mapping TEXT,
    ansatz TEXT,
    optimizer TEXT,
    backend TEXT,
    shots INTEGER,
    seed INTEGER,
    noise_model TEXT,
    vqe_energy REAL,
    hf_energy REAL,
    fci_energy REAL,
    error_ha REAL,
    uncertainty_ha REAL,
    iterations INTEGER,
    converged INTEGER,
    runtime_sec REAL,
    qubits INTEGER,
    depth INTEGER,
    gate_count INTEGER,
    reliability TEXT,
    verdict TEXT
);
"""


def get_connection(db_path: Union[Path, str] = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row factory set to sqlite3.Row."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Union[Path, str] = DEFAULT_DB_PATH) -> Path:
    """Create the SQLite database file and `experiments` table if they do not exist."""
    resolved_path = Path(db_path)
    if str(resolved_path) != ":memory:":
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(resolved_path) as conn:
        conn.executescript(EXPERIMENTS_SCHEMA_SQL)
        conn.commit()

    return resolved_path


def get_table_columns(
    table_name: str = "experiments", db_path: Union[Path, str] = DEFAULT_DB_PATH
) -> List[str]:
    """Return column names for a given table (useful for schema verification tests)."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(f"PRAGMA table_info({table_name});")
        return [str(row["name"]) for row in cursor.fetchall()]
