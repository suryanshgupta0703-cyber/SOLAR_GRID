"""
SolarGrid — db_manager.py
==========================
Handles all SQLite database operations:
  - Establishing a connection to the local .db file
  - Initialising the schema (creates tables if they don't exist)
  - Providing helper functions used by data_gen.py and app.py

Design decision: We use a *file-based* SQLite database (solargrid.db) so data
persists between Streamlit reruns without needing a separate server process.
"""

import sqlite3
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve DB path relative to this file so the app works from any cwd
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent          # .../solargrid/database/
DB_PATH = _HERE.parent / "solargrid.db"  # .../solargrid/solargrid.db


def get_connection() -> sqlite3.Connection:
    """
    Open (or create) the SQLite database file and return a connection.

    We enable WAL (Write-Ahead Logging) mode for better concurrent read
    performance — important when Streamlit auto-reruns while data_gen writes.

    Returns:
        sqlite3.Connection: Active connection with row_factory set to
        sqlite3.Row so columns are accessible by name as well as index.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row       # enables dict-like row access
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """
    Create all required tables if they do not already exist.

    Tables
    ------
    historical_energy
        Stores 30 days of hourly mock sensor/weather readings plus derived
        solar-generation and baseline-demand values.

    forecasts  (optional, for future use)
        Would store model-generated 24-hour forecasts so they can be
        retrieved without re-running inference on every page load.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ------------------------------------------------------------------
    # Table: historical_energy
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_energy (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp                   TEXT    NOT NULL UNIQUE,
            temperature_c               REAL    NOT NULL,   -- ambient temp (°C)
            cloud_cover_pct             REAL    NOT NULL,   -- 0–100 %
            solar_irradiance_wm2        REAL    NOT NULL,   -- W/m²
            historical_solar_gen_kw     REAL    NOT NULL,   -- kW produced
            historical_baseline_demand_kw REAL  NOT NULL    -- kW consumed
        );
    """)

    # ------------------------------------------------------------------
    # Table: forecasts  (stored results of the ML models)
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at          TEXT    NOT NULL,
            forecast_hour       INTEGER NOT NULL,   -- 0–23
            predicted_solar_kw  REAL    NOT NULL,
            predicted_demand_kw REAL    NOT NULL
        );
    """)

    conn.commit()
    conn.close()
    print(f"[db_manager] Database initialised at: {DB_PATH}")


def table_has_data(table_name: str) -> bool:
    """
    Quick check to see if a table already contains rows.

    Used by data_gen.py to avoid re-inserting duplicate data when the
    script is accidentally run twice.

    Args:
        table_name: Name of the SQLite table to check.

    Returns:
        True if at least one row exists, False otherwise.
    """
    conn = get_connection()
    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    conn.close()
    return count > 0


if __name__ == "__main__":
    # Allow this module to be run directly to set up the database.
    init_db()
