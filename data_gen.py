"""
SolarGrid — data_gen.py
========================
Generates 30 days of realistic synthetic hourly energy data and persists it
to the SQLite database.

Physics-based heuristics used:
  - Solar irradiance follows a clipped sinusoidal curve peaking at noon.
  - Cloud cover randomly attenuates irradiance (Beer-Lambert approximation).
  - Temperature follows a daily cycle with added Gaussian noise.
  - Solar generation is a noisy linear function of effective irradiance.
  - Baseline demand peaks in the morning (7–9 AM) and evening (6–9 PM),
    representing typical residential load profiles.

Run once before starting the Streamlit app:
    python data_gen.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Ensure the project root is importable regardless of how the script is called
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db_manager import get_connection, init_db, table_has_data

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------
RANDOM_SEED      = 42
N_DAYS           = 30          # days of history to generate
HOURS_PER_DAY    = 24
TOTAL_HOURS      = N_DAYS * HOURS_PER_DAY

# Solar panel system specs (representative residential community)
PANEL_CAPACITY_KW    = 50.0    # kW peak capacity of entire community array
PANEL_EFFICIENCY     = 0.18    # 18% conversion efficiency
PANEL_AREA_M2        = 300.0   # total panel area in m²

# Community baseline demand scale
PEAK_DEMAND_KW       = 45.0    # realistic peak kW for ~20-home community

# Start date for the synthetic dataset
START_DATE = datetime(2024, 1, 1, 0, 0, 0)


def _solar_irradiance(hour: np.ndarray, cloud_cover: np.ndarray) -> np.ndarray:
    """
    Compute solar irradiance (W/m²) using a sinusoidal day-light model.

    The sun is above the horizon roughly from hour 6 to hour 18.
    We model peak irradiance at solar noon (hour 12) and attenuate
    by cloud cover using a transmittance factor.

    Args:
        hour:        Array of hour-of-day values (0–23).
        cloud_cover: Array of fractional cloud cover values (0–1).

    Returns:
        Array of irradiance values clipped to [0, 1100] W/m².
    """
    # Sunrise at 6, sunset at 18 → 12-hour daylight window
    # Shift hour so that noon maps to the peak of a half-sine
    daylight_fraction = np.clip((hour - 6) / 12.0, 0, 1)
    base_irradiance   = 1000.0 * np.sin(np.pi * daylight_fraction)

    # Cloud transmittance: clear sky (0 cloud) → 1.0, overcast → ~0.1
    transmittance = 1.0 - 0.9 * cloud_cover

    return np.clip(base_irradiance * transmittance, 0, 1100)


def _temperature(hour: np.ndarray, day_of_year: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Simulate ambient temperature with daily and seasonal variation.

    Model:
        T = T_base + T_seasonal * sin(DOY) + T_daily * sin(pi * h/24) + noise

    Args:
        hour:        Hour-of-day (0–23).
        day_of_year: Day-of-year (1–365).
        rng:         Seeded random number generator.

    Returns:
        Temperature array in °C.
    """
    # Seasonal variation: warmer mid-year (summer for northern hemisphere)
    seasonal = 10.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Daily cycle: coldest at 4 AM, warmest at 3 PM
    daily = 5.0 * np.sin(np.pi * (hour - 4) / 12.0)

    noise = rng.normal(0, 1.5, size=len(hour))

    return 20.0 + seasonal + daily + noise


def _baseline_demand(hour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Synthesise residential electricity demand using a two-peak profile.

    Peaks:
        Morning: 7–9 AM  (appliances, breakfast, EV departure charging)
        Evening: 18–21   (cooking, HVAC, entertainment)
    Overnight: low but non-zero base load (fridges, standby devices)

    Args:
        hour: Array of hour-of-day values (0–23).
        rng:  Seeded random number generator.

    Returns:
        Array of demand values in kW.
    """
    # Vectorised piece-wise demand shape
    morning_peak = 0.7 * np.exp(-0.5 * ((hour - 8.0) / 1.5) ** 2)
    evening_peak = 1.0 * np.exp(-0.5 * ((hour - 19.5) / 1.8) ** 2)
    base_load    = 0.2  # fraction of PEAK_DEMAND_KW always consumed

    profile = base_load + morning_peak + evening_peak
    noise   = rng.normal(0, 0.05, size=len(hour))

    # Normalise so the max is ~PEAK_DEMAND_KW and add jitter
    demand  = PEAK_DEMAND_KW * (profile / profile.max()) + noise * PEAK_DEMAND_KW * 0.05
    return np.clip(demand, 5.0, PEAK_DEMAND_KW * 1.1)


def generate_data() -> pd.DataFrame:
    """
    Build the complete 30-day hourly synthetic dataset.

    Returns:
        pd.DataFrame with columns matching the historical_energy DB schema.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # Build timestamp series
    timestamps = [START_DATE + timedelta(hours=i) for i in range(TOTAL_HOURS)]
    ts_array   = np.array(timestamps)

    hours        = np.array([t.hour              for t in timestamps], dtype=float)
    day_of_year  = np.array([t.timetuple().tm_yday for t in timestamps], dtype=float)

    # Cloud cover: slow-varying Gaussian random walk clamped to [0, 1]
    cloud_steps  = rng.normal(0, 0.05, size=TOTAL_HOURS)
    cloud_cover  = np.cumsum(cloud_steps)
    cloud_cover  = (cloud_cover - cloud_cover.min()) / (cloud_cover.max() - cloud_cover.min())
    # Re-inject randomness so we don't get monotone cloud cover
    cloud_cover  = np.clip(cloud_cover + rng.uniform(-0.2, 0.2, TOTAL_HOURS), 0, 1)

    irradiance   = _solar_irradiance(hours, cloud_cover)
    temperature  = _temperature(hours, day_of_year, rng)

    # Solar generation: physically proportional to effective irradiance
    solar_gen    = (irradiance * PANEL_EFFICIENCY * PANEL_AREA_M2) / 1000.0   # → kW
    # Add realistic sensor noise
    solar_gen   += rng.normal(0, 0.5, TOTAL_HOURS)
    solar_gen    = np.clip(solar_gen, 0, PANEL_CAPACITY_KW)

    demand       = _baseline_demand(hours, rng)

    df = pd.DataFrame({
        "timestamp":                     [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "temperature_c":                 np.round(temperature, 2),
        "cloud_cover_pct":               np.round(cloud_cover * 100, 2),
        "solar_irradiance_wm2":          np.round(irradiance, 2),
        "historical_solar_gen_kw":       np.round(solar_gen, 3),
        "historical_baseline_demand_kw": np.round(demand, 3),
    })

    return df


def save_to_db(df: pd.DataFrame) -> None:
    """
    Persist the generated DataFrame to the historical_energy SQLite table.

    Skips insertion if the table already contains data to prevent duplicates.

    Args:
        df: DataFrame produced by generate_data().
    """
    # Ensure the schema exists before inserting
    init_db()

    if table_has_data("historical_energy"):
        print("[data_gen] historical_energy table already populated — skipping insert.")
        print("           Delete solargrid.db to regenerate data.")
        return

    conn = get_connection()
    df.to_sql(
        name="historical_energy",
        con=conn,
        if_exists="append",   # append to existing (empty) table
        index=False,
    )
    conn.commit()
    conn.close()
    print(f"[data_gen] Inserted {len(df)} rows into historical_energy.")


if __name__ == "__main__":
    print("SolarGrid — Mock Data Generator")
    print("=" * 40)
    print(f"Generating {N_DAYS} days of hourly data ({TOTAL_HOURS} rows)...")
    df = generate_data()
    print(df.head())
    print(f"\nSolar range : {df['historical_solar_gen_kw'].min():.2f} – "
          f"{df['historical_solar_gen_kw'].max():.2f} kW")
    print(f"Demand range: {df['historical_baseline_demand_kw'].min():.2f} – "
          f"{df['historical_baseline_demand_kw'].max():.2f} kW")
    save_to_db(df)
    print("\n[data_gen] Done! You can now run the model training step.")
