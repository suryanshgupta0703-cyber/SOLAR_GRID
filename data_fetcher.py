"""
SolarGrid — data_fetcher.py
============================
Real-data layer for the SolarGrid dashboard.

Replaces synthetic weather data with live readings from the Open-Meteo API
(completely free, no API key required).

Public API
----------
  fetch_live_forecast(lat, lon, city_name, target_date)  → pd.DataFrame | None
  fetch_historical_archive(lat, lon, start_date, end_date) → pd.DataFrame | None
  CITY_PRESETS  — dict of preset city coordinates

Dependencies
------------
  Only stdlib (urllib.request, json, math, datetime) + numpy + pandas.
  No `requests` library needed.
"""

import json
import math
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta, timezone
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ---------------------------------------------------------------------------
# User-Agent header (courteous to API servers)
# ---------------------------------------------------------------------------
USER_AGENT = "SolarGrid/1.0 (Academic Project)"

# ---------------------------------------------------------------------------
# Network timeout (seconds)
# ---------------------------------------------------------------------------
_TIMEOUT = 15

# ---------------------------------------------------------------------------
# Solar physics constants (mirrors data_gen.py)
# ---------------------------------------------------------------------------
_P_RATED_KW  = 50.0   # kW — peak rated capacity of the community array
_PANEL_EFF   = 0.95   # overall system de-rating factor (wiring, inverter, etc.)
_TEMP_COEFF  = 0.004  # fractional power loss per °C above 25 °C (STC)
_PEAK_DEMAND = 45.0   # kW — community peak demand used in the demand model

# ---------------------------------------------------------------------------
# Preset cities
# ---------------------------------------------------------------------------
CITY_PRESETS: dict[str, dict[str, float]] = {
    "Jammu":     {"lat": 32.7266, "lon": 74.8570},
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Jaipur":    {"lat": 26.9124, "lon": 75.7873},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
}


# ===========================================================================
# Internal helpers
# ===========================================================================

def _get_json(url: str) -> dict | None:
    """
    Perform an HTTP GET and return the parsed JSON body.

    Uses stdlib urllib so no external `requests` package is needed.
    Sets a custom User-Agent and enforces a 15-second timeout.

    Args:
        url: Fully-formed URL to fetch.

    Returns:
        Parsed JSON dict, or None if any network/HTTP error occurs.
    """
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8"))
    except HTTPError as exc:
        print(f"[data_fetcher] ⚠ HTTP {exc.code} fetching {url}: {exc.reason}")
    except URLError as exc:
        print(f"[data_fetcher] ⚠ Network error fetching {url}: {exc.reason}")
    except Exception as exc:  # noqa: BLE001
        print(f"[data_fetcher] ⚠ Unexpected error fetching {url}: {exc}")
    return None


def _add_cyclical_encodings(df: pd.DataFrame, dt_series: pd.Series) -> pd.DataFrame:
    """
    Add circular time encodings to *df* in-place using timestamps in *dt_series*.

    Columns added:
        hour_sin, hour_cos  — hour-of-day cyclical encoding
        dow_sin,  dow_cos   — day-of-week cyclical encoding
        day_of_year         — integer 1–366

    Args:
        df:        DataFrame to augment.
        dt_series: pandas Series of datetime objects (one per row).

    Returns:
        The same DataFrame with new columns appended.
    """
    hours      = dt_series.dt.hour.astype(float)
    dow        = dt_series.dt.dayofweek.astype(float)
    doy        = dt_series.dt.dayofyear.astype(float)

    df["hour_sin"]    = np.sin(2 * math.pi * hours / 24)
    df["hour_cos"]    = np.cos(2 * math.pi * hours / 24)
    df["dow_sin"]     = np.sin(2 * math.pi * dow   /  7)
    df["dow_cos"]     = np.cos(2 * math.pi * dow   /  7)
    df["day_of_year"] = doy
    return df


def _solar_gen_from_gti(gti: np.ndarray, temp_c: np.ndarray) -> np.ndarray:
    """
    Estimate solar generation (kW) from Global Tilted Irradiance.

    Physics formula (same as brief specification):
        P = P_rated × (GTI / 1000) × [1 − 0.004 × (T − 25)] × 0.95

    Args:
        gti:    Array of GTI values in W/m².
        temp_c: Array of ambient temperature in °C.

    Returns:
        Array of estimated solar generation in kW, clipped to [0, P_rated].
    """
    temp_factor = 1.0 - _TEMP_COEFF * (temp_c - 25.0)
    solar_kw    = _P_RATED_KW * (gti / 1000.0) * temp_factor * _PANEL_EFF
    return np.clip(solar_kw, 0.0, _P_RATED_KW)


def _baseline_demand(hours: np.ndarray) -> np.ndarray:
    """
    Estimate residential baseline demand (kW) using the dual-peak profile
    from data_gen.py.

    Peaks at 08:00 (morning) and 19:30 (evening).

    Args:
        hours: Array of hour-of-day values (0–23) as floats.

    Returns:
        Array of demand values in kW.
    """
    morning_peak = 0.7 * np.exp(-0.5 * ((hours - 8.0)  / 1.5) ** 2)
    evening_peak = 1.0 * np.exp(-0.5 * ((hours - 19.5) / 1.8) ** 2)
    base_load    = 0.2

    profile = base_load + morning_peak + evening_peak
    # Normalise so the max maps to PEAK_DEMAND_KW (deterministic version)
    demand  = _PEAK_DEMAND * (profile / profile.max())
    return np.clip(demand, 5.0, _PEAK_DEMAND * 1.1)


# ===========================================================================
# Public function 1 — Live 24-hour forecast
# ===========================================================================

def fetch_live_forecast(
    lat: float = 32.7266,
    lon: float = 74.8570,
    city_name: str = "Jammu",
    target_date: date | None = None,
) -> pd.DataFrame | None:
    """
    Fetch one full local calendar day (00:00–23:00) of hourly weather from the
    Open-Meteo Forecast API.

    Returns a DataFrame whose columns match the inference feature schema
    expected by ``models/forecast.py``:
        temperature_c, cloud_cover_pct, solar_irradiance_wm2,
        hour_sin, hour_cos, dow_sin, dow_cos, day_of_year

    Args:
        lat:       Latitude of the location (decimal degrees).
        lon:       Longitude of the location (decimal degrees).
        city_name: Human-readable label used only for log messages.
        target_date: Calendar day to fetch (defaults to today). Open-Meteo
                     serves roughly the past 3 months up to 16 days ahead.

    Returns:
        24-row DataFrame (row index == hour of day) with the feature columns
        described above, or **None** if the API call fails.

    Notes:
        - ``global_tilted_irradiance`` → ``solar_irradiance_wm2``
        - ``cloud_cover`` / 100       → ``cloud_cover_pct``  (kept as %)
        - Timestamps are in the city's local timezone (``timezone=auto``).
    """
    day = (target_date or date.today()).isoformat()
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,cloud_cover,direct_normal_irradiance,global_tilted_irradiance"
        f"&start_date={day}&end_date={day}&timezone=auto"
    )

    print(f"[data_fetcher] Fetching live forecast for {city_name} on {day} (lat={lat}, lon={lon})…")
    data = _get_json(url)
    if data is None:
        print(f"[data_fetcher] ⚠ Could not retrieve live forecast for {city_name}. "
              "Returning None — caller should use synthetic fallback.")
        return None

    try:
        hourly     = data["hourly"]
        times      = pd.to_datetime(hourly["time"])
        temp_c     = np.array(hourly["temperature_2m"],            dtype=float)
        cloud_raw  = np.array(hourly["cloud_cover"],               dtype=float)
        gti        = np.array(hourly["global_tilted_irradiance"],  dtype=float)

        # Build base DataFrame
        df = pd.DataFrame({
            "_time":               times,
            "temperature_c":       np.round(temp_c,              2),
            "cloud_cover_pct":     np.round(cloud_raw,           2),   # already in %
            "solar_irradiance_wm2":np.round(np.clip(gti, 0, None), 2),
        })

        # Interpolate (not drop) occasional nulls so row index stays aligned to hour of day
        value_cols = ["temperature_c", "cloud_cover_pct", "solar_irradiance_wm2"]
        df[value_cols] = df[value_cols].interpolate(limit_direction="both")

        if len(df) != 24 or df[value_cols].isna().any().any():
            print(f"[data_fetcher] ⚠ Incomplete day ({len(df)} hours) in API response for {city_name}.")
            return None

        # Add cyclical encodings
        df = _add_cyclical_encodings(df, df["_time"])

        # Drop internal helper column
        df.drop(columns=["_time"], inplace=True)

        # Reorder to match expected feature schema
        feature_cols = [
            "temperature_c", "cloud_cover_pct", "solar_irradiance_wm2",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "day_of_year",
        ]
        df = df[feature_cols].reset_index(drop=True)

        print(f"[data_fetcher] ✓ Live forecast fetched: {len(df)} hours for {city_name}.")
        return df

    except (KeyError, ValueError, TypeError) as exc:
        print(f"[data_fetcher] ⚠ Failed to parse forecast response: {exc}")
        return None


# ===========================================================================
# Public function 2 — Historical archive
# ===========================================================================

def fetch_historical_archive(
    lat: float = 32.7266,
    lon: float = 74.8570,
    start_date: str = "2024-01-01",
    end_date: str   = "2024-01-30",
) -> pd.DataFrame | None:
    """
    Fetch historical hourly weather from the Open-Meteo Archive API.

    Returns a DataFrame matching the ``historical_energy`` DB table schema,
    with two additional physics-derived columns:
        timestamp, temperature_c, cloud_cover_pct, solar_irradiance_wm2,
        historical_solar_gen_kw, historical_baseline_demand_kw

    Solar generation uses the same physics formula as the brief:
        P = P_rated × (GTI / 1000) × [1 − 0.004 × (T − 25)] × 0.95
        with P_rated = 50 kW.

    Demand uses the same dual-peak Gaussian model from ``data_gen.py``.

    Args:
        lat:        Latitude (decimal degrees).
        lon:        Longitude (decimal degrees).
        start_date: ISO 8601 date string, e.g. ``"2024-01-01"``.
        end_date:   ISO 8601 date string, e.g. ``"2024-01-30"`` (inclusive).

    Returns:
        DataFrame described above, or **None** on any API or parse failure.
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,cloud_cover,direct_normal_irradiance,global_tilted_irradiance"
        f"&timezone=auto"
    )

    print(f"[data_fetcher] Fetching historical archive {start_date} → {end_date} "
          f"(lat={lat}, lon={lon})…")
    data = _get_json(url)
    if data is None:
        print("[data_fetcher] ⚠ Could not retrieve historical archive. "
              "Returning None — caller should use synthetic fallback.")
        return None

    try:
        hourly    = data["hourly"]
        times     = pd.to_datetime(hourly["time"])
        temp_c    = np.array(hourly["temperature_2m"],           dtype=float)
        cloud_raw = np.array(hourly["cloud_cover"],              dtype=float)
        gti       = np.array(hourly["global_tilted_irradiance"], dtype=float)

        # Replace NaN GTI/temp with 0 so physics formula stays well-defined
        gti    = np.nan_to_num(gti,    nan=0.0)
        temp_c = np.nan_to_num(temp_c, nan=25.0)

        # Derived physics columns
        solar_gen_kw  = _solar_gen_from_gti(gti, temp_c)
        hours_arr     = times.hour.to_numpy(dtype=float)
        demand_kw     = _baseline_demand(hours_arr)

        df = pd.DataFrame({
            "timestamp":                     times.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature_c":                 np.round(temp_c,                     2),
            "cloud_cover_pct":               np.round(np.nan_to_num(cloud_raw),   2),
            "solar_irradiance_wm2":          np.round(np.clip(gti, 0, None),      2),
            "historical_solar_gen_kw":       np.round(solar_gen_kw,              3),
            "historical_baseline_demand_kw": np.round(demand_kw,                 3),
        })

        print(f"[data_fetcher] ✓ Historical archive fetched: {len(df)} rows "
              f"({start_date} → {end_date}).")
        return df

    except (KeyError, ValueError, TypeError) as exc:
        print(f"[data_fetcher] ⚠ Failed to parse archive response: {exc}")
        return None


# ===========================================================================
# Quick smoke-test (run as script)
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SolarGrid — data_fetcher.py  (smoke test)")
    print("=" * 60)

    print("\n--- Live forecast (Jammu) ---")
    df_live = fetch_live_forecast()
    if df_live is not None:
        print(df_live.to_string(index=True))
    else:
        print("Live forecast returned None.")

    print("\n--- Historical archive (Jammu, Jan 2024) ---")
    df_hist = fetch_historical_archive(end_date="2024-01-07")
    if df_hist is not None:
        print(df_hist.head(5).to_string(index=False))
        print(f"…total rows: {len(df_hist)}")
    else:
        print("Historical archive returned None.")
