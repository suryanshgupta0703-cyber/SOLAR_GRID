"""
SolarGrid — models/forecast.py
================================
Inference layer — generates a 24-hour hourly forecast of solar generation
and electricity demand using the trained LSTM and XGBoost models.

Strategy:
  • Solar prediction is an **ensemble**: LSTM (0.55 weight) + XGBoost (0.45).
    The LSTM captures temporal carry-over (e.g., a sunny morning predicts a
    sunny afternoon). XGBoost captures weather-feature nonlinearities.
  • Demand prediction uses **XGBoost only** (demand is less temporally
    autocorrelated in a residential context and more feature-driven).

Graceful fallback:
  If saved model files are missing (models haven't been trained yet), the
  module returns a physics-based synthetic forecast so the Streamlit UI
  still renders without crashing.

Usage (from project root):
    from models.forecast import get_24hr_forecast
    solar, demand = get_24hr_forecast()
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import sys
from pathlib import Path
from datetime import datetime

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
MODELS_DIR       = Path(__file__).parent / "saved"
LSTM_PATH        = MODELS_DIR / "lstm_solar.pt"
XGB_SOLAR_JSON   = MODELS_DIR / "xgb_solar.json"
XGB_DEMAND_JSON  = MODELS_DIR / "xgb_demand.json"
XGB_SOLAR_JOBLIB = MODELS_DIR / "xgb_solar.joblib"
XGB_DEMAND_JOBLIB= MODELS_DIR / "xgb_demand.joblib"
SCALER_FEAT      = MODELS_DIR / "scaler_features.pkl"
SCALER_SOL       = MODELS_DIR / "scaler_solar.pkl"
SCALER_DEM       = MODELS_DIR / "scaler_demand.pkl"
FEAT_COLS_F      = MODELS_DIR / "feature_cols.json"

# Ensemble blend weight for LSTM vs XGBoost (must sum to 1.0)
LSTM_WEIGHT = 0.55
XGB_WEIGHT  = 1 - LSTM_WEIGHT

# ---------------------------------------------------------------------------
# Lazy-loaded model cache (avoid reloading on every Streamlit rerun)
# ---------------------------------------------------------------------------
_cache: dict = {}


# ===========================================================================
# Helper: Check if all model artefacts exist
# ===========================================================================

def models_are_trained() -> bool:
    """Return True only if all saved model files exist on disk."""
    solar_exists = XGB_SOLAR_JOBLIB.exists() or XGB_SOLAR_JSON.exists()
    demand_exists = XGB_DEMAND_JOBLIB.exists() or XGB_DEMAND_JSON.exists()
    required = [LSTM_PATH, SCALER_FEAT, SCALER_SOL, SCALER_DEM]
    return solar_exists and demand_exists and all(p.exists() for p in required)


# ===========================================================================
# Feature Engineering for Inference
# ===========================================================================

def _build_inference_features(target_date: datetime | None = None) -> pd.DataFrame:
    """
    Construct a 24-row DataFrame of weather features for the next 24 hours.

    In a production system this would call a real weather API.  Here we use
    the same physics-based model as data_gen.py but for "today".

    Args:
        target_date: Starting datetime for the 24-hour window. Defaults to
                     today at midnight (local time).

    Returns:
        DataFrame with the same feature columns used during training.
    """
    if target_date is None:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        today = target_date

    hours = np.arange(24, dtype=float)

    # --- Synthetic weather for inference day ---
    # Cloud cover: mildly random, seeded by calendar day for reproducibility
    day_seed = today.timetuple().tm_yday
    rng      = np.random.default_rng(day_seed)

    cloud_cover = rng.uniform(0.1, 0.6, 24)      # 10–60 % cloud cover

    # Solar irradiance from sinusoidal model
    daylight_fraction = np.clip((hours - 6) / 12.0, 0, 1)
    transmittance     = 1.0 - 0.9 * cloud_cover
    irradiance        = np.clip(1000.0 * np.sin(np.pi * daylight_fraction) * transmittance, 0, 1100)

    # Temperature: daily cycle
    seasonal  = 10.0 * np.sin(2 * np.pi * (today.timetuple().tm_yday - 80) / 365)
    daily     = 5.0  * np.sin(np.pi * (hours - 4) / 12.0)
    temp      = 20.0 + seasonal + daily + rng.normal(0, 0.5, 24)

    # Circular time encodings
    hour_sin  = np.sin(2 * np.pi * hours / 24)
    hour_cos  = np.cos(2 * np.pi * hours / 24)
    dow       = today.weekday()
    dow_sin   = np.full(24, np.sin(2 * np.pi * dow / 7))
    dow_cos   = np.full(24, np.cos(2 * np.pi * dow / 7))
    day_of_yr = np.full(24, today.timetuple().tm_yday, dtype=float)

    df = pd.DataFrame({
        "temperature_c":       np.round(temp,        2),
        "cloud_cover_pct":     np.round(cloud_cover * 100, 2),
        "solar_irradiance_wm2":np.round(irradiance,  2),
        "hour_sin":            hour_sin,
        "hour_cos":            hour_cos,
        "dow_sin":             dow_sin,
        "dow_cos":             dow_cos,
        "day_of_year":         day_of_yr,
    })

    return df


# ===========================================================================
# Fallback: Physics-only forecast (no ML models needed)
# ===========================================================================

def _physics_fallback() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a plausible 24-hour forecast using only physics rules.

    Called when trained model files are absent (e.g., first run before
    `python models/train_models.py` has been executed).

    Returns:
        Tuple (solar_kw, demand_kw) — each a 24-element float array.
    """
    hours  = np.arange(24, dtype=float)
    rng    = np.random.default_rng(42)

    # Solar: sinusoidal peaking at noon
    day_frac  = np.clip((hours - 6) / 12.0, 0, 1)
    solar     = 50.0 * 0.18 * np.sin(np.pi * day_frac) * 0.85  # kW
    solar    += rng.normal(0, 0.5, 24)
    solar     = np.clip(solar, 0, None)

    # Demand: two-hump residential profile
    demand  = 45.0 * (0.2
                + 0.7 * np.exp(-0.5 * ((hours - 8.0)  / 1.5) ** 2)
                + 1.0 * np.exp(-0.5 * ((hours - 19.5) / 1.8) ** 2))
    demand += rng.normal(0, 0.5, 24)
    demand  = np.clip(demand, 5.0, 50.0)

    return solar, demand


# ===========================================================================
# Main Inference Function
# ===========================================================================

def get_24hr_forecast(target_date: datetime | None = None,
                      use_lstm: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a 24-hour hourly forecast of solar generation and demand.

    Ensemble approach:
        predicted_solar  = LSTM_WEIGHT × lstm_pred + XGB_WEIGHT × xgb_solar_pred
        predicted_demand = xgb_demand_pred

    Args:
        target_date: Forecast start date (defaults to today at midnight).
        use_lstm:    If False, skip LSTM and use only XGBoost.
                     Useful for fast iteration during development.

    Returns:
        Tuple (predicted_solar_kw, predicted_demand_kw):
            Each is a numpy array of shape (24,).
    """
    # ---- Fallback if models not trained yet ----------------------------
    if not models_are_trained():
        print("[forecast] ⚠ Model files not found — returning physics fallback.")
        return _physics_fallback()

    # ---- Load models & scalers (cached after first call) ---------------
    global _cache
    if not _cache:
        print("[forecast] Loading model artefacts from disk...")

        # XGBoost — prefer joblib for seamless platform loading
        from xgboost import XGBRegressor
        if XGB_SOLAR_JOBLIB.exists():
            xgb_sol = joblib.load(XGB_SOLAR_JOBLIB)
        else:
            xgb_sol = XGBRegressor()
            xgb_sol.load_model(str(XGB_SOLAR_JSON))

        if XGB_DEMAND_JOBLIB.exists():
            xgb_dem = joblib.load(XGB_DEMAND_JOBLIB)
        else:
            xgb_dem = XGBRegressor()
            xgb_dem.load_model(str(XGB_DEMAND_JSON))

        # Scalers
        sc_feat  = joblib.load(SCALER_FEAT)
        sc_sol   = joblib.load(SCALER_SOL)
        sc_dem   = joblib.load(SCALER_DEM)

        # LSTM
        ckpt     = torch.load(LSTM_PATH, map_location="cpu", weights_only=False)
        from models.lstm_model import SolarLSTM
        lstm     = SolarLSTM(
            input_size  = ckpt["input_size"],
            hidden_size = ckpt["hidden_size"],
            num_layers  = ckpt["num_layers"],
        )
        lstm.load_state_dict(ckpt["state_dict"])
        lstm.eval()
        seq_len  = ckpt["seq_len"]

        _cache = dict(xgb_sol=xgb_sol, xgb_dem=xgb_dem,
                      sc_feat=sc_feat, sc_sol=sc_sol, sc_dem=sc_dem,
                      lstm=lstm, seq_len=seq_len)

    xgb_sol = _cache["xgb_sol"]
    xgb_dem = _cache["xgb_dem"]
    sc_feat  = _cache["sc_feat"]
    sc_sol   = _cache["sc_sol"]
    sc_dem   = _cache["sc_dem"]
    lstm     = _cache["lstm"]
    seq_len  = _cache["seq_len"]

    # ---- Build feature DataFrame for the 24 inference hours ------------
    feat_df      = _build_inference_features(target_date)
    feat_scaled  = sc_feat.transform(feat_df.values).astype(np.float32)

    # ---- XGBoost inference ---------------------------------------------
    xgb_solar_scaled  = xgb_sol.predict(feat_scaled)
    xgb_demand_scaled = xgb_dem.predict(feat_scaled)

    xgb_solar_kw  = sc_sol.inverse_transform(xgb_solar_scaled.reshape(-1, 1)).ravel()
    xgb_demand_kw = sc_dem.inverse_transform(xgb_demand_scaled.reshape(-1, 1)).ravel()

    # ---- LSTM inference (if enabled) -----------------------------------
    if use_lstm:
        # We need seq_len past feature vectors; we replicate the first row as
        # "warm-up" history (acceptable for a 24-hour look-ahead prototype).
        warmup     = np.tile(feat_scaled[0], (seq_len, 1))  # (seq_len, n_feat)
        full_seq   = np.vstack([warmup, feat_scaled])        # (seq_len+24, n_feat)

        lstm_preds_scaled = []
        lstm.eval()
        with torch.no_grad():
            for h in range(24):
                window  = full_seq[h : h + seq_len]          # (seq_len, n_feat)
                x_t     = torch.tensor(window).unsqueeze(0)  # (1, seq_len, n_feat)
                y_pred  = lstm(x_t).item()
                lstm_preds_scaled.append(y_pred)

        lstm_solar_kw = sc_sol.inverse_transform(
            np.array(lstm_preds_scaled).reshape(-1, 1)).ravel()

        # Ensemble blend
        solar_kw = LSTM_WEIGHT * lstm_solar_kw + XGB_WEIGHT * xgb_solar_kw
    else:
        solar_kw = xgb_solar_kw

    demand_kw = xgb_demand_kw

    # ---- Post-process: physical constraints ----------------------------
    solar_kw  = np.clip(solar_kw,  0.0, 55.0)   # Can't exceed panel capacity
    demand_kw = np.clip(demand_kw, 3.0, 55.0)   # Minimum base load

    return solar_kw.astype(np.float64), demand_kw.astype(np.float64)


if __name__ == "__main__":
    solar, demand = get_24hr_forecast()
    print("Hour | Solar (kW) | Demand (kW) | Net (kW)")
    print("-" * 45)
    for h, (s, d) in enumerate(zip(solar, demand)):
        print(f"{h:4d} | {s:10.2f} | {d:11.2f} | {s-d:+.2f}")
