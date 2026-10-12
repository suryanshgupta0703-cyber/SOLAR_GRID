"""
SolarGrid — models/train_models.py
====================================
Trains and persists two complementary forecasting models:

1. **LSTM (PyTorch)**  — Captures temporal dependencies across sequential
   hourly windows to forecast solar generation.

2. **XGBoost**         — Models the nonlinear relationship between
   weather features (irradiance, temperature, cloud cover) and both
   solar generation AND baseline demand.

Saved artefacts (to solargrid/models/saved/):
    lstm_solar.pt          PyTorch LSTM state-dict
    xgb_solar.json         XGBoost solar model
    xgb_demand.json        XGBoost demand model
    scaler_features.pkl    sklearn StandardScaler for input features
    scaler_solar.pkl       sklearn StandardScaler for solar target
    scaler_demand.pkl      sklearn StandardScaler for demand target

Usage:
    cd solargrid/
    python models/train_models.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db_manager import get_connection, init_db
from models.lstm_model import SolarLSTM

SAVED_DIR = Path(__file__).parent / "saved"
SAVED_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
LSTM_SEQ_LEN     = 24      # Look-back window: 24 hours of history
LSTM_HIDDEN      = 64      # Number of LSTM hidden units
LSTM_LAYERS      = 2       # Stacked LSTM depth
LSTM_EPOCHS      = 30      # Training epochs (kept low for fast prototype)
LSTM_LR          = 1e-3    # Adam learning rate
LSTM_BATCH       = 32
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH    = 6
RANDOM_SEED      = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ===========================================================================
# 1. Data Loading
# ===========================================================================

def load_data() -> pd.DataFrame:
    """
    Load the historical_energy table from SQLite into a DataFrame.

    Raises:
        RuntimeError: If the table is empty (data_gen.py hasn't been run).

    Returns:
        pd.DataFrame sorted by timestamp.
    """
    init_db()
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM historical_energy ORDER BY timestamp", conn)
    conn.close()

    if df.empty:
        raise RuntimeError(
            "historical_energy table is empty. Run `python data_gen.py` first."
        )

    df["timestamp"]      = pd.to_datetime(df["timestamp"])
    df["hour_of_day"]    = df["timestamp"].dt.hour
    df["day_of_week"]    = df["timestamp"].dt.dayofweek
    df["day_of_year"]    = df["timestamp"].dt.dayofyear
    df["month"]          = df["timestamp"].dt.month

    # Sine/cosine encoding for circular time features avoids the 23→0 gap
    df["hour_sin"]       = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]       = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]        = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]        = np.cos(2 * np.pi * df["day_of_week"] / 7)

    print(f"[train] Loaded {len(df)} rows from DB.")
    return df


# ===========================================================================
# 2. LSTM Model Definition
# ===========================================================================

class SolarLSTM(nn.Module):
    """
    A stacked LSTM that ingests a sliding window of hourly feature vectors
    and predicts the next hour's solar generation (single-step regression).

    Architecture:
        Input  → LSTM (seq_len × input_size) → Dropout → FC → output (1)

    We use only the last hidden state (many-to-one) for prediction.
    """

    def __init__(self, input_size: int, hidden_size: int = LSTM_HIDDEN,
                 num_layers: int = LSTM_LAYERS, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,    # expects (batch, seq, features)
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_size)
        Returns:
            Tensor of shape (batch, 1) — predicted value
        """
        out, _   = self.lstm(x)       # (batch, seq, hidden)
        last_out = out[:, -1, :]      # Take only the last time-step
        dropped  = self.dropout(last_out)
        return self.fc(dropped)


def build_sequences(features: np.ndarray, target: np.ndarray,
                    seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping (X, y) sliding-window pairs for LSTM training.

    Given a time series of length T, produces:
        X: (T - seq_len, seq_len, n_features)
        y: (T - seq_len, 1)

    Args:
        features: 2-D array (T × n_features), already scaled.
        target:   1-D array (T,), already scaled.
        seq_len:  Number of past time-steps in each input window.

    Returns:
        Tuple of numpy arrays (X, y).
    """
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i])
        y.append(target[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm(df: pd.DataFrame,
               feature_cols: list[str],
               scaler_feat: StandardScaler,
               scaler_solar: StandardScaler) -> SolarLSTM:
    """
    Train the LSTM model and save its state-dict.

    Args:
        df:           Full training DataFrame.
        feature_cols: List of column names used as LSTM inputs.
        scaler_feat:  Fitted StandardScaler for features.
        scaler_solar: Fitted StandardScaler for solar target.

    Returns:
        Trained SolarLSTM instance (in eval mode).
    """
    print("\n[train] Training LSTM for solar generation...")

    X_scaled = scaler_feat.transform(df[feature_cols].values)
    y_scaled = scaler_solar.transform(
        df["historical_solar_gen_kw"].values.reshape(-1, 1)
    ).ravel()

    X_seq, y_seq = build_sequences(X_scaled, y_scaled, LSTM_SEQ_LEN)

    # Train / val split (80 / 20), no shuffle to preserve temporal order
    split        = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    # Convert to tensors
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(1)
    X_val_t   = torch.tensor(X_val)
    y_val_t   = torch.tensor(y_val).unsqueeze(1)

    model     = SolarLSTM(input_size=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    criterion = nn.HuberLoss()        # robust to outliers vs plain MSELoss

    dataset   = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader    = torch.utils.data.DataLoader(dataset, batch_size=LSTM_BATCH, shuffle=False)

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation pass
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        # Save best weights via early-stop pattern
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{LSTM_EPOCHS}  "
                  f"train_loss={epoch_loss/len(loader):.4f}  val_loss={val_loss:.4f}")

    # Restore best weights
    model.load_state_dict(best_state)
    model.eval()

    # Save state-dict
    lstm_path = SAVED_DIR / "lstm_solar.pt"
    torch.save({
        "state_dict":  model.state_dict(),
        "input_size":  len(feature_cols),
        "hidden_size": LSTM_HIDDEN,
        "num_layers":  LSTM_LAYERS,
        "seq_len":     LSTM_SEQ_LEN,
    }, lstm_path)
    print(f"  [✓] LSTM saved → {lstm_path}")
    return model


# ===========================================================================
# 3. XGBoost Models
# ===========================================================================

def train_xgboost(df: pd.DataFrame,
                  feature_cols: list[str],
                  scaler_feat: StandardScaler,
                  scaler_solar: StandardScaler,
                  scaler_demand: StandardScaler) -> tuple[XGBRegressor, XGBRegressor]:
    """
    Train two separate XGBoost regressors — one for solar, one for demand.

    XGBoost is powerful for capturing the nonlinear interaction between
    temperature, cloud cover, and irradiance without needing a sequence window.

    Args:
        df:            Full training DataFrame.
        feature_cols:  Input feature column names.
        scaler_feat:   Fitted StandardScaler for features.
        scaler_solar:  Fitted StandardScaler for solar target.
        scaler_demand: Fitted StandardScaler for demand target.

    Returns:
        Tuple (xgb_solar, xgb_demand).
    """
    print("\n[train] Training XGBoost models...")

    X = scaler_feat.transform(df[feature_cols].values)
    y_solar  = scaler_solar.transform(
        df["historical_solar_gen_kw"].values.reshape(-1, 1)).ravel()
    y_demand = scaler_demand.transform(
        df["historical_baseline_demand_kw"].values.reshape(-1, 1)).ravel()

    # Temporal split — last 20% as hold-out test
    split   = int(0.8 * len(X))
    X_train, X_test   = X[:split], X[split:]
    ys_train, ys_test = y_solar[:split], y_solar[split:]
    yd_train, yd_test = y_demand[:split], y_demand[split:]

    common_params = dict(
        n_estimators      = XGB_N_ESTIMATORS,
        max_depth         = XGB_MAX_DEPTH,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        tree_method       = "hist",    # fast CPU histogram method
        random_state      = RANDOM_SEED,
        eval_metric       = "mae",
        early_stopping_rounds = 20,
        n_jobs            = 1,
        verbosity         = 0,
    )

    # --- Solar model ---
    xgb_solar = XGBRegressor(**common_params)
    xgb_solar.fit(X_train, ys_train,
                  eval_set=[(X_test, ys_test)],
                  verbose=False)

    ys_pred = scaler_solar.inverse_transform(
        xgb_solar.predict(X_test).reshape(-1, 1)).ravel()
    ys_true = scaler_solar.inverse_transform(ys_test.reshape(-1, 1)).ravel()

    # --- Metrics helper (MAE, RMSE, MAPE) ---
    def calc_metrics(y_true, y_pred, label):
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        # Non-zero mask to avoid division by zero (e.g. night hours for solar)
        mask = y_true > 1.0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if np.any(mask) else 0.0
        print(f"  {label:12s} | MAE: {mae:6.3f} kW | RMSE: {rmse:6.3f} kW | MAPE: {mape:5.2f}%")
        return {"mae": round(mae, 3), "rmse": round(rmse, 3), "mape": round(mape, 2)}

    print("\n--- Model Evaluation (Holdout Test Set) ---")
    metrics_solar = calc_metrics(ys_true, ys_pred, "XGB Solar")

    # --- Demand model ---
    xgb_demand = XGBRegressor(**common_params)
    xgb_demand.fit(X_train, yd_train,
                   eval_set=[(X_test, yd_test)],
                   verbose=False)

    yd_pred = scaler_demand.inverse_transform(
        xgb_demand.predict(X_test).reshape(-1, 1)).ravel()
    yd_true = scaler_demand.inverse_transform(yd_test.reshape(-1, 1)).ravel()
    metrics_demand = calc_metrics(yd_true, yd_pred, "XGB Demand")

    # Persist metrics to JSON
    metrics_all = {
        "solar": metrics_solar,
        "demand": metrics_demand,
        "test_samples": len(X_test),
        "split_ratio": "80/20 train/test",
    }
    with open(SAVED_DIR / "metrics.json", "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"  [✓] Evaluation metrics saved → {SAVED_DIR / 'metrics.json'}")

    # Save models
    solar_path_json   = SAVED_DIR / "xgb_solar.json"
    demand_path_json  = SAVED_DIR / "xgb_demand.json"
    solar_path_joblib = SAVED_DIR / "xgb_solar.joblib"
    demand_path_joblib= SAVED_DIR / "xgb_demand.joblib"
    try:
        xgb_solar.save_model(str(solar_path_json))
        xgb_demand.save_model(str(demand_path_json))
    except Exception:
        pass
    joblib.dump(xgb_solar, solar_path_joblib)
    joblib.dump(xgb_demand, demand_path_joblib)
    print(f"  [✓] XGB Solar  saved → {solar_path_joblib}")
    print(f"  [✓] XGB Demand saved → {demand_path_joblib}")

    return xgb_solar, xgb_demand


# ===========================================================================
# 4. Main Training Pipeline
# ===========================================================================

def main():
    print("SolarGrid — Model Training Pipeline")
    print("=" * 45)

    df = load_data()

    # ------------------------------------------------------------------
    # Feature columns shared by both model families
    # ------------------------------------------------------------------
    feature_cols = [
        "temperature_c",
        "cloud_cover_pct",
        "solar_irradiance_wm2",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "day_of_year",
    ]

    # ------------------------------------------------------------------
    # Fit scalers on the full dataset (we're generating, not leaking test)
    # In production you'd fit only on train split.
    # ------------------------------------------------------------------
    scaler_feat   = StandardScaler()
    scaler_solar  = StandardScaler()
    scaler_demand = StandardScaler()

    scaler_feat.fit(df[feature_cols].values)
    scaler_solar.fit(df[["historical_solar_gen_kw"]].values)
    scaler_demand.fit(df[["historical_baseline_demand_kw"]].values)

    # Persist scalers for inference
    joblib.dump(scaler_feat,   SAVED_DIR / "scaler_features.pkl")
    joblib.dump(scaler_solar,  SAVED_DIR / "scaler_solar.pkl")
    joblib.dump(scaler_demand, SAVED_DIR / "scaler_demand.pkl")

    # Also save feature column list for runtime validation
    with open(SAVED_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"\nScalers saved to {SAVED_DIR}/")

    # ------------------------------------------------------------------
    # Train models
    # ------------------------------------------------------------------
    train_lstm(df, feature_cols, scaler_feat, scaler_solar)
    train_xgboost(df, feature_cols, scaler_feat, scaler_solar, scaler_demand)

    print("\n[train] All models trained and saved successfully!")
    print(f"        Artefacts directory: {SAVED_DIR}")


if __name__ == "__main__":
    main()
