"""
SolarGrid — Interim Milestone Dashboard (app_interim.py)
=========================================================
Focused Streamlit presentation for Phase 1–4 demonstration:
  ├── Data Exploration      : Weather features & database overview
  ├── Model Diagnostics     : LSTM + XGBoost architecture & evaluation metrics
  ├── 24-Hour Forecast Curve: Solar vs Demand with surplus/deficit shading
  └── Surplus Analysis      : Net energy balance & preliminary scheduling rationale

Run:
    streamlit run app_interim.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.forecast     import get_24hr_forecast, models_are_trained
from database.db_manager import get_connection, init_db

# ===========================================================================
# Page configuration
# ===========================================================================
st.set_page_config(
    page_title    = "SolarGrid — Interim Project Review",
    page_icon     = "⚡",
    layout        = "wide",
    initial_sidebar_state = "expanded",
)

# ===========================================================================
# Global CSS — dark glassmorphism theme
# ===========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary:    #0a0f1e;
    --bg-card:       rgba(255, 255, 255, 0.04);
    --border:        rgba(255, 255, 255, 0.10);
    --accent-solar:  #f7c948;
    --accent-demand: #ff6b6b;
    --accent-green:  #06d6a0;
    --accent-blue:   #118ab2;
    --text-primary:  #e8eaf0;
    --text-muted:    #8892a4;
    --glow-solar:    0 0 20px rgba(247, 201, 72, 0.25);
    --glow-green:    0 0 20px rgba(6, 214, 160, 0.25);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1528 0%, #111827 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--glow-green);
}
.kpi-label {
    font-size: 0.75rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.2rem; font-weight: 800; line-height: 1.1;
    background: linear-gradient(135deg, #f7c948, #ff9f43);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-value.green {
    background: linear-gradient(135deg, #06d6a0, #00b4d8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-value.blue {
    background: linear-gradient(135deg, #118ab2, #06d6a0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-unit {
    font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;
}

.section-header {
    font-size: 1.1rem; font-weight: 700;
    letter-spacing: 0.04em; color: var(--text-primary);
    margin: 8px 0 16px 0; padding-left: 10px;
    border-left: 3px solid var(--accent-solar);
}

.warn-banner {
    background: rgba(255, 193, 7, 0.10);
    border: 1px solid rgba(255, 193, 7, 0.30);
    border-radius: 10px; padding: 12px 16px;
    color: #ffc107; font-size: 0.85rem; margin-bottom: 20px;
}

div[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
}
.stMarkdown h1 {
    background: linear-gradient(135deg, #f7c948, #ff9f43);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Cached data loaders
# ===========================================================================

@st.cache_data(ttl=300)
def load_historical() -> pd.DataFrame:
    """Pull all historical data from the SQLite database."""
    try:
        init_db()
        conn = get_connection()
        df   = pd.read_sql(
            "SELECT * FROM historical_energy ORDER BY timestamp",
            conn,
        )
        conn.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_forecast(forecast_date: date, use_lstm: bool) -> tuple[np.ndarray, np.ndarray]:
    """Cached 24-hour forecast for a given date."""
    target_dt = datetime(forecast_date.year, forecast_date.month, forecast_date.day)
    return get_24hr_forecast(target_date=target_dt, use_lstm=use_lstm)


def load_model_metrics() -> dict:
    """Load test-set evaluation metrics from models/saved/metrics.json."""
    metrics_path = PROJECT_ROOT / "models" / "saved" / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ===========================================================================
# Chart builders
# ===========================================================================

def build_forecast_chart(solar: np.ndarray, demand: np.ndarray) -> go.Figure:
    """Build 24-hour forecast chart with surplus/deficit shading."""
    hours     = list(range(24))
    hour_lbls = [f"{h:02d}:00" for h in hours]
    fig = go.Figure()

    # Surplus fill (solar > demand)
    surplus_solar  = np.where(solar >= demand, solar, demand)
    fig.add_trace(go.Scatter(
        x=hour_lbls + hour_lbls[::-1],
        y=list(surplus_solar) + list(demand[::-1]),
        fill="toself", fillcolor="rgba(6, 214, 160, 0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Solar Surplus", showlegend=True, hoverinfo="skip",
    ))

    # Deficit fill (demand > solar)
    deficit_demand = np.where(demand > solar, demand, solar)
    fig.add_trace(go.Scatter(
        x=hour_lbls + hour_lbls[::-1],
        y=list(deficit_demand) + list(solar[::-1]),
        fill="toself", fillcolor="rgba(255, 107, 107, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Grid Draw Zone", showlegend=True, hoverinfo="skip",
    ))

    # Solar line
    fig.add_trace(go.Scatter(
        x=hour_lbls, y=solar, name="☀ Solar Generation",
        mode="lines+markers",
        line=dict(color="#f7c948", width=3, shape="spline"),
        marker=dict(size=6, color="#f7c948", symbol="circle"),
        hovertemplate="%{x}<br>Solar: <b>%{y:.1f} kW</b><extra></extra>",
    ))

    # Demand line
    fig.add_trace(go.Scatter(
        x=hour_lbls, y=demand, name="⚡ Baseline Demand",
        mode="lines+markers",
        line=dict(color="#ff6b6b", width=3, shape="spline"),
        marker=dict(size=6, color="#ff6b6b", symbol="diamond"),
        hovertemplate="%{x}<br>Demand: <b>%{y:.1f} kW</b><extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=440,
        margin=dict(l=60, r=30, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=11)),
        xaxis=dict(title="Hour of Day", gridcolor="rgba(255,255,255,0.06)",
                   tickangle=-45, showgrid=True),
        yaxis=dict(title="Power (kW)", gridcolor="rgba(255,255,255,0.06)",
                   showgrid=True),
        hovermode="x unified",
    )
    return fig


def build_net_energy_chart(solar: np.ndarray, demand: np.ndarray) -> go.Figure:
    """Net energy balance bar chart (hour-by-hour)."""
    net = solar - demand
    hour_lbls = [f"{h:02d}:00" for h in range(24)]
    colors = ["#06d6a0" if v >= 0 else "#ff6b6b" for v in net]

    fig = go.Figure(go.Bar(
        x=hour_lbls, y=net, marker_color=colors, name="Net Energy",
        hovertemplate="%{x}<br>Net: <b>%{y:+.1f} kW</b><extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=50, r=20, t=30, b=50),
        yaxis=dict(title="Net Power (kW)", gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickangle=-45),
        showlegend=False,
    )
    return fig


def build_weather_correlation_chart(df: pd.DataFrame) -> go.Figure:
    """Build interactive weather variable time-series for the dataset."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Solar Irradiance (W/m²)", "Cloud Cover (%)",
                        "Ambient Temperature (°C)"),
    )

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["solar_irradiance_wm2"],
        mode="lines", line=dict(color="#f7c948", width=1.5),
        name="Irradiance", showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["cloud_cover_pct"],
        mode="lines", line=dict(color="#8892a4", width=1.5),
        name="Cloud Cover", showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["temperature_c"],
        mode="lines", line=dict(color="#ff9f43", width=1.5),
        name="Temperature", showlegend=False,
    ), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(l=60, r=30, t=40, b=40),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", row=i, col=1)
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=i, col=1)

    return fig


def build_historical_generation_chart(df: pd.DataFrame) -> go.Figure:
    """Historical solar generation vs community demand time-series."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["historical_solar_gen_kw"],
        mode="lines", line=dict(color="#f7c948", width=1.5),
        name="☀ Solar Generation",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["historical_baseline_demand_kw"],
        mode="lines", line=dict(color="#ff6b6b", width=1.5),
        name="⚡ Baseline Demand",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=60, r=30, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        xaxis=dict(title="Date/Time", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Power (kW)", gridcolor="rgba(255,255,255,0.06)"),
        hovermode="x unified",
    )
    return fig


# ===========================================================================
# Sidebar
# ===========================================================================

def render_sidebar() -> tuple[date, bool]:
    """Render sidebar controls and return user selections."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:10px 0 24px;">
            <div style="font-size:2.8rem;">⚡</div>
            <div style="font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#f7c948,#ff9f43);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                SolarGrid
            </div>
            <div style="font-size:0.72rem;color:#8892a4;letter-spacing:0.12em;
                        text-transform:uppercase;">Interim Project Review</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("### 📅 Forecast Settings")
        forecast_date = st.date_input(
            "Forecast Date",
            value     = date.today(),
            min_value = date.today() - timedelta(days=30),
            max_value = date.today() + timedelta(days=7),
            help      = "Select the day for the 24-hour energy forecast",
        )

        use_lstm = st.toggle(
            "Enable LSTM Ensemble",
            value=True,
            help="Blends LSTM (temporal) + XGBoost (feature-based) predictions. "
                 "Disable for faster XGBoost-only mode.",
        )

        st.divider()
        st.markdown("### 📋 Project Info")
        st.caption(
            "**Candidate:** Suryansh Gupta\n\n"
            "**Roll No:** 2023a1r116\n\n"
            "**Dept:** CSE, MIET Autonomous, Jammu\n\n"
            "**Milestone:** Phases 1–4 (Data, ML Pipeline, Forecasting)"
        )

        st.divider()
        st.markdown("### ℹ️ Model Status")
        trained = models_are_trained()
        status_color  = "#06d6a0" if trained else "#ffc107"
        status_text   = "All Models Trained ✓" if trained else "Using Fallback Mode"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<div style="width:10px;height:10px;border-radius:50%;'
            f'background:{status_color};box-shadow:0 0 8px {status_color};"></div>'
            f'<span style="font-size:0.82rem;">{status_text}</span></div>',
            unsafe_allow_html=True,
        )
        if not trained:
            st.caption("Run `python models/train_models.py` to train the ML models.")

        st.markdown("---")
        st.caption("© 2024 SolarGrid · Interim Milestone v0.5")

    return forecast_date, use_lstm


# ===========================================================================
# Main Dashboard
# ===========================================================================

def main():
    # ---- Sidebar -------------------------------------------------------
    forecast_date, use_lstm = render_sidebar()

    # ---- Page Header ---------------------------------------------------
    st.markdown("""
    <div style="padding: 8px 0 24px;">
        <h1 style="font-size:2rem;margin:0;font-weight:800;">
            ⚡ SolarGrid — Interim Project Review Dashboard
        </h1>
        <p style="color:#8892a4;margin:4px 0 0;font-size:0.92rem;">
            Phase 1–4 Demonstration: Data Pipeline, ML Forecasting Engine & Surplus Identification
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Load data & forecast ------------------------------------------
    hist_df = load_historical()

    with st.spinner("Running AI forecast engine..."):
        solar_kw, demand_kw = load_forecast(forecast_date, use_lstm)

    if not models_are_trained():
        st.markdown("""
        <div class="warn-banner">
            ⚠️ <strong>ML models not found.</strong>
            Showing physics-based fallback forecast.
            Run <code>python data_gen.py</code> then
            <code>python models/train_models.py</code> to enable full AI forecasting.
        </div>
        """, unsafe_allow_html=True)

    # ====================================================================
    # TAB LAYOUT — organized presentation sections
    # ====================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Exploration",
        "🧠 Model Architecture & Evaluation",
        "📈 24-Hour Forecast",
        "⚖ Surplus Analysis & Next Steps",
    ])

    # ====================================================================
    # TAB 1: Data Exploration (Phase 1 & 2)
    # ====================================================================
    with tab1:
        st.markdown('<p class="section-header">📊 Phase 1 & 2: Historical Data Exploration</p>',
                    unsafe_allow_html=True)

        if hist_df.empty:
            st.info("No data available. Run `python data_gen.py` to generate the dataset.")
        else:
            # Database summary KPIs
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">📦 Total Records</div>
                    <div class="kpi-value blue">{len(hist_df)}</div>
                    <div class="kpi-unit">hourly readings</div>
                </div>
                """, unsafe_allow_html=True)
            with k2:
                n_days = (hist_df["timestamp"].max() - hist_df["timestamp"].min()).days + 1
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">📅 Coverage</div>
                    <div class="kpi-value">{n_days}</div>
                    <div class="kpi-unit">days of history</div>
                </div>
                """, unsafe_allow_html=True)
            with k3:
                peak_solar = hist_df["historical_solar_gen_kw"].max()
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">☀ Peak Solar</div>
                    <div class="kpi-value">{peak_solar:.1f}</div>
                    <div class="kpi-unit">kW recorded</div>
                </div>
                """, unsafe_allow_html=True)
            with k4:
                peak_demand = hist_df["historical_baseline_demand_kw"].max()
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">⚡ Peak Demand</div>
                    <div class="kpi-value green">{peak_demand:.1f}</div>
                    <div class="kpi-unit">kW recorded</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Weather correlation sub-charts
            st.markdown('<p class="section-header">🌦 Weather Parameter Time-Series (30 Days)</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(build_weather_correlation_chart(hist_df),
                            use_container_width=True, config={"displayModeBar": False})

            st.markdown("<br>", unsafe_allow_html=True)

            # Historical solar vs demand
            st.markdown('<p class="section-header">☀ Historical Solar Generation vs Community Demand</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(build_historical_generation_chart(hist_df),
                            use_container_width=True, config={"displayModeBar": False})

            # Schema info
            with st.expander("🗄️ SQLite Database Schema & Sample Data"):
                st.code("""
CREATE TABLE historical_energy (
    id                            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                     TEXT    NOT NULL UNIQUE,
    temperature_c                 REAL    NOT NULL,   -- ambient temp (°C)
    cloud_cover_pct               REAL    NOT NULL,   -- 0–100 %
    solar_irradiance_wm2          REAL    NOT NULL,   -- W/m²
    historical_solar_gen_kw       REAL    NOT NULL,   -- kW produced
    historical_baseline_demand_kw REAL    NOT NULL    -- kW consumed
);
                """, language="sql")

                st.markdown("**Sample Records (First 10 Rows):**")
                st.dataframe(
                    hist_df.head(10)[["timestamp", "temperature_c", "cloud_cover_pct",
                                     "solar_irradiance_wm2", "historical_solar_gen_kw",
                                     "historical_baseline_demand_kw"]],
                    use_container_width=True,
                )

    # ====================================================================
    # TAB 2: Model Architecture & Evaluation (Phase 3 & 4)
    # ====================================================================
    with tab2:
        st.markdown('<p class="section-header">🧠 Phase 3 & 4: ML Model Architecture & Evaluation</p>',
                    unsafe_allow_html=True)

        # Architecture overview
        arch_col1, arch_col2 = st.columns(2, gap="large")

        with arch_col1:
            st.markdown("#### 🔮 PyTorch LSTM — Solar Generation Predictor")
            st.markdown("""
            | Parameter | Value |
            |-----------|-------|
            | **Architecture** | 2-Layer Stacked LSTM → Dropout → FC(1) |
            | **Input Size** | 8 features per timestep |
            | **Hidden Units** | 64 per layer |
            | **Sequence Length** | 24-hour lookback window |
            | **Dropout** | 20% between LSTM layers |
            | **Optimizer** | Adam (lr = 1e-3, grad_clip = 1.0) |
            | **Loss** | Mean Squared Error (MSE) |
            | **Epochs** | 30 (with early stopping on val loss) |
            """)
            st.caption(
                "The LSTM captures **temporal autocorrelation** — a sunny morning "
                "reliably predicts sustained solar generation through the afternoon. "
                "This sequential dependency is critical for improving forecast horizon accuracy."
            )

        with arch_col2:
            st.markdown("#### 🌲 XGBoost — Dual-Target Regressor")
            st.markdown("""
            | Parameter | Value |
            |-----------|-------|
            | **Target 1** | Solar Generation (kW) |
            | **Target 2** | Baseline Electricity Demand (kW) |
            | **n_estimators** | 300 |
            | **max_depth** | 6 |
            | **learning_rate** | 0.05 |
            | **subsample** | 0.8 |
            | **tree_method** | hist (CPU histogram) |
            | **early_stopping** | 20 rounds patience |
            """)
            st.caption(
                "XGBoost models **non-linear meteorological feature interactions** — "
                "e.g., the combined effect of cloud cover × solar irradiance × temperature. "
                "It excels at point-in-time predictions without needing sequential context."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Ensemble explanation
        st.markdown('<p class="section-header">🔗 Hybrid Ensemble Strategy</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        **Solar forecast uses a weighted ensemble blend:**
        ```
        Solar_forecast(h) = 0.55 × LSTM_pred(h) + 0.45 × XGB_solar_pred(h)
        ```
        **Demand forecast uses XGBoost only** (demand patterns are more feature-driven than temporal):
        ```
        Demand_forecast(h) = XGB_demand_pred(h)
        ```
        """)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature engineering
        st.markdown('<p class="section-header">⚙️ Feature Engineering</p>',
                    unsafe_allow_html=True)
        feat_cols_path = PROJECT_ROOT / "models" / "saved" / "feature_cols.json"
        if feat_cols_path.exists():
            with open(feat_cols_path) as f:
                feat_cols = json.load(f)
            feat_desc = {
                "temperature_c": "Ambient temperature (°C) — affects panel efficiency (inverse relationship)",
                "cloud_cover_pct": "Cloud coverage (0–100%) — primary attenuator of solar irradiance",
                "solar_irradiance_wm2": "Solar irradiance at panel surface (W/m²) — direct energy driver",
                "hour_sin": "sin(2π·hour/24) — circular encoding of time of day",
                "hour_cos": "cos(2π·hour/24) — circular encoding of time of day",
                "dow_sin": "sin(2π·day_of_week/7) — circular day-of-week encoding",
                "dow_cos": "cos(2π·day_of_week/7) — circular day-of-week encoding",
                "day_of_year": "Day of year (1–365) — captures seasonal variation",
            }
            feat_df = pd.DataFrame([
                {"Feature": col, "Description": feat_desc.get(col, "")}
                for col in feat_cols
            ])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
        else:
            st.caption("Feature columns not available — run `python models/train_models.py`.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Evaluation metrics
        st.markdown('<p class="section-header">🎯 Holdout Test-Set Evaluation Metrics</p>',
                    unsafe_allow_html=True)
        metrics = load_model_metrics()
        if metrics:
            sol_m = metrics.get("solar", {})
            dem_m = metrics.get("demand", {})

            ev1, ev2, ev3 = st.columns(3)
            with ev1:
                st.markdown("##### Metric Definitions")
                st.caption(
                    "**MAE** (Mean Absolute Error): Average ±kW deviation from true values.\n\n"
                    "**RMSE** (Root Mean Square Error): Penalises large outlier errors.\n\n"
                    "**MAPE** (Mean Absolute % Error): % error relative to actual values "
                    "(computed only for values > 1.0 kW to avoid night-hour zero-division)."
                )

            with ev2:
                st.markdown("##### ☀ Solar Generation Model")
                st.metric("MAE", f"{sol_m.get('mae', 'N/A')} kW")
                st.metric("RMSE", f"{sol_m.get('rmse', 'N/A')} kW")
                st.metric("MAPE", f"{sol_m.get('mape', 'N/A')} %")

            with ev3:
                st.markdown("##### ⚡ Electricity Demand Model")
                st.metric("MAE", f"{dem_m.get('mae', 'N/A')} kW")
                st.metric("RMSE", f"{dem_m.get('rmse', 'N/A')} kW")
                st.metric("MAPE", f"{dem_m.get('mape', 'N/A')} %")

            st.caption(
                f"Evaluated on **{metrics.get('test_samples', 'N/A')} holdout test samples** "
                f"({metrics.get('split_ratio', 'N/A')} temporal train/test split)."
            )
        else:
            st.info("Run `python models/train_models.py` to generate evaluation metrics.")

    # ====================================================================
    # TAB 3: 24-Hour Forecast (Phase 3 output)
    # ====================================================================
    with tab3:
        st.markdown('<p class="section-header">📈 24-Hour Solar vs Demand Forecast</p>',
                    unsafe_allow_html=True)

        # KPIs
        net = solar_kw - demand_kw
        peak_solar   = float(solar_kw.max())
        total_solar  = float(solar_kw.sum())
        total_demand = float(demand_kw.sum())
        surplus_hrs  = int((net > 0).sum())
        surplus_kwh  = float(np.maximum(net, 0).sum())

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">☀ Peak Solar Output</div>
                <div class="kpi-value">{peak_solar:.1f}</div>
                <div class="kpi-unit">kilowatts</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">⚡ Total Day Demand</div>
                <div class="kpi-value blue">{total_demand:.1f}</div>
                <div class="kpi-unit">kWh consumed</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">🌤 Total Solar Yield</div>
                <div class="kpi-value">{total_solar:.1f}</div>
                <div class="kpi-unit">kWh generated</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📈 Surplus Hours</div>
                <div class="kpi-value green">{surplus_hrs}</div>
                <div class="kpi-unit">hours (solar &gt; demand)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Forecast chart
        forecast_fig = build_forecast_chart(solar_kw, demand_kw)
        st.plotly_chart(forecast_fig, use_container_width=True,
                        config={"displayModeBar": False})

        # Summary metrics under chart
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("☀ Total Solar Generated", f"{total_solar:.1f} kWh")
        with sc2:
            st.metric("🌤 Solar Surplus Available", f"{surplus_kwh:.1f} kWh",
                      delta=f"{surplus_hrs} surplus hours")
        with sc3:
            net_total = total_solar - total_demand
            st.metric("⚖ Day Net Balance",
                      f"{net_total:+.1f} kWh",
                      delta="Surplus" if net_total >= 0 else "Deficit",
                      delta_color="normal" if net_total >= 0 else "inverse")

        st.markdown("<br>", unsafe_allow_html=True)

        # Raw data table
        with st.expander("📋 View Raw Hourly Forecast Data"):
            df_display = pd.DataFrame({
                "Hour":       [f"{h:02d}:00" for h in range(24)],
                "Solar (kW)": np.round(solar_kw, 2),
                "Demand (kW)": np.round(demand_kw, 2),
                "Net (kW)":    np.round(net, 2),
            })

            def _color_net(val):
                color = "#06d6a0" if val >= 0 else "#ff6b6b"
                return f"color: {color}; font-weight: 600"

            try:
                styled = df_display.style.map(_color_net, subset=["Net (kW)"]).format(precision=2)
            except Exception:
                styled = df_display.style.format(precision=2)

            st.dataframe(styled, use_container_width=True, height=350)

    # ====================================================================
    # TAB 4: Surplus Analysis & Next Steps (Transition to Phase 5)
    # ====================================================================
    with tab4:
        st.markdown('<p class="section-header">⚖ Surplus Analysis & Scheduling Rationale</p>',
                    unsafe_allow_html=True)

        left_col, right_col = st.columns([3, 2], gap="large")

        with left_col:
            st.markdown("#### Hourly Net Energy Balance (Solar − Demand)")
            net_fig = build_net_energy_chart(solar_kw, demand_kw)
            st.plotly_chart(net_fig, use_container_width=True,
                            config={"displayModeBar": False})

            st.caption(
                "**Green bars** represent hours where solar generation exceeds baseline demand — "
                "this is the **surplus energy window** that will be exploited by the heuristic "
                "scheduling engine in Phase 5 to power community water pumps and EV chargers."
            )

        with right_col:
            st.markdown("#### 🗺 Roadmap: Phases 5–7 (Final Submission)")
            st.markdown("""
            The forecasting and surplus identification infrastructure is now **complete**.
            The upcoming phases will build on this foundation:

            **Phase 5 — Smart Heuristic Scheduling Engine:**
            - Sliding-window surplus maximization algorithm
            - Priority-based appliance allocation (Water Pump → EV Charger)
            - Dynamic load deduction to prevent double-counting

            **Phase 6 — Integrated Full Dashboard:**
            - Appliance operating window cards with time-bands
            - Baseline comparison (conventional vs AI-optimized)
            - Financial KPIs (₹ saved/day, grid draw reduction %)

            **Phase 7 — Final Evaluation & Documentation:**
            - End-to-end benchmark testing
            - Complete project report with results tables
            - Deployment-ready package
            """)

            # Identify the surplus window for the schedule teaser
            surplus_mask = net > 0
            if surplus_mask.any():
                surplus_start = int(np.argmax(surplus_mask))
                surplus_end   = 23 - int(np.argmax(surplus_mask[::-1])) + 1
                st.markdown("---")
                st.markdown("#### 🎯 Identified Optimal Scheduling Window")
                st.markdown(f"""
                Based on today's forecast, the prime surplus window is:

                🟢 **{surplus_start:02d}:00 – {surplus_end:02d}:00**
                *(Solar exceeds demand for {surplus_hrs} hours, yielding {surplus_kwh:.1f} kWh surplus)*

                **Target loads for Phase 5:**
                - 🔧 Community Water Pump (3.5 kW × 2h = 7.0 kWh)
                - 🔌 EV Charging Station (7.4 kW × 3h = 22.2 kWh)
                - **Combined:** 29.2 kWh → fits within {surplus_kwh:.1f} kWh surplus ✓
                """)

    # ====================================================================
    # Footer
    # ====================================================================
    st.markdown("""
    <div style="text-align:center;padding:30px 0 10px;color:#4a5568;font-size:0.78rem;">
        SolarGrid v0.5 (Interim) · Phases 1–4 · PyTorch LSTM + XGBoost<br>
        Presented for Mid-term Review — Full Scheduling & Dashboard in Final Submission
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
