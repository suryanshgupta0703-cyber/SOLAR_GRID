"""
SolarGrid — Dashboard  (app_v2.py)
==================================
Tabs: Data Exploration · Model Architecture & Evaluation · 24-Hour Forecast ·
Surplus Analysis · Carbon & Savings · Smart Scheduler.

Sidebar: data source (Historical Benchmark | Live City Forecast), city, forecast date.

Run:
    streamlit run app.py
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

from models.forecast import get_24hr_forecast, models_are_trained, _build_inference_features
from database.db_manager import get_connection, init_db

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------
try:
    from data_fetcher import fetch_live_forecast, CITY_PRESETS
    _HAS_DATA_FETCHER = True
except ImportError:
    _HAS_DATA_FETCHER = False
    CITY_PRESETS = {}
    fetch_live_forecast = None  # type: ignore

try:
    from models.scheduler import (
        run_scheduler,
        compute_tod_savings,
        compute_carbon_savings,
        DEFAULT_APPLIANCES,
        Appliance,
        ScheduleResult,
    )
    _HAS_SCHEDULER = True
except ImportError:
    _HAS_SCHEDULER = False
    DEFAULT_APPLIANCES = []  # type: ignore

# ===========================================================================
# Page configuration
# ===========================================================================
st.set_page_config(
    page_title="SolarGrid Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# Global CSS — dark glassmorphism theme  (copied verbatim from app_interim.py)
# + new .ai-summary-box class for Tab 5
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

.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 100% !important;
    padding: 3rem 2rem 2rem 2rem !important;
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

/* ---- New CSS for Phase 5–7 ---- */
.ai-summary-box {
    background: linear-gradient(135deg, rgba(6,214,160,0.08), rgba(17,138,178,0.08));
    border: 1px solid rgba(6,214,160,0.25);
    border-radius: 16px; padding: 24px 28px; margin-bottom: 24px;
    backdrop-filter: blur(12px);
}

.live-banner {
    background: rgba(17, 138, 178, 0.12);
    border: 1px solid rgba(17, 138, 178, 0.35);
    border-radius: 10px; padding: 12px 16px;
    color: #00b4d8; font-size: 0.88rem; margin-bottom: 16px;
}

.safety-ok {
    background: rgba(6, 214, 160, 0.10);
    border: 1px solid rgba(6, 214, 160, 0.30);
    border-radius: 10px; padding: 12px 16px;
    color: #06d6a0; font-size: 0.88rem;
}

.safety-warn {
    background: rgba(255, 107, 107, 0.10);
    border: 1px solid rgba(255, 107, 107, 0.35);
    border-radius: 10px; padding: 12px 16px;
    color: #ff6b6b; font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Cached data loaders  (all from app_interim.py + new live-forecast loader)
# ===========================================================================

@st.cache_data(ttl=300)
def load_historical() -> pd.DataFrame:
    """Pull all historical data from the SQLite database."""
    try:
        init_db()
        conn = get_connection()
        df = pd.read_sql(
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
    """Cached 24-hour benchmark forecast for a given date."""
    target_dt = datetime(forecast_date.year, forecast_date.month, forecast_date.day)
    return get_24hr_forecast(target_date=target_dt, use_lstm=use_lstm)


@st.cache_data(ttl=300)
def load_live_forecast(lat: float, lon: float, city_name: str, forecast_date: date,
                       use_lstm: bool) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Fetch live weather from Open-Meteo for one calendar day and run ML models on it.

    Returns (solar_kw, demand_kw, used_live) where used_live=True when real
    data was successfully fetched.
    """
    if not _HAS_DATA_FETCHER or fetch_live_forecast is None:
        solar, demand = load_forecast(forecast_date, use_lstm)
        return solar, demand, False

    try:
        live_df = fetch_live_forecast(lat=lat, lon=lon, city_name=city_name,
                                      target_date=forecast_date)
        if live_df is None or len(live_df) != 24:
            raise ValueError("Live forecast unavailable or incomplete")

        # Feed into ML models via forecast module internals
        from models.forecast import _cache as _fcache, models_are_trained as _mtr
        from models.forecast import _physics_fallback, LSTM_WEIGHT, XGB_WEIGHT

        if not _mtr():
            solar, demand = _physics_fallback()
            return solar, demand, True  # live weather used but physics fallback for power

        import joblib, torch
        from models.forecast import (
            SCALER_FEAT, SCALER_SOL, SCALER_DEM,
            LSTM_PATH, XGB_SOLAR_JOBLIB, XGB_SOLAR_JSON,
            XGB_DEMAND_JOBLIB, XGB_DEMAND_JSON,
        )

        # Ensure cache is warm
        if not _fcache:
            # Trigger lazy load by calling get_24hr_forecast once
            get_24hr_forecast(use_lstm=use_lstm)

        sc_feat = _fcache["sc_feat"]
        sc_sol  = _fcache["sc_sol"]
        sc_dem  = _fcache["sc_dem"]
        xgb_sol = _fcache["xgb_sol"]
        xgb_dem = _fcache["xgb_dem"]
        lstm    = _fcache["lstm"]
        seq_len = _fcache["seq_len"]

        feat_scaled = sc_feat.transform(live_df.values).astype(np.float32)

        xgb_solar_scaled  = xgb_sol.predict(feat_scaled)
        xgb_demand_scaled = xgb_dem.predict(feat_scaled)
        xgb_solar_kw  = sc_sol.inverse_transform(xgb_solar_scaled.reshape(-1, 1)).ravel()
        xgb_demand_kw = sc_dem.inverse_transform(xgb_demand_scaled.reshape(-1, 1)).ravel()

        if use_lstm:
            warmup   = np.tile(feat_scaled[0], (seq_len, 1))
            full_seq = np.vstack([warmup, feat_scaled])
            lstm_preds_scaled = []
            lstm.eval()
            with torch.no_grad():
                for h in range(24):
                    window = full_seq[h: h + seq_len]
                    x_t = torch.tensor(window).unsqueeze(0)
                    lstm_preds_scaled.append(lstm(x_t).item())
            lstm_solar_kw = sc_sol.inverse_transform(
                np.array(lstm_preds_scaled).reshape(-1, 1)).ravel()
            solar_kw = LSTM_WEIGHT * lstm_solar_kw + XGB_WEIGHT * xgb_solar_kw
        else:
            solar_kw = xgb_solar_kw

        solar_kw  = np.clip(solar_kw,  0.0, 55.0)
        demand_kw = np.clip(xgb_demand_kw, 3.0, 55.0)
        return solar_kw.astype(np.float64), demand_kw.astype(np.float64), True

    except Exception as exc:
        print(f"[app_v2] ⚠ Live forecast failed: {exc}. Falling back to benchmark.")
        solar, demand = load_forecast(forecast_date, use_lstm)
        return solar, demand, False


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
# AI summary generator
# ===========================================================================

def generate_ai_summary(
    weather_cond: str,
    peak_start: int,
    peak_end: int,
    surplus_kwh: float,
    num_appliances: int,
    rupees_saved: float,
    co2_kg: float,
) -> str:
    weather_emojis = {
        "clear": "☀️",
        "partly cloudy": "⛅",
        "overcast": "🌧️",
        "cloudy": "🌥️",
    }
    emoji = weather_emojis.get(weather_cond.lower(), "🌤️")
    trees = co2_kg / 21.77
    return f"""
{emoji} **Today's Conditions:** {weather_cond.title()}. Peak solar generation window: **{peak_start:02d}:00 – {peak_end:02d}:00**.

🤖 **AI Actions Taken:** Automatically shifted **{num_appliances} appliance(s)** into the solar surplus window, consuming **{surplus_kwh:.1f} kWh** of otherwise-wasted solar energy.

💰 **Financial Impact:** Saved **₹{rupees_saved:,.0f}** vs. unmanaged peak-hour tariff today.

🌱 **Environmental Impact:** Avoided **{co2_kg:.1f} kg of CO₂** emissions — equivalent to planting **{trees:.2f} mature trees** today.
"""


# ===========================================================================
# Chart builders  (verbatim from app_interim.py)
# ===========================================================================

def build_forecast_chart(solar: np.ndarray, demand: np.ndarray) -> go.Figure:
    """Build 24-hour forecast chart with surplus/deficit shading."""
    hours = list(range(24))
    hour_lbls = [f"{h:02d}:00" for h in hours]
    fig = go.Figure()

    surplus_solar = np.where(solar >= demand, solar, demand)
    fig.add_trace(go.Scatter(
        x=hour_lbls + hour_lbls[::-1],
        y=list(surplus_solar) + list(demand[::-1]),
        fill="toself", fillcolor="rgba(6, 214, 160, 0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Solar Surplus", showlegend=True, hoverinfo="skip",
    ))

    deficit_demand = np.where(demand > solar, demand, solar)
    fig.add_trace(go.Scatter(
        x=hour_lbls + hour_lbls[::-1],
        y=list(deficit_demand) + list(solar[::-1]),
        fill="toself", fillcolor="rgba(255, 107, 107, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Grid Draw Zone", showlegend=True, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=hour_lbls, y=solar, name="☀ Solar Generation",
        mode="lines+markers",
        line=dict(color="#f7c948", width=3, shape="spline"),
        marker=dict(size=6, color="#f7c948", symbol="circle"),
        hovertemplate="%{x}<br>Solar: <b>%{y:.1f} kW</b><extra></extra>",
    ))

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
        height=520,
        margin=dict(l=60, r=30, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        xaxis=dict(title="Date/Time", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Power (kW)", gridcolor="rgba(255,255,255,0.06)"),
        hovermode="x unified",
    )
    return fig


# ===========================================================================
# NEW chart builders — Phase 5–7
# ===========================================================================

def build_gantt_chart(
    schedule: dict,
    solar_kw: np.ndarray,
    demand_kw: np.ndarray,
    peak_cap_kw: float = 35.0,
) -> go.Figure:
    """
    Plotly horizontal bar Gantt chart for the appliance schedule.

    Bars are green when the hour is in solar surplus, orange when grid-draw.
    A light-green background band highlights the solar surplus window.
    A phantom 'Before (Unmanaged)' row shows the worst-case baseline.
    """
    hour_lbls = [f"{h:02d}:00" for h in range(24)]
    surplus_mask = (solar_kw - demand_kw) > 0

    # Surplus window for background shading
    surplus_hours = [h for h in range(24) if surplus_mask[h]]
    surplus_start_h = min(surplus_hours) if surplus_hours else 10
    surplus_end_h   = max(surplus_hours) + 1 if surplus_hours else 15

    fig = go.Figure()

    # -- Background green band for surplus window --
    fig.add_vrect(
        x0=surplus_start_h - 0.5,
        x1=surplus_end_h - 0.5,
        fillcolor="rgba(6, 214, 160, 0.06)",
        line_width=0,
        annotation_text="☀ Solar Surplus Window",
        annotation_position="top left",
        annotation_font_color="#06d6a0",
        annotation_font_size=10,
    )

    appliance_names = list(schedule.keys())
    # Add phantom unmanaged row at the top (before scheduled)
    all_names = ["Before (Unmanaged)"] + appliance_names

    # Phantom row: assume all appliances would run 08:00–11:00 (morning peak)
    phantom_hours = list(range(8, 11))
    for h in phantom_hours:
        fig.add_trace(go.Bar(
            x=[1],
            y=["Before (Unmanaged)"],
            base=[h],
            orientation="h",
            marker_color="rgba(255, 107, 107, 0.45)",
            marker_line_color="rgba(255, 107, 107, 0.8)",
            marker_line_width=1,
            name="Unmanaged (baseline)",
            showlegend=(h == phantom_hours[0]),
            hovertemplate=f"Unmanaged: {h:02d}:00–{h+1:02d}:00<extra></extra>",
        ))

    # Appliance bars
    colors_surplus = "#06d6a0"
    colors_grid    = "#ff9f43"

    shown_surplus = False
    shown_grid    = False

    for appl_name, hours_active in schedule.items():
        if not hours_active:
            continue
        for h in hours_active:
            is_surplus = bool(surplus_mask[h]) if h < len(surplus_mask) else False
            color = colors_surplus if is_surplus else colors_grid
            lname_surplus = "Solar-Powered ✓" if not shown_surplus else None
            lname_grid    = "Grid Draw" if not shown_grid else None

            show_legend = False
            legend_name = appl_name
            if is_surplus and not shown_surplus:
                legend_name = "Solar-Powered ✓"
                show_legend = True
                shown_surplus = True
            elif not is_surplus and not shown_grid:
                legend_name = "Grid Draw"
                show_legend = True
                shown_grid = True

            fig.add_trace(go.Bar(
                x=[1],
                y=[appl_name],
                base=[h],
                orientation="h",
                marker_color=color,
                marker_line_color=color,
                marker_line_width=0.5,
                name=legend_name,
                showlegend=show_legend,
                hovertemplate=(
                    f"{appl_name}<br>{h:02d}:00–{h+1:02d}:00<br>"
                    f"{'☀ Solar' if is_surplus else '⚡ Grid'}<extra></extra>"
                ),
            ))

    # X-axis ticks
    tick_vals = list(range(0, 24))
    tick_text = [f"{h:02d}:00" for h in range(0, 24)]

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="stack",
        height=max(260, 60 * (len(appliance_names) + 2)),
        margin=dict(l=160, r=30, t=50, b=60),
        xaxis=dict(
            title="Hour of Day",
            range=[-0.5, 23.5],
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=-45,
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            categoryorder="array",
            categoryarray=all_names[::-1],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=11)),
        title=dict(text="📅 Appliance Schedule Gantt Chart", font=dict(size=14)),
    )
    return fig


def build_power_balance_chart(
    solar_kw: np.ndarray,
    demand_kw: np.ndarray,
    schedule: dict,
    appliance_powers: dict,
) -> go.Figure:
    """
    24-hour stacked area chart:
      • Base demand (bottom fill)
      • Each appliance load stacked on top (one colour per appliance)
      • Solar generation curve overlay
    """
    hour_lbls = [f"{h:02d}:00" for h in range(24)]

    # Colour palette for appliances
    appl_colours = ["#f7c948", "#06d6a0", "#118ab2", "#ff9f43", "#a29bfe", "#fd79a8"]

    fig = go.Figure()

    # Base demand fill
    fig.add_trace(go.Scatter(
        x=hour_lbls, y=demand_kw,
        fill="tozeroy",
        fillcolor="rgba(255, 107, 107, 0.18)",
        line=dict(color="#ff6b6b", width=2),
        name="⚡ Base Demand",
        stackgroup=None,
        hovertemplate="%{x}<br>Base Demand: <b>%{y:.1f} kW</b><extra></extra>",
    ))

    # Stacked appliance loads
    cumulative = demand_kw.copy()
    for idx, (appl_name, hours_active) in enumerate(schedule.items()):
        if not hours_active:
            continue
        pwr = appliance_powers.get(appl_name, 0.0)
        load = np.zeros(24)
        for h in hours_active:
            if 0 <= h < 24:
                load[h] = pwr
        colour = appl_colours[idx % len(appl_colours)]
        r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
        fig.add_trace(go.Scatter(
            x=hour_lbls,
            y=cumulative + load,
            fill="tonexty",
            fillcolor=f"rgba({r}, {g}, {b}, 0.22)",
            line=dict(color=colour, width=1.5, dash="dot"),
            name=f"🔌 {appl_name}",
            hovertemplate=f"{appl_name}<br>" + "%{x}<br>Load: <b>%{y:.1f} kW</b><extra></extra>",
        ))
        cumulative = cumulative + load

    # Solar generation overlay
    fig.add_trace(go.Scatter(
        x=hour_lbls, y=solar_kw,
        mode="lines",
        line=dict(color="#f7c948", width=3, shape="spline"),
        name="☀ Solar Generation",
        hovertemplate="%{x}<br>Solar: <b>%{y:.1f} kW</b><extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=440,
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=11)),
        xaxis=dict(title="Hour of Day", gridcolor="rgba(255,255,255,0.06)", tickangle=-45),
        yaxis=dict(title="Power (kW)", gridcolor="rgba(255,255,255,0.06)"),
        hovermode="x unified",
        title=dict(text="⚡ 24-Hour Power Balance", font=dict(size=14)),
    )
    return fig


# ===========================================================================
# Sidebar
# ===========================================================================

def render_sidebar() -> tuple:
    """
    Render the full sidebar and return user selections.

    Returns
    -------
    forecast_date, use_lstm, data_mode, selected_city, city_lat, city_lon
    """
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:10px 0 24px;">
            <div style="font-size:2.8rem;">⚡</div>
            <div style="font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#f7c948,#ff9f43);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                SolarGrid
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ---- Data Source Mode ----------------------------------------
        st.markdown("### 🌐 Data Source")
        mode_options = ["🔬 Historical Benchmark"]
        if _HAS_DATA_FETCHER:
            mode_options.append("🌐 Live City Forecast")

        data_mode = st.radio(
            "Select data source",
            options=mode_options,
            index=0,
            help="Historical Benchmark uses synthetic physics-based data. "
                 "Live City Forecast fetches real weather from Open-Meteo.",
        )

        # City selector (only when live mode is chosen)
        selected_city = "Jammu"
        city_lat, city_lon = 32.7266, 74.8570

        if _HAS_DATA_FETCHER and "Live" in data_mode:
            city_names = list(CITY_PRESETS.keys())
            selected_city = st.selectbox(
                "Select City",
                options=city_names,
                index=0,
                help="Fetches real hourly weather data from Open-Meteo for this city.",
            )
            coords = CITY_PRESETS[selected_city]
            city_lat = coords["lat"]
            city_lon = coords["lon"]
            st.caption(f"📍 lat={city_lat}, lon={city_lon}")

        st.divider()

        # ---- Forecast Settings ----------------------------------------
        st.markdown("### 📅 Forecast Settings")
        forecast_date = st.date_input(
            "Forecast Date",
            value=date.today(),
            min_value=date.today() - timedelta(days=30),
            max_value=date.today() + timedelta(days=7),
            help="Day for the 24-hour (00:00–23:00) forecast. "
                 "In Live mode, weather for this day is fetched from Open-Meteo.",
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
            "**Dept:** CSE, MIET Autonomous, Jammu"
        )

        st.divider()
        st.markdown("### ℹ️ Model Status")
        trained = models_are_trained()
        status_color = "#06d6a0" if trained else "#ffc107"
        status_text  = "All Models Trained ✓" if trained else "Using Fallback Mode"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<div style="width:10px;height:10px;border-radius:50%;'
            f'background:{status_color};box-shadow:0 0 8px {status_color};"></div>'
            f'<span style="font-size:0.82rem;">{status_text}</span></div>',
            unsafe_allow_html=True,
        )
        if not trained:
            st.caption("Run `python models/train_models.py` to train the ML models.")

    return forecast_date, use_lstm, data_mode, selected_city, city_lat, city_lon


# ===========================================================================
# Main entry point
# ===========================================================================

def organized_main():
    """Full Phase 5–7 dashboard — entry point (backwards-compatible name)."""

    # ---- Sidebar -------------------------------------------------------
    forecast_date, use_lstm, data_mode, selected_city, city_lat, city_lon = render_sidebar()
    is_live = _HAS_DATA_FETCHER and "Live" in data_mode

    # ---- Page Header ---------------------------------------------------
    st.markdown("""
    <h1 style="font-size:2rem;margin:0 0 16px;font-weight:800;">⚡ SolarGrid Dashboard</h1>
    """, unsafe_allow_html=True)

    # ---- Load data & forecast ------------------------------------------
    hist_df = load_historical()

    if is_live:
        with st.spinner(f"🌐 Fetching live weather for {selected_city} from Open-Meteo…"):
            solar_kw, demand_kw, used_live = load_live_forecast(
                city_lat, city_lon, selected_city, forecast_date, use_lstm
            )
        if used_live:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.markdown(
                f'<div class="live-banner">🌐 Live Weather — Open-Meteo forecast for '
                f'<b>{selected_city}</b> on <b>{forecast_date:%d %b %Y}</b> '
                f'(00:00–23:00 local time) · fetched at {now_str}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="warn-banner">⚠️ Live weather fetch failed — showing '
                'benchmark synthetic forecast instead.</div>',
                unsafe_allow_html=True,
            )
    else:
        with st.spinner("Running AI forecast engine…"):
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

    # Derived quantities used across tabs
    net = solar_kw - demand_kw
    surplus_mask = net > 0
    surplus_kwh  = float(np.maximum(net, 0).sum())
    surplus_hrs  = int(surplus_mask.sum())
    total_solar  = float(solar_kw.sum())
    total_demand = float(demand_kw.sum())
    peak_solar   = float(solar_kw.max())

    # Surplus window
    if surplus_mask.any():
        surplus_start = int(np.argmax(surplus_mask))
        surplus_end   = 23 - int(np.argmax(surplus_mask[::-1])) + 1
    else:
        surplus_start, surplus_end = 10, 14

    metrics = load_model_metrics()

    # ====================================================================
    # TAB LAYOUT — 6 tabs total
    # ====================================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Exploration",
        "🧠 Model Architecture & Evaluation",
        "📈 24-Hour Forecast",
        "⚖ Surplus Analysis",
        "🌿 Carbon & Savings",
        "⚙️ Smart Scheduler",
    ])

    # ====================================================================
    # TAB 1: Data Exploration  (Phase 1 & 2 — verbatim from app_interim.py)
    # ====================================================================
    with tab1:
        st.markdown('<p class="section-header">📊 Historical Data Exploration (Synthetic Dataset)</p>',
                    unsafe_allow_html=True)

        if hist_df.empty:
            st.info("No data available. Run `python data_gen.py` to generate the dataset.")
        else:
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
                pk_sol = hist_df["historical_solar_gen_kw"].max()
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">☀ Peak Solar</div>
                    <div class="kpi-value">{pk_sol:.1f}</div>
                    <div class="kpi-unit">kW recorded</div>
                </div>
                """, unsafe_allow_html=True)
            with k4:
                pk_dem = hist_df["historical_baseline_demand_kw"].max()
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">⚡ Peak Demand</div>
                    <div class="kpi-value green">{pk_dem:.1f}</div>
                    <div class="kpi-unit">kW recorded</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            wc1, wc2 = st.columns(2, gap="large")
            with wc1:
                st.markdown('<p class="section-header">🌦 Weather Parameter Time-Series (30 Days)</p>',
                            unsafe_allow_html=True)
                st.plotly_chart(build_weather_correlation_chart(hist_df),
                                use_container_width=True, config={"displayModeBar": False})
            with wc2:
                st.markdown('<p class="section-header">☀ Historical Solar Generation vs Community Demand</p>',
                            unsafe_allow_html=True)
                st.plotly_chart(build_historical_generation_chart(hist_df),
                                use_container_width=True, config={"displayModeBar": False})

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
    # TAB 2: Model Architecture & Evaluation  (verbatim from app_interim.py)
    # ====================================================================
    with tab2:
        st.markdown('<p class="section-header">🧠 ML Model Architecture & Evaluation</p>',
                    unsafe_allow_html=True)

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
        ens_col, feat_col = st.columns(2, gap="large")
        with ens_col:
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

        with feat_col:
            st.markdown('<p class="section-header">⚙️ Feature Engineering</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="section-header">🎯 Holdout Test-Set Evaluation Metrics</p>',
                    unsafe_allow_html=True)
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
                st.metric("MAE",  f"{sol_m.get('mae',  'N/A')} kW")
                st.metric("RMSE", f"{sol_m.get('rmse', 'N/A')} kW")
                st.metric("MAPE", f"{sol_m.get('mape', 'N/A')} %")
            with ev3:
                st.markdown("##### ⚡ Electricity Demand Model")
                st.metric("MAE",  f"{dem_m.get('mae',  'N/A')} kW")
                st.metric("RMSE", f"{dem_m.get('rmse', 'N/A')} kW")
                st.metric("MAPE", f"{dem_m.get('mape', 'N/A')} %")
            st.caption(
                f"Evaluated on **{metrics.get('test_samples', 'N/A')} holdout test samples** "
                f"({metrics.get('split_ratio', 'N/A')} temporal train/test split)."
            )
        else:
            st.info("Run `python models/train_models.py` to generate evaluation metrics.")

    # ====================================================================
    # TAB 3: 24-Hour Forecast  (verbatim from app_interim.py + live banner)
    # ====================================================================
    with tab3:
        st.markdown('<p class="section-header">📈 24-Hour Solar vs Demand Forecast</p>',
                    unsafe_allow_html=True)

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
        chart_col, table_col = st.columns([2, 1], gap="large")
        with chart_col:
            st.plotly_chart(build_forecast_chart(solar_kw, demand_kw),
                            use_container_width=True, config={"displayModeBar": False},
                            key="tab3_forecast")

            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric("🌤 Solar Surplus Available", f"{surplus_kwh:.1f} kWh",
                          delta=f"{surplus_hrs} surplus hours")
            with sc2:
                net_total = total_solar - total_demand
                st.metric("⚖ Day Net Balance",
                          f"{net_total:+.1f} kWh",
                          delta="Surplus" if net_total >= 0 else "Deficit",
                          delta_color="normal" if net_total >= 0 else "inverse")

        with table_col:
            st.markdown("**📋 Hourly Forecast Data**")
            df_display = pd.DataFrame({
                "Hour":        [f"{h:02d}:00" for h in range(24)],
                "Solar (kW)":  np.round(solar_kw,  2),
                "Demand (kW)": np.round(demand_kw, 2),
                "Net (kW)":    np.round(net,        2),
            })

            def _color_net(val):
                color = "#06d6a0" if val >= 0 else "#ff6b6b"
                return f"color: {color}; font-weight: 600"

            try:
                styled = df_display.style.map(_color_net, subset=["Net (kW)"]).format(precision=2)
            except Exception:
                styled = df_display.style.format(precision=2)

            st.dataframe(styled, use_container_width=True, height=520, hide_index=True)

    # ====================================================================
    # TAB 4: Surplus Analysis
    # ====================================================================
    with tab4:
        st.markdown('<p class="section-header">⚖ Hourly Net Energy Balance (Solar − Demand)</p>',
                    unsafe_allow_html=True)

        left_col, right_col = st.columns([3, 1], gap="large")

        with left_col:
            st.plotly_chart(build_net_energy_chart(solar_kw, demand_kw),
                            use_container_width=True, config={"displayModeBar": False},
                            key="tab4_net")
            st.caption("🟢 Green = solar surplus (solar > demand) · 🔴 Red = grid draw (demand > solar)")

        with right_col:
            if surplus_mask.any():
                st.metric("🎯 Surplus Window", f"{surplus_start:02d}:00 – {surplus_end:02d}:00")
            else:
                st.metric("🎯 Surplus Window", "None")
            st.metric("📈 Surplus Hours", f"{surplus_hrs} h")
            st.metric("☀ Surplus Energy", f"{surplus_kwh:.1f} kWh")
            st.metric("⚡ Grid Deficit Energy", f"{float(np.maximum(-net, 0).sum()):.1f} kWh")

    # ====================================================================
    # TAB 5: 🌿 Carbon & Savings  (NEW — Phase 6)
    # ====================================================================
    with tab5:
        st.markdown('<p class="section-header">🌿 Carbon Impact & Financial Savings</p>',
                    unsafe_allow_html=True)

        if not _HAS_SCHEDULER:
            st.warning(
                "⚠️ `models/scheduler.py` could not be imported. "
                "Carbon & Savings tab requires the scheduler module."
            )
        else:
            # Run scheduler to get savings data
            try:
                sched_result: ScheduleResult = run_scheduler(solar_kw, demand_kw)
                tod_data  = compute_tod_savings(sched_result, solar_kw, demand_kw)
                carbon_data = compute_carbon_savings(sched_result.surplus_used_kwh)

                # Determine weather condition from cloud cover proxy
                avg_cloud = float(np.mean(np.where(solar_kw > 0, 1 - solar_kw / (solar_kw.max() + 1e-6), 1)))
                if avg_cloud < 0.2:
                    weather_cond = "clear"
                elif avg_cloud < 0.45:
                    weather_cond = "partly cloudy"
                elif avg_cloud < 0.70:
                    weather_cond = "cloudy"
                else:
                    weather_cond = "overcast"

                scheduled_appliances = [
                    name for name, hrs in sched_result.schedule.items() if hrs
                ]
                num_scheduled = len(scheduled_appliances)
                rupees_saved  = tod_data["savings_rs"]
                co2_kg        = carbon_data["co2_saved_kg"]
                trees_today   = carbon_data["trees_equivalent"]
                annual_co2_t  = carbon_data["annual_co2_kg"] / 1000.0
                annual_trees  = carbon_data["annual_trees"]

                # ---- AI Summary Box ------------------------------------
                ai_text = generate_ai_summary(
                    weather_cond=weather_cond,
                    peak_start=surplus_start,
                    peak_end=surplus_end,
                    surplus_kwh=sched_result.surplus_used_kwh,
                    num_appliances=num_scheduled,
                    rupees_saved=rupees_saved,
                    co2_kg=co2_kg,
                )

                st.markdown('<p class="section-header">🤖 Ask SolarGrid — AI Summary</p>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="ai-summary-box">{ai_text}</div>',
                    unsafe_allow_html=True,
                )

                # ---- Carbon & Trees KPI Row ----------------------------
                st.markdown('<p class="section-header">🌱 Carbon Impact</p>',
                            unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">🌱 CO₂ Saved Today</div>
                        <div class="kpi-value green">{co2_kg:.2f}</div>
                        <div class="kpi-unit">kg of CO₂ avoided</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">🌳 Trees Equivalent</div>
                        <div class="kpi-value green">{trees_today:.3f}</div>
                        <div class="kpi-unit">tree-days of absorption</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">📅 Annual CO₂ Saving</div>
                        <div class="kpi-value green">{annual_co2_t:.2f}</div>
                        <div class="kpi-unit">tonnes CO₂ / year</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">🌲 Annual Trees Planted</div>
                        <div class="kpi-value green">{annual_trees:.1f}</div>
                        <div class="kpi-unit">tree-equivalents / year</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ---- Financial Savings KPI Row -------------------------
                st.markdown('<p class="section-header">💰 Financial Savings</p>',
                            unsafe_allow_html=True)
                f1, f2, f3, f4 = st.columns(4)
                with f1:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">💰 Daily Savings</div>
                        <div class="kpi-value">₹{rupees_saved:,.0f}</div>
                        <div class="kpi-unit">vs. unmanaged peak tariff</div>
                    </div>
                    """, unsafe_allow_html=True)
                with f2:
                    monthly = rupees_saved * 30
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">📅 Monthly Savings</div>
                        <div class="kpi-value">₹{monthly:,.0f}</div>
                        <div class="kpi-unit">projected (×30 days)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with f3:
                    annual_sav = rupees_saved * 365
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">📈 Annual Savings</div>
                        <div class="kpi-value">₹{annual_sav:,.0f}</div>
                        <div class="kpi-unit">projected (×365 days)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with f4:
                    grid_avoided = sched_result.surplus_used_kwh
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">⚡ Grid Energy Avoided</div>
                        <div class="kpi-value green">{grid_avoided:.1f}</div>
                        <div class="kpi-unit">kWh avoided today</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ---- Tariff breakdown ----------------------------------
                with st.expander("📊 Tariff Cost Breakdown"):
                    tc1, tc2, tc3 = st.columns(3)
                    tc1.metric("Baseline Cost (unmanaged)", f"₹{tod_data['baseline_cost_rs']:,.2f}")
                    tc2.metric("Optimised Cost (scheduled)", f"₹{tod_data['optimized_cost_rs']:,.2f}")
                    tc3.metric("Net Saving", f"₹{tod_data['savings_rs']:,.2f}",
                               delta=f"{tod_data['savings_pct']:.1f}% reduction")

                    st.markdown("""
                    **Tariff bands used (India CEA):**
                    | Period | Hours | Rate |
                    |--------|-------|------|
                    | Peak   | 18:00–21:00 | ₹8.50 / kWh |
                    | Normal | 06:00–17:00, 22:00 | ₹5.50 / kWh |
                    | Off-peak | 00:00–05:00, 23:00 | ₹4.00 / kWh |

                    *Baseline assumes all appliances ran at peak tariff window (worst-case).*
                    """)

            except Exception as exc:
                st.error(f"❌ Failed to compute Carbon & Savings: {exc}")
                st.exception(exc)

    # ====================================================================
    # TAB 6: ⚙️ Smart Scheduler  (NEW — Phase 5)
    # ====================================================================
    with tab6:
        st.markdown('<p class="section-header">⚙️ Smart Heuristic Load Scheduler</p>',
                    unsafe_allow_html=True)

        if not _HAS_SCHEDULER:
            st.warning(
                "⚠️ `models/scheduler.py` could not be imported. "
                "Install the scheduler module to enable this tab."
            )
        else:
            # ---- Appliance Configuration Panel -------------------------
            st.markdown('<p class="section-header">🔧 Appliance Configuration</p>',
                        unsafe_allow_html=True)
            st.caption(
                "Toggle appliances on/off before running the scheduler. "
                "Disabled appliances are excluded from the optimisation."
            )

            enabled_appliances = []
            appl_cols = st.columns(len(DEFAULT_APPLIANCES))
            for col, appl in zip(appl_cols, DEFAULT_APPLIANCES):
                with col:
                    is_enabled = st.toggle(
                        label=appl.name,
                        value=True,
                        key=f"appl_toggle_{appl.name}",
                    )
                    st.caption(
                        f"⚡ {appl.power_kw} kW · "
                        f"⏱ {appl.duration_hours}h · "
                        f"🏁 by {appl.deadline_hour:02d}:00"
                    )
                    if is_enabled:
                        enabled_appliances.append(appl)

            st.markdown("<br>", unsafe_allow_html=True)

            try:
                sched_res_t6: ScheduleResult = run_scheduler(
                    solar_kw, demand_kw, appliances=enabled_appliances
                )

                schedule   = sched_res_t6.schedule
                total_load = sched_res_t6.total_load_per_hour
                appl_pwr   = {a.name: a.power_kw for a in DEFAULT_APPLIANCES}

                # ---- Transformer Safety indicator ----------------------
                PEAK_CAP_KW = 35.0
                max_load    = float(total_load.max())
                is_safe     = max_load <= PEAK_CAP_KW
                if is_safe:
                    st.markdown(
                        f'<div class="safety-ok">✅ <b>Transformer Safe</b> — '
                        f'Peak total load: <b>{max_load:.1f} kW</b> '
                        f'(within {PEAK_CAP_KW:.0f} kW capacity)</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    over_hours = [h for h in range(24) if total_load[h] > PEAK_CAP_KW]
                    st.markdown(
                        f'<div class="safety-warn">⚠️ <b>Transformer Overload Warning</b> — '
                        f'Peak total load: <b>{max_load:.1f} kW</b> exceeds '
                        f'{PEAK_CAP_KW:.0f} kW limit at hours: '
                        f'{[f"{h:02d}:00" for h in over_hours]}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # ---- Schedule result summary ---------------------------
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric("☀ Solar Surplus Used",
                              f"{sched_res_t6.surplus_used_kwh:.2f} kWh")
                with res_col2:
                    st.metric("⚡ Grid Fallback",
                              f"{sched_res_t6.grid_fallback_kwh:.2f} kWh")
                with res_col3:
                    unscheduled = sched_res_t6.unscheduled
                    if unscheduled:
                        st.metric("⚠ Unscheduled", f"{len(unscheduled)} appliance(s)")
                        st.caption(f"Could not schedule: {', '.join(unscheduled)}")
                    else:
                        st.metric("✅ All Scheduled", f"{len(schedule)} appliance(s)")

                st.markdown("<br>", unsafe_allow_html=True)

                # ---- Schedule Gantt Chart ------------------------------
                st.markdown('<p class="section-header">📅 Appliance Schedule Gantt Chart</p>',
                            unsafe_allow_html=True)
                gantt_fig = build_gantt_chart(schedule, solar_kw, demand_kw)
                st.plotly_chart(gantt_fig, use_container_width=True,
                                config={"displayModeBar": False},
                                key="tab6_gantt")

                st.caption(
                    "🟢 Green bars = appliance powered by solar surplus  |  "
                    "🟠 Orange bars = grid draw  |  "
                    "🔴 'Before (Unmanaged)' = baseline with no scheduling (peak morning hours)"
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # ---- Power Balance Chart -------------------------------
                st.markdown('<p class="section-header">⚡ 24-Hour Power Balance</p>',
                            unsafe_allow_html=True)
                power_fig = build_power_balance_chart(
                    solar_kw, demand_kw, schedule, appl_pwr
                )
                st.plotly_chart(power_fig, use_container_width=True,
                                config={"displayModeBar": False},
                                key="tab6_power")

                # ---- Per-hour schedule table ---------------------------
                with st.expander("📋 Hourly Schedule Detail"):
                    sched_rows = []
                    for h in range(24):
                        active = [
                            name for name, hrs in schedule.items() if h in hrs
                        ]
                        sched_rows.append({
                            "Hour": f"{h:02d}:00",
                            "Active Appliances": ", ".join(active) if active else "—",
                            "Total Load (kW)": round(float(total_load[h]), 2),
                            "Solar (kW)": round(float(solar_kw[h]), 2),
                            "Net (kW)": round(float(solar_kw[h] - demand_kw[h]), 2),
                        })
                    sched_df = pd.DataFrame(sched_rows)
                    st.dataframe(sched_df, use_container_width=True, hide_index=True)

            except Exception as exc:
                st.error(f"❌ Scheduler error: {exc}")
                st.exception(exc)


# Alias for backwards compatibility
main = organized_main

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    organized_main()
