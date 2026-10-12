"""
SolarGrid — app.py
====================
Main Streamlit dashboard for the AI-Enabled Energy Optimization System.

Layout
------
├── Sidebar        : Controls (forecast date, model blend, appliance config)
├── KPI Row        : Peak solar | Total demand | Grid savings %
├── Forecast Chart : Solar vs Demand with surplus shading + appliance bands
├── Net Energy Bar : Hour-by-hour net power balance
└── Schedule Table : Recommended appliance operating windows

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
# Ensure project root is on sys.path (important when Streamlit changes cwd)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.forecast     import get_24hr_forecast, models_are_trained
from scheduler.heuristic import schedule_appliances, compute_kpis, APPLIANCES
from database.db_manager import get_connection, init_db

# ===========================================================================
# Page configuration — must be first Streamlit call
# ===========================================================================
st.set_page_config(
    page_title    = "SolarGrid — AI Energy Optimizer",
    page_icon     = "⚡",
    layout        = "wide",
    initial_sidebar_state = "expanded",
)

# ===========================================================================
# Global CSS — dark glassmorphism theme
# ===========================================================================
st.markdown("""
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---------- Root palette ---------- */
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

/* ---------- Base ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1528 0%, #111827 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ---------- KPI Cards ---------- */
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
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
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
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ---------- Section headers ---------- */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: var(--text-primary);
    margin: 8px 0 16px 0;
    padding-left: 10px;
    border-left: 3px solid var(--accent-solar);
}

/* ---------- Schedule table ---------- */
.schedule-row {
    display: flex;
    align-items: center;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 10px;
    gap: 20px;
    transition: background 0.15s;
}
.schedule-row:hover { background: rgba(255,255,255,0.07); }
.schedule-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
}
.schedule-name  { font-weight: 600; flex: 2; }
.schedule-time  { font-size: 0.9rem; color: var(--accent-solar); flex: 1.5; font-weight: 700; }
.schedule-badge {
    font-size: 0.75rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    background: rgba(6,214,160,0.15);
    color: var(--accent-green);
    border: 1px solid rgba(6,214,160,0.3);
    flex-shrink: 0;
}
.schedule-savings { font-size: 0.85rem; color: var(--text-muted); flex: 1; }

/* ---------- Warning banner ---------- */
.warn-banner {
    background: rgba(255, 193, 7, 0.10);
    border: 1px solid rgba(255, 193, 7, 0.30);
    border-radius: 10px;
    padding: 12px 16px;
    color: #ffc107;
    font-size: 0.85rem;
    margin-bottom: 20px;
}

/* ---------- Plotly chart containers ---------- */
.element-container iframe { border-radius: 16px; }

/* ---------- Streamlit overrides ---------- */
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

@st.cache_data(ttl=300)  # refresh every 5 minutes
def load_historical() -> pd.DataFrame:
    """Pull the last 7 days of historical data for the trend sparkline."""
    try:
        init_db()
        conn = get_connection()
        df   = pd.read_sql(
            """
            SELECT timestamp, historical_solar_gen_kw, historical_baseline_demand_kw
            FROM historical_energy
            ORDER BY timestamp DESC
            LIMIT 168
            """,
            conn,
        )
        conn.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp")
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

def build_forecast_chart(solar: np.ndarray, demand: np.ndarray,
                         schedule: list[dict]) -> go.Figure:
    """
    Build the main interactive 24-hour forecast chart.

    Features:
    - Solar generation line (amber gradient)
    - Demand line (coral)
    - Green-shaded area wherever solar > demand (surplus)
    - Red-shaded area wherever demand > solar (deficit)
    - Semi-transparent vertical bands for each scheduled appliance
    """
    hours     = list(range(24))
    hour_lbls = [f"{h:02d}:00" for h in hours]

    fig = go.Figure()

    # ---- Surplus fill (solar > demand) ---------------------------------
    surplus_solar  = np.where(solar >= demand, solar, demand)
    surplus_demand = demand.copy()

    fig.add_trace(go.Scatter(
        x    = hour_lbls + hour_lbls[::-1],
        y    = list(surplus_solar) + list(surplus_demand[::-1]),
        fill = "toself",
        fillcolor = "rgba(6, 214, 160, 0.18)",
        line      = dict(color="rgba(0,0,0,0)"),
        name      = "Solar Surplus",
        showlegend= True,
        hoverinfo = "skip",
    ))

    # ---- Deficit fill (demand > solar) ---------------------------------
    deficit_demand = np.where(demand > solar, demand, solar)
    deficit_solar  = solar.copy()

    fig.add_trace(go.Scatter(
        x    = hour_lbls + hour_lbls[::-1],
        y    = list(deficit_demand) + list(deficit_solar[::-1]),
        fill = "toself",
        fillcolor = "rgba(255, 107, 107, 0.12)",
        line      = dict(color="rgba(0,0,0,0)"),
        name      = "Grid Draw Zone",
        showlegend= True,
        hoverinfo = "skip",
    ))

    # ---- Solar generation line -----------------------------------------
    fig.add_trace(go.Scatter(
        x         = hour_lbls,
        y         = solar,
        name      = "☀ Solar Generation",
        mode      = "lines+markers",
        line      = dict(color="#f7c948", width=3, shape="spline"),
        marker    = dict(size=6, color="#f7c948", symbol="circle"),
        hovertemplate = "%{x}<br>Solar: <b>%{y:.1f} kW</b><extra></extra>",
    ))

    # ---- Demand line ---------------------------------------------------
    fig.add_trace(go.Scatter(
        x         = hour_lbls,
        y         = demand,
        name      = "⚡ Baseline Demand",
        mode      = "lines+markers",
        line      = dict(color="#ff6b6b", width=3, shape="spline"),
        marker    = dict(size=6, color="#ff6b6b", symbol="diamond"),
        hovertemplate = "%{x}<br>Demand: <b>%{y:.1f} kW</b><extra></extra>",
    ))

    # ---- Appliance operating bands -------------------------------------
    y_max = max(solar.max(), demand.max()) * 1.15
    for sched in schedule:
        sh, eh = sched["start_hour"], sched["end_hour"]
        color  = sched["color"]
        # Band as shape
        fig.add_vrect(
            x0          = hour_lbls[sh],
            x1          = hour_lbls[min(eh, 23)],
            fillcolor   = color,
            opacity     = 0.12,
            layer       = "below",
            line_width  = 0,
        )
        # Label at top of band
        mid_hour = hour_lbls[(sh + eh) // 2]
        fig.add_annotation(
            x         = mid_hour,
            y         = y_max * 0.96,
            text      = f"<b>{sched['appliance_name'].split()[0]}</b><br>"
                        f"{sched['start_time_str']}–{sched['end_time_str']}",
            showarrow = False,
            font      = dict(size=10, color=color),
            align     = "center",
            bgcolor   = "rgba(10,15,30,0.75)",
            bordercolor = color,
            borderwidth = 1,
            borderpad   = 4,
        )

    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(family="Inter", color="#e8eaf0"),
        legend        = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.02,
            xanchor     = "left",
            x           = 0,
            bgcolor     = "rgba(0,0,0,0)",
            font        = dict(size=12),
        ),
        xaxis = dict(
            title          = "Hour of Day",
            showgrid       = True,
            gridcolor      = "rgba(255,255,255,0.06)",
            zeroline       = False,
            tickfont       = dict(size=11),
        ),
        yaxis = dict(
            title          = "Power (kW)",
            showgrid       = True,
            gridcolor      = "rgba(255,255,255,0.06)",
            zeroline       = False,
            tickfont       = dict(size=11),
            range          = [0, y_max],
        ),
        hovermode    = "x unified",
        margin       = dict(l=10, r=10, t=30, b=10),
        height       = 420,
    )

    return fig


def build_net_energy_chart(solar: np.ndarray, demand: np.ndarray) -> go.Figure:
    """
    Horizontal bar chart showing net energy balance per hour.
    Green = surplus, Red = deficit.
    """
    net       = solar - demand
    hour_lbls = [f"{h:02d}:00" for h in range(24)]
    colors    = ["#06d6a0" if v >= 0 else "#ff6b6b" for v in net]

    fig = go.Figure(go.Bar(
        x         = net,
        y         = hour_lbls,
        orientation = "h",
        marker_color= colors,
        marker_line = dict(width=0),
        hovertemplate = "<b>%{y}</b><br>Net: %{x:.1f} kW<extra></extra>",
        name      = "Net Energy",
    ))

    fig.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=1)

    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(family="Inter", color="#e8eaf0"),
        xaxis = dict(
            title     = "Net Power (kW)  ← Grid Draw | Surplus →",
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.06)",
            zeroline  = False,
        ),
        yaxis = dict(
            showgrid  = False,
            tickfont  = dict(size=10),
            autorange = "reversed",
        ),
        margin = dict(l=10, r=10, t=10, b=10),
        height = 500,
        showlegend = False,
    )
    return fig


def build_historical_sparkline(hist_df: pd.DataFrame) -> go.Figure | None:
    """Build a 7-day historical trend mini-chart."""
    if hist_df.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": False}]])

    fig.add_trace(go.Scatter(
        x    = hist_df["timestamp"],
        y    = hist_df["historical_solar_gen_kw"],
        name = "Historical Solar",
        line = dict(color="#f7c948", width=1.5),
        fill = "tozeroy",
        fillcolor = "rgba(247,201,72,0.08)",
        hovertemplate = "%{x|%b %d %H:%M}<br>Solar: %{y:.1f} kW<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x    = hist_df["timestamp"],
        y    = hist_df["historical_baseline_demand_kw"],
        name = "Historical Demand",
        line = dict(color="#ff6b6b", width=1.5),
        hovertemplate = "%{x|%b %d %H:%M}<br>Demand: %{y:.1f} kW<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(family="Inter", color="#e8eaf0", size=11),
        legend        = dict(
            orientation="h", yanchor="top", y=1.12, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis = dict(showgrid=False, showticklabels=True, tickformat="%b %d"),
        yaxis = dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                     title="kW"),
        margin  = dict(l=0, r=0, t=10, b=0),
        height  = 220,
        hovermode = "x unified",
    )
    return fig


# ===========================================================================
# Schedule table renderer
# ===========================================================================

def render_schedule_table(schedule: list[dict]) -> None:
    """
    Render a visually rich schedule as HTML cards instead of a plain table.
    """
    st.markdown('<p class="section-header">🗓 Recommended Appliance Schedule</p>',
                unsafe_allow_html=True)

    for item in schedule:
        color = item["color"]
        grid_draw = item["grid_draw_kwh"]
        grid_badge = (f'<span style="background:rgba(255,107,107,0.15);color:#ff6b6b;'
                      f'border:1px solid rgba(255,107,107,0.3);padding:2px 8px;'
                      f'border-radius:12px;font-size:0.72rem;font-weight:600;">'
                      f'Grid: {grid_draw:.1f} kWh</span>'
                      if grid_draw > 0.1 else
                      '<span style="background:rgba(6,214,160,0.15);color:#06d6a0;'
                      'border:1px solid rgba(6,214,160,0.3);padding:2px 8px;'
                      'border-radius:12px;font-size:0.72rem;font-weight:600;">'
                      '100% Solar</span>')

        st.markdown(f"""
        <div class="schedule-row">
            <div class="schedule-dot" style="background:{color};box-shadow:0 0 8px {color};"></div>
            <div class="schedule-name">{item['appliance_name']}</div>
            <div class="schedule-time">⏰ {item['start_time_str']} – {item['end_time_str']}</div>
            <div style="flex:1;font-size:0.82rem;color:#8892a4;">
                {item['power_kw']} kW · {item['duration_hours']}h · {item['total_kwh']} kWh
            </div>
            {grid_badge}
            <div class="schedule-savings" style="text-align:right;">
                💰 ₹{item['estimated_saving_inr']:.0f} saved
            </div>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# Sidebar
# ===========================================================================

def render_sidebar() -> tuple[date, bool]:
    """Render sidebar controls and return user selections."""
    with st.sidebar:
        # Logo / title
        st.markdown("""
        <div style="text-align:center;padding:10px 0 24px;">
            <div style="font-size:2.8rem;">⚡</div>
            <div style="font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#f7c948,#ff9f43);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                SolarGrid
            </div>
            <div style="font-size:0.72rem;color:#8892a4;letter-spacing:0.12em;
                        text-transform:uppercase;">AI Energy Optimizer</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("### 📅 Forecast Settings")
        forecast_date = st.date_input(
            "Forecast Date",
            value      = date.today(),
            min_value  = date.today() - timedelta(days=30),
            max_value  = date.today() + timedelta(days=7),
            help       = "Select the day for the 24-hour energy forecast",
        )

        use_lstm = st.toggle(
            "Enable LSTM Ensemble",
            value = True,
            help  = "Blends LSTM (temporal) + XGBoost (feature-based) predictions. "
                    "Disable for faster XGBoost-only mode.",
        )

        st.divider()
        st.markdown("### ⚙️ Grid Settings")
        grid_rate = st.slider(
            "Grid Tariff (₹/kWh)",
            min_value = 4.0,
            max_value = 15.0,
            value     = 8.0,
            step      = 0.5,
        )

        st.divider()
        st.markdown("### ℹ️ System Info")
        trained = models_are_trained()
        status_color  = "#06d6a0" if trained else "#ffc107"
        status_text   = "Models Trained ✓" if trained else "Using Fallback Mode"
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
        st.caption("© 2024 SolarGrid · v1.0.0")

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
            ⚡ SolarGrid Energy Dashboard
        </h1>
        <p style="color:#8892a4;margin:4px 0 0;font-size:0.92rem;">
            AI-powered solar forecasting & smart appliance scheduling for residential communities
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Load & compute ------------------------------------------------
    with st.spinner("Running AI forecast engine..."):
        solar_kw, demand_kw = load_forecast(forecast_date, use_lstm)

    schedule = schedule_appliances(solar_kw, demand_kw)
    kpis     = compute_kpis(solar_kw, demand_kw, schedule)

    # ---- Training status warning ---------------------------------------
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
    # KPI Row
    # ====================================================================
    st.markdown('<p class="section-header">📊 Key Performance Indicators</p>',
                unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">☀ Peak Solar Output</div>
            <div class="kpi-value">{kpis['peak_solar_kw']}</div>
            <div class="kpi-unit">kilowatts</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">⚡ Total Day Demand</div>
            <div class="kpi-value blue">{kpis['total_demand_kwh']}</div>
            <div class="kpi-unit">kWh consumed</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🌿 Grid Savings</div>
            <div class="kpi-value green">{kpis['grid_savings_pct']}%</div>
            <div class="kpi-unit">self-consumption ratio</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">💰 Estimated Savings</div>
            <div class="kpi-value green">₹{kpis['total_saving_inr']:.0f}</div>
            <div class="kpi-unit">from scheduled loads</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ====================================================================
    # Main Forecast Chart
    # ====================================================================
    st.markdown('<p class="section-header">📈 24-Hour Solar vs Demand Forecast</p>',
                unsafe_allow_html=True)

    forecast_fig = build_forecast_chart(solar_kw, demand_kw, schedule)
    st.plotly_chart(forecast_fig, use_container_width=True, config={"displayModeBar": False})

    # ---- Surplus summary metrics beneath chart -------------------------
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("☀ Total Solar Generated",  f"{kpis['total_solar_kwh']:.1f} kWh")
    with sc2:
        st.metric("🌤 Solar Surplus Available", f"{kpis['surplus_kwh']:.1f} kWh",
                  delta=f"{kpis['surplus_hours']} surplus hours")
    with sc3:
        net_total = kpis["total_solar_kwh"] - kpis["total_demand_kwh"]
        st.metric("⚖ Day Net Balance",
                  f"{net_total:+.1f} kWh",
                  delta="Surplus" if net_total >= 0 else "Deficit",
                  delta_color="normal" if net_total >= 0 else "inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # ====================================================================
    # Schedule + Net Energy (side by side)
    # ====================================================================
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        render_schedule_table(schedule)

        # Data table (expandable)
        with st.expander("📋 View Raw Hourly Forecast Data"):
            df_display = pd.DataFrame({
                "Hour":       [f"{h:02d}:00" for h in range(24)],
                "Solar (kW)": np.round(solar_kw, 2),
                "Demand (kW)": np.round(demand_kw, 2),
                "Net (kW)":    np.round(solar_kw - demand_kw, 2),
            })
            def _color_net(val):
                """Color Net (kW) cells green for surplus, red for deficit."""
                color = "#06d6a0" if val >= 0 else "#ff6b6b"
                return f"color: {color}; font-weight: 600"

            try:
                styled = (
                    df_display.style
                    .map(_color_net, subset=["Net (kW)"])
                    .format(precision=2)
                )
            except Exception:
                styled = df_display.style.format(precision=2)

            st.dataframe(
                styled,
                use_container_width=True,
                height=350,
            )

    with right_col:
        st.markdown('<p class="section-header">⚖ Hourly Net Energy Balance</p>',
                    unsafe_allow_html=True)
        net_fig = build_net_energy_chart(solar_kw, demand_kw)
        st.plotly_chart(net_fig, use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ====================================================================
    # Benchmark Evaluation & Baseline Comparison (PDF Workflow Step 6)
    # ====================================================================
    st.markdown('<p class="section-header">🔬 Evaluation & Benchmark Comparison (PDF Step 6)</p>',
                unsafe_allow_html=True)

    ev_col1, ev_col2 = st.columns(2, gap="large")

    base_cmp = kpis.get("baseline_comparison", {})
    with ev_col1:
        st.markdown("#### ⚡ Heuristic Optimizer vs Unmanaged Baseline")
        st.caption(
            "Comparing AI solar-surplus scheduling against uncoordinated fixed-hour operations "
            "(Water Pump @ 07:00 morning peak, EV Charger @ 19:00 evening peak)."
        )
        if base_cmp:
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Unmanaged Grid Draw", f"{base_cmp.get('baseline_grid_kwh', 0):.1f} kWh",
                          delta=f"Cost: ₹{base_cmp.get('baseline_cost_inr', 0):.0f}", delta_color="inverse")
            with bc2:
                st.metric("SolarGrid Grid Draw", f"{base_cmp.get('optimized_grid_kwh', 0):.1f} kWh",
                          delta=f"Cost: ₹{base_cmp.get('optimized_cost_inr', 0):.0f}", delta_color="normal")
            with bc3:
                st.metric("Grid Draw Cut", f"-{base_cmp.get('grid_reduction_pct', 0):.1f}%",
                          delta=f"₹{base_cmp.get('inr_saved', 0):.0f} saved/day", delta_color="normal")

    with ev_col2:
        st.markdown("#### 🎯 ML Forecast Accuracy (Holdout Test Set)")
        st.caption(
            "Evaluation metrics (**MAE**, **RMSE**, and **MAPE**) on holdout 20% unseen test data "
            "as specified in system architecture evaluation."
        )
        metrics = load_model_metrics()
        if metrics:
            sol_m = metrics.get("solar", {})
            dem_m = metrics.get("demand", {})
            m1, m2 = st.columns(2)
            with m1:
                st.markdown("**☀ Solar Generation Model**")
                st.caption(f"• **MAE:** {sol_m.get('mae', 'N/A')} kW<br>"
                           f"• **RMSE:** {sol_m.get('rmse', 'N/A')} kW<br>"
                           f"• **MAPE:** {sol_m.get('mape', 'N/A')}%", unsafe_allow_html=True)
            with m2:
                st.markdown("**⚡ Electricity Demand Model**")
                st.caption(f"• **MAE:** {dem_m.get('mae', 'N/A')} kW<br>"
                           f"• **RMSE:** {dem_m.get('rmse', 'N/A')} kW<br>"
                           f"• **MAPE:** {dem_m.get('mape', 'N/A')}%", unsafe_allow_html=True)
        else:
            st.caption("Run `python models/train_models.py` to generate holdout evaluation metrics.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ====================================================================
    # Historical Trend (last 7 days)
    # ====================================================================
    st.markdown('<p class="section-header">📅 Historical Trend (Last 7 Days)</p>',
                unsafe_allow_html=True)

    hist_df  = load_historical()
    hist_fig = build_historical_sparkline(hist_df)

    if hist_fig:
        st.plotly_chart(hist_fig, use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.info("Historical data not available. Run `python data_gen.py` to generate it.")

    # ====================================================================
    # Footer
    # ====================================================================
    st.markdown("""
    <div style="text-align:center;padding:30px 0 10px;color:#4a5568;font-size:0.78rem;">
        SolarGrid v1.0 · Built with Streamlit + PyTorch + XGBoost · AI Energy Optimization<br>
        Solar predictions are illustrative. For production use, integrate a real weather API.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
