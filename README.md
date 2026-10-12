# ⚡ SolarGrid: AI-Enabled Solar Energy Prediction & Smart Load Optimization

> **Autonomous Residential Energy Management System**  
> **Candidate:** Suryansh Gupta (Roll No: 2023a1r116)  
> **Department:** Department of Computer Science and Engineering, Model Institute of Engineering and Technology (Autonomous), Jammu, India (2026)

---

## 📌 Project Overview
SolarGrid is an AI-powered energy management software prototype designed for residential communities with rooftop solar photovoltaic (PV) arrays.

It bridges the operational gap between volatile solar generation and residential electricity demand by:
1. **Physics-informed Meteorological Modeling:** Simulating clear-sky solar irradiance ($W/m^2$), quadratic cloud attenuation, and ambient temperature cycles.
2. **Hybrid Machine Learning Forecasting (Phases 3 & 4):** Combining Deep Learning (**PyTorch Stacked LSTM**) for sequential autocorrelation and **XGBoost Regressor** for non-linear meteorological feature boundaries to predict next-day 24-hour solar generation and community demand.
3. **Priority Heuristic Load Scheduler (Phase 5):** Automatically dispatching flexible heavy community loads (central water pumps, EV chargers) into predicted midday solar surplus windows (10:00 AM – 3:00 PM), maximizing self-consumption and slashing peak grid draw.
4. **Interactive Glassmorphic Dashboard (Phase 6):** Real-time monitoring, financial tariff savings tracking, appliance load control, and model diagnostics.

---

## 📊 Benchmark Model Performance (Holdout Test Set)

| Target Variable | MAE | RMSE | MAPE | Primary Model Architecture |
| :--- | :---: | :---: | :---: | :--- |
| **Solar Generation (kW)** | **1.34 kW** | **3.13 kW** | **9.14%** | Hybrid Ensemble (60% XGBoost + 40% PyTorch LSTM) |
| **Community Demand (kW)** | **0.91 kW** | **2.21 kW** | **6.41%** | Feature-Engineered XGBoost Regressor |

*Ablation analysis: The hybrid ensemble reduces MAE by 15.2% over standalone LSTM and 5.0% over standalone XGBoost.*

---

## 🗂️ Repository Structure

```plaintext
SOLAR_GRID/
├── app.py                       # Complete SolarGrid Dashboard (5 Tabs + Heuristic Scheduler)
├── app_interim.py               # Focused Interim Milestone Review Dashboard (4 Tabs)
├── PROJECT_REPORT.txt           # Comprehensive Final Technical Specification Report
├── PROJECT_MILESTONE_REPORT.txt # Interim Milestone Defense Report (Phases 1–4)
├── data_gen.py                  # Synthetic physics & weather generator
├── solargrid.db                 # Embedded SQLite database (720 hourly records)
├── requirements.txt             # Python dependencies
├── database/
│   ├── __init__.py
│   └── db_manager.py            # SQLite schema & query manager
├── scheduler/
│   ├── __init__.py
│   └── heuristic.py             # Greedy-priority heuristic load optimization engine
└── models/
    ├── __init__.py
    ├── lstm_model.py            # PyTorch stacked LSTM network architecture
    ├── train_models.py          # Model training & holdout validation pipeline
    ├── forecast.py              # 24-hr multi-model inference engine
    └── saved/                   # Serialized weights, scalers & evaluation metrics
        ├── lstm_solar.pt
        ├── xgb_solar.joblib & xgb_solar.json
        ├── xgb_demand.joblib & xgb_demand.json
        ├── scaler_features.pkl, scaler_solar.pkl, scaler_demand.pkl
        ├── feature_cols.json
        └── metrics.json
```

---

## 🚀 Quickstart & Local Execution

### 1. Clone the repository
```bash
git clone https://github.com/suryanshgupta0703-cyber/SOLAR_GRID.git
cd SOLAR_GRID
```

### 2. Set up virtual environment & dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Complete Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` to view the complete system.

*(To view the interim milestone presentation dashboard specifically: `streamlit run app_interim.py`)*.

---

## 🌐 Online Deployment (Streamlit Community Cloud)
1. Fork or push this repository to GitHub (`suryanshgupta0703-cyber/SOLAR_GRID`).
2. Log in at [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **"New app"**, select the `SOLAR_GRID` repository, and set **Main file path** to `app.py`.
4. Click **"Deploy!"** — Streamlit automatically provisions the container, loads the embedded database and pre-trained models, and provides a permanent public `https://...streamlit.app` URL!
