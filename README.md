# ⚡ SolarGrid: AI-Enabled Solar Energy Prediction & Smart Load Optimization

> **Interim Milestone Package (Phases 1–4: Data Engineering & Machine Learning Forecasting Engine)**  
> **Candidate:** Suryansh Gupta (Roll No: 2023a1r116)  
> **Department:** Department of Computer Science and Engineering, Model Institute of Engineering and Technology (Autonomous), Jammu, India (2026)

---

## 📌 Project Overview
SolarGrid is an AI-powered energy management software prototype designed for residential communities with rooftop solar photovoltaic (PV) arrays.

This repository hosts the **Phase 1–4 Interim Demonstration Package**, proving the mathematical and machine learning foundations before coupling the heuristic scheduler:
1. **Phase 1 & 2:** Physics-informed meteorological data pipeline (solar irradiance, cloud cover attenuation, temperature) and dual-peak residential demand modeling persisted to SQLite.
2. **Phase 3:** Deep Learning (PyTorch stacked LSTM) + Gradient Boosting (XGBoost) hybrid solar yield forecaster.
3. **Phase 4:** Machine learning residential community demand forecaster.
4. **Energy Balance Analytics:** 24-hour predictive curves identifying daytime surplus windows for load shifting.

---

## 📊 Benchmark Model Performance (Holdout Test Set)

| Target Variable | MAE | RMSE | MAPE | Primary Model |
| :--- | :---: | :---: | :---: | :--- |
| **Solar Generation (kW)** | **1.34 kW** | **3.13 kW** | **9.14%** | Hybrid Ensemble (60% XGBoost + 40% PyTorch LSTM) |
| **Community Demand (kW)** | **0.91 kW** | **2.21 kW** | **6.41%** | Feature-Engineered XGBoost Regressor |

*Ablation study: The hybrid ensemble reduces MAE by 15.2% compared to standalone LSTM.*

---

## 🗂️ Repository Structure

```plaintext
SOLAR_GRID/
├── app.py                     # Primary Streamlit entrypoint
├── app_interim.py             # Dedicated 4-tab review dashboard
├── PROJECT_MILESTONE_REPORT.txt # Comprehensive technical academic report
├── data_gen.py                # Synthetic physics & weather generator
├── solargrid.db               # Embedded SQLite database (720 hourly records)
├── requirements.txt           # Python dependencies
├── database/
│   ├── __init__.py
│   └── db_manager.py          # SQLite schema & query manager
└── models/
    ├── __init__.py
    ├── lstm_model.py          # PyTorch stacked LSTM network architecture
    ├── train_models.py        # Model training & validation pipeline
    ├── forecast.py            # 24-hr multi-model inference engine
    └── saved/                 # Serialized weights, scalers & evaluation metrics
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

### 3. Launch the Streamlit Review Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Deploy Online with Streamlit Community Cloud
1. Fork or push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/).
3. Connect your repository and select `app.py`.
4. Click **Deploy** — Streamlit Cloud will automatically build and launch the application!
