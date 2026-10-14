# SolarGrid — Interim Submission (Phases 1–4)

This folder is the **partial seminar submission** for SolarGrid. It contains the work completed so far: synthetic data generation, SQLite persistence, solar and demand forecasting, model evaluation, and solar-surplus analysis.

It deliberately does **not** include the Phase 5 scheduling engine, Phase 6 final optimization dashboard, or Phase 7 deployment work. Those components belong only to the complete `../solargrid/` project.

## Completed scope

| Phase | Deliverable | Status |
| --- | --- | --- |
| 1 | Dataset preparation and synthetic energy data | Complete |
| 2 | Weather and demand data pipeline with SQLite persistence | Complete |
| 3 | Solar-generation forecasting with LSTM and XGBoost | Complete |
| 4 | Community-demand forecasting and 24-hour surplus analysis | Complete |

See [PROJECT_MILESTONE_REPORT.txt](PROJECT_MILESTONE_REPORT.txt) for the technical explanation, model metrics, and planned roadmap.

## What the demonstration shows

- Historical synthetic weather, solar, and demand data
- LSTM/XGBoost model architecture and holdout evaluation metrics
- 24-hour solar-versus-demand forecast
- Solar-surplus window identification
- The design rationale for the future scheduler (without implementing it)

The model files included in this folder are trained artefacts from this
synthetic dataset. The forecast uses the saved models when present; it has a
physics-based fallback only for the case where those files are missing.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The normal entry point, `app.py`, launches the Phase 1–4 interim dashboard.

## Roadmap after the partial submission

- **Phase 5:** Heuristic load scheduler and priority allocation
- **Phase 6:** Integrated optimization dashboard, appliance controls, and savings KPIs
- **Phase 7:** End-to-end testing, deployment, and documentation
