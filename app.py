"""
SolarGrid — Interim Milestone Entry Point (app.py)
==================================================
Standard entrypoint aliased to app_interim.py for seamless Streamlit Cloud deployment.
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).parent / "app_interim.py"
    runpy.run_path(str(target), run_name="__main__")
