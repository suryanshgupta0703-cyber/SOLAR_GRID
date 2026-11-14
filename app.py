"""SolarGrid — Smart Solar Load Optimization Dashboard.

Entry point for the full Phase 1–7 production system.
Run with: streamlit run app.py
"""

try:
    from app_v2 import organized_main as main
except ImportError:
    from app_interim import organized_main as main

if __name__ == "__main__":
    main()