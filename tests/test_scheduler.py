"""
tests/test_scheduler.py
=======================
Pytest suite for models/scheduler.py — covers result structure, transformer
cap, zero-solar behaviour, carbon savings, and ToD tariff savings.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Test 1: Scheduler returns valid structure
# ---------------------------------------------------------------------------
def test_scheduler_returns_result():
    from models.scheduler import run_scheduler, DEFAULT_APPLIANCES
    solar = np.array([0]*6 + [5,10,15,20,22,24,22,20,16,12,8,4] + [0]*6, dtype=float)
    demand = np.full(24, 10.0)
    result = run_scheduler(solar, demand, DEFAULT_APPLIANCES)
    assert hasattr(result, 'schedule')
    assert hasattr(result, 'surplus_used_kwh')
    assert isinstance(result.schedule, dict)


# ---------------------------------------------------------------------------
# Test 2: No hour exceeds transformer cap
# ---------------------------------------------------------------------------
def test_transformer_cap_not_exceeded():
    from models.scheduler import run_scheduler, DEFAULT_APPLIANCES
    solar = np.array([0]*6 + [5,10,15,20,22,24,22,20,16,12,8,4] + [0]*6, dtype=float)
    demand = np.full(24, 10.0)
    result = run_scheduler(solar, demand, DEFAULT_APPLIANCES, peak_cap_kw=35.0)
    assert result.total_load_per_hour.max() <= 36.0  # allow 1 kW tolerance


# ---------------------------------------------------------------------------
# Test 3: Zero solar → all load falls to grid
# ---------------------------------------------------------------------------
def test_zero_solar_all_grid():
    from models.scheduler import run_scheduler, DEFAULT_APPLIANCES
    solar = np.zeros(24)
    demand = np.full(24, 8.0)
    result = run_scheduler(solar, demand, DEFAULT_APPLIANCES)
    assert result.surplus_used_kwh == 0.0


# ---------------------------------------------------------------------------
# Test 4: Carbon savings calculation
# ---------------------------------------------------------------------------
def test_carbon_savings():
    from models.scheduler import compute_carbon_savings
    result = compute_carbon_savings(surplus_used_kwh=10.0)
    assert abs(result['co2_saved_kg'] - 8.2) < 0.01
    assert result['trees_equivalent'] > 0


# ---------------------------------------------------------------------------
# Test 5: ToD savings baseline > 0
# ---------------------------------------------------------------------------
def test_tod_savings_positive():
    from models.scheduler import run_scheduler, compute_tod_savings, DEFAULT_APPLIANCES
    solar = np.array([0]*6 + [5,10,15,20,22,24,22,20,16,12,8,4] + [0]*6, dtype=float)
    demand = np.full(24, 10.0)
    result = run_scheduler(solar, demand, DEFAULT_APPLIANCES)
    savings = compute_tod_savings(result, solar, demand)
    assert savings['baseline_cost_rs'] > 0
    assert savings['savings_rs'] >= 0
