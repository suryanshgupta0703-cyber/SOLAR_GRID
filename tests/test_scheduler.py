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


# ---------------------------------------------------------------------------
# Test 6: Custom power rating scales the scheduled energy
# ---------------------------------------------------------------------------
def test_custom_power_scales_energy():
    from dataclasses import replace
    from models.scheduler import run_scheduler, Appliance
    solar = np.array([0]*6 + [5,10,15,20,22,24,22,20,16,12,8,4] + [0]*6, dtype=float)
    demand = np.full(24, 10.0)
    load = Appliance("Test Load", power_kw=4.0, duration_hours=2, priority=1,
                     interruptible=True, deadline_hour=18, earliest_start=6)
    r1 = run_scheduler(solar, demand, [load])
    r2 = run_scheduler(solar, demand, [replace(load, power_kw=8.0)])
    e1 = r1.surplus_used_kwh + r1.grid_fallback_kwh
    e2 = r2.surplus_used_kwh + r2.grid_fallback_kwh
    assert e1 == pytest.approx(8.0)
    assert e2 == pytest.approx(2 * e1)


# ---------------------------------------------------------------------------
# Test 7: ToD savings honour renamed per-unit instances
# ---------------------------------------------------------------------------
def test_tod_savings_uses_instance_names():
    from models.scheduler import run_scheduler, compute_tod_savings, Appliance
    solar = np.array([0]*6 + [5,10,15,20,22,24,22,20,16,12,8,4] + [0]*6, dtype=float)
    demand = np.full(24, 10.0)
    units = [Appliance(f"EV Charger #{i}", power_kw=11.0, duration_hours=3,
                       priority=1, interruptible=False, deadline_hour=17,
                       earliest_start=6) for i in (1, 2)]
    result = run_scheduler(solar, demand, units)
    savings = compute_tod_savings(result, solar, demand, appliances=units)
    assert savings['baseline_cost_rs'] == pytest.approx(2 * 33.0 * 8.50)
    assert savings['savings_rs'] > 0


# ---------------------------------------------------------------------------
# Test 8: Two units interleave into separate surplus hours
# ---------------------------------------------------------------------------
def test_instances_interleave():
    from models.scheduler import run_scheduler, Appliance
    solar = np.zeros(24)
    solar[10] = 25.0
    solar[11] = 25.0
    demand = np.full(24, 10.0)  # 15 kW surplus at 10:00 and 11:00
    units = [Appliance(f"EV Charger #{i}", power_kw=11.0, duration_hours=1,
                       priority=1, interruptible=False, deadline_hour=17,
                       earliest_start=6) for i in (1, 2)]
    result = run_scheduler(solar, demand, units)
    assert result.schedule["EV Charger #1"] != result.schedule["EV Charger #2"]
    assert result.surplus_used_kwh == pytest.approx(22.0)


# ---------------------------------------------------------------------------
# Test 9: Lowered transformer cap blocks oversized loads
# ---------------------------------------------------------------------------
def test_lower_cap_blocks_oversized_load():
    from models.scheduler import run_scheduler, Appliance
    solar = np.full(24, 40.0)
    demand = np.full(24, 10.0)
    load = Appliance("Big Load", power_kw=20.0, duration_hours=2, priority=1,
                     interruptible=True, deadline_hour=18, earliest_start=6)
    result = run_scheduler(solar, demand, [load], peak_cap_kw=25.0)
    assert result.total_load_per_hour.max() <= 25.0
    assert "Big Load" in result.unscheduled


# ---------------------------------------------------------------------------
# Test 10: Empty appliance list is safe end to end
# ---------------------------------------------------------------------------
def test_empty_appliance_list():
    from models.scheduler import run_scheduler, compute_tod_savings
    solar = np.full(24, 30.0)
    demand = np.full(24, 10.0)
    result = run_scheduler(solar, demand, [])
    assert result.schedule == {}
    assert result.surplus_used_kwh == 0.0
    savings = compute_tod_savings(result, solar, demand, appliances=[])
    assert savings['baseline_cost_rs'] == 0.0
    assert savings['savings_pct'] == 0.0
