"""
SolarGrid — scheduler/heuristic.py
=====================================
Rule-based scheduling engine that maps controllable community appliances to
optimal operating windows within the 24-hour forecast horizon.

Philosophy
----------
Grid savings are maximised when high-draw appliances run **during periods of
solar surplus** (solar generation > baseline demand). The surplus energy would
otherwise be exported to the grid at a lower feed-in tariff, so consuming it
on-site is economically better.

Algorithm (per appliance)
-------------------------
1. Compute net energy = solar_kw[h] - demand_kw[h]  for each hour h.
2. Restrict candidate hours to the allowed operating window (configurable).
3. Use a sliding-window sum to find the contiguous block of `duration` hours
   that maximises total net energy (i.e., best solar surplus coverage).
4. If no all-surplus window exists, fall back to the window with the least
   net deficit (greedy "least grid draw" strategy).

Appliances configured
---------------------
  • Community Water Pump : 3.5 kW, 2 continuous hours
  • EV Charging Station  : 7.4 kW, 3 continuous hours

Returns
-------
A list of dicts, one per appliance, with keys:
    appliance_name, power_kw, duration_hours, start_hour, end_hour,
    start_time_str, end_time_str, surplus_covered_kwh, estimated_saving_rs
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ===========================================================================
# Appliance definition
# ===========================================================================

@dataclass
class Appliance:
    """
    Represents a controllable electrical load in the community.

    Attributes:
        name:           Human-readable display name.
        power_kw:       Rated power draw in kilowatts.
        duration_hours: Minimum continuous run time required (integer hours).
        window_start:   Earliest allowed start hour (0-23, inclusive).
        window_end:     Latest allowed start hour (0-23, inclusive).
                        The appliance can START at any hour in this range,
                        but must FINISH before hour 24.
        priority:       Lower number = scheduled first (useful if power
                        budgets become binding in future versions).
    """
    name:           str
    power_kw:       float
    duration_hours: int
    window_start:   int   = 0
    window_end:     int   = 23
    priority:       int   = 1
    color:          str   = "#00b4d8"   # for Plotly annotations


# ---------------------------------------------------------------------------
# Community appliance registry
# ---------------------------------------------------------------------------
APPLIANCES: list[Appliance] = [
    Appliance(
        name           = "Community Water Pump",
        power_kw       = 3.5,
        duration_hours = 2,
        window_start   = 6,     # Don't run pump before dawn
        window_end     = 20,    # Must finish by 22:00
        priority       = 1,
        color          = "#06d6a0",
    ),
    Appliance(
        name           = "EV Charging Station",
        power_kw       = 7.4,
        duration_hours = 3,
        window_start   = 7,     # EVs typically available from 7 AM
        window_end     = 20,    # Must finish by 23:00
        priority       = 2,
        color          = "#118ab2",
    ),
]

# Electricity rate used to compute monetary savings estimate (INR per kWh)
GRID_RATE_INR_PER_KWH = 8.0


# ===========================================================================
# Core scheduling logic
# ===========================================================================

def _best_window(net_energy: np.ndarray, appliance: Appliance) -> int:
    """
    Find the start hour that maximises total net surplus over the run window.

    Uses a sliding sum of width `duration_hours` restricted to valid hours.

    Args:
        net_energy: Array of shape (24,) with values = solar - demand.
        appliance:  Appliance being scheduled.

    Returns:
        Best start hour (int, 0-23).
    """
    dur   = appliance.duration_hours
    best_start    = appliance.window_start
    best_net_sum  = -np.inf

    # Iterate over all valid start hours
    for h in range(appliance.window_start, appliance.window_end + 1):
        end_h = h + dur
        if end_h > 24:
            break                                     # Would run past midnight
        window_sum = float(net_energy[h:end_h].sum())
        if window_sum > best_net_sum:
            best_net_sum = window_sum
            best_start   = h

    return best_start


def schedule_appliances(
    solar_kw:  np.ndarray,
    demand_kw: np.ndarray,
    appliances: list[Appliance] | None = None,
) -> list[dict]:
    """
    Schedule each appliance to its optimal operating window.

    The net energy array is updated after each appliance is scheduled to
    model the additional load — preventing the same solar surplus from being
    double-counted by multiple appliances.

    Args:
        solar_kw:   24-element array of predicted solar generation.
        demand_kw:  24-element array of predicted baseline demand.
        appliances: List of Appliance objects to schedule. Defaults to
                    the community registry APPLIANCES.

    Returns:
        List of schedule dictionaries, one per appliance.
    """
    if appliances is None:
        appliances = APPLIANCES

    # Sort by priority so high-priority loads claim the best surplus first
    sorted_apps = sorted(appliances, key=lambda a: a.priority)

    # Working copy of net energy — modified as appliances are placed
    net_energy = solar_kw.copy() - demand_kw.copy()

    results = []

    for app in sorted_apps:
        start_h = _best_window(net_energy, app)
        end_h   = start_h + app.duration_hours

        # Clamp to valid range (defensive programming)
        start_h = max(0, min(start_h, 23))
        end_h   = min(end_h, 24)

        # How much of the appliance's consumption is covered by solar surplus?
        total_consumption_kwh = app.power_kw * (end_h - start_h)
        window_net            = net_energy[start_h:end_h]
        surplus_used_kwh      = float(np.sum(np.minimum(window_net, app.power_kw)))
        surplus_used_kwh      = max(0.0, surplus_used_kwh)

        # Monetary saving if surplus_used_kwh avoids grid purchase
        estimated_saving_inr  = surplus_used_kwh * GRID_RATE_INR_PER_KWH

        # Deduct the appliance load from net energy for subsequent appliances
        net_energy[start_h:end_h] -= app.power_kw

        # Determine grid supplementation needed
        grid_draw_kwh = total_consumption_kwh - surplus_used_kwh

        results.append({
            "appliance_name":       app.name,
            "power_kw":             app.power_kw,
            "duration_hours":       end_h - start_h,
            "start_hour":           start_h,
            "end_hour":             end_h,
            "start_time_str":       f"{start_h:02d}:00",
            "end_time_str":         f"{end_h:02d}:00",
            "total_kwh":            round(total_consumption_kwh, 2),
            "surplus_covered_kwh":  round(max(0.0, surplus_used_kwh), 2),
            "grid_draw_kwh":        round(max(0.0, grid_draw_kwh), 2),
            "estimated_saving_inr": round(estimated_saving_inr, 2),
            "color":                app.color,
        })

    return results


# ===========================================================================
# Baseline Comparison (Unmanaged vs AI Optimized)
# ===========================================================================

def compare_with_baseline(solar_kw: np.ndarray, demand_kw: np.ndarray,
                          schedule: list[dict]) -> dict:
    """
    Compare SolarGrid heuristic schedule against a conventional unmanaged baseline.

    Conventional unmanaged behavior:
      - Water Pump: runs at 07:00–09:00 (morning rush, low solar irradiance).
      - EV Charger: runs at 19:00–22:00 (evening peak arrival, zero solar irradiance).

    Returns:
        Dict comparing energy draw, grid reliance, and costs for baseline vs optimized.
    """
    # Fixed conventional operating hours
    baseline_slots = {
        "Community Water Pump": (7, 9),    # 7 AM - 9 AM
        "EV Charging Station":  (19, 22),  # 7 PM - 10 PM
    }

    baseline_grid_kwh = 0.0
    baseline_solar_kwh = 0.0
    baseline_details = []

    net_energy_base = solar_kw.copy() - demand_kw.copy()

    for app in APPLIANCES:
        sh, eh = baseline_slots.get(app.name, (8, 8 + app.duration_hours))
        total_kwh = app.power_kw * (eh - sh)
        window_net = net_energy_base[sh:eh]
        surplus_avail = float(np.sum(np.maximum(0.0, np.minimum(window_net, app.power_kw))))
        grid_draw = max(0.0, total_kwh - surplus_avail)

        baseline_grid_kwh += grid_draw
        baseline_solar_kwh += surplus_avail
        baseline_details.append({
            "appliance_name": app.name,
            "start_time_str": f"{sh:02d}:00",
            "end_time_str":   f"{eh:02d}:00",
            "total_kwh":      total_kwh,
            "grid_draw_kwh":  grid_draw,
            "surplus_kwh":    surplus_avail,
            "cost_inr":       grid_draw * GRID_RATE_INR_PER_KWH,
        })

    # Optimized totals
    opt_grid_kwh  = sum(s["grid_draw_kwh"] for s in schedule)
    opt_solar_kwh = sum(s["surplus_covered_kwh"] for s in schedule)
    total_app_kwh = sum(s["total_kwh"] for s in schedule)

    baseline_cost_inr = baseline_grid_kwh * GRID_RATE_INR_PER_KWH
    opt_cost_inr      = opt_grid_kwh * GRID_RATE_INR_PER_KWH
    kwh_saved         = max(0.0, baseline_grid_kwh - opt_grid_kwh)
    inr_saved         = max(0.0, baseline_cost_inr - opt_cost_inr)
    grid_reduction_pct= (kwh_saved / baseline_grid_kwh * 100) if baseline_grid_kwh > 0 else 0.0

    return {
        "total_app_kwh":       round(total_app_kwh, 2),
        "baseline_grid_kwh":   round(baseline_grid_kwh, 2),
        "baseline_solar_kwh":  round(baseline_solar_kwh, 2),
        "baseline_cost_inr":   round(baseline_cost_inr, 2),
        "optimized_grid_kwh":  round(opt_grid_kwh, 2),
        "optimized_solar_kwh": round(opt_solar_kwh, 2),
        "optimized_cost_inr":  round(opt_cost_inr, 2),
        "kwh_saved":           round(kwh_saved, 2),
        "inr_saved":           round(inr_saved, 2),
        "grid_reduction_pct":  round(grid_reduction_pct, 1),
        "baseline_details":    baseline_details,
    }


# ===========================================================================
# KPI Calculations
# ===========================================================================

def compute_kpis(solar_kw: np.ndarray, demand_kw: np.ndarray,
                 schedule: list[dict]) -> dict:
    """
    Compute high-level key performance indicators for the dashboard.

    Args:
        solar_kw:  24-hour predicted solar array.
        demand_kw: 24-hour predicted demand array.
        schedule:  Output of schedule_appliances().

    Returns:
        Dict with KPI keys and values including baseline comparative metrics.
    """
    peak_solar_kw      = float(solar_kw.max())
    total_demand_kwh   = float(demand_kw.sum())
    total_solar_kwh    = float(solar_kw.sum())

    # Hours where solar > demand (natural surplus)
    net                = solar_kw - demand_kw
    surplus_hours      = int((net > 0).sum())
    surplus_kwh        = float(np.maximum(net, 0).sum())

    # Baseline comparison
    comparison         = compare_with_baseline(solar_kw, demand_kw, schedule)

    # Grid savings estimate: total surplus used by appliances + natural surplus
    total_saving_inr   = sum(s["estimated_saving_inr"] for s in schedule)
    total_sched_kwh    = sum(s["surplus_covered_kwh"]  for s in schedule)

    # Grid savings % = solar energy consumed locally / total demand
    self_consumption_kwh = min(total_solar_kwh, total_demand_kwh)
    grid_savings_pct     = (self_consumption_kwh / total_demand_kwh * 100
                            if total_demand_kwh > 0 else 0.0)

    return {
        "peak_solar_kw":         round(peak_solar_kw, 2),
        "total_demand_kwh":      round(total_demand_kwh, 2),
        "total_solar_kwh":       round(total_solar_kwh, 2),
        "surplus_kwh":           round(surplus_kwh, 2),
        "surplus_hours":         surplus_hours,
        "grid_savings_pct":      round(grid_savings_pct, 1),
        "total_saving_inr":      round(total_saving_inr, 2),
        "total_sched_surplus_kwh": round(total_sched_kwh, 2),
        "baseline_comparison":   comparison,
    }



# ===========================================================================
# Quick self-test
# ===========================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.forecast import get_24hr_forecast

    solar, demand = get_24hr_forecast()
    print("24-Hour Forecast:")
    for h, (s, d) in enumerate(zip(solar, demand)):
        bar = "█" * int(s / 2)
        print(f"  {h:02d}:00  Solar={s:5.1f} kW  Demand={d:5.1f} kW  {bar}")

    print("\nScheduling appliances...")
    sched = schedule_appliances(solar, demand)

    print("\nRecommended Schedule:")
    for s in sched:
        print(f"  {s['appliance_name']:30s} | "
              f"{s['start_time_str']} – {s['end_time_str']} | "
              f"Surplus covered: {s['surplus_covered_kwh']:.1f} kWh | "
              f"Saving: ₹{s['estimated_saving_inr']:.2f}")

    kpis = compute_kpis(solar, demand, sched)
    print("\nKPIs:")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
