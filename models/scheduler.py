"""
models/scheduler.py
===================
Phase 5 — Heuristic Load Scheduler for SolarGrid.

Implements a greedy priority bin-packing algorithm that shifts flexible
community loads (appliances) into available solar-surplus windows, minimising
grid draw and time-of-day electricity costs.

Public API
----------
- Appliance          : dataclass describing a shiftable load
- DEFAULT_APPLIANCES : sensible defaults for the Jammu pilot site
- ScheduleResult     : dataclass returned by run_scheduler()
- run_scheduler()    : core scheduling function
- compute_tod_savings()  : Time-of-Day tariff cost/savings analysis
- compute_carbon_savings(): CO₂ displacement metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class Appliance:
    """
    Represents a shiftable electrical load in the community micro-grid.

    Attributes
    ----------
    name : str
        Human-readable appliance identifier.
    power_kw : float
        Rated power draw while running (kW).
    duration_hours : int
        Number of hours the appliance must run in total.
    priority : int
        Scheduling priority — 1 is highest (served first).
    interruptible : bool
        If True the runtime may be spread across non-consecutive hours.
        If False the appliance must run as one continuous block.
    deadline_hour : int
        Latest hour by which the appliance must have *finished* (0-23).
        The last start-slot for a block is therefore deadline_hour - duration_hours.
    earliest_start : int
        Earliest hour at which the appliance may begin (0-23).
    """

    name: str
    power_kw: float
    duration_hours: int
    priority: int
    interruptible: bool
    deadline_hour: int
    earliest_start: int


# Default pilot-site appliances (Jammu community micro-grid)
DEFAULT_APPLIANCES: List[Appliance] = [
    Appliance(
        name="EV Charging Fleet",
        power_kw=11.0,
        duration_hours=3,
        priority=1,
        interruptible=False,
        deadline_hour=17,
        earliest_start=6,
    ),
    Appliance(
        name="Community Water Pump",
        power_kw=5.0,
        duration_hours=2,
        priority=2,
        interruptible=True,
        deadline_hour=18,
        earliest_start=6,
    ),
    Appliance(
        name="HVAC Pre-cooling",
        power_kw=3.5,
        duration_hours=2,
        priority=3,
        interruptible=True,
        deadline_hour=16,
        earliest_start=9,
    ),
]


@dataclass
class ScheduleResult:
    """
    Output of :func:`run_scheduler`.

    Attributes
    ----------
    schedule : dict[str, list[int]]
        Maps each appliance name to the list of hours (0-23) it is active.
    surplus_used_kwh : float
        Total energy (kWh) sourced from solar surplus by all scheduled loads.
    grid_fallback_kwh : float
        Total energy (kWh) that had to be drawn from the grid because solar
        surplus was insufficient for some/all assigned slots.
    total_load_per_hour : np.ndarray
        Shape (24,) array of combined load (base demand + all scheduled loads)
        for every hour of the day.
    unscheduled : list[str]
        Appliance names that could not be scheduled at all (no feasible slot).
    """

    schedule: dict
    surplus_used_kwh: float
    grid_fallback_kwh: float
    total_load_per_hour: np.ndarray
    unscheduled: list


# ---------------------------------------------------------------------------
# Tariff helpers
# ---------------------------------------------------------------------------

def _tariff_for_hour(hour: int) -> float:
    """
    Return the Time-of-Day electricity tariff (₹/kWh) for a given hour.

    Tariff bands (India-inspired):

    - Peak     18-21 h  → ₹8.50 / kWh
    - Normal    6-17 h and 22 h → ₹5.50 / kWh
    - Off-peak  0- 5 h and 23 h → ₹4.00 / kWh

    Parameters
    ----------
    hour : int
        Hour of the day in [0, 23].

    Returns
    -------
    float
        Applicable tariff in ₹ per kWh.
    """
    if 18 <= hour <= 21:
        return 8.50
    if (6 <= hour <= 17) or hour == 22:
        return 5.50
    # hours 0-5 and 23
    return 4.00


# ---------------------------------------------------------------------------
# Core scheduler
# ---------------------------------------------------------------------------


def run_scheduler(
    solar_kw: np.ndarray,
    demand_kw: np.ndarray,
    appliances: Optional[List[Appliance]] = None,
    peak_cap_kw: float = 35.0,
) -> ScheduleResult:
    """
    Greedy priority bin-packing scheduler.

    Shifts flexible community loads into solar-surplus windows to minimise
    grid import and time-of-day electricity costs.

    Algorithm
    ---------
    1. Compute hourly surplus: ``surplus[t] = solar_kw[t] - demand_kw[t]``.
    2. Sort appliances by priority ascending (priority 1 served first).
    3. For each appliance:

       a. Identify *candidate* hours inside [earliest_start, deadline_hour).
       b. **Non-interruptible**: Find the consecutive block of
          ``duration_hours`` that maximises total surplus consumed while
          keeping every hour below ``peak_cap_kw``.  If no pure-surplus block
          exists, fall back to the cheapest-tariff block that fits the capacity
          constraint.
       c. **Interruptible**: Greedily pick individual hours with the most
          remaining surplus first, within the deadline window, respecting the
          capacity cap.
       d. Update the running ``total_load`` array and reduce remaining surplus
          accordingly.

    4. Return a :class:`ScheduleResult`.

    Parameters
    ----------
    solar_kw : np.ndarray
        Hourly solar generation forecast, shape (24,), in kW.
    demand_kw : np.ndarray
        Hourly baseline demand forecast, shape (24,), in kW.
    appliances : list[Appliance] or None
        Appliances to schedule.  Defaults to :data:`DEFAULT_APPLIANCES`.
    peak_cap_kw : float
        Hard upper limit on total load per hour (kW).  Defaults to 35.0 kW.

    Returns
    -------
    ScheduleResult
        Fully populated scheduling result.

    Raises
    ------
    ValueError
        If ``solar_kw`` or ``demand_kw`` do not have exactly 24 elements.
    """
    if len(solar_kw) != 24 or len(demand_kw) != 24:
        raise ValueError("solar_kw and demand_kw must each have exactly 24 elements.")

    if appliances is None:
        appliances = DEFAULT_APPLIANCES

    # Working copies — modified as loads are assigned
    surplus = solar_kw.astype(float) - demand_kw.astype(float)
    total_load = demand_kw.astype(float).copy()

    schedule: dict[str, list[int]] = {}
    unscheduled: list[str] = []
    surplus_used_kwh: float = 0.0
    grid_fallback_kwh: float = 0.0

    # Sort by priority (1 = highest)
    sorted_appliances = sorted(appliances, key=lambda a: a.priority)

    for appl in sorted_appliances:
        name = appl.name
        pwr = appl.power_kw
        dur = appl.duration_hours

        # Window of valid hours for this appliance
        win_start = appl.earliest_start
        win_end = appl.deadline_hour  # exclusive: must FINISH by deadline_hour

        # Feasibility check
        if win_end - win_start < dur:
            unscheduled.append(name)
            schedule[name] = []
            continue

        assigned_hours: list[int] = []

        if not appl.interruptible:
            # ----------------------------------------------------------------
            # Non-interruptible: find best consecutive block of `dur` hours
            # ----------------------------------------------------------------
            # Latest start so the block ends at or before deadline_hour
            latest_start = win_end - dur  # block occupies [start, start+dur)

            best_start: Optional[int] = None
            best_surplus: float = -np.inf

            # Pass 1: prefer blocks where ALL hours have positive net surplus
            for s in range(win_start, latest_start + 1):
                block = range(s, s + dur)
                # Check capacity constraint
                if any(total_load[t] + pwr > peak_cap_kw for t in block):
                    continue
                block_surplus = sum(surplus[t] for t in block)
                if all(surplus[t] >= pwr for t in block):
                    # Fully solar-covered block
                    if block_surplus > best_surplus:
                        best_surplus = block_surplus
                        best_start = s

            # Pass 2: if no fully-covered block, find cheapest valid slot
            if best_start is None:
                best_cost: float = np.inf
                for s in range(win_start, latest_start + 1):
                    block = range(s, s + dur)
                    if any(total_load[t] + pwr > peak_cap_kw for t in block):
                        continue
                    cost = sum(_tariff_for_hour(t) for t in block)
                    if cost < best_cost:
                        best_cost = cost
                        best_start = s

            if best_start is None:
                # Truly infeasible
                unscheduled.append(name)
                schedule[name] = []
                continue

            # Commit the block
            for t in range(best_start, best_start + dur):
                assigned_hours.append(t)
                solar_contrib = max(0.0, min(pwr, surplus[t]))
                grid_contrib = pwr - solar_contrib
                surplus_used_kwh += solar_contrib
                grid_fallback_kwh += grid_contrib
                surplus[t] -= solar_contrib
                total_load[t] += pwr

        else:
            # ----------------------------------------------------------------
            # Interruptible: greedy pick of best individual surplus hours
            # ----------------------------------------------------------------
            # Candidate hours within the window that don't exceed peak cap
            candidate = [
                t for t in range(win_start, win_end)
                if total_load[t] + pwr <= peak_cap_kw
            ]
            # Sort by surplus descending (maximise solar use), break ties by hour
            candidate.sort(key=lambda t: (-surplus[t], t))

            if len(candidate) < dur:
                unscheduled.append(name)
                schedule[name] = []
                continue

            chosen = candidate[:dur]
            for t in chosen:
                assigned_hours.append(t)
                solar_contrib = max(0.0, min(pwr, surplus[t]))
                grid_contrib = pwr - solar_contrib
                surplus_used_kwh += solar_contrib
                grid_fallback_kwh += grid_contrib
                surplus[t] -= solar_contrib
                total_load[t] += pwr

        schedule[name] = sorted(assigned_hours)

    return ScheduleResult(
        schedule=schedule,
        surplus_used_kwh=round(surplus_used_kwh, 4),
        grid_fallback_kwh=round(grid_fallback_kwh, 4),
        total_load_per_hour=total_load,
        unscheduled=unscheduled,
    )


# ---------------------------------------------------------------------------
# Time-of-Day cost / savings analysis
# ---------------------------------------------------------------------------


def compute_tod_savings(
    schedule_result: ScheduleResult,
    solar_kw: np.ndarray,
    demand_kw: np.ndarray,
    appliances: Optional[List[Appliance]] = None,
) -> dict:
    """
    Compute Time-of-Day (ToD) tariff cost savings versus unmanaged operation.

    Assumes the worst-case baseline is that every appliance runs entirely
    during the peak tariff window (18-21 h), which is the default pattern when
    there is no active demand-shifting.

    Tariff bands used:

    =====================  =================  ============
    Period                 Hours              Rate (₹/kWh)
    =====================  =================  ============
    Peak                   18-21 h            ₹8.50
    Normal                 6-17 h, 22 h       ₹5.50
    Off-peak               0-5 h, 23 h        ₹4.00
    =====================  =================  ============

    Parameters
    ----------
    schedule_result : ScheduleResult
        Output of :func:`run_scheduler`.
    solar_kw : np.ndarray
        Hourly solar generation used to run the scheduler, shape (24,).
    demand_kw : np.ndarray
        Hourly baseline demand used to run the scheduler, shape (24,).
    appliances : list[Appliance] or None
        The same appliance list passed to :func:`run_scheduler`, so custom
        power ratings and per-unit instances are costed correctly.
        Defaults to :data:`DEFAULT_APPLIANCES`.

    Returns
    -------
    dict with keys:
        - ``baseline_cost_rs``  : float — worst-case unmanaged cost (₹)
        - ``optimized_cost_rs`` : float — scheduled cost (₹)
        - ``savings_rs``        : float — absolute saving (₹)
        - ``savings_pct``       : float — percentage saving vs baseline
    """
    if appliances is None:
        appliances = DEFAULT_APPLIANCES
    power_lookup = {a.name: a.power_kw for a in appliances}

    # Baseline cost: all scheduled loads run at peak tariff (₹8.50)
    PEAK_TARIFF = 8.50
    baseline_cost_rs = 0.0
    optimized_cost_rs = 0.0

    for appl_name, hours in schedule_result.schedule.items():
        if not hours:
            continue
        pwr = power_lookup.get(appl_name, 0.0)
        energy_kwh = pwr * len(hours)
        baseline_cost_rs += energy_kwh * PEAK_TARIFF
        for h in hours:
            optimized_cost_rs += pwr * _tariff_for_hour(h)

    savings_rs = baseline_cost_rs - optimized_cost_rs
    savings_pct = (savings_rs / baseline_cost_rs * 100.0) if baseline_cost_rs > 0 else 0.0

    return {
        "baseline_cost_rs": round(baseline_cost_rs, 2),
        "optimized_cost_rs": round(optimized_cost_rs, 2),
        "savings_rs": round(savings_rs, 2),
        "savings_pct": round(savings_pct, 2),
    }


# ---------------------------------------------------------------------------
# Carbon / environmental impact
# ---------------------------------------------------------------------------


def compute_carbon_savings(surplus_used_kwh: float) -> dict:
    """
    Estimate CO₂ displacement from using solar surplus instead of grid power.

    Uses the Central Electricity Authority (CEA) India grid carbon intensity
    of **0.82 kg CO₂ / kWh** and an annual CO₂ absorption of **21.77 kg per
    mature tree** (Indian Forest Research Institute estimate).

    Parameters
    ----------
    surplus_used_kwh : float
        Total kWh sourced from solar surplus (from :attr:`ScheduleResult.surplus_used_kwh`).

    Returns
    -------
    dict with keys:
        - ``co2_saved_kg``    : float — CO₂ avoided today (kg)
        - ``trees_equivalent``: float — equivalent tree-days of absorption
        - ``annual_co2_kg``   : float — projected annual CO₂ saving (kg)
        - ``annual_trees``    : float — projected annual tree-equivalents
    """
    CEA_INTENSITY = 0.82          # kg CO₂ per kWh (CEA India 2023)
    TREE_ABSORPTION_KG = 21.77    # kg CO₂ absorbed per mature tree per year

    co2_saved_kg = surplus_used_kwh * CEA_INTENSITY
    trees_equivalent = co2_saved_kg / TREE_ABSORPTION_KG
    annual_co2_kg = co2_saved_kg * 365
    annual_trees = trees_equivalent * 365

    return {
        "co2_saved_kg": round(co2_saved_kg, 4),
        "trees_equivalent": round(trees_equivalent, 4),
        "annual_co2_kg": round(annual_co2_kg, 2),
        "annual_trees": round(annual_trees, 2),
    }


# ---------------------------------------------------------------------------
# Quick self-test (run directly: python -m models.scheduler)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Synthetic 24-h solar profile peaking at noon
    hours_arr = np.arange(24)
    solar = np.clip(50 * np.exp(-0.5 * ((hours_arr - 13) / 3) ** 2), 0, None)
    demand = 20 + 5 * np.sin(np.pi * hours_arr / 12)

    result = run_scheduler(solar, demand)

    print("=== Schedule ===")
    for name, hrs in result.schedule.items():
        print(f"  {name:30s}: {hrs}")
    print(f"  Unscheduled      : {result.unscheduled}")
    print(f"  Surplus used     : {result.surplus_used_kwh} kWh")
    print(f"  Grid fallback    : {result.grid_fallback_kwh} kWh")

    tod = compute_tod_savings(result, solar, demand)
    print("\n=== ToD Savings ===")
    for k, v in tod.items():
        print(f"  {k}: {v}")

    carbon = compute_carbon_savings(result.surplus_used_kwh)
    print("\n=== Carbon Savings ===")
    for k, v in carbon.items():
        print(f"  {k}: {v}")
