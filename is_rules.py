"""
is_rules.py
---------------
Central repository of IS code–based constants and helper functions
for CivilGPT (concrete mix designer).
"""

import numpy as np
import pandas as pd

# =========================
# IS 456 Durability Rules
# =========================
EXPOSURE_WB_LIMITS = {
    "Mild": 0.60,
    "Moderate": 0.55,
    "Severe": 0.50,
    "Very Severe": 0.45,
    "Marine": 0.40,
}

EXPOSURE_MIN_CEMENT = {
    "Mild": 300,
    "Moderate": 300,
    "Severe": 320,
    "Very Severe": 340,
    "Marine": 360,
}

# =========================
# Concrete Grades (IS 456)
# =========================
GRADE_STRENGTH = {
    "M20": 20,
    "M25": 25,
    "M30": 30,
    "M35": 35,
    "M40": 40,
}

# =========================
# IS 10262 – Water Content
# =========================
WATER_BASELINE = {10: 200, 12.5: 195, 20: 186, 40: 165}

def water_for_slump(nom_max_mm: int, slump_mm: int, uses_sp: bool=False, sp_reduction_frac: float=0.0) -> float:
    """
    Estimate mixing water for given slump & aggregate size (IS 10262 style).
    Adjusts for superplasticizer if used.
    """
    base = WATER_BASELINE.get(int(nom_max_mm), 186)
    if slump_mm <= 50:
        water = base
    else:
        extra_25 = max(0, (slump_mm - 50) / 25.0)
        water = base * (1 + 0.03 * extra_25)  # +3% per 25 mm increment
    if uses_sp and sp_reduction_frac > 0:
        water *= (1 - sp_reduction_frac)
    return float(water)

def aggregate_correction(delta_moisture_pct: float, agg_mass_ssd: float):
    """
    Adjust aggregate batch mass for field moisture deviation.
    Returns: (free_water_contribution, corrected_mass).
    """
    water_delta = (delta_moisture_pct / 100.0) * agg_mass_ssd
    corrected_mass = agg_mass_ssd * (1 + delta_moisture_pct / 100.0)
    return float(water_delta), float(corrected_mass)

# =========================
# IS 383 – Sieve Limits
# =========================
FINE_AGG_ZONE_LIMITS = {
    "Zone I":  {"10.0": (100,100),"4.75": (90,100),"2.36": (60,95),"1.18": (30,70),"0.600": (15,34),"0.300": (5,20),"0.150": (0,10)},
    "Zone II": {"10.0": (100,100),"4.75": (90,100),"2.36": (75,100),"1.18": (55,90),"0.600": (35,59),"0.300": (8,30),"0.150": (0,10)},
    "Zone III":{"10.0": (100,100),"4.75": (90,100),"2.36": (85,100),"1.18": (75,100),"0.600": (60,79),"0.300": (12,40),"0.150": (0,10)},
    "Zone IV": {"10.0": (100,100),"4.75": (95,100),"2.36": (95,100),"1.18": (90,100),"0.600": (80,100),"0.300": (15,50),"0.150": (0,15)},
}

COARSE_20MM_LIMITS = {
    "40.0": (95,100),
    "20.0": (95,100),
    "10.0": (25,55),
    "4.75": (0,10),
}

def sieve_check_fa(df: pd.DataFrame, zone: str):
    """
    Verify fine aggregate sieve analysis against IS 383 limits.
    """
    limits = FINE_AGG_ZONE_LIMITS[zone]
    ok = True
    msgs = []
    for sieve, (lo, hi) in limits.items():
        row = df.loc[df["Sieve_mm"].astype(str) == sieve]
        if row.empty:
            msgs.append(f"Missing sieve {sieve} mm.")
            ok = False
            continue
        p = float(row["PercentPassing"].iloc[0])
        if not (lo <= p <= hi):
            ok = False
            msgs.append(f"{sieve} mm → {p:.1f}% (req {lo}-{hi}%)")
    if ok and not msgs:
        msgs = [f"Fine aggregate meets IS 383 {zone}."]
    return ok, msgs

def sieve_check_ca20(df: pd.DataFrame):
    """
    Verify coarse aggregate (20 mm nominal) sieve analysis against IS 383 limits.
    """
    ok = True
    msgs = []
    for sieve, (lo, hi) in COARSE_20MM_LIMITS.items():
        row = df.loc[df["Sieve_mm"].astype(str) == sieve]
        if row.empty:
            msgs.append(f"Missing sieve {sieve} mm.")
            ok = False
            continue
        p = float(row["PercentPassing"].iloc[0])
        if not (lo <= p <= hi):
            ok = False
            msgs.append(f"{sieve} mm → {p:.1f}% (req {lo}-{hi}%)")
    if ok and not msgs:
        msgs = ["Coarse aggregate meets IS 383 (20 mm graded)."]
    return ok, msgs
