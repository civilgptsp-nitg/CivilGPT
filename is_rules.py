# is_rules.py
from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd

# ---- IS 456 durability snippets (you already had these in app.py; mirrored here for centralization)
EXPOSURE_WB_LIMITS = {
    "Mild": 0.60, "Moderate": 0.55, "Severe": 0.50, "Very Severe": 0.45, "Marine": 0.40
}
EXPOSURE_MIN_CEMENT = {
    "Mild": 300, "Moderate": 300, "Severe": 320, "Very Severe": 340, "Marine": 360
}

# ---- IS 383 (1970) Table 4 — Fine aggregate grading limits (Zones I–IV)
# Source: IS 383:1970 Table 4. :contentReference[oaicite:0]{index=0}
FINE_AGG_ZONE_LIMITS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "Zone I": {
        "10.0": (100, 100),
        "4.75": (90, 100),
        "2.36": (60, 95),
        "1.18": (30, 70),
        "0.600": (15, 34),
        "0.300": (5, 20),
        "0.150": (0, 10),
    },
    "Zone II": {
        "10.0": (100, 100),
        "4.75": (90, 100),
        "2.36": (75, 100),
        "1.18": (55, 90),
        "0.600": (35, 59),
        "0.300": (8, 30),
        "0.150": (0, 10),
    },
    "Zone III": {
        "10.0": (100, 100),
        "4.75": (90, 100),
        "2.36": (85, 100),
        "1.18": (75, 100),
        "0.600": (60, 79),
        "0.300": (12, 40),
        "0.150": (0, 10),
    },
    "Zone IV": {
        "10.0": (100, 100),
        "4.75": (95, 100),
        "2.36": (95, 100),
        "1.18": (90, 100),
        "0.600": (80, 100),
        "0.300": (15, 50),
        "0.150": (0, 15),  # note: 20% allowed for crushed stone sand; see IS note
    },
}

# ---- IS 383 (1970) Table 2 — Graded coarse aggregate, Nominal size 20 mm
# Source: IS 383:1970 Table 2 (Graded 20 mm column). :contentReference[oaicite:1]{index=1}
COARSE_GRADED_20MM_LIMITS: Dict[str, Tuple[int, int]] = {
    "40.0": (95, 100),
    "20.0": (95, 100),
    "10.0": (25, 55),
    "4.75": (0, 10),
}

# ---- IS 10262 baseline water content (for slump 25–50 mm, angular aggregate)
# 20 mm: 186 kg/m³; plus ~3% water per extra 25 mm slump (without SP), typical teaching from IS 10262:2009/2019. 
# Sources: Table in IS 10262 summaries. :contentReference[oaicite:2]{index=2}
WATER_BASELINE = {
    10: 200, 12.5: 195, 20: 186, 40: 165
}

def water_for_slump(nom_max_mm: int, slump_mm: int, uses_sp: bool=False, sp_reduction_frac: float=0.0) -> float:
    """Compute target mixing water per m³ based on IS 10262 baseline + slump/SP adjustments (generic guidance)."""
    base = WATER_BASELINE.get(nom_max_mm, 186)
    if slump_mm <= 50:
        water = base
    else:
        # +3% per extra 25 mm (heuristic from IS 10262 guidance)
        extra_25mm = max(0, (slump_mm - 50) / 25.0)
        water = base * (1 + 0.03 * extra_25mm)

    if uses_sp and sp_reduction_frac > 0:
        # reduce water by superplasticizer efficiency (e.g., 0.15–0.25)
        water *= (1 - sp_reduction_frac)
    return water

# --- Moisture / absorption correction helpers (IS 10262 workflow convention)
def aggregate_correction(delta_moisture_pct: float, agg_mass_ssd: float) -> Tuple[float, float]:
    """
    Returns (added_free_water, corrected_agg_mass_as_batched).
    delta_moisture_pct = (moisture% - absorption%)
    If positive => aggregate carries free water => subtract from batch water and reduce added water;
    If negative => aggregate is drier than SSD => it will absorb water => add to batch water.
    """
    water_delta = (delta_moisture_pct / 100.0) * agg_mass_ssd
    corrected_mass = agg_mass_ssd * (1 + delta_moisture_pct / 100.0)
    return (water_delta, corrected_mass)

@dataclass
class GradationResult:
    ok: bool
    messages: List[str]

def check_fine_zone(df: pd.DataFrame, zone: str) -> GradationResult:
    limits = FINE_AGG_ZONE_LIMITS[zone]
    msgs = []
    ok_all = True
    for sieve, (lo, hi) in limits.items():
        row = df.loc[df["Sieve_mm"].astype(str) == sieve]
        if row.empty: 
            msgs.append(f"Missing sieve {sieve} mm in upload.")
            ok_all = False
            continue
        p = float(row["PercentPassing"].iloc[0])
        if not (lo <= p <= hi):
            msgs.append(f"Zone {zone}: {sieve} mm -> {p:.1f}% (req {lo}-{hi}%)")
            ok_all = False
    return GradationResult(ok_all, msgs if msgs else [f"Fine aggregate meets IS 383 {zone}."])

def check_coarse_20mm(df: pd.DataFrame) -> GradationResult:
    msgs = []
    ok_all = True
    for sieve, (lo, hi) in COARSE_GRADED_20MM_LIMITS.items():
        row = df.loc[df["Sieve_mm"].astype(str) == sieve]
        if row.empty:
            msgs.append(f"Missing sieve {sieve} mm in upload.")
            ok_all = False
            continue
        p = float(row["PercentPassing"].iloc[0])
        if not (lo <= p <= hi):
            msgs.append(f"20 mm graded: {sieve} mm -> {p:.1f}% (req {lo}-{hi}%)")
            ok_all = False
    return GradationResult(ok_all, msgs if msgs else ["Coarse aggregate meets IS 383 (20 mm graded)."])
