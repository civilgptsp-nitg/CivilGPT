# app.py - CivilGPT v2.5 (Refactored UI & IS-Code Compliant Logic)
# Backend logic preserved from v2.0
# UI refactored for a professional, modern, and intuitive experience
# Clarification step for free-text parsing added.
# v2.2 - Corrected aggregate proportioning logic to align with IS 10262:2019, Table 5.
# v2.3 - Added developer calibration panel to tune optimizer search parameters.
# v2.4 - Added Lab Calibration Dataset Upload + Error Analysis feature.
# v2.5 - Integrated Material Library, Binder Range Checks, Judge Prompts, and Calculation Walkthrough.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
import json
import traceback
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from matplotlib.ticker import ScalarFormatter


# ==============================================================================
# PART 1: BACKEND LOGIC (CORRECTED & ENHANCED) - DO NOT MODIFY
# ==============================================================================

# Groq client (optional)
try:
    from groq import Groq
    client = Groq(api_key=st.secrets.get("GROQ_API_KEY", None))
except (ImportError, Exception):
    client = None

# App Config
st.set_page_config(
    page_title="CivilGPT - Sustainable Concrete Mix Designer",
    page_icon="🧱",
    layout="wide"
)

# --- IS Code Rules & Tables (IS 456 & IS 10262) ---
EXPOSURE_WB_LIMITS = {"Mild": 0.60,"Moderate": 0.55,"Severe": 0.50,"Very Severe": 0.45,"Marine": 0.40}
EXPOSURE_MIN_CEMENT = {"Mild": 300, "Moderate": 300, "Severe": 320,"Very Severe": 340, "Marine": 360}
EXPOSURE_MIN_GRADE = {"Mild": "M20", "Moderate": "M25", "Severe": "M30","Very Severe": "M35", "Marine": "M40"}
GRADE_STRENGTH = {"M10": 10, "M15": 15, "M20": 20, "M25": 25,"M30": 30, "M35": 35, "M40": 40, "M45": 45, "M50": 50}
WATER_BASELINE = {10: 208, 12.5: 202, 20: 186, 40: 165} # IS 10262, Table 4 (for 50mm slump)
AGG_SHAPE_WATER_ADJ = {"Angular (baseline)": 0.00, "Sub-angular": -0.03,"Sub-rounded": -0.05, "Rounded": -0.07,"Flaky/Elongated": +0.03}
QC_STDDEV = {"Good": 5.0, "Fair": 7.5, "Poor": 10.0} # IS 10262, Table 2

# Typical binder ranges by grade
BINDER_RANGES = {
    "M10": (220, 320), "M15": (250, 350), "M20": (300, 400),
    "M25": (320, 420), "M30": (340, 450), "M35": (360, 480),
    "M40": (380, 500), "M45": (400, 520), "M50": (420, 540)
}

# IS 10262:2019, Table 5 - Volume of Coarse Aggregate per unit volume of Total Aggregate
COARSE_AGG_FRAC_BY_ZONE = {
    10: {"Zone I": 0.50, "Zone II": 0.48, "Zone III": 0.46, "Zone IV": 0.44},
    12.5: {"Zone I": 0.59, "Zone II": 0.57, "Zone III": 0.55, "Zone IV": 0.53},
    20: {"Zone I": 0.66, "Zone II": 0.64, "Zone III": 0.62, "Zone IV": 0.60},
    40: {"Zone I": 0.71, "Zone II": 0.69, "Zone III": 0.67, "Zone IV": 0.65}
}

FINE_AGG_ZONE_LIMITS = {
    "Zone I":   {"10.0": (100,100),"4.75": (90,100),"2.36": (60,95),"1.18": (30,70),"0.600": (15,34),"0.300": (5,20),"0.150": (0,10)},
    "Zone II":  {"10.0": (100,100),"4.75": (90,100),"2.36": (75,100),"1.18": (55,90),"0.600": (35,59),"0.300": (8,30),"0.150": (0,10)},
    "Zone III": {"10.0": (100,100),"4.75": (90,100),"2.36": (85,100),"1.18": (75,90),"0.600": (60,79),"0.300": (12,40),"0.150": (0,10)},
    "Zone IV":  {"10.0": (95,100),"4.75": (95,100),"2.36": (95,100),"1.18": (90,100),"0.600": (80,100),"0.300": (15,50),"0.150": (0,15)},
}

COARSE_LIMITS = {
    10: {"20.0": (100,100), "10.0": (85,100),  "4.75": (0,20)},
    20: {"40.0": (100,100),  "20.0": (95,100),  "10.0": (25,55), "4.75": (0,10)},
    40: {"80.0": (100,100),  "40.0": (95,100),  "20.0": (30,70), "10.0": (0,15)}
}

# Parsers
def simple_parse(text: str) -> dict:
    result = {}
    grade_match = re.search(r"\bM\s*(10|15|20|25|30|35|40|45|50)\b", text, re.IGNORECASE)
    if grade_match: result["grade"] = "M" + grade_match.group(1)
    for exp in EXPOSURE_WB_LIMITS.keys():
        if re.search(r'\b' + exp + r'\b', text, re.IGNORECASE): result["exposure"] = exp; break
    slump_match = re.search(r"(slump\s*(of\s*)?|)\b(\d{2,3})\s*mm", text, re.IGNORECASE)
    if slump_match: result["slump"] = int(slump_match.group(3))
    cement_types = ["OPC 43"] # Restricted to OPC 43 only
    for ctype in cement_types:
        if re.search(ctype.replace(" ", r"\s*"), text, re.IGNORECASE):
            result["cement"] = ctype; break
    nom_match = re.search(r"(\d{2}(\.5)?)\s*mm", text, re.IGNORECASE)
    if nom_match:
        try: result["nom_max"] = float(nom_match.group(1))
        except: pass
    return result

def parse_input_with_llm(user_text: str) -> dict:
    if client is None:
        return simple_parse(user_text)
    prompt = f"Extract grade, exposure, slump (mm), cement type, and nominal max aggregate from: {user_text}. Return JSON."
    resp = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    try:
        parsed = json.loads(resp.choices[0].message.content)
    except Exception:
        parsed = simple_parse(user_text)
    return parsed

# Helpers
@st.cache_data
def load_data(emissions_file=None, cost_file=None):
    def _safe_read_csv(file, default_path):
        if file is not None:
            try: return pd.read_csv(file)
            except Exception: return pd.read_csv(default_path)
        return pd.read_csv(default_path)

    try:
        emissions = _safe_read_csv(emissions_file, "emission_factors.csv")
    except Exception:
        emissions = pd.DataFrame(columns=["Material", "CO2_Factor(kg_CO2_per_kg)"])

    try:
        costs = _safe_read_csv(cost_file, "cost_factors.csv")
    except Exception:
        costs = pd.DataFrame(columns=["Material", "Cost(₹/kg)"])

    return emissions, costs


def pareto_front(df, x_col="cost", y_col="co2"):
    if df.empty:
        return pd.DataFrame(columns=df.columns)
    sorted_df = df.sort_values(by=[x_col, y_col], ascending=[True, True])
    pareto_points = []
    last_y = float('inf')
    for _, row in sorted_df.iterrows():
        if row[y_col] < last_y:
            pareto_points.append(row)
            last_y = row[y_col]
    if not pareto_points:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(pareto_points).reset_index(drop=True)


def water_for_slump_and_shape(nom_max_mm: int, slump_mm: int,
                              agg_shape: str, uses_sp: bool=False,
                              sp_reduction_frac: float=0.0) -> float:
    base = WATER_BASELINE.get(int(nom_max_mm), 186.0)
    if slump_mm <= 50: water = base
    else: water = base * (1 + 0.03 * ((slump_mm - 50) / 25.0))
    water *= (1.0 + AGG_SHAPE_WATER_ADJ.get(agg_shape, 0.0))
    if uses_sp and sp_reduction_frac > 0: water *= (1 - sp_reduction_frac)
    return float(water)

def reasonable_binder_range(grade: str):
    return BINDER_RANGES.get(grade, (300, 500))

def get_coarse_agg_fraction(nom_max_mm: float, fa_zone: str, wb_ratio: float):
    base_fraction = COARSE_AGG_FRAC_BY_ZONE.get(nom_max_mm, {}).get(fa_zone, 0.62)
    wb_diff = 0.50 - wb_ratio
    correction = (wb_diff / 0.05) * 0.01
    corrected_fraction = base_fraction + correction
    return max(0.4, min(0.8, corrected_fraction))

def run_lab_calibration(lab_df):
    if lab_df is None or lab_df.empty:
        return None, {}
    required_cols = ['grade', 'actual_strength']
    if not all(col in lab_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in lab_df.columns]
        raise ValueError(f"Lab data CSV is missing required columns: {', '.join(missing)}")

    results = []
    default_qc_level = "Good"
    std_dev_S = QC_STDDEV[default_qc_level]

    for _, row in lab_df.iterrows():
        try:
            grade = str(row['grade']).strip()
            actual_strength = float(row['actual_strength'])
            if grade not in GRADE_STRENGTH:
                continue
            fck = GRADE_STRENGTH[grade]
            predicted_strength = fck + 1.65 * std_dev_S

            results.append({
                "Grade": grade,
                "Exposure": row.get('exposure', 'N/A'),
                "Slump (mm)": row.get('slump', 'N/A'),
                "Lab Strength (MPa)": actual_strength,
                "Predicted Target Strength (MPa)": predicted_strength,
                "Error (MPa)": predicted_strength - actual_strength
            })
        except (ValueError, TypeError):
            pass

    if not results:
        return None, {}

    results_df = pd.DataFrame(results)
    mae = results_df["Error (MPa)"].abs().mean()
    rmse = np.sqrt((results_df["Error (MPa)"] ** 2).mean())
    bias = results_df["Error (MPa)"].mean()
    metrics = {"Mean Absolute Error (MPa)": mae, "Root Mean Squared Error (MPa)": rmse, "Mean Bias (MPa)": bias}
    return results_df, metrics

# ==============================================================================
# PART 2: CORE MIX LOGIC (UPDATED) - DO NOT MODIFY
# ==============================================================================

def evaluate_mix(components_dict, emissions_df, costs_df=None):
    comp_items = [(m.strip().lower(), q) for m, q in components_dict.items()]
    comp_df = pd.DataFrame(comp_items, columns=["Material_norm", "Quantity (kg/m3)"])
    emissions_df = emissions_df.copy()
    emissions_df["Material_norm"] = emissions_df["Material"].str.strip().str.lower()
    df = comp_df.merge(emissions_df[["Material_norm","CO2_Factor(kg_CO2_per_kg)"]], on="Material_norm", how="left")
    df["CO2_Factor(kg_CO2_per_kg)"] = df["CO2_Factor(kg_CO2_per_kg)"].fillna(0.0)
    df["CO2_Emissions (kg/m3)"] = df["Quantity (kg/m3)"] * df["CO2_Factor(kg_CO2_per_kg)"]

    if costs_df is not None and "Cost(₹/kg)" in costs_df.columns and not costs_df.empty:
        costs_df = costs_df.copy()
        costs_df["Material_norm"] = costs_df["Material"].str.strip().str.lower()
        df = df.merge(costs_df[["Material_norm", "Cost(₹/kg)"]], on="Material_norm", how="left")
        df["Cost(₹/kg)"] = df["Cost(₹/kg)"].fillna(0.0)
        df["Cost (₹/m3)"] = df["Quantity (kg/m3)"] * df["Cost(₹/kg)"]
    else:
        df["Cost(₹/kg)"] = 0.0
        df["Cost (₹/m3)"] = 0.0

    df["Material"] = df["Material_norm"].str.title()
    return df[["Material","Quantity (kg/m3)","CO2_Factor(kg_CO2_per_kg)","CO2_Emissions (kg/m3)","Cost(₹/kg)","Cost (₹/m3)"]]

def aggregate_correction(delta_moisture_pct: float, agg_mass_ssd: float):
    water_delta = (delta_moisture_pct / 100.0) * agg_mass_ssd
    corrected_mass = agg_mass_ssd * (1 + delta_moisture_pct / 100.0)
    return float(water_delta), float(corrected_mass)

def compute_aggregates(cementitious, water, sp, coarse_agg_fraction,
                       density_fa=2650.0, density_ca=2700.0):
    vol_cem = cementitious / 3150.0
    vol_wat = water / 1000.0
    vol_sp  = sp / 1200.0
    vol_binder = vol_cem + vol_wat + vol_sp
    vol_agg = 1.0 - vol_binder
    if vol_agg <= 0: vol_agg = 0.60

    vol_coarse = vol_agg * coarse_agg_fraction
    vol_fine = vol_agg * (1.0 - coarse_agg_fraction)

    mass_fine, mass_coarse = vol_fine * density_fa, vol_coarse * density_ca
    return float(mass_fine), float(mass_coarse)

def compliance_checks(mix_df, meta, exposure):
    checks = {}
    try: checks["W/B ≤ exposure limit"] = float(meta["w_b"]) <= EXPOSURE_WB_LIMITS[exposure]
    except: checks["W/B ≤ exposure limit"] = False
    try: checks["Min cementitious met"] = float(meta["cementitious"]) >= float(EXPOSURE_MIN_CEMENT[exposure])
    except: checks["Min cementitious met"] = False
    try: checks["SCM ≤ 50%"] = float(meta.get("scm_total_frac", 0.0)) <= 0.50
    except: checks["SCM ≤ 50%"] = False
    try:
        total_mass = float(mix_df["Quantity (kg/m3)"].sum())
        checks["Unit weight 2200–2600 kg/m³"] = 2200.0 <= total_mass <= 2600.0
    except: checks["Unit weight 2200–2600 kg/m³"] = False
    derived = {
        "w/b used": round(float(meta.get("w_b", 0.0)), 3),
        "cementitious (kg/m³)": round(float(meta.get("cementitious", 0.0)), 1),
        "SCM % of cementitious": round(100 * float(meta.get("scm_total_frac", 0.0)), 1),
        "total mass (kg/m³)": round(float(mix_df["Quantity (kg/m3)"].sum()), 1) if "Quantity (kg/m3)" in mix_df.columns else None,
        "water target (kg/m³)": round(float(meta.get("water_target", 0.0)), 1),
        "cement (kg/m³)": round(float(meta.get("cement", 0.0)), 1),
        "fly ash (kg/m³)": round(float(meta.get("flyash", 0.0)), 1),
        "ggbs (kg/m³)": round(float(meta.get("ggbs", 0.0)), 1),
        "fine agg (kg/m³)": round(float(meta.get("fine", 0.0)), 1),
        "coarse agg (kg/m³)": round(float(meta.get("coarse", 0.0)), 1),
        "SP (kg/m³)": round(float(meta.get("sp", 0.0)), 2),
        "fck (MPa)": meta.get("fck"), "fck,target (MPa)": meta.get("fck_target"), "QC (S, MPa)": meta.get("stddev_S"),
    }
    return checks, derived

def sanity_check_mix(meta, df):
    warnings = []
    try:
        cement, water, fine, coarse, sp = float(meta.get("cement", 0)), float(meta.get("water_target", 0)), float(meta.get("fine", 0)), float(meta.get("coarse", 0)), float(meta.get("sp", 0))
    except Exception: return ["Insufficient data to run sanity checks."]

    if cement > 500: warnings.append(f"High cement content ({cement:.1f} kg/m³). Increases cost, shrinkage, and CO₂.")
    if water < 140 or water > 220: warnings.append(f"Water content ({water:.1f} kg/m³) is outside the typical range of 140-220 kg/m³.")
    if fine < 500 or fine > 900: warnings.append(f"Fine aggregate quantity ({fine:.1f} kg/m³) is unusual.")
    if coarse < 1000 or coarse > 1300: warnings.append(f"Coarse aggregate quantity ({coarse:.1f} kg/m³) is unusual.")
    if sp > 20: warnings.append(f"Superplasticizer dosage ({sp:.1f} kg/m³) is unusually high.")
    return warnings

def check_feasibility(mix_df, meta, exposure):
    checks, derived = compliance_checks(mix_df, meta, exposure)
    warnings = sanity_check_mix(meta, mix_df)
    reasons_fail = [f"IS Code Fail: {k}" for k, v in checks.items() if not v]
    feasible = len(reasons_fail) == 0
    return feasible, reasons_fail, warnings, derived, checks

def sieve_check_fa(df: pd.DataFrame, zone: str):
    try:
        limits, ok, msgs = FINE_AGG_ZONE_LIMITS[zone], True, []
        for sieve, (lo, hi) in limits.items():
            row = df.loc[df["Sieve_mm"].astype(str) == sieve]
            if row.empty:
                ok = False; msgs.append(f"Missing sieve size: {sieve} mm."); continue
            p = float(row["PercentPassing"].iloc[0])
            if not (lo <= p <= hi): ok = False; msgs.append(f"Sieve {sieve} mm: {p:.1f}% passing is outside the required range of {lo}-{hi}%.")
        if ok: msgs = [f"Fine aggregate conforms to IS 383 for {zone}."]
        return ok, msgs
    except Exception: return False, ["Invalid fine aggregate CSV format. Ensure 'Sieve_mm' and 'PercentPassing' columns exist."]

def sieve_check_ca(df: pd.DataFrame, nominal_mm: int):
    try:
        limits, ok, msgs = COARSE_LIMITS[int(nominal_mm)], True, []
        for sieve, (lo, hi) in limits.items():
            row = df.loc[df["Sieve_mm"].astype(str) == sieve]
            if row.empty:
                ok = False; msgs.append(f"Missing sieve size: {sieve} mm."); continue
            p = float(row["PercentPassing"].iloc[0])
            if not (lo <= p <= hi): ok = False; msgs.append(f"Sieve {sieve} mm: {p:.1f}% passing is outside the required range of {lo}-{hi}%.")
        if ok: msgs = [f"Coarse aggregate conforms to IS 383 for {nominal_mm} mm graded aggregate."]
        return ok, msgs
    except Exception: return False, ["Invalid coarse aggregate CSV format. Ensure 'Sieve_mm' and 'PercentPassing' columns exist."]

def generate_mix(grade, exposure, nom_max, target_slump, agg_shape, fine_zone, emissions, costs, cement_choice, material_props, use_sp=True, sp_reduction=0.18, optimize_cost=False, wb_min=0.35, wb_steps=6, max_flyash_frac=0.3, max_ggbs_frac=0.5, scm_step=0.1, fine_fraction_override=None):
    w_b_limit, min_cem_exp = float(EXPOSURE_WB_LIMITS[exposure]), float(EXPOSURE_MIN_CEMENT[exposure])
    target_water = water_for_slump_and_shape(nom_max_mm=nom_max, slump_mm=int(target_slump), agg_shape=agg_shape, uses_sp=use_sp, sp_reduction_frac=sp_reduction)
    best_df, best_meta, best_score = None, None, float("inf")
    trace = []

    wb_values = np.linspace(float(wb_min), float(w_b_limit), int(wb_steps))
    flyash_options = np.arange(0.0, max_flyash_frac + 1e-9, scm_step)
    ggbs_options = np.arange(0.0, max_ggbs_frac + 1e-9, scm_step)
    min_b_grade, max_b_grade = reasonable_binder_range(grade)

    for wb in wb_values:
        for flyash_frac in flyash_options:
            for ggbs_frac in ggbs_options:
                if flyash_frac + ggbs_frac > 0.50: continue

                binder_for_strength = target_water / wb
                binder = max(binder_for_strength, min_cem_exp, min_b_grade)
                binder = min(binder, max_b_grade)
                actual_wb = target_water / binder

                cement, flyash, ggbs = binder * (1 - flyash_frac - ggbs_frac), binder * flyash_frac, binder * ggbs_frac
                sp = 0.01 * binder if use_sp else 0.0

                density_fa = material_props['sg_fa'] * 1000
                density_ca = material_props['sg_ca'] * 1000

                if fine_fraction_override is not None:
                    coarse_agg_frac = 1.0 - fine_fraction_override
                else:
                    coarse_agg_frac = get_coarse_agg_fraction(nom_max, fine_zone, actual_wb)

                fine_ssd, coarse_ssd = compute_aggregates(binder, target_water, sp, coarse_agg_frac, density_fa, density_ca)
                water_delta_fa, fine_wet = aggregate_correction(material_props['moisture_fa'], fine_ssd)
                water_delta_ca, coarse_wet = aggregate_correction(material_props['moisture_ca'], coarse_ssd)
                water_final = target_water - water_delta_fa - water_delta_ca

                mix = {cement_choice: cement,"Fly Ash": flyash,"GGBS": ggbs,"Water": water_final,"PCE Superplasticizer": sp,"Fine Aggregate": fine_wet,"Coarse Aggregate": coarse_wet}
                df = evaluate_mix(mix, emissions, costs)
                co2_total, cost_total = float(df["CO2_Emissions (kg/m3)"].sum()), float(df["Cost (₹/m3)"].sum())

                candidate_meta = {"w_b": actual_wb, "cementitious": binder, "cement": cement, "flyash": flyash, "ggbs": ggbs, "water_target": target_water, "water_final": water_final, "sp": sp, "fine": fine_wet, "coarse": coarse_wet, "scm_total_frac": flyash_frac + ggbs_frac, "grade": grade, "exposure": exposure, "nom_max": nom_max, "slump": target_slump, "co2_total": co2_total, "cost_total": cost_total, "coarse_agg_fraction": coarse_agg_frac, "binder_range": (min_b_grade, max_b_grade), "material_props": material_props}
                feasible, reasons_fail, _, _, _ = check_feasibility(df, candidate_meta, exposure)
                score = co2_total if not optimize_cost else cost_total
                trace.append({"wb": float(actual_wb), "flyash_frac": float(flyash_frac), "ggbs_frac": float(ggbs_frac),"co2": float(co2_total), "cost": float(cost_total),"score": float(score), "feasible": bool(feasible),"reasons": ", ".join(reasons_fail)})
                if feasible and score < best_score:
                    best_df, best_score, best_meta = df.copy(), score, candidate_meta.copy()
    return best_df, best_meta, trace

def generate_baseline(grade, exposure, nom_max, target_slump, agg_shape, fine_zone, emissions, costs, cement_choice, material_props, use_sp=True, sp_reduction=0.18):
    w_b_limit, min_cem_exp = float(EXPOSURE_WB_LIMITS[exposure]), float(EXPOSURE_MIN_CEMENT[exposure])
    water_target = water_for_slump_and_shape(nom_max_mm=nom_max, slump_mm=int(target_slump), agg_shape=agg_shape, uses_sp=use_sp, sp_reduction_frac=sp_reduction)

    min_b_grade, max_b_grade = reasonable_binder_range(grade)
    binder_for_wb = water_target / w_b_limit
    cementitious = max(binder_for_wb, min_cem_exp, min_b_grade)
    cementitious = min(cementitious, max_b_grade)
    actual_wb = water_target / cementitious
    sp = 0.01 * cementitious if use_sp else 0.0

    coarse_agg_frac = get_coarse_agg_fraction(nom_max, fine_zone, actual_wb)
    density_fa = material_props['sg_fa'] * 1000
    density_ca = material_props['sg_ca'] * 1000
    fine_ssd, coarse_ssd = compute_aggregates(cementitious, water_target, sp, coarse_agg_frac, density_fa, density_ca)

    water_delta_fa, fine_wet = aggregate_correction(material_props['moisture_fa'], fine_ssd)
    water_delta_ca, coarse_wet = aggregate_correction(material_props['moisture_ca'], coarse_ssd)
    water_final = water_target - water_delta_fa - water_delta_ca

    mix = {cement_choice: cementitious,"Fly Ash": 0.0,"GGBS": 0.0,"Water": water_final, "PCE Superplasticizer": sp,"Fine Aggregate": fine_wet,"Coarse Aggregate": coarse_wet}
    df = evaluate_mix(mix, emissions, costs)
    meta = {"w_b": actual_wb, "cementitious": cementitious, "cement": cementitious, "flyash": 0.0, "ggbs": 0.0, "water_target": water_target, "water_final": water_final, "sp": sp, "fine": fine_wet, "coarse": coarse_wet, "scm_total_frac": 0.0, "grade": grade, "exposure": exposure, "nom_max": nom_max, "slump": target_slump, "co2_total": float(df["CO2_Emissions (kg/m3)"].sum()), "cost_total": float(df["Cost (₹/m3)"].sum()), "coarse_agg_fraction": coarse_agg_frac, "material_props": material_props}
    return df, meta

def apply_parser(user_text, current_inputs):
    if not user_text.strip():
        return current_inputs, [], {}
    try:
        parsed = parse_input_with_llm(user_text) if use_llm_parser else simple_parse(user_text)
    except Exception as e:
        st.warning(f"Parser error: {e}, falling back to regex")
        parsed = simple_parse(user_text)
    messages, updated = [], current_inputs.copy()
    if "grade" in parsed and parsed["grade"] in GRADE_STRENGTH:
        updated["grade"] = parsed["grade"]; messages.append(f"✅ Parser set Grade to **{parsed['grade']}**")
    if "exposure" in parsed and parsed["exposure"] in EXPOSURE_WB_LIMITS:
        updated["exposure"] = parsed["exposure"]; messages.append(f"✅ Parser set Exposure to **{parsed['exposure']}**")
    if "slump" in parsed:
        s = max(25, min(180, int(parsed["slump"])))
        updated["target_slump"] = s; messages.append(f"✅ Parser set Target Slump to **{s} mm**")
    if "cement" in parsed:
        updated["cement_choice"] = parsed["cement"]; messages.append(f"✅ Parser set Cement Type to **{parsed['cement']}**")
    if "nom_max" in parsed and parsed["nom_max"] in [10, 12.5, 20, 40]:
        updated["nom_max"] = parsed["nom_max"]; messages.append(f"✅ Parser set Aggregate Size to **{parsed['nom_max']} mm**")
    return updated, messages, parsed

# ==============================================================================
# PART 3: REFACTORED USER INTERFACE
# ==============================================================================

# --- UI Helper Functions (New) ---
def plot_gradation_chart(user_df, limits_dict, title, zone_name=""):
    """Generates a matplotlib figure for sieve gradation analysis."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sieves = sorted([float(k) for k in limits_dict.keys()], reverse=True)
    lower_bounds = [limits_dict[str(s)][0] if str(s) in limits_dict else limits_dict[s][0] for s in sieves]
    upper_bounds = [limits_dict[str(s)][1] if str(s) in limits_dict else limits_dict[s][1] for s in sieves]

    ax.plot(sieves, upper_bounds, 'r--', label='IS 383 Limits')
    ax.plot(sieves, lower_bounds, 'r--')
    ax.fill_between(sieves, lower_bounds, upper_bounds, color='red', alpha=0.1, label=f'IS 383 Compliance Zone {zone_name}')

    user_df['Sieve_mm_num'] = pd.to_numeric(user_df['Sieve_mm'])
    user_df_sorted = user_df.sort_values('Sieve_mm_num', ascending=False)
    ax.plot(user_df_sorted['Sieve_mm_num'], user_df_sorted['PercentPassing'], 'b-o', label='Uploaded Aggregate', zorder=10)

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([s for s in sieves if s > 0.1])
    ax.minorticks_off()
    ax.set_xlabel("Sieve Size (mm) - Log Scale")
    ax.set_ylabel("Percent Passing (%)")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    plt.gca().invert_xaxis()
    return fig

def generate_pareto_plot_fig(feasible_mixes, pareto_df, opt_meta, best_compromise_mix, optimize_for):
    """Generates a reusable matplotlib figure of the Pareto front."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if not feasible_mixes.empty:
        ax.scatter(feasible_mixes["cost"], feasible_mixes["co2"], color='grey', alpha=0.4, label='All Feasible Mixes')
    if not pareto_df.empty:
        ax.plot(pareto_df["cost"], pareto_df["co2"], '-o', color='#2196F3', label='Pareto Front (Efficient Mixes)')

    ax.plot(opt_meta['cost_total'], opt_meta['co2_total'], '*', markersize=18, color='#F44336', label=f'Chosen Mix (Lowest {optimize_for.split(" ")[1]})', zorder=12, markeredgecolor='black')

    if best_compromise_mix is not None:
        ax.plot(best_compromise_mix['cost'], best_compromise_mix['co2'], 'D', markersize=12, color='#4CAF50', label='Best Compromise (from slider)', zorder=11, markeredgecolor='black')

    ax.set_xlabel("Material Cost (₹/m³)", fontsize=12)
    ax.set_ylabel("Embodied Carbon (kg CO₂e / m³)", fontsize=12)
    ax.set_title("Pareto Front of Feasible Concrete Mixes", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    return fig


# --- Page Styling ---
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 5rem; padding-right: 5rem; }
    .st-emotion-cache-1y4p8pa { max-width: 100%; }
    .stTextArea [data-baseweb=base-input] { border-color: #4A90E2; box-shadow: 0 0 5px #4A90E2; background-color: #f0f8ff; }
    .stMetric { background-color: #FFFFFF; border: 1px solid #E0E0E0; border-left: 6px solid #4A90E2; padding: 1rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stMetric[data-testid="stMetricValue"] { font-size: 2.2em; }
    .badge-compliant { background-color: #e8f5e9; color: #2e7d32; padding: 10px; border-radius: 5px; border-left: 5px solid #4CAF50; font-weight: bold; text-align: center; margin-bottom: 1rem; }
    .badge-non-compliant { background-color: #ffebee; color: #c62828; padding: 10px; border-radius: 5px; border-left: 5px solid #F44336; font-weight: bold; text-align: center; margin-bottom: 1rem; }
    .glow-success { border: 2px solid #4CAF50; border-radius: 10px; box-shadow: 0 0 15px #4CAF50; animation: glow-animation 2s infinite alternate; padding: 1rem; background-color: #f1f8e9; }
    @keyframes glow-animation { from { box-shadow: 0 0 5px #4CAF50; } to { box-shadow: 0 0 20px #66BB6A; } }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; justify-content: center; }
</style>
""", unsafe_allow_html=True)


# --- Landing Page / Main Interface ---
st.title("🧱 CivilGPT: Sustainable Concrete Mix Designer")
st.markdown("<h4 style='text-align: center; color: #4A90E2;'>An AI-powered tool for creating <b>IS 10262:2019 compliant</b> concrete mixes, optimized for low carbon footprint.</h4>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([0.7, 0.3])
with col1:
    user_text = st.text_area(
        "**Describe Your Requirements**",
        height=100,
        placeholder="e.g., Design an M30 grade concrete for severe exposure using OPC 43. Target a slump of 125 mm with 20 mm aggregates.",
        label_visibility="collapsed",
        key="user_text_input"
    )
with col2:
    st.write("")
    st.write("")
    run_button = st.button("🚀 Generate Mix Design", use_container_width=True, type="primary")

manual_mode = st.toggle("⚙️ Switch to Advanced Manual Input", help="Toggle for detailed control over all mix design parameters.")

# --- Sidebar ---
if 'user_text_input' not in st.session_state:
    st.session_state.user_text_input = ""

if manual_mode:
    st.sidebar.header("📝 Manual Mix Inputs")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Core Requirements")
    grade = st.sidebar.selectbox("Concrete Grade", list(GRADE_STRENGTH.keys()), index=4, help="Target characteristic compressive strength at 28 days.")
    exposure = st.sidebar.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()), index=2, help="Determines durability requirements like min. cement content and max. water-binder ratio as per IS 456.")

    st.sidebar.subheader("Workability & Materials")
    target_slump = st.sidebar.slider("Target Slump (mm)", 25, 180, 100, 5, help="Specifies the desired consistency and workability of the fresh concrete.")
    cement_choice = st.sidebar.selectbox("Cement Type", ["OPC 43"], index=0, help="Type of Ordinary Portland Cement.")
    nom_max = st.sidebar.selectbox("Nominal Max. Aggregate Size (mm)", [10, 12.5, 20, 40], index=2, help="Largest practical aggregate size, influences water demand.")
    agg_shape = st.sidebar.selectbox("Coarse Aggregate Shape", list(AGG_SHAPE_WATER_ADJ.keys()), index=0, help="Shape affects water demand; angular requires more water than rounded.")
    fine_zone = st.sidebar.selectbox("Fine Aggregate Zone (IS 383)", ["Zone I","Zone II","Zone III","Zone IV"], index=1, help="Grading zone as per IS 383. This is crucial for determining aggregate proportions per IS 10262.")

    st.sidebar.subheader("Optimization & Admixtures")
    use_sp = st.sidebar.checkbox("Use Superplasticizer (PCE)", True, help="Chemical admixture to increase workability or reduce water content.")
    optimize_for = st.sidebar.radio("Optimize For", ["Lowest CO₂", "Lowest Cost"], help="The optimizer will prioritize finding a feasible mix that minimizes either carbon emissions or material cost.")
    optimize_cost = (optimize_for == "Lowest Cost")

    st.sidebar.subheader("Advanced Parameters")
    with st.sidebar.expander("QA/QC"):
        qc_level = st.selectbox("Quality Control Level", list(QC_STDDEV.keys()), index=0, help="Assumed site quality control, affecting the target strength calculation (f_target = fck + 1.65 * S).")

    with st.sidebar.expander("Material Properties (from Library or Manual)"):
        materials_file = st.file_uploader("Upload Materials Library CSV", type=["csv"], key="materials_csv", help="CSV with 'Material', 'SpecificGravity', 'MoistureContent' columns.")
        sg_fa_default, moisture_fa_default = 2.65, 1.0
        sg_ca_default, moisture_ca_default = 2.70, 0.5

        if materials_file is not None:
            try:
                mat_df = pd.read_csv(materials_file)
                mat_df.columns = [col.strip().lower().replace(" ", "") for col in mat_df.columns]
                mat_df['material'] = mat_df['material'].str.strip().lower()

                fa_row = mat_df[mat_df['material'] == 'fine aggregate']
                if not fa_row.empty:
                    if 'specificgravity' in fa_row.columns: sg_fa_default = float(fa_row['specificgravity'].iloc[0])
                    if 'moisturecontent' in fa_row.columns: moisture_fa_default = float(fa_row['moisturecontent'].iloc[0])

                ca_row = mat_df[mat_df['material'] == 'coarse aggregate']
                if not ca_row.empty:
                    if 'specificgravity' in ca_row.columns: sg_ca_default = float(ca_row['specificgravity'].iloc[0])
                    if 'moisturecontent' in ca_row.columns: moisture_ca_default = float(ca_row['moisturecontent'].iloc[0])
                st.success("Materials library CSV loaded and properties updated.")
            except Exception as e:
                st.error(f"Failed to parse materials CSV: {e}")

        st.markdown("###### Fine Aggregate")
        sg_fa = st.number_input("Specific Gravity (FA)", 2.0, 3.0, sg_fa_default, 0.01)
        moisture_fa = st.number_input("Free Moisture Content % (FA)", -2.0, 5.0, moisture_fa_default, 0.1, help="Moisture beyond SSD condition. Negative if absorbent.")

        st.markdown("###### Coarse Aggregate")
        sg_ca = st.number_input("Specific Gravity (CA)", 2.0, 3.0, sg_ca_default, 0.01)
        moisture_ca = st.number_input("Free Moisture Content % (CA)", -2.0, 5.0, moisture_ca_default, 0.1, help="Moisture beyond SSD condition. Negative if absorbent.")

    st.sidebar.subheader("File Uploads (Optional)")
    with st.sidebar.expander("Upload Sieve Analysis & Financials"):
        fine_csv = st.file_uploader("Fine Aggregate CSV", type=["csv"], key="fine_csv", help="CSV with 'Sieve_mm' and 'PercentPassing' columns.")
        coarse_csv = st.file_uploader("Coarse Aggregate CSV", type=["csv"], key="coarse_csv", help="CSV with 'Sieve_mm' and 'PercentPassing' columns.")
        emissions_file = st.file_uploader("Emission Factors (kgCO₂/kg)", type=["csv"], key="emissions_csv")
        cost_file = st.file_uploader("Cost Factors (₹/kg)", type=["csv"], key="cost_csv")

    with st.sidebar.expander("🔬 Lab Calibration Dataset"):
        st.markdown("Upload CSV with lab results. Required columns: `grade`, `actual_strength`.")
        lab_csv = st.file_uploader("Upload Lab Data CSV", type=["csv"], key="lab_csv")

    st.sidebar.markdown("---")
    use_llm_parser = st.sidebar.checkbox("Use Groq LLM Parser", value=(client is not None), help="Use a Large Language Model for parsing the text prompt. Requires API key.")

else: # Default values when manual mode is off
    grade, exposure, cement_choice = "M30", "Severe", "OPC 43"
    nom_max, agg_shape, target_slump = 20, "Angular (baseline)", 125
    use_sp, optimize_cost, fine_zone = True, False, "Zone II"
    qc_level = "Good"
    sg_fa, moisture_fa = 2.65, 1.0
    sg_ca, moisture_ca = 2.70, 0.5
    fine_csv, coarse_csv, lab_csv = None, None, None
    emissions_file, cost_file, materials_file = None, None, None
    use_llm_parser = (client is not None)

with st.sidebar.expander("🎭 Judge Demo Prompts"):
    prompts = {
        "M30 slab, moderate exposure, OPC+Fly Ash": "M30 slab, moderate exposure, OPC+Fly Ash",
        "M40 pumped concrete, severe, GGBS, 150mm slump": "M40 pumped concrete, severe exposure, GGBS, slump 150 mm",
        "A durable concrete mix": "good durable mix"
    }
    for label, text in prompts.items():
        if st.button(label, use_container_width=True, key=f"prompt_{label}"):
            st.session_state.user_text_input = text
            st.rerun()

with st.sidebar.expander("🛠️ Calibration & Tuning (Developer)"):
    enable_calibration_overrides = st.checkbox("Enable calibration overrides", False)
    calib_wb_min = st.number_input("W/B search minimum (wb_min)", 0.30, 0.45, 0.35, 0.01)
    calib_wb_steps = st.slider("W/B search steps (wb_steps)", 3, 15, 6, 1)
    calib_fine_fraction = st.slider("Fine Aggregate Fraction Override", 0.30, 0.50, 0.40, 0.01)
    calib_max_flyash_frac = st.slider("Max Fly Ash fraction", 0.0, 0.5, 0.30, 0.05)
    calib_max_ggbs_frac = st.slider("Max GGBS fraction", 0.0, 0.5, 0.50, 0.05)
    calib_scm_step = st.slider("SCM fraction step (scm_step)", 0.05, 0.25, 0.10, 0.05)


emissions_df, costs_df = load_data(emissions_file, cost_file)

if 'clarification_needed' not in st.session_state: st.session_state.clarification_needed = False
if 'run_generation' not in st.session_state: st.session_state.run_generation = False
if 'final_inputs' not in st.session_state: st.session_state.final_inputs = {}

CLARIFICATION_WIDGETS = {
    "grade": lambda v: st.selectbox("Concrete Grade", list(GRADE_STRENGTH.keys()), index=list(GRADE_STRENGTH.keys()).index(v) if v in GRADE_STRENGTH else 4),
    "exposure": lambda v: st.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()), index=list(EXPOSURE_WB_LIMITS.keys()).index(v) if v in EXPOSURE_WB_LIMITS else 2),
    "target_slump": lambda v: st.slider("Target Slump (mm)", 25, 180, v if isinstance(v, int) else 100, 5),
    "cement_choice": lambda v: st.selectbox("Cement Type", ["OPC 43"], index=0),
    "nom_max": lambda v: st.selectbox("Nominal Max. Aggregate Size (mm)", [10, 12.5, 20, 40], index=[10, 12.5, 20, 40].index(v) if v in [10, 12.5, 20, 40] else 2),
}

if run_button:
    st.session_state.run_generation = False
    st.session_state.clarification_needed = False

    material_props = {'sg_fa': sg_fa, 'moisture_fa': moisture_fa, 'sg_ca': sg_ca, 'moisture_ca': moisture_ca}
    inputs = { "grade": grade, "exposure": exposure, "cement_choice": cement_choice, "nom_max": nom_max, "agg_shape": agg_shape, "target_slump": target_slump, "use_sp": use_sp, "optimize_cost": optimize_cost, "qc_level": qc_level, "fine_zone": fine_zone, "material_props": material_props }

    if user_text.strip() and not manual_mode:
        with st.spinner("🤖 Parsing your request..."):
            inputs, msgs, _ = apply_parser(user_text, inputs)
        if msgs: st.info(" ".join(msgs), icon="💡")

        required_fields = ["grade", "exposure", "target_slump", "nom_max", "cement_choice"]
        missing_fields = [f for f in required_fields if inputs.get(f) is None]

        if missing_fields:
            st.session_state.clarification_needed = True
            st.session_state.final_inputs = inputs
            st.session_state.missing_fields = missing_fields
        else:
            st.session_state.run_generation = True
            st.session_state.final_inputs = inputs
    else:
        st.session_state.run_generation = True
        st.session_state.final_inputs = inputs

if st.session_state.get('clarification_needed', False):
    st.markdown("---")
    st.warning("Your request is missing some details. Please confirm the following to continue.", icon="🤔")
    st.markdown("Please confirm the missing values below. Once submitted, mix design will start automatically.")
    with st.form("clarification_form"):
        st.subheader("Please Clarify Your Requirements")
        current_inputs = st.session_state.final_inputs
        missing_fields_list = st.session_state.missing_fields

        num_cols = min(len(missing_fields_list), 3)
        cols = st.columns(num_cols)
        for i, field in enumerate(missing_fields_list):
            with cols[i % num_cols]:
                widget_func = CLARIFICATION_WIDGETS[field]
                current_value = current_inputs.get(field)
                new_value = widget_func(current_value)
                current_inputs[field] = new_value

        if st.form_submit_button("✅ Confirm & Continue", use_container_width=True, type="primary"):
            st.session_state.final_inputs = current_inputs
            st.session_state.clarification_needed = False
            st.session_state.run_generation = True
            st.rerun()

if st.session_state.get('run_generation', False):
    st.markdown("---")
    try:
        inputs = st.session_state.final_inputs
        min_grade_req = EXPOSURE_MIN_GRADE[inputs["exposure"]]
        grade_order = list(GRADE_STRENGTH.keys())
        if grade_order.index(inputs["grade"]) < grade_order.index(min_grade_req):
            st.warning(f"For **{inputs['exposure']}** exposure, IS 456 recommends a minimum grade of **{min_grade_req}**. The grade has been automatically updated.", icon="⚠️")
            inputs["grade"] = min_grade_req

        calibration_kwargs = {}
        if enable_calibration_overrides:
            calibration_kwargs = {
                "wb_min": calib_wb_min, "wb_steps": calib_wb_steps,
                "max_flyash_frac": calib_max_flyash_frac, "max_ggbs_frac": calib_max_ggbs_frac,
                "scm_step": calib_scm_step, "fine_fraction_override": calib_fine_fraction
            }
            st.info("Developer calibration overrides are enabled.", icon="🛠️")

        with st.spinner("⚙️ Running IS-code calculations and optimizing for sustainability..."):
            fck, S = GRADE_STRENGTH[inputs["grade"]], QC_STDDEV[inputs.get("qc_level", "Good")]
            fck_target = fck + 1.65 * S
            opt_df, opt_meta, trace = generate_mix(
                inputs["grade"], inputs["exposure"], inputs["nom_max"],
                inputs["target_slump"], inputs["agg_shape"], inputs["fine_zone"],
                emissions_df, costs_df, inputs["cement_choice"],
                material_props=inputs["material_props"],
                use_sp=inputs["use_sp"], optimize_cost=inputs["optimize_cost"],
                **calibration_kwargs
            )
            base_df, base_meta = generate_baseline(
                inputs["grade"], inputs["exposure"], inputs["nom_max"], inputs["target_slump"],
                inputs["agg_shape"], inputs["fine_zone"], emissions_df, costs_df,
                inputs["cement_choice"], material_props=inputs["material_props"], use_sp=inputs["use_sp"]
            )

        if opt_df is None or base_df is None:
            st.error("Could not find a feasible mix design with the given constraints. Try adjusting the parameters, such as a higher grade or less restrictive exposure condition.", icon="❌")
            st.dataframe(pd.DataFrame(trace))
        else:
            for m in (opt_meta, base_meta):
                m["fck"], m["fck_target"], m["stddev_S"] = fck, round(fck_target, 1), S
            st.success(f"Successfully generated mix designs for **{inputs['grade']}** concrete in **{inputs['exposure']}** conditions.", icon="✅")

            tab1, tab2, tab_pareto, tab4, tab5, tab6 = st.tabs([
                "📊 **Overview**", "🌱 **Mix Comparison**", "⚖️ **Trade-off Explorer**",
                "📋 **QA/QC & Gradation**", "📥 **Downloads & Reports**", "🔬 **Lab Calibration**"
            ])

            with tab1:
                co2_opt, cost_opt = opt_meta["co2_total"], opt_meta["cost_total"]
                co2_base, cost_base = base_meta["co2_total"], base_meta["cost_total"]
                reduction = (co2_base - co2_opt) / co2_base * 100 if co2_base > 0 else 0.0
                cost_savings = cost_base - cost_opt

                st.subheader("Performance At a Glance")
                c1, c2, c3 = st.columns(3)
                c1.metric("🌱 CO₂ Reduction", f"{reduction:.1f}%", f"{co2_base - co2_opt:.1f} kg/m³ saved", help="Percentage reduction in embodied carbon compared to the standard OPC baseline mix.")
                c2.metric("💰 Cost Savings", f"₹{cost_savings:,.0f} / m³", f"{cost_savings/cost_base*100 if cost_base>0 else 0:.1f}% cheaper", help="Cost difference per cubic meter compared to the baseline mix.")
                c3.metric("♻️ SCM Content", f"{opt_meta['scm_total_frac']*100:.0f}%", f"{base_meta['scm_total_frac']*100:.0f}% in baseline", help="Supplementary Cementitious Materials (Fly Ash, GGBS) replace high-carbon cement.")
                st.markdown("---")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Embodied Carbon (CO₂e)")
                    chart_data = pd.DataFrame({'Mix Type': ['Baseline OPC', 'CivilGPT Optimized'], 'CO₂ (kg/m³)': [co2_base, co2_opt]})
                    fig, ax = plt.subplots(figsize=(6, 4))
                    bars = ax.bar(chart_data['Mix Type'], chart_data['CO₂ (kg/m³)'], color=['#B0BEC5', '#4CAF50'])
                    ax.set_ylabel("Embodied Carbon (kg CO₂e / m³)")
                    ax.bar_label(bars, fmt='{:,.1f}')
                    st.pyplot(fig)
                with col2:
                    st.subheader("💵 Material Cost")
                    chart_data_cost = pd.DataFrame({'Mix Type': ['Baseline OPC', 'CivilGPT Optimized'], 'Cost (₹/m³)': [cost_base, cost_opt]})
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    bars2 = ax2.bar(chart_data_cost['Mix Type'], chart_data_cost['Cost (₹/m³)'], color=['#B0BEC5', '#2196F3'])
                    ax2.set_ylabel("Material Cost (₹ / m³)")
                    ax2.bar_label(bars2, fmt='₹{:,.0f}')
                    st.pyplot(fig2)

                with st.expander("📝 Judge Explanation (How CivilGPT Works)"):
                    st.markdown("""
                    “CivilGPT uses strict IS-code constraints combined with a constrained optimization search to produce construction-ready, low-CO₂ concrete mixes.”
                    
                    “It leverages local material properties and India-specific emission factors so recommendations are context-aware and verifiable.”
                    """)

            def display_mix_details(df, meta, exposure):
                is_feasible, fail_reasons, warnings, _, _ = check_feasibility(df, meta, exposure)
                if is_feasible:
                    st.markdown('<div class="badge-compliant">✅ IS 10262 Compliant</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-non-compliant">❌ Non-Compliant: {", ".join(fail_reasons)}</div>', unsafe_allow_html=True)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("💧 W/B Ratio", f"{meta['w_b']:.3f}", help="Water/Binder Ratio. Lower values generally lead to higher strength.")
                c2.metric("📦 Binder (kg/m³)", f"{meta['cementitious']:.1f}", help="Total mass of cementitious materials (Cement + SCMs).")
                c3.metric("🎯 Target Strength", f"{meta['fck_target']:.1f} MPa", help="f_target = fck + 1.65 * S. Strength the mix is designed for.")
                c4.metric("⚖️ Unit Weight", f"{df['Quantity (kg/m3)'].sum():.1f} kg/m³", help="Total mass per cubic meter.")

                st.markdown("##### Mix Proportions (per m³)")
                styled_df = df.style.format({
                    "Quantity (kg/m3)": "{:.2f}", "CO2_Factor(kg_CO2_per_kg)": "{:.3f}",
                    "CO2_Emissions (kg/m3)": "{:.2f}", "Cost(₹/kg)": "₹{:.2f}", "Cost (₹/m3)": "₹{:,.2f}"
                }).background_gradient(cmap='Greens', subset=['CO2_Emissions (kg/m3)']) \
                  .background_gradient(cmap='Blues', subset=['Cost (₹/m3)'])
                st.dataframe(styled_df, use_container_width=True)

                if warnings:
                    with st.expander("⚠️ View Sanity Check Warnings"):
                        for warning in warnings: st.warning(warning, icon="⚠️")
            
            def display_calculation_walkthrough(meta):
                st.header("Step-by-Step Calculation Walkthrough (IS 10262:2019)")
                st.markdown(f"""This is a summary of how the **Optimized Mix** was designed.
                #### 1. Target Mean Strength
                - **Characteristic Strength (fck):** `{meta['fck']}` MPa (from Grade {meta['grade']})
                - **Assumed Standard Deviation (S):** `{meta['stddev_S']}` MPa (for '{inputs['qc_level']}' quality control)
                - **Target Mean Strength (f'ck):** `fck + 1.65 * S = {meta['fck']} + 1.65 * {meta['stddev_S']} =` **`{meta['fck_target']:.2f}` MPa**
                #### 2. Water Content
                - **Basis:** IS 10262, Table 4, for `{meta['nom_max']}` mm nominal max aggregate size.
                - **Adjustments:** Slump (`{meta['slump']}` mm), aggregate shape ('{inputs['agg_shape']}'), and superplasticizer use.
                - **Final Target Water (SSD basis):** **`{meta['water_target']:.1f}` kg/m³**
                #### 3. Water-Binder (w/b) Ratio
                - **Constraint:** Maximum w/b ratio for `{meta['exposure']}` exposure is `{EXPOSURE_WB_LIMITS[meta['exposure']]}`.
                - **Optimizer Selection:** The optimizer selected the lowest w/b ratio that resulted in a feasible, low-carbon mix.
                - **Selected w/b Ratio:** **`{meta['w_b']:.3f}`**
                #### 4. Binder Content
                - **Initial Binder (from w/b):** `{meta['water_target']:.1f} / {meta['w_b']:.3f} = {(meta['water_target']/meta['w_b']):.1f}` kg/m³
                - **Constraints Check:**
                    - Min. for `{meta['exposure']}` exposure: `{EXPOSURE_MIN_CEMENT[meta['exposure']]}` kg/m³
                    - Typical range for `{meta['grade']}`: `{meta['binder_range'][0]}` - `{meta['binder_range'][1]}` kg/m³
                - **Final Adjusted Binder Content:** **`{meta['cementitious']:.1f}` kg/m³**
                #### 5. SCM & Cement Content
                - **Optimizer Goal:** Minimize CO₂/cost by replacing cement with SCMs (Fly Ash, GGBS).
                - **Selected SCM Fraction:** `{meta['scm_total_frac']*100:.0f}%`
                - **Material Quantities:**
                    - **Cement:** `{meta['cement']:.1f}` kg/m³ | **Fly Ash:** `{meta['flyash']:.1f}` kg/m³ | **GGBS:** `{meta['ggbs']:.1f}` kg/m³
                #### 6. Aggregate Proportioning (IS 10262, Table 5)
                - **Basis:** Volume of coarse aggregate for `{meta['nom_max']}` mm aggregate and fine aggregate `{inputs['fine_zone']}`.
                - **Adjustment:** Corrected for the final w/b ratio of `{meta['w_b']:.3f}`.
                - **Coarse Aggregate Fraction (by volume):** **`{meta['coarse_agg_fraction']:.3f}`**
                #### 7. Final Quantities (with Moisture Correction)
                - **Moisture Correction:** Adjusted for `{meta['material_props']['moisture_fa']}%` free moisture in fine and `{meta['material_props']['moisture_ca']}%` in coarse aggregate.
                - **Final Batch Weights:**
                    - **Water:** **`{meta['water_final']:.1f}` kg/m³** | **Fine Aggregate:** **`{meta['fine']:.1f}` kg/m³** | **Coarse Aggregate:** **`{meta['coarse']:.1f}` kg/m³**
                """)

            with tab2:
                col_opt, col_base = st.columns(2)
                with col_opt:
                    st.subheader("🌱 Optimized Low-Carbon Mix Design")
                    display_mix_details(opt_df, opt_meta, inputs['exposure'])
                with col_base:
                    st.subheader("🏗️ Standard OPC Baseline Mix")
                    display_mix_details(base_df, base_meta, inputs['exposure'])
                st.markdown("---")
                if st.toggle("📖 Show Step-by-Step IS Calculation for Optimized Mix", key="toggle_walkthrough_tab2"):
                    display_calculation_walkthrough(opt_meta)

            with tab_pareto:
                st.header("Cost vs. Carbon Trade-off Analysis")
                st.markdown("This chart displays all IS-code compliant mixes found. The **blue line** is the **Pareto Front**—the set of most efficient mixes where you can't improve one objective (e.g., lower CO₂) without worsening the other (e.g., increasing cost).")
                if trace:
                    trace_df = pd.DataFrame(trace)
                    feasible_mixes = trace_df[trace_df['feasible']].copy()
                    if not feasible_mixes.empty:
                        pareto_df = pareto_front(feasible_mixes, x_col="cost", y_col="co2")
                        best_compromise_mix = None
                        if not pareto_df.empty:
                            alpha = st.slider( "Prioritize Sustainability (CO₂) ↔ Cost", 0.0, 1.0, 0.5, 0.05, help="Slide to find the best compromise on the Pareto Front for your preference.")
                            
                            pareto_df_norm = pareto_df.copy()
                            cost_min, cost_max = pareto_df_norm['cost'].min(), pareto_df_norm['cost'].max()
                            co2_min, co2_max = pareto_df_norm['co2'].min(), pareto_df_norm['co2'].max()
                            
                            pareto_df_norm['norm_cost'] = 0.0 if (cost_max - cost_min) == 0 else (pareto_df_norm['cost'] - cost_min) / (cost_max - cost_min)
                            pareto_df_norm['norm_co2'] = 0.0 if (co2_max - co2_min) == 0 else (pareto_df_norm['co2'] - co2_min) / (co2_max - co2_min)
                            pareto_df_norm['score'] = alpha * pareto_df_norm['norm_co2'] + (1 - alpha) * pareto_df_norm['norm_cost']
                            best_compromise_mix = pareto_df_norm.loc[pareto_df_norm['score'].idxmin()]

                        pareto_fig = generate_pareto_plot_fig(feasible_mixes, pareto_df, opt_meta, best_compromise_mix, optimize_for)
                        st.pyplot(pareto_fig)

                        if best_compromise_mix is not None:
                            st.markdown("---")
                            st.subheader("Details of Selected 'Best Compromise' Mix")
                            st.info("This point represents the optimal trade-off on the Pareto Front based on your current slider preference.", icon="🎯")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("💰 Cost", f"₹{best_compromise_mix['cost']:.0f} / m³")
                            c2.metric("🌱 CO₂", f"{best_compromise_mix['co2']:.1f} kg / m³")
                            c3.metric("💧 W/B Ratio", f"{best_compromise_mix['wb']:.3f}")
                            c4.metric("♻️ SCM (Fly Ash %)", f"{best_compromise_mix['flyash_frac']*100:.0f}%")
                    else:
                        st.warning("No feasible mixes were found by the optimizer, so no trade-off plot can be generated.", icon="⚠️")
                else:
                    st.error("Optimizer trace data is missing.", icon="❌")

            with tab4:
                st.header("Quality Assurance & Sieve Analysis (IS 383)")
                sample_fa_data = "Sieve_mm,PercentPassing\n10.0,100\n4.75,95\n2.36,80\n1.18,60\n0.600,40\n0.300,15\n0.150,5"
                sample_ca_data = "Sieve_mm,PercentPassing\n40.0,100\n20.0,98\n10.0,40\n4.75,5"
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Fine Aggregate Gradation")
                    if fine_csv is not None:
                        try:
                            df_fine = pd.read_csv(fine_csv)
                            ok_fa, msgs_fa = sieve_check_fa(df_fine, inputs.get("fine_zone", "Zone II"))
                            if ok_fa: st.success(msgs_fa[0], icon="✅")
                            else:
                                for m in msgs_fa: st.error(m, icon="❌")
                            fig_fa = plot_gradation_chart(df_fine, FINE_AGG_ZONE_LIMITS[inputs['fine_zone']], "Fine Aggregate: Uploaded vs. IS 383 Limits", f"({inputs['fine_zone']})")
                            st.pyplot(fig_fa)
                        except Exception as e: st.error(f"Could not process Fine Aggregate CSV: {e}")
                    else:
                        st.info("Upload a Fine Aggregate CSV in the sidebar to perform a gradation check.", icon="ℹ️")
                        st.download_button("Download Sample FA CSV", sample_fa_data, "sample_fine_aggregate.csv", "text/csv")
                with col2:
                    st.subheader("Coarse Aggregate Gradation")
                    if coarse_csv is not None:
                        try:
                            df_coarse = pd.read_csv(coarse_csv)
                            ok_ca, msgs_ca = sieve_check_ca(df_coarse, inputs["nom_max"])
                            if ok_ca: st.success(msgs_ca[0], icon="✅")
                            else:
                                for m in msgs_ca: st.error(m, icon="❌")
                            fig_ca = plot_gradation_chart(df_coarse, COARSE_LIMITS[int(inputs['nom_max'])], f"Coarse Aggregate ({inputs['nom_max']}mm): Uploaded vs. IS 383 Limits")
                            st.pyplot(fig_ca)
                        except Exception as e: st.error(f"Could not process Coarse Aggregate CSV: {e}")
                    else:
                        st.info("Upload a Coarse Aggregate CSV in the sidebar to perform a gradation check.", icon="ℹ️")
                        st.download_button("Download Sample CA CSV", sample_ca_data, "sample_coarse_aggregate.csv", "text/csv")
                
                with st.expander("🔬 View Optimizer Trace (Advanced)"):
                    if trace:
                        st.markdown("The table below shows every mix combination attempted by the optimizer. 'Feasible' mixes met all IS-code checks.")
                        st.dataframe(pd.DataFrame(trace).style.apply(lambda s: ['background-color: #e8f5e9' if v else 'background-color: #ffebee' for v in s], subset=['feasible']), use_container_width=True)
                    else: st.info("Trace not available.")

            with tab5:
                st.header("Download Reports")
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    comp_df = opt_df[['Material', 'Quantity (kg/m3)']].copy().rename(columns={'Quantity (kg/m3)': 'Optimized Mix (kg/m3)'})
                    comp_df = comp_df.merge(base_df[['Material', 'Quantity (kg/m3)']], on='Material', how='outer').rename(columns={'Quantity (kg/m3)': 'Baseline Mix (kg/m3)'}).fillna(0)
                    comp_df.to_excel(writer, sheet_name="Side-by-Side Comparison", index=False)
                    opt_df.to_excel(writer, sheet_name="Optimized_Mix_Details", index=False)
                    base_df.to_excel(writer, sheet_name="Baseline_Mix_Details", index=False)
                excel_buffer.seek(0)
                
                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=(8.5*inch, 11*inch), topMargin=0.5*inch, bottomMargin=0.5*inch)
                styles = getSampleStyleSheet()
                story = [Paragraph("CivilGPT Sustainable Mix Report", styles['h1']), Spacer(1, 0.2*inch)]
                story.append(Paragraph(f"<b>Design for:</b> {inputs['grade']} / {inputs['exposure']} Exposure / {inputs['target_slump']}mm Slump", styles['h2']))

                data_opt = [["Optimized Mix (kg/m³)",""]] + [[k, f"{v:.2f}"] for k,v in opt_df.set_index('Material')['Quantity (kg/m3)'].items() if v>0]
                data_base = [["Baseline Mix (kg/m³)",""]] + [[k, f"{v:.2f}"] for k,v in base_df.set_index('Material')['Quantity (kg/m3)'].items() if v>0]
                
                max_len = max(len(data_opt), len(data_base))
                data_opt.extend([["", ""]] * (max_len - len(data_opt)))
                data_base.extend([["", ""]] * (max_len - len(data_base)))
                
                table_data = [[Paragraph(f"<b>{opt[0]}</b>", styles['Normal']), opt[1], "", Paragraph(f"<b>{base[0]}</b>", styles['Normal']), base[1]] for opt, base in zip(data_opt, data_base)]

                pdf_table = Table(table_data, hAlign='LEFT', colWidths=[2.0*inch, 1.0*inch, 0.5*inch, 2.0*inch, 1.0*inch])
                pdf_table.setStyle(TableStyle([
                    ('GRID', (0,0), (1,-1), 1, colors.black), ('GRID', (3,0), (4,-1), 1, colors.black),
                    ('BACKGROUND', (0,0), (1,0), colors.palegreen), ('BACKGROUND', (3,0), (4,0), colors.lightgrey),
                ]))
                story.extend([Spacer(1, 0.2*inch), pdf_table, Spacer(1, 0.2*inch)])

                if 'trace' in locals() and trace and not feasible_mixes.empty:
                    pareto_df_pdf = pareto_front(feasible_mixes, x_col="cost", y_col="co2")
                    pdf_pareto_fig = generate_pareto_plot_fig(feasible_mixes, pareto_df_pdf, opt_meta, None, optimize_for)
                    img_buffer = BytesIO()
                    pdf_pareto_fig.savefig(img_buffer, format='PNG', bbox_inches='tight', dpi=150)
                    img_buffer.seek(0)
                    story.append(Image(img_buffer, width=6*inch, height=4*inch))

                doc.build(story)
                pdf_buffer.seek(0)

                d1, d2 = st.columns(2)
                d1.download_button("📄 Download PDF Report", pdf_buffer, "CivilGPT_Report.pdf", "application/pdf", use_container_width=True)
                d2.download_button("📈 Download Excel Report", excel_buffer, "CivilGPT_Mix_Designs.xlsx", "application/vnd.ms-excel", use_container_width=True)

            with tab6:
                st.header("🔬 Lab Calibration Analysis")
                if lab_csv is not None:
                    try:
                        lab_results_df = pd.read_csv(lab_csv)
                        comparison_df, error_metrics = run_lab_calibration(lab_results_df)

                        if comparison_df is not None and not comparison_df.empty:
                            st.subheader("Error Metrics")
                            st.markdown("Comparing lab-tested 28-day strength against the IS code's required target strength (`f_target = fck + 1.65 * S`).")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Mean Absolute Error (MAE)", f"{error_metrics['Mean Absolute Error (MPa)']:.2f} MPa")
                            m2.metric("Root Mean Squared Error (RMSE)", f"{error_metrics['Root Mean Squared Error (MPa)']:.2f} MPa")
                            m3.metric("Mean Bias (Over/Under-prediction)", f"{error_metrics['Mean Bias (MPa)']:.2f} MPa")
                            st.markdown("---")
                            st.subheader("Comparison: Lab vs. Predicted Target Strength")
                            st.dataframe(comparison_df.style.format({
                                "Lab Strength (MPa)": "{:.2f}", "Predicted Target Strength (MPa)": "{:.2f}", "Error (MPa)": "{:+.2f}"
                            }).background_gradient(cmap='RdYlGn_r', subset=['Error (MPa)']), use_container_width=True)
                            
                            st.subheader("Prediction Accuracy Scatter Plot")
                            fig, ax = plt.subplots()
                            ax.scatter(comparison_df["Lab Strength (MPa)"], comparison_df["Predicted Target Strength (MPa)"], alpha=0.7, label="Data Points", edgecolors='k')
                            lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
                            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Perfect Prediction (y=x)")
                            ax.set_xlabel("Actual Lab Strength (MPa)"); ax.set_ylabel("Predicted Target Strength (MPa)")
                            ax.set_title("Lab Strength vs. Predicted Target Strength"); ax.legend(); ax.grid(True)
                            st.pyplot(fig)
                        else:
                            st.warning("Could not process the uploaded lab data CSV. Please check the file format and data.", icon="⚠️")
                    except Exception as e:
                        st.error(f"Failed to process lab data: {e}", icon="💥")
                else:
                    st.info("Upload a lab data CSV in the sidebar to compare CivilGPT's target strength calculations against your real-world results.", icon="ℹ️")

    except Exception as e:
        st.error(f"An unexpected error occurred during mix design generation. Please check your inputs.", icon="💥")
        st.code(traceback.format_exc())
    finally:
        st.session_state.run_generation = False

elif not st.session_state.get('clarification_needed'):
    st.info("Enter your concrete requirements in the prompt box above, or switch to manual mode to specify parameters.", icon="👆")
    st.markdown("---")
    st.subheader("How It Works")
    st.markdown("""
    1.  **Input Requirements**: Describe your project needs in plain English (e.g., "M25 concrete for moderate exposure") or use the manual sidebar for detailed control.
    2.  **IS Code Compliance**: The app generates dozens of candidate mixes, ensuring each one adheres to the durability and strength requirements of Indian Standards **IS 10262** and **IS 456**.
    3.  **Sustainability Optimization**: It then calculates the embodied carbon (CO₂e) and cost for every compliant mix.
    4.  **Best Mix Selection**: Finally, it presents the mix with the lowest carbon footprint (or cost) alongside a standard OPC baseline for comparison.
    """)
