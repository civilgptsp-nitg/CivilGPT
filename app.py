# app.py — CivilGPT v2.1 (Refactored UI & IS-Code Compliant Logic)
# Backend logic preserved from v2.0
# UI refactored for a professional, modern, and intuitive experience
# Clarification step for free-text parsing added.
# v2.2 - Corrected aggregate proportioning logic to align with IS 10262:2019, Table 5.
# v2.3 - Added developer calibration panel to tune optimizer search parameters.
# v2.4 - Added Lab Calibration Dataset Upload + Error Analysis feature.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
import json
import traceback
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ==============================================================================
# PART 1: BACKEND LOGIC (CORRECTED & ENHANCED)
# ==============================================================================

# Groq client (optional)
try:
    from groq import Groq
    client = Groq(api_key=st.secrets.get("GROQ_API_KEY", None))
except Exception:
    client = None

# App Config
st.set_page_config(
    page_title="CivilGPT - Sustainable Concrete Mix Designer",
    page_icon="🧱",
    layout="wide"
)

# Dataset Path Handling
LAB_FILE = "lab_processed_mgrades_only.xlsx"
MIX_FILE = "concrete_mix_design_data_cleaned_standardized.xlsx"

def safe_load_excel(name):
    for p in [name, f"data/{name}"]:
        if os.path.exists(p):
            try:
                return pd.read_excel(p)
            except Exception:
                try:
                    return pd.read_excel(p, engine="openpyxl")
                except Exception:
                    return None
    return None

lab_df = safe_load_excel(LAB_FILE)
mix_df = safe_load_excel(MIX_FILE)


# --- IS Code Rules & Tables (IS 456 & IS 10262) ---
EXPOSURE_WB_LIMITS = {"Mild": 0.60,"Moderate": 0.55,"Severe": 0.50,"Very Severe": 0.45,"Marine": 0.40}
EXPOSURE_MIN_CEMENT = {"Mild": 300, "Moderate": 300, "Severe": 320,"Very Severe": 340, "Marine": 360}
EXPOSURE_MIN_GRADE = {"Mild": "M20", "Moderate": "M25", "Severe": "M30","Very Severe": "M35", "Marine": "M40"}
GRADE_STRENGTH = {"M10": 10, "M15": 15, "M20": 20, "M25": 25,"M30": 30, "M35": 35, "M40": 40, "M45": 45, "M50": 50}
WATER_BASELINE = {10: 208, 12.5: 202, 20: 186, 40: 165} # IS 10262, Table 4 (for 50mm slump)
AGG_SHAPE_WATER_ADJ = {"Angular (baseline)": 0.00, "Sub-angular": -0.03,"Sub-rounded": -0.05, "Rounded": -0.07,"Flaky/Elongated": +0.03}
QC_STDDEV = {"Good": 5.0, "Fair": 7.5, "Poor": 10.0} # IS 10262, Table 2

# NEW: IS 10262:2019, Table 5 - Volume of Coarse Aggregate per unit volume of Total Aggregate
# FIXED: Added 12.5 as a key to prevent KeyError
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
    20: {"40.0": (95,100),  "20.0": (95,100),  "10.0": (25,55), "4.75": (0,10)},
    40: {"80.0": (95,100),  "40.0": (95,100),  "20.0": (30,70), "10.0": (0,15)}
}

# Parsers
def simple_parse(text: str) -> dict:
    result = {}
    grade_match = re.search(r"\bM\s*(10|15|20|25|30|35|40|45|50)\b", text, re.IGNORECASE)
    if grade_match: result["grade"] = "M" + grade_match.group(1)
    for exp in EXPOSURE_WB_LIMITS.keys():
        if re.search(exp, text, re.IGNORECASE): result["exposure"] = exp; break
    slump_match = re.search(r"(slump\s*(of\s*)?|)\b(\d{2,3})\s*mm", text, re.IGNORECASE)
    if slump_match: result["slump"] = int(slump_match.group(3))
    cement_types = ["OPC 33", "OPC 43", "OPC 53", "PPC"]
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
def _read_csv_try(path): return pd.read_csv(path)

@st.cache_data
def load_data(materials_file=None, emissions_file=None, cost_file=None):
    def _safe_read(file, default):
        if file is not None:
            try: return pd.read_csv(file)
            except: return default
        return default
    materials = _safe_read(materials_file, None)
    emissions = _safe_read(emissions_file, None)
    costs = _safe_read(cost_file, None)
    if materials is None:
        try: materials = _read_csv_try("materials_library.csv")
        except: materials = pd.DataFrame(columns=["Material"])
    if emissions is None:
        try: emissions = _read_csv_try("emission_factors.csv")
        except: emissions = pd.DataFrame(columns=["Material","CO2_Factor(kg_CO2_per_kg)"])
    if costs is None:
        try: costs = _read_csv_try("cost_factors.csv")
        except: costs = pd.DataFrame(columns=["Material","Cost(₹/kg)"])
    return materials, emissions, costs

def water_for_slump_and_shape(nom_max_mm: int, slump_mm: int,
                                agg_shape: str, uses_sp: bool=False,
                                sp_reduction_frac: float=0.0) -> float:
    base = WATER_BASELINE.get(int(nom_max_mm), 186.0)
    # IS 10262: Increase water by ~3% for every 25mm slump increase over 50mm
    if slump_mm <= 50: water = base
    else: water = base * (1 + 0.03 * ((slump_mm - 50) / 25.0))
    water *= (1.0 + AGG_SHAPE_WATER_ADJ.get(agg_shape, 0.0))
    if uses_sp and sp_reduction_frac > 0: water *= (1 - sp_reduction_frac)
    return float(water)

# NEW: Helper function to get coarse aggregate fraction as per IS 10262, Table 5
def get_coarse_agg_fraction(nom_max_mm: float, fa_zone: str, wb_ratio: float):
    """
    Calculates the volume of coarse aggregate per unit volume of total aggregate.
    Adjusts for w/b ratio as per IS 10262:2019 note in Table 5.
    The baseline w/b ratio for Table 5 is 0.5.
    """
    # Get the baseline coarse aggregate volume fraction from the table
    base_fraction = COARSE_AGG_FRAC_BY_ZONE.get(nom_max_mm, {}).get(fa_zone, 0.62) # Default to a reasonable value if not found

    # Adjust for w/b ratio.
    # The code suggests increasing the fraction by 0.01 for every decrease of 0.05 in w/b ratio.
    # And vice-versa for an increase in w/b ratio.
    wb_diff = 0.50 - wb_ratio
    correction = (wb_diff / 0.05) * 0.01
    
    corrected_fraction = base_fraction + correction
    
    # Ensure fraction is within reasonable bounds (e.g., 0.4 to 0.8)
    return max(0.4, min(0.8, corrected_fraction))

# NEW: Function for lab calibration analysis
def run_lab_calibration(lab_df):
    """
    Compares lab-tested strengths against CivilGPT's IS-code based target strength.
    """
    results = []
    # Assume "Good" Quality Control for standard deviation as per the app's default
    default_qc_level = "Good"
    std_dev_S = QC_STDDEV[default_qc_level]

    for _, row in lab_df.iterrows():
        try:
            # Extract inputs from the lab data row
            grade = str(row['grade']).strip()
            actual_strength = float(row['actual_strength'])

            # Validate grade and get characteristic strength (fck)
            if grade not in GRADE_STRENGTH:
                continue # Skip rows with invalid grade
            fck = GRADE_STRENGTH[grade]

            # CivilGPT's prediction is the target strength required by IS code
            predicted_strength = fck + 1.65 * std_dev_S

            results.append({
                "Grade": grade,
                "Exposure": row.get('exposure', 'N/A'),
                "Slump (mm)": row.get('slump', 'N/A'),
                "Lab Strength (MPa)": actual_strength,
                "Predicted Target Strength (MPa)": predicted_strength,
                "Error (MPa)": predicted_strength - actual_strength
            })
        except (KeyError, ValueError, TypeError):
            # Skip rows with malformed data (e.g., non-numeric strength)
            pass

    if not results:
        return None, {}

    results_df = pd.DataFrame(results)

    # Calculate error metrics
    mae = results_df["Error (MPa)"].abs().mean()
    rmse = np.sqrt((results_df["Error (MPa)"] ** 2).mean())
    bias = results_df["Error (MPa)"].mean()

    metrics = {"Mean Absolute Error (MPa)": mae, "Root Mean Squared Error (MPa)": rmse, "Mean Bias (MPa)": bias}

    return results_df, metrics

# ==============================================================================
# PART 2: CORE MIX LOGIC (UPDATED)
# ==============================================================================

def evaluate_mix(components_dict, emissions_df, costs_df=None):
    comp_items = [(m.strip().lower(), q) for m, q in components_dict.items()]
    comp_df = pd.DataFrame(comp_items, columns=["Material_norm", "Quantity (kg/m3)"])
    emissions_df = emissions_df.copy()
    emissions_df["Material_norm"] = emissions_df["Material"].str.strip().str.lower()
    df = comp_df.merge(emissions_df[["Material_norm","CO2_Factor(kg_CO2_per_kg)"]],
                         on="Material_norm", how="left")
    if "CO2_Factor(kg_CO2_per_kg)" not in df.columns:
        df["CO2_Factor(kg_CO2_per_kg)"] = 0.0
    df["CO2_Factor(kg_CO2_per_kg)"] = df["CO2_Factor(kg_CO2_per_kg)"].fillna(0.0)
    df["CO2_Emissions (kg/m3)"] = df["Quantity (kg/m3)"] * df["CO2_Factor(kg_CO2_per_kg)"]
    if costs_df is not None and "Cost(₹/kg)" in costs_df.columns:
        costs_df = costs_df.copy()
        costs_df["Material_norm"] = costs_df["Material"].str.strip().str.lower()
        df = df.merge(costs_df[["Material_norm","Cost(₹/kg)"]], on="Material_norm", how="left")
        df["Cost(₹/kg)"] = df["Cost(₹/kg)"].fillna(0.0)
        df["Cost (₹/m3)"] = df["Quantity (kg/m3)"] * df["Cost(₹/kg)"]
    else:
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
        "GGBS (kg/m³)": round(float(meta.get("ggbs", 0.0)), 1),
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
        unit_wt = float(df["Quantity (kg/m3)"].sum())
    except Exception: return ["Insufficient data to run sanity checks."]
    if cement < 250: warnings.append(f"Low cement content ({cement:.1f} kg/m³). May affect durability.")
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
    except: return False, ["Invalid fine aggregate CSV format. Ensure 'Sieve_mm' and 'PercentPassing' columns exist."]

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
    except: return False, ["Invalid coarse aggregate CSV format. Ensure 'Sieve_mm' and 'PercentPassing' columns exist."]

# Calibration overrides added: wb_min, wb_steps, max_flyash_frac, max_ggbs_frac, scm_step, fine_fraction_override
def generate_mix(grade, exposure, nom_max, target_slump, agg_shape, fine_zone, emissions, costs, cement_choice, use_sp=True, sp_reduction=0.18, optimize_cost=False, wb_min=0.35, wb_steps=6, max_flyash_frac=0.3, max_ggbs_frac=0.5, scm_step=0.1, fine_fraction_override=None):
    w_b_limit, min_cem = float(EXPOSURE_WB_LIMITS[exposure]), float(EXPOSURE_MIN_CEMENT[exposure])
    target_water = water_for_slump_and_shape(nom_max_mm=nom_max, slump_mm=int(target_slump), agg_shape=agg_shape, uses_sp=use_sp, sp_reduction_frac=sp_reduction)
    best_df, best_meta, best_score = None, None, float("inf")
    trace = []
    
    # Use calibrated optimizer values
    wb_values = np.linspace(float(wb_min), float(w_b_limit), int(wb_steps))
    flyash_options = np.arange(0.0, max_flyash_frac + 1e-9, scm_step)
    ggbs_options = np.arange(0.0, max_ggbs_frac + 1e-9, scm_step)

    for wb in wb_values:
        for flyash_frac in flyash_options:
            for ggbs_frac in ggbs_options:
                if flyash_frac + ggbs_frac > 0.50: continue
                binder = max(target_water / wb, min_cem)
                cement, flyash, ggbs = binder * (1 - flyash_frac - ggbs_frac), binder * flyash_frac, binder * ggbs_frac
                sp = 0.01 * binder if use_sp else 0.0 # Typical SP dosage is ~1% of binder
                
                # Check for developer override of aggregate proportioning
                if fine_fraction_override is not None:
                    coarse_agg_frac = 1.0 - fine_fraction_override
                else:
                    # IS-Code Compliant Logic
                    coarse_agg_frac = get_coarse_agg_fraction(nom_max, fine_zone, wb)
                
                fine, coarse = compute_aggregates(binder, target_water, sp, coarse_agg_frac)

                mix = {cement_choice: cement,"Fly Ash": flyash,"GGBS": ggbs,"Water": target_water,"PCE Superplasticizer": sp,"Fine Aggregate": fine,"Coarse Aggregate": coarse}
                df = evaluate_mix(mix, emissions, costs)
                co2_total, cost_total = float(df["CO2_Emissions (kg/m3)"].sum()), float(df["Cost (₹/m3)"].sum())
                candidate_meta = {"w_b": wb, "cementitious": binder, "cement": cement, "flyash": flyash, "ggbs": ggbs, "water_target": target_water, "sp": sp, "fine": fine, "coarse": coarse, "scm_total_frac": flyash_frac + ggbs_frac, "grade": grade, "exposure": exposure, "nom_max": nom_max, "slump": target_slump, "co2_total": co2_total, "cost_total": cost_total}
                feasible, reasons_fail, _, _, _ = check_feasibility(df, candidate_meta, exposure)
                score = co2_total if not optimize_cost else cost_total
                trace.append({"wb": float(wb), "flyash_frac": float(flyash_frac), "ggbs_frac": float(ggbs_frac),"co2": float(co2_total), "cost": float(cost_total),"score": float(score), "feasible": bool(feasible),"reasons": ", ".join(reasons_fail)})
                if feasible and score < best_score:
                    best_df, best_score, best_meta = df.copy(), score, candidate_meta.copy()
    return best_df, best_meta, trace

def generate_baseline(grade, exposure, nom_max, target_slump, agg_shape, fine_zone, emissions, costs, cement_choice, use_sp=True, sp_reduction=0.18):
    w_b_limit, min_cem = float(EXPOSURE_WB_LIMITS[exposure]), float(EXPOSURE_MIN_CEMENT[exposure])
    water_target = water_for_slump_and_shape(nom_max_mm=nom_max, slump_mm=int(target_slump), agg_shape=agg_shape, uses_sp=use_sp, sp_reduction_frac=sp_reduction)
    cementitious = max(water_target / w_b_limit, min_cem)
    sp = 0.01 * cementitious if use_sp else 0.0

    # UPDATED LOGIC: Get aggregate proportions from IS Code method
    coarse_agg_frac = get_coarse_agg_fraction(nom_max, fine_zone, w_b_limit)
    fine, coarse = compute_aggregates(cementitious, water_target, sp, coarse_agg_frac)
    
    mix = {cement_choice: cementitious,"Fly Ash": 0.0,"GGBS": 0.0,"Water": water_target, "PCE Superplasticizer": sp,"Fine Aggregate": fine,"Coarse Aggregate": coarse}
    df = evaluate_mix(mix, emissions, costs)
    meta = {"w_b": w_b_limit, "cementitious": cementitious, "cement": cementitious, "flyash": 0.0, "ggbs": 0.0, "water_target": water_target, "sp": sp, "fine": fine, "coarse": coarse, "scm_total_frac": 0.0, "grade": grade, "exposure": exposure, "nom_max": nom_max, "slump": target_slump, "co2_total": float(df["CO2_Emissions (kg/m3)"].sum()), "cost_total": float(df["Cost (₹/m3)"].sum())}
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

# --- Page Styling ---
st.markdown("""
<style>
    /* Center the title and main interface elements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 100%;
    }
    /* Style the main text area like a prompt box */
    .stTextArea [data-baseweb=base-input] {
        border-color: #4A90E2;
        box-shadow: 0 0 5px #4A90E2;
    }
</style>
""", unsafe_allow_html=True)


# --- Landing Page / Main Interface ---
st.title("🧱 CivilGPT: Sustainable Concrete Mix Designer")
st.markdown("##### An AI-powered tool for creating **IS 10262:2019 compliant** concrete mixes, optimized for low carbon footprint.")

# Main input area
col1, col2 = st.columns([0.7, 0.3])
with col1:
    user_text = st.text_area(
        "**Describe Your Requirements**",
        height=100,
        placeholder="e.g., Design an M30 grade concrete for severe exposure using OPC 53. Target a slump of 125 mm with 20 mm aggregates.",
        label_visibility="collapsed"
    )
with col2:
    st.write("") # for vertical alignment
    st.write("")
    run_button = st.button("🚀 Generate Mix Design", use_container_width=True, type="primary")

manual_mode = st.toggle("⚙️ Switch to Advanced Manual Input")

# --- Sidebar for Manual Inputs ---
if manual_mode:
    st.sidebar.header("📝 Manual Mix Inputs")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Core Requirements")
    grade = st.sidebar.selectbox("Concrete Grade", list(GRADE_STRENGTH.keys()), index=4, help="Target characteristic compressive strength at 28 days.")
    exposure = st.sidebar.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()), index=2, help="Determines durability requirements like min. cement content and max. water-binder ratio as per IS 456.")
    
    st.sidebar.subheader("Workability & Materials")
    target_slump = st.sidebar.slider("Target Slump (mm)", 25, 180, 100, 5, help="Specifies the desired consistency and workability of the fresh concrete.")
    cement_choice = st.sidebar.selectbox("Cement Type", ["OPC 33", "OPC 43", "OPC 53", "PPC"], index=2, help="Type of Ordinary Portland Cement or Pozzolana Portland Cement.")
    nom_max = st.sidebar.selectbox("Nominal Max. Aggregate Size (mm)", [10, 12.5, 20, 40], index=2, help="Largest practical aggregate size, influences water demand.")
    agg_shape = st.sidebar.selectbox("Coarse Aggregate Shape", list(AGG_SHAPE_WATER_ADJ.keys()), index=0, help="Shape affects water demand; angular requires more water than rounded.")
    # UPDATED: Fine aggregate zone is now a direct input for the IS-code calculation
    fine_zone = st.sidebar.selectbox("Fine Aggregate Zone (IS 383)", ["Zone I","Zone II","Zone III","Zone IV"], index=1, help="Grading zone as per IS 383. This is crucial for determining aggregate proportions per IS 10262.")

    st.sidebar.subheader("Optimization & Admixtures")
    use_sp = st.sidebar.checkbox("Use Superplasticizer (PCE)", True, help="Chemical admixture to increase workability or reduce water content.")
    optimize_for = st.sidebar.radio("Optimize For", ["Lowest CO₂", "Lowest Cost"], help="The optimizer will prioritize finding a feasible mix that minimizes either carbon emissions or material cost.")
    optimize_cost = (optimize_for == "Lowest Cost")

    st.sidebar.subheader("Advanced Parameters")
    with st.sidebar.expander("QA/QC"):
        qc_level = st.selectbox("Quality Control Level", list(QC_STDDEV.keys()), index=0, help="Assumed site quality control, affecting the target strength calculation (f_target = fck + 1.65 * S).")

    st.sidebar.subheader("File Uploads (Optional)")
    with st.sidebar.expander("Upload Material Data & Gradation"):
        st.markdown("###### Sieve Analysis (IS 383)")
        fine_csv = st.file_uploader("Fine Aggregate CSV", type=["csv"], key="fine_csv", help="CSV with 'Sieve_mm' and 'PercentPassing' columns.")
        coarse_csv = st.file_uploader("Coarse Aggregate CSV", type=["csv"], key="coarse_csv", help="CSV with 'Sieve_mm' and 'PercentPassing' columns.")
        
        st.markdown("###### Cost & Emissions Data")
        emissions_file = st.file_uploader("Emission Factors (kgCO₂/kg)", type=["csv"], key="emissions_csv")
        cost_file = st.file_uploader("Cost Factors (₹/kg)", type=["csv"], key="cost_csv")

    # NEW: Expander for Lab Calibration
    with st.sidebar.expander("🔬 Lab Calibration Dataset"):
        st.markdown("""
        Upload a CSV with lab results to compare against CivilGPT's predictions.
        **Required columns:**
        - `grade` (e.g., M30)
        - `exposure` (e.g., Severe)
        - `slump` (mm)
        - `nom_max` (mm)
        - `cement_choice` (e.g., OPC 53)
        - `actual_strength` (MPa)
        """)
        lab_csv = st.file_uploader("Upload Lab Data CSV", type=["csv"], key="lab_csv")

    st.sidebar.markdown("---")
    use_llm_parser = st.sidebar.checkbox("Use Groq LLM Parser", value=False, help="Use a Large Language Model for parsing the text prompt. Requires API key.")

else: # Default values when manual mode is off
    grade, exposure, cement_choice = "M30", "Severe", "OPC 53"
    nom_max, agg_shape, target_slump = 20, "Angular (baseline)", 125
    use_sp, optimize_cost, fine_zone = True, False, "Zone II"
    qc_level = "Good"
    fine_csv, coarse_csv, lab_csv = None, None, None
    emissions_file, cost_file = None, None
    use_llm_parser = False

# NEW: Calibration controls, always visible
with st.sidebar.expander("Calibration & Tuning (Developer)"):
    enable_calibration_overrides = st.checkbox("Enable calibration overrides", False, help="Override default optimizer search parameters with the values below.")
    calib_wb_min = st.number_input("W/B search minimum (wb_min)", 0.30, 0.45, 0.35, 0.01, help="Lower bound for the Water/Binder ratio search space.")
    calib_wb_steps = st.slider("W/B search steps (wb_steps)", 3, 15, 6, 1, help="Number of W/B ratios to test between min and the exposure limit.")
    calib_fine_fraction = st.slider("Fine Aggregate Fraction (fine_fraction)", 0.30, 0.50, 0.40, 0.01, help="Manually overrides the IS 10262 calculation for aggregate proportions.")
    calib_max_flyash_frac = st.slider("Max Fly Ash fraction", 0.0, 0.5, 0.30, 0.05, help="Maximum Fly Ash replacement percentage to test.")
    calib_max_ggbs_frac = st.slider("Max GGBS fraction", 0.0, 0.5, 0.50, 0.05, help="Maximum GGBS replacement percentage to test.")
    calib_scm_step = st.slider("SCM fraction step (scm_step)", 0.05, 0.25, 0.10, 0.05, help="Step size for testing different SCM replacement percentages.")


# Load datasets
_, emissions_df, costs_df = load_data(None, emissions_file, cost_file)


# --- Main Execution Block ---

# Initialize session state for clarification workflow
if 'clarification_needed' not in st.session_state:
    st.session_state.clarification_needed = False
if 'run_generation' not in st.session_state:
    st.session_state.run_generation = False
if 'final_inputs' not in st.session_state:
    st.session_state.final_inputs = {}

# Map internal keys to UI widgets for the clarification form
CLARIFICATION_WIDGETS = {
    "grade": lambda v: st.selectbox("Concrete Grade", list(GRADE_STRENGTH.keys()), index=list(GRADE_STRENGTH.keys()).index(v) if v in GRADE_STRENGTH else 4),
    "exposure": lambda v: st.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()), index=list(EXPOSURE_WB_LIMITS.keys()).index(v) if v in EXPOSURE_WB_LIMITS else 2),
    "target_slump": lambda v: st.slider("Target Slump (mm)", 25, 180, v if isinstance(v, int) else 100, 5),
    "cement_choice": lambda v: st.selectbox("Cement Type", ["OPC 33", "OPC 43", "OPC 53", "PPC"], index=["OPC 33", "OPC 43", "OPC 53", "PPC"].index(v) if v in ["OPC 33", "OPC 43", "OPC 53", "PPC"] else 2),
    "nom_max": lambda v: st.selectbox("Nominal Max. Aggregate Size (mm)", [10, 12.5, 20, 40], index=[10, 12.5, 20, 40].index(v) if v in [10, 12.5, 20, 40] else 2),
}

# The button press is the main trigger to start or reset the process
if run_button:
    # Reset state flags on a new run
    st.session_state.run_generation = False
    st.session_state.clarification_needed = False
    
    # Get initial inputs from sidebar (if manual) or defaults
    inputs = { "grade": grade, "exposure": exposure, "cement_choice": cement_choice, "nom_max": nom_max, "agg_shape": agg_shape, "target_slump": target_slump, "use_sp": use_sp, "optimize_cost": optimize_cost, "qc_level": qc_level, "fine_zone": fine_zone }
    
    # If the user entered text (and not in manual mode), parse it and check for missing info
    if user_text.strip() and not manual_mode:
        with st.spinner("🤖 Parsing your request..."):
            inputs, msgs, _ = apply_parser(user_text, inputs)
        
        if msgs:
            st.info(" ".join(msgs), icon="💡")
            
        # Check for the required fields for mix design
        required_fields = ["grade", "exposure", "target_slump", "nom_max", "cement_choice"]
        missing_fields = [f for f in required_fields if inputs.get(f) is None]

        if missing_fields:
            # If fields are missing, trigger the clarification step
            st.session_state.clarification_needed = True
            st.session_state.final_inputs = inputs  # Store partial inputs
            st.session_state.missing_fields = missing_fields
        else:
            # If all fields are present, proceed to generation
            st.session_state.run_generation = True
            st.session_state.final_inputs = inputs
    else:
        # If in manual mode or no text was entered, proceed directly to generation
        st.session_state.run_generation = True
        st.session_state.final_inputs = inputs

# Display the clarification form if triggered
if st.session_state.get('clarification_needed', False):
    st.markdown("---")
    st.warning("Your request is missing some details. Please confirm the following to continue.", icon="🤔")
    with st.form("clarification_form"):
        st.subheader("Please Clarify Your Requirements")
        current_inputs = st.session_state.final_inputs
        missing_fields_list = st.session_state.missing_fields
        
        # Dynamically create widgets only for the missing fields
        num_cols = min(len(missing_fields_list), 3) # Max 3 columns for neatness
        cols = st.columns(num_cols)
        for i, field in enumerate(missing_fields_list):
            with cols[i % num_cols]:
                widget_func = CLARIFICATION_WIDGETS[field]
                current_value = current_inputs.get(field)
                new_value = widget_func(current_value)
                current_inputs[field] = new_value

        submitted = st.form_submit_button("✅ Confirm & Continue", use_container_width=True, type="primary")

        if submitted:
            # When form is submitted, update state and rerun to start generation
            st.session_state.final_inputs = current_inputs
            st.session_state.clarification_needed = False
            st.session_state.run_generation = True
            st.rerun()

# Run the main generation and display logic if the flag is set
if st.session_state.get('run_generation', False):
    st.markdown("---")
    try:
        inputs = st.session_state.final_inputs
        
        # Validate grade against exposure
        min_grade_req = EXPOSURE_MIN_GRADE[inputs["exposure"]]
        grade_order = list(GRADE_STRENGTH.keys())
        if grade_order.index(inputs["grade"]) < grade_order.index(min_grade_req):
            st.warning(f"For **{inputs['exposure']}** exposure, IS 456 recommends a minimum grade of **{min_grade_req}**. The grade has been automatically updated.", icon="⚠️")
            inputs["grade"] = min_grade_req

        # Prepare calibration overrides if enabled
        calibration_kwargs = {}
        if enable_calibration_overrides:
            calibration_kwargs = {
                "wb_min": calib_wb_min,
                "wb_steps": calib_wb_steps,
                "max_flyash_frac": calib_max_flyash_frac,
                "max_ggbs_frac": calib_max_ggbs_frac,
                "scm_step": calib_scm_step,
                "fine_fraction_override": calib_fine_fraction
            }
            st.info("Developer calibration overrides are enabled.", icon="🛠️")

        # Generate mixes
        with st.spinner("⚙️ Running IS-code calculations and optimizing for sustainability..."):
            fck, S = GRADE_STRENGTH[inputs["grade"]], QC_STDDEV[inputs.get("qc_level", "Good")]
            fck_target = fck + 1.65 * S
            opt_df, opt_meta, trace = generate_mix(
                inputs["grade"], inputs["exposure"], inputs["nom_max"], 
                inputs["target_slump"], inputs["agg_shape"], inputs["fine_zone"], 
                emissions_df, costs_df, inputs["cement_choice"], 
                use_sp=inputs["use_sp"], optimize_cost=inputs["optimize_cost"],
                **calibration_kwargs
            )
            base_df, base_meta = generate_baseline(inputs["grade"], inputs["exposure"], inputs["nom_max"], inputs["target_slump"], inputs["agg_shape"], inputs["fine_zone"], emissions_df, costs_df, inputs["cement_choice"], use_sp=inputs["use_sp"])

        if opt_df is None or base_df is None:
            st.error("Could not find a feasible mix design with the given constraints. Try adjusting the parameters, such as a higher grade or less restrictive exposure condition.", icon="❌")
            st.dataframe(pd.DataFrame(trace))
        else:
            for m in (opt_meta, base_meta):
                m["fck"], m["fck_target"], m["stddev_S"] = fck, round(fck_target, 1), S
            st.success(f"Successfully generated mix designs for **{inputs['grade']}** concrete in **{inputs['exposure']}** conditions.", icon="✅")

            # --- Results Display ---
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 **Overview**", "🌱 **Optimized Mix**", "🏗️ **Baseline Mix**", "📋 **QA/QC & Gradation**", "📥 **Downloads & Reports**", "🔬 **Lab Calibration**"])

            # -- Overview Tab --
            with tab1:
                co2_opt, cost_opt = opt_meta["co2_total"], opt_meta["cost_total"]
                co2_base, cost_base = base_meta["co2_total"], base_meta["cost_total"]
                reduction = (co2_base - co2_opt) / co2_base * 100 if co2_base > 0 else 0.0
                cost_savings = cost_base - cost_opt
                
                st.subheader("Performance At a Glance")
                c1, c2, c3 = st.columns(3)
                c1.metric("🌱 CO₂ Reduction", f"{reduction:.1f}%", f"{co2_base - co2_opt:.1f} kg/m³ saved")
                c2.metric("💰 Cost Savings", f"₹{cost_savings:,.0f} / m³", f"{cost_savings/cost_base*100 if cost_base>0 else 0:.1f}% cheaper")
                c3.metric("♻️ SCM Content", f"{opt_meta['scm_total_frac']*100:.0f}%", f"{base_meta['scm_total_frac']*100:.0f}% in baseline", help="Supplementary Cementitious Materials (Fly Ash, GGBS) replace high-carbon cement.")
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Embodied Carbon (CO₂e)")
                    chart_data = pd.DataFrame({'Mix Type': ['Baseline OPC', 'CivilGPT Optimized'], 'CO₂ (kg/m³)': [co2_base, co2_opt]})
                    fig, ax = plt.subplots(figsize=(6, 4))
                    bars = ax.bar(chart_data['Mix Type'], chart_data['CO₂ (kg/m³)'], color=['#D3D3D3', '#4CAF50'])
                    ax.set_ylabel("Embodied Carbon (kg CO₂e / m³)")
                    ax.bar_label(bars, fmt='{:,.1f}')
                    st.pyplot(fig)
                with col2:
                    st.subheader("💵 Material Cost")
                    chart_data_cost = pd.DataFrame({'Mix Type': ['Baseline OPC', 'CivilGPT Optimized'], 'Cost (₹/m³)': [cost_base, cost_opt]})
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    bars2 = ax2.bar(chart_data_cost['Mix Type'], chart_data_cost['Cost (₹/m³)'], color=['#D3D3D3', '#2196F3'])
                    ax2.set_ylabel("Material Cost (₹ / m³)")
                    ax2.bar_label(bars2, fmt='₹{:,.0f}')
                    st.pyplot(fig2)
            
            def display_mix_details(title, df, meta, exposure):
                st.header(title)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("💧 Water/Binder Ratio", f"{meta['w_b']:.3f}")
                c2.metric("📦 Total Binder (kg/m³)", f"{meta['cementitious']:.1f}")
                c3.metric("🎯 Target Strength (MPa)", f"{meta['fck_target']:.1f}")
                c4.metric("⚖️ Unit Weight (kg/m³)", f"{df['Quantity (kg/m3)'].sum():.1f}")
                
                st.subheader("Mix Proportions (per m³)")
                # UI PATCH: Add explanatory note for CO2 factors.
                st.info(
                    "CO₂ factors represent cradle-to-gate emissions: the amount of CO₂ released per kg of material during its manufacture. These values do not reduce the material mass in the mix — they are an environmental footprint, not a physical subtraction.",
                    icon="ℹ️"
                )
                st.dataframe(df.style.format({
                    "Quantity (kg/m3)": "{:.2f}",
                    "CO2_Factor(kg_CO2_per_kg)": "{:.3f}",
                    "CO2_Emissions (kg/m3)": "{:.2f}",
                    "Cost(₹/kg)": "₹{:.2f}",
                    "Cost (₹/m3)": "₹{:.2f}"
                }), use_container_width=True)

                st.subheader("Compliance & Sanity Checks (IS 10262 & IS 456)")
                is_feasible, fail_reasons, warnings, derived, checks_dict = check_feasibility(df, meta, exposure)
                
                if is_feasible:
                    st.success("✅ This mix design is compliant with IS code requirements.", icon="👍")
                else:
                    st.error(f"❌ This mix fails {len(fail_reasons)} IS code compliance check(s): " + ", ".join(fail_reasons), icon="🚨")
                
                if warnings:
                    for warning in warnings:
                        st.warning(warning, icon="⚠️")
                
                with st.expander("Show detailed calculation parameters"):
                    st.json(derived)

            # -- Optimized & Baseline Mix Tabs --
            with tab2:
                display_mix_details("🌱 Optimized Low-Carbon Mix Design", opt_df, opt_meta, inputs['exposure'])
            with tab3:
                display_mix_details("🏗️ Standard OPC Baseline Mix Design", base_df, base_meta, inputs['exposure'])
            
            # -- QA/QC & Gradation Tab --
            with tab4:
                st.header("Quality Assurance & Sieve Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Fine Aggregate Gradation")
                    if fine_csv is not None:
                        df_fine = pd.read_csv(fine_csv)
                        ok_fa, msgs_fa = sieve_check_fa(df_fine, inputs.get("fine_zone", "Zone II"))
                        if ok_fa: st.success(msgs_fa[0], icon="✅")
                        else: 
                            for m in msgs_fa: st.error(m, icon="❌")
                        st.dataframe(df_fine, use_container_width=True)
                    else:
                        st.info("Upload a Fine Aggregate CSV in the sidebar to perform a gradation check against IS 383.", icon="ℹ️")
                with col2:
                    st.subheader("Coarse Aggregate Gradation")
                    if coarse_csv is not None:
                        df_coarse = pd.read_csv(coarse_csv)
                        ok_ca, msgs_ca = sieve_check_ca(df_coarse, inputs["nom_max"])
                        if ok_ca: st.success(msgs_ca[0], icon="✅")
                        else:
                            for m in msgs_ca: st.error(m, icon="❌")
                        st.dataframe(df_coarse, use_container_width=True)
                    else:
                        st.info("Upload a Coarse Aggregate CSV in the sidebar to perform a gradation check against IS 383.", icon="ℹ️")
                
                st.markdown("---")
                with st.expander("🔬 View Optimizer Trace (Advanced)"):
                    if trace:
                        trace_df = pd.DataFrame(trace)
                        st.markdown("The table below shows every mix combination attempted by the optimizer. 'Feasible' mixes met all IS-code checks.")
                        st.dataframe(trace_df.style.apply(lambda s: ['background-color: #e8f5e9' if v else 'background-color: #ffebee' for v in s], subset=['feasible']), use_container_width=True)
                        st.markdown("#### CO₂ vs. Cost of All Candidate Mixes")
                        fig, ax = plt.subplots()
                        scatter_colors = ["#4CAF50" if f else "#F44336" for f in trace_df["feasible"]]
                        ax.scatter(trace_df["cost"], trace_df["co2"], c=scatter_colors, alpha=0.6)
                        ax.set_xlabel("Material Cost (₹/m³)")
                        ax.set_ylabel("Embodied Carbon (kg CO₂e/m³)")
                        ax.grid(True, linestyle='--', alpha=0.6)
                        st.pyplot(fig)
                    else:
                        st.info("Trace not available.")
            
            # -- Downloads Tab --
            with tab5:
                st.header("Download Reports")
                
                # Excel Report
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    opt_df.to_excel(writer, sheet_name="Optimized_Mix", index=False)
                    base_df.to_excel(writer, sheet_name="Baseline_Mix", index=False)
                    pd.DataFrame([opt_meta]).T.to_excel(writer, sheet_name="Optimized_Meta")
                    pd.DataFrame([base_meta]).T.to_excel(writer, sheet_name="Baseline_Meta")
                
                # PDF Report
                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=(8.5*inch, 11*inch))
                styles = getSampleStyleSheet()
                story = [Paragraph("CivilGPT Sustainable Mix Report", styles['h1']), Spacer(1, 0.2*inch)]
                
                # Summary table
                summary_data = [
                    ["Metric", "Optimized Mix", "Baseline Mix"],
                    ["CO₂ (kg/m³)", f"{opt_meta['co2_total']:.1f}", f"{base_meta['co2_total']:.1f}"],
                    ["Cost (₹/m³)", f"₹{opt_meta['cost_total']:,.2f}", f"₹{base_meta['cost_total']:,.2f}"],
                    ["w/b Ratio", f"{opt_meta['w_b']:.3f}", f"{base_meta['w_b']:.3f}"],
                    ["Binder (kg/m³)", f"{opt_meta['cementitious']:.1f}", f"{base_meta['cementitious']:.1f}"],
                ]
                summary_table = Table(summary_data, hAlign='LEFT', colWidths=[2*inch, 1.5*inch, 1.5*inch])
                summary_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)]))
                story.extend([Paragraph(f"Design for <b>{inputs['grade']} / {inputs['exposure']} Exposure</b>", styles['h2']), summary_table, Spacer(1, 0.2*inch)])
                
                # Optimized Mix Table
                opt_data_pdf = [opt_df.columns.values.tolist()] + opt_df.values.tolist()
                opt_table = Table(opt_data_pdf, hAlign='LEFT')
                opt_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.palegreen)]))
                story.extend([Paragraph("Optimized Mix Proportions (kg/m³)", styles['h2']), opt_table])
                doc.build(story)

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button("📄 Download PDF Report", data=pdf_buffer.getvalue(), file_name="CivilGPT_Report.pdf", mime="application/pdf", use_container_width=True)
                    st.download_button("📈 Download Excel Report", data=excel_buffer.getvalue(), file_name="CivilGPT_Mix_Designs.xlsx", mime="application/vnd.ms-excel", use_container_width=True)
                with d2:
                    st.download_button("✔️ Optimized Mix (CSV)", data=opt_df.to_csv(index=False).encode("utf-8"), file_name="optimized_mix.csv", mime="text/csv", use_container_width=True)
                    st.download_button("✖️ Baseline Mix (CSV)", data=base_df.to_csv(index=False).encode("utf-8"), file_name="baseline_mix.csv", mime="text/csv", use_container_width=True)
            
            # -- NEW: Lab Calibration Tab --
            with tab6:
                st.header("🔬 Lab Calibration Analysis")
                if lab_csv is not None:
                    try:
                        lab_results_df = pd.read_csv(lab_csv)
                        # Run the calibration analysis
                        comparison_df, error_metrics = run_lab_calibration(lab_results_df)

                        if comparison_df is not None and not comparison_df.empty:
                            st.subheader("Error Metrics")
                            st.markdown("Comparing lab-tested 28-day strength against the IS code's required target strength (`f_target = fck + 1.65 * S`).")
                            m1, m2, m3 = st.columns(3)
                            m1.metric(label="Mean Absolute Error (MAE)", value=f"{error_metrics['Mean Absolute Error (MPa)']:.2f} MPa")
                            m2.metric(label="Root Mean Squared Error (RMSE)", value=f"{error_metrics['Root Mean Squared Error (MPa)']:.2f} MPa")
                            m3.metric(label="Mean Bias (Over/Under-prediction)", value=f"{error_metrics['Mean Bias (MPa)']:.2f} MPa")
                            st.markdown("---")

                            st.subheader("Comparison: Lab vs. Predicted Target Strength")
                            st.dataframe(comparison_df.style.format({
                                "Lab Strength (MPa)": "{:.2f}",
                                "Predicted Target Strength (MPa)": "{:.2f}",
                                "Error (MPa)": "{:+.2f}"
                            }), use_container_width=True)

                            st.subheader("Prediction Accuracy Scatter Plot")
                            fig, ax = plt.subplots()
                            ax.scatter(comparison_df["Lab Strength (MPa)"], comparison_df["Predicted Target Strength (MPa)"], alpha=0.7, label="Data Points")
                            # Add y=x line
                            lims = [
                                np.min([ax.get_xlim(), ax.get_ylim()]),
                                np.max([ax.get_xlim(), ax.get_ylim()]),
                            ]
                            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Perfect Prediction (y=x)")
                            ax.set_xlabel("Actual Lab Strength (MPa)")
                            ax.set_ylabel("Predicted Target Strength (MPa)")
                            ax.set_title("Lab Strength vs. Predicted Target Strength")
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)
                        else:
                            st.warning("Could not process the uploaded lab data CSV. Please check the file format, column names, and ensure it contains valid data.", icon="⚠️")
                    except Exception as e:
                        st.error(f"Failed to read or process the lab data CSV file: {e}", icon="💥")
                else:
                    st.info(
                        "Upload a lab data CSV in the sidebar to automatically compare CivilGPT's "
                        "target strength calculations against your real-world results.",
                        icon="ℹ️"
                    )

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}", icon="💥")
        st.code(traceback.format_exc())
    finally:
        # VERY IMPORTANT: Reset the generation flag after the run is complete
        st.session_state.run_generation = False

# This block runs only if no action (button press, form submission) has been initiated
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
