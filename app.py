# app.py — CivilGPT v1.9 (Clarification flow + unified feasibility + optional trace)
# - Based on v1.8 (all features preserved)
# - New in v1.9: clarification flow when LLM parser misses required fields,
#   unified check_feasibility() function used by optimizer to skip invalid candidates,
#   optional optimizer trace returned to UI (hidden by default),
#   positive message when all checks pass.

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

# Groq client
try:
    from groq import Groq
    client = Groq(api_key=st.secrets.get("GROQ_API_KEY", None))
except Exception:
    client = None

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="CivilGPT - Sustainable Concrete Mix Designer",
    page_icon="🧱",
    layout="wide"
)

# =========================
# Dataset Path Handling
# =========================
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

# =========================
# IS-style Rules & Tables
# =========================
EXPOSURE_WB_LIMITS = {
    "Mild": 0.60,
    "Moderate": 0.55,
    "Severe": 0.50,
    "Very Severe": 0.45,
    "Marine": 0.40,
}

EXPOSURE_MIN_CEMENT = {
    "Mild": 300, "Moderate": 300, "Severe": 320,
    "Very Severe": 340, "Marine": 360,
}

EXPOSURE_MIN_GRADE = {
    "Mild": "M20", "Moderate": "M25", "Severe": "M30",
    "Very Severe": "M35", "Marine": "M40",
}

GRADE_STRENGTH = {
    "M10": 10, "M15": 15, "M20": 20, "M25": 25,
    "M30": 30, "M35": 35, "M40": 40, "M45": 45, "M50": 50
}

WATER_BASELINE = {10: 208, 12.5: 202, 20: 186, 40: 165}

AGG_SHAPE_WATER_ADJ = {
    "Angular (baseline)": 0.00, "Sub-angular": -0.03,
    "Sub-rounded": -0.05, "Rounded": -0.07,
    "Flaky/Elongated": +0.03,
}

QC_STDDEV = {"Good": 5.0, "Fair": 7.5, "Poor": 10.0}

FINE_AGG_ZONE_LIMITS = {
    "Zone I":  {"10.0": (100,100),"4.75": (90,100),"2.36": (60,95),"1.18": (30,70),"0.600": (15,34),"0.300": (5,20),"0.150": (0,10)},
    "Zone II": {"10.0": (100,100),"4.75": (90,100),"2.36": (75,100),"1.18": (55,90),"0.600": (35,59),"0.300": (8,30),"0.150": (0,10)},
    "Zone III":{"10.0": (100,100),"4.75": (90,100),"2.36": (85,100),"1.18": (75,90),"0.600": (60,79),"0.300": (12,40),"0.150": (0,10)},
    "Zone IV": {"10.0": (95,100),"4.75": (95,100),"2.36": (95,100),"1.18": (90,100),"0.600": (80,100),"0.300": (15,50),"0.150": (0,15)},
}

COARSE_LIMITS = {
    10: {"20.0": (100,100), "10.0": (85,100),  "4.75": (0,20)},
    20: {"40.0": (95,100),  "20.0": (95,100),  "10.0": (25,55), "4.75": (0,10)},
    40: {"80.0": (95,100),  "40.0": (95,100),  "20.0": (30,70), "10.0": (0,15)}
}

# =========================
# Parsers
# =========================
def simple_parse(text: str) -> dict:
    result = {}
    grade_match = re.search(r"\bM(10|15|20|25|30|35|40|45|50)\b", text, re.IGNORECASE)
    if grade_match: result["grade"] = "M" + grade_match.group(1)

    for exp in EXPOSURE_WB_LIMITS.keys():
        if re.search(exp, text, re.IGNORECASE): result["exposure"] = exp; break

    slump_match = re.search(r"slump\s*(\d+)", text, re.IGNORECASE)
    if slump_match: result["slump"] = int(slump_match.group(1))

    cement_types = ["OPC 33", "OPC 43", "OPC 53", "PPC"]
    for ctype in cement_types:
        if re.search(ctype.replace(" ", r"\s*"), text, re.IGNORECASE):
            result["cement"] = ctype; break

    # try capture nominal max aggregate like "20mm" or "20 mm"
    nom_match = re.search(r"(10|12\.5|20|40)\s*-?\s*mm", text, re.IGNORECASE)
    if nom_match:
        try:
            val = float(nom_match.group(1))
            result["nom_max"] = val
        except:
            pass

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

# =========================
# Helpers
# =========================
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
    if slump_mm <= 50: water = base
    else: water = base * (1 + 0.03 * ((slump_mm - 50) / 25.0))
    water *= (1.0 + AGG_SHAPE_WATER_ADJ.get(agg_shape, 0.0))
    if uses_sp and sp_reduction_frac > 0: water *= (1 - sp_reduction_frac)
    return float(water)

# =========================
# Mix Evaluation
# =========================
def evaluate_mix(components_dict, emissions_df, costs_df=None):
    comp_df = pd.DataFrame(list(components_dict.items()), columns=["Material", "Quantity (kg/m3)"])
    df = comp_df.merge(emissions_df, on="Material", how="left")
    if "CO2_Factor(kg_CO2_per_kg)" not in df.columns:
        df["CO2_Factor(kg_CO2_per_kg)"] = 0.0
    df["CO2_Factor(kg_CO2_per_kg)"] = df["CO2_Factor(kg_CO2_per_kg)"].fillna(0.0)
    df["CO2_Emissions (kg/m3)"] = df["Quantity (kg/m3)"] * df["CO2_Factor(kg_CO2_per_kg)"]

    if costs_df is not None and "Cost(₹/kg)" in costs_df.columns:
        df = df.merge(costs_df, on="Material", how="left")
        df["Cost(₹/kg)"] = df["Cost(₹/kg)"].fillna(0.0)
        df["Cost (₹/m3)"] = df["Quantity (kg/m3)"] * df["Cost(₹/kg)"]
    else:
        df["Cost (₹/m3)"] = 0.0
    return df

# =========================
# Moisture Correction
# =========================
def aggregate_correction(delta_moisture_pct: float, agg_mass_ssd: float):
    water_delta = (delta_moisture_pct / 100.0) * agg_mass_ssd
    corrected_mass = agg_mass_ssd * (1 + delta_moisture_pct / 100.0)
    return float(water_delta), float(corrected_mass)

# =========================
# Aggregate Volume Balance
# =========================
def compute_aggregates(cementitious, water, sp, fine_fraction=0.40,
                       density_fa=2650.0, density_ca=2700.0):
    """
    Compute aggregate masses (kg/m³) by volume method (per 1 m³ concrete).
    - Inputs: cementitious, water, sp are in kg/m³.
    - Densities are in kg/m³ (typical: fa ~2650, ca ~2700).
    Returns: (mass_fine_kg_per_m3, mass_coarse_kg_per_m3)
    """
    # volumes (m³) of binder components (per m³ concrete)
    vol_cem = cementitious / 3150.0     # cement density ~3150 kg/m³
    vol_wat = water / 1000.0            # water: 1000 kg/m³
    vol_sp  = sp / 1200.0               # SP density ~1200 kg/m³ (approx)

    vol_binder = vol_cem + vol_wat + vol_sp

    # available aggregate volume (m³) in 1 m³ concrete
    total_vol = 1.0
    vol_agg = total_vol - vol_binder

    # fallback if binder volume exceeds 1 m³ (shouldn't happen normally)
    if vol_agg <= 0:
        vol_agg = 0.60  # reasonable fallback

    vol_fine = vol_agg * fine_fraction
    vol_coarse = vol_agg * (1.0 - fine_fraction)

    # mass = volume (m³) * density (kg/m³) -> kg
    mass_fine = vol_fine * density_fa
    mass_coarse = vol_coarse * density_ca

    return float(mass_fine), float(mass_coarse)

# =========================
# Compliance Checks
# =========================
def compliance_checks(mix_df, meta, exposure):
    checks = {}
    try:
        checks["W/B ≤ exposure limit"] = float(meta["w_b"]) <= EXPOSURE_WB_LIMITS[exposure]
    except: checks["W/B ≤ exposure limit"] = False

    try:
        checks["Min cementitious met"] = float(meta["cementitious"]) >= float(EXPOSURE_MIN_CEMENT[exposure])
    except: checks["Min cementitious met"] = False

    try:
        checks["SCM ≤ 50%"] = float(meta.get("scm_total_frac", 0.0)) <= 0.50
    except:
        checks["SCM ≤ 50%"] = False

    try:
        total_mass = float(mix_df["Quantity (kg/m3)"].sum())
        checks["Unit weight 2200–2600 kg/m³"] = 2200.0 <= total_mass <= 2600.0
    except:
        checks["Unit weight 2200–2600 kg/m³"] = False

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
        "fck (MPa)": meta.get("fck"),
        "fck,target (MPa)": meta.get("fck_target"),
        "QC (S, MPa)": meta.get("stddev_S"),
    }
    return checks, derived

# =========================
# Sanity Check Helper
# =========================
def sanity_check_mix(meta, df):
    warnings = []
    try:
        cement = float(meta.get("cement", 0))
        water = float(meta.get("water_target", 0))
        fine = float(meta.get("fine", 0))
        coarse = float(meta.get("coarse", 0))
        sp = float(meta.get("sp", 0))
        unit_wt = float(df["Quantity (kg/m3)"].sum())
    except Exception:
        # if something missing, return a warning
        return ["Insufficient data to run sanity checks."]

    if cement < 250:
        warnings.append(f"Cement too low ({cement:.1f} kg/m³) — IS 456 min is 300–360 depending on exposure.")
    if cement > 500:
        warnings.append(f"Cement unusually high ({cement:.1f} kg/m³).")
    if water < 140 or water > 220:
        warnings.append(f"Water out of typical range ({water:.1f} kg/m³).")
    if fine < 500 or fine > 900:
        warnings.append(f"Fine aggregate unusual ({fine:.1f} kg/m³).")
    if coarse < 1000 or coarse > 1300:
        warnings.append(f"Coarse aggregate unusual ({coarse:.1f} kg/m³).")
    if sp > 20:
        warnings.append(f"SP dosage unusually high ({sp:.1f} kg/m³).")
    if unit_wt < 2200 or unit_wt > 2600:
        warnings.append(f"Unit weight {unit_wt:.1f} kg/m³ outside IS 10262 range (2200–2600).")

    return warnings

# =========================
# Unified Feasibility Function
# =========================
def check_feasibility(mix_df, meta, exposure):
    """
    Returns: (feasible: bool, reasons: list[str], derived: dict, checks: dict)
    Combines compliance checks and sanity checks to provide a consolidated decision.
    """
    checks, derived = compliance_checks(mix_df, meta, exposure)
    warnings = sanity_check_mix(meta, mix_df)

    reasons = []
    for k, v in checks.items():
        if not v:
            reasons.append(f"Failed check: {k}")
    reasons.extend(warnings)

    feasible = len(reasons) == 0
    return feasible, reasons, derived, checks

# =========================
# Sieve Checks
# =========================
def sieve_check_fa(df: pd.DataFrame, zone: str):
    try:
        limits = FINE_AGG_ZONE_LIMITS[zone]
        ok, msgs = True, []
        for sieve, (lo, hi) in limits.items():
            row = df.loc[df["Sieve_mm"].astype(str) == sieve]
            if row.empty:
                ok = False; msgs.append(f"Missing sieve {sieve} mm."); continue
            p = float(row["PercentPassing"].iloc[0])
            if not (lo <= p <= hi):
                ok = False; msgs.append(f"{sieve} mm → {p:.1f}% (req {lo}-{hi}%)")
        if ok and not msgs: msgs = [f"Fine aggregate meets IS 383 {zone} limits."]
        return ok, msgs
    except: return False, ["Invalid fine aggregate CSV format. Expected columns: Sieve_mm, PercentPassing"]

def sieve_check_ca(df: pd.DataFrame, nominal_mm: int):
    try:
        limits = COARSE_LIMITS[int(nominal_mm)]
        ok, msgs = True, []
        for sieve, (lo, hi) in limits.items():
            row = df.loc[df["Sieve_mm"].astype(str) == sieve]
            if row.empty:
                ok = False; msgs.append(f"Missing sieve {sieve} mm."); continue
            p = float(row["PercentPassing"].iloc[0])
            if not (lo <= p <= hi):
                ok = False; msgs.append(f"{sieve} mm → {p:.1f}% (req {lo}-{hi}%)")
        if ok and not msgs: msgs = [f"Coarse aggregate meets IS 383 ({nominal_mm} mm graded)."]
        return ok, msgs
    except: return False, ["Invalid coarse aggregate CSV format. Expected columns: Sieve_mm, PercentPassing"]

# =========================
# Mix Generators (IS-code compliant + trace)
# =========================
def generate_mix(grade, exposure, nom_max, target_slump, agg_shape,
                 emissions, costs, cement_choice, use_sp=True, sp_reduction=0.18,
                 optimize_cost=False, fine_fraction=0.40):

    w_b_limit = float(EXPOSURE_WB_LIMITS[exposure])
    min_cem = float(EXPOSURE_MIN_CEMENT[exposure])

    # IS 10262 compliant water calculation
    target_water = water_for_slump_and_shape(
        nom_max_mm=nom_max, slump_mm=int(target_slump),
        agg_shape=agg_shape, uses_sp=use_sp, sp_reduction_frac=sp_reduction
    )

    best_df, best_meta, best_score = None, None, float("inf")
    trace = []

    wb_values = np.linspace(0.35, w_b_limit, 6)
    flyash_options = [0.0, 0.2, 0.3]
    ggbs_options = [0.0, 0.3, 0.5]

    for wb in wb_values:
        for flyash_frac in flyash_options:
            for ggbs_frac in ggbs_options:
                if flyash_frac + ggbs_frac > 0.50:
                    continue
                binder = max(target_water / wb, min_cem)
                cement = binder * (1 - flyash_frac - ggbs_frac)
                flyash = binder * flyash_frac
                ggbs = binder * ggbs_frac
                sp = 2.5 if use_sp else 0.0

                fine, coarse = compute_aggregates(
                    cementitious=binder, water=target_water, sp=sp,
                    fine_fraction=fine_fraction
                )

                mix = {
                    cement_choice: cement,
                    "Fly Ash": flyash,
                    "GGBS": ggbs,
                    "Water": target_water,
                    "PCE Superplasticizer": sp,
                    "M-Sand": fine,
                    "20mm Coarse Aggregate": coarse,
                }

                df = evaluate_mix(mix, emissions, costs)
                co2_total = float(df["CO2_Emissions (kg/m3)"].sum())
                cost_total = float(df["Cost (₹/m3)"].sum())

                # candidate meta for feasibility checking
                candidate_meta = {
                    "w_b": wb, "cementitious": binder, "cement": cement,
                    "flyash": flyash, "ggbs": ggbs, "water_target": target_water,
                    "sp": sp, "fine": fine, "coarse": coarse,
                    "scm_total_frac": flyash_frac + ggbs_frac,
                    "grade": grade, "exposure": exposure,
                    "nom_max": nom_max, "slump": target_slump,
                    "co2_total": co2_total, "cost_total": cost_total
                }

                # feasibility check
                feasible, reasons, derived, checks = check_feasibility(df, candidate_meta, exposure)

                score = co2_total if not optimize_cost else co2_total + 0.001 * cost_total

                trace.append({
                    "wb": float(wb), "flyash_frac": float(flyash_frac), "ggbs_frac": float(ggbs_frac),
                    "co2": float(co2_total), "cost": float(cost_total), "score": float(score),
                    "feasible": bool(feasible), "reasons": reasons
                })

                if feasible and score < best_score:
                    best_df = df.copy()
                    best_score = score
                    best_meta = candidate_meta.copy()

    return best_df, best_meta, trace


def generate_baseline(grade, exposure, nom_max, target_slump, agg_shape,
                      emissions, costs, cement_choice, use_sp=True, sp_reduction=0.18,
                      fine_fraction=0.40):

    w_b_limit = float(EXPOSURE_WB_LIMITS[exposure])
    min_cem = float(EXPOSURE_MIN_CEMENT[exposure])

    water_target = water_for_slump_and_shape(
        nom_max_mm=nom_max, slump_mm=int(target_slump),
        agg_shape=agg_shape, uses_sp=use_sp, sp_reduction_frac=sp_reduction
    )

    cementitious = max(water_target / w_b_limit, min_cem)
    sp = 2.5 if use_sp else 0.0

    fine, coarse = compute_aggregates(
        cementitious=cementitious, water=water_target, sp=sp,
        fine_fraction=fine_fraction
    )

    mix = {
        cement_choice: cementitious,
        "Fly Ash": 0.0,
        "GGBS": 0.0,
        "Water": water_target,
        "PCE Superplasticizer": sp,
        "M-Sand": fine,
        "20mm Coarse Aggregate": coarse,
    }

    df = evaluate_mix(mix, emissions, costs)
    meta = {
        "w_b": w_b_limit, "cementitious": cementitious, "cement": cementitious,
        "flyash": 0.0, "ggbs": 0.0, "water_target": water_target,
        "sp": sp, "fine": fine, "coarse": coarse,
        "scm_total_frac": 0.0, "grade": grade, "exposure": exposure,
        "nom_max": nom_max, "slump": target_slump,
        "co2_total": float(df["CO2_Emissions (kg/m3)"].sum()),
        "cost_total": float(df["Cost (₹/m3)"].sum())
    }
    return df, meta

# =========================
# Sidebar UI
# =========================
st.sidebar.header("📝 Mix Inputs")

# Parser input
user_text = st.sidebar.text_area("Describe your mix in English (optional)", height=100)
use_llm_parser = st.sidebar.checkbox("Use Groq LLM Parser", value=False)

# Trace toggle (optional, hidden by default)
show_trace = st.sidebar.checkbox("Show optimizer trace (advanced)", value=False)

# Grade + exposure
grade = st.sidebar.selectbox("Concrete Grade", list(GRADE_STRENGTH.keys()), index=2)
exposure = st.sidebar.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()), index=2)
cement_choice = st.sidebar.selectbox("Cement Type", ["OPC 33", "OPC 43", "OPC 53", "PPC"], index=2)

# Aggregates
nom_max = st.sidebar.selectbox("Nominal max aggregate (mm)", [10, 12.5, 20, 40], index=2)
agg_shape = st.sidebar.selectbox("Aggregate shape", list(AGG_SHAPE_WATER_ADJ.keys()), index=0)
target_slump = st.sidebar.slider("Target slump (mm)", 25, 180, 100, step=5)

# SP + cost
use_sp = st.sidebar.checkbox("Use Superplasticizer (PCE)", True)
optimize_cost = st.sidebar.checkbox("Optimize for CO₂ + Cost", False)

# Fine fraction
fine_fraction = st.sidebar.slider("Fine aggregate fraction", 0.3, 0.5, 0.40, step=0.01)

# QC + moisture
qc_level = st.sidebar.selectbox("Quality control level", list(QC_STDDEV.keys()), index=0)
air_pct = st.sidebar.number_input("Entrapped air (%)", 1.0, 3.0, 2.0, step=0.5)
fa_moist = st.sidebar.number_input("Fine agg moisture (%)", 0.0, 10.0, 0.0, step=0.1)
ca_moist = st.sidebar.number_input("Coarse agg moisture (%)", 0.0, 5.0, 0.0, step=0.1)
fa_abs, ca_abs = 1.0, 0.5

st.sidebar.markdown("---")
st.sidebar.markdown("#### IS 383 Sieve (optional uploads)")
fine_zone = st.sidebar.selectbox("Fine agg zone (IS 383)", ["Zone I","Zone II","Zone III","Zone IV"], index=1)
fine_csv = st.sidebar.file_uploader("Fine sieve CSV (Sieve_mm,PercentPassing)", type=["csv"], key="fine_csv")
coarse_csv = st.sidebar.file_uploader("Coarse sieve CSV (Sieve_mm,PercentPassing)", type=["csv"], key="coarse_csv")

st.sidebar.markdown("---")
st.sidebar.markdown("#### (Optional) Provide CSVs here")
materials_file = st.sidebar.file_uploader("materials_library.csv", type=["csv"], key="materials_csv")
emissions_file = st.sidebar.file_uploader("emission_factors.csv", type=["csv"], key="emissions_csv")
cost_file = st.sidebar.file_uploader("cost_factors.csv", type=["csv"], key="cost_csv")

# Load datasets
materials_df, emissions_df, costs_df = load_data(materials_file, emissions_file, cost_file)

# =========================
# Parser Override (updated to return parsed too)
# =========================
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
        updated["grade"] = parsed["grade"]; messages.append(f"Parser set grade → {parsed['grade']}")
    if "exposure" in parsed and parsed["exposure"] in EXPOSURE_WB_LIMITS:
        updated["exposure"] = parsed["exposure"]; messages.append(f"Parser set exposure → {parsed['exposure']}")
    if "slump" in parsed:
        s = max(25, min(180, int(parsed["slump"])))
        updated["target_slump"] = s; messages.append(f"Parser set slump → {s} mm")
    if "cement" in parsed:
        updated["cement_choice"] = parsed["cement"]; messages.append(f"Parser set cement → {parsed['cement']}")
    if "nom_max" in parsed and parsed["nom_max"] in [10, 12.5, 20, 40]:
        updated["nom_max"] = parsed["nom_max"]; messages.append(f"Parser set agg size → {parsed['nom_max']} mm")

    return updated, messages, parsed

# =========================
# Main Run Button
# =========================
st.header("CivilGPT — Sustainable Concrete Mix Designer (v1.9)")
st.markdown("**v1.9:** Clarification flow · Unified feasibility checks · Optional optimizer trace (advanced)")

if st.button("Generate Sustainable Mix (v1.9)"):
    try:
        inputs = {
            "grade": grade, "exposure": exposure, "cement_choice": cement_choice,
            "nom_max": nom_max, "agg_shape": agg_shape, "target_slump": target_slump,
            "use_sp": use_sp, "optimize_cost": optimize_cost, "fine_fraction": fine_fraction
        }
        inputs, msgs, parsed = apply_parser(user_text, inputs)
        for m in msgs: st.info(m)

        # Clarification flow (only when user explicitly chose LLM parser and supplied text)
        if use_llm_parser and user_text.strip():
            # required parsed keys
            required = {"grade":"grade", "exposure":"exposure", "slump":"slump", "nom_max":"nom_max"}
            missing = []
            for k in ["grade","exposure","slump","nom_max"]:
                if k not in parsed:
                    missing.append(k)

            if missing:
                st.warning(f"Parser couldn't extract: {', '.join(missing)}. Please provide the missing fields below (one-shot).")
                with st.form("clarify_form"):
                    # show only missing widgets, prefilled from inputs
                    if "grade" in missing:
                        c_grade = st.selectbox('Concrete Grade', list(GRADE_STRENGTH.keys()), index=list(GRADE_STRENGTH.keys()).index(inputs['grade']))
                    else:
                        c_grade = inputs['grade']
                    if "exposure" in missing:
                        c_exposure = st.selectbox('Exposure Condition', list(EXPOSURE_WB_LIMITS.keys()), index=list(EXPOSURE_WB_LIMITS.keys()).index(inputs['exposure']))
                    else:
                        c_exposure = inputs['exposure']
                    if "slump" in missing:
                        c_slump = st.slider('Target slump (mm)', 25, 180, inputs['target_slump'], step=5)
                    else:
                        c_slump = inputs['target_slump']
                    if "nom_max" in missing:
                        c_nom_max = st.selectbox('Nominal max aggregate (mm)', [10, 12.5, 20, 40], index=[10,12.5,20,40].index(inputs['nom_max']))
                    else:
                        c_nom_max = inputs['nom_max']

                    submit = st.form_submit_button("Submit clarification and regenerate")

                if not submit:
                    # stop execution until user provides clarification
                    st.stop()

                # update inputs with clarified values
                inputs['grade'] = c_grade
                inputs['exposure'] = c_exposure
                inputs['target_slump'] = c_slump
                inputs['nom_max'] = c_nom_max

        # proceed with inputs
        grade, exposure, cement_choice = inputs["grade"], inputs["exposure"], inputs["cement_choice"]
        nom_max, agg_shape, target_slump = inputs["nom_max"], inputs["agg_shape"], inputs["target_slump"]
        use_sp, optimize_cost, fine_fraction = inputs["use_sp"], inputs["optimize_cost"], inputs["fine_fraction"]

        # Exposure–minimum grade enforcement
        min_grade_required = EXPOSURE_MIN_GRADE[exposure]
        grade_order = list(GRADE_STRENGTH.keys())
        if grade_order.index(grade) < grade_order.index(min_grade_required):
            st.warning(f"Exposure {exposure} requires ≥ {min_grade_required}. Adjusted automatically.")
            grade = min_grade_required

        fck, S = GRADE_STRENGTH[grade], QC_STDDEV[qc_level]
        fck_target = fck + 1.65 * S

        # Generate mixes
        opt_df, opt_meta, trace = generate_mix(
            grade, exposure, nom_max, target_slump, agg_shape,
            emissions_df, costs_df, cement_choice,
            use_sp=use_sp, optimize_cost=optimize_cost, fine_fraction=fine_fraction
        )
        base_df, base_meta = generate_baseline(
            grade, exposure, nom_max, target_slump, agg_shape,
            emissions_df, costs_df, cement_choice,
            use_sp=use_sp, fine_fraction=fine_fraction
        )

        if opt_df is None or base_df is None:
            st.error("No feasible mix found.")
        else:
            for m in (opt_meta, base_meta):
                m["fck"], m["fck_target"], m["stddev_S"] = fck, round(fck_target, 1), S

            st.success(f"Mixes generated for **{grade}** under **{exposure}** exposure using {cement_choice}.")

            # Display mixes
            st.subheader("Optimized Sustainable Mix")
            st.dataframe(opt_df, use_container_width=True)
            st.subheader("Baseline Mix")
            st.dataframe(base_df, use_container_width=True)

            co2_opt, cost_opt = opt_meta["co2_total"], opt_meta["cost_total"]
            co2_base, cost_base = base_meta["co2_total"], base_meta["cost_total"]
            reduction = (co2_base - co2_opt) / co2_base * 100 if co2_base > 0 else 0.0
            cost_diff = cost_opt - cost_base

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🌱 Optimized CO₂", f"{co2_opt:.1f} kg/m³")
            k2.metric("🏗️ Baseline CO₂", f"{co2_base:.1f} kg/m³")
            k3.metric("📉 CO₂ Reduction", f"{reduction:.1f}%")
            k4.metric("💰 Cost Δ", f"{cost_diff:+.2f} ₹/m³")

            # =========================
            # Detailed Compliance Displays (restored)
            # =========================
            st.markdown("### ✅ Assumptions, Strength & Compliance")

            # Optimized mix details
            opt_checks, opt_derived = compliance_checks(opt_df, opt_meta, exposure)
            # use unified feasibility for positive/negative message
            opt_feasible, opt_reasons, opt_derived_full, opt_checks_full = check_feasibility(opt_df, opt_meta, exposure)

            with st.expander("Optimized Mix — Details", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.json(opt_derived)
                    st.caption(f"Entrapped air assumed: {air_pct:.1f} %")
                    try:
                        fa_free_w, _ = aggregate_correction(fa_moist - fa_abs, opt_meta["fine"])
                        ca_free_w, _ = aggregate_correction(ca_moist - ca_abs, opt_meta["coarse"])
                        st.write(f"Free water (report): {fa_free_w + ca_free_w:.1f} kg/m³")
                    except: st.write("Free water (report): N/A")
                with c2:
                    st.table(compliance_table(opt_checks))
                    warnings = sanity_check_mix(opt_meta, opt_df)
                    if warnings:
                        st.warning("Sanity Check Warnings:")
                        for w in warnings:
                            st.write("⚠️ " + w)
                    else:
                        st.success("✅ Optimized mix passes all sanity & IS compliance checks.")


            # Baseline mix details
            base_checks, base_derived = compliance_checks(base_df, base_meta, exposure)
            base_feasible, base_reasons, base_derived_full, base_checks_full = check_feasibility(base_df, base_meta, exposure)

            with st.expander("Baseline Mix — Details", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.json(base_derived)
                    st.caption(f"Entrapped air assumed: {air_pct:.1f} %")
                    try:
                        fa_free_w_b, _ = aggregate_correction(fa_moist - fa_abs, base_meta["fine"])
                        ca_free_w_b, _ = aggregate_correction(ca_moist - ca_abs, base_meta["coarse"])
                        st.write(f"Free water (report): {fa_free_w_b + ca_free_w_b:.1f} kg/m³")
                    except: st.write("Free water (report): N/A")
                with c2:
                    st.table(compliance_table(base_checks))
                    warnings = sanity_check_mix(base_meta, base_df)
                    if warnings:
                        st.warning("Sanity Check Warnings:")
                        for w in warnings:
                            st.write("⚠️ " + w)
                    else:
                        st.success("✅ Baseline mix passes all sanity & IS compliance checks.")

            # Moisture corrections
            st.markdown("### 💧 Moisture Corrections")
            try:
                fa_free_w, _ = aggregate_correction(fa_moist - fa_abs, opt_meta["fine"])
                ca_free_w, _ = aggregate_correction(ca_moist - ca_abs, opt_meta["coarse"])
                st.write(f"Free water adj (Optimized): {fa_free_w + ca_free_w:.1f} kg/m³")
            except: st.write("Free water adjustment: N/A")

            # Sieve checks
            st.markdown("### IS 383 Sieve Compliance")
            if fine_csv is not None:
                try:
                    df_fine = pd.read_csv(fine_csv)
                    ok_fa, msgs_fa = sieve_check_fa(df_fine, fine_zone)
                    for m in msgs_fa: st.write(("✅ " if ok_fa else "❌ ") + m)
                except Exception as e: st.warning(f"Fine sieve error: {e}")
            else: st.info("Fine sieve CSV not provided.")

            if coarse_csv is not None:
                try:
                    df_coarse = pd.read_csv(coarse_csv)
                    ok_ca, msgs_ca = sieve_check_ca(df_coarse, nom_max)
                    for m in msgs_ca: st.write(("✅ " if ok_ca else "❌ ") + m)
                except Exception as e: st.warning(f"Coarse sieve error: {e}")
            else: st.info("Coarse sieve CSV not provided.")

            # =========================
            # Optimizer Trace (optional)
            # =========================
            if show_trace and trace:
                st.markdown("### 🔍 Optimizer Trace (all evaluated candidates)")
                try:
                    trace_df = pd.DataFrame(trace)
                    st.dataframe(trace_df, use_container_width=True)
                except Exception:
                    st.write(trace)

            # CO₂ plot
            st.markdown("### 📊 CO₂ Comparison")
            fig, ax = plt.subplots()
            ax.bar(["Optimized", "Baseline"], [co2_opt, co2_base])
            ax.set_ylabel("CO₂ Emissions (kg/m³)")
            st.pyplot(fig)

            # Downloads
            csv_opt = opt_df.to_csv(index=False).encode("utf-8")
            csv_base = base_df.to_csv(index=False).encode("utf-8")

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                opt_df.to_excel(writer, sheet_name="Optimized Mix", index=False)
                base_df.to_excel(writer, sheet_name="Baseline Mix", index=False)
            excel_bytes = buffer.getvalue()

            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph("CivilGPT Sustainable Mix Report", styles["Title"]))
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"Grade: {grade} | Exposure: {exposure} | Cement: {cement_choice}", styles["Normal"]))
            story.append(Paragraph(f"Target mean strength: {round(fck_target,1)} MPa", styles["Normal"]))
            story.append(Spacer(1, 8))
            data_summary = [
                ["Metric", "Optimized", "Baseline"],
                ["CO₂ (kg/m³)", f"{co2_opt:.1f}", f"{co2_base:.1f}"],
                ["Cost (₹/m³)", f"{cost_opt:.2f}", f"{cost_base:.2f}"],
                ["Reduction (%)", f"{reduction:.1f}", "-"]
            ]
            tbl = Table(data_summary, hAlign="LEFT")
            tbl.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.grey)]))
            story.append(tbl)
            doc.build(story)
            pdf_bytes = pdf_buffer.getvalue()

            with st.expander("📥 Downloads", expanded=True):
                st.download_button("Optimized Mix (CSV)", csv_opt, "optimized_mix.csv", "text/csv")
                st.download_button("Baseline Mix (CSV)", csv_base, "baseline_mix.csv", "text/csv")
                st.download_button("Report (Excel)", excel_bytes, "CivilGPT_Report.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                st.download_button("Report (PDF)", pdf_bytes, "CivilGPT_Report.pdf", "application/pdf")

    except Exception as e:
        st.error(f"Error: {e}")
        st.text(traceback.format_exc())
else:
    st.info("Set parameters and click **Generate Sustainable Mix (v1.9)**.")

st.markdown("---")
st.caption("CivilGPT v1.9 | Clarification flow · Feasibility checks · Optional trace | Groq Mixtral")
