# app.py — CivilGPT v2.1 (Part 1/3)
# Backend logic preserved from v2.0
# UI refactored (landing page + toggle sidebar)

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

# Groq client (optional)
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
EXPOSURE_WB_LIMITS = {"Mild": 0.60,"Moderate": 0.55,"Severe": 0.50,"Very Severe": 0.45,"Marine": 0.40}
EXPOSURE_MIN_CEMENT = {"Mild": 300, "Moderate": 300, "Severe": 320,"Very Severe": 340, "Marine": 360}
EXPOSURE_MIN_GRADE = {"Mild": "M20", "Moderate": "M25", "Severe": "M30","Very Severe": "M35", "Marine": "M40"}
GRADE_STRENGTH = {"M10": 10, "M15": 15, "M20": 20, "M25": 25,"M30": 30, "M35": 35, "M40": 40, "M45": 45, "M50": 50}
WATER_BASELINE = {10: 208, 12.5: 202, 20: 186, 40: 165}
AGG_SHAPE_WATER_ADJ = {"Angular (baseline)": 0.00, "Sub-angular": -0.03,"Sub-rounded": -0.05, "Rounded": -0.07,"Flaky/Elongated": +0.03}
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
    # Grade
    grade_match = re.search(r"\bM\s*(10|15|20|25|30|35|40|45|50)\b", text, re.IGNORECASE)
    if grade_match: result["grade"] = "M" + grade_match.group(1)
    # Exposure
    for exp in EXPOSURE_WB_LIMITS.keys():
        if re.search(exp, text, re.IGNORECASE): result["exposure"] = exp; break
    # Slump (more flexible)
    slump_match = re.search(r"(slump\s*(of\s*)?|)\b(\d{2,3})\s*mm", text, re.IGNORECASE)
    if slump_match: result["slump"] = int(slump_match.group(3))
    # Cement
    cement_types = ["OPC 33", "OPC 43", "OPC 53", "PPC"]
    for ctype in cement_types:
        if re.search(ctype.replace(" ", r"\s*"), text, re.IGNORECASE):
            result["cement"] = ctype; break
    # Aggregate size
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
# app.py — CivilGPT v2.1 (Part 2/3)

# =========================
# Mix Evaluation
# =========================
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
    vol_cem = cementitious / 3150.0
    vol_wat = water / 1000.0
    vol_sp  = sp / 1200.0
    vol_binder = vol_cem + vol_wat + vol_sp
    total_vol = 1.0
    vol_agg = total_vol - vol_binder
    if vol_agg <= 0: vol_agg = 0.60
    vol_fine = vol_agg * fine_fraction
    vol_coarse = vol_agg * (1.0 - fine_fraction)
    mass_fine = vol_fine * density_fa
    mass_coarse = vol_coarse * density_ca
    return float(mass_fine), float(mass_coarse)

# =========================
# Compliance Checks
# =========================
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

def compliance_table(checks: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(checks.items()), columns=["Check", "Status"])
    df["Result"] = df["Status"].apply(lambda x: "✅ Pass" if x else "❌ Fail")
    return df[["Check", "Result"]]

# =========================
# Sanity Checks
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
        return ["Insufficient data to run sanity checks."]
    if cement < 250: warnings.append(f"Cement too low ({cement:.1f} kg/m³).")
    if cement > 500: warnings.append(f"Cement unusually high ({cement:.1f} kg/m³).")
    if water < 140 or water > 220: warnings.append(f"Water out of typical range ({water:.1f} kg/m³).")
    if fine < 500 or fine > 900: warnings.append(f"Fine aggregate unusual ({fine:.1f} kg/m³).")
    if coarse < 1000 or coarse > 1300: warnings.append(f"Coarse aggregate unusual ({coarse:.1f} kg/m³).")
    if sp > 20: warnings.append(f"SP dosage unusually high ({sp:.1f} kg/m³).")
    if unit_wt < 2200 or unit_wt > 2600: warnings.append(f"Unit weight {unit_wt:.1f} kg/m³ outside IS 10262 range.")
    return warnings

# =========================
# Unified Feasibility
# =========================
def check_feasibility(mix_df, meta, exposure):
    checks, derived = compliance_checks(mix_df, meta, exposure)
    warnings = sanity_check_mix(meta, mix_df)
    reasons = [f"Failed check: {k}" for k, v in checks.items() if not v]
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
            if not (lo <= p <= hi): ok = False; msgs.append(f"{sieve} mm → {p:.1f}% (req {lo}-{hi}%)")
        if ok and not msgs: msgs = [f"Fine aggregate meets IS 383 {zone} limits."]
        return ok, msgs
    except: return False, ["Invalid fine aggregate CSV format."]

def sieve_check_ca(df: pd.DataFrame, nominal_mm: int):
    try:
        limits = COARSE_LIMITS[int(nominal_mm)]
        ok, msgs = True, []
        for sieve, (lo, hi) in limits.items():
            row = df.loc[df["Sieve_mm"].astype(str) == sieve]
            if row.empty:
                ok = False; msgs.append(f"Missing sieve {sieve} mm."); continue
            p = float(row["PercentPassing"].iloc[0])
            if not (lo <= p <= hi): ok = False; msgs.append(f"{sieve} mm → {p:.1f}% (req {lo}-{hi}%)")
        if ok and not msgs: msgs = [f"Coarse aggregate meets IS 383 ({nominal_mm} mm graded)."]
        return ok, msgs
    except: return False, ["Invalid coarse aggregate CSV format."]

# =========================
# Mix Generators
# =========================
def generate_mix(grade, exposure, nom_max, target_slump, agg_shape,
                 emissions, costs, cement_choice, use_sp=True, sp_reduction=0.18,
                 optimize_cost=False, fine_fraction=0.40):
    w_b_limit = float(EXPOSURE_WB_LIMITS[exposure])
    min_cem = float(EXPOSURE_MIN_CEMENT[exposure])
    target_water = water_for_slump_and_shape(
        nom_max_mm=nom_max, slump_mm=int(target_slump),
        agg_shape=agg_shape, uses_sp=use_sp, sp_reduction_frac=sp_reduction
    )
    best_df, best_meta, best_score = None, None, float("inf")
    trace = []
    wb_values = np.linspace(0.35, w_b_limit, 6)
    flyash_options, ggbs_options = [0.0, 0.2, 0.3], [0.0, 0.3, 0.5]
    for wb in wb_values:
        for flyash_frac in flyash_options:
            for ggbs_frac in ggbs_options:
                if flyash_frac + ggbs_frac > 0.50: continue
                binder = max(target_water / wb, min_cem)
                cement = binder * (1 - flyash_frac - ggbs_frac)
                flyash, ggbs = binder * flyash_frac, binder * ggbs_frac
                sp = 2.5 if use_sp else 0.0
                fine, coarse = compute_aggregates(binder, target_water, sp, fine_fraction)
                mix = {cement_choice: cement,"Fly Ash": flyash,"GGBS": ggbs,"Water": target_water,
                       "PCE Superplasticizer": sp,"M-Sand": fine,"20mm Coarse Aggregate": coarse}
                df = evaluate_mix(mix, emissions, costs)
                co2_total, cost_total = float(df["CO2_Emissions (kg/m3)"].sum()), float(df["Cost (₹/m3)"].sum())
                candidate_meta = {"w_b": wb, "cementitious": binder, "cement": cement,
                                  "flyash": flyash, "ggbs": ggbs, "water_target": target_water,
                                  "sp": sp, "fine": fine, "coarse": coarse,
                                  "scm_total_frac": flyash_frac + ggbs_frac,
                                  "grade": grade, "exposure": exposure,
                                  "nom_max": nom_max, "slump": target_slump,
                                  "co2_total": co2_total, "cost_total": cost_total}
                feasible, reasons, derived, checks = check_feasibility(df, candidate_meta, exposure)
                score = co2_total if not optimize_cost else co2_total + 0.001 * cost_total
                trace.append({"wb": float(wb), "flyash_frac": float(flyash_frac),
                              "ggbs_frac": float(ggbs_frac),"co2": float(co2_total),
                              "cost": float(cost_total),"score": float(score),
                              "feasible": bool(feasible),"reasons": reasons})
                if feasible and score < best_score:
                    best_df, best_score, best_meta = df.copy(), score, candidate_meta.copy()
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
    fine, coarse = compute_aggregates(cementitious, water_target, sp, fine_fraction)
    mix = {cement_choice: cementitious,"Fly Ash": 0.0,"GGBS": 0.0,"Water": water_target,
           "PCE Superplasticizer": sp,"M-Sand": fine,"20mm Coarse Aggregate": coarse}
    df = evaluate_mix(mix, emissions, costs)
    meta = {"w_b": w_b_limit, "cementitious": cementitious, "cement": cementitious,
            "flyash": 0.0, "ggbs": 0.0, "water_target": water_target,
            "sp": sp, "fine": fine, "coarse": coarse,
            "scm_total_frac": 0.0, "grade": grade, "exposure": exposure,
            "nom_max": nom_max, "slump": target_slump,
            "co2_total": float(df["CO2_Emissions (kg/m3)"].sum()),
            "cost_total": float(df["Cost (₹/m3)"].sum())}
    return df, meta
# app.py — CivilGPT v2.1 (Part 3/3)

# =========================
# Parser Override
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
# Landing Page UI
# =========================
st.title("🧱 CivilGPT — Sustainable Concrete Mix Designer (v2.1)")
st.caption("Professional Tabbed UI · IS:10262 compliant checks · CO₂ optimized")

user_text = st.text_area("💬 Describe your mix requirements in plain English", height=120,
                         placeholder="e.g., Design M30 concrete, moderate exposure, slump of 100 mm, OPC 53, 20 mm aggregates, with superplasticizer.")

manual_mode = st.toggle("Switch to Manual Input Mode")

# Manual sidebar only appears if toggle ON
if manual_mode:
    st.sidebar.header("📝 Manual Mix Inputs")
    use_llm_parser = st.sidebar.checkbox("Use Groq LLM Parser", value=False)
    show_trace = st.sidebar.checkbox("Show optimizer trace (advanced)", value=False)

    grade = st.sidebar.selectbox("Concrete Grade", list(GRADE_STRENGTH.keys()), index=2)
    exposure = st.sidebar.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()), index=2)
    cement_choice = st.sidebar.selectbox("Cement Type", ["OPC 33", "OPC 43", "OPC 53", "PPC"], index=2)

    nom_max = st.sidebar.selectbox("Nominal max aggregate (mm)", [10, 12.5, 20, 40], index=2)
    agg_shape = st.sidebar.selectbox("Aggregate shape", list(AGG_SHAPE_WATER_ADJ.keys()), index=0)
    target_slump = st.sidebar.slider("Target slump (mm)", 25, 180, 100, step=5)

    use_sp = st.sidebar.checkbox("Use Superplasticizer (PCE)", True)
    optimize_cost = st.sidebar.checkbox("Optimize for CO₂ + Cost", False)

    fine_fraction = st.sidebar.slider("Fine aggregate fraction", 0.3, 0.5, 0.40, step=0.01)

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
else:
    # Defaults for non-manual mode
    use_llm_parser, show_trace = False, False
    grade, exposure, cement_choice = "M30", "Moderate", "OPC 53"
    nom_max, agg_shape, target_slump = 20, "Angular (baseline)", 100
    use_sp, optimize_cost, fine_fraction = True, False, 0.40
    qc_level, air_pct, fa_moist, ca_moist = "Good", 2.0, 0.0, 0.0
    fa_abs, ca_abs = 1.0, 0.5
    fine_zone, fine_csv, coarse_csv = "Zone II", None, None
    materials_file, emissions_file, cost_file = None, None, None

# Load datasets
materials_df, emissions_df, costs_df = load_data(materials_file, emissions_file, cost_file)

# =========================
# Main Run Button
# =========================
if st.button("🚀 Generate Sustainable Mix"):
    try:
        inputs = {
            "grade": grade, "exposure": exposure, "cement_choice": cement_choice,
            "nom_max": nom_max, "agg_shape": agg_shape, "target_slump": target_slump,
            "use_sp": use_sp, "optimize_cost": optimize_cost, "fine_fraction": fine_fraction
        }
        inputs, msgs, parsed = apply_parser(user_text, inputs)
        for m in msgs: st.info(m)

        min_grade_required = EXPOSURE_MIN_GRADE[inputs["exposure"]]
        grade_order = list(GRADE_STRENGTH.keys())
        if grade_order.index(inputs["grade"]) < grade_order.index(min_grade_required):
            st.warning(f"Exposure {inputs['exposure']} requires ≥ {min_grade_required}. Adjusted automatically.")
            inputs["grade"] = min_grade_required

        fck, S = GRADE_STRENGTH[inputs["grade"]], QC_STDDEV[qc_level]
        fck_target = fck + 1.65 * S

        opt_df, opt_meta, trace = generate_mix(
            inputs["grade"], inputs["exposure"], inputs["nom_max"], inputs["target_slump"], inputs["agg_shape"],
            emissions_df, costs_df, inputs["cement_choice"],
            use_sp=inputs["use_sp"], optimize_cost=inputs["optimize_cost"], fine_fraction=inputs["fine_fraction"]
        )
        base_df, base_meta = generate_baseline(
            inputs["grade"], inputs["exposure"], inputs["nom_max"], inputs["target_slump"], inputs["agg_shape"],
            emissions_df, costs_df, inputs["cement_choice"],
            use_sp=inputs["use_sp"], fine_fraction=inputs["fine_fraction"]
        )

        if opt_df is None or base_df is None:
            st.error("No feasible mix found.")
        else:
            for m in (opt_meta, base_meta):
                m["fck"], m["fck_target"], m["stddev_S"] = fck, round(fck_target, 1), S
            st.success(f"Mixes generated for **{inputs['grade']}** under **{inputs['exposure']}** exposure.")

            # Tabs
            tabs = st.tabs(["Overview","Optimized Mix","Baseline Mix","Trace & Calculations","Sieve & QA","Downloads"])

            # ---- Overview
            with tabs[0]:
                co2_opt, cost_opt = opt_meta["co2_total"], opt_meta["cost_total"]
                co2_base, cost_base = base_meta["co2_total"], base_meta["cost_total"]
                reduction = (co2_base - co2_opt) / co2_base * 100 if co2_base > 0 else 0.0
                cost_diff = cost_opt - cost_base
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("🌱 Optimized CO₂ (kg/m³)", f"{co2_opt:.1f}")
                with col2: st.metric("🏗 Baseline CO₂ (kg/m³)", f"{co2_base:.1f}")
                with col3: st.metric("📉 Reduction (%)", f"{reduction:.1f}")
                with col4: st.metric("💰 Cost Δ (₹/m³)", f"{cost_diff:+.2f}")
                st.markdown("#### 📊 CO₂ Comparison")
                fig, ax = plt.subplots()
                ax.bar(["Optimized", "Baseline"], [co2_opt, co2_base], color=["green","gray"])
                ax.set_ylabel("CO₂ (kg/m³)")
                st.pyplot(fig)

            # ---- Optimized Mix
            with tabs[1]:
                st.dataframe(opt_df, use_container_width=True)
                feasible, reasons, derived, _ = check_feasibility(opt_df, opt_meta, inputs["exposure"])
                st.json(derived)
                if reasons: st.warning("\n".join(reasons))
                else: st.success("✅ Optimized mix passes all IS compliance checks.")

            # ---- Baseline Mix
            with tabs[2]:
                st.dataframe(base_df, use_container_width=True)
                feasible, reasons, derived, _ = check_feasibility(base_df, base_meta, inputs["exposure"])
                st.json(derived)
                if reasons: st.warning("\n".join(reasons))
                else: st.success("✅ Baseline mix passes all IS compliance checks.")

            # ---- Trace & Calculations
            with tabs[3]:
                if trace:
                    trace_df = pd.DataFrame(trace)
                    st.dataframe(trace_df, use_container_width=True)
                    st.markdown("#### Scatter: CO₂ vs Cost (all candidates)")
                    fig, ax = plt.subplots()
                    ax.scatter(trace_df["co2"], trace_df["cost"], c=["green" if f else "red" for f in trace_df["feasible"]])
                    ax.set_xlabel("CO₂ (kg/m³)")
                    ax.set_ylabel("Cost (₹/m³)")
                    st.pyplot(fig)
                else: st.info("Trace not available.")

            # ---- Sieve & QA
            with tabs[4]:
                if fine_csv is not None:
                    df_fine = pd.read_csv(fine_csv)
                    ok_fa, msgs_fa = sieve_check_fa(df_fine, fine_zone)
                    for m in msgs_fa: st.write(("✅ " if ok_fa else "❌ ") + m)
                else: st.info("Fine sieve CSV not provided.")
                if coarse_csv is not None:
                    df_coarse = pd.read_csv(coarse_csv)
                    ok_ca, msgs_ca = sieve_check_ca(df_coarse, inputs["nom_max"])
                    for m in msgs_ca: st.write(("✅ " if ok_ca else "❌ ") + m)
                else: st.info("Coarse sieve CSV not provided.")

            # ---- Downloads
            with tabs[5]:
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
                story = [
                    Paragraph("CivilGPT Sustainable Mix Report", styles["Title"]),
                    Spacer(1, 8),
                    Paragraph(f"Grade: {inputs['grade']} | Exposure: {inputs['exposure']} | Cement: {inputs['cement_choice']}", styles["Normal"]),
                    Spacer(1, 8),
                ]
                data_summary = [
                    ["Metric", "Optimized", "Baseline"],
                    ["CO₂ (kg/m³)", f"{opt_meta['co2_total']:.1f}", f"{base_meta['co2_total']:.1f}"],
                    ["Cost (₹/m³)", f"{opt_meta['cost_total']:.2f}", f"{base_meta['cost_total']:.2f}"],
                    ["Reduction (%)", f"{reduction:.1f}", "-"]
                ]
                tbl = Table(data_summary, hAlign="LEFT")
                tbl.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.grey)]))
                story.append(tbl)
                doc.build(story)
                pdf_bytes = pdf_buffer.getvalue()
                st.download_button("Optimized Mix (CSV)", csv_opt, "optimized_mix.csv", "text/csv")
                st.download_button("Baseline Mix (CSV)", csv_base, "baseline_mix.csv", "text/csv")
                st.download_button("Report (Excel)", excel_bytes, "CivilGPT_Report.xlsx")
                st.download_button("Report (PDF)", pdf_bytes, "CivilGPT_Report.pdf")

    except Exception as e:
        st.error(f"Error: {e}")
        st.text(traceback.format_exc())
else:
    st.info("Enter a prompt above or switch to Manual Input Mode to set parameters.")

st.caption("CivilGPT v2.1 | ChatGPT-like UI + IS-code backend preserved")
