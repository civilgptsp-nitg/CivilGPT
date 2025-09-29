# app.py — CivilGPT v2.1 (Backend + Landing UI)
# Part 1/2 — Backend functions (full, uncut)

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

# =========================
# Material Aliasing
# =========================
ALIASES = {
    "opc53": "OPC 53",
    "53gradeopc": "OPC 53",
    "ordinaryportlandcement53": "OPC 53",
    "opc43": "OPC 43",
    "43gradeopc": "OPC 43",
    "opc33": "OPC 33",
    "33gradeopc": "OPC 33",
    "ppc": "PPC",
    "portlandpozzolanacement": "PPC",
}

def normalize_material(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return ALIASES.get(key, name)

# =========================
# Parsers
# =========================
def simple_parse(text: str) -> dict:
    result = {}
    grade_match = re.search(r"\bM(10|15|20|25|30|35|40|45|50)\b", text, re.IGNORECASE)
    if grade_match: result["grade"] = "M" + grade_match.group(1)

    for exp in EXPOSURE_WB_LIMITS.keys():
        if re.search(exp, text, re.IGNORECASE):
            result["exposure"] = exp; break

    slump_match = re.search(r"slump\s*(\d+)", text, re.IGNORECASE)
    if slump_match: result["slump"] = int(slump_match.group(1))

    cement_types = ["OPC 33", "OPC 43", "OPC 53", "PPC"]
    for ctype in cement_types:
        if re.search(ctype.replace(" ", r"\s*"), text, re.IGNORECASE):
            result["cement"] = ctype; break

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
# Clarification Flow Helper
# =========================
def missing_required(parsed: dict) -> list:
    """Check if essential fields are missing after parsing."""
    required = ["grade", "slump", "exposure"]
    return [r for r in required if r not in parsed]

# =========================
# (Rest of backend: evaluation, feasibility, compliance, generate_mix, baseline, etc.)
# Will continue in Part 2
# =========================
# Mix Evaluation & Feasibility (continuation from Part 1)
# =========================
def evaluate_mix(components_dict, emissions_df, costs_df=None):
    comp_items = [(normalize_material(m.strip()), q) for m, q in components_dict.items()]
    comp_df = pd.DataFrame(comp_items, columns=["Material_norm", "Quantity (kg/m3)"])

    emissions_df = emissions_df.copy()
    emissions_df["Material_norm"] = emissions_df["Material"].str.strip().apply(normalize_material)

    df = comp_df.merge(emissions_df[["Material_norm","CO2_Factor(kg_CO2_per_kg)"]],
                       on="Material_norm", how="left")

    if "CO2_Factor(kg_CO2_per_kg)" not in df.columns:
        df["CO2_Factor(kg_CO2_per_kg)"] = 0.0
    df["CO2_Factor(kg_CO2_per_kg)"] = df["CO2_Factor(kg_CO2_per_kg)"].fillna(0.0)
    df["CO2_Emissions (kg/m3)"] = df["Quantity (kg/m3)"] * df["CO2_Factor(kg_CO2_per_kg)"]

    if costs_df is not None and "Cost(₹/kg)" in costs_df.columns:
        costs_df = costs_df.copy()
        costs_df["Material_norm"] = costs_df["Material"].str.strip().apply(normalize_material)
        df = df.merge(costs_df[["Material_norm","Cost(₹/kg)"]], on="Material_norm", how="left")
        df["Cost(₹/kg)"] = df["Cost(₹/kg)"].fillna(0.0)
        df["Cost (₹/m3)"] = df["Quantity (kg/m3)"] * df["Cost(₹/kg)"]
    else:
        df["Cost (₹/m3)"] = 0.0

    df["Material"] = df["Material_norm"]
    df = df[["Material","Quantity (kg/m3)","CO2_Factor(kg_CO2_per_kg)","CO2_Emissions (kg/m3)","Cost(₹/kg)","Cost (₹/m3)"]]
    return df


# --- Sanity & compliance helpers (same as v2.0, trimmed here for brevity) ---
def compute_aggregates(cementitious, water, sp, fine_fraction=0.40,
                       density_fa=2650.0, density_ca=2700.0):
    vol_cem = cementitious / 3150.0
    vol_wat = water / 1000.0
    vol_sp  = sp / 1200.0
    vol_binder = vol_cem + vol_wat + vol_sp
    total_vol = 1.0
    vol_agg = total_vol - vol_binder
    if vol_agg <= 0:
        vol_agg = 0.60
    vol_fine = vol_agg * fine_fraction
    vol_coarse = vol_agg * (1.0 - fine_fraction)
    mass_fine = vol_fine * density_fa
    mass_coarse = vol_coarse * density_ca
    return float(mass_fine), float(mass_coarse)

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
    return checks


# --- Mix generation (same as v2.0, kept intact) ---
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
    flyash_options = [0.0, 0.2, 0.3]
    ggbs_options = [0.0, 0.3, 0.5]
    for wb in wb_values:
        for flyash_frac in flyash_options:
            for ggbs_frac in ggbs_options:
                if flyash_frac + ggbs_frac > 0.50: continue
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
                candidate_meta = {
                    "w_b": wb, "cementitious": binder, "cement": cement,
                    "flyash": flyash, "ggbs": ggbs, "water_target": target_water,
                    "sp": sp, "fine": fine, "coarse": coarse,
                    "scm_total_frac": flyash_frac + ggbs_frac,
                    "grade": grade, "exposure": exposure,
                    "nom_max": nom_max, "slump": target_slump,
                    "co2_total": co2_total
                }
                trace.append({"wb": float(wb), "flyash_frac": float(flyash_frac),
                              "ggbs_frac": float(ggbs_frac), "co2": float(co2_total)})
                if not sanity_check_mix(candidate_meta, df) and all(compliance_checks(df, candidate_meta, exposure).values()):
                    score = co2_total
                    if score < best_score:
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
        "Fly Ash": 0.0, "GGBS": 0.0, "Water": water_target,
        "PCE Superplasticizer": sp, "M-Sand": fine, "20mm Coarse Aggregate": coarse,
    }
    df = evaluate_mix(mix, emissions, costs)
    meta = {"w_b": w_b_limit, "cementitious": cementitious, "cement": cementitious,
            "flyash": 0.0, "ggbs": 0.0, "water_target": water_target,
            "sp": sp, "fine": fine, "coarse": coarse,
            "scm_total_frac": 0.0, "grade": grade, "exposure": exposure,
            "nom_max": nom_max, "slump": target_slump,
            "co2_total": float(df["CO2_Emissions (kg/m3)"].sum())}
    return df, meta


# =========================
# UI: Landing Page + Simplified Sidebar
# =========================
st.markdown("<h1 style='text-align:center'>What can I help with?</h1>", unsafe_allow_html=True)
user_text = st.text_input(" ", placeholder="Describe your concrete mix (e.g., M30, moderate exposure, OPC 53)...")
manual_mode = st.toggle("Manual Input Mode")

# Sidebar (only if manual mode)
if manual_mode:
    with st.sidebar:
        st.header("Manual Mix Inputs")
        grade = st.selectbox("Concrete Grade", list(GRADE_STRENGTH.keys()), index=2)
        exposure = st.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()), index=2)
        cement_choice = st.selectbox("Cement Type", ["OPC 33", "OPC 43", "OPC 53", "PPC"], index=2)
        nom_max = st.selectbox("Nominal max aggregate (mm)", [10, 12.5, 20, 40], index=2)
        target_slump = st.slider("Target slump (mm)", 25, 180, 100, step=5)
        agg_shape = st.selectbox("Aggregate shape", list(AGG_SHAPE_WATER_ADJ.keys()), index=0)
        use_sp = st.checkbox("Use Superplasticizer (PCE)", True)
        fine_fraction = st.slider("Fine aggregate fraction", 0.3, 0.5, 0.40, step=0.01)
else:
    # Defaults if manual mode off
    grade, exposure, cement_choice, nom_max = "M30", "Moderate", "OPC 53", 20
    target_slump, agg_shape, use_sp, fine_fraction = 100, "Angular (baseline)", True, 0.40

# =========================
# Run button
# =========================
if st.button("Generate Sustainable Mix"):
    try:
        parsed = simple_parse(user_text) if not manual_mode else {}
        if missing_required(parsed) and not manual_mode:
            st.error("Please include grade, exposure, and slump in your input.")
            st.stop()

        opt_df, opt_meta, trace = generate_mix(
            grade, exposure, nom_max, target_slump, agg_shape,
            emissions_df, costs_df, cement_choice, use_sp=use_sp, fine_fraction=fine_fraction
        )
        base_df, base_meta = generate_baseline(
            grade, exposure, nom_max, target_slump, agg_shape,
            emissions_df, costs_df, cement_choice, use_sp=use_sp, fine_fraction=fine_fraction
        )

        if opt_df is None:
            st.error("No feasible mix found.")
        else:
            st.success(f"Mix generated for **{grade}** under **{exposure}** exposure.")

            tabs = st.tabs(["Overview","Optimized Mix","Baseline Mix","Trace","Downloads"])
            with tabs[0]:
                st.metric("Optimized CO₂", f"{opt_meta['co2_total']:.1f} kg/m³")
                st.metric("Baseline CO₂", f"{base_meta['co2_total']:.1f} kg/m³")

            with tabs[1]:
                st.dataframe(opt_df, use_container_width=True)

            with tabs[2]:
                st.dataframe(base_df, use_container_width=True)

            with tabs[3]:
                st.dataframe(pd.DataFrame(trace), use_container_width=True)

            with tabs[4]:
                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer)
                styles = getSampleStyleSheet()
                story = [Paragraph("CivilGPT Sustainable Mix Report", styles["Title"])]
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"Grade: {grade}, Exposure: {exposure}, Cement: {cement_choice}", styles["Normal"]))
                doc.build(story)
                st.download_button("Download Report (PDF)", pdf_buffer.getvalue(), "CivilGPT_Report.pdf")
    except Exception as e:
        st.error(f"Error: {e}")
        st.text(traceback.format_exc())
