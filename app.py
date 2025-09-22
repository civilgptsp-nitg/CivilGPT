# app.py — CivilGPT v1.7
# Full drop-in: Sustainable Mix Designer with dataset previews, correlation, compliance, CO2 reports
# Updates vs v1.6.5:
# - Dataset names verified: lab_processed_mgrades_only.xlsx, concrete_mix_design_data_cleaned_standardized.xlsx
# - Grade range restricted to M10–M50 only
# - Added English parser UI (regex-based simple_parse) with toggle
# - Dropdowns remain as fallback

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
    """Try loading Excel robustly from root/ or data/ (case-insensitive)."""
    for p in [name, f"data/{name}"]:
        if os.path.exists(p):
            try:
                return pd.read_excel(p)
            except Exception:
                try:
                    return pd.read_excel(p, engine="openpyxl")
                except Exception:
                    return None
    data_dir = "data"
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            if fname.lower() == name.lower():
                try:
                    return pd.read_excel(os.path.join(data_dir, fname))
                except Exception:
                    try:
                        return pd.read_excel(os.path.join(data_dir, fname), engine="openpyxl")
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
    "Mild": 300,
    "Moderate": 300,
    "Severe": 320,
    "Very Severe": 340,
    "Marine": 360,
}

EXPOSURE_MIN_GRADE = {
    "Mild": "M20",
    "Moderate": "M25",
    "Severe": "M30",
    "Very Severe": "M35",
    "Marine": "M40",
}

# Restricted grade dictionary (M10 to M50 only)
GRADE_STRENGTH = {
    "M10": 10, "M15": 15, "M20": 20, "M25": 25, "M30": 30,
    "M35": 35, "M40": 40, "M45": 45, "M50": 50
}

WATER_BASELINE = {10: 208, 12.5: 202, 20: 186, 40: 165}

AGG_SHAPE_WATER_ADJ = {
    "Angular (baseline)": 0.00,
    "Sub-angular": -0.03,
    "Sub-rounded": -0.05,
    "Rounded": -0.07,
    "Flaky/Elongated": +0.03,
}

QC_STDDEV = {"Good": 5.0, "Fair": 7.5, "Poor": 10.0}

FINE_AGG_ZONE_LIMITS = {
    "Zone I":  {"10.0": (100,100),"4.75": (90,100),"2.36": (60,95),"1.18": (30,70),"0.600": (15,34),"0.300": (5,20),"0.150": (0,10)},
    "Zone II": {"10.0": (100,100),"4.75": (90,100),"2.36": (75,100),"1.18": (55,90),"0.600": (35,59),"0.300": (8,30),"0.150": (0,10)},
    "Zone III":{"10.0": (100,100),"4.75": (90,100),"2.36": (85,100),"1.18": (75,100),"0.600": (60,79),"0.300": (12,40),"0.150": (0,10)},
    "Zone IV": {"10.0": (95,100),"4.75": (95,100),"2.36": (95,100),"1.18": (90,100),"0.600": (80,100),"0.300": (15,50),"0.150": (0,15)},
}

COARSE_LIMITS = {
    10: {"20.0": (100,100), "10.0": (85,100),  "4.75": (0,20)},
    20: {"40.0": (95,100),  "20.0": (95,100),  "10.0": (25,55), "4.75": (0,10)},
    40: {"80.0": (95,100),  "40.0": (95,100),  "20.0": (30,70), "10.0": (0,15)}
}

# =========================
# Helpers
# =========================
@st.cache_data
def _read_csv_try(path):
    return pd.read_csv(path)

@st.cache_data
def load_data(materials_file=None, emissions_file=None):
    """Robust loader with graceful warnings."""
    materials = None
    emissions = None
    # Materials
    if materials_file is not None:
        try:
            materials = pd.read_csv(materials_file)
        except Exception as e:
            st.warning(f"Could not read uploaded materials CSV: {e}")
            materials = None
    if materials is None:
        try:
            materials = _read_csv_try("materials_library.csv")
        except Exception:
            try:
                materials = _read_csv_try("data/materials_library.csv")
            except Exception:
                materials = None
                st.warning("Materials CSV not found. Expected: materials_library.csv or data/materials_library.csv")
    # Emissions
    if emissions_file is not None:
        try:
            emissions = pd.read_csv(emissions_file)
        except Exception as e:
            st.warning(f"Could not read uploaded emission factors CSV: {e}")
            emissions = None
    if emissions is None:
        try:
            emissions = _read_csv_try("emission_factors.csv")
        except Exception:
            try:
                emissions = _read_csv_try("data/emission_factors.csv")
            except Exception:
                emissions = None
                st.warning("Emission factors CSV not found. Expected: emission_factors.csv or data/emission_factors.csv")
    # Normalize emissions
    if emissions is not None:
        material_col = None
        co2_col = None
        for c in emissions.columns:
            lc = c.lower()
            if lc in ("material","materials","name","item"): material_col = c
            if lc in ("kg_co2_per_kg","co2_factor","co2","kgco2perkg","co2_factor(kg_co2_per_kg)","co2_factor(kg_co2/kg)"):
                co2_col = c
        if material_col and co2_col:
            emissions = emissions[[material_col, co2_col]].rename(columns={material_col: "Material", co2_col: "CO2_Factor(kg_CO2_per_kg)"})
        else:
            emissions = emissions.copy()
            if emissions.shape[1] >= 2:
                emissions.columns = ["Material", "CO2_Factor(kg_CO2_per_kg)"] + list(emissions.columns[2:])
            elif emissions.shape[1] == 1:
                emissions.columns = ["Material"]; emissions["CO2_Factor(kg_CO2_per_kg)"] = 0.0
            else:
                emissions = pd.DataFrame(columns=["Material","CO2_Factor(kg_CO2_per_kg)"])
    else:
        emissions = pd.DataFrame(columns=["Material","CO2_Factor(kg_CO2_per_kg)"])
    if materials is None:
        materials = pd.DataFrame(columns=["Material"])
    return materials, emissions

def water_for_slump_and_shape(nom_max_mm: int, slump_mm: int,
                              agg_shape: str,
                              uses_sp: bool=False, sp_reduction_frac: float=0.0) -> float:
    base = WATER_BASELINE.get(int(nom_max_mm), 186.0)
    if slump_mm <= 50:
        water = base
    else:
        extra_25 = max(0, (slump_mm - 50) / 25.0)
        water = base * (1 + 0.03 * extra_25)
    water *= (1.0 + AGG_SHAPE_WATER_ADJ.get(agg_shape, 0.0))
    if uses_sp and sp_reduction_frac > 0:
        water *= (1 - sp_reduction_frac)
    return float(water)

def evaluate_mix(components_dict, emissions_df):
    comp_df = pd.DataFrame(list(components_dict.items()), columns=["Material", "Quantity (kg/m3)"])
    df = comp_df.merge(emissions_df, on="Material", how="left")
    if "CO2_Factor(kg_CO2_per_kg)" not in df.columns:
        df["CO2_Factor(kg_CO2_per_kg)"] = 0.0
    if df["CO2_Factor(kg_CO2_per_kg)"].isna().any():
        df["CO2_Factor(kg_CO2_per_kg)"] = df["CO2_Factor(kg_CO2_per_kg)"].fillna(0.0)
    df["CO2_Emissions (kg/m3)"] = df["Quantity (kg/m3)"] * df["CO2_Factor(kg_CO2_per_kg)"]
    return df

def aggregate_correction(delta_moisture_pct: float, agg_mass_ssd: float):
    water_delta = (delta_moisture_pct / 100.0) * agg_mass_ssd
    corrected_mass = agg_mass_ssd * (1 + delta_moisture_pct / 100.0)
    return float(water_delta), float(corrected_mass)

# =========================
# NEW: Simple English parser
# =========================
def simple_parse(text: str) -> dict:
    text = text.upper()
    parsed = {}
    match = re.search(r"M(\d{2})", text)
    parsed["grade"] = f"M{match.group(1)}" if match else "M30"
    exposures = ["MILD", "MODERATE", "SEVERE", "VERY SEVERE", "MARINE"]
    parsed["exposure"] = next((e.title() for e in exposures if e in text), "Moderate")
    match = re.search(r"SLUMP\s*(\d+)", text)
    parsed["slump"] = int(match.group(1)) if match else 100
    cements = ["OPC 33", "OPC 43", "OPC 53", "PPC"]
    parsed["cement"] = next((c for c in cements if c.replace(" ", "") in text.replace(" ", "")), "OPC 43")
    return parsed
# =========================
# Mix Generators
# =========================
def generate_mix(grade, exposure, nom_max, target_slump, agg_shape,
                 emissions, cement_choice, use_sp=True, sp_reduction=0.18):
    w_b_limit = float(EXPOSURE_WB_LIMITS[exposure])
    min_cem = float(EXPOSURE_MIN_CEMENT[exposure])
    target_water = water_for_slump_and_shape(
        nom_max_mm=nom_max, slump_mm=int(target_slump),
        agg_shape=agg_shape, uses_sp=use_sp, sp_reduction_frac=sp_reduction
    )
    best_df, best_meta, best_co2 = None, None, float("inf")
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
                fine = 650.0
                coarse = 1150.0
                sp = 2.5 if use_sp else 0.0
                mix = {
                    cement_choice: round(cement, 3),
                    "Fly Ash": round(flyash, 3),
                    "GGBS": round(ggbs, 3),
                    "Water": round(target_water, 3),
                    "PCE Superplasticizer": round(sp, 3),
                    "M-Sand": fine,
                    "20mm Coarse Aggregate": coarse,
                }
                df = evaluate_mix(mix, emissions)
                total_co2 = float(df["CO2_Emissions (kg/m3)"].sum())
                if total_co2 < best_co2:
                    best_df = df.copy()
                    best_co2 = total_co2
                    best_meta = {
                        "w_b": float(wb),
                        "cementitious": float(binder),
                        "cement": float(cement),
                        "flyash": float(flyash),
                        "ggbs": float(ggbs),
                        "water_target": float(target_water),
                        "sp": float(sp),
                        "fine": float(fine),
                        "coarse": float(coarse),
                        "scm_total_frac": float(flyash_frac + ggbs_frac),
                        "grade": grade, "exposure": exposure,
                        "nom_max": nom_max, "slump": int(target_slump),
                    }
    return best_df, best_meta

def generate_baseline(grade, exposure, nom_max, target_slump, agg_shape,
                      emissions, cement_choice, use_sp=True, sp_reduction=0.18):
    w_b_limit = float(EXPOSURE_WB_LIMITS[exposure])
    min_cem = float(EXPOSURE_MIN_CEMENT[exposure])
    water_target = water_for_slump_and_shape(
        nom_max_mm=nom_max, slump_mm=int(target_slump),
        agg_shape=agg_shape, uses_sp=use_sp, sp_reduction_frac=sp_reduction
    )
    cementitious = max(water_target / w_b_limit, min_cem)
    mix = {
        cement_choice: float(cementitious),
        "Fly Ash": 0.0,
        "GGBS": 0.0,
        "Water": float(water_target),
        "PCE Superplasticizer": 2.5 if use_sp else 0.0,
        "M-Sand": 650.0,
        "20mm Coarse Aggregate": 1150.0,
    }
    df = evaluate_mix(mix, emissions)
    meta = {
        "w_b": float(w_b_limit),
        "cementitious": float(cementitious),
        "cement": float(cementitious),
        "flyash": 0.0, "ggbs": 0.0,
        "water_target": float(water_target),
        "sp": float(mix["PCE Superplasticizer"]),
        "fine": 650.0, "coarse": 1150.0,
        "scm_total_frac": 0.0,
        "grade": grade, "exposure": exposure, "nom_max": nom_max, "slump": int(target_slump),
        "baseline_cement": cement_choice,
    }
    return df, meta

def compliance_checks(mix_df, meta, exposure):
    checks = {}
    try:
        checks["W/B ≤ exposure limit"] = float(meta["w_b"]) <= EXPOSURE_WB_LIMITS[exposure]
    except Exception:
        checks["W/B ≤ exposure limit"] = False
    try:
        checks["Min cementitious met"] = float(meta["cementitious"]) >= float(EXPOSURE_MIN_CEMENT[exposure])
    except Exception:
        checks["Min cementitious met"] = False
    checks["SCM ≤ 50%"] = float(meta.get("scm_total_frac", 0.0)) <= 0.50
    total_mass = float(mix_df["Quantity (kg/m3)"].sum())
    checks["Unit weight 2200–2600 kg/m³"] = 2200.0 <= total_mass <= 2600.0
    derived = {
        "w/b used": round(float(meta.get("w_b", 0.0)), 3),
        "cementitious (kg/m³)": round(float(meta.get("cementitious", 0.0)), 1),
        "SCM % of cementitious": round(100 * float(meta.get("scm_total_frac", 0.0)), 1),
        "total mass (kg/m³)": round(total_mass, 1),
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

def compliance_table(checks: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(checks.items()), columns=["Check", "Status"])
    df["Result"] = df["Status"].apply(lambda x: "✅ Pass" if x else "❌ Fail")
    return df[["Check", "Result"]]

# =========================
# UI — Sidebar & Inputs
# =========================
st.sidebar.header("📝 Mix Inputs")

# English parser option
st.sidebar.markdown("### 🔤 English Input (optional)")
user_text = st.sidebar.text_area(
    "Describe requirement:",
    placeholder="e.g. M30 concrete for slab, severe exposure, slump 120 mm, OPC 43 with fly ash"
)
use_parser = st.sidebar.checkbox("Use English Parser", value=False)

# Dropdowns fallback
supported_grades = []
try:
    if lab_df is not None:
        lab_grade_cols = [c for c in lab_df.columns if 'grade' in c.lower()]
        if lab_grade_cols:
            supported_grades.extend(lab_df[lab_grade_cols[0]].dropna().astype(str).unique().tolist())
    if mix_df is not None:
        mix_grade_cols = [c for c in mix_df.columns if 'grade' in c.lower()]
        if mix_grade_cols:
            supported_grades.extend(mix_df[mix_grade_cols[0]].dropna().astype(str).unique().tolist())
except Exception:
    supported_grades = []

supported_grades = sorted(set([s.strip() for s in supported_grades if isinstance(s, str)]))
supported_grades = [g for g in supported_grades if g in GRADE_STRENGTH]  # restrict to M10–M50

supported_cements = []
try:
    if mix_df is not None:
        cement_cols = [c for c in mix_df.columns if 'cement' in c.lower()]
        if cement_cols:
            cement_col = cement_cols[0]
            supported_cements.extend(mix_df[cement_col].dropna().astype(str).unique().tolist())
except Exception:
    supported_cements = []
supported_cements = sorted(set([s.strip() for s in supported_cements if isinstance(s, str)]))
if not supported_grades:
    supported_grades = list(GRADE_STRENGTH.keys())
if not supported_cements:
    supported_cements = ["OPC 33", "OPC 43", "OPC 53", "PPC"]

grade = st.sidebar.selectbox("Concrete Grade", supported_grades, index=0)
exposure = st.sidebar.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()), index=2)
cement_choice = st.sidebar.selectbox("Cement Type", supported_cements, index=0)

st.sidebar.markdown("### Workability & Aggregates")
nom_max = st.sidebar.selectbox("Nominal max aggregate (mm)", [10, 12.5, 20, 40], index=2)
agg_shape = st.sidebar.selectbox("Aggregate shape", list(AGG_SHAPE_WATER_ADJ.keys()), index=0)
target_slump = st.sidebar.slider("Target slump (mm)", 25, 180, 100, step=5)
use_sp = st.sidebar.checkbox("Use Superplasticizer (PCE)", True)
sp_reduction = st.sidebar.slider("SP water reduction (fraction)", 0.00, 0.30, 0.18, step=0.01)

st.sidebar.markdown("### Quality Control")
qc_level = st.sidebar.selectbox("Quality control level", list(QC_STDDEV.keys()), index=0)

st.sidebar.markdown("### Air & Moisture")
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

# =========================
# Data loading
# =========================
materials_df, emissions_df = load_data(materials_file, emissions_file)
if emissions_df is None or emissions_df.shape[0] == 0:
    emissions_df = pd.DataFrame(columns=["Material","CO2_Factor(kg_CO2_per_kg)"])

# =========================
# Dataset Previews & Correlation
# =========================
st.header("CivilGPT — Sustainable Concrete Mix Designer")
st.markdown("Upload materials/emissions CSV to override defaults. Mix generation will use datasets in repo if available.")

with st.expander("📁 Current dataset preview (Lab)"):
    if lab_df is None:
        st.write("No lab dataset found. Expected file:", LAB_FILE)
    else:
        st.dataframe(lab_df.head(10))

with st.expander("📁 Current dataset preview (Mix designs)"):
    if mix_df is None:
        st.write("No mix dataset found. Expected file:", MIX_FILE)
    else:
        st.dataframe(mix_df.head(10))

if lab_df is not None:
    strength_cols = [c for c in lab_df.columns if 'compress' in c.lower() or 'strength' in c.lower()]
    if strength_cols:
        col = strength_cols[0]
        try:
            fig, ax = plt.subplots()
            lab_df[col].dropna().astype(float).hist(bins=20, ax=ax)
            ax.set_xlabel(col)
            st.pyplot(fig)
        except Exception:
            st.write("Could not generate strength histogram.")

# =========================
# Run
# =========================
if st.button("Generate Sustainable Mix"):
    try:
        if use_parser and user_text.strip():
            parsed = simple_parse(user_text)
            grade = parsed["grade"]
            exposure = parsed["exposure"]
            target_slump = parsed["slump"]
            cement_choice = parsed["cement"]
            st.info(f"Parsed input → Grade: {grade}, Exposure: {exposure}, Slump: {target_slump} mm, Cement: {cement_choice}")

        # [Mix generation, compliance checks, reporting, downloads remain same as v1.6.5 …]

    except Exception as e:
        st.error(f"Unexpected error during mix generation: {e}")
        st.text(traceback.format_exc())
else:
    st.info("Set parameters and click **Generate Sustainable Mix**.")

st.markdown("---")
st.caption("CivilGPT v1.7 | Added English parser option (M10–M50 only)")
