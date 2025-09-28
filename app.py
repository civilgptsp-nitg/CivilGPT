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

QC_STDDEV = {
    "Good": 5.0,
    "Fair": 7.5,
    "Poor": 10.0,
}

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
# English Parser
# =========================
def simple_parse(text: str) -> dict:
    result = {}
    grade_match = re.search(r"\bM(10|15|20|25|30|35|40|45|50)\b", text, re.IGNORECASE)
    if grade_match:
        result["grade"] = "M" + grade_match.group(1)
    exposures = ["Mild", "Moderate", "Severe", "Very Severe", "Marine"]
    for exp in exposures:
        if re.search(exp, text, re.IGNORECASE):
            result["exposure"] = exp
            break
    slump_match = re.search(r"slump\s*(\d+)", text, re.IGNORECASE)
    if slump_match:
        try:
            result["slump"] = int(slump_match.group(1))
        except Exception:
            pass
    cement_types = ["OPC 33", "OPC 43", "OPC 53", "PPC"]
    for ctype in cement_types:
        if re.search(ctype.replace(" ", r"\s*"), text, re.IGNORECASE):
            result["cement"] = ctype
            break
    return result

# =========================
# Helpers
# =========================
@st.cache_data
def _read_csv_try(path):
    return pd.read_csv(path)

@st.cache_data
def load_data(materials_file=None, emissions_file=None):
    materials = None
    emissions = None
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
                st.warning("Materials CSV not found.")
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
                st.warning("Emission factors CSV not found.")
    if emissions is not None:
        if "Material" not in emissions.columns:
            emissions.columns = ["Material","CO2_Factor(kg_CO2_per_kg)"][:len(emissions.columns)]
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
        missing = df.loc[df["CO2_Factor(kg_CO2_per_kg)"].isna(), "Material"].unique().tolist()
        st.warning(f"Missing CO₂ factors for materials: {missing}. Treated as 0 in CO₂ calc.")
        df["CO2_Factor(kg_CO2_per_kg)"] = df["CO2_Factor(kg_CO2_per_kg)"].fillna(0.0)
    df["CO2_Emissions (kg/m3)"] = df["Quantity (kg/m3)"] * df["CO2_Factor(kg_CO2_per_kg)"]
    return df

def aggregate_correction(delta_moisture_pct: float, agg_mass_ssd: float):
    water_delta = (delta_moisture_pct / 100.0) * agg_mass_ssd
    corrected_mass = agg_mass_ssd * (1 + delta_moisture_pct / 100.0)
    return float(water_delta), float(corrected_mass)

def sieve_check_fa(df: pd.DataFrame, zone: str):
    try:
        limits = FINE_AGG_ZONE_LIMITS[zone]
        ok, msgs = True, []
        for sieve, (lo, hi) in limits.items():
            row = df.loc[df["Sieve_mm"].astype(str) == sieve]
            if row.empty:
                ok = False; msgs.append(f"Missing sieve {sieve} mm.")
                continue
            p = float(row["PercentPassing"].iloc[0])
            if not (lo <= p <= hi):
                ok = False; msgs.append(f"{sieve} mm → {p:.1f}% (req {lo}-{hi}%)")
        if ok and not msgs:
            msgs = [f"Fine aggregate meets IS 383 {zone} limits."]
        return ok, msgs
    except Exception:
        return False, ["Invalid fine aggregate CSV format. Expected columns: Sieve_mm, PercentPassing"]

def sieve_check_ca(df: pd.DataFrame, nominal_mm: int):
    try:
        limits = COARSE_LIMITS[int(nominal_mm)]
        ok, msgs = True, []
        for sieve, (lo, hi) in limits.items():
            row = df.loc[df["Sieve_mm"].astype(str) == sieve]
            if row.empty:
                ok = False; msgs.append(f"Missing sieve {sieve} mm.")
                continue
            p = float(row["PercentPassing"].iloc[0])
            if not (lo <= p <= hi):
                ok = False; msgs.append(f"{sieve} mm → {p:.1f}% (req {lo}-{hi}%)")
        if ok and not msgs:
            msgs = [f"Coarse aggregate meets IS 383 ({nominal_mm} mm graded)."]
        return ok, msgs
    except Exception:
        return False, ["Invalid coarse aggregate CSV format. Expected columns: Sieve_mm, PercentPassing"]

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

# =========================
# UI — Sidebar & Inputs
# =========================
st.sidebar.header("📝 Mix Inputs")

# Determine supported grades from datasets (restricted to M10–M50)
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

supported_grades = sorted(set([s.strip().upper() for s in supported_grades if isinstance(s, str)]))
allowed_grades = sorted(GRADE_STRENGTH.keys(), key=lambda x: int(x.lstrip("M")))
supported_grades = [g for g in supported_grades if g in allowed_grades]
if not supported_grades:
    supported_grades = allowed_grades.copy()

# Determine supported cement types
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
if not supported_cements:
    supported_cements = ["OPC 33", "OPC 43", "OPC 53", "PPC"]

# === Parser UI additions ===
st.sidebar.markdown("### Natural Language Input")
user_text = st.sidebar.text_area("Describe your mix in English (optional)", height=100)
use_parser = st.sidebar.checkbox("Use parser to auto-fill inputs", value=False)

# Default UI selectors
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
# Data loading (materials & emissions)
# =========================
materials_df, emissions_df = load_data(materials_file, emissions_file)

if emissions_df is None or emissions_df.shape[0] == 0:
    emissions_df = pd.DataFrame(columns=["Material","CO2_Factor(kg_CO2_per_kg)"])

# =========================
# Dataset Previews
# =========================
st.header("CivilGPT — Sustainable Concrete Mix Designer (v1.7)")
st.markdown("Upload materials/emissions CSV to override defaults. Mix generation will use datasets in repo if available. UI restricted to grades M10–M50. You can optionally describe your desired mix in English and enable the parser to auto-fill inputs.")

with st.expander("📁 Current dataset preview (Lab)"):
    if lab_df is None:
        st.write("No lab dataset found. Expected file:", LAB_FILE)
    else:
        st.write("Showing first 10 rows of the lab dataset:")
        st.dataframe(lab_df.head(10))

with st.expander("📁 Current dataset preview (Mix designs)"):
    if mix_df is None:
        st.write("No mix dataset found. Expected file:", MIX_FILE)
    else:
        st.write("Showing first 10 rows of the mix dataset:")
        st.dataframe(mix_df.head(10))

# NOTE: histogram removed here
# =========================
# Parser override logic
# =========================
def _apply_parser_overrides(parsed: dict, current_inputs: dict):
    messages = []
    updated = current_inputs.copy()
    if not parsed:
        return updated, messages

    if "grade" in parsed:
        parsed_grade = parsed["grade"].upper()
        if parsed_grade in GRADE_STRENGTH:
            updated["grade"] = parsed_grade
            messages.append(f"Parser: set grade → {parsed_grade}")
        else:
            messages.append(f"Parser: grade {parsed_grade} not in allowed range M10–M50; ignored.")

    if "exposure" in parsed:
        exp = parsed["exposure"]
        if exp in EXPOSURE_WB_LIMITS:
            updated["exposure"] = exp
            messages.append(f"Parser: set exposure → {exp}")
        else:
            messages.append(f"Parser: exposure {exp} not recognized; ignored.")

    if "slump" in parsed:
        try:
            s = int(parsed["slump"])
            s_clamped = max(25, min(180, s))
            updated["target_slump"] = s_clamped
            messages.append(f"Parser: set slump → {s_clamped} mm")
        except Exception:
            messages.append("Parser: slump value invalid; ignored.")

    if "cement" in parsed:
        cement = parsed["cement"]
        for c in supported_cements:
            if cement.lower().replace(" ", "") in c.lower().replace(" ", ""):
                updated["cement_choice"] = c
                messages.append(f"Parser: set cement → {c}")
                break
        else:
            if cement in ["OPC 33", "OPC 43", "OPC 53", "PPC"]:
                updated["cement_choice"] = cement
                messages.append(f"Parser: set cement → {cement}")
            else:
                messages.append(f"Parser: cement '{cement}' not recognized among available types; ignored.")

    return updated, messages

# =========================
# Run (Generate button)
# =========================
csv_opt = None
csv_base = None
excel_bytes = None
pdf_bytes = None

if st.button("Generate Sustainable Mix"):
    try:
        current_inputs = {
            "grade": grade,
            "exposure": exposure,
            "cement_choice": cement_choice,
            "nom_max": nom_max,
            "agg_shape": agg_shape,
            "target_slump": target_slump,
            "use_sp": use_sp,
            "sp_reduction": sp_reduction,
            "qc_level": qc_level,
            "air_pct": air_pct,
            "fa_moist": fa_moist,
            "ca_moist": ca_moist,
            "fine_zone": fine_zone
        }

        parsed = {}
        if use_parser and user_text and user_text.strip():
            parsed = simple_parse(user_text)
            updated_inputs, parser_msgs = _apply_parser_overrides(parsed, current_inputs)
            grade = updated_inputs["grade"]
            exposure = updated_inputs["exposure"]
            cement_choice = updated_inputs["cement_choice"]
            nom_max = updated_inputs["nom_max"]
            agg_shape = updated_inputs["agg_shape"]
            target_slump = updated_inputs["target_slump"]
            use_sp = updated_inputs["use_sp"]
            sp_reduction = updated_inputs["sp_reduction"]
            qc_level = updated_inputs["qc_level"]
            air_pct = updated_inputs["air_pct"]
            fa_moist = updated_inputs["fa_moist"]
            ca_moist = updated_inputs["ca_moist"]
            fine_zone = updated_inputs["fine_zone"]

            if parser_msgs:
                for m in parser_msgs:
                    st.info(m)
            if not parsed:
                st.info("Parser did not detect any recognized parameters in your text; using UI inputs.")

        if materials_df is None or emissions_df is None:
            st.error("CSV files missing or invalid. Fix and retry.")
        else:
            min_grade_required = EXPOSURE_MIN_GRADE[exposure]
            grade_order = list(GRADE_STRENGTH.keys())
            parsed_grade = grade
            if parsed_grade not in grade_order:
                parsed_grade = grade_order[0]

            if grade_order.index(parsed_grade) < grade_order.index(min_grade_required):
                st.warning(f"Exposure **{exposure}** requires minimum grade **{min_grade_required}** by IS 456. Proceeding with {min_grade_required}.")
                parsed_grade = min_grade_required

            fck = GRADE_STRENGTH.get(parsed_grade, 30)
            S = QC_STDDEV[qc_level]
            fck_target = fck + 1.65 * S

            opt_df, opt_meta = generate_mix(
                grade=parsed_grade, exposure=exposure, nom_max=nom_max, target_slump=target_slump,
                agg_shape=agg_shape, emissions=emissions_df, cement_choice=cement_choice,
                use_sp=use_sp, sp_reduction=sp_reduction
            )
            base_df, base_meta = generate_baseline(
                grade=parsed_grade, exposure=exposure, nom_max=nom_max, target_slump=target_slump,
                agg_shape=agg_shape, emissions=emissions_df, cement_choice=cement_choice,
                use_sp=use_sp, sp_reduction=sp_reduction
            )

            if opt_df is None or base_df is None:
                st.error("No feasible mix found.")
            else:
                for m in (opt_meta, base_meta):
                    m["fck"] = fck
                    m["fck_target"] = round(fck_target, 1)
                    m["stddev_S"] = S

                st.success(f"Mixes generated for **{parsed_grade}** under **{exposure}** exposure using {cement_choice}.")

                st.subheader("Optimized Sustainable Mix")
                st.dataframe(opt_df, use_container_width=True)

                st.subheader(f"{cement_choice} Baseline Mix")
                st.dataframe(base_df, use_container_width=True)

                co2_opt = float(opt_df["CO2_Emissions (kg/m3)"].sum())
                co2_base = float(base_df["CO2_Emissions (kg/m3)"].sum())
                reduction = (co2_base - co2_opt) / co2_base * 100 if co2_base > 0 else 0.0

                k1, k2, k3 = st.columns(3)
                k1.metric("🌱 Optimized CO₂", f"{co2_opt:.1f} kg/m³")
                k2.metric(f"🏗️ {cement_choice} Baseline CO₂", f"{co2_base:.1f} kg/m³")
                k3.metric("📉 Reduction", f"{reduction:.1f}%")

                st.markdown("### ✅ Assumptions, Strength & Compliance")
                opt_checks, opt_derived = compliance_checks(opt_df, opt_meta, exposure)
                base_checks, base_derived = compliance_checks(base_df, base_meta, exposure)

                with st.expander("Optimized Mix — Details", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.json(opt_derived)
                        st.caption(f"Entrapped air assumed: {air_pct:.1f} %")
                    with c2:
                        st.table(compliance_table(opt_checks))

                with st.expander(f"{cement_choice} Baseline — Details", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.json(base_derived)
                        st.caption(f"Entrapped air assumed: {air_pct:.1f} %")
                    with c2:
                        st.table(compliance_table(base_checks))

                st.markdown("### IS 383 Sieve Compliance")
                if fine_csv is not None:
                    try:
                        df_fine = pd.read_csv(fine_csv)
                        ok_fa, msgs_fa = sieve_check_fa(df_fine, fine_zone)
                        for m in msgs_fa:
                            st.write(("✅ " if ok_fa else "❌ ") + m)
                    except Exception as e:
                        st.warning(f"Could not read fine sieve CSV: {e}")
                if coarse_csv is not None:
                    try:
                        df_coarse = pd.read_csv(coarse_csv)
                        ok_ca, msgs_ca = sieve_check_ca(df_coarse, nom_max)
                        for m in msgs_ca:
                            st.write(("✅ " if ok_ca else "❌ ") + m)
                    except Exception as e:
                        st.warning(f"Could not read coarse sieve CSV: {e}")

                st.markdown("### 📊 CO₂ Comparison")
                fig, ax = plt.subplots()
                bars = ax.bar(["Optimized", f"{cement_choice} Baseline"], [co2_opt, co2_base])
                ax.set_ylabel("CO₂ Emissions (kg/m³)")
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f"{height:.1f}",
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=9)
                st.pyplot(fig)

                csv_opt = opt_df.to_csv(index=False).encode("utf-8")
                csv_base = base_df.to_csv(index=False).encode("utf-8")

                try:
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        opt_df.to_excel(writer, sheet_name="Optimized Mix", index=False)
                        base_df.to_excel(writer, sheet_name="Baseline Mix", index=False)
                    excel_bytes = buffer.getvalue()
                except Exception:
                    excel_bytes = None

                try:
                    pdf_buffer = BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer)
                    styles = getSampleStyleSheet()
                    story = []

                    story.append(Paragraph("CivilGPT Sustainable Mix Report", styles["Title"]))
                    story.append(Spacer(1, 8))
                    story.append(Paragraph(f"Grade: {parsed_grade} | Exposure: {exposure} | Cement: {cement_choice}", styles["Normal"]))
                    story.append(Paragraph(f"Target mean strength: {round(fck_target,1)} MPa", styles["Normal"]))
                    story.append(Spacer(1, 8))

                    data_summary = [
                        ["Metric", "Optimized", "Baseline"],
                        ["CO₂ (kg/m³)", f"{co2_opt:.1f}", f"{co2_base:.1f}"],
                        ["Reduction (%)", f"{reduction:.1f}", "-"]
                    ]
                    tbl = Table(data_summary, hAlign="LEFT")
                    tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F0F0F0")),
                        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                    ]))
                    story.append(tbl)
                    story.append(Spacer(1, 8))
                    doc.build(story)
                    pdf_bytes = pdf_buffer.getvalue()
                except Exception:
                    pdf_bytes = None

                with st.expander("📥 Downloads", expanded=True):
                    st.download_button("📥 Download Optimized Mix (CSV)", data=csv_opt,
                                       file_name=f"CivilGPT_{parsed_grade}_optimized.csv", mime="text/csv")
                    st.download_button("📥 Download Baseline Mix (CSV)", data=csv_base,
                                       file_name=f"CivilGPT_{parsed_grade}_{cement_choice}_baseline.csv", mime="text/csv")
                    if excel_bytes is not None:
                        st.download_button("📊 Download Report (Excel)", data=excel_bytes,
                                           file_name=f"CivilGPT_{parsed_grade}_Report.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    if pdf_bytes is not None:
                        st.download_button("📄 Download Report (PDF)", data=pdf_bytes,
                                           file_name=f"CivilGPT_{parsed_grade}_Report.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.text(traceback.format_exc())

else:
    st.info("Set parameters and click **Generate Sustainable Mix**.")

st.markdown("---")
st.caption("CivilGPT v1.7 (cleaned) | Histogram removed")
