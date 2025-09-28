import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
import json
import traceback
import re

from groq import Groq
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="CivilGPT - Sustainable Concrete Mix Designer v1.8",
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

# IS 383 limits
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
# Moisture Corrections
# =========================
def aggregate_correction(delta_moisture_pct: float, agg_mass_ssd: float):
    water_delta = (delta_moisture_pct / 100.0) * agg_mass_ssd
    corrected_mass = agg_mass_ssd * (1 + delta_moisture_pct / 100.0)
    return float(water_delta), float(corrected_mass)

# =========================
# Parsers (Regex + Groq LLM)
# =========================
def simple_parse(text: str) -> dict:
    """Regex-based parser for grade, exposure, slump, cement."""
    result = {}
    grade_match = re.search(r"\bM(10|15|20|25|30|35|40|45|50)\b", text, re.IGNORECASE)
    if grade_match:
        result["grade"] = "M" + grade_match.group(1)
    for exp in ["Mild", "Moderate", "Severe", "Very Severe", "Marine"]:
        if re.search(exp, text, re.IGNORECASE):
            result["exposure"] = exp
            break
    slump_match = re.search(r"slump\s*(\d+)", text, re.IGNORECASE)
    if slump_match:
        try:
            result["slump"] = int(slump_match.group(1))
        except:
            pass
    for ctype in ["OPC 33", "OPC 43", "OPC 53", "PPC"]:
        if re.search(ctype.replace(" ", r"\s*"), text, re.IGNORECASE):
            result["cement"] = ctype
            break
    return result

def parse_input_with_llm(user_text: str) -> dict:
    """Groq LLM parser with regex fallback."""
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Extract concrete mix design parameters as JSON."},
                {"role": "user", "content": user_text}
            ],
            response_format={"type": "json_schema", "json_schema": {
                "name": "mix_parser",
                "schema": {
                    "type": "object",
                    "properties": {
                        "grade": {"type": "string"},
                        "exposure": {"type": "string"},
                        "slump": {"type": "integer"},
                        "cement": {"type": "string"},
                        "nom_max": {"type": "integer"}
                    },
                    "required": ["grade"]
                }
            }}
        )
        return resp.choices[0].message.parsed
    except Exception as e:
        st.warning(f"LLM parser failed: {e}. Falling back to regex parser.")
        return simple_parse(user_text)

# =========================
# CSV Loaders (Materials, Emissions, Costs)
# =========================
@st.cache_data
def load_data(materials_file=None, emissions_file=None, cost_file=None):
    def _try_read_csv(path): 
        return pd.read_csv(path)
    materials, emissions, costs = None, None, None

    if materials_file: 
        try: materials = pd.read_csv(materials_file)
        except: pass
    if materials is None:
        for p in ["materials_library.csv", "data/materials_library.csv"]:
            if os.path.exists(p): 
                materials = _try_read_csv(p)

    if emissions_file: 
        try: emissions = pd.read_csv(emissions_file)
        except: pass
    if emissions is None:
        for p in ["emission_factors.csv", "data/emission_factors.csv"]:
            if os.path.exists(p): 
                emissions = _try_read_csv(p)

    if cost_file: 
        try: costs = pd.read_csv(cost_file)
        except: pass
    if costs is None:
        for p in ["cost_factors.csv", "data/cost_factors.csv"]:
            if os.path.exists(p): 
                costs = _try_read_csv(p)

    if emissions is None:
        emissions = pd.DataFrame(columns=["Material","CO2_Factor(kg_CO2_per_kg)"])
    if costs is None:
        costs = pd.DataFrame(columns=["Material","Cost_per_kg"])
    if materials is None:
        materials = pd.DataFrame(columns=["Material"])
    return materials, emissions, costs
# =========================
# Mix Evaluation Helpers
# =========================
def evaluate_mix(components_dict, emissions_df, cost_df):
    """Compute CO₂ and cost for each component mix."""
    comp_df = pd.DataFrame(list(components_dict.items()), columns=["Material", "Quantity (kg/m3)"])

    # Merge CO₂ factors
    df = comp_df.merge(emissions_df, on="Material", how="left")
    if "CO2_Factor(kg_CO2_per_kg)" not in df.columns:
        df["CO2_Factor(kg_CO2_per_kg)"] = 0.0
    df["CO2_Factor(kg_CO2_per_kg)"] = df["CO2_Factor(kg_CO2_per_kg)"].fillna(0.0)
    df["CO2_Emissions (kg/m3)"] = df["Quantity (kg/m3)"] * df["CO2_Factor(kg_CO2_per_kg)"]

    # Merge cost
    df = df.merge(cost_df, on="Material", how="left")
    if "Cost_per_kg" not in df.columns:
        df["Cost_per_kg"] = 0.0
    df["Cost_per_kg"] = df["Cost_per_kg"].fillna(0.0)
    df["Cost (₹/m3)"] = df["Quantity (kg/m3)"] * df["Cost_per_kg"]

    return df

# =========================
# Aggregate Volume Balance
# =========================
def compute_aggregates(binder, water, sp, density_target=2400, fine_fraction=0.40):
    """
    Split aggregates by volume method instead of fixed values.
    """
    nonagg_mass = binder + water + sp
    agg_mass = density_target - nonagg_mass
    fine = agg_mass * fine_fraction
    coarse = agg_mass * (1 - fine_fraction)
    return fine, coarse

# =========================
# Sieve Compliance (IS 383)
# =========================
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
        return False, ["Invalid fine aggregate CSV format. Expected: Sieve_mm, PercentPassing"]

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
        return False, ["Invalid coarse aggregate CSV format. Expected: Sieve_mm, PercentPassing"]

# =========================
# Compliance Checks
# =========================
def compliance_checks(mix_df, meta, exposure):
    checks = {}
    try:
        checks["W/B ≤ exposure limit"] = float(meta["w_b"]) <= EXPOSURE_WB_LIMITS[exposure]
    except:
        checks["W/B ≤ exposure limit"] = False
    try:
        checks["Min cementitious met"] = float(meta["cementitious"]) >= float(EXPOSURE_MIN_CEMENT[exposure])
    except:
        checks["Min cementitious met"] = False
    checks["SCM ≤ 50%"] = float(meta.get("scm_total_frac", 0.0)) <= 0.50
    total_mass = float(mix_df["Quantity (kg/m3)"].sum())
    checks["Unit weight 2200–2600 kg/m³"] = 2200.0 <= total_mass <= 2600.0
    return checks

def compliance_table(checks: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(checks.items()), columns=["Check", "Status"])
    df["Result"] = df["Status"].apply(lambda x: "✅ Pass" if x else "❌ Fail")
    return df[["Check", "Result"]]
# =========================
# Mix Generators with CO₂ + Cost + Trace
# =========================
def generate_mix(grade, exposure, nom_max, target_slump, agg_shape,
                 emissions, costs, cement_choice, use_sp=True, sp_reduction=0.18,
                 optimize_cost=False, fine_fraction=0.40):

    w_b_limit = float(EXPOSURE_WB_LIMITS[exposure])
    min_cem = float(EXPOSURE_MIN_CEMENT[exposure])
    target_water = WATER_BASELINE.get(int(nom_max), 186.0)

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
                fine, coarse = compute_aggregates(binder, target_water, sp, fine_fraction=fine_fraction)

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

                score = co2_total
                if optimize_cost:
                    score = co2_total + 0.001 * cost_total  # cost influence

                trace.append({
                    "wb": wb, "flyash_frac": flyash_frac, "ggbs_frac": ggbs_frac,
                    "co2": co2_total, "cost": cost_total, "score": score
                })

                if score < best_score:
                    best_df = df.copy()
                    best_score = score
                    best_meta = {
                        "w_b": wb, "cementitious": binder, "cement": cement,
                        "flyash": flyash, "ggbs": ggbs, "water": target_water,
                        "sp": sp, "fine": fine, "coarse": coarse,
                        "scm_total_frac": flyash_frac + ggbs_frac,
                        "grade": grade, "exposure": exposure,
                        "nom_max": nom_max, "slump": target_slump,
                        "co2_total": co2_total, "cost_total": cost_total
                    }

    return best_df, best_meta, trace

def generate_baseline(grade, exposure, nom_max, target_slump, agg_shape,
                      emissions, costs, cement_choice, use_sp=True, sp_reduction=0.18,
                      fine_fraction=0.40):

    w_b_limit = float(EXPOSURE_WB_LIMITS[exposure])
    min_cem = float(EXPOSURE_MIN_CEMENT[exposure])
    water_target = WATER_BASELINE.get(int(nom_max), 186.0)
    cementitious = max(water_target / w_b_limit, min_cem)
    sp = 2.5 if use_sp else 0.0
    fine, coarse = compute_aggregates(cementitious, water_target, sp, fine_fraction=fine_fraction)

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
        "flyash": 0.0, "ggbs": 0.0, "water": water_target,
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

# Basic selectors
grade = st.sidebar.selectbox("Concrete Grade", list(GRADE_STRENGTH.keys()), index=2)
exposure = st.sidebar.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()), index=2)
cement_choice = st.sidebar.selectbox("Cement Type", ["OPC 33", "OPC 43", "OPC 53", "PPC"], index=2)

nom_max = st.sidebar.selectbox("Nominal max aggregate (mm)", [10, 12.5, 20, 40], index=2)
agg_shape = st.sidebar.selectbox("Aggregate shape", list(AGG_SHAPE_WATER_ADJ.keys()), index=0)
target_slump = st.sidebar.slider("Target slump (mm)", 25, 180, 100, step=5)

# Superplasticizer + cost toggle
use_sp = st.sidebar.checkbox("Use Superplasticizer (PCE)", True)
optimize_cost = st.sidebar.checkbox("Optimize for CO₂ + Cost", False)

# Fine fraction
fine_fraction = st.sidebar.slider("Fine aggregate fraction", 0.3, 0.5, 0.40, step=0.01)

# QC & moisture
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
# Parser Override Logic
# =========================
def apply_parser(user_text, current_inputs):
    if not user_text.strip():
        return current_inputs, []
    try:
        if use_llm_parser:
            parsed = parse_input_with_llm(user_text)
        else:
            parsed = simple_parse(user_text)
    except Exception as e:
        st.warning(f"Parser error: {e}, falling back to regex")
        parsed = simple_parse(user_text)

    messages = []
    updated = current_inputs.copy()

    if "grade" in parsed and parsed["grade"] in GRADE_STRENGTH:
        updated["grade"] = parsed["grade"]; messages.append(f"Parser set grade → {parsed['grade']}")
    if "exposure" in parsed and parsed["exposure"] in EXPOSURE_WB_LIMITS:
        updated["exposure"] = parsed["exposure"]; messages.append(f"Parser set exposure → {parsed['exposure']}")
    if "slump" in parsed:
        s = int(parsed["slump"]); s = max(25, min(180, s))
        updated["target_slump"] = s; messages.append(f"Parser set slump → {s} mm")
    if "cement" in parsed:
        updated["cement_choice"] = parsed["cement"]; messages.append(f"Parser set cement → {parsed['cement']}")
    if "nom_max" in parsed and parsed["nom_max"] in [10, 12.5, 20, 40]:
        updated["nom_max"] = parsed["nom_max"]; messages.append(f"Parser set aggregate size → {parsed['nom_max']} mm")

    return updated, messages

# =========================
# Main Run Button
# =========================
st.header("CivilGPT — Sustainable Concrete Mix Designer (v1.8)")
st.markdown("This version integrates **Groq LLM parsing, cost optimization, aggregate volume balance, and optimizer trace**, while preserving all v1.7 features (sieve checks, moisture corrections, reports).")

if st.button("Generate Sustainable Mix (v1.8)"):
    try:
        inputs = {
            "grade": grade, "exposure": exposure, "cement_choice": cement_choice,
            "nom_max": nom_max, "agg_shape": agg_shape, "target_slump": target_slump,
            "use_sp": use_sp, "optimize_cost": optimize_cost, "fine_fraction": fine_fraction
        }

        inputs, msgs = apply_parser(user_text, inputs)
        for m in msgs: st.info(m)

        grade = inputs["grade"]; exposure = inputs["exposure"]
        cement_choice = inputs["cement_choice"]; nom_max = inputs["nom_max"]
        agg_shape = inputs["agg_shape"]; target_slump = inputs["target_slump"]
        use_sp = inputs["use_sp"]; optimize_cost = inputs["optimize_cost"]
        fine_fraction = inputs["fine_fraction"]

        fck = GRADE_STRENGTH[grade]
        S = QC_STDDEV[qc_level]
        fck_target = fck + 1.65 * S

        # Generate mixes
        opt_df, opt_meta, trace = generate_mix(
            grade, exposure, nom_max, target_slump, agg_shape,
            emissions_df, costs_df, cement_choice,
            use_sp=use_sp, optimize_cost=optimize_cost,
            fine_fraction=fine_fraction
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
                m["fck"] = fck
                m["fck_target"] = round(fck_target, 1)
                m["stddev_S"] = S

            st.success(f"Mixes generated for **{grade}** under **{exposure}** exposure using {cement_choice}.")

            # Display mixes
            st.subheader("Optimized Sustainable Mix")
            st.dataframe(opt_df, use_container_width=True)

            st.subheader("Baseline Mix")
            st.dataframe(base_df, use_container_width=True)

            co2_opt, cost_opt = opt_meta["co2_total"], opt_meta["cost_total"]
            co2_base, cost_base = base_meta["co2_total"], base_meta["cost_total"]

            reduction = (co2_base - co2_opt) / co2_base * 100 if co2_base > 0 else 0.0
            cost_diff = (cost_opt - cost_base)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🌱 Optimized CO₂", f"{co2_opt:.1f} kg/m³")
            k2.metric("🏗️ Baseline CO₂", f"{co2_base:.1f} kg/m³")
            k3.metric("📉 CO₂ Reduction", f"{reduction:.1f}%")
            k4.metric("💰 Cost Difference", f"{cost_diff:+.2f} ₹/m³")

            # Compliance
            st.markdown("### ✅ Compliance Checks")
            st.table(compliance_table(compliance_checks(opt_df, opt_meta, exposure)))

            # Optimizer trace
            st.markdown("### 🔍 Optimizer Trace (Top 5 candidates)")
            trace_df = pd.DataFrame(trace).sort_values("score").head(5)
            st.dataframe(trace_df, use_container_width=True)

            # Moisture corrections
            st.markdown("### 💧 Moisture Corrections")
            try:
                fa_free_w, _ = aggregate_correction(fa_moist - fa_abs, opt_meta["fine"])
                ca_free_w, _ = aggregate_correction(ca_moist - ca_abs, opt_meta["coarse"])
                st.write(f"Free water adjustment (Optimized): {fa_free_w + ca_free_w:.1f} kg/m³")
            except Exception:
                st.write("Free water adjustment: N/A")

            # Sieve checks
            st.markdown("### IS 383 Sieve Compliance")
            if fine_csv is not None:
                try:
                    df_fine = pd.read_csv(fine_csv)
                    ok_fa, msgs_fa = sieve_check_fa(df_fine, fine_zone)
                    for m in msgs_fa:
                        st.write(("✅ " if ok_fa else "❌ ") + m)
                except Exception as e:
                    st.warning(f"Could not read fine sieve CSV: {e}")
            else:
                st.info("Fine sieve CSV not provided — skipping check.")

            if coarse_csv is not None:
                try:
                    df_coarse = pd.read_csv(coarse_csv)
                    ok_ca, msgs_ca = sieve_check_ca(df_coarse, nom_max)
                    for m in msgs_ca:
                        st.write(("✅ " if ok_ca else "❌ ") + m)
                except Exception as e:
                    st.warning(f"Could not read coarse sieve CSV: {e}")
            else:
                st.info("Coarse sieve CSV not provided — skipping check.")

            # CO₂ plot
            st.markdown("### 📊 CO₂ Comparison")
            fig, ax = plt.subplots()
            ax.bar(["Optimized", "Baseline"], [co2_opt, co2_base])
            ax.set_ylabel("CO₂ Emissions (kg/m³)")
            st.pyplot(fig)

            # Prepare downloads
            csv_opt = opt_df.to_csv(index=False).encode("utf-8")
            csv_base = base_df.to_csv(index=False).encode("utf-8")

            # Excel workbook
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                opt_df.to_excel(writer, sheet_name="Optimized Mix", index=False)
                base_df.to_excel(writer, sheet_name="Baseline Mix", index=False)
            excel_bytes = buffer.getvalue()

            # PDF report
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

            # Downloads
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
    st.info("Set parameters and click **Generate Sustainable Mix (v1.8)**.")

st.markdown("---")
st.caption("CivilGPT v1.8 | Full merged: Groq LLM parser + Cost optimization + Volume balance + Trace + IS 383 + Moisture + Reports")
