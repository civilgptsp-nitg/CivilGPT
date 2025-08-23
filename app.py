import streamlit as st
import pandas as pd
import numpy as np

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="CivilGPT - Sustainable Concrete Mix Designer",
    page_icon="🧱",
    layout="wide"
)

# =========================
# IS-style Rules & Tables
# =========================
# IS 456 durability-style caps (representative values)
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

# Characteristic strengths (MPa)
GRADE_STRENGTH = {"M20": 20, "M25": 25, "M30": 30, "M35": 35, "M40": 40}

# IS 10262-style baseline mixing water (kg/m³)
WATER_BASELINE = {10: 200, 12.5: 195, 20: 186, 40: 165}

def water_for_slump(nom_max_mm: int, slump_mm: int, uses_sp: bool=False, sp_reduction_frac: float=0.0) -> float:
    base = WATER_BASELINE.get(int(nom_max_mm), 186)
    if slump_mm <= 50:
        water = base
    else:
        extra_25 = max(0, (slump_mm - 50) / 25.0)
        water = base * (1 + 0.03 * extra_25)
    if uses_sp and sp_reduction_frac > 0:
        water *= (1 - sp_reduction_frac)
    return float(water)

def aggregate_correction(delta_moisture_pct: float, agg_mass_ssd: float):
    water_delta = (delta_moisture_pct / 100.0) * agg_mass_ssd
    corrected_mass = agg_mass_ssd * (1 + delta_moisture_pct / 100.0)
    return float(water_delta), float(corrected_mass)

# IS 383: Fine aggregate grading limits (Zones I–IV)
FINE_AGG_ZONE_LIMITS = {
    "Zone I":  {"10.0": (100,100),"4.75": (90,100),"2.36": (60,95),"1.18": (30,70),"0.600": (15,34),"0.300": (5,20),"0.150": (0,10)},
    "Zone II": {"10.0": (100,100),"4.75": (90,100),"2.36": (75,100),"1.18": (55,90),"0.600": (35,59),"0.300": (8,30),"0.150": (0,10)},
    "Zone III":{"10.0": (100,100),"4.75": (90,100),"2.36": (85,100),"1.18": (75,100),"0.600": (60,79),"0.300": (12,40),"0.150": (0,10)},
    "Zone IV": {"10.0": (100,100),"4.75": (95,100),"2.36": (95,100),"1.18": (90,100),"0.600": (80,100),"0.300": (15,50),"0.150": (0,15)},
}
# IS 383: Coarse aggregate graded (20 mm nominal)
COARSE_20MM_LIMITS = {"40.0": (95,100), "20.0": (95,100), "10.0": (25,55), "4.75": (0,10)}

# =========================
# Data Loading
# =========================
@st.cache_data
def load_data():
    try:
        materials = pd.read_csv("materials_library.csv")
        emissions = pd.read_csv("emission_factors.csv")
        return materials, emissions
    except Exception as e:
        st.error(f"Error loading CSVs: {e}")
        return None, None

materials_df, emissions_df = load_data()

# =========================
# Core Mix Helpers
# =========================
def evaluate_mix(components_dict, emissions_df):
    df = pd.DataFrame(list(components_dict.items()), columns=["Material", "Quantity (kg/m3)"])
    df = df.merge(emissions_df, on="Material", how="left")
    df["CO2_Emissions (kg/m3)"] = df["Quantity (kg/m3)"] * df["CO2_Factor(kg_CO2_per_kg)"]
    return df

def generate_mix(grade, exposure, nom_max, target_slump, emissions,
                 cement_choice="OPC 43", use_sp=True, sp_reduction=0.18):
    w_b_limit = float(EXPOSURE_WB_LIMITS[exposure])
    min_cem = float(EXPOSURE_MIN_CEMENT[exposure])
    target_water = water_for_slump(nom_max, int(target_slump), use_sp, sp_reduction)

    best_df, best_meta, best_co2 = None, None, float("inf")
    for wb in np.linspace(0.35, w_b_limit, 6):
        for flyash_frac in [0.0, 0.2, 0.3]:
            for ggbs_frac in [0.0, 0.3, 0.5]:
                if flyash_frac + ggbs_frac > 0.50:
                    continue
                cementitious = max(target_water / wb, min_cem)
                cement = cementitious * (1 - flyash_frac - ggbs_frac)
                flyash = cementitious * flyash_frac
                ggbs = cementitious * ggbs_frac
                mix = {
                    cement_choice: cement,
                    "Fly Ash": flyash,
                    "GGBS": ggbs,
                    "Water": target_water,
                    "PCE Superplasticizer": 2.5 if use_sp else 0.0,
                    "M-Sand": 650.0,
                    "20mm Coarse Aggregate": 1150.0,
                }
                df = evaluate_mix(mix, emissions)
                total_co2 = float(df["CO2_Emissions (kg/m3)"].sum())
                if total_co2 < best_co2:
                    best_df, best_co2 = df.copy(), total_co2
                    best_meta = {
                        "w_b": float(wb), "cementitious": float(cementitious),
                        "cement": float(cement), "flyash": float(flyash), "ggbs": float(ggbs),
                        "water_target": float(target_water), "sp": float(mix["PCE Superplasticizer"]),
                        "fine": 650.0, "coarse": 1150.0,
                        "scm_total_frac": float(flyash_frac + ggbs_frac),
                        "grade": grade, "exposure": exposure,
                        "nom_max": nom_max, "slump": int(target_slump),
                        "cement_choice": cement_choice
                    }
    return best_df, best_meta

def generate_baseline(grade, exposure, nom_max, target_slump, emissions,
                      cement_choice="OPC 43", use_sp=True, sp_reduction=0.18):
    w_b_limit = float(EXPOSURE_WB_LIMITS[exposure])
    min_cem = float(EXPOSURE_MIN_CEMENT[exposure])
    water_target = water_for_slump(nom_max, int(target_slump), use_sp, sp_reduction)
    cementitious = max(water_target / w_b_limit, min_cem)
    mix = {
        cement_choice: cementitious,
        "Fly Ash": 0.0, "GGBS": 0.0,
        "Water": water_target,
        "PCE Superplasticizer": 2.5 if use_sp else 0.0,
        "M-Sand": 650.0,
        "20mm Coarse Aggregate": 1150.0,
    }
    df = evaluate_mix(mix, emissions)
    meta = {
        "w_b": float(w_b_limit), "cementitious": float(cementitious), "cement": float(cementitious),
        "flyash": 0.0, "ggbs": 0.0,
        "water_target": float(water_target), "sp": float(mix["PCE Superplasticizer"]),
        "fine": 650.0, "coarse": 1150.0,
        "scm_total_frac": 0.0,
        "grade": grade, "exposure": exposure, "nom_max": nom_max, "slump": int(target_slump),
        "cement_choice": cement_choice
    }
    return df, meta

def compliance_checks(mix_df, meta, exposure):
    checks = {}
    checks["W/B ≤ exposure limit"] = meta["w_b"] <= EXPOSURE_WB_LIMITS[exposure]
    checks["Min cementitious met"] = meta["cementitious"] >= EXPOSURE_MIN_CEMENT[exposure]
    checks["SCM ≤ 50%"] = meta.get("scm_total_frac", 0.0) <= 0.50
    total_mass = float(mix_df["Quantity (kg/m3)"].sum())
    checks["Unit weight 2200–2600 kg/m³"] = 2200.0 <= total_mass <= 2600.0
    derived = {
        "w/b used": round(meta["w_b"], 3),
        "cementitious (kg/m³)": round(meta["cementitious"], 1),
        "SCM % of cementitious": round(100 * meta.get("scm_total_frac", 0.0), 1),
        "total mass (kg/m³)": round(total_mass, 1),
        "water target (kg/m³)": round(meta.get("water_target", 0.0), 1),
        "cement (kg/m³)": round(meta["cement"], 1),
        "fly ash (kg/m³)": round(meta.get("flyash", 0.0), 1),
        "GGBS (kg/m³)": round(meta.get("ggbs", 0.0), 1),
        "fine agg (kg/m³)": round(meta["fine"], 1),
        "coarse agg (kg/m³)": round(meta["coarse"], 1),
        "SP (kg/m³)": round(meta["sp"], 2),
    }
    return checks, derived

def sieve_check_fa(df: pd.DataFrame, zone: str):
    limits = FINE_AGG_ZONE_LIMITS[zone]; ok, msgs = True, []
    for sieve, (lo, hi) in limits.items():
        row = df.loc[df["Sieve_mm"].astype(str) == sieve]
        if row.empty:
            msgs.append(f"Missing sieve {sieve} mm."); ok = False; continue
        p = float(row["PercentPassing"].iloc[0])
        if not (lo <= p <= hi):
            ok = False; msgs.append(f"{sieve} mm → {p:.1f}% (req {lo}-{hi}%)")
    if ok and not msgs: msgs = [f"Fine aggregate meets IS 383 {zone}."]
    return ok, msgs

def sieve_check_ca20(df: pd.DataFrame):
    ok, msgs = True, []
    for sieve, (lo, hi) in COARSE_20MM_LIMITS.items():
        row = df.loc[df["Sieve_mm"].astype(str) == sieve]
        if row.empty:
            msgs.append(f"Missing sieve {sieve} mm."); ok = False; continue
        p = float(row["PercentPassing"].iloc[0])
        if not (lo <= p <= hi):
            ok = False; msgs.append(f"{sieve} mm → {p:.1f}% (req {lo}-{hi}%)")
    if ok and not msgs: msgs = ["Coarse aggregate meets IS 383 (20 mm graded)."]
    return ok, msgs

# =========================
# UI
# =========================
st.title("🌍 CivilGPT")
st.subheader("AI-Powered Sustainable Concrete Mix Designer (IS-aware)")

st.markdown(
    "Generates **eco-optimized, IS-style concrete mix designs** and compares against **OPC/PPC baselines** with CO₂ footprint and compliance checks."
)

# ---- Sidebar Inputs
st.sidebar.header("📝 Mix Inputs")
grade = st.sidebar.selectbox("Concrete Grade", list(GRADE_STRENGTH.keys()), index=2)
exposure = st.sidebar.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()), index=2)
cement_choice = st.sidebar.selectbox("Cement Type / Grade", ["OPC 33", "OPC 43", "OPC 53", "PPC"], index=1)

st.sidebar.markdown("### Workability & Aggregates")
nom_max = st.sidebar.selectbox("Nominal max aggregate (mm)", [10, 12.5, 20, 40], index=2)
target_slump = st.sidebar.slider("Target slump (mm)", 25, 180, 100, step=5)
use_sp = st.sidebar.checkbox("Use Superplasticizer (PCE)", True)
sp_reduction = st.sidebar.slider("SP water reduction (fraction)", 0.00, 0.30, 0.18, step=0.01)

st.sidebar.markdown("### Air & Moisture (for reporting)")
air_pct = st.sidebar.number_input("Entrapped air (%)", 1.0, 3.0, 2.0, step=0.5)
fa_moist = st.sidebar.number_input("Fine agg moisture (%)", 0.0, 10.0, 0.0, step=0.1)
ca_moist = st.sidebar.number_input("Coarse agg moisture (%)", 0.0, 5.0, 0.0, step=0.1)
fa_abs, ca_abs = 1.0, 0.5

st.sidebar.markdown("---")
st.sidebar.markdown("#### IS 383 Sieve (optional uploads)")
fine_zone = st.sidebar.selectbox("Fine agg zone (IS 383)", ["Zone I","Zone II","Zone III","Zone IV"], index=1)
fine_csv = st.sidebar.file_uploader("Fine sieve CSV (Sieve_mm,PercentPassing)", type=["csv"], key="fine_csv")
coarse_csv = st.sidebar.file_uploader("Coarse sieve CSV 20 mm (Sieve_mm,PercentPassing)", type=["csv"], key="coarse_csv")

# =========================
# Run
# =========================
if st.button("Generate Sustainable Mix"):
    if materials_df is None or emissions_df is None:
        st.error("CSV files missing in repo. Ensure materials_library.csv and emission_factors.csv exist.")
    else:
        opt_df, opt_meta = generate_mix(
            grade=grade, exposure=exposure, nom_max=nom_max, target_slump=target_slump,
            emissions=emissions_df, cement_choice=cement_choice, use_sp=use_sp, sp_reduction=sp_reduction
        )
        base_df, base_meta = generate_baseline(
            grade=grade, exposure=exposure, nom_max=nom_max, target_slump=target_slump,
            emissions=emissions_df, cement_choice=cement_choice, use_sp=use_sp, sp_reduction=sp_reduction
        )

        if opt_df is None or base_df is None:
            st.error("No feasible mix found under given constraints.")
        else:
            st.success(f"Mixes generated for {grade} with {cement_choice} under {exposure} exposure.")

            st.subheader("Optimized Sustainable Mix")
            st.dataframe(opt_df, use_container_width=True)

            st.subheader(f"{cement_choice} Baseline Mix")
            st.dataframe(base_df, use_container_width=True)

            # KPIs
            co2_opt = float(opt_df["CO2_Emissions (kg/m3)"].sum())
            co2_base = float(base_df["CO2_Emissions (kg/m3)"].sum())
            reduction = (co2_base - co2_opt) / co2_base * 100 if co2_base > 0 else 0.0

            k1, k2, k3 = st.columns(3)
            k1.metric("🌱 Optimized CO₂", f"{co2_opt:.1f} kg/m³")
            k2.metric(f"🏗️ {cement_choice} Baseline CO₂", f"{co2_base:.1f} kg/m³")
            k3.metric("📉 Reduction", f"{reduction:.1f}%")

            # Compliance & Assumptions
            st.markdown("### ✅ Assumptions & Compliance")
            opt_checks, opt_derived = compliance_checks(opt_df, opt_meta, exposure)
            base_checks, base_derived = compliance_checks(base_df, base_meta, exposure)

            with st.expander("Optimized Mix — Assumptions & Checks", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Derived Parameters**")
                    st.json(opt_derived)
                    st.caption(f"Entrapped air assumed: {air_pct:.1f} %")
                    fa_free_w, _ = aggregate_correction(fa_moist - fa_abs, opt_meta["fine"])
                    ca_free_w, _ = aggregate_correction(ca_moist - ca_abs, opt_meta["coarse"])
                    st.write(f"Free water from aggregates (report): {fa_free_w + ca_free_w:.1f} kg/m³")
                with c2:
                    st.markdown("**Compliance**")
                    for k, v in opt_checks.items():
                        st.write(f"• {k}: {'✅' if v else '❌'}")

            with st.expander(f"{cement_choice} Baseline — Assumptions & Checks", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Derived Parameters**")
                    st.json(base_derived)
                    st.caption(f"Entrapped air assumed: {air_pct:.1f} %")
                    fa_free_w_b, _ = aggregate_correction(fa_moist - fa_abs, base_meta["fine"])
                    ca_free_w
