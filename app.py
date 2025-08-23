import streamlit as st
import pandas as pd
import numpy as np

# Import IS rules & helpers
from is_rules import (
    EXPOSURE_WB_LIMITS, EXPOSURE_MIN_CEMENT, GRADE_STRENGTH,
    water_for_slump, aggregate_correction,
    sieve_check_fa, sieve_check_ca20
)

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="CivilGPT - Sustainable Concrete Mix Designer",
    page_icon="🧱",
    layout="wide"
)

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
    """components_dict -> {'Material': qty_kgm3, ...} -> returns dataframe with CO2."""
    df = pd.DataFrame(list(components_dict.items()), columns=["Material", "Quantity (kg/m3)"])
    df = df.merge(emissions_df, on="Material", how="left")
    df["CO2_Emissions (kg/m3)"] = df["Quantity (kg/m3)"] * df["CO2_Factor(kg_CO2_per_kg)"]
    return df

def generate_mix(grade, exposure, nom_max, target_slump, emissions, use_sp=True, sp_reduction=0.18):
    """
    Rule-based optimizer (grid) using IS-style w/b caps and IS 10262 water target.
    Returns (mix_df, meta)
    """
    w_b_limit = float(EXPOSURE_WB_LIMITS[exposure])
    min_cem = float(EXPOSURE_MIN_CEMENT[exposure])

    # IS 10262 water target based on slump & SP
    target_water = water_for_slump(nom_max_mm=nom_max, slump_mm=int(target_slump), uses_sp=use_sp, sp_reduction_frac=sp_reduction)

    best_df, best_meta, best_co2 = None, None, float("inf")

    for wb in np.linspace(0.35, w_b_limit, 6):          # candidate w/b ratios
        for flyash_frac in [0.0, 0.2, 0.3]:
            for ggbs_frac in [0.0, 0.3, 0.5]:
                if flyash_frac + ggbs_frac > 0.50:
                    continue

                cementitious = max(target_water / wb, min_cem)   # kg/m³
                cement = cementitious * (1 - flyash_frac - ggbs_frac)
                flyash = cementitious * flyash_frac
                ggbs = cementitious * ggbs_frac

                # Simple aggregate pack (kept constant for demo stability)
                fine = 650.0
                coarse = 1150.0
                sp = 2.5 if use_sp else 0.0

                mix = {
                    "OPC 43": cement,  # default binder (will compare with PPC baseline if chosen)
                    "Fly Ash": flyash,
                    "GGBS": ggbs,
                    "Water": target_water,
                    "PCE Superplasticizer": sp,
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
                        "cementitious": float(cementitious),
                        "cement": float(cement),
                        "flyash": float(flyash),
                        "ggbs": float(ggbs),
                        "water_target": float(target_water),
                        "sp": float(sp),
                        "fine": float(fine),
                        "coarse": float(coarse),
                        "scm_total_frac": float(flyash_frac + ggbs_frac),
                        "grade": grade, "exposure": exposure, "nom_max": nom_max, "slump": int(target_slump),
                    }

    return best_df, best_meta

def generate_baseline(grade, exposure, nom_max, target_slump, emissions, baseline_type="OPC 43", use_sp=True, sp_reduction=0.18):
    """Baseline with 100% cementitious as selected cement (OPC 33, OPC 43, OPC 53, PPC)."""
    w_b_limit = float(EXPOSURE_WB_LIMITS[exposure])
    min_cem = float(EXPOSURE_MIN_CEMENT[exposure])

    water_target = water_for_slump(nom_max_mm=nom_max, slump_mm=int(target_slump), uses_sp=use_sp, sp_reduction_frac=sp_reduction)
    cementitious = max(water_target / w_b_limit, min_cem)

    cement_name = baseline_type
    mix = {
        cement_name: cementitious,
        "Fly Ash": 0.0,
        "GGBS": 0.0,
        "Water": water_target,
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
        "baseline_type": baseline_type,
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

# =========================
# UI
# =========================
st.title("🌍 CivilGPT")
st.subheader("AI-Powered Sustainable Concrete Mix Designer (IS-aware)")

st.markdown(
    "Generates **eco-optimized, IS-style concrete mix designs** and compares against cement baselines (OPC/PPC) with CO₂ footprint and compliance checks."
)

# ---- Sidebar Inputs
st.sidebar.header("📝 Mix Inputs")
grade = st.sidebar.selectbox("Concrete Grade", list(GRADE_STRENGTH.keys()), index=2)
exposure = st.sidebar.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()), index=2)
baseline_choice = st.sidebar.selectbox("Baseline Cement Type", ["OPC 33", "OPC 43", "OPC 53", "PPC"], index=1)

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
            emissions=emissions_df, use_sp=use_sp, sp_reduction=sp_reduction
        )
        base_df, base_meta = generate_baseline(
            grade=grade, exposure=exposure, nom_max=nom_max, target_slump=target_slump,
            emissions=emissions_df, baseline_type=baseline_choice, use_sp=use_sp, sp_reduction=sp_reduction
        )

        if opt_df is None or base_df is None:
            st.error("No feasible mix found under given constraints.")
        else:
            st.success(f"Mixes generated for {grade} under {exposure} exposure.")

            st.subheader("Optimized Sustainable Mix")
            st.dataframe(opt_df, use_container_width=True)

            st.subheader(f"{baseline_choice} Baseline Mix")
            st.dataframe(base_df, use_container_width=True)

            # KPIs
            co2_opt = float(opt_df["CO2_Emissions (kg/m3)"].sum())
            co2_base = float(base_df["CO2_Emissions (kg/m3)"].sum())
            reduction = (co2_base - co2_opt) / co2_base * 100 if co2_base > 0 else 0.0

            k1, k2, k3 = st.columns(3)
            k1.metric("🌱 Optimized CO₂", f"{co2_opt:.1f} kg/m³")
            k2.metric(f"🏗️ {baseline_choice} Baseline CO₂", f"{co2_base:.1f} kg/m³")
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

            with st.expander(f"{baseline_choice} Baseline — Assumptions & Checks", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Derived Parameters**")
                    st.json(base_derived)
                    st.caption(f"Entrapped air assumed: {air_pct:.1f} %")
                    fa_free_w_b, _ = aggregate_correction(fa_moist - fa_abs, base_meta["fine"])
                    ca_free_w_b, _ = aggregate_correction(ca_moist - ca_abs, base_meta["coarse"])
                    st.write(f"Free water from aggregates (report): {fa_free_w_b + ca_free_w_b:.1f} kg/m³")
                with c2:
                    st.markdown("**Compliance**")
                    for k, v in base_checks.items():
                        st.write(f"• {k}: {'✅' if v else '❌'}")

            # IS 383 sieve compliance
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
                    ok_ca, msgs_ca = sieve_check_ca20(df_coarse)
                    for m in msgs_ca:
                        st.write(("✅ " if ok_ca else "❌ ") + m)
                except Exception as e:
                    st.warning(f"Could not read coarse sieve CSV: {e}")
            if fine_csv is None and coarse_csv is None:
                st.info("Upload sieve CSVs (optional) to auto-check IS 383 fine zone & 20 mm coarse gradation.")

            # Downloads
            csv_opt = opt_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Optimized Mix (CSV)", data=csv_opt,
                               file_name=f"CivilGPT_{grade}_optimized.csv", mime="text/csv")
            csv_base = base_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Baseline Mix (CSV)", data=csv_base,
                               file_name=f"CivilGPT_{grade}_{baseline_choice}_baseline.csv", mime="text/csv")

else:
    st.info("Set parameters in the sidebar and click **Generate Sustainable Mix**.")

st.markdown("---")
st.caption("CivilGPT v1.4 | IS-aware Sustainable Concrete Mix Designer")
