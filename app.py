# app.py — CivilGPT v1.6.1 (IS-aware refinements + robust CSV handling + meta JSON downloads)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import json

# For PDF export
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
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
# IS-style Rules & Tables
# =========================

# IS 456 durability-style w/b caps (representative)
EXPOSURE_WB_LIMITS = {
    "Mild": 0.60,
    "Moderate": 0.55,
    "Severe": 0.50,
    "Very Severe": 0.45,
    "Marine": 0.40,
}

# Minimum cementitious contents (kg/m³) from IS 456 durability table (typical teaching values)
EXPOSURE_MIN_CEMENT = {
    "Mild": 300,
    "Moderate": 300,
    "Severe": 320,
    "Very Severe": 340,
    "Marine": 360,
}

# Minimum concrete grade required by exposure (IS 456; reinforced concrete)
EXPOSURE_MIN_GRADE = {
    "Mild": "M20",
    "Moderate": "M25",
    "Severe": "M30",
    "Very Severe": "M35",
    "Marine": "M40",
}

# All common IS 456 grades (normal to high strength)
GRADE_STRENGTH = {
    "M10": 10, "M15": 15, "M20": 20, "M25": 25, "M30": 30, "M35": 35, "M40": 40,
    "M45": 45, "M50": 50, "M55": 55, "M60": 60, "M65": 65, "M70": 70, "M75": 75, "M80": 80
}

# IS 10262 baseline mixing water (kg/m³) for slump 25–50 mm, angular aggregate (typical teaching values)
WATER_BASELINE = {10: 208, 12.5: 202, 20: 186, 40: 165}

# Aggregate shape water adjustment factors (IS 10262 qualitative guidance)
AGG_SHAPE_WATER_ADJ = {
    "Angular (baseline)": 0.00,
    "Sub-angular": -0.03,
    "Sub-rounded": -0.05,
    "Rounded": -0.07,
    "Flaky/Elongated": +0.03,
}

# Standard deviation S (MPa) per quality control (IS 10262 table)
QC_STDDEV = {
    "Good": 5.0,
    "Fair": 7.5,
    "Poor": 10.0,
}

# IS 383: Fine aggregate grading limits
FINE_AGG_ZONE_LIMITS = {
    "Zone I":  {"10.0": (100,100),"4.75": (90,100),"2.36": (60,95),"1.18": (30,70),"0.600": (15,34),"0.300": (5,20),"0.150": (0,10)},
    "Zone II": {"10.0": (100,100),"4.75": (90,100),"2.36": (75,100),"1.18": (55,90),"0.600": (35,59),"0.300": (8,30),"0.150": (0,10)},
    "Zone III":{"10.0": (100,100),"4.75": (90,100),"2.36": (85,100),"1.18": (75,100),"0.600": (60,79),"0.300": (12,40),"0.150": (0,10)},
    "Zone IV": {"10.0": (95,100),"4.75": (95,100),"2.36": (95,100),"1.18": (90,100),"0.600": (80,100),"0.300": (15,50),"0.150": (0,15)},
}

# IS 383: Coarse aggregate graded limits
COARSE_LIMITS = {
    10: {"20.0": (100,100), "10.0": (85,100),  "4.75": (0,20)},
    20: {"40.0": (95,100),  "20.0": (95,100),  "10.0": (25,55), "4.75": (0,10)},
    40: {"80.0": (95,100),  "40.0": (95,100),  "20.0": (30,70), "10.0": (0,15)}
}

# =========================
# Helpers
# =========================
def _normalize_emissions_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts various common schemas and normalizes to:
      columns: ["Material", "CO2_Factor(kg_CO2_per_kg)"]
    """
    df2 = df.copy()
    # Lowercase mapping
    lower_cols = {c.lower(): c for c in df2.columns}
    # Material column options
    mat_col = None
    for cand in ["material", "materials", "name"]:
        if cand in lower_cols:
            mat_col = lower_cols[cand]; break
    if mat_col is None and "Material" in df2.columns:
        mat_col = "Material"
    # CO2 factor column options
    co2_col = None
    for cand in ["kg_co2_per_kg", "co2_factor", "co2", "kgco2perkg", "co2_factor(kg_co2_per_kg)"]:
        if cand in lower_cols:
            co2_col = lower_cols[cand]; break
    if co2_col is None and "CO2_Factor(kg_CO2_per_kg)" in df2.columns:
        co2_col = "CO2_Factor(kg_CO2_per_kg)"

    if mat_col is None or co2_col is None:
        # If we can't find expected columns, just return as-is; downstream will raise clear error.
        return df2

    df2 = df2.rename(columns={
        mat_col: "Material",
        co2_col: "CO2_Factor(kg_CO2_per_kg)"
    })
    return df2[["Material", "CO2_Factor(kg_CO2_per_kg]".replace("]",")")]]  # safe slice fix

@st.cache_data
def _read_csv_flexible(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_data(materials_file=None, emissions_file=None):
    """
    Robust loader:
      1) If sidebar uploaders provided, use those
      2) Else try repo root
      3) Else try data/
    """
    materials = None
    emissions = None
    errors = []

    # Try uploaded files first
    if materials_file is not None:
        try:
            materials = pd.read_csv(materials_file)
        except Exception as e:
            errors.append(f"Uploaded materials CSV error: {e}")
    if emissions_file is not None:
        try:
            emissions = pd.read_csv(emissions_file)
        except Exception as e:
            errors.append(f"Uploaded emissions CSV error: {e}")

    # Try repo root if still None
    if materials is None:
        try:
            materials = _read_csv_flexible("materials_library.csv")
        except Exception as e:
            errors.append(f"materials_library.csv not found in root: {e}")
    if emissions is None:
        try:
            emissions = _read_csv_flexible("emission_factors.csv")
        except Exception as e:
            errors.append(f"emission_factors.csv not found in root: {e}")

    # Try data/ folder
    if materials is None:
        try:
            materials = _read_csv_flexible("data/materials_library.csv")
        except Exception as e:
            errors.append(f"data/materials_library.csv not found: {e}")
    if emissions is None:
        try:
            emissions = _read_csv_flexible("data/emission_factors.csv")
        except Exception as e:
            errors.append(f"data/emission_factors.csv not found: {e}")

    if materials is None or emissions is None:
        st.error("Error loading CSVs. Details:\n" + "\n".join(errors))
        return None, None

    # Normalize emissions schema if needed
    emissions = _normalize_emissions_df(emissions)
    if "Material" not in emissions.columns or "CO2_Factor(kg_CO2_per_kg)" not in emissions.columns:
        st.error("Emission factors CSV must have columns: Material, CO2_Factor(kg_CO2_per_kg). "
                 "Tip: rename your columns or update the CSV.")
        return None, None

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
    if df["CO2_Factor(kg_CO2_per_kg)"].isna().any():
        missing = df.loc[df["CO2_Factor(kg_CO2_per_kg)"].isna(), "Material"].unique().tolist()
        st.warning(f"Missing CO₂ factors for: {missing}. Treated as 0.")
        df["CO2_Factor(kg_CO2_per_kg)"] = df["CO2_Factor(kg_CO2_per_kg)"].fillna(0.0)
    df["CO2_Emissions (kg/m3)"] = df["Quantity (kg/m3)"] * df["CO2_Factor(kg_CO2_per_kg)"]
    return df

def aggregate_correction(delta_moisture_pct: float, agg_mass_ssd: float):
    water_delta = (delta_moisture_pct / 100.0) * agg_mass_ssd
    corrected_mass = agg_mass_ssd * (1 + delta_moisture_pct / 100.0)
    return float(water_delta), float(corrected_mass)

def sieve_check_fa(df: pd.DataFrame, zone: str):
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

def sieve_check_ca(df: pd.DataFrame, nominal_mm: int):
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
        "fck (MPa)": meta.get("fck"),
        "fck,target (MPa)": meta.get("fck_target"),
        "QC (S, MPa)": meta.get("stddev_S"),
    }
    return checks, derived

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
    for wb in np.linspace(0.35, w_b_limit, 6):
        for flyash_frac in [0.0, 0.2, 0.3]:
            for ggbs_frac in [0.0, 0.3, 0.5]:
                if flyash_frac + ggbs_frac > 0.50:
                    continue
                cementitious = max(target_water / wb, min_cem)
                cement = cementitious * (1 - flyash_frac - ggbs_frac)
                flyash = cementitious * flyash_frac
                ggbs   = cementitious * ggbs_frac
                # Simple aggregate placeholders (volume method can replace in future)
                fine = 650.0
                coarse = 1150.0
                sp = 2.5 if use_sp else 0.0
                mix = {
                    cement_choice: cement,
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
        cement_choice: cementitious,
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
        "baseline_cement": cement_choice,
    }
    return df, meta

# =========================
# UI
# =========================
st.title("🌍 CivilGPT")
st.subheader("AI-Powered Sustainable Concrete Mix Designer (IS-aware)")

st.markdown(
    "Generates **eco-optimized, IS-style concrete mix designs** and compares against baselines "
    "with CO₂ footprint and compliance checks."
)

# ---- Sidebar Inputs
st.sidebar.header("📝 Mix Inputs")
grade = st.sidebar.selectbox("Concrete Grade", list(GRADE_STRENGTH.keys()), index=4)
exposure = st.sidebar.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()), index=2)
cement_choice = st.sidebar.selectbox("Cement Type", ["OPC 33", "OPC 43", "OPC 53", "PPC"], index=1)

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
# Data
# =========================
materials_df, emissions_df = load_data(materials_file, emissions_file)

# =========================
# Run
# =========================
if st.button("Generate Sustainable Mix"):
    if materials_df is None or emissions_df is None:
        st.error("CSV files missing or invalid. Fix and retry.")
    else:
        # Enforce minimum grade per exposure (without deleting user choice; we warn and proceed)
        min_grade_required = EXPOSURE_MIN_GRADE[exposure]
        grade_order = list(GRADE_STRENGTH.keys())
        if grade_order.index(grade) < grade_order.index(min_grade_required):
            st.warning(f"Exposure **{exposure}** requires minimum grade **{min_grade_required}** by IS 456. "
                       f"Proceeding with {min_grade_required}.")
            grade = min_grade_required

        fck = GRADE_STRENGTH[grade]
        S = QC_STDDEV[qc_level]
        fck_target = fck + 1.65 * S

        opt_df, opt_meta = generate_mix(
            grade=grade, exposure=exposure, nom_max=nom_max, target_slump=target_slump,
            agg_shape=agg_shape, emissions=emissions_df, cement_choice=cement_choice,
            use_sp=use_sp, sp_reduction=sp_reduction
        )
        base_df, base_meta = generate_baseline(
            grade=grade, exposure=exposure, nom_max=nom_max, target_slump=target_slump,
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

            st.success(f"Mixes generated for **{grade}** under **{exposure}** exposure using {cement_choice}.")

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
                    fa_free_w, _ = aggregate_correction(fa_moist - fa_abs, opt_meta["fine"])
                    ca_free_w, _ = aggregate_correction(ca_moist - ca_abs, opt_meta["coarse"])
                    st.write(f"Free water (report): {fa_free_w + ca_free_w:.1f} kg/m³")
                with c2:
                    for k, v in opt_checks.items():
                        st.write(f"• {k}: {'✅' if v else '❌'}")

            with st.expander(f"{cement_choice} Baseline — Details", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.json(base_derived)
                    st.caption(f"Entrapped air assumed: {air_pct:.1f} %")
                    fa_free_w_b, _ = aggregate_correction(fa_moist - fa_abs, base_meta["fine"])
                    ca_free_w_b, _ = aggregate_correction(ca_moist - ca_abs, base_meta["coarse"])
                    st.write(f"Free water (report): {fa_free_w_b + ca_free_w_b:.1f} kg/m³")
                with c2:
                    for k, v in base_checks.items():
                        st.write(f"• {k}: {'✅' if v else '❌'}")

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
            if fine_csv is None and coarse_csv is None:
                st.info("Upload sieve CSVs to auto-check IS 383 compliance.")

            st.markdown("### 📊 CO₂ Comparison")
            fig, ax = plt.subplots()
            ax.bar(["Optimized", f"{cement_choice} Baseline"], [co2_opt, co2_base])
            ax.set_ylabel("CO₂ Emissions (kg/m³)")
            st.pyplot(fig)

            # ---- Downloads (CSV)
            csv_opt = opt_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Optimized Mix (CSV)", data=csv_opt,
                               file_name=f"CivilGPT_{grade}_optimized.csv", mime="text/csv")
            csv_base = base_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Baseline Mix (CSV)", data=csv_base,
                               file_name=f"CivilGPT_{grade}_{cement_choice}_baseline.csv", mime="text/csv")

            # ---- Downloads (JSON meta)
            st.download_button("🧾 Download Optimized Meta (JSON)",
                               data=json.dumps(opt_meta, indent=2),
                               file_name=f"CivilGPT_{grade}_optimized_meta.json",
                               mime="application/json")
            st.download_button("🧾 Download Baseline Meta (JSON)",
                               data=json.dumps(base_meta, indent=2),
                               file_name=f"CivilGPT_{grade}_{cement_choice}_baseline_meta.json",
                               mime="application/json")

            # ---- Excel export
            try:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    opt_df.to_excel(writer, sheet_name="Optimized Mix", index=False)
                    base_df.to_excel(writer, sheet_name="Baseline Mix", index=False)
                st.download_button("📊 Download Report (Excel)", data=buffer.getvalue(),
                                   file_name=f"CivilGPT_{grade}_Report.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.warning(f"Excel export unavailable: {e}")

            # ---- PDF export (high-level summary)
            try:
                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer)
                styles = getSampleStyleSheet()
                story = []
                story.append(Paragraph("CivilGPT Sustainable Mix Report", styles["Title"]))
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"Grade: {grade} | Exposure: {exposure} | Cement: {cement_choice}", styles["Normal"]))
                story.append(Paragraph(f"Target mean strength (IS 10262): {fck_target:.1f} MPa (S={S:.1f})", styles["Normal"]))
                story.append(Paragraph(f"Optimized CO₂: {co2_opt:.1f} kg/m³", styles["Normal"]))
                story.append(Paragraph(f"{cement_choice} Baseline CO₂: {co2_base:.1f} kg/m³", styles["Normal"]))
                story.append(Paragraph(f"Reduction: {reduction:.1f} %", styles["Normal"]))
                doc.build(story)
                st.download_button("📄 Download Report (PDF)", data=pdf_buffer.getvalue(),
                                   file_name=f"CivilGPT_{grade}_Report.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"PDF export unavailable: {e}")

else:
    st.info("Set parameters and click **Generate Sustainable Mix**.")

st.markdown("---")
st.caption("CivilGPT v1.6.1 | IS-aware Sustainable Concrete Mix Designer — OPC 33/43/53, PPC • Grades M10–M80 • IS 456/10262/383 compliant")
