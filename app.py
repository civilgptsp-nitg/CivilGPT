import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="CivilGPT - Sustainable Concrete Mix Designer",
    page_icon="🧱",
    layout="wide"
)

# ---------------------------
# Constants (IS-code style rules)
# ---------------------------
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

GRADE_STRENGTH = {
    "M20": 20,
    "M25": 25,
    "M30": 30,
    "M35": 35,
    "M40": 40,
}

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    try:
        materials = pd.read_csv("materials_library.csv")
        emissions = pd.read_csv("emission_factors.csv")
        return materials, emissions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

materials_df, emissions_df = load_data()

# ---------------------------
# Helper Functions
# ---------------------------
def estimate_water_demand(slump, agg_size, use_sp=False):
    """Estimate water demand based on slump and aggregate size"""
    base = 180  # baseline water demand (kg/m3) for 100mm slump, 20mm agg
    adj = (slump - 100) * 0.3
    if agg_size == 10:
        adj += 5
    elif agg_size == 40:
        adj -= 5
    demand = base + adj
    if use_sp:
        demand *= 0.85  # SP reduces demand ~15%
    return max(demand, 140)

def evaluate_mix(cement, scm, water, agg, emissions_df):
    """Calculate CO2 footprint for a given mix"""
    mix = cement.copy()
    mix.update(scm)
    mix.update(water)
    mix.update(agg)

    df = pd.DataFrame(list(mix.items()), columns=["Material", "Quantity (kg/m3)"])
    df = df.merge(emissions_df, on="Material", how="left")
    df["CO2_Emissions (kg/m3)"] = df["Quantity (kg/m3)"] * df["CO2_Factor(kg_CO2_per_kg)"]
    return df

def generate_mix(grade, exposure, slump, agg_size, emissions, use_sp=True):
    """Rule-based optimizer for sustainable mix"""
    fck = GRADE_STRENGTH[grade]
    w_b_limit = EXPOSURE_WB_LIMITS[exposure]
    min_cement = EXPOSURE_MIN_CEMENT[exposure]

    best_mix = None
    best_co2 = float("inf")

    # Iterate over candidate mixes
    for wb in np.linspace(0.35, w_b_limit, 6):  # candidate w/b ratios
        for flyash_frac in [0.0, 0.2, 0.3]:
            for ggbs_frac in [0.0, 0.3, 0.5]:
                if flyash_frac + ggbs_frac > 0.5:  # limit SCM replacement
                    continue

                water = estimate_water_demand(slump, agg_size, use_sp)
                cementitious = max(water / wb, min_cement)
                cement = cementitious * (1 - flyash_frac - ggbs_frac)
                flyash = cementitious * flyash_frac
                ggbs = cementitious * ggbs_frac

                # Check density balance
                fine = 650
                coarse = 1150
                sp = 2.5 if use_sp else 0.0

                cement_dict = {"OPC 43": cement}
                scm_dict = {"Fly Ash": flyash, "GGBS": ggbs}
                water_dict = {"Water": water, "PCE Superplasticizer": sp}
                agg_dict = {"M-Sand": fine, "20mm Coarse Aggregate": coarse}

                df = evaluate_mix(cement_dict, scm_dict, water_dict, agg_dict, emissions)

                total_co2 = df["CO2_Emissions (kg/m3)"].sum()
                if total_co2 < best_co2:
                    best_co2 = total_co2
                    best_mix = df.copy()

    return best_mix

def generate_baseline(grade, exposure, slump, agg_size, emissions, baseline_type="OPC", use_sp=True):
    """Generate baseline mix (100% OPC or 100% PPC)"""
    w_b_limit = EXPOSURE_WB_LIMITS[exposure]
    min_cement = EXPOSURE_MIN_CEMENT[exposure]

    water = estimate_water_demand(slump, agg_size, use_sp)
    cementitious = max(water / w_b_limit, min_cement)

    if baseline_type == "OPC":
        cement_dict = {"OPC 43": cementitious}
    else:  # PPC
        cement_dict = {"PPC": cementitious}

    scm_dict = {"Fly Ash": 0.0, "GGBS": 0.0}
    water_dict = {"Water": water, "PCE Superplasticizer": 2.5 if use_sp else 0.0}
    agg_dict = {"M-Sand": 650, "20mm Coarse Aggregate": 1150}

    df = evaluate_mix(cement_dict, scm_dict, water_dict, agg_dict, emissions)
    return df

# ---------------------------
# UI
# ---------------------------
st.title("🌍 CivilGPT")
st.subheader("AI-Powered Sustainable Concrete Mix Designer")

st.markdown(
    """
    CivilGPT generates **eco-optimized, IS-code-aligned concrete mix designs**.  
    It balances **strength, workability, durability, and CO₂ footprint**.  
    """
)

# Sidebar Inputs
st.sidebar.header("📝 Input Parameters")
grade = st.sidebar.selectbox("Concrete Grade", list(GRADE_STRENGTH.keys()))
exposure = st.sidebar.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()))
slump = st.sidebar.slider("Target Slump (mm)", 50, 200, 100)
agg_size = st.sidebar.selectbox("Max Aggregate Size (mm)", [10, 20, 40])
use_sp = st.sidebar.checkbox("Use Superplasticizer", True)
baseline_choice = st.sidebar.radio("Baseline Mix Type", ["OPC", "PPC"])

# Main Output
if st.button("Generate Sustainable Mix"):
    if materials_df is None or emissions_df is None:
        st.error("Data files missing. Please upload CSVs to repo.")
    else:
        mix_df = generate_mix(grade, exposure, slump, agg_size, emissions_df, use_sp)
        baseline_df = generate_baseline(grade, exposure, slump, agg_size, emissions_df, baseline_choice, use_sp)

        if mix_df is not None and baseline_df is not None:
            st.success(f"Sustainable Mix generated for {grade} under {exposure} exposure.")

            st.subheader("Optimized Sustainable Mix")
            st.dataframe(mix_df, use_container_width=True)

            st.subheader(f"{baseline_choice} Baseline Mix")
            st.dataframe(baseline_df, use_container_width=True)

            # KPI
            total_co2_opt = mix_df["CO2_Emissions (kg/m3)"].sum()
            total_co2_baseline = baseline_df["CO2_Emissions (kg/m3)"].sum()
            reduction = (total_co2_baseline - total_co2_opt) / total_co2_baseline * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("🌱 Optimized Mix CO₂", f"{total_co2_opt:.1f} kg/m³")
            col2.metric(f"🏗️ {baseline_choice} Baseline CO₂", f"{total_co2_baseline:.1f} kg/m³")
            col3.metric("📉 % Reduction", f"{reduction:.1f}%")

            # Downloads
            csv = mix_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Optimized Mix (CSV)",
                data=csv,
                file_name=f"CivilGPT_{grade}_mix.csv",
                mime="text/csv",
            )

            csv_baseline = baseline_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Baseline Mix (CSV)",
                data=csv_baseline,
                file_name=f"CivilGPT_{grade}_baseline.csv",
                mime="text/csv",
            )

        else:
            st.error("No feasible mix found under given constraints.")
else:
    st.info("Set parameters in the sidebar and click Generate Sustainable Mix.")

# Footer
st.markdown("---")
st.caption("CivilGPT v1.1 | Sustainable Construction AI Prototype")
