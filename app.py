# app.py — CivilGPT v1.6.6 (robust dataset path handling)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import json
import traceback

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

GRADE_STRENGTH = {
    "M10": 10, "M15": 15, "M20": 20, "M25": 25, "M30": 30, "M35": 35, "M40": 40,
    "M45": 45, "M50": 50, "M55": 55, "M60": 60, "M65": 65, "M70": 70, "M75": 75, "M80": 80
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

# =========================
# Helpers
# =========================
@st.cache_data
def _read_csv_try(path):
    return pd.read_csv(path)

@st.cache_data
def load_data(materials_file=None, emissions_file=None):
    materials, emissions = None, None
    if materials_file is not None:
        try:
            materials = pd.read_csv(materials_file)
        except Exception as e:
            st.warning(f"Could not read uploaded materials CSV: {e}")
    if materials is None:
        for p in ["materials_library.csv", "data/materials_library.csv"]:
            try:
                materials = pd.read_csv(p)
                break
            except: pass
        if materials is None:
            st.warning("Materials CSV not found.")
            materials = pd.DataFrame(columns=["Material"])

    if emissions_file is not None:
        try:
            emissions = pd.read_csv(emissions_file)
        except Exception as e:
            st.warning(f"Could not read uploaded emission factors CSV: {e}")
    if emissions is None:
        for p in ["emission_factors.csv", "data/emission_factors.csv"]:
            try:
                emissions = pd.read_csv(p)
                break
            except: pass
        if emissions is None:
            st.warning("Emission factors CSV not found.")
            emissions = pd.DataFrame(columns=["Material","CO2_Factor(kg_CO2_per_kg)"])
    return materials, emissions

# =========================
# NEW: Load Real Datasets (robust paths)
# =========================
@st.cache_data
def load_real_datasets():
    def try_paths(filename, reader):
        for p in [f"data/{filename}", filename]:
            try:
                return reader(p)
            except: pass
        return None

    lab_df = try_paths("lab_processed.xlsx", pd.read_excel)
    mix_df = try_paths("concrete_mix_design_data_cleaned.xlsx", pd.read_excel)
    slump_df = try_paths("slump_test.data", lambda p: pd.read_csv(p, header=None))

    if lab_df is None:
        st.warning("lab_processed.xlsx not found in root or data/")
    if mix_df is None:
        st.warning("concrete_mix_design_data_cleaned.xlsx not found in root or data/")
    if slump_df is None:
        st.warning("slump_test.data not found in root or data/")

    return lab_df, mix_df, slump_df

# =========================
# Correlation Merge
# =========================
def correlate_mix_strength(lab_df, mix_df):
    if lab_df is None or mix_df is None:
        return None
    try:
        merged = pd.merge(
            mix_df,
            lab_df,
            how="left",
            left_on=["Grade of concrete", "Age_days"],
            right_on=["Grade of concrete", "Age_days"]
        )
        return merged
    except Exception as e:
        st.warning(f"Correlation merge failed: {e}")
        return None

# =========================
# UI
# =========================
st.title("🌍 CivilGPT")
st.subheader("AI-Powered Sustainable Concrete Mix Designer (IS-aware)")

st.markdown(
    "Generates **eco-optimized, IS-style concrete mix designs** and compares against baselines "
    "with CO₂ footprint and compliance checks."
)

# =========================
# Load datasets
# =========================
materials_df, emissions_df = load_data()
lab_df, mix_df, slump_df = load_real_datasets()

# Show previews
with st.expander("📂 Dataset Previews", expanded=False):
    if lab_df is not None:
        st.write("**Lab Strength Data (lab_processed.xlsx)**")
        st.dataframe(lab_df.head(), use_container_width=True)
    if mix_df is not None:
        st.write("**Concrete Mix Data (concrete_mix_design_data_cleaned.xlsx)**")
        st.dataframe(mix_df.head(), use_container_width=True)
    if slump_df is not None:
        st.write("**Slump Test Data (slump_test.data)**")
        st.dataframe(slump_df.head(), use_container_width=True)

# Correlation section
with st.expander("📊 Mix ↔ Strength Correlation", expanded=False):
    merged_df = correlate_mix_strength(lab_df, mix_df)
    if merged_df is not None:
        st.dataframe(merged_df.head(20), use_container_width=True)
        try:
            fig, ax = plt.subplots()
            merged_df.plot(
                kind="scatter", x="Age_days", y="Compressive strength(N/mm^2)",
                c="Grade of concrete", colormap="viridis", ax=ax
            )
            plt.title("Strength vs Age by Grade")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not plot correlation: {e}")

st.markdown("---")
st.caption("CivilGPT v1.6.6 | Robust dataset path handling")
