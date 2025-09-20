# app.py — CivilGPT v1.6.4 (adds real dataset loaders)
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
    materials = None
    emissions = None
    if materials_file is not None:
        try:
            materials = pd.read_csv(materials_file)
        except Exception as e:
            st.warning(f"Could not read uploaded materials CSV: {e}")
    if materials is None:
        try:
            materials = _read_csv_try("materials_library.csv")
        except Exception:
            try:
                materials = _read_csv_try("data/materials_library.csv")
            except Exception:
                st.warning("Materials CSV not found in repo.")
                materials = pd.DataFrame(columns=["Material"])

    if emissions_file is not None:
        try:
            emissions = pd.read_csv(emissions_file)
        except Exception as e:
            st.warning(f"Could not read uploaded emission factors CSV: {e}")
    if emissions is None:
        try:
            emissions = _read_csv_try("emission_factors.csv")
        except Exception:
            try:
                emissions = _read_csv_try("data/emission_factors.csv")
            except Exception:
                st.warning("Emission factors CSV not found in repo.")
                emissions = pd.DataFrame(columns=["Material","CO2_Factor(kg_CO2_per_kg)"])
    return materials, emissions

# =========================
# NEW: Load Real Datasets
# =========================
@st.cache_data
def load_real_datasets():
    lab_df, mix_df, slump_df = None, None, None
    try:
        lab_df = pd.read_excel("data/lab_processed.xlsx")
    except Exception:
        st.warning("lab_processed.xlsx not found in data/")
    try:
        mix_df = pd.read_excel("data/concrete_mix_design_data_cleaned.xlsx")
    except Exception:
        st.warning("concrete_mix_design_data_cleaned.xlsx not found in data/")
    try:
        slump_df = pd.read_csv("data/slump_test.data", header=None)
    except Exception:
        st.warning("slump_test.data not found in data/")
    return lab_df, mix_df, slump_df

# =========================
# Mix Functions (unchanged)
# =========================
# ... [KEEP all your original functions here without changes]
# generate_mix, generate_baseline, compliance_checks, etc.

# =========================
# UI
# =========================
st.title("🌍 CivilGPT")
st.subheader("AI-Powered Sustainable Concrete Mix Designer (IS-aware)")

st.markdown(
    "Generates **eco-optimized, IS-style concrete mix designs** and compares against baselines "
    "with CO₂ footprint and compliance checks."
)

# Sidebar Inputs (unchanged)
# ... [KEEP all your original sidebar inputs here]

# =========================
# Load datasets
# =========================
materials_df, emissions_df = load_data()
lab_df, mix_df, slump_df = load_real_datasets()

# Show previews in collapsible sections
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

# =========================
# Run (your original run block)
# =========================
# ... [KEEP all your original run block here unchanged]

st.markdown("---")
st.caption("CivilGPT v1.6.4 | Added real dataset loaders & previews")
