# app.py — CivilGPT v1.6.4 (full drop-in: original app v1.6.3 preserved + robust dataset loading + previews + correlation)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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
# Version note
# =========================
APP_VERSION = "v1.6.4"

# =========================
# IS-style Rules & Tables (original)
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
# Helpers (original)
# =========================
@st.cache_data
def _read_csv_try(path):
    return pd.read_csv(path)

@st.cache_data
def load_data(materials_file=None, emissions_file=None):
    """
    Robust loader with graceful warnings (does not raise exceptions).
    Prefers uploaded files, then repo root, then data/ folder.
    """
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
                st.warning("Materials CSV not found in repo. Expected: materials_library.csv or data/materials_library.csv")
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
                st.warning("Emission factors CSV not found in repo. Expected: emission_factors.csv or data/emission_factors.csv")
    # Validate emissions columns
    if emissions is not None:
        expected_cols = [c.lower() for c in emissions.columns]
        # Accept flexible names but map them to canonical for internal use
        material_col = None
        co2_col = None
        for c in emissions.columns:
            lc = c.lower()
            if lc in ("material","materials","name","item"): material_col = c
            if lc in ("kg_co2_per_kg","co2_factor","co2","kgco2perkg","co2_factor(kg_co2_per_kg)","co2_factor(kg_co2/kg)"):
                co2_col = c
        if material_col is None or co2_col is None:
            st.warning("Emission CSV missing expected columns. Required: a material column and a CO2 factor column (e.g. 'CO2_Factor(kg_CO2_per_kg)'). You can still proceed but CO₂ will be treated as 0 for missing materials.")
            # create fallback with zero factors if necessary
            if material_col is None:
                if "material" in emissions.columns:
                    material_col = "material"
                else:
                    # try first column as Material
                    material_col = emissions.columns[0]
            if co2_col is None:
                # try to find numeric column as CO2
                numeric_cols = [c for c in emissions.columns if pd.api.types.is_numeric_dtype(emissions[c])]
                co2_col = numeric_cols[0] if numeric_cols else None
        # Build normalized emissions_df with canonical columns
        if material_col is not None and co2_col is not None:
            try:
                emissions = emissions[[material_col, co2_col]].rename(columns={material_col: "Material", co2_col: "CO2_Factor(kg_CO2_per_kg)"})
            except Exception:
                emissions = emissions.copy()
                emissions.columns = ["Material", "CO2_Factor(kg_CO2_per_kg)"][:len(emissions.columns)]
        else:
            # last resort
            emissions = emissions.copy()
            if emissions.shape[1] >= 2:
                emissions.columns = ["Material", "CO2_Factor(kg_CO2_per_kg)"] + list(emissions.columns[2:])
            elif emissions.shape[1] == 1:
                emissions.columns = ["Material"]; emissions["CO2_Factor(kg_CO2_per_kg)"] = 0.0
            else:
                emissions = pd.DataFrame(columns=["Material","CO2_Factor(kg_CO2_per_kg)"])
    else:
        # make a default empty emissions DF
        emissions = pd.DataFrame(columns=["Material","CO2_Factor(kg_CO2_per_kg)"])
    # Ensure materials exists as DataFrame (if not, provide empty)
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
    # Merge emissions; if material missing, warn and treat as 0
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
