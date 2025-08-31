# app.py — CivilGPT v1.6.2 (polish update, full code, no deletions)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import json

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
def _normalize_emissions_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    lower_cols = {c.lower(): c for c in df2.columns}
    mat_col = None
    for cand in ["material", "materials", "name"]:
        if cand in lower_cols: mat_col = lower_cols[cand]; break
    if mat_col is None and "Material" in df2.columns: mat_col = "Material"
    co2_col = None
    for cand in ["kg_co2_per_kg", "co2_factor", "co2", "kgco2perkg", "co2_factor(kg_co2_per_kg)"]:
        if cand in lower_cols: co2_col = lower_cols[cand]; break
    if co2_col is None and "CO2_Factor(kg_CO2_per_kg)" in df2.columns: co2_col = "CO2_Factor(kg_CO2_per_kg)"
    if mat_col is None or co2_col is None: return df2
    df2 = df2.rename(columns={mat_col: "Material", co2_col: "CO2_Factor(kg_CO2_per_kg)"})
    return df2[["Material", "CO2_Factor(kg_CO2_per_kg)"]]

@st.cache_data
def load_data(materials_file=None, emissions_file=None):
    mats, ems = None, None
    try:
        if materials_file: mats = pd.read_csv(materials_file)
        else: mats = pd.read_csv("materials_library.csv")
    except: 
        try: mats = pd.read_csv("data/materials_library.csv")
        except: pass
    try:
        if emissions_file: ems = pd.read_csv(emissions_file)
        else: ems = pd.read_csv("emission_factors.csv")
    except: 
        try: ems = pd.read_csv("data/emission_factors.csv")
        except: pass
    if ems is not None: ems = _normalize_emissions_df(ems)
    return mats, ems

def water_for_slump_and_shape(nom_max_mm, slump_mm, agg_shape, uses_sp=False, sp_reduction_frac=0.0):
    base = WATER_BASELINE.get(int(nom_max_mm), 186.0)
    water = base if slump_mm <= 50 else base*(1+0.03*((slump_mm-50)/25))
    water *= (1.0 + AGG_SHAPE_WATER_ADJ.get(agg_shape,0.0))
    if uses_sp and sp_reduction_frac>0: water *= (1-sp_reduction_frac)
    return float(water)

def evaluate_mix(components, emissions_df):
    df = pd.DataFrame(list(components.items()), columns=["Material","Quantity (kg/m3)"])
    df = df.merge(emissions_df, on="Material", how="left")
    if "CO2_Factor(kg_CO2_per_kg)" not in df.columns:
        df["CO2_Factor(kg_CO2_per_kg)"]=0
    df["CO2_Emissions (kg/m3)"]=df["Quantity (kg/m3)"]*df["CO2_Factor(kg_CO2_per_kg)"].fillna(0)
    return df

def aggregate_correction(delta_moisture_pct, agg_mass_ssd):
    water_delta=(delta_moisture_pct/100.0)*agg_mass_ssd
    corrected_mass=agg_mass_ssd*(1+delta_moisture_pct/100.0)
    return float(water_delta), float(corrected_mass)

def sieve_check_fa(df, zone):
    limits=FINE_AGG_ZONE_LIMITS[zone]; ok, msgs=True,[]
    for sieve,(lo,hi) in limits.items():
        row=df.loc[df["Sieve_mm"].astype(str)==sieve]
        if row.empty: ok=False; msgs.append(f"Missing sieve {sieve} mm."); continue
        p=float(row["PercentPassing"].iloc[0])
        if not(lo<=p<=hi): ok=False; msgs.append(f"{sieve} mm → {p:.1f}% (req {lo}-{hi}%)")
    if ok and not msgs: msgs=[f"Fine aggregate meets IS 383 {zone} limits."]
    return ok,msgs

def sieve_check_ca(df, nominal_mm):
    limits=COARSE_LIMITS[int(nominal_mm)]; ok, msgs=True,[]
    for sieve,(lo,hi) in limits.items():
        row=df.loc[df["Sieve_mm"].astype(str)==sieve]
        if row.empty: ok=False; msgs.append(f"Missing sieve {sieve} mm."); continue
        p=float(row["PercentPassing"].iloc[0])
        if not(lo<=p<=hi): ok=False; msgs.append(f"{sieve} mm → {p:.1f}% (req {lo}-{hi}%)")
    if ok and not msgs: msgs=[f"Coarse aggregate meets IS 383 ({nominal_mm} mm graded)."]
    return ok,msgs

def compliance_checks(mix_df, meta, exposure):
    checks={}
    checks["W/B ≤ exposure limit"]=meta["w_b"]<=EXPOSURE_WB_LIMITS[exposure]
    checks["Min cementitious met"]=meta["cementitious"]>=EXPOSURE_MIN_CEMENT[exposure]
    checks["SCM ≤ 50%"]=meta.get("scm_total_frac",0.0)<=0.50
    total_mass=float(mix_df["Quantity (kg/m3)"].sum())
    checks["Unit weight 2200–2600 kg/m³"]=2200.0<=total_mass<=2600.0
    derived={"w/b used":round(meta["w_b"],3),
             "cementitious (kg/m³)":round(meta["cementitious"],1),
             "SCM % of cementitious":round(100*meta.get("scm_total_frac",0.0),1),
             "total mass (kg/m³)":round(total_mass,1),
             "water target (kg/m³)":round(meta.get("water_target",0.0),1),
             "cement (kg/m³)":round(meta["cement"],1),
             "fly ash (kg/m³)":round(meta.get("flyash",0.0),1),
             "GGBS (kg/m³)":round(meta.get("ggbs",0.0),1),
             "fine agg (kg/m³)":round(meta["fine"],1),
             "coarse agg (kg/m³)":round(meta["coarse"],1),
             "SP (kg/m³)":round(meta["sp"],2),
             "fck (MPa)":meta.get("fck"),
             "fck,target (MPa)":meta.get("fck_target"),
             "QC (S, MPa)":meta.get("stddev_S")}
    return checks,derived

def compliance_table(checks:dict)->pd.DataFrame:
    df=pd.DataFrame(list(checks.items()),columns=["Check","Status"])
    df["Result"]=df["Status"].apply(lambda x:"✅ Pass" if x else "❌ Fail")
    return df[["Check","Result"]]

# =========================
# Mix Generators
# =========================
def generate_mix(grade,exposure,nom_max,target_slump,agg_shape,emissions,cement_choice,use_sp=True,sp_reduction=0.18):
    w_b_limit=EXPOSURE_WB_LIMITS[exposure]; min_cem=EXPOSURE_MIN_CEMENT[exposure]
    target_water=water_for_slump_and_shape(nom_max,target_slump,agg_shape,use_sp,sp_reduction)
    best_df,best_meta,best_co2=None,None,float("inf")
    for wb in np.linspace(0.35,w_b_limit,6):
        for flyash_frac in [0.0,0.2,0.3]:
            for ggbs_frac in [0.0,0.3,0.5]:
                if flyash_frac+ggbs_frac>0.50: continue
                cementitious=max(target_water/wb,min_cem)
                cement=cementitious*(1-flyash_frac-ggbs_frac)
                flyash=cementitious*flyash_frac; ggbs=cementitious*ggbs_frac
                fine,coarse=650.0,1150.0; sp=2.5 if use_sp else 0.0
                mix={cement_choice:cement,"Fly Ash":flyash,"GGBS":ggbs,"Water":target_water,
                     "PCE Superplasticizer":sp,"M-Sand":fine,"20mm Coarse Aggregate":coarse}
                df=evaluate_mix(mix,emissions); total_co2=float(df["CO2_Emissions (kg/m3)"].sum())
                if total_co2<best_co2:
                    best_df, best_co2 = df.copy(), total_co2
                    best_meta={"w_b":float(wb),"cementitious":float(cementitious),
                               "cement":float(cement),"flyash":float(flyash),"ggbs":float(ggbs),
                               "water_target":float(target_water),"sp":float(sp),
                               "fine":float(fine),"coarse":float(coarse),
                               "scm_total_frac":float(flyash_frac+ggbs_frac),
                               "grade":grade,"exposure":exposure,"nom_max":nom_max,"slump":int(target_slump)}
    return best_df,best_meta

def generate_baseline(grade,exposure,nom_max,target_slump,agg_shape,emissions,cement_choice,use_sp=True,sp_reduction=0.18):
    w_b_limit=EXPOSURE_WB_LIMITS[exposure]; min_cem=EXPOSURE_MIN_CEMENT[exposure]
    water_target=water_for_slump_and_shape(nom_max,target_slump,agg_shape,use_sp,sp_reduction)
    cementitious=max(water_target/w_b_limit,min_cem)
    mix={cement_choice:cementitious,"Fly Ash":0.0,"GGBS":0.0,"Water":water_target,
         "PCE Superplasticizer":2.5 if use_sp else 0.0,"M-Sand":650.0,"20mm Coarse Aggregate":1150.0}
    df=evaluate_mix(mix,emissions)
    meta={"w_b":float(w_b_limit),"cementitious":float(cementitious),"cement":float(cementitious),
          "flyash":0.0,"ggbs":0.0,"water_target":float(water_target),"sp":float(mix["PCE Superplasticizer"]),
          "fine":650.0,"coarse":1150.0,"scm_total_frac":0.0,"grade":grade,"exposure":exposure,
          "nom_max":nom_max,"slump":int(target_slump),"baseline_cement":cement_choice}
    return df,meta

# =========================
# UI
# =========================
st.title("🌍 CivilGPT")
st.subheader("AI-Powered Sustainable Concrete Mix Designer (IS-aware)")

st.sidebar.header("📝 Mix Inputs")
grade=st.sidebar.selectbox("Concrete Grade",list(GRADE_STRENGTH.keys()),index=4)
exposure=st.sidebar.selectbox("Exposure Condition",list(EXPOSURE_WB_LIMITS.keys()),index=2)
cement_choice=st.sidebar.selectbox("Cement Type",["OPC 33","OPC 43","OPC 53","PPC"],index=1)

nom_max=st.sidebar.selectbox("Nominal max aggregate (mm)",[10,12.5,20,40],index=2)
agg_shape=st.sidebar.selectbox("Aggregate shape",list(AGG_SHAPE_WATER_ADJ.keys()),index=0)
target_slump=st.sidebar.slider("Target slump (mm)",25,180,100,step=5)
use_sp=st.sidebar.checkbox("Use Superplasticizer (PCE)",True)
sp_reduction=st.sidebar.slider("SP water reduction (fraction)",0.00,0.30,0.18,step=0.01)

qc_level=st.sidebar.selectbox("Quality control level",list(QC_STDDEV.keys()),index=0)

air_pct=st.sidebar.number_input("Entrapped air (%)",1.0,3.0,2.0,step=0.5)
fa_moist=st.sidebar.number_input("Fine agg moisture (%)",0.0,10.0,0.0,step=0.1)
ca_moist=st.sidebar.number_input("Coarse agg moisture (%)",0.0,5.0,0.0,step=0.1)
fa_abs,ca_abs=1.0,0.5

fine_zone=st.sidebar.selectbox("Fine agg zone (IS 383)",["Zone I","Zone II","Zone III","Zone IV"],index=1)
fine_csv=st.sidebar.file_uploader("Fine sieve CSV (Sieve_mm,PercentPassing)",type=["csv"],key="fine_csv")
coarse_csv=st.sidebar.file_uploader("Coarse sieve CSV (Sieve_mm,PercentPassing)",type=["csv"],key="coarse_csv")

materials_file=st.sidebar.file_uploader("materials_library.csv",type=["csv"],key="materials_csv")
emissions_file=st.sidebar.file_uploader("emission_factors.csv",type=["csv"],key="emissions_csv")

materials_df,emissions_df=load_data(materials_file,emissions_file)

# =========================
# Run
# =========================
if st.button("Generate Sustainable Mix"):
    if materials_df is None or emissions_df is None:
        st.error("CSV files missing or invalid.")
    else:
        min_grade=EXPOSURE_MIN_GRADE[exposure]
        if list(GRADE_STRENGTH.keys()).index(grade)<list(GRADE_STRENGTH.keys()).index(min_grade):
            st.warning(f"Exposure {exposure} requires min grade {min_grade}. Using {min_grade}.
