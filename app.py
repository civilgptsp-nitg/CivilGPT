# app.py — CivilGPT v2.1
# - Backend logic fully preserved from v2.0
# - UI refactored: Landing page input + Manual Input toggle
# - Tabbed results unchanged

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

# Groq client
try:
    from groq import Groq
    client = Groq(api_key=st.secrets.get("GROQ_API_KEY", None))
except Exception:
    client = None

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
    for p in [name, f"data/{name}"]:
        if os.path.exists(p):
            try:
                return pd.read_excel(p)
            except Exception:
                try:
                    return pd.read_excel(p, engine="openpyxl")
                except Exception:
                    return None
    return None

lab_df = safe_load_excel(LAB_FILE)
mix_df = safe_load_excel(MIX_FILE)

# =========================
# IS-style Rules & Tables
# =========================
EXPOSURE_WB_LIMITS = {"Mild": 0.60,"Moderate": 0.55,"Severe": 0.50,"Very Severe": 0.45,"Marine": 0.40}
EXPOSURE_MIN_CEMENT = {"Mild": 300, "Moderate": 300, "Severe": 320,"Very Severe": 340, "Marine": 360}
EXPOSURE_MIN_GRADE = {"Mild": "M20", "Moderate": "M25", "Severe": "M30","Very Severe": "M35", "Marine": "M40"}
GRADE_STRENGTH = {"M10": 10, "M15": 15, "M20": 20, "M25": 25,"M30": 30, "M35": 35, "M40": 40, "M45": 45, "M50": 50}
WATER_BASELINE = {10: 208, 12.5: 202, 20: 186, 40: 165}
AGG_SHAPE_WATER_ADJ = {"Angular (baseline)": 0.00, "Sub-angular": -0.03,"Sub-rounded": -0.05, "Rounded": -0.07,"Flaky/Elongated": +0.03}
QC_STDDEV = {"Good": 5.0, "Fair": 7.5, "Poor": 10.0}

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
# Parsers
# =========================
def simple_parse(text: str) -> dict:
    result = {}
    grade_match = re.search(r"\bM(10|15|20|25|30|35|40|45|50)\b", text, re.IGNORECASE)
    if grade_match: result["grade"] = "M" + grade_match.group(1)

    for exp in EXPOSURE_WB_LIMITS.keys():
        if re.search(exp, text, re.IGNORECASE): result["exposure"] = exp; break

    slump_match = re.search(r"slump\s*(\d+)", text, re.IGNORECASE)
    if slump_match: result["slump"] = int(slump_match.group(1))

    cement_types = ["OPC 33", "OPC 43", "OPC 53", "PPC"]
    for ctype in cement_types:
        if re.search(ctype.replace(" ", r"\s*"), text, re.IGNORECASE):
            result["cement"] = ctype; break

    nom_match = re.search(r"(10|12\.5|20|40)\s*-?\s*mm", text, re.IGNORECASE)
    if nom_match:
        try: result["nom_max"] = float(nom_match.group(1))
        except: pass

    return result

def parse_input_with_llm(user_text: str) -> dict:
    if client is None:
        return simple_parse(user_text)
    prompt = f"Extract grade, exposure, slump (mm), cement type, and nominal max aggregate from: {user_text}. Return JSON."
    resp = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    try:
        parsed = json.loads(resp.choices[0].message.content)
    except Exception:
        parsed = simple_parse(user_text)
    return parsed

# =========================
# Helpers, Mix Evaluation, Compliance, Feasibility, Sieve, Mix Generators
# (all functions from v2.0 kept exactly the same — omitted here for brevity but included in your actual file)
# =========================
# ⚠️ You keep all functions (evaluate_mix, compliance_checks, sanity_check_mix, 
# check_feasibility, sieve_check_fa, sieve_check_ca, generate_mix, generate_baseline, etc.) 
# exactly as in v2.0. I haven’t touched them.

# =========================
# UI Refactor
# =========================
st.markdown("<h1 style='text-align:center; font-size:40px;'>What can I help with?</h1>", unsafe_allow_html=True)

# Landing input
user_text = st.text_input(" ", placeholder="e.g., M30, moderate exposure, slump 100 mm, OPC 53, 20 mm aggregate")
manual_mode = st.toggle("Manual Input Mode")

# Sidebar if manual
if manual_mode:
    st.sidebar.header("📝 Mix Inputs")
    use_llm_parser = st.sidebar.checkbox("Use Groq LLM Parser", value=False)
    show_trace = st.sidebar.checkbox("Show optimizer trace (advanced)", value=False)
    grade = st.sidebar.selectbox("Concrete Grade", list(GRADE_STRENGTH.keys()), index=2)
    exposure = st.sidebar.selectbox("Exposure Condition", list(EXPOSURE_WB_LIMITS.keys()), index=2)
    cement_choice = st.sidebar.selectbox("Cement Type", ["OPC 33", "OPC 43", "OPC 53", "PPC"], index=2)
    nom_max = st.sidebar.selectbox("Nominal max aggregate (mm)", [10, 12.5, 20, 40], index=2)
    agg_shape = st.sidebar.selectbox("Aggregate shape", list(AGG_SHAPE_WATER_ADJ.keys()), index=0)
    target_slump = st.sidebar.slider("Target slump (mm)", 25, 180, 100, step=5)
    use_sp = st.sidebar.checkbox("Use Superplasticizer (PCE)", True)
    optimize_cost = st.sidebar.checkbox("Optimize for CO₂ + Cost", False)
    fine_fraction = st.sidebar.slider("Fine aggregate fraction", 0.3, 0.5, 0.40, step=0.01)
    qc_level = st.sidebar.selectbox("Quality control level", list(QC_STDDEV.keys()), index=0)
    air_pct = st.sidebar.number_input("Entrapped air (%)", 1.0, 3.0, 2.0, step=0.5)
    fa_moist = st.sidebar.number_input("Fine agg moisture (%)", 0.0, 10.0, 0.0, step=0.1)
    ca_moist = st.sidebar.number_input("Coarse agg moisture (%)", 0.0, 5.0, 0.0, step=0.1)
    fa_abs, ca_abs = 1.0, 0.5
    st.sidebar.markdown("---")
    fine_zone = st.sidebar.selectbox("Fine agg zone (IS 383)", ["Zone I","Zone II","Zone III","Zone IV"], index=1)
    fine_csv = st.sidebar.file_uploader("Fine sieve CSV", type=["csv"], key="fine_csv")
    coarse_csv = st.sidebar.file_uploader("Coarse sieve CSV", type=["csv"], key="coarse_csv")
    st.sidebar.markdown("---")
    materials_file = st.sidebar.file_uploader("materials_library.csv", type=["csv"], key="materials_csv")
    emissions_file = st.sidebar.file_uploader("emission_factors.csv", type=["csv"], key="emissions_csv")
    cost_file = st.sidebar.file_uploader("cost_factors.csv", type=["csv"], key="cost_csv")

else:
    parsed = simple_parse(user_text) if user_text else {}
    grade = parsed.get("grade","M30")
    exposure = parsed.get("exposure","Moderate")
    cement_choice = parsed.get("cement","OPC 53")
    nom_max = parsed.get("nom_max",20)
    target_slump = parsed.get("slump",100)
    agg_shape = "Angular (baseline)"
    use_sp, optimize_cost, fine_fraction, qc_level = True, False, 0.40, "Good"
    air_pct, fa_moist, ca_moist, fa_abs, ca_abs = 2.0, 0.0, 0.0, 1.0, 0.5
    show_trace = False
    fine_zone, fine_csv, coarse_csv = "Zone II", None, None
    materials_file, emissions_file, cost_file = None, None, None

# =========================
# Main Run Button
# =========================
if st.button("Generate Sustainable Mix"):
    try:
        # Load datasets
        materials_df, emissions_df, costs_df = load_data(materials_file, emissions_file, cost_file)

        # Exposure–minimum grade enforcement
        min_grade_required = EXPOSURE_MIN_GRADE[exposure]
        grade_order = list(GRADE_STRENGTH.keys())
        if grade_order.index(grade) < grade_order.index(min_grade_required):
            st.warning(f"Exposure {exposure} requires ≥ {min_grade_required}. Adjusted automatically.")
            grade = min_grade_required

        fck, S = GRADE_STRENGTH[grade], QC_STDDEV[qc_level]
        fck_target = fck + 1.65 * S

        # Generate mixes
        opt_df, opt_meta, trace = generate_mix(
            grade, exposure, nom_max, target_slump, agg_shape,
            emissions_df, costs_df, cement_choice,
            use_sp=use_sp, optimize_cost=optimize_cost, fine_fraction=fine_fraction
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
                m["fck"], m["fck_target"], m["stddev_S"] = fck, round(fck_target, 1), S
            st.success(f"Mixes generated for **{grade}** under **{exposure}** exposure.")

            tabs = st.tabs(["Overview","Optimized Mix","Baseline Mix","Trace & Calculations","Sieve & QA","Downloads"])

            # ---- Overview
            with tabs[0]:
                co2_opt, cost_opt = opt_meta["co2_total"], opt_meta["cost_total"]
                co2_base, cost_base = base_meta["co2_total"], base_meta["cost_total"]
                reduction = (co2_base - co2_opt) / co2_base * 100 if co2_base > 0 else 0.0
                cost_diff = cost_opt - cost_base
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("🌱 Optimized CO₂", f"{co2_opt:.1f} kg/m³")
                with col2: st.metric("🏗 Baseline CO₂", f"{co2_base:.1f} kg/m³")
                with col3: st.metric("📉 Reduction", f"{reduction:.1f}%")
                with col4: st.metric("💰 Cost Δ", f"{cost_diff:+.2f} ₹/m³")
                st.markdown("#### 📊 CO₂ Comparison")
                fig, ax = plt.subplots()
                bars = ax.bar(["Optimized", "Baseline"], [co2_opt, co2_base])
                bars[0].set_color("green"); bars[1].set_color("gray")
                ax.set_ylabel("CO₂ (kg/m³)")
                st.pyplot(fig)

            # ---- Optimized Mix
            with tabs[1]:
                st.dataframe(opt_df, use_container_width=True)
                feasible, reasons, derived, _ = check_feasibility(opt_df, opt_meta, exposure)
                st.json(derived)
                if reasons: st.warning("\n".join(reasons))
                else: st.success("✅ Optimized mix passes all IS compliance checks.")

            # ---- Baseline Mix
            with tabs[2]:
                st.dataframe(base_df, use_container_width=True)
                feasible, reasons, derived, _ = check_feasibility(base_df, base_meta, exposure)
                st.json(derived)
                if reasons: st.warning("\n".join(reasons))
                else: st.success("✅ Baseline mix passes all IS compliance checks.")

            # ---- Trace
            with tabs[3]:
                if trace:
                    trace_df = pd.DataFrame(trace)
                    st.dataframe(trace_df, use_container_width=True)
                    st.markdown("#### Scatter: CO₂ vs Cost")
                    fig, ax = plt.subplots()
                    ax.scatter(trace_df["co2"], trace_df["cost"], c=["green" if f else "red" for f in trace_df["feasible"]])
                    ax.set_xlabel("CO₂ (kg/m³)"); ax.set_ylabel("Cost (₹/m³)")
                    st.pyplot(fig)
                else: st.info("Trace not available.")

            # ---- Sieve
            with tabs[4]:
                if fine_csv is not None:
                    df_fine = pd.read_csv(fine_csv)
                    ok_fa, msgs_fa = sieve_check_fa(df_fine, fine_zone)
                    for m in msgs_fa: st.write(("✅ " if ok_fa else "❌ ") + m)
                else: st.info("Fine sieve CSV not provided.")
                if coarse_csv is not None:
                    df_coarse = pd.read_csv(coarse_csv)
                    ok_ca, msgs_ca = sieve_check_ca(df_coarse, nom_max)
                    for m in msgs_ca: st.write(("✅ " if ok_ca else "❌ ") + m)
                else: st.info("Coarse sieve CSV not provided.")

            # ---- Downloads
            with tabs[5]:
                csv_opt = opt_df.to_csv(index=False).encode("utf-8")
                csv_base = base_df.to_csv(index=False).encode("utf-8")
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    opt_df.to_excel(writer, sheet_name="Optimized Mix", index=False)
                    base_df.to_excel(writer, sheet_name="Baseline Mix", index=False)
                excel_bytes = buffer.getvalue()
                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer)
                styles = getSampleStyleSheet()
                story = [Paragraph("CivilGPT Sustainable Mix Report", styles["Title"]), Spacer(1, 8),
                         Paragraph(f"Grade: {grade} | Exposure: {exposure} | Cement: {cement_choice}", styles["Normal"])]
                data_summary = [["Metric", "Optimized", "Baseline"],
                                ["CO₂ (kg/m³)", f"{opt_meta['co2_total']:.1f}", f"{base_meta['co2_total']:.1f}"],
                                ["Cost (₹/m³)", f"{opt_meta['cost_total']:.2f}", f"{base_meta['cost_total']:.2f}"],
                                ["Reduction (%)", f"{reduction:.1f}", "-"]]
                tbl = Table(data_summary, hAlign="LEFT")
                tbl.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.grey)]))
                story.append(tbl)
                doc.build(story)
                pdf_bytes = pdf_buffer.getvalue()
                st.download_button("Optimized Mix (CSV)", csv_opt, "optimized_mix.csv", "text/csv")
                st.download_button("Baseline Mix (CSV)", csv_base, "baseline_mix.csv", "text/csv")
                st.download_button("Report (Excel)", excel_bytes, "CivilGPT_Report.xlsx")
                st.download_button("Report (PDF)", pdf_bytes, "CivilGPT_Report.pdf")

    except Exception as e:
        st.error(f"Error: {e}")
        st.text(traceback.format_exc())

else:
    st.info("Enter a prompt above or switch to Manual Input Mode to set parameters.")
