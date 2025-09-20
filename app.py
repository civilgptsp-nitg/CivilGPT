# app.py — CivilGPT v1.6.7 (drop-in: robust dataset path fix + Mix↔Strength correlation preserved)
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

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
# canonical dataset filenames
LAB_FILE = "lab_processed.xlsx"
MIX_FILE = "concrete_mix_design_data_cleaned.xlsx"

def safe_load_excel(name):
    """Try loading Excel from root/ or data/ folder with robust fallback."""
    # exact match priority
    for p in [name, f"data/{name}"]:
        if os.path.exists(p):
            return pd.read_excel(p)
    # case-insensitive fallback
    data_dir = "data"
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            if fname.lower() == name.lower():
                return pd.read_excel(os.path.join(data_dir, fname))
    return None

# Load datasets
lab_df = safe_load_excel(LAB_FILE)
mix_df = safe_load_excel(MIX_FILE)

# =========================
# UI
# =========================
st.title("🌍 CivilGPT")
st.subheader("AI-Powered Sustainable Concrete Mix Designer (IS-aware)")

st.markdown("Generates **eco-optimized, IS-style concrete mix designs** and correlates with lab-tested strengths.")

# ---- Dataset Previews
with st.expander("📂 Dataset Previews", expanded=False):
    if lab_df is not None:
        st.write("**Lab Processed Data**")
        st.dataframe(lab_df.head(), use_container_width=True)
    else:
        st.warning(f"{LAB_FILE} not found in root/ or data/")

    if mix_df is not None:
        st.write("**Concrete Mix Design Data**")
        st.dataframe(mix_df.head(), use_container_width=True)
    else:
        st.warning(f"{MIX_FILE} not found in root/ or data/")

# ---- Mix ↔ Strength Correlation
if lab_df is not None and mix_df is not None:
    with st.expander("📊 Mix ↔ Strength Correlation", expanded=False):
        try:
            # Normalize column names
            lab = lab_df.rename(columns=lambda x: x.strip().lower())
            mix = mix_df.rename(columns=lambda x: x.strip().lower())

            # Expected columns
            lab_cols = {"age", "grade", "average compressive strength"}
            mix_cols = {"grade", "age", "compressive strength"}

            if lab_cols.issubset(set(lab.columns)) and mix_cols.issubset(set(mix.columns)):
                merged = pd.merge(
                    lab, mix, on=["grade", "age"], how="inner", suffixes=("_lab", "_mix")
                )
                st.write("**Merged Dataset**")
                st.dataframe(merged.head(), use_container_width=True)

                # Scatter plot: Strength vs Age
                fig, ax = plt.subplots()
                for grade, gdf in merged.groupby("grade"):
                    ax.scatter(gdf["age"], gdf["average compressive strength"], label=f"Lab {grade}", marker="o")
                    ax.scatter(gdf["age"], gdf["compressive strength"], label=f"Mix {grade}", marker="x")
                ax.set_xlabel("Age (days)")
                ax.set_ylabel("Strength (MPa)")
                ax.set_title("Lab vs Mix Strength Correlation")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("Column mismatch: please check lab & mix dataset headers.")
        except Exception as e:
            st.error(f"Error building correlation: {e}")

st.markdown("---")
st.caption("CivilGPT v1.6.7 | Robust dataset path fix + Mix↔Strength correlation")
