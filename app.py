import streamlit as st
import pandas as pd

# ---------------------------
# App Configuration
# ---------------------------
st.set_page_config(
    page_title="CivilGPT - Sustainable Concrete Design",
    page_icon="🧱",
    layout="wide"
)

# ---------------------------
# Header
# ---------------------------
st.title("🌍 CivilGPT")
st.subheader("AI-Powered Sustainable Concrete Mix Design Prototype")

st.markdown(
    """
    CivilGPT helps you design **construction-ready, eco-friendly concrete mixes**  
    by balancing **IS-code compliance, CO₂ footprint, and cost efficiency**.  
    """
)

# ---------------------------
# Sidebar - User Input
# ---------------------------
st.sidebar.header("📝 Input Parameters")

grade = st.sidebar.selectbox("Concrete Grade", ["M20", "M25", "M30", "M35", "M40"])
exposure = st.sidebar.selectbox("Exposure Condition", ["Mild", "Moderate", "Severe", "Very Severe", "Marine"])
slump = st.sidebar.slider("Target Slump (mm)", 50, 200, 100)
agg_size = st.sidebar.selectbox("Maximum Aggregate Size (mm)", [10, 20, 40])

materials = st.sidebar.multiselect(
    "Available Materials",
    ["OPC 43", "OPC 53", "PPC", "Fly Ash", "GGBS", "M-Sand", "Natural Sand", "20mm Coarse Aggregate", "PCE Superplasticizer"],
    default=["OPC 43", "Fly Ash", "M-Sand", "20mm Coarse Aggregate"]
)

# ---------------------------
# Main Section - Placeholder for AI/Optimizer
# ---------------------------
st.markdown("### ⚙️ Generated Sustainable Mix")

if st.button("Generate Mix Design"):
    # Placeholder dataframe (later replaced by optimizer results)
    data = {
        "Material": ["OPC 43", "Fly Ash", "M-Sand", "20mm Coarse Aggregate", "Water", "PCE Superplasticizer"],
        "Quantity (kg/m³)": [280, 120, 650, 1150, 160, 2.5],
        "CO₂ Factor (kg/kg)": [0.9, 0.1, 0.01, 0.005, 0, 0.02],
    }
    df = pd.DataFrame(data)
    df["CO₂ Emissions (kg/m³)"] = df["Quantity (kg/m³)"] * df["CO₂ Factor (kg/kg)"]

    st.success(f"Sustainable Mix generated for **{grade}** concrete under **{exposure}** exposure.")
    st.dataframe(df, use_container_width=True)

    # Show CO₂ summary
    total_co2 = df["CO₂ Emissions (kg/m³)"].sum()
    st.metric("🌱 Estimated CO₂ Footprint", f"{total_co2:.2f} kg/m³")

    # Download options
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Mix as CSV",
        data=csv,
        file_name=f"CivilGPT_{grade}_mix.csv",
        mime="text/csv",
    )
else:
    st.info("Enter your parameters on the left and click **Generate Mix Design**.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("CivilGPT v1.0 | Prototype for Sustainable Construction AI")
