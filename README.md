# 🌍 CivilGPT – Sustainable Concrete Mix Designer

**CivilGPT** is an **AI-assisted, IS-code–aware concrete mix design tool** built with [Streamlit](https://streamlit.io).  
It generates **eco-optimized concrete mix proportions**, compares them against **OPC/PPC baselines**, and ensures compliance with **Indian Standards**.

---

## 🚀 Features

- ✅ **IS 456 Durability Checks**  
  - Exposure-based **max w/b ratio** and **minimum cementitious content**.

- ✅ **IS 10262 Water Estimation**  
  - Baseline water content (186 kg/m³ for 20 mm)  
  - Slump-based adjustment (+3% per 25 mm)  
  - Superplasticizer-based water reduction.

- ✅ **IS 383 Aggregate Gradation Compliance**  
  - Fine aggregate Zones I–IV (sieve check).  
  - Coarse aggregate 20 mm graded (sieve check).  
  - Upload your sieve analysis CSVs for **PASS/FAIL** check.

- ✅ **Baseline Comparison**  
  - **OPC** and **PPC** baseline mixes.  
  - Optimized sustainable mix vs baseline.  
  - CO₂ footprint comparison + % reduction.

- ✅ **Moisture & Absorption Corrections**  
  - Fine & coarse aggregate moisture vs absorption adjustment.  
  - Free water correction reported.

- ✅ **Outputs**  
  - Optimized mix table (kg/m³).  
  - Baseline mix table (kg/m³).  
  - KPIs: CO₂ emissions & % reduction.  
  - Compliance check report.  
  - **Download mixes as CSV**.  

---

## 📊 Example Workflow

1. Select **Grade (M20–M40)** and **Exposure condition (Mild–Marine)**.  
2. Choose **baseline** (OPC or PPC).  
3. Adjust **slump, nominal max aggregate size, and SP usage**.  
4. *(Optional)* Upload sieve analysis CSVs for FA/CA.  
5. Click **Generate Mix**.  
6. Review outputs (tables, KPIs, compliance).  
7. Download as CSV.

---

## 🖼️ Screenshots (to be added later)

- App homepage  
- Optimized vs Baseline mix tables  
- IS 383 sieve compliance check panel  

---

## 🛠️ Tech Stack

- **Streamlit**, **pandas**, **numpy**  
- Data: `materials_library.csv`, `emission_factors.csv`  
- Deployment: Streamlit Cloud  

---

## ⚙️ Installation & Usage

```bash
# Clone this repo
git clone https://github.com/civilgptsp-nitg/CivilGPT.git
cd CivilGPT

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
