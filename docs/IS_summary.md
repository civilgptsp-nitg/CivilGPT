# IS Code Summary for CivilGPT

CivilGPT implements the provisions of **IS 456:2000** (Plain & Reinforced Concrete – Code of Practice) and **IS 10262:2019** (Concrete Mix Proportioning – Guidelines) to ensure that every generated mix is **code-compliant, durable, and practical**.

---

## 1. IS 456:2000 – Durability & Minimum Requirements

- **Maximum Water–Binder Ratio (w/b) by Exposure**
  - Mild: 0.60
  - Moderate: 0.55
  - Severe: 0.50
  - Very Severe: 0.45
  - Marine: 0.40

- **Minimum Cementitious Content (kg/m³)**
  - Mild: 300
  - Moderate: 300
  - Severe: 320
  - Very Severe: 340
  - Marine: 360

- **Minimum Grade of Concrete by Exposure**
  - Mild: M20
  - Moderate: M25
  - Severe: M30
  - Very Severe: M35
  - Marine: M40

---

## 2. IS 10262:2019 – Mix Design Guidelines

- **Target Strength Calculation**
  \[
  f_{target} = f_{ck} + 1.65 \times S
  \]
  where \(S\) = standard deviation (based on site quality control).

- **Water Content (Baseline for 50 mm slump)**
  - 10 mm: 208 kg
  - 12.5 mm: 202 kg
  - 20 mm: 186 kg
  - 40 mm: 165 kg

  +3% water for each 25 mm increase in slump beyond 50 mm.

- **Coarse Aggregate Proportion (Table 5)**
  - Adjusted by aggregate size, fine aggregate zone, and w/b ratio.
  - Correction: ±0.01 in volume fraction for every ±0.05 change in w/b from 0.50.

- **Supplementary Cementitious Materials (SCMs)**
  - Fly Ash: up to 30%
  - GGBS: up to 50%
  - Total SCM ≤ 50% of binder content.

---

## 3. Quality Control (Standard Deviation S)

As per IS 10262 Table 2:
- Good QC → 5.0 MPa
- Fair QC → 7.5 MPa
- Poor QC → 10.0 MPa

---

## 4. Additional Sanity Checks

CivilGPT enforces:
- Unit weight between 2200–2600 kg/m³
- Warnings for unusual ranges:
  - Fine aggregate <500 or >900 kg/m³
  - Coarse aggregate <1000 or >1300 kg/m³
  - Superplasticizer dosage >20 kg/m³

---

✅ This ensures every CivilGPT mix is not just optimized for cost and carbon footprint, but also remains **durable, safe, and IS-code compliant**.
