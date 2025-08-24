# prepare_dataset.py
# Robust loader + normalizer for slump/strength Excel datasets (xls/xlsx)
# - Auto-finds the Excel file in /data
# - Renames columns by fuzzy matching (no fixed-length renaming)
# - Fills optional fields sensibly (e.g., GGBS=0 if absent)
# - Writes /data/processed/dataset_processed.csv and train/val splits

import os
import re
import sys
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = pathlib.Path("data")
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Find a likely Excel file in /data
def find_excel():
    candidates = []
    # common names first
    preferred = [
        DATA_DIR / "dataset.scc.xlsx",
        DATA_DIR / "concrete_data.xlsx",
        DATA_DIR / "concrete_data.xls",
        DATA_DIR / "data.xlsx",
        DATA_DIR / "data.xls",
    ]
    for p in preferred:
        if p.exists():
            return p

    # fallback: any .xlsx/.xls in /data
    for p in DATA_DIR.glob("*.xlsx"):
        candidates.append(p)
    for p in DATA_DIR.glob("*.xls"):
        candidates.append(p)

    if not candidates:
        print("❌ No Excel dataset found in /data. Put an .xlsx or .xls there.")
        sys.exit(1)
    # pick the first
    return candidates[0]

# 2) Fuzzy mapper from existing columns -> standard names
STD_COLS = {
    "cement": "Cement",
    "flyash": "FlyAsh",          # fly ash / FA
    "ggbs": "GGBS",              # GGBS / GGBFS / slag
    "water": "Water",
    "superplasticizer": "SP",    # SP / superplasticizer / admixture
    "coarse": "CoarseAgg",       # coarse agg
    "fine": "FineAgg",           # fine agg / sand / m-sand
    "slump": "Slump",
    "flow": "Flow",
    "compressive": "CompressiveStrength",  # compressive strength
    "strength": "CompressiveStrength",     # (fallback if 'compressive' not present)
    "age": "Age",
}

def normalize_name(col: str) -> str:
    c = re.sub(r"[^a-z0-9]+", "", col.lower())
    # direct pattern checks (order matters)
    if "cement" in c:
        return "Cement"
    if "fly" in c or c.endswith("fa"):
        return "FlyAsh"
    if "ggbs" in c or "ggbfs" in c or "slag" in c:
        return "GGBS"
    if "water" in c or c == "w":
        return "Water"
    if "super" in c or c == "sp" or "admixture" in c or "pce" in c:
        return "SP"
    if "coarse" in c:
        return "CoarseAgg"
    if "fine" in c or "sand" in c or "msand" in c:
        return "FineAgg"
    if "slump" in c:
        return "Slump"
    if "flow" in c or "slumpflow" in c:
        return "Flow"
    if "compress" in c or "fc" in c or c.startswith("cs"):
        return "CompressiveStrength"
    if "strength" in c:
        return "CompressiveStrength"
    if c == "age" or "days" in c or c.endswith("d"):
        return "Age"
    # keep original if unknown (we won't rely on it)
    return col

def load_excel(path: pathlib.Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)  # auto engine: openpyxl for xlsx, xlrd for xls
    except ImportError as e:
        print("❌ Missing Excel engine. If file is .xlsx, run: pip install openpyxl")
        print("   If file is .xls, run: pip install xlrd")
        raise
    return df

def to_numeric(df: pd.DataFrame, cols: list):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def main():
    f = find_excel()
    print(f"📄 Using dataset: {f}")

    df = load_excel(f)
    print("🔎 Original columns:", df.columns.tolist())

    # Create a rename map via fuzzy normalization
    rename_map = {}
    for c in df.columns:
        std = normalize_name(str(c))
        rename_map[c] = std

    df = df.rename(columns=rename_map)
    print("✅ Mapped columns:", df.columns.tolist())

    # Ensure key columns exist or create sensible defaults
    required = ["Cement", "Water", "CoarseAgg", "FineAgg", "CompressiveStrength"]
    optional_defaults = {
        "FlyAsh": 0.0,
        "GGBS": 0.0,
        "SP": 0.0,
        "Slump": None,
        "Flow": None,
        "Age": 28,
    }

    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        print("❌ Missing required columns after mapping:", missing_required)
        print("   Please make sure your file contains these fields (any reasonable name works—this script maps them).")
        sys.exit(1)

    # Add optional columns if absent
    for k, v in optional_defaults.items():
        if k not in df.columns:
            df[k] = v

    # Keep only the standard subset (plus keep any extra columns at the end for traceability)
    std_order = [
        "Cement", "FlyAsh", "GGBS", "Water", "SP",
        "CoarseAgg", "FineAgg", "Slump", "Flow",
        "CompressiveStrength", "Age"
    ]
    extras = [c for c in df.columns if c not in std_order]
    df = df[ [c for c in std_order if c in df.columns] + extras ]

    # Make numerics numeric (core 11 columns)
    to_numeric(df, [c for c in std_order if c != "Slump" and c != "Flow"])  # Slump/Flow may be numeric too, we’ll coerce anyway:
    to_numeric(df, ["Slump", "Flow"])

    # Drop rows with no strength
    before = len(df)
    df = df.dropna(subset=["CompressiveStrength"])
    after = len(df)
    if after < before:
        print(f"ℹ️ Dropped {before - after} rows without strength data.")

    # Save processed + split
    out_all = OUT_DIR / "dataset_processed.csv"
    df.to_csv(out_all, index=False)
    print(f"💾 Saved: {out_all}  ({len(df)} rows)")

    # Train/val split for later modeling
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    print(f"📦 Train: {len(train_df)}  |  Val: {len(val_df)}")

    print("✅ Done.")

if __name__ == "__main__":
    main()
