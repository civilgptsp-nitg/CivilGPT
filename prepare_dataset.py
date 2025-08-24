print("🚀 Running NEW prepare_dataset.py")

import pandas as pd
from pathlib import Path

INPUT_XLS = "data/concrete_Data.xls"
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "dataset_processed.csv"

# Load the Excel file
df = pd.read_excel(INPUT_XLS)

print("Original columns:", df.columns.tolist())

# If there is an extra index-like column, drop it safely
# Common names: "No", "No.", "ID", "Index", "Unnamed: 0"
index_like = {"no", "no.", "id", "index", "unnamed: 0"}
first_col_name = str(df.columns[0]).strip().lower()
if len(df.columns) == 11 and (first_col_name in index_like or "unnamed" in first_col_name):
    df = df.drop(df.columns[0], axis=1)

# Expected 10 columns for UCI Concrete Slump dataset
expected_names = [
    "cement", "slag", "fly_ash", "water", "superplasticizer",
    "coarse_agg", "fine_agg", "slump_cm", "flow_cm", "strength_mpa"
]

if len(df.columns) == 10:
    df.columns = expected_names
else:
    # Don't crash—show you what we see so we can adjust once
    print("⚠️ Unexpected number of columns:", len(df.columns))
    print("Columns:", df.columns.tolist())
    print("Not renaming. Saving raw copy so we can inspect.")
    
# Save processed (or raw if unexpected) to CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved: {OUTPUT_CSV}")
print(df.head(3))
