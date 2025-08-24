import pandas as pd
from pathlib import Path

# Input Excel file
input_file = "data/concrete_Data.xls"

# Load dataset
df_slump = pd.read_excel(input_file)

# 🔎 Print original columns to verify
print("Original columns in Excel:", df_slump.columns.tolist())

# If mismatch occurs, auto-generate safe column names
expected_cols = [
    "Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer",
    "CoarseAggregate", "FineAggregate", "Age", "Slump", "Flow", "CompressiveStrength"
]

# If numbers match, rename → else fallback
if len(df_slump.columns) == len(expected_cols):
    df_slump.columns = expected_cols
else:
    df_slump.columns = [f"col_{i}" for i in range(len(df_slump.columns))]
    print("⚠️ Column count mismatch. Assigned generic names instead.")

# Save cleaned dataset
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "dataset_processed.csv"
df_slump.to_csv(output_file, index=False)

print(f"✅ Processed dataset saved at {output_file}")
