import pandas as pd
from pathlib import Path

# Input Excel file
input_file = "data/concrete_Data.xls"

# Load dataset
df_slump = pd.read_excel(input_file)

# 🔎 Print original columns from Excel
print("Original columns in Excel:", df_slump.columns.tolist())

# Save dataset without renaming columns yet
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "dataset_processed.csv"
df_slump.to_csv(output_file, index=False)

print(f"✅ Dataset saved at {output_file}")
