import pandas as pd

# Load datasets
slump = pd.read_csv("data/slump_test.data")
concrete = pd.read_excel("data/Concrete_Data.xls")
scc = pd.read_excel("data/dataset.SCC.xlsx")

# Show basic info
print("Slump shape:", slump.shape)
print("Concrete shape:", concrete.shape)
print("SCC shape:", scc.shape)

# Save cleaned copies into processed folder
slump.to_csv("data/processed/slump.csv", index=False)
concrete.to_csv("data/processed/concrete.csv", index=False)
scc.to_csv("data/processed/scc.csv", index=False)

print("✅ All datasets processed and saved in data/processed/")
