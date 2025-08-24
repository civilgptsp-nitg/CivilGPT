import pandas as pd

# --------------------
# Load Concrete Strength dataset (UCI)
# --------------------
df_strength = pd.read_excel("data/Concrete_Data.xls")
df_strength.columns = [
    "cement", "slag", "flyash", "water", "superplasticizer",
    "coarse_agg", "fine_agg", "age_days", "compressive_strength"
]
df_strength["slump"] = None
df_strength["flow"] = None
df_strength["source"] = "UCI_Compressive"

# --------------------
# Load Slump Test dataset (UCI)
# --------------------
df_slump = pd.read_csv("data/slump_test.data", header=None)
df_slump.columns = [
    "cement", "slag", "flyash", "water", "superplasticizer",
    "coarse_agg", "fine_agg", "slump", "flow", "compressive_strength"
]
df_slump["age_days"] = 28   # assumption: dataset is 28-day strength
df_slump["source"] = "UCI_Slump"

# --------------------
# Load SCC dataset (Mendeley)
# --------------------
df_scc = pd.read_excel("data/dataset.scc.xlsx")
# Adjust column names depending on Excel structure
df_scc = df_scc.rename(columns={
    "Cement": "cement",
    "FA": "flyash",
    "GGBS": "slag",
    "Water": "water",
    "SP": "superplasticizer",
    "CA": "coarse_agg",
    "FAg": "fine_agg",
    "Slump": "slump",
    "Strength": "compressive_strength"
})
df_scc["age_days"] = 28
df_scc["flow"] = None
df_scc["source"] = "Mendeley_SCC"

# --------------------
# Merge all datasets
# --------------------
df_all = pd.concat([df_strength, df_slump, df_scc], ignore_index=True)

# Save master CSV
df_all.to_csv("data/civilgpt_master.csv", index=False)

print("✅ civilgpt_master.csv created with", len(df_all), "rows")
