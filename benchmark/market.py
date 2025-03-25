import pandas as pd
import matplotlib.pyplot as plt
import json

results = {
    "returns": [],
    "riskless":[]
}

ff_file = "benchmark_data\market.CSV"

# Load Fama-French Research Factors
df_ff = pd.read_csv(ff_file)
df_ff.columns = ["Date", "Mkt-RF", "SMB", "HML", "RF"]
df_ff = df_ff.dropna()

# Clean and parse Date
df_ff["Date"] = df_ff["Date"].astype(str).str.strip()
df_ff["Date"] = pd.to_datetime(df_ff["Date"], format="%Y%m", errors="coerce")
df_ff = df_ff.dropna(subset=["Date"])
df_ff["Mkt"] = df_ff["Mkt-RF"] + df_ff["RF"]

# only get data after 2004
df_ff = df_ff[df_ff["Date"] >= "2004-02-01"]

results["returns"] = df_ff["Mkt"].tolist()
results["riskless"] = df_ff["RF"].tolist()

file_name = f"./result/market.json"
with open(file_name, "w") as f:
    json.dump(results, f, indent=4)