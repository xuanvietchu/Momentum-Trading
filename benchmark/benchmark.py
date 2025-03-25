import pandas as pd
import matplotlib.pyplot as plt
import json

# File paths (update if needed)
momentum_file = "benchmark_data\\VW_benchmark.CSV"
stnum = "benchmark_data\\stnum.CSV"

results = {
    "returns": [],
    "longs": [],
    "shorts": [],
    "longs_stdev": [],
    "shorts_stdev": [],
    "stnum": [],
}

# Load STNUM
df_stnum = pd.read_csv(stnum)
df_stnum.columns = ["Date"] + [f"PRIOR {i}" for i in range(1, len(df_stnum.columns))]
df_stnum.rename(columns={"PRIOR 1": "Lo PRIOR", "PRIOR 10": "Hi PRIOR"}, inplace=True)
df_stnum = df_stnum[["Date", "Lo PRIOR", "PRIOR 2", "PRIOR 9", "Hi PRIOR"]]

# Clean and parse Date
df_stnum["Date"] = df_stnum["Date"].astype(str).str.strip()
df_stnum["Date"] = pd.to_datetime(df_stnum["Date"], format="%Y%m", errors="raise")
df_stnum = df_stnum.dropna(subset=["Date"])

# only get data after 2004
df_stnum = df_stnum[df_stnum["Date"] >= "2004-01-01"]

df_stnum["stnum_long"] = (df_stnum["Hi PRIOR"] + df_stnum["PRIOR 9"]) 
df_stnum["stnum_short"] = (df_stnum["Lo PRIOR"] + df_stnum["PRIOR 2"])
df_stnum["stnum"] =  (df_stnum["stnum_long"] + df_stnum["stnum_short"])

# Load Momentum Portfolio Returns
df_momentum = pd.read_csv(momentum_file)

# Ensure correct number of column names
df_momentum.columns = ["Date"] + [f"PRIOR {i}" for i in range(1, len(df_momentum.columns))]
df_momentum.rename(columns={"PRIOR 1": "Lo PRIOR", "PRIOR 10": "Hi PRIOR"}, inplace=True)
df_momentum = df_momentum[["Date", "Lo PRIOR", "PRIOR 2", "PRIOR 9", "Hi PRIOR"]]
df_momentum = df_momentum.dropna()

# Clean and parse Date
df_momentum["Date"] = df_momentum["Date"].astype(str).str.strip()
df_momentum["Date"] = pd.to_datetime(df_momentum["Date"], format="%Y%m", errors="raise")
df_momentum = df_momentum.dropna(subset=["Date"])

# only get data after 2004
df_momentum = df_momentum[df_momentum["Date"] >= "2004-02-01"]

# Convert return columns to numeric
df_momentum[["Lo PRIOR", "PRIOR 2", "PRIOR 9", "Hi PRIOR"]] = df_momentum[["Lo PRIOR", "PRIOR 2", "PRIOR 9", "Hi PRIOR"]].apply(pd.to_numeric, errors="coerce")

# Compute benchmark portfolio returns
df_momentum["Winner"] = (df_momentum["Hi PRIOR"] + df_momentum["PRIOR 9"]) / 2
df_momentum["Loser"] = (df_momentum["Lo PRIOR"] + df_momentum["PRIOR 2"]) / 2
df_momentum["Momentum Profit"] = df_momentum["Winner"] - df_momentum["Loser"]

initial_investment = 1  # Starting with $1
# current investment = initial_investment * (1 + previous return)
df_momentum["Cumulative Winner"] = initial_investment * (1 + df_momentum["Winner"]/100).cumprod()
df_momentum["Cumulative Loser"] = initial_investment * (1 + df_momentum["Loser"]/100).cumprod()
df_momentum["Cumulative Momentum Profit"] = initial_investment * (1 + df_momentum["Momentum Profit"]/100).cumprod()


results["returns"] = df_momentum["Momentum Profit"].tolist()
results["longs"] = df_momentum["Winner"].tolist()
results["shorts"] = df_momentum["Loser"].tolist()
# results["longs_stdev"] = df_momentum["Winner"].std()
# results["shorts_stdev"] = df_momentum["Loser"].std()
results["stnum"] = df_stnum["stnum"].tolist()

# Plot time series
def plot_returns(df, column, title):
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))  # Two subplots (2 rows, 1 column)

    # print(df[column])
    # date to YYYY-MM
    # --- Plot 1: Return Dynamics ---
    axes[0].plot(df["Date"], df[column], label=column, linestyle="-", color="red")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Returns (%)")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True)

    column = "Cumulative " + column
    # --- Plot 2: Cumulative Dollar Returns ---
    axes[1].plot(df["Date"], df[column], label=column, linestyle="-", color="red")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Cumulative Dollar Returns")
    axes[1].set_title(f"Cumulative Dollar Returns")
    axes[1].legend()
    axes[1].grid(True)

    # Rotate x-axis labels for better visibility
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    plt.subplots_adjust(hspace=0.5)  # Increase space between subplots
    # Show the plots
    plt.show()

plot_returns(df_momentum, "Momentum Profit", "Benchmark Momentum Profit Over Time")
plot_returns(df_momentum, "Winner", "Benchmark Winner Portfolio Returns")
plot_returns(df_momentum, "Loser", "Benchmark Loser Portfolio Returns")

file_name = f"./result/benchmark.json"
with open(file_name, "w") as f:
    json.dump(results, f, indent=4)
