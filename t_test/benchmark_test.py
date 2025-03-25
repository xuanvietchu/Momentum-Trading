import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta

def load_data(file, mode):
    path = f"./result/{file}"
    with open(path, "r") as f:
        data = json.load(f)
        data = data[mode]
    return np.array(data)

def compute_statistics(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    n = data.shape[0]

    return mean, std, min_val, max_val, n

def compute_diff_statistics(data, modified):
    # Compute the differences between data and modified
    data_diff = data - modified

    # Compute statistics
    mean_diff = np.mean(data_diff)
    std_diff = np.std(data_diff, ddof=1)
    n_diff = len(data_diff)
    t_statistic_diff = mean_diff / (std_diff / np.sqrt(n_diff))

    return mean_diff, std_diff, n_diff, t_statistic_diff


VW = True
SKIP = False
model_no = 7           
reversed = False

# "returns" or "longs" or "shorts" 
mode = "returns"
modes = ["returns", "longs", "shorts"]

if mode:
    modes = [mode]

for mode in modes:
    baseline_file = f"baseline_{'VW' if VW else 'EW'}_{'skip' if SKIP else 'noskip'}.json"
    modified_file = f"modified_{'VW' if VW else 'EW'}_{'skip' if SKIP else 'noskip'}_model{model_no}_{'reversed' if reversed else ''}.json"
    benchmark_file = f"benchmark.json"
    market_file = f"market.json"

    # Load the data
    baseline = load_data(baseline_file, mode)
    modified = load_data(modified_file, mode)
    benchmark = load_data(benchmark_file, mode)
    market = load_data(market_file, "returns")
    riskless = load_data(market_file, "riskless")

    # Generate x-axis dates: Start from Jul 2004, increment every 1 months
    start_date = datetime.datetime(2004, 2, 1)
    dates = [start_date + relativedelta(months=i) for i in range(len(baseline))]

    # Compute cumulative returns: Cumulative product of (1 + return)
    initial_investment = 1  # Starting with $1
    cumulative_baseline = initial_investment * np.cumprod((100 + baseline) / 100)
    cumulative_modified = initial_investment * np.cumprod((100 + modified) / 100)
    cumulative_benchmark = initial_investment * np.cumprod((100 + benchmark) / 100)
    cumulative_market = initial_investment * np.cumprod((100 + market) / 100)

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))  # Two subplots (2 rows, 1 column)

    # --- Plot 1: Return Dynamics ---
    axes[0].plot(dates, baseline, label="Baseline", linestyle="--", color="blue")
    # axes[0].plot(dates, modified, label="Modified", linestyle="-", color="red")
    axes[0].plot(dates, benchmark, label="Benchmark", linestyle="-.", color="green")
    # axes[0].plot(dates, market, label="Market", linestyle="-.", color="purple")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Returns (%)")
    axes[0].set_title(f"New vs. Base {mode} Strategy Return Dynamics {'(VW)' if VW else '(EW)'} model {model_no} {'(reversed)' if reversed else ''}")
    axes[0].legend()
    axes[0].grid(True)

    # --- Plot 2: Cumulative Dollar Returns ---
    axes[1].plot(dates, cumulative_baseline, label="Baseline", linestyle="--", color="blue")
    axes[1].plot(dates, cumulative_modified, label="Modified", linestyle="-", color="red")
    axes[1].plot(dates, cumulative_benchmark, label="Benchmark", linestyle="-.", color="green")
    axes[1].plot(dates, cumulative_market, label="Market", linestyle="-", color="purple")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Cumulative Dollar Returns")
    axes[1].set_title(f"New vs. Base {mode} Strategy Cumulative Dollar Returns")
    axes[1].legend()
    axes[1].grid(True)

    # Rotate x-axis labels for better visibility
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    plt.subplots_adjust(hspace=0.5)  # Increase space between subplots

    # plt.savefig(f"./result/comparison_{'VW' if VW else 'EW'}_{mode}_{'skip' if SKIP else 'noskip'}_model{model_no}_{'reversed' if reversed else ''}.png", dpi=300)

    # Show the plots
    plt.show()


    # Compute statistics
    mean_modified, std_modified, min_modified, max_modified, n_modified = compute_statistics(modified)
    mean_baseline, std_baseline, min_baseline, max_baseline, n_baseline = compute_statistics(baseline)
    mean_bench, std_bench, min_bench, max_bench, n_bench = compute_statistics(benchmark)
    mean_market, std_market, min_market, max_market, n_market = compute_statistics(market)
    mean_riskless, std_riskless, min_riskless, max_riskless, n_riskless = compute_statistics(riskless)

    # Compute t-statistic
    t_statistic_modified = (mean_modified) / (std_modified / np.sqrt(n_modified))
    t_statistic_baseline = (mean_baseline) / (std_baseline / np.sqrt(n_baseline))
    t_statistic_bench = (mean_bench) / (std_bench / np.sqrt(n_bench))
    t_statistic_market = (mean_market) / (std_market / np.sqrt(n_market))
    t_statistic_riskless = (mean_riskless) / (std_riskless / np.sqrt(n_riskless))

    risk_free_rate = np.mean(riskless)

    # Sharpe ratio
    sharpe_baseline = (mean_baseline - risk_free_rate) / std_baseline if std_baseline != 0 else np.nan
    sharpe_modified = (mean_modified - risk_free_rate) / std_modified if std_modified != 0 else np.nan
    sharpe_bench = (mean_bench - risk_free_rate) / std_bench if std_bench != 0 else np.nan
    sharpe_market = (mean_market - risk_free_rate) / std_market if std_market != 0 else np.nan

    # Compute correlation
    correlation_base = np.corrcoef(baseline, modified)[0, 1]
    correlation_bench = np.corrcoef(baseline, benchmark)[0, 1]
    correlation_market = np.corrcoef(baseline, market)[0, 1]
    correlation_riskless = np.corrcoef(baseline, riskless)[0, 1]
    
    # Compute mean of differences and t-statistic of differences
    mean_diff, std_diff, n_diff, t_statistic_diff = compute_diff_statistics(baseline, modified)
    mean_diff_bench, std_diff_bench, n_diff_bench, t_statistic_diff_bench = compute_diff_statistics(benchmark, modified)
    mean_diff_market, std_diff_market, n_diff_market, t_statistic_diff_market = compute_diff_statistics(market, modified)
    mean_diff_riskless, std_diff_riskless, n_diff_riskless, t_statistic_diff_riskless = compute_diff_statistics(riskless, modified)

    # Create DataFrame
    df = pd.DataFrame({
        "Metric": ["Mean", "Standard Deviation", "Minimum", "Maximum", "Number of Observations", "T-Statistic", "Sharpe Ratio", "Correlation with New", "Mean of Differences w.r.t. New", "T-Stat of Differences w.r.t. New"],
        "Modified": [mean_modified, std_modified, min_modified, max_modified, n_modified, t_statistic_modified, sharpe_modified, "N/A", "N/A", "N/A"],
        "Baseline": [mean_baseline, std_baseline, min_baseline, max_baseline, n_baseline, t_statistic_baseline, sharpe_baseline, correlation_base, mean_diff, t_statistic_diff],
        "Benchmark": [mean_bench, std_bench, min_bench, max_bench, n_bench, t_statistic_bench, sharpe_bench, correlation_bench, mean_diff_bench, t_statistic_diff_bench],
        "Market": [mean_market, std_market, min_market, max_market, n_market, t_statistic_market, sharpe_market, correlation_market, mean_diff_market, t_statistic_diff_market],
        "Riskless": [mean_riskless, std_riskless, min_riskless, max_riskless, n_riskless, t_statistic_riskless, "N/A", correlation_riskless, mean_diff_riskless, t_statistic_diff_riskless]
    })

    # Print the table
    print(f"Comparing New vs. Base {mode} Strategy {'(VW)' if VW else '(EW)'} model {model_no} {'(reversed)' if reversed else ''}")
    print(df.to_string(index=False))

# python t_test/benchmark_test.py