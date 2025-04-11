import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta

def compute_statistics(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    n = data.shape[0]

    return mean, std, min_val, max_val, n

VW = True
SKIP = False
model_no = 7           
reversed = False

# "returns" or "longs" or "shorts" 
mode = 'shorts'
modes = ["returns", "longs", "shorts"]

if mode:
    modes = [mode]

for mode in modes:
    baseline_file = f"baseline_{'VW' if VW else 'EW'}_{'skip' if SKIP else 'noskip'}.json"
    modified_file = f"modified_{'VW' if VW else 'EW'}_{'skip' if SKIP else 'noskip'}_model{model_no}_{'reversed' if reversed else ''}.json"

    # Load the data
    with open(f"./result/{baseline_file}", "r") as f:
        baseline = json.load(f)
        baseline = baseline[mode]

    with open(f"./result/{modified_file}", "r") as f:
        modified = json.load(f)
        modified = modified[mode]

    # Convert to numpy arrays (in case they are lists)
    baseline = np.array(baseline)
    modified = np.array(modified)

    # Generate x-axis dates: Start from Jul 2004, increment every 1 months
    start_date = datetime.datetime(2004, 2, 1)
    dates = [start_date + relativedelta(months=i) for i in range(len(baseline))]

    # Compute cumulative returns: Cumulative product of (1 + return)
    initial_investment = 1  # Starting with $1
    cumulative_baseline = initial_investment * np.cumprod((100 + baseline) / 100)
    cumulative_modified = initial_investment * np.cumprod((100 + modified) / 100)

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))  # Two subplots (2 rows, 1 column)

    # --- Plot 1: Return Dynamics ---
    axes[0].plot(dates, baseline, label="Baseline", linestyle="--", color="blue")
    axes[0].plot(dates, modified, label="Modified", linestyle="-", color="red")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Returns (%)")
    axes[0].set_title(f"New vs. Base {mode} Strategy Return Dynamics {'(VW)' if VW else '(EW)'} model {model_no} {'(reversed)' if reversed else ''}")
    axes[0].legend()
    axes[0].grid(True)

    # --- Plot 2: Cumulative Dollar Returns ---
    axes[1].plot(dates, cumulative_baseline, label="Baseline", linestyle="--", color="blue")
    axes[1].plot(dates, cumulative_modified, label="Modified", linestyle="-", color="red")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Cumulative Dollar Returns")
    axes[1].set_title(f"New vs. Base {mode} Strategy Cumulative Dollar Returns")
    axes[1].legend()
    axes[1].grid(True)

    # Rotate x-axis labels for better visibility
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    plt.subplots_adjust(hspace=0.5)  # Increase space between subplots

    plt.savefig(f"./result/comparison_{'VW' if VW else 'EW'}_{mode}_{'skip' if SKIP else 'noskip'}_model{model_no}_{'reversed' if reversed else ''}.png", dpi=300)

    # Show the plots
    plt.show()


    # Compute statistics
    mean_modified, std_modified, min_modified, max_modified, n_modified = compute_statistics(modified)
    mean_baseline, std_baseline, min_baseline, max_baseline, n_baseline = compute_statistics(baseline)

    # Compute return differences
    differences = modified - baseline
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)  # Sample standard deviation
    n = len(differences)

    # Compute t-statistic
    t_statistic_modified = (mean_modified) / (std_modified / np.sqrt(n))
    t_statistic_baseline = (mean_baseline) / (std_baseline / np.sqrt(n))
    t_statistic_diff = mean_diff / (std_diff / np.sqrt(n))

    risk_free_rate = 0.05  # 5%

    # Sharpe ratio
    sharpe_baseline = (mean_baseline - risk_free_rate) / std_baseline if std_baseline != 0 else np.nan
    sharpe_modified = (mean_modified - risk_free_rate) / std_modified if std_modified != 0 else np.nan

    # Compute correlation (handling potential singularity cases)
    if np.all(baseline == baseline[0]) or np.all(modified == modified[0]):
        correlation = np.nan  # Avoid singular matrix error when all values are identical
    else:
        correlation = np.corrcoef(baseline, modified)[0, 1]

    # Create DataFrame
    df = pd.DataFrame({
        "Metric": ["Mean", "Standard Deviation", "Minimum", "Maximum", "Number of Observations", "T-Statistic", "Sharpe Ratio", "Correlation"],
        "Modified": [mean_modified, std_modified, min_modified, max_modified, n_modified, t_statistic_modified, sharpe_modified, correlation],
        "Baseline": [mean_baseline, std_baseline, min_baseline, max_baseline, n_baseline, t_statistic_baseline, sharpe_baseline, "N/A"],
        "Modified - Baseline": [mean_modified - mean_baseline, std_modified - std_baseline, min_modified - min_baseline, 
                                max_modified - max_baseline, n_modified - n_baseline, t_statistic_diff, "N/A", correlation]
    })

    # Print the table
    print(f"Comparing New vs. Base {mode} Strategy {'(VW)' if VW else '(EW)'} model {model_no} {'(reversed)' if reversed else ''}")
    print(df.to_string(index=False))

# python t_test/t_test.py