import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
# Generate x-axis dates: Start from Jul 2004, increment every 6 months
start_date = datetime.datetime(2004, 7, 30)

def generate_dates(data_length, start_date):
    return [start_date + datetime.timedelta(days=6 * 30 * i) for i in range(data_length)]

VW = True
SKIP = False
model_no = 7
reversed = False


# Load the JSON files
baseline_path = f"baseline_{'VW' if VW else 'EW'}_{'skip' if SKIP else 'noskip'}.json"
modified_path = f"modified_{'VW' if VW else 'EW'}_{'skip' if SKIP else 'noskip'}_model{model_no}_{'reversed' if reversed else ''}.json"


with open(f"./result/{baseline_path}", "r") as file:
    baseline_data = json.load(file)

with open(f"./result/{modified_path}", "r") as file:
    modified_data = json.load(file)

# Extract statistics for each category: Base Long, Base Short, New Long, New Short
summary_data_detailed = {
    "Metric": [
        "Average Number of Stocks",
        "Average of Mean Returns",
        "Average of Standard Deviation of Returns",
        "Minimum Return",
        "Maximum Return"
    ],
    "Base Long": [
        np.mean(baseline_data["stnum"]) // 2,
        np.mean(baseline_data["longs"]),
        np.mean(baseline_data["longs_stdev"]),
        min(baseline_data["longs"]),
        max(baseline_data["longs"])
    ],
    "Base Short": [
        np.mean(baseline_data["stnum"]) // 2,
        np.mean(baseline_data["shorts"]),
        np.mean(baseline_data["shorts_stdev"]),
        min(baseline_data["shorts"]),
        max(baseline_data["shorts"])
    ],
    "New Long": [
        np.mean(modified_data["stnum"]) // 2,
        np.mean(modified_data["longs"]),
        np.mean(modified_data["longs_stdev"]),
        min(modified_data["longs"]),
        max(modified_data["longs"])
    ],
    "New Short": [
        np.mean(modified_data["stnum"]) // 2,
        np.mean(modified_data["shorts"]),
        np.mean(modified_data["shorts_stdev"]),
        min(modified_data["shorts"]),
        max(modified_data["shorts"])
    ]
}

# Create DataFrame
detailed_summary_df = pd.DataFrame(summary_data_detailed)

print(f"Detailed Summary: Baseline {'VW' if VW else 'EW'} vs. Modified model {model_no}")
print("----------------------------------------------------")
# Display the table
print(detailed_summary_df.to_string(index=False))


# # Extract returns
# baseline_returns = np.array(baseline_data["returns"])
# modified_returns = np.array(modified_data["returns"])

# dates = generate_dates(len(baseline_returns), start_date)

# # Plot the comparison
# plt.figure(figsize=(12, 6))
# plt.plot(dates, baseline_returns, label="Baseline Returns", linestyle="--", color="blue")
# plt.plot(dates, modified_returns, label="Modified Returns", linestyle="-", color="red")

# plt.xlabel("Date")
# plt.ylabel("Returns")
# plt.title(f"Comparison of Returns: Baseline vs. Modified Model {model_no} ({'VW' if VW else 'EW'})")
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)

# # Save the plot
# plot_filename = f"./result/comparison_returns_model{model_no}.png"
# plt.savefig(plot_filename, dpi=300)
# plt.show()