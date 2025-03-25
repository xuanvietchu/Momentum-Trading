import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta


# Generate x-axis dates: Start from Jul 2004, increment every 6 months
start_date = datetime.datetime(2004, 2, 1)


def generate_dates(data_length, start_date):
    return [start_date + relativedelta(months=i) for i in range(data_length)]

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
        "Average of Mean V1",
        "Average of Standard Deviation of V1",
        "Minimum V1",
        "Maximum V1",
        "Average of Mean V2",
        "Average of Standard Deviation of V2",
        "Minimum V2",
        "Maximum V2"
    ],
    "Base Long": [
        np.mean(baseline_data["stnum"]) // 2,
        np.mean(baseline_data["longs"]),
        np.mean(baseline_data["longs_stdev"]),
        min(baseline_data["longs"]),
        max(baseline_data["longs"]),
        np.mean(modified_data["longs"]),
        np.mean(modified_data["longs_stdev"]),
        min(modified_data["longs"]),
        max(modified_data["longs"])
    ],
    "Base Short": [
        np.mean(baseline_data["stnum"]) // 2,
        np.mean(baseline_data["shorts"]),
        np.mean(baseline_data["shorts_stdev"]),
        min(baseline_data["shorts"]),
        max(baseline_data["shorts"]),
        np.mean(modified_data["shorts"]),
        np.mean(modified_data["shorts_stdev"]),
        min(modified_data["shorts"]),
        max(modified_data["shorts"])
    ],
    "New Long": [
        np.mean(modified_data["stnum"]) // 2,
        np.mean(baseline_data["longs"]),
        np.mean(baseline_data["longs_stdev"]),
        min(baseline_data["longs"]),
        max(baseline_data["longs"]),
        np.mean(modified_data["longs"]),
        np.mean(modified_data["longs_stdev"]),
        min(modified_data["longs"]),
        max(modified_data["longs"])
    ],
    "New Short": [
        np.mean(modified_data["stnum"]) // 2,
        np.mean(baseline_data["shorts"]),
        np.mean(baseline_data["shorts_stdev"]),
        min(baseline_data["shorts"]),
        max(baseline_data["shorts"]),
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
baseline_longs = np.array(baseline_data["longs"])
modified_longs = np.array(modified_data["longs"])

baseline_shorts = np.array(baseline_data["shorts"])
modified_shorts = np.array(modified_data["shorts"])

dates = generate_dates(len(baseline_longs), start_date)

# Create a figure with two subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

# Plot Baseline Comparison (V1)
axes[0].plot(dates, baseline_longs, label="Baseline longs", linestyle="--", color="blue")
axes[0].plot(dates, baseline_shorts, label="Baseline shorts", linestyle="-", color="red")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Returns")
axes[0].set_title(f"V1 Dynamic model={model_no} ({'VW' if VW else 'EW'})")
axes[0].legend()
axes[0].grid(True)
axes[0].tick_params(axis='x', rotation=45)

# Plot Modified Comparison (V2)
axes[1].plot(dates, modified_longs, label="Modified longs", linestyle="--", color="blue")
axes[1].plot(dates, modified_shorts, label="Modified shorts", linestyle="-", color="red")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Returns")
axes[1].set_title(f"V2 Dynamic model={model_no} ({'VW' if VW else 'EW'})")
axes[1].legend()
axes[1].grid(True)
axes[1].tick_params(axis='x', rotation=45)

# Adjust layout to prevent overlap
plt.tight_layout()
fig.subplots_adjust(hspace=0.4)  # Increased spacing between subplots

# Save the plot
plot_filename = f"./result/comparison_model{model_no}.png"
plt.savefig(plot_filename, dpi=300)
plt.show()