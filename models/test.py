import pandas as pd
import numpy as np
import time
import json

start = time.time()

# Load and preprocess data
df = pd.read_csv('.\\baseline_data\\stock_price_monthly_2003_2024.csv')
df = df.dropna(subset=["past_return", "past_return_skip"])

# Set multi-index on ['date', 'NCUSIP'] and sort for efficient slicing
df = df.set_index(['date', 'NCUSIP'])
df.sort_index(inplace=True)

# Pre-index by date: build a dictionary with date keys for fast lookup in the formation period
dates = df.index.get_level_values('date').unique()
df_by_date = {date: df.loc[date] for date in dates}

start_date = '2004-01'
VW = True   # Value-weighted flag
SKIP = False # Skip flag
rebalance = 6

results = {
    "returns": [],
    "longs": [],
    "shorts": [],
    "longs_stdev": [],
    "shorts_stdev": [],
    "stnum": [],
}

while start_date in df_by_date:
    end_date = (pd.to_datetime(start_date) + pd.DateOffset(months=rebalance)).strftime('%Y-%m')
    if end_date not in df_by_date:
        break

    # Formation period: retrieve data for the formation period using the pre-indexed dictionary
    df1 = df_by_date[start_date].copy()
    sort_col = "past_return_skip" if SKIP else "past_return"

    df1['portfolio'] = np.where(
        df1[sort_col] >= np.percentile(df1[sort_col], 99.99), 'W',  # Winners are the top 10%
        np.where(df1[sort_col] <= np.percentile(df1[sort_col], 0.01), 'L', 'N')  # Losers are the bottom 10%
    )
    # print(df1.sort_values(by='past_return')[['past_return', 'portfolio']])

    # Select winners and losers portfolios
    buy = df1[df1['portfolio'] == 'W'].copy()
    sell = df1[df1['portfolio'] == 'L'].copy()

    # Assign weights (equal-weighted or value-weighted)
    buy['weight'] = buy['ME'] / buy['ME'].sum() if VW else 1 / len(buy)
    sell['weight'] = sell['ME'] / sell['ME'].sum() if VW else 1 / len(sell)

    # print(f"Buying {len(buy)} stocks and selling {len(sell)} stocks")
    # print(f"They are {buy.index.get_level_values('NCUSIP')} and {sell.index.get_level_values('NCUSIP')}")

    current_date = start_date
    test = 1
    
    # --- Compute Portfolio Returns ---
    # holding_data = df.loc[pd.IndexSlice[start_date:end_date, :], :].reset_index()
    holding_start = (pd.to_datetime(start_date) + pd.DateOffset(months=1)).strftime('%Y-%m')
    holding_data = df.loc[pd.IndexSlice[holding_start:end_date, :], :].reset_index()

    holding_data['one_plus_ret'] = 1 + holding_data['RET']
    cumulative_returns = holding_data.groupby('NCUSIP')['one_plus_ret'].prod() - 1

    # print(cumulative_returns.reindex(buy.index))

    buy_return = (buy['weight'] * cumulative_returns.reindex(buy.index)).sum()
    buy_stdev = (buy['weight']**2 * cumulative_returns.reindex(buy.index)**2).sum()**0.5
    sell_return = (sell['weight'] * cumulative_returns.reindex(sell.index)).sum()
    sell_stdev = (sell['weight']**2 * cumulative_returns.reindex(sell.index)**2).sum()**0.5
    period_return = buy_return - sell_return

    results['returns'].append(period_return *100)
    results['longs'].append(buy_return*100)
    results['shorts'].append(sell_return*100)
    results['longs_stdev'].append(buy_stdev*100)
    results['shorts_stdev'].append(sell_stdev*100)
    results['stnum'].append(len(buy) + len(sell))
    
    # Print the period return with color coding (green for positive, red for negative)
    color = '\033[92m' if period_return > 0 else '\033[91m'
    reset = '\033[0m'
    print(f"{start_date} {end_date} {color}{round(period_return * 100, 2)}%{reset}")

    start_date = end_date
    break

# Save results as JSON
file_name = f"./result/baseline_{'VW' if VW else 'EW'}_{'skip' if SKIP else 'noskip'}.json"
with open(file_name, "w") as f:
    json.dump(results, f, indent=4)

# Final output of settings and average return
print("Value-weighted" if VW else "Equal-weighted")
print("Skip" if SKIP else "No Skip")
print(f"Average return: {round(np.nanmean(results['returns']), 2)}%")
print(f"Time taken: {time.time() - start} seconds")