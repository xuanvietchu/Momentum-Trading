import pandas as pd
import numpy as np
import time

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
VW = False   # Value-weighted flag
SKIP = False # Skip flag
rebalance = 3

returns = []

while True:
    if start_date not in df_by_date:
        break

    # Formation period: retrieve data for the formation period using the pre-indexed dictionary
    df1 = df_by_date[start_date].copy()
    sort_col = "past_return_skip" if SKIP else "past_return"
    df1['rank'] = df1[sort_col].rank(method="first", ascending=False)

    # Classify stocks: Winners (top 10%), Losers (bottom 10%), Neutral otherwise
    df1['portfolio'] = np.where(
        df1['rank'] <= np.percentile(df1['rank'], 10), 'W',
        np.where(df1['rank'] >= np.percentile(df1['rank'], 90), 'L', 'N')
    )

    # Select winners and losers portfolios
    buy = df1[df1['portfolio'] == 'W'].copy()
    sell = df1[df1['portfolio'] == 'L'].copy()

    # Assign weights (equal-weighted or value-weighted)
    buy['weight'] = buy['ME'] / buy['ME'].sum() if VW else 1 / len(buy)
    sell['weight'] = sell['ME'] / sell['ME'].sum() if VW else 1 / len(sell)

    # Determine the holding period end date (6 months later)
    end_date_pd = pd.to_datetime(start_date) + pd.DateOffset(months=rebalance)
    end_date = end_date_pd.strftime('%Y-%m')

    if int(end_date.split('-')[0]) == 2025:
        break

    # Retrieve holding period data using multi-index slicing
    idx = pd.IndexSlice
    df_hold = df.loc[idx[start_date:end_date, :], :]

    # --- Vectorized cumulative return computation ---
    # Reset index so that 'NCUSIP' becomes a column
    df_hold_reset = df_hold.reset_index()
    # Create a new column for 1 + RET
    df_hold_reset['one_plus_ret'] = 1 + df_hold_reset['RET']
    # Compute cumulative product per stock over the holding period and subtract 1
    accu_returns = df_hold_reset.groupby('NCUSIP')['one_plus_ret'].prod() - 1

    # Compute weighted returns by aligning on stock identifier
    buy_return = (buy['weight'] * accu_returns.reindex(buy.index)).sum()
    sell_return = (sell['weight'] * accu_returns.reindex(sell.index)).sum()

    # Overall portfolio return: long winners minus short losers
    portfolio_return = buy_return - sell_return
    returns.append(portfolio_return)

    color = '\033[92m' if portfolio_return > 0 else '\033[91m'
    reset = '\033[0m'
    print(f"{start_date} {end_date} {color}{round(portfolio_return * 100, 2)}%{reset}")
    # print(f"{color}{round(portfolio_return * 100, 2)}%{reset}")


    start_date = end_date

# Final output of settings and average return
print("Value-weighted" if VW else "Equal-weighted")
print("Skip" if SKIP else "No Skip")
print(f"Average return: {round(np.nanmean(returns) * 100, 2)}%")
print(f"Time taken: {time.time() - start} seconds")
