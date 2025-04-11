import pandas as pd
import numpy as np
import time
import json
import yaml


if __name__ == "__main__":
    # Load configuration from YAML file
    config = yaml.load(open(".\\config\\baseline_config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

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

    start_date = config['start_date']  # Start date for the backtest
    VW = config['VW']  # Value-weighted or equal-weighted
    SKIP = config['SKIP']  # Skip the first month of returns
    rebalance = config['rebalance']  # Rebalance period in months

    results = {
        "returns": [],
        "longs": [],
        "shorts": [],
        "longs_stdev": [],
        "shorts_stdev": [],
        "stnum": [],
    }
    tests = []

    while start_date in df_by_date:
        end_date = (pd.to_datetime(start_date) + pd.DateOffset(months=rebalance)).strftime('%Y-%m')

        # Formation period: retrieve data for the formation period using the pre-indexed dictionary
        df1 = df_by_date[start_date].copy()
        sort_col = "past_return_skip" if SKIP else "past_return"

        df1['portfolio'] = np.where(
            df1[sort_col] >= np.percentile(df1[sort_col], 80), 'W',  # Winners are the top 20%
            np.where(df1[sort_col] <= np.percentile(df1[sort_col], 20), 'L', 'N')  # Losers are the bottom 20%
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
        
        while current_date < end_date:
            next_date = (pd.to_datetime(current_date) + pd.DateOffset(months=1)).strftime('%Y-%m')
            if next_date not in df_by_date:
                break
            
            df_hold = df_by_date[next_date]
            monthly_returns = df_hold['RET']
            
            # Compute weighted returns
            # Filter out missing returns
            valid_buy = monthly_returns.reindex(buy.index).dropna()
            valid_sell = monthly_returns.reindex(sell.index).dropna()

            # Recalculate weights based on surviving stocks
            buy_weights = buy.loc[valid_buy.index, 'weight']
            sell_weights = sell.loc[valid_sell.index, 'weight']

            buy_weights /= buy_weights.sum()  # Normalize weights to sum to 1
            sell_weights /= sell_weights.sum()

            # Compute weighted returns
            buy_return = (buy_weights * valid_buy).sum()
            buy_stdev = (buy_weights**2 * valid_buy**2).sum()**0.5
            sell_return = (sell_weights * valid_sell).sum()
            sell_stdev = (sell_weights**2 * valid_sell**2).sum()**0.5

            portfolio_return = buy_return - sell_return
            
            results["returns"].append(portfolio_return * 100)
            results["longs"].append(buy_return * 100)
            results["shorts"].append(sell_return * 100)
            results["longs_stdev"].append(buy_stdev * 100)
            results["shorts_stdev"].append(sell_stdev * 100)
            results['stnum'].append(len(buy) + len(sell))

            color = '\033[92m' if portfolio_return > 0 else '\033[91m'
            reset = '\033[0m'
            # print(f"{current_date} {next_date} {color}{round(portfolio_return * 100, 2)}%{reset}")
            
            current_date = next_date
            test *= 1 + portfolio_return

        start_date = end_date
        # break

    # Save results as JSON
    file_name = f"./result/baseline_{'VW' if VW else 'EW'}_{'skip' if SKIP else 'noskip'}.json"
    with open(file_name, "w") as f:
        json.dump(results, f, indent=4)

    # Final output of settings and average return
    print(f"Average return: {round(np.nanmean(results['returns']), 2)}%")
    print(f"Time taken: {time.time() - start} seconds")