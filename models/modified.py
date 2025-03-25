import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import json

def load_data():
    # Load and preprocess df_proxy
    df_proxy = pd.read_csv('./baseline_data/ibes_eps_semi_annual_by_date_2003_2024.csv').drop(columns=['Year'])
    df_proxy['FPEDATS'] = pd.to_datetime(df_proxy['FPEDATS']).dt.strftime('%Y-%m')
    
    # Load and preprocess df
    df = pd.read_csv('./baseline_data/stock_price_monthly_2003_2024.csv').dropna(subset=["past_return", "past_return_skip"])
    df = df.set_index(['date', 'NCUSIP']).sort_index()

    # remove bottom 20% of market cap
    # df = df[df['ME'] > df['ME'].quantile(0.1)]
    
    # Pre-index formation data by date for fast lookup
    dates = df.index.get_level_values('date').unique()
    df_by_date = {date: df.loc[date] for date in dates}
    
    return df_proxy, df, df_by_date

def run_model(df_proxy, df, df_by_date, VW, model_no, reversed = False, start_time = 0):
    SKIP = False           # Skip flag
    
    # Configuration parameters
    start_date = '2004-01'
    rebalance_period = 6   # in months

    
    results = {
        "returns": [],
        "longs": [],
        "shorts": [],
        "longs_stdev": [],
        "shorts_stdev": [],
        "stnum": [],
    }
    R2s = []
    corr_buys = []
    corr_sells = []
    
    while start_date in df_by_date:
        # Define holding period end date
        end_date = (pd.to_datetime(start_date) + pd.DateOffset(months=rebalance_period)).strftime('%Y-%m')
        
        # --- Formation Period ---
        formation_data = df_by_date[start_date].copy()
        sort_col = "past_return_skip" if SKIP else "past_return"
        formation_data['portfolio'] = np.where(
            formation_data[sort_col] >= np.percentile(formation_data[sort_col], 80), 'W',  # Winners are the top 10%
            np.where(formation_data[sort_col] <= np.percentile(formation_data[sort_col], 20), 'L', 'N')  # Losers are the bottom 10%
        )
        
        # Extract buy and sell groups and reset index for merging later
        buy = formation_data[formation_data['portfolio'] == 'W'].reset_index().copy()
        sell = formation_data[formation_data['portfolio'] == 'L'].reset_index().copy()
        
        # Unique identifier pairs from formation data
        unique_pairs = formation_data.reset_index()[['TICKER', 'NCUSIP']].drop_duplicates()
        
        # --- Proxy Data Processing ---
        # Filter proxy data for the formation period 
        # ensuring that measurement of analyst coverage isn't directly influenced by the stock's recent performance.
        _6_months_before = (pd.to_datetime(start_date) - pd.DateOffset(months=6)).strftime('%Y-%m')

        proxy_period = df_proxy[(df_proxy['FPEDATS'] >= _6_months_before) & (df_proxy['FPEDATS'] < start_date)].copy()
        proxy_period = proxy_period.drop(columns=['FPEDATS', 'FPI'], errors='ignore')

        numest_total = proxy_period['NUMEST'].sum()
        
        # Identify missing identifier pairs and add them with NUMEST = 0
        merged = unique_pairs.merge(proxy_period, left_on=['TICKER', 'NCUSIP'],
                                    right_on=['OFTIC', 'CUSIP'], how='left', indicator=True)
        missing_pairs = merged[merged['_merge'] == 'left_only'][['TICKER', 'NCUSIP']] \
                        .rename(columns={'TICKER': 'OFTIC', 'NCUSIP': 'CUSIP'})
        missing_pairs['NUMEST'] = 0
        proxy_period = pd.concat([proxy_period, missing_pairs], ignore_index=True)
        proxy_period = proxy_period.dropna(subset=['OFTIC', 'CUSIP'])
        
        # Merge to get additional info (market cap and Nasdaq dummy)
        formation_info = formation_data.reset_index()[['NCUSIP', 'TICKER',
                                                        'ME', 'NasdaqDummy',
                                                        'past_variance', 'PRC',
                                                        'past_return', 'R1', 'R2', 'R3', 'R4']]
        proxy_period = proxy_period.merge(
            formation_info,
            left_on=['CUSIP', 'OFTIC'],
            right_on=['NCUSIP', 'TICKER'],
            how='left'
        )
        proxy_period.drop(columns=['NCUSIP', 'TICKER'], inplace=True)

        if model_no == 1:
            proxy_period = proxy_period.dropna(subset=['ME'])
        elif model_no == 7:
            proxy_period = proxy_period.dropna(subset=['ME', 'past_variance', 'past_return'])
            # 'R1', 'R2', 'R3', 'R4' impute with 0
            proxy_period[['R1', 'R2', 'R3', 'R4']] = proxy_period[['R1', 'R2', 'R3', 'R4']].fillna(0)


        # --- Regression ---
        # Model: ln(1 + # Analysts) = β0 + β1 * ln(ME) + β2 * NasdaqDummy
        proxy_period['ln_analysts'] = np.log1p(proxy_period['NUMEST'])
        proxy_period['ln_ME'] = np.log(proxy_period['ME'])

        if model_no == 1:
            X = proxy_period[['ln_ME', 'NasdaqDummy']]
        elif model_no == 7:
            proxy_period['1/P'] = 1 / abs(proxy_period['PRC'])
            X = proxy_period[['ln_ME', 'NasdaqDummy', '1/P', 'past_variance', 'past_return', 'R1', 'R2', 'R3', 'R4']]

        y = proxy_period['ln_analysts']
        reg = LinearRegression().fit(X, y)
        # beta0 = reg.intercept_
        # beta1, beta2, beta3 = reg.coef_
        y_pred = np.maximum(reg.predict(X), 0)
        proxy_period["predicted_ln_analysts"] = y_pred
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        R2s.append(r2)
        
        # --- Visualization ---
        # plt.figure(figsize=(10, 6))
        # sns.scatterplot(x=proxy_period['ln_ME'], y=proxy_period['ln_analysts'],
        #                 hue=proxy_period['NasdaqDummy'], alpha=0.7, palette='coolwarm')
        # ln_ME_range = np.linspace(proxy_period['ln_ME'].min(), proxy_period['ln_ME'].max(), 100)
        # # Nasdaq regression line (NasdaqDummy = 1)
        # plt.plot(ln_ME_range, np.maximum(beta0 + beta1 * ln_ME_range + beta2, 0),
        #          color='orange', linewidth=2, label="Nasdaq Regression")
        # # Non-Nasdaq regression line (NasdaqDummy = 0)
        # plt.plot(ln_ME_range, np.maximum(beta0 + beta1 * ln_ME_range, 0),
        #          color='blue', linewidth=2, label="Non-Nasdaq Regression")
        # textstr = (rf"$\ln(1+\#\text{{Analysts}}) = {beta0:.4f} + {beta1:.4f}\ln(\text{{Size}}) + {beta2:.4f}\,\text{{Nasdaq}}$"
        #            f"\nRMSE = {rmse:.4f}\nR2 = {r2:.4f}")
        # plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
        #          verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
        # plt.xlabel("ln(Size) (Market Capitalization)")
        # plt.ylabel("ln(1 + # of Analysts)")
        # plt.title(f"Linear Regression: Analysts vs. Market Capitalization {start_date} - {end_date}")
        # plt.legend(title="Nasdaq Dummy", loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
        # plt.grid(True)
        # plt.show()
        
        # --- Residual Calculation & Portfolio Construction ---
        proxy_period["residuals"] = proxy_period["NUMEST"] - np.expm1(proxy_period["predicted_ln_analysts"])
        
        # Merge residuals into buy and sell data
        buy = buy.merge(proxy_period[['OFTIC', 'CUSIP', 'residuals']],
                        left_on=['NCUSIP', 'TICKER'], right_on=['CUSIP', 'OFTIC'], how='left') \
                 .drop(columns=['CUSIP', 'OFTIC'])
        sell = sell.merge(proxy_period[['OFTIC', 'CUSIP', 'residuals']],
                          left_on=['NCUSIP', 'TICKER'], right_on=['CUSIP', 'OFTIC'], how='left') \
                   .drop(columns=['CUSIP', 'OFTIC'])
        buy = buy.dropna(subset=['residuals'])
        sell = sell.dropna(subset=['residuals'])
        
        if reversed:
            buy['portfolio'] = np.where(buy['residuals'] >= np.percentile(buy['residuals'], 90), 'W', 'N')
            sell['portfolio'] = np.where(sell['residuals'] >= np.percentile(sell['residuals'], 90), 'L', 'N')
        else:   
            buy['portfolio'] = np.where(buy['residuals'] <= np.percentile(buy['residuals'], 10), 'W', 'N')
            sell['portfolio'] = np.where(sell['residuals'] <= np.percentile(sell['residuals'], 10), 'L', 'N')
        
        # Print the number of selected stocks
        # print(f"Number of stocks in Buy Portfolio (W): {sum(buy['portfolio'] == 'W')}")
        # print(f"Number of stocks in Sell Portfolio (L): {sum(sell['portfolio'] == 'L')}")

        buy = buy[buy['portfolio'] == 'W'].copy()
        sell = sell[sell['portfolio'] == 'L'].copy()
        
        # Assign weights (equal-weighted or value-weighted)
        if VW:
            buy['weight'] = buy['ME'] / buy['ME'].sum()
            sell['weight'] = sell['ME'] / sell['ME'].sum()
        else:
            buy['weight'] = 1 / len(buy)
            sell['weight'] = 1 / len(sell)
        buy = buy.set_index("NCUSIP")
        sell = sell.set_index("NCUSIP")
        
        # compare residuals with returns of 6 months later
        buy['RET_temp'] = buy['RET'].shift(-6)
        sell['RET_temp'] = sell['RET'].shift(-6)
        corr_buy = buy['residuals'].corr(buy['RET_temp'])
        corr_sell = sell['residuals'].corr(sell['RET_temp']) 
        # drop
        buy = buy.drop(columns=['RET_temp'])
        sell = sell.drop(columns=['RET_temp'])

        corr_buys.append(corr_buy)    
        corr_sells.append(corr_sell)  
        
        current_date = start_date

        # --- Compute Portfolio Returns ---
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
        
        start_date = end_date  # Move to the next period
    
    # --- Summary ---
    print("Value-weighted" if VW else "Equal-weighted")
    print("Skip" if SKIP else "No Skip")
    print(f"Average return: {round(np.nanmean(results['returns']), 2)}%")
    print(f"Time taken: {time.time() - start_time} seconds")

    # value-weighted and skip
    file_name = f"modified_{'VW' if VW else 'EW'}_{'skip' if SKIP else 'noskip'}_model{model_no}_{'reversed' if reversed else ''}.json"
    with open(f"./result/{file_name}", "w") as f:
        json.dump(results, f, indent=4)
    
    
    print(f"Completed: VW={VW}, Model={model_no}, Reversed={reversed}")

    return R2s, corr_buys, corr_sells

def run_all(start_time):
    df_proxy, df, df_by_date = load_data()
    
    R = []
    CB = []
    CS = []
    start_date = '2004-01'
    dates = []
    dates.append(start_date)
    while start_date in df_by_date:
        end_date = (pd.to_datetime(start_date) + pd.DateOffset(months=6)).strftime('%Y-%m')
        dates.append(end_date)
        start_date = end_date
    for VW in [True]:
        for model_no in [1, 7]:
            for reversed in [False]:
                R2s, corr_buys, corr_sells  = run_model(df_proxy, df, df_by_date, VW, model_no, reversed, start_time)
                R.append(R2s)
                CB.append(corr_buys)
                CS.append(corr_sells)
    
    # plot R2s
    plt.figure(figsize=(10, 6))
    for i, R2s in enumerate(R):
        plt.plot(dates[1:], R2s, label=f"VW, Model {1 if i ==0 else 7}")
    plt.xlabel("Period")
    plt.ylabel("R2")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.legend()
    plt.title("R2 for models over time")
    plt.savefig('./result/R2.png')
    plt.show()
    # save 

    # plot corr_buys
    # plt.figure(figsize=(10, 6))
    # for i, corr_buys in enumerate(CB):
    #     plt.plot(dates[1:], corr_buys, label=f"VW, Model {1 if i ==0 else 7}")
    #     print(f"Model {1 if i ==0 else 7}: ", np.mean(corr_buys))
    # plt.xlabel("Period")
    # plt.ylabel("Correlation")
    # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # plt.legend()
    # plt.title("Correlation between residuals and returns for buys")
    # plt.show()

    # plot corr_sells
    # plt.figure(figsize=(10, 6))
    # for i, corr_sells in enumerate(CS):
    #     plt.plot(dates[1:], corr_sells, label=f"VW, Model {1 if i ==0 else 7}")
    #     print(f"Model {1 if i ==0 else 7}: ", np.mean(corr_sells))
    # plt.xlabel("Period")
    # plt.ylabel("Correlation")
    # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # plt.legend()
    # plt.title("Correlation between residuals and returns for sells")
    # plt.show()

if __name__ == "__main__":
    start_time = time.time()
    # df_proxy, df, df_by_date = load_data()

    # run_model(df_proxy, df, df_by_date, True, 1, False, start_time)
    # run_model(df_proxy, df, df_by_date, False, 1, False, start_time)
    # run_model(df_proxy, df, df_by_date, True, 7, False, start_time)
    # run_model(df_proxy, df, df_by_date, False, 7, False, start_time)

    run_all(start_time)
