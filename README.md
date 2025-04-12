# Momentum Trading
Implementation of momentum trading using U.S. market data from 2003 to 2024

More details in [Presentation slide](https://docs.google.com/presentation/d/12ly-klGXJP00pRBN3lptlji4UzezfFZwZJq_PUALpLU/edit?slide=id.g33e3490a09c_3_267#slide=id.g33e3490a09c_3_267)

## 🧠 Implementation Details

The **baseline momentum strategy** examines the past 1-year return of all publicly traded stocks in the U.S. market. It then takes a **long position** in the top 20% of performers and a **short position** in the bottom 20%. The portfolio is **value-weighted** and **rebalanced every 6 months** (configurable)

The **modified strategy** builds on this baseline by introducing an additional filter: it only selects stocks in the **bottom 10% of analyst coverage** (based on lagged 3-6 months residual analyst coverage) from both the long and short portfolios. This refinement results in **superior returns**.

![modified](<./assets/modified.png>)

### 📊 Residual Analyst Coverage

Analyst coverage is quantified using **Residual Analyst Coverage (ResAnalyst)**, calculated as:

`ResAnalyst = Actual Number of Analysts - Predicted (from Linear Regression)`

A linear regression is fitted to model expected analyst coverage based on firm characteristics, and the residuals from this model represent the unexplained portion — i.e., **information diffusion speed** as stated by Hong, Lim and Stein (2000).

Model 0: 
`ResAnalyst = Actual Number of Analysts` (no regression)

Model 1: 
`log(1 + Predicted) = α + β₁ × log(Size) + β₂ × Nasdaq + ε`

Model 7: 
`log(1 + Predicted) = α + β₁ × log(Size) + β₂ × Nasdaq + β₃ × 1/P + β₄ × Variance + β₅ × R₀ + β₆ × R₁ + β₇ × R₂ + β₈ × R₃ + β₉ × R₄ + ε`

Model 8:
`log(1 + Predicted) = α + β₁ × log(Size) + β₂ × Nasdaq + β₃ × Turnover + β₅ × Turnover × Nasdaq + ε`

- `log(1 + Predicted)`: Natural log of one plus the number of predicted analysts covering the firm.
- `log(Size)`: Natural log of the firm's market capitalization.
- `Nasdaq`: Dummy variable equal to 1 if the firm is listed on Nasdaq; 0 otherwise.
- `ε`: Error term.
- `1/P`: 1 / Stock Price
- `Variance`: Variance of the firm's monthly returns for the past year.
- `R₀ to R₄`: Annual returns lagged by 0 to 4 years, respectively.
- `Turnover`: Ratio of trading volume over the prior six months to total shares outstanding.

## Setup Instructions

### Compatibility
python >=3.9,<=3.12

### Clone the repo
```bash
git clone https://github.com/xuanvietchu/Momentum-Trading.git
cd Momentum-Trading
```

### 📥 Download Data

Download the required dataset from [this Google Drive folder](https://drive.google.com/drive/folders/1oNgrl9AGhgINerKbnNA3SlnJty-CsLpH?usp=sharing).

Before placing the files, make sure to create the following folders in the root directory of the project:
```bash
mkdir benchmark_data
mkdir data
mkdir models_data
mkdir processed_data
mkdir result
```

After downloading, place the files into their respective directories so that your project structure resembles the following:

```bash
Momentum-Trading/
├── benchmark_data/  
│   ├── market.CSV  
│   ├── stnum.CSV  
│   └── VW_benchmark.CSV  
├── data/  
│   ├── IBES_eps_quarters_2003_2024.csv  
│   └── stock_price_monthly_2003_2024.csv
```

### 🐍 For pip + virtualenv users:
```bash
python -m venv env_fin
source env_fin/bin/activate  # or `env_fin\Scripts\activate` on Windows
pip install -r requirements.txt
```
## ▶️ How to Run

### Step 1: Preprocessing

Execute the following Jupyter notebooks in order to clean and prepare the data:

1. Open `clean.ipynb` from the `preprocess` folder and run all cells.
   > 💡 Make sure to select the `env_fin` Python kernel. If you don't see it listed, try reopening the `Momentum-Trading` folder with VSCode and reactivating the `env_fin` environment.

2. Open `baseline_preprocess.ipynb` and run all cells.
3. Open `proxy_preprocess.ipynb` and run all cells.

You can run these in Jupyter Notebook or any IDE that supports `.ipynb` files (e.g., VSCode, JupyterLab).

If the notebooks run successfully, your project directory should now include:
```bash
Momentum-Trading/
├───models_data/
│       ibes_eps_semi_annual_by_date_2003_2024.csv
│       stock_price_monthly_2003_2024.csv
```
---


### Step 2: Configure the Model (Optional)

If needed, customize parameters or strategy logic by editing the model's configuration:
- `config/` folder for constants and parameters

---

### Step 3: Run Strategies

Run the following Python scripts to execute the trading strategies:

```bash
# Run baseline strategy
python ./models/baseline.py

# Run modified strategy
python ./models/modified.py

# Run benchmark
python ./benchmark/benchmark.py

# Run market
python ./benchmark/market.py
```

### Step 4: Statistical Testing

Use these scripts to evaluate strategy performance with statistical tests:
```bash
# Paired t-test between strategies
python ./t_test/t_test.py

# Compare strategy to benchmark
python ./t_test/test_benchmark.py

# Compare screening variables
python ./t_test/v1_vs_v2.py
```

### 🙏 Acknowledgements

Special thanks to **Professor Akiko Watanabe** for her guidance and support throughout the course. This project would not have been possible without her mentorship.

## Sources

Jegadeesh and Titman, “Returns to Buying Winners and Selling Losers,” Journal of Finance 1993. 

Hong, Harrison, and Jeremy C. Stein. "A unified theory of underreaction, momentum trading, and overreaction in asset markets." The Journal of finance 54.6 (1999): 2143-2184.

Daniel, Kent, and Tobias J. Moskowitz. "Momentum crashes." Journal of Financial economics 122.2 (2016): 221-247.

Hou, Kewei, Chen Xue, and Lu Zhang. "Replicating anomalies." The Review of financial studies 33.5 (2020): 2019-2133.

Hong, Harrison, Terence Lim, and Jeremy C. Stein. "Bad news travels slowly: Size, analyst coverage, and the profitability of momentum strategies." The Journal of finance 55.1 (2000): 265-295.

Goyal, Amit, Narasimhan Jegadeesh, and Avanidhar Subrahmanyam. "What explains momentum? a perspective from international data." Working Paper. 2022.
