# Momentum Trading
Implementation of momentum trading using data from 2003 to 2024

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

Momentum-Trading/
├── benchmark_data/  
│   ├── market.CSV  
│   ├── stnum.CSV  
│   └── VW_benchmark.CSV  
├── data/  
│   ├── IBES_eps_quarters_2003_2024.csv  
│   └── stock_price_monthly_2003_2024.csv

### 🐍 For pip + virtualenv users:
```bash
python -m venv env_fin
source env_fin/bin/activate  # or `env_fin\Scripts\activate` on Windows
pip install -r requirements.txt
```
## ▶️ How to Run

### Step 1: Preprocessing

Execute the following Jupyter notebooks in order to clean and prepare the data:

1. Open `clean.ipynb` and run all cells.
   > 💡 Make sure to select the `env_fin` Python kernel. If you don't see it listed, try restarting VS Code and reactivating the `env_fin` environment.

2. Open `baseline_preprocess.ipynb` and run all cells.
3. Open `proxy_preprocess.ipynb` and run all cells.

You can run these in Jupyter Notebook or any IDE that supports `.ipynb` files (e.g., VSCode, JupyterLab).

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

