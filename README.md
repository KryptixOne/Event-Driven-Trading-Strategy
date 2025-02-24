# 📈 Event-Driven Trading Strategy Using Mention Spikes & Price Momentum

## **Overview**

This repository contains a **quantitative trading framework** designed to identify **profitable trading signals** based on **mention spikes** and **price momentum**. By systematically analyzing how unusual surges in social media mentions affect stock returns, this project aims to optimize **event-driven trading strategies**.

Data available here: https://drive.google.com/drive/folders/1-npp29XExE_BhdrMnJKI1Glr_N8kU9Yh

## **🚀 Key Features**

- **📊 Hyperparameter Optimization:**
  - Searches across different **mention window sizes**, **z-score thresholds**, and **minimum price increases** to define the best event signals.
- **📈 Multi-Horizon Event Study:**
  - Evaluates stock returns at **1, 5, 10, 20, 30, and 40-day** forward periods to measure short-term & long-term price reactions.
- **📌 Statistical & Performance Metrics:**
  - Computes **average return differences (**``**)**, **statistical significance (**``**)**, and **event count** to rank parameter sets.
- **🔎 Signal Filtering for Trading:**
  - Filters for the **strongest** mention-driven price movements with sustained effects to create a **momentum-based strategy**.
- **📊 Data Visualization & Insights:**
  - Generates **pretty tables & plots** to highlight the most effective event definitions.

## **🔬 How It Works**

1️⃣ **Define an Event:**

- A day is an **event day** if: ✅ Mentions **spike** above a rolling average (`mention_window` & `mention_z`). ✅ The stock price **increases by at least **X**%**.

2️⃣ **Test Different Hyperparameters:**

- Evaluates multiple `mention_window`, `mention_z`, and `price_up_min` values.
- Measures **forward stock returns** after event days vs. non-event days.

3️⃣ **Rank the Best Trading Signals:**

- Finds the **most profitable** (`avg_diff`) and **statistically significant** (low `p-value`) parameter sets.

## **💡 Key Use Cases**

- **Algorithmic Trading**: Identify social sentiment-driven trading opportunities.
- **Quantitative Research**: Understand how social media mentions influence stock prices.
- **Backtesting & Strategy Development**: Use historical data to optimize trade execution timing.

## **📂 Repository Structure**

```
./data/                # Holds all datasets (QuiverQuant, Yahoo Finance, Mentions, etc.)
./event_study/         # Event study functions, including hyperparameter search
    ├── event_studies.py
./preproc/             # Data preprocessing functions
./utils/               # Utility functions for plotting and OLS regression
./UpdateMentionData.py # Script to create & update data from QuiverQuant, Mentions, and Yahoo Finance
```

## **📌 Getting Started**

### **1️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2️⃣ Update the Data**

```bash
python UpdateMentionData.py
```

This will fetch and preprocess new mention and stock price data from QuiverQuant, Yahoo Finance, and other sources.

### **3️⃣ Run the Hyperparameter Search**

```python
from event_study.event_studies import hyperparam_search

df_results = hyperparam_search(df, ticker='AAPL')
```

### **4️⃣ View the Best Event Signals**

- **Check results**: The system will output a ranked table of mention spike configurations with the highest predictive power.
- **Visualize**: Use `utils/plotting.py` to generate plots of post-event return trajectories.

## **🔍 Future Work**

- Expand analysis to **multiple tickers**
- Integrate **alternative sentiment sources** (Twitter, Reddit, etc.)
- Develop **real-time signal monitoring** for trading execution
- Implement **backtesting framework** for strategy validation

## **📌 Contributions**

If you have ideas to enhance the strategy, feel free to **open a pull request**! 🚀

