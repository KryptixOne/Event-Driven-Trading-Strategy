# 📈 Event-Driven Trading Strategy Using Mention Spikes & Price Momentum

## **Overview**

This repository contains a **quantitative trading framework** designed to identify **profitable trading signals** based on **mention spikes**, **price momentum**, and now **classical indicators** like RSI, MACD, CVD, and Chaikin Oscillator. The framework supports **hyperparameter tuning**, **visual signal analysis**, and **backtesting** for **event-driven trading strategies**.

Data available here: [📂 Google Drive](https://drive.google.com/drive/folders/1-npp29XExE_BhdrMnJKI1Glr_N8kU9Yh)

---

## **🚀 Key Features**

- **📊 Hyperparameter Optimization**
- **📈 Multi-Horizon Event Study**
- **📌 Statistical & Performance Metrics**
- **🔎 Signal Filtering for Trading**
- **🧠 Classical Technical Indicator Studies** (RSI, MACD, CVD, Chaikin)
- **📊 Data Visualization & Insights**
- **⏱️ Backtesting Interface** to simulate and visualize event signals

---

## **🧠 New Additions**

- 🔁 **Backtesting Framework**: Simulate event-based trades using `run_study.py` and `symbol_data.py`
- 🧮 **Classical Trading Studies**: Add custom indicator studies to augment or replace mention-driven events
- 🖼️ **Visual Signal Explorer**: Use `plot_fcn.py` to plot annotated trading signals

### 🔍 Example Signal Output

<p align="center">
  <img src="docs/Example%20Signals%20and%20Indicators.png" alt="Example Signal Plot" width="700"/>
</p>

---

## **📂 Repository Structure**



```
./data/                     # Datasets (QuiverQuant, Yahoo Finance, Mentions, etc.)
./event_study/              # Event study logic and hyperparameter search
    ├── event_studies.py
./preproc/                  # Data preprocessing
./utils/                    # Utility functions (plotting, stats)
./UpdateMentionData.py      # Script to update data sources

./backtesting/              # Backtesting framework
    ├── plotting/
        └── plot_fcn.py     # Signal plotting logic
    ├── run_study.py        # Execute backtest on chosen signal definitions
    └── symbol_data.py      # Symbol-level data abstraction

./classical_trading/        # Classical technical studies
    ├── custom_studies/
        └── rsi_macd_cvd_chai_custom_study.py
    └── indicators/
        ├── base_indicators_tos.py
        └── custom_indicators.py

./docs/
    └── Example Signals and Indicators.png

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

