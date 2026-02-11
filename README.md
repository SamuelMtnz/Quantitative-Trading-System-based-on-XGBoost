# 📊 Quantitative Trading System based on XGBoost

Machine Learning trading system designed to prioritize **risk control, capital efficiency, and robustness** over raw returns.

This project explores whether an ML-driven strategy can achieve **comparable returns to Buy & Hold** while significantly reducing market exposure and drawdowns.

---

## 🎯 Research Hypothesis

A **robust trading system** should:

- Reduce **market exposure**
- Control **drawdowns**
- Produce **stable equity growth**
- Optimize **risk-adjusted metrics** (Sharpe, MAR)
- Focus on **consistency over volatility**

The goal is **not to maximize profit**, but to **lose less during adverse market regimes**.

---

## 🧠 Model

- **Algorithm:** XGBoost Classifier  
- **Training Method:** Walk Forward Optimization  
- **Prediction Target:** Directional market movement  
- **Position Sizing:** Fractional capital exposure  
- **Risk Approach:** Volatility-aware and capital preservation focused  

---

## 🔄 Walk Forward Framework

The system uses a rolling Walk Forward structure:

```
Train → Test → Slide Window → Retrain
```

This avoids look-ahead bias and simulates live deployment conditions.

---

## 📈 Performance Philosophy

Unlike traditional systems that maximize CAGR, this system optimizes:

| Metric | Purpose |
|--------|--------|
| **Sharpe Ratio** | Return per unit of volatility |
| **MAR Ratio** | Return relative to max drawdown |
| **Max Drawdown** | Capital preservation |
| **Exposure %** | Time in market |
| **Capital Efficiency** | Return achieved per unit of exposure |

---

## ⚖️ Why Compare vs Buy & Hold?

Buy & Hold has:
- Full exposure
- High volatility
- Large drawdowns

Our system aims to achieve **similar returns** with:
- Lower exposure
- Lower drawdown
- Higher risk efficiency

---

## 🖥️ Dashboard

A Streamlit dashboard allows full exploration of:

- Walk Forward metrics  
- XGB vs Buy & Hold comparison  
- Trade-level analytics  
- Equity curve analysis  

Run locally:

```bash
streamlit run dashboard.py
```

---

## 📂 Project Structure

```
Quant/
│
├── Scripts/   # Model training and WF pipeline
│   ├── 01_EDA.py
│   ├── 02_FE.py        
│   └── 03_WF.py           # (ignored in Git)           
├── app.py                 # Streamlit dashboard
├── Models_WF/             # Saved WF models (ignored in Git)
├── Results_WF/            # Metrics and comparisons (ignored)
├── Graphs_WF/             # Generated charts (ignored)
├── README.md
├── .gitignore
└── reuirements.txt
```

---

## 🧪 Key Findings

✔ Comparable CAGR to Buy & Hold  
✔ Lower Max Drawdown  
✔ Reduced Market Exposure  
✔ Improved Capital Efficiency  
✔ Smoother equity curve  

The model behaves as a **risk-managed alternative** to passive investing.

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**.  
It does not constitute financial advice.

---

## 👤 Author

Samuel Martínez  
MSc Thesis Project — Quantitative Finance & Machine Learning
