# 📊 Quantitative Trading System based on XGBoost

**MSc Thesis Project — Quantitative Finance & Machine Learning**

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
Train → Validation → Test → Slide Window → Retrain
```

This structure avoids look-ahead bias and closely simulates real-world deployment conditions.

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

The focus is on **risk efficiency**, not raw return magnitude.

---

## ⚖️ Why Compare vs Buy & Hold?

Buy & Hold has:
- Full exposure
- High volatility
- Large drawdowns

This system aims to achieve **similar long-term return behavior** while offering:
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

## 🌐 Live Demo

Interactive dashboard available at:

https://quantitative-trading-system-based-on-xgboost.streamlit.app/


---
## 🌐 Public Demo Version

The online dashboard contains:

- Real equity curve visualizations  
- Demonstrative tabular metrics (non-sensitive values)  
- No trading signals  
- No trained models  
- No proprietary datasets  

This ensures **intellectual property protection** while preserving **methodological transparency**.

---
## 📂 Project Structure

```
Quantitative-Trading-System-based-on-XGBoost/
│
├── Scripts/   # Model training and WF pipeline
│   ├── 01_EDA.py
│   └── 02_FE.py                 
├── app.py                
├── public_graphs/ 
├── README.md
├── requirements.txt
└── .gitignore
```

---
## 🔒 Private Components (Not Included)

The following elements exist locally but are intentionally excluded:

- Walk Forward training pipeline  
- Feature engineering modules  
- Model training scripts  
- Trained model files  
- Raw backtesting datasets  
- Detailed performance outputs  

These components constitute the **core intellectual property** of the research.

---

## 🧪 Research Findings (Summary)

Backtesting results suggest that the system:

✔ Comparable CAGR to Buy & Hold  
✔ Lower Max Drawdown  
✔ Reduced Market Exposure  
✔ Improved Capital Efficiency  
✔ Smoother equity curve  

The model behaves as a **risk-managed alternative** to passive investing.

---

## ⚠️ Disclaimer

This project is provided for **research and educational purposes only**.  
It does **not** constitute financial advice or investment recommendations.

---

## 👤 Author

Samuel Martínez  
MSc Thesis Project — Quantitative Finance & Machine Learning

## 🔐 Intellectual Property Notice

The Walk Forward training pipeline, feature engineering methodology, model configurations, and raw backtesting data are intentionally withheld from the public repository.

This repository contains only the **demonstration dashboard** and a **high-level methodological overview**.