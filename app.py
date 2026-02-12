import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(layout="wide", page_title="XGB Trading System Dashboard")

MODELS = Path("Models_WF")
RESULTS = Path("Results_WF/metrics")
GRAPHS = Path("Graphs_WF")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    demo_mode = not (MODELS / "wf_results.pkl").exists()

    if demo_mode:
        # ======================
        # DEMO DATA
        # ======================

        df_metrics = pd.DataFrame({
            "Ticker": ["AAPL", "SPY", "NVDA"],
            "Final Capital": [82000, 22500, 150000],
            "Total Return %": [720, 125, 1400],
            "CAGR %": [33, 11, 44],
            "Max DD %": [-18, -20, -37],
            "Sharpe": [2.9, 1.6, 2.2],
            "MAR": [1.8, 0.56, 1.2],
            "Expectancy": [0.0028, 0.0010, 0.0044],
            "Trades": [310, 280, 295]
        })

        df_comp_aligned = pd.DataFrame({
            "Ticker": ["AAPL", "SPY", "NVDA"],
            "Modelo": ["XGB", "XGB", "XGB"],
            "CAGR %": [33, 11, 44],
            "Max DD %": [-18, -20, -37],
            "Sharpe": [2.9, 1.6, 2.2],
            "Exposure %": [28, 31, 26],
        })

        df_comp_aligned["Capital Efficiency"] = (
            df_comp_aligned["CAGR %"] /
            df_comp_aligned["Exposure %"]
        )

        df_comp = df_comp_aligned.copy()

        df_trade_stats = pd.DataFrame({
            "Win Rate %": [54],
            "Avg Win": [0.021],
            "Avg Loss": [-0.015],
            "Payoff Ratio": [1.4],
            "Expectancy": [0.0025]
        })

        all_wf = None

    else:
        # ======================
        # REAL DATA (LOCAL ONLY)
        # ======================
        all_wf = joblib.load(MODELS / "wf_results.pkl")
        df_metrics = joblib.load(RESULTS / "wf_metrics.pkl")
        df_comp = joblib.load(RESULTS / "comparison.pkl")
        df_comp_aligned = joblib.load(RESULTS / "comparison_aligned.pkl")
        df_trade_stats = joblib.load(RESULTS / "trade_stats.pkl")

    return all_wf, df_metrics, df_comp, df_comp_aligned, df_trade_stats

# 🔹 SIEMPRE SE DEFINEN
all_wf, df_metrics, df_comp, df_comp_aligned, df_trade_stats = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Panel de Control")
section = st.sidebar.radio(
    "Secciones",
    [
        "📊 Visión General del Sistema",
        "📈 Robustez Walk Forward",
        "⚖️ Sistema vs Buy & Hold",
        "🧠 Microestructura de Trades",
        "📉 Distribución Estadística",
        "📂 Curvas de Equity Alineadas",
        "📑 Conclusiones del Sistema"
    ]
)

# ==========================================================
# 1️⃣ VISIÓN GENERAL
# ==========================================================
if section == "📊 Visión General del Sistema":
    st.title("Sistema de Trading Cuantitativo basado en XGBoost")

    st.markdown("""
### 🎯 Hipótesis de Investigación

El objetivo no es maximizar beneficio absoluto, sino diseñar un sistema **robusto y consistente**, priorizando:

- Control del riesgo  
- Reducción del drawdown  
- Menor exposición estructural al mercado  
- Estabilidad de la curva de capital  

Se evalúa el sistema bajo un marco de **Walk Forward**, simulando condiciones reales de uso.
""")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CAGR Medio", f"{df_metrics['CAGR %'].mean():.2f}%")
    col2.metric("Sharpe Medio", f"{df_metrics['Sharpe'].mean():.2f}")
    col3.metric("Trades Totales", int(df_metrics['Trades'].sum()))
    col4.metric("Max DD Medio", f"{df_metrics['Max DD %'].mean():.2f}%")

    st.markdown("### Equity Global Normalizada")
    st.image(GRAPHS / "eq_global_normalized.png", use_container_width=True)

# ==========================================================
# 2️⃣ WALK FORWARD
# ==========================================================
elif section == "📈 Robustez Walk Forward":
    st.title("Validación Walk Forward")

    st.markdown("""
El Walk Forward permite evaluar el sistema como si se operara en tiempo real:

- Entrenamiento → Validación → Test
- Ventanas deslizantes
- Sin fuga de información futura

### Métricas Clave
| Métrica | Qué mide |
|--------|----------|
| CAGR | Crecimiento anual medio |
| Max DD | Riesgo real sufrido |
| Sharpe | Retorno ajustado a volatilidad |
| MAR | Retorno ajustado a drawdown |
| Expectancy | Ventaja estadística por trade |
""")

    st.dataframe(df_metrics, use_container_width=True)

# ==========================================================
# 3️⃣ COMPARACIÓN
# ==========================================================
elif section == "⚖️ Sistema vs Buy & Hold":
    st.title("Comparación contra el Mercado")

    st.markdown("""
Aquí se valida la hipótesis principal:

> *¿Puede el sistema mantener retornos comparables con menor riesgo estructural?*

### Métricas Diferenciales
| Métrica | Interpretación |
|--------|----------------|
| Exposure | % del capital realmente expuesto al mercado |
| Capital Efficiency | Retorno generado por unidad de capital realmente invertido |

Estas métricas miden **calidad del riesgo**, no solo rentabilidad.
""")

    st.subheader("Periodo Completo")
    st.dataframe(df_comp, use_container_width=True)

    st.subheader("Periodo Alineado")
    st.dataframe(df_comp_aligned, use_container_width=True)

# ==========================================================
# 4️⃣ TRADES
# ==========================================================
elif section == "🧠 Microestructura de Trades":
    st.title("Análisis de la Estructura de Operaciones")

    st.markdown("""
Este bloque analiza la consistencia interna del sistema.

| Métrica | Significado |
|---------|------------|
| Win Rate | % operaciones ganadoras |
| Avg Win / Avg Loss | Tamaño medio de ganancias y pérdidas |
| Payoff Ratio | Relación beneficio/riesgo por trade |
| Expectancy | Ventaja estadística media |
""")

    st.dataframe(df_trade_stats, use_container_width=True)

# ==========================================================
# 5️⃣ DISTRIBUCIÓN
# ==========================================================
elif section == "📉 Distribución Estadística":
    st.title("Distribución de Retornos por Trade")
    st.image(GRAPHS / "trade_return_distribution.png", use_container_width=True)

# ==========================================================
# 6️⃣ EQUITY ALINEADA
# ==========================================================
elif section == "📂 Curvas de Equity Alineadas":
    st.title("Curvas de Capital Alineadas Temporalmente")

    global_graph = GRAPHS / "equity_aligned_global_dates.png"
    if global_graph.exists():
        st.image(global_graph, use_container_width=True)

    aligned_graphs = sorted(GRAPHS.glob("equity_aligned_dates_*.png"))
    for g in aligned_graphs:
        st.image(g, caption=g.name.replace("equity_aligned_dates_", "").replace(".png", ""), use_container_width=True)
# ==========================
# 7️⃣ CONCLUSIONES 
# =========================

elif section == "📑 Conclusiones del Sistema":
    st.title("📑 Conclusiones Cuantitativas del Sistema")

    avg_cagr = df_metrics["CAGR %"].mean()
    avg_sharpe = df_metrics["Sharpe"].mean()
    avg_dd = df_metrics["Max DD %"].mean()
    total_trades = df_metrics["Trades"].sum()

    avg_exposure = df_comp_aligned[df_comp_aligned["Modelo"]=="XGB"]["Exposure %"].mean()
    avg_efficiency = df_comp_aligned[df_comp_aligned["Modelo"]=="XGB"]["Capital Efficiency"].mean()

    st.markdown(f"""
### 📊 Rendimiento Global

El sistema presenta un **CAGR medio del {avg_cagr:.2f}%** con un **Sharpe medio de {avg_sharpe:.2f}**, 
lo que indica generación consistente de retornos ajustados a la volatilidad.

El **drawdown medio ({avg_dd:.2f}%)** se mantiene controlado, alineado con el objetivo de estabilidad del modelo.

---

### ⚙️ Eficiencia del Capital

El modelo solo mantiene capital expuesto al mercado un **{avg_exposure:.1f}% del tiempo**, 
frente al 100% de Buy & Hold.

A pesar de esta menor exposición, logra una **eficiencia del capital de {avg_efficiency:.2f}**, 
mostrando que los retornos se obtienen con menor riesgo estructural.

---

### 🧠 Consistencia Operativa

Se ejecutaron **{int(total_trades)} operaciones**, indicando que el rendimiento no depende de pocos eventos extremos,
sino de una ventaja estadística distribuida.

---

### 🎯 Conclusión Final

✔ Reducción estructural de exposición  
✔ Control del drawdown  
✔ Retornos ajustados al riesgo superiores a Buy & Hold  
✔ Consistencia estadística en la microestructura  

**Resultado:** Sistema cuantitativo robusto, orientado a estabilidad más que a volatilidad.
""")
