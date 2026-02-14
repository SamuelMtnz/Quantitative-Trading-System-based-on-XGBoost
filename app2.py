import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from Scripts.f import predict

st.set_page_config(layout="wide", page_title="XGB Trading System Dashboard")



BASE_DIR = Path(__file__).resolve().parent
PUBLIC = BASE_DIR / "public_data"
GRAPHS = BASE_DIR / "public_graphs"


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    required_files = [
        "wf_results.pkl",
        "wf_metrics.pkl",
        "comparison.pkl",
        "comparison_aligned.pkl",
        "trade_stats.pkl"
    ]

    for f in required_files:
        if not (PUBLIC / f).exists():
            st.error(f"No se encontró el archivo requerido: {f}")
            st.stop()

    wf_results = joblib.load(PUBLIC / "wf_results.pkl")
    wf_metrics = joblib.load(PUBLIC / "wf_metrics.pkl")
    comparison = joblib.load(PUBLIC / "comparison.pkl")
    comparison_aligned = joblib.load(PUBLIC / "comparison_aligned.pkl")
    trade_stats = joblib.load(PUBLIC / "trade_stats.pkl")

    return wf_results, wf_metrics, comparison, comparison_aligned, trade_stats


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
        "🔮 Señal en Vivo",
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

Se evalúa el sistema bajo un marco de **Walk Forward**, simulando condiciones reales.
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


# ==========================================================
# 7️⃣ SEÑAL EN VIVO
# ==========================================================
elif section == "🔮 Señal en Vivo":

    st.title("🔮 Generador de Señal en Tiempo Real")

    st.markdown("""
El modelo final fue entrenado sobre todo el histórico validado mediante Walk Forward.

Se carga el modelo persistido y se evalúan las features actuales.

Además, se realiza una comparación alineada contra Buy & Hold
para evaluar si el modelo mejora la eficiencia del capital.
""")

    ticker = st.text_input("Introduce el ticker", value="AAPL")

    if st.button("Generar Señal"):

        with st.spinner("Calculando señal y métricas..."):

            try:
                result = predict(ticker.upper())

                prob = result["probability"]
                signal = result["signal"]
                comparison = result["comparison"]  # <- debes devolver esto desde predict()

                st.subheader(f"Resultado para {ticker.upper()}")

                col1, col2 = st.columns(2)

                col1.metric("Probabilidad LONG", f"{prob:.2%}")

                if signal == 1:
                    col2.success("Señal: LONG")
                else:
                    col2.warning("Señal: NO TRADE")

                st.markdown("---")
                st.subheader("📊 Comparativa vs Buy & Hold (Periodo Alineado)")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("CAGR Modelo", f"{comparison['model_cagr']:.2f}%")
                col2.metric("CAGR Buy & Hold", f"{comparison['bh_cagr']:.2f}%")
                col3.metric ("Sharpe Modelo", f"{comparison['model_sharpe']:.2f}")
                col4.metric("Sharpe Buy & Hold", f"{comparison['bh_sharpe']:.2f}")
               
                col5, col6, col7, col8 = st.columns(4)
                
                col5.metric("Exposure Modelo", f"{comparison['model_exposure']:.1f}%")
                col6.metric("Exposure Buy & Hold", "100%")
                col7.metric("DD Modelo", f"{comparison['model_max_dd']:.2f}%")                         
                col8.metric("DD Buy & Hold", f"{comparison['bh_max_dd']:.2f}%")
                
               
                if "equity_plot_path" in comparison:
                    st.image(comparison["equity_plot_path"], use_container_width=True)

            except Exception as e:
                st.error(f"Error generando señal: {e}")



# ==========================================================
# 8️⃣ CONCLUSIONES
# ==========================================================
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