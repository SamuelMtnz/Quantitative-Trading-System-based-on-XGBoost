# ======================================================
# FASE 1: EDA 
# ======================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('ggplot')

# =========================
# RUTAS DEL REPO
# =========================
GRAPH_PATH = "Graphs_EDA"
DATA_PATH = "data_EDA"

os.makedirs(GRAPH_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# =========================
# PARÁMETROS
# =========================
TICKERS = ["AAPL", "SPY", "NVDA"]
START = "2015-01-01"
ROLL_WINDOW = 20

# =========================
# FUNCIONES
# =========================

def download_data(tickers, start):
    """Descarga datos históricos y los guarda en CSV"""
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(os.path.join(DATA_PATH, f"{ticker}.csv"))
        data[ticker] = df
    return data

def eda_summary(df, ticker):
    """Muestra tipos de datos, nulos y estadísticas descriptivas"""
    print(f"=== {ticker} ===")
    print("Tipos y nulos:")
    print(df.info())
    print("\nEstadísticas descriptivas (numéricas):")
    print(df.describe())
    print("\nEstadísticas descriptivas (todas las columnas):")
    print(df.describe(include='all'))
    print("\n-----------------------------------\n")

def plot_individual_eda(df, ticker):
    """Crea figura 3x3 con métricas individuales de cada ticker"""
    df['ret_1'] = df['Close'].pct_change()
    df['vol_20'] = df['ret_1'].rolling(ROLL_WINDOW).std()
    df['equity'] = (1 + df['ret_1']).cumprod()
    df['drawdown'] = df['equity'] / df['equity'].cummax() - 1

    fig, axes = plt.subplots(3, 3, figsize=(20,15))
    fig.suptitle(f"EDA Compacto 3x3 - {ticker}", fontsize=18)

    axes[0,0].plot(df['Close'], color='blue')
    axes[0,0].set_title('Precio de Cierre Histórico')
    axes[0,0].set_ylabel('Precio ($)')

    axes[0,1].bar(df.index, df['Volume'], color='#6a0dad', alpha=1.0)
    axes[0,1].set_title("Volumen Diario")
    axes[0,1].set_ylabel("Volumen")

    axes[0,2].plot(df['ret_1'], color='orange')
    axes[0,2].set_title('Volatilidad Diaria (retornos)')
    axes[0,2].set_ylabel('Retorno diario')

    axes[1,0].plot(df['drawdown'], color='red')
    axes[1,0].set_title('Drawdown Histórico')
    axes[1,0].set_ylabel('Drawdown')

    corr_df = df[['Close','Volume','vol_20','drawdown']].dropna()
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1,1])
    axes[1,1].set_title("Correlaciones internas")

    axes[1,2].scatter(df['ret_1'], df['Volume'], alpha=0.5, color='#6a0dad')
    axes[1,2].set_title("Scatter Retorno vs Volumen")
    axes[1,2].set_xlabel("Retorno diario")
    axes[1,2].set_ylabel("Volumen")

    for ax in [axes[2,0], axes[2,1], axes[2,2]]:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(GRAPH_PATH, f"EDA_3x3_{ticker}.png")
    plt.savefig(fig_path)
    plt.close(fig)

def plot_comparative_eda(eda_data):
    """Crea figura comparativa 2x2 entre tickers"""
    fig, axes = plt.subplots(2, 2, figsize=(18,12))
    fig.suptitle("Comparativa Tickers - 2x2", fontsize=18)

    # Precio cierre + MA50
    for ticker, df in eda_data.items():
        df['close_ma50'] = df['Close'].rolling(50).mean()
        axes[0,0].plot(df['Close'], label=f"{ticker} Close")
        axes[0,0].plot(df['close_ma50'], label=f"{ticker} MA50")
    axes[0,0].set_title("Precio + MA50 Comparativa")
    axes[0,0].set_xlabel("Fecha")
    axes[0,0].set_ylabel("Precio ($)")
    axes[0,0].legend()

    # Distribución retornos
    for ticker, df in eda_data.items():
        axes[0,1].hist(df['ret_1'].dropna(), bins=50, alpha=0.5, label=ticker)
    axes[0,1].set_title("Distribución Retornos Diarios")
    axes[0,1].set_xlabel("Retorno diario")
    axes[0,1].set_ylabel("Frecuencia")
    axes[0,1].legend()

    # Heatmap correlación retornos
    returns_df = pd.DataFrame({ticker: df['ret_1'] for ticker, df in eda_data.items()}).dropna()
    sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1,0])
    axes[1,0].set_title("Correlación Retornos Diarios")

    # Scatter retorno vs volumen comparativo
    for ticker, df in eda_data.items():
        axes[1,1].scatter(df['ret_1'], df['Volume'], alpha=0.5, label=ticker)
    axes[1,1].set_title("Scatter Retorno vs Volumen Comparativo")
    axes[1,1].set_xlabel("Retorno diario")
    axes[1,1].set_ylabel("Volumen")
    axes[1,1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(GRAPH_PATH, "Comparativa_Tickers_2x2.png")
    plt.savefig(fig_path)
    plt.close(fig)

# =========================
# EJECUCIÓN
# =========================
eda_data = download_data(TICKERS, START)

for ticker, df in eda_data.items():
    eda_summary(df, ticker)
    plot_individual_eda(df, ticker)

plot_comparative_eda(eda_data)


#El EDA se mantiene compacto para evitar sobreajuste visual.