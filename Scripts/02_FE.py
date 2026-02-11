# ======================================================
# FEATURE ENGINEERING 
# ======================================================
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

plt.style.use("ggplot")

# ======================================================
# RUTAS
# ======================================================
GRAPH_PATH = "Graphs_FE"
DATA_PATH = "data_FE"
os.makedirs(GRAPH_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# ======================================================
# CONFIG
# ======================================================
TICKER = "AAPL"
START = "2015-01-01"
RANDOM_STATE = 42
TEST_SIZE = 0.3
HORIZONS = {"daily": 1, "weekly": 5, "monthly": 21}

# ======================================================
# FUNCIONES AUXILIARES
# ======================================================
def download_data(ticker=TICKER, start=START):
    df = yf.download(ticker, start=start, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df

def create_features(df):
    df = df.copy()
    # Retornos y price features
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_20"] = df["Close"].pct_change(20)
    df["gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["range"] = (df["High"] - df["Low"]) / df["Close"]
    df["body"] = (df["Close"] - df["Open"]) / df["Close"]
    df["upper_wick"] = (df["High"] - np.maximum(df["Close"], df["Open"])) / df["Close"]
    df["lower_wick"] = (np.minimum(df["Close"], df["Open"]) - df["Low"]) / df["Close"]
    
    # Volumen
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_rel"] = df["Volume"] / df["vol_ma20"]
    df["vol_trend"] = df["vol_rel"].rolling(5).mean()
    
    # Máximos y mínimos
    df["high_20"] = df["High"].rolling(20).max()
    df["low_20"] = df["Low"].rolling(20).min()
    df["distance_high_20"] = (df["Close"] - df["high_20"]) / df["Close"]
    df["distance_low_20"] = (df["Close"] - df["low_20"]) / df["Close"]
    
    # Volatilidad
    df["volatility_20"] = df["Close"].pct_change().rolling(20).std()
    df["ret_norm"] = df["ret_1"] / df["volatility_20"]
    df["range_norm"] = df["range"] / df["volatility_20"]
    df["vol_regime"] = (df["volatility_20"] > df["volatility_20"].rolling(100).median()).astype(int)
    
    # EMAs
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    df["EMA_200"] = df["Close"].ewm(span=200).mean()
    
    # MACD
    ema_12 = df["Close"].ewm(span=12).mean()
    ema_26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema_12 - ema_26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"] = df["MACD"] - df["Signal"]
    
    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(14).mean() / down.ewm(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # Stochastic
    low_14 = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()
    df["Stoch"] = 100 * (df["Close"] - low_14) / (high_14 - low_14)
    df = df.dropna()
    
    return df

def save_heatmaps(df, tech_cols, final_features):
    
    # Heatmap tech
    plt.figure(figsize=(14,10))
    sns.heatmap(df[tech_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlaciones TECH features")
    plt.savefig(os.path.join(GRAPH_PATH, "heatmap_tech.png"))
    plt.close()
    
    # Heatmap final
    plt.figure(figsize=(14,10))
    sns.heatmap(df[final_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlaciones FINAL features")
    plt.savefig(os.path.join(GRAPH_PATH, "heatmap_final.png"))
    plt.close()

# ======================================================
# MODELOS Y MÉTRICAS
# ======================================================
models = {
    "Logistic": (Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]), {"model__C":[0.01,0.1,1,5]}),
    "Ridge": (Pipeline([("scaler", StandardScaler()), ("model", RidgeClassifier())]), {"model__alpha":[0.1,1,5]}),
    "RandomForest": (RandomForestClassifier(random_state=RANDOM_STATE), {"n_estimators":[200,400],"max_depth":[4,6]}),
    "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE), {"n_estimators":[200,400],"max_depth":[3,4],"learning_rate":[0.03,0.05],"subsample":[0.7],"colsample_bytree":[0.7]})
}

FINAL_FEATURES = [
    "ret_1","ret_5","ret_20",
    "gap","range","body",
    "upper_wick","lower_wick",
    "vol_rel","vol_trend",
    "distance_high_20","distance_low_20",
    "ret_norm","range_norm"
]

TECH_COLS = ["EMA_20","EMA_50","EMA_200","MACD","MACD_hist","RSI","Stoch","ret_1","ret_5","volatility_20"]

# ======================================================
# FUNCIONES DE EJECUCIÓN DE MODELOS
# ======================================================
def apply_filter(probs, rets, outliers=None):
    return (probs >= 0.5).astype(int)  # FE no usa el 0.6, justo comparación

def run_no_grid(df, horizon_name, horizon, smote=False, outliers=None):
    data = df.copy()
    
    data["future_ret"] = data["Close"].shift(-horizon) / data["Close"] - 1
    data["Target"] = (data["future_ret"] > 0).astype(int)
    data = data.dropna()
    
    X = data[FINAL_FEATURES]
    y = data["Target"]
    rets = data["future_ret"]
    split = int(len(data) * (1 - TEST_SIZE))
    
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    rets_test = rets.iloc[split:]
    
    if smote:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    rows = []
    
    for name, (model, _) in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        preds = apply_filter(probs, rets_test, outliers)
        rows.append({
            "Horizon": horizon_name,
            "Model": name,
            "Grid": False,
            "SMOTE": smote,
            "Outliers": outliers,
            "Precision": precision_score(y_test, preds, zero_division=0),
            "AUC": roc_auc_score(y_test, probs),
            "F1": f1_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "ACC": accuracy_score(y_test, preds)
        })
        
    return pd.DataFrame(rows)

def run_with_grid(df, horizon_name, horizon):
    data = df.copy()
    
    data["future_ret"] = data["Close"].shift(-horizon) / data["Close"] - 1
    data["Target"] = (data["future_ret"] > 0).astype(int)
    data = data.dropna()
    
    X = data[FINAL_FEATURES]
    y = data["Target"]
    rets = data["future_ret"]
    split = int(len(data) * (1 - TEST_SIZE))
    
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    rets_test = rets.iloc[split:]
    rows = []
    
    for name, (model, params) in models.items():
        gs = GridSearchCV(model, params, scoring="precision", cv=TimeSeriesSplit(5), n_jobs=1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        probs = best.predict_proba(X_test)[:,1] if hasattr(best, "predict_proba") else best.decision_function(X_test)
        preds = apply_filter(probs, rets_test)
        rows.append({
            "Horizon": horizon_name,
            "Model": name,
            "Grid": True,
            "SMOTE": False,
            "Outliers": None,
            "Precision": precision_score(y_test, preds, zero_division=0),
            "AUC": roc_auc_score(y_test, probs),
            "F1": f1_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "ACC": accuracy_score(y_test, preds)
        })
        
    return pd.DataFrame(rows)

# ======================================================
# WRAPPERS DE COMPARACIÓN
# ======================================================
def compare_horizons(df, final_features, horizons):
    return pd.concat([run_no_grid(df, h, v) for h,v in horizons.items()])

def compare_grid(df, final_features, horizon):
    return pd.concat([run_no_grid(df, "weekly", horizon), run_with_grid(df, "weekly", horizon)])

def compare_outliers(df, final_features, horizon):
    return pd.concat([run_no_grid(df, "weekly", horizon, outliers=None),
                      run_no_grid(df, "weekly", horizon, outliers="inferior")])

def compare_smote(df, final_features, horizon):
    return pd.concat([run_no_grid(df, "weekly", horizon, smote=False),
                      run_no_grid(df, "weekly", horizon, smote=True)])

def final_weekly_model(df, final_features, horizon, smote=False, outliers=None):
    return run_no_grid(
        df,
        "weekly",
        horizon,
        smote=smote,
        outliers=outliers
    )

# ======================================================
# EJECUCIÓN
# ======================================================
df = download_data(TICKER, START)
df = create_features(df)
save_heatmaps(df, TECH_COLS, FINAL_FEATURES)

# Guardar CSV con features
df.to_csv(os.path.join(DATA_PATH, f"{TICKER}_FE.csv"))

# Comparaciones
df_horizons = compare_horizons(df, FINAL_FEATURES, HORIZONS)
df_grid = compare_grid(df, FINAL_FEATURES, horizon=5)
df_smote = compare_smote(df, FINAL_FEATURES, horizon=5)
df_outliers = compare_outliers(df, FINAL_FEATURES, horizon=5)
df_final = final_weekly_model(df, FINAL_FEATURES, horizon=5, smote=False, outliers="inferior")
    
# Mostrar resultados por pantalla
print("=== COMPARACIÓN POR HORIZON ===")
print(df_horizons)

print("\n=== COMPARACIÓN GRID VS NO GRID ===")
print(df_grid)

print("\n=== COMPARACIÓN SMOTE ===")
print(df_smote)

print("\n=== COMPARACIÓN OUTLIERS ===")
print(df_outliers)

print("\n=== MODELO FINAL SEMANAL ===")
print(df_final)

# Guardar resultados
# df_horizons.to_csv(os.path.join(DATA_PATH, "compare_horizons.csv"), index=False)
# df_grid.to_csv(os.path.join(DATA_PATH, "compare_grid.csv"), index=False)
# df_outliers.to_csv(os.path.join(DATA_PATH, "compare_outliers.csv"), index=False)
# df_smote.to_csv(os.path.join(DATA_PATH, "compare_smote.csv"), index=False)
# df_final.to_csv(os.path.join(DATA_PATH, "final_weekly.csv"), index=False)

print("✅ FE completo ejecutado y resultados guardados.")
