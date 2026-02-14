# ======================================================
# WALK FORWARD Y COMPARACIÓN CON BUY & HOLD
# ======================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from scipy.stats import skew, kurtosis
from datetime import datetime, timedelta
    
    
TICKERS = ["AAPL", "SPY", "NVDA"]
START = pd.to_datetime("2015-01-01")
END   = pd.to_datetime("2026-01-01")

CAPITAL_INIT   = 10_000
COST           = 0.002
POSITION_SIZE  = 0.25
HORIZON        = 5
TOP_FEATURES   = 10
OUTLIER_PCTL   = 5
RANDOM_STATE   = 42

TRAIN_YEARS    = 3
VAL_MONTHS     = 6
TEST_MONTHS    = 6
PROB_FILTER_XGB = 0.6

MODELS = Path("public_data")
RESULTS = Path("public_data")
GRAPHS = Path("public_graphs")

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
GRAPHS.mkdir(exist_ok=True)

def simulate_equity(preds, rets, capital, position_size, cost):
    equity, trade_rets = [], []

    for r, p in zip(rets, preds):
        if p == 1:
            tr = position_size * (r - cost)
            capital *= (1 + tr)
            trade_rets.append(position_size * (r - cost))
        equity.append(capital)

    return pd.Series(equity), capital, trade_rets

def apply_filter(pred_probs, rets, prob_threshold, outlier_pct):
    preds = (pred_probs >= prob_threshold).astype(int)
    mask = preds == 1

    if mask.sum() > 0:
        cutoff = np.percentile(rets[mask], outlier_pct)
        preds[(mask) & (rets < cutoff)] = 0

    return preds

def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)

def max_dd(eq):
    return (eq / eq.cummax() - 1).min()

def CAGR(final_capital, initial_capital, start_date, end_date):

    if isinstance(start_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp):
        years = (end_date - start_date).days / 365.25
    else:
        years = len(range(start_date, end_date + 1)) / 252

    if years == 0:
        return 0

    return (final_capital / initial_capital) ** (1 / years) - 1

def MAR(cagr, max_dd):
    return cagr / abs(max_dd)


def download_ticker(ticker, start=START, end=END):
    df = yf.download(ticker, start=start, end=end, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)

    return df

def market_exposure(signals: pd.Series, position_size: float = 0.25):
    exposure = (signals.abs() * position_size).mean()
    return exposure
   
def capital_efficiency(cagr: float, exposure: float):
    if exposure == 0:
        return np.nan
    return cagr / exposure



def prepare_features(df):
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

    df["future_ret"] = df["Close"].shift(-HORIZON) / df["Close"] - 1
    df["Target"] = (df["future_ret"] > 0).astype(int)

    return df.dropna()


def run_walkforward(df, wf_config, model_config):
    capital = wf_config["capital_init"]
    capital_bh = wf_config["capital_init"]   

    wf_start = df.index.min()

    equity_segments_xgb = []
    equity_segments_bh  = []

    trade_rets, trade_count = [], 0

    last_model = None
    last_feats = None
    last_test_end = None

    while True:
        train_end = wf_start + pd.DateOffset(years=wf_config["train_years"])
        val_end   = train_end + pd.DateOffset(months=wf_config["val_months"])
        test_end  = val_end   + pd.DateOffset(months=wf_config["test_months"])

        if test_end > df.index.max():
            break

        train = df.loc[wf_start:train_end]
        val   = df.loc[train_end:val_end]
        test  = df.loc[val_end:test_end]

        X_train, y_train = train.drop(columns=["Target","future_ret"]), train["Target"]
        X_val,   y_val   = val.drop(columns=["Target","future_ret"]),   val["Target"]
        X_test           = test.drop(columns=["Target","future_ret"])
        rets_test        = test["future_ret"]

        fs = XGBClassifier(**model_config)
        fs.fit(X_train, y_train)

        gain = fs.get_booster().get_score(importance_type="gain")
        top_feats = sorted(gain, key=gain.get, reverse=True)[:wf_config["top_features"]]

        model = XGBClassifier(**model_config)
        model.fit(pd.concat([X_train[top_feats], X_val[top_feats]]),
                  pd.concat([y_train, y_val]))

        probs = model.predict_proba(X_test[top_feats])[:, 1]
        preds = apply_filter(probs, rets_test,
                             wf_config["prob_threshold"],
                             wf_config["outlier_pct"])
        
        signal_segments = []
        signal_segments.append(pd.Series(preds, index=rets_test.index))

        equity_xgb, capital, tr = simulate_equity(
            preds, rets_test, capital,
            wf_config["position_size"],
            wf_config["cost"]
        )
        equity_xgb.index = rets_test.index
        equity_segments_xgb.append(equity_xgb)

        eq_bh_block = (1 + rets_test).cumprod() * capital_bh
        capital_bh = eq_bh_block.iloc[-1]
        eq_bh_block.index = rets_test.index
        equity_segments_bh.append(eq_bh_block)

        trade_rets.extend(tr)
        trade_count += len(tr)

        last_model = model
        last_feats = top_feats
        last_test_end = test_end

        wf_start += pd.DateOffset(months=wf_config["test_months"])

    if last_test_end is not None and last_test_end < df.index.max():
        final_test = df.loc[last_test_end:]
        X_final = final_test[last_feats]
        rets_final = final_test["future_ret"]

        probs = last_model.predict_proba(X_final)[:, 1]
        preds = apply_filter(probs, rets_final,
                             wf_config["prob_threshold"],
                             wf_config["outlier_pct"])

        equity_xgb, capital, tr = simulate_equity(
            preds, rets_final, capital,
            wf_config["position_size"],
            wf_config["cost"]
        )
        equity_xgb.index = final_test.index
        equity_segments_xgb.append(equity_xgb)

        eq_bh_final = (1 + rets_final).cumprod() * capital_bh
        eq_bh_final.index = final_test.index
        equity_segments_bh.append(eq_bh_final)

        trade_rets.extend(tr)
        trade_count += len(tr)

    equity_xgb = pd.concat(equity_segments_xgb).sort_index()
    equity_bh  = pd.concat(equity_segments_bh).sort_index()

    all_signals = pd.concat(signal_segments).sort_index()

    return {
        "equity_xgb": equity_xgb,
        "equity_bh": equity_bh,
        "trade_returns": trade_rets,
        "trade_count": trade_count,
        "final_capital": capital,
        "signals": all_signals
    }
    
def train_final_model():
    print("Entrenando modelo final sobre todo el histórico...")

    df_all = []

    for t in TICKERS:
        df = prepare_features(download_ticker(t))
        df["Ticker"] = t
        df_all.append(df)

    df_all = pd.concat(df_all)

    X = df_all.drop(columns=["Target", "future_ret", "Ticker"])
    y = df_all["Target"]

    model = XGBClassifier(**model_config)
    model.fit(X, y)

    joblib.dump(model, MODELS / "final_model.pkl")
    joblib.dump(X.columns.tolist(), MODELS / "feature_list.pkl")


def compute_metrics(ticker, wf):
    eq = wf["equity_xgb"]
    tr = np.array(wf["trade_returns"])
    start_real = wf["equity_xgb"].index[0]
    end_real   = wf["equity_xgb"].index[-1]
    cagr = CAGR(wf["final_capital"], CAPITAL_INIT, start_real, end_real)
    mdd  = max_dd(eq)
    
    return {
        "Ticker": ticker,
        "Final Capital": wf["final_capital"],
        "Total Return %": (wf["final_capital"]/CAPITAL_INIT - 1)*100,
        "CAGR %": CAGR(wf["final_capital"], CAPITAL_INIT, start_real, end_real)*100,
        "Max DD %": max_dd(eq)*100,
        "Sharpe": sharpe(wf["equity_xgb"].pct_change().dropna()),
        "MAR": MAR(cagr, mdd),
        "Expectancy": tr.mean() if len(tr) > 0 else 0,
        "Trades": wf["trade_count"]
    }


def run_full_walkforward_pipeline():
    all_wf, metrics = {}, []

    for t in TICKERS:
        df = prepare_features(download_ticker(t))
        wf = run_walkforward(df, wf_config, model_config)
        all_wf[t] = wf
        metrics.append(compute_metrics(t, wf))

    df_metrics = pd.DataFrame(metrics)

    joblib.dump(all_wf, MODELS / "wf_results.pkl")
    joblib.dump(df_metrics, RESULTS / "wf_metrics.pkl")

    print("💾 Características WF guardados")
    return all_wf, df_metrics


wf_config = {
    "capital_init": CAPITAL_INIT,
    "train_years": TRAIN_YEARS,
    "val_months": VAL_MONTHS,
    "test_months": TEST_MONTHS,
    "prob_threshold": PROB_FILTER_XGB,
    "position_size": POSITION_SIZE,
    "cost": COST,
    "top_features": TOP_FEATURES,
    "outlier_pct": OUTLIER_PCTL
}

model_config = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss"
}

def save_configs():
    joblib.dump(wf_config, MODELS / "wf_config.pkl")
    joblib.dump(model_config, MODELS / "model_config.pkl")

def run_wf_analysis():
    all_wf, metrics = {}, []

    for t in TICKERS:
        df = prepare_features(download_ticker(t))
        wf = run_walkforward(df, wf_config, model_config)
        all_wf[t] = wf
        metrics.append(compute_metrics(t, wf))

    df_metrics = pd.DataFrame(metrics)

    print("\n--- MÉTRICAS WALK FOWARD DE XGB ---")
    print(
        df_metrics.to_string(
            index=False,
            formatters={
                "Final Capital": "{:,.2f}".format,
                "Total Return %": "{:.2f}".format,
                "CAGR %": "{:.2f}".format,
                "Max DD %": "{:.2f}".format,
                "Expectancy": "{:.4f}".format,
                "Sharpe": "{:.2f}".format
            }
        )
    )

    joblib.dump(all_wf, MODELS / "wf_results.pkl")
    joblib.dump(df_metrics, RESULTS / "wf_metrics.pkl")

    return all_wf, df_metrics


def plot_wf_normalized_equities(all_wf):
    plt.figure(figsize=(10,6))
    for t in TICKERS:
        eq_norm = all_wf[t]["equity_xgb"] / all_wf[t]["equity_xgb"].iloc[0]
        plt.plot(eq_norm.values, label=t)

    plt.title("Ticker Normalized Equity")
    plt.xlabel("Days")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS / "eq_normalized_by_ticker.png")
    plt.close()

    equity_total = pd.concat(
        [all_wf[t]["equity_xgb"].reset_index(drop=True) for t in TICKERS], axis=1
    ).sum(axis=1)
    equity_total_norm = equity_total / equity_total.iloc[0]

    plt.figure(figsize=(10,6))
    plt.plot(equity_total_norm.values, color="black", label="Global Equity")
    plt.title("Normalized Global Equity")
    plt.xlabel("Days")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS / "eq_global_normalized.png")
    plt.close()

def run_comparison(all_wf):
    comparison = []

    for t in TICKERS:
        wf = all_wf[t]
        eq_xgb = wf["equity_xgb"]
        tr_xgb = np.array(wf["trade_returns"])

        start_real = eq_xgb.index[0]
        end_real   = eq_xgb.index[-1]
        cagr = CAGR(wf["final_capital"], CAPITAL_INIT, start_real, end_real)
        mdd  = max_dd(eq_xgb)
        
        signals = wf["signals"]              
        exposure = market_exposure(signals, position_size=0.25)
        
        comparison.append({
            "Ticker": t,
            "Modelo": "XGB",
            "Final Capital": wf["final_capital"],
            "Total Return %": (wf["final_capital"] / CAPITAL_INIT - 1) * 100,
            "CAGR %": CAGR(wf["final_capital"], CAPITAL_INIT, start_real, end_real) * 100,
            "Max DD %": max_dd(eq_xgb) * 100,            
            "Sharpe": sharpe(wf["equity_xgb"].pct_change().dropna()),
            "MAR": MAR(cagr, mdd),
            "Expectancy": tr_xgb.mean() if len(tr_xgb) > 0 else np.nan,
            "Exposure %": exposure * 100,
            "Capital Efficiency": capital_efficiency(cagr, exposure)
        })

        df_bh = download_ticker(t)
        start_xgb = wf["equity_xgb"].index[0]

        rets_bh = df_bh["Close"].pct_change().fillna(0)
        eq_bh = (1 + rets_bh).cumprod() * CAPITAL_INIT
        cagr_bh = CAGR(eq_bh.iloc[-1], CAPITAL_INIT,
               eq_bh.index[0], eq_bh.index[-1])
        mdd_bh = max_dd(eq_bh)

        exposure_bh = 1.0
        
        comparison.append({
            "Ticker": t,
            "Modelo": "Buy & Hold",
            "Final Capital": eq_bh.iloc[-1],
            "Total Return %": (eq_bh.iloc[-1] / CAPITAL_INIT - 1) * 100,
            "CAGR %": CAGR(eq_bh.iloc[-1], CAPITAL_INIT,
                           eq_bh.index[0], eq_bh.index[-1]) * 100,
            "Max DD %": max_dd(eq_bh) * 100,
            "Sharpe": sharpe(rets_bh),
            "MAR": MAR(cagr_bh, mdd_bh),
            "Expectancy": np.nan,
            "Exposure %": 100,
            "Capital Efficiency": capital_efficiency(cagr_bh, exposure_bh)
        })

    df_comparison = pd.DataFrame(comparison)

    joblib.dump(df_comparison, RESULTS / "comparison.pkl")

    return df_comparison


def run_comparison_aligned(all_wf):
    comparison_aligned = []

    for t in TICKERS:
        wf = all_wf[t]
        eq_xgb = wf["equity_xgb"].copy()
        tr_xgb = np.array(wf["trade_returns"])

        start_xgb = pd.to_datetime(eq_xgb.index[0])

        df_bh = download_ticker(t)
        df_bh.index = pd.to_datetime(df_bh.index)
        df_bh_aligned = df_bh.loc[df_bh.index >= start_xgb]
        rets_bh = df_bh_aligned["Close"].pct_change().fillna(0)
        eq_bh_aligned = (1 + rets_bh).cumprod() * CAPITAL_INIT

        min_len = min(len(eq_xgb), len(eq_bh_aligned))
        eq_bh_aligned = eq_bh_aligned.iloc[:min_len]

        start_real = eq_xgb.index[0]
        end_real   = eq_xgb.index[-1]
        
        cagr = CAGR(wf["final_capital"], CAPITAL_INIT, start_real, end_real)
        mdd  = max_dd(eq_xgb)
        
        signals = wf["signals"]              
        exposure = market_exposure(signals, position_size=0.25)
        
        comparison_aligned.append({
            "Ticker": t,
            "Modelo": "XGB",
            "Final Capital": wf["final_capital"],
            "Total Return %": (wf["final_capital"] / CAPITAL_INIT - 1) * 100,
            "CAGR %": CAGR(wf["final_capital"], CAPITAL_INIT, start_real, end_real) * 100,
            "Max DD %": max_dd(eq_xgb) * 100,
            "Sharpe": sharpe(wf["equity_xgb"].pct_change().dropna()),
            "MAR": MAR(cagr, mdd),
            "Expectancy": tr_xgb.mean() if len(tr_xgb) > 0 else np.nan,
            "Exposure %": exposure * 100,
            "Capital Efficiency": capital_efficiency(cagr, exposure)
        })

        cagr_bh = CAGR(eq_bh_aligned.iloc[-1], CAPITAL_INIT,
               eq_bh_aligned.index[0], eq_bh_aligned.index[-1])
        mdd_bh = max_dd(eq_bh_aligned)
        exposure_bh = 1.0
        
        comparison_aligned.append({
            "Ticker": t,
            "Modelo": "Buy & Hold",
            "Final Capital": eq_bh_aligned.iloc[-1],
            "Total Return %": (eq_bh_aligned.iloc[-1] / CAPITAL_INIT - 1) * 100,
            "CAGR %": CAGR(eq_bh_aligned.iloc[-1], CAPITAL_INIT,
                           eq_bh_aligned.index[0], eq_bh_aligned.index[-1]) * 100,
            "Max DD %": max_dd(eq_bh_aligned) * 100,
            "Sharpe": sharpe(eq_bh_aligned.pct_change().fillna(0)),
            "MAR": MAR(cagr_bh, mdd_bh),
            "Expectancy": np.nan,
            "Exposure %": 100,
            "Capital Efficiency": capital_efficiency(cagr_bh, exposure_bh)
        })

    df_comparison_aligned = pd.DataFrame(comparison_aligned)

    joblib.dump(df_comparison_aligned, RESULTS / "comparison_aligned.pkl")

    return df_comparison_aligned


def plot_comparison_aligned(all_wf):

    equity_global_xgb_list = []
    equity_global_bh_list  = []
    dates_global_list = []  

    for t in all_wf.keys():
        wf = all_wf[t]        
        eq_xgb_rel = wf["equity_xgb"]           
        eq_xgb_cap = wf["equity_xgb"].copy()  

        start_date = wf["equity_xgb"].index[0]

        df_full = download_ticker(t)
        df_full.index = pd.to_datetime(df_full.index)
                
        df_bh = df_full.loc[df_full.index >= start_date].copy()

        rets_bh = df_bh["Close"].pct_change().fillna(0)
        eq_bh = (1 + rets_bh).cumprod() * CAPITAL_INIT  

        min_len = min(len(eq_xgb_cap), len(eq_bh))
        if min_len == 0:            
            print(f"⚠️  {t}: longitud 0 al alinear, saltando gráfico.")
            continue

        eq_xgb_pos = eq_xgb_cap.iloc[:min_len].reset_index(drop=True)
        eq_bh_pos  = eq_bh.iloc[:min_len].reset_index(drop=True)

        equity_global_xgb_list.append(eq_xgb_pos)
        equity_global_bh_list.append(eq_bh_pos)

        dates_segment = df_bh.index[:min_len]
        dates_global_list.append(dates_segment)

        plt.figure(figsize=(10,6))
        plt.plot(dates_segment, eq_xgb_cap.iloc[:min_len].values, label="XGB (equity)")
        plt.plot(dates_segment, eq_bh.iloc[:min_len].values,  label="Buy & Hold (aligned, equity)", linestyle="--")
        plt.title(f"Equity Aligned - {t} (Dates)")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(GRAPHS / f"equity_aligned_dates_{t}.png")
        plt.close('all')

    if len(dates_global_list) > 0:
        min_len_all = min(len(s) for s in dates_global_list)
        
        eq_g_xgb_dates = pd.concat([s.iloc[:min_len_all] for s in equity_global_xgb_list], axis=1).sum(axis=1)
        eq_g_bh_dates  = pd.concat([s.iloc[:min_len_all] for s in equity_global_bh_list], axis=1).sum(axis=1)
        
        ref_dates = dates_global_list[0][:min_len_all]

        plt.figure(figsize=(12,6))
        plt.plot(ref_dates, eq_g_xgb_dates.values, label="XGB Global")
        plt.plot(ref_dates, eq_g_bh_dates.values,  label="Buy & Hold Global (aligned)", linestyle="--")
        plt.title("Equity Global Aligned (Dates)")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(GRAPHS / "equity_aligned_global_dates.png")
        plt.close()

def plot_comparison_real_timeline(all_wf):

    equity_global_xgb = []
    equity_global_bh = []

    for t in all_wf.keys():
        wf = all_wf[t]
        eq_xgb = wf["equity_xgb"].copy()
        eq_xgb.index = pd.to_datetime(eq_xgb.index)

        df_bh = yf.download(t, start=START, end=END, progress=False)
        df_bh = df_bh[["Close"]].dropna()
        df_bh.index = pd.to_datetime(df_bh.index)

        rets_bh = df_bh["Close"].pct_change().fillna(0)
        eq_bh = (1 + rets_bh).cumprod() * CAPITAL_INIT

        equity_global_xgb.append(eq_xgb)
        equity_global_bh.append(eq_bh)

        plt.figure(figsize=(10,6))
        plt.plot(eq_bh.index, eq_bh.values, label="Buy & Hold", alpha=0.7)
        plt.plot(eq_xgb.index, eq_xgb.values, label="XGB", linewidth=2)

        plt.title(f"XGB vs Buy & Hold (Timeline real) — {t}")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(GRAPHS / f"equity_real_timeline_{t}.png")
        plt.close()

    plt.figure(figsize=(10,6))

    eq_global_xgb = pd.concat(equity_global_xgb, axis=1).sum(axis=1)
    eq_global_bh = pd.concat(equity_global_bh, axis=1).sum(axis=1)

    plt.plot(eq_global_bh.index, eq_global_bh.values, label="Buy & Hold")
    plt.plot(eq_global_xgb.index, eq_global_xgb.values, label="XGB", linewidth=2)

    plt.title("XGB vs Buy & Hold — Global")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS / "equity_real_timeline_global.png")
    plt.close()
  

def trade_statistics_table(all_wf):
   
    rows = []

    for t in all_wf:
        tr = np.array(all_wf[t]["trade_returns"])

        if len(tr) == 0:
            continue

        wins = tr[tr > 0]
        losses = tr[tr <= 0]

        win_rate = len(wins) / len(tr)
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
        expectancy = tr.mean()

        rows.append({
            "Ticker": t,
            "Trades": len(tr),
            "Win Rate %": win_rate * 100,
            "Avg Win": avg_win,
            "Avg Loss": avg_loss,
            "Payoff Ratio": payoff,
            "Expectancy": expectancy
        })

    df_stats = pd.DataFrame(rows)

    joblib.dump(df_stats, RESULTS / "trade_stats.pkl")

    return df_stats

def plot_trade_return_distribution(all_wf):
    trade_returns = []

    for t in all_wf.keys():
        trade_returns.extend(all_wf[t]["trade_returns"])

    trade_returns = np.array(trade_returns)

    mean_ret   = trade_returns.mean()
    median_ret = np.median(trade_returns)
    std_ret    = trade_returns.std()
    skew_ret   = pd.Series(trade_returns).skew()
    n_trades   = len(trade_returns)

    plt.figure(figsize=(10,6))

    sns.histplot(trade_returns, bins=80, stat="density", alpha=0.4)
    sns.kdeplot(trade_returns, linewidth=2)

    plt.axvline(mean_ret, linestyle="--", label=f"Media")
    plt.axvline(0, color="black", linewidth=1)

    stats_text = (
        f"N Trades: {n_trades}\n"
        f"Mean: {mean_ret:.4f}\n"
        f"Median: {median_ret:.4f}\n"
        f"Std Dev: {std_ret:.4f}\n"
        f"Skewness: {skew_ret:.2f}"
    )

    plt.text(
        0.98, 0.98, stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)
    )

    plt.title("Distribution of returns per trade")
    plt.xlabel("Return per trade")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS / "trade_return_distribution.png")
    plt.close()
    
def predict(ticker):

    model = joblib.load(MODELS / "final_model.pkl")
    feature_list = joblib.load(MODELS / "feature_list.pkl")

    end = datetime.today()
    start = end - timedelta(days=365*5)

    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = prepare_features(df)
    df = df.dropna()

    if len(df) < 600:
        raise ValueError("Histórico insuficiente para mini Walk Forward")

    X_live = df[feature_list].iloc[-1:]
    prob_live = model.predict_proba(X_live)[0][1]
    signal_live = int(prob_live >= PROB_FILTER_XGB)

    test = df.iloc[-252:].copy()

    X_test = test[feature_list]

    rets_test = test["Close"].pct_change().shift(-1).fillna(0)

    probs_test = model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= PROB_FILTER_XGB).astype(int)

    rets_test = rets_test.loc[test.index]

    equity_model, final_capital, _ = simulate_equity(
        preds_test,
        rets_test,
        CAPITAL_INIT,
        POSITION_SIZE,
        COST
    )

    equity_model.index = test.index

    rets_bh = test["Close"].pct_change().fillna(0)
    equity_bh = (1 + rets_bh).cumprod() * CAPITAL_INIT

    start_date = equity_model.index[0]
    end_date = equity_model.index[-1]

    years = (end_date - start_date).days / 365.25

    model_cagr = (equity_model.iloc[-1] / equity_model.iloc[0]) ** (1 / years) - 1
    bh_cagr = (equity_bh.iloc[-1] / equity_bh.iloc[0]) ** (1 / years) - 1

    model_sharpe = sharpe(equity_model.pct_change().dropna())
    bh_sharpe = sharpe(rets_bh)

    model_exposure = market_exposure(
        pd.Series(preds_test, index=test.index),
        POSITION_SIZE
    )

    roll_max_model = equity_model.cummax()
    dd_model = (equity_model / roll_max_model - 1).min()

    roll_max_bh = equity_bh.cummax()
    dd_bh = (equity_bh / roll_max_bh - 1).min()

    return {
        "probability": prob_live,
        "signal": signal_live,
        "comparison": {
            "model_cagr": model_cagr * 100,
            "bh_cagr": bh_cagr * 100,
            "model_sharpe": model_sharpe,
            "bh_sharpe": bh_sharpe,
            "model_exposure": model_exposure * 100,
            "model_max_dd": dd_model * 100,
            "bh_max_dd": dd_bh * 100
        }
    }



if __name__ == "__main__":
        
    all_wf, df_metrics = run_wf_analysis()

    df_comparison = run_comparison(all_wf)
    df_comparison_aligned = run_comparison_aligned(all_wf)

    df_trade_stats = trade_statistics_table(all_wf)
  
    plot_wf_normalized_equities(all_wf)
    plot_comparison_aligned(all_wf)
    plot_comparison_real_timeline(all_wf)
    plot_trade_return_distribution(all_wf)
   
    save_configs()
    train_final_model()


    print("\n--- COMPARACIÓN ---")
    print(df_comparison.to_string(index=False))

    print("\n--- COMPARACIÓN ALINEADA ---")
    print(df_comparison_aligned.to_string(index=False))

    print("\n--- ESTADÍSTICAS DE TRADES ---")
    print(df_trade_stats.to_string(index=False))

    print("✅ PIPELINE COMPLETO FINALIZADO")

    
    
  