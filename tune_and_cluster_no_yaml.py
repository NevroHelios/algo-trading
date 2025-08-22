"""
Tuning script variant that avoids PyYAML by building a minimal config dict in Python.
This tries to run a 100-cluster statistical clusters backtest using existing project code.
"""
import traceback
import os
import pandas as pd
from strategies.statistical_clusters_strategy import StatisticalClustersStrategy
from backtester.backtest import run_statistical_clusters_backtest


def build_config():
    # Minimal config taken from repo defaults; adjust as needed
    return {
        "data_source": "local",
        "mode": "paper",
        "ticker": "RELIANCE.NS",
        "start_date": "2023-06-01",
        "end_date": "2024-08-22",
        "initial_cash": 200000,
        "strategy": "statistical_clusters",
        "cluster_strategy": {
            "mode": "predefined",
            "predefined_strategy": "adaptive_trend",
            "num_clusters": 100,
        },
        "primary_timeframe": "1d",
        "timeframes": ["1d"],
        "ml_algorithms": {"enabled": True},
        "strategy_params": {
            "fast_ma_windows": [5, 10, 15, 20],
            "slow_ma_windows": [20, 30, 50, 100],
            "rsi_periods": [14, 21],
            "bb_periods": [20, 30],
            "bb_std_devs": [2.0, 2.5],
            "atr_periods": [14, 21],
            "support_resistance_periods": [20, 30, 50],
            "ichimoku_conversion": [9, 12],
            "ichimoku_base": [26, 30],
            "ichimoku_span_b": [52, 60],
            "mse_periods": [20, 30],
        },
    }


def load_local_data(path: str, cfg: dict):
    df = pd.read_csv(path)
    # Build a business-day DateTime index matching start_date
    try:
        start = pd.to_datetime(cfg.get("start_date"))
    except Exception:
        start = pd.Timestamp("2023-06-01")

    # Use business day frequency and match number of rows
    idx = pd.bdate_range(start=start, periods=len(df))
    df.index = idx

    # Ensure columns names match expected ones
    expected = ["Open", "High", "Low", "Close", "Volume"]
    # If CSV has different capitalization or column order, try to map
    cols = {c.lower(): c for c in df.columns}
    for col in expected:
        if col not in df.columns:
            low = col.lower()
            if low in cols:
                df.rename(columns={cols[low]: col}, inplace=True)

    # Do not compute technical indicators here to avoid extra dependencies.
    # The strategy implementation is resilient and will operate using Close prices.

    return df


if __name__ == "__main__":
    cfg = build_config()
    try:
        local_path = os.path.join("data", "data.csv")
        if os.path.exists(local_path):
            print("Loading local data file:", local_path)
            data = load_local_data(local_path, cfg)
        else:
            # Fallback to remote loader if local missing
            from backtester.data_handler import get_data

            print("Local data not found; fetching remote data...")
            data = get_data(cfg)

        print(f"Data ready: {len(data)} rows")

        print("Initializing strategy...")
        strategy = StatisticalClustersStrategy(cfg)

        print("Running statistical clusters backtest with 100 clusters...")
        portfolio = run_statistical_clusters_backtest(strategy, data, cfg)

        if portfolio is not None:
            if getattr(portfolio, 'history', None):
                last_price = portfolio.history[-1][1]
                print("Final portfolio value:", portfolio.get_current_value(last_price))
            else:
                print("No trades executed. Final cash:", portfolio.cash)

    except Exception as e:
        print("Error during tuning run:")
        traceback.print_exc()
