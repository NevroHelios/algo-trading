"""
Small tuning harness: run quick grid search over Random Forest hyperparameters
using the existing statistical clusters backtester. This script modifies the
loaded config in-memory to set num_clusters=100 and adjusts the ML config.

It runs fast backtests (short data window) to get comparative returns. Designed
for running inside the project's venv.
"""

import copy
import yaml
import time
from pathlib import Path

from backtester.backtest import run_backtest
import tempfile
import os

CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def quick_backtest_for_config(cfg):
    # Keep a short testing window to make runs fast
    # We'll set the start_date to a recent 6-month window if available
    cfg = copy.deepcopy(cfg)
    # Reduce data range for speed
    cfg["start_date"] = cfg.get("start_date", "2023-06-01")
    cfg["end_date"] = cfg.get("end_date", "2024-08-22")
    # Enforce yahoo CSV-less mode if local data exists
    # Primary change: ensure cluster count is large
    cfg.setdefault("cluster_strategy", {})
    cfg["cluster_strategy"]["num_clusters"] = 100

    # Ensure ML is enabled and configure RandomForest hyperparams
    cfg.setdefault("ml_algorithms", {})
    cfg["ml_algorithms"]["enabled"] = True
    # Place random forest settings under ensemble for this simple harness
    cfg["ml_algorithms"]["random_forest"] = cfg["ml_algorithms"].get("random_forest", {})

    # Write a temporary config file and call the project's run_backtest loader
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
        # Ensure custom mode and activate all clusters
        cfg.setdefault("cluster_strategy", {})
        cfg["cluster_strategy"]["mode"] = "custom"
        cfg["cluster_strategy"]["custom_clusters"] = [f"cluster_{i}" for i in range(1, 101)]
        yaml.safe_dump(cfg, tf)
        tmp_path = tf.name

    try:
        # run_backtest reads a config path by default; it will perform the backtest and return a portfolio
        # Note: run_backtest prints output; it returns portfolio only for statistical clusters path
        result = run_backtest(str(tmp_path))
        # run_backtest may return None or a portfolio depending on execution path
        portfolio = result if result is not None else None

        # Try to compute simple return metric if portfolio available
        if portfolio is None:
            return None, None

        try:
            final_val = portfolio.get_current_value(portfolio.history[-1][1]) if getattr(portfolio, 'history', None) else portfolio.cash
            total_return = final_val - portfolio.initial_cash
            pct = (total_return / portfolio.initial_cash) * 100
            return float(pct), portfolio
        except Exception:
            return None, portfolio
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    cfg = load_config()

    # Grid for Random Forest (small grid for speed)
    n_estimators_grid = [10, 50, 100]
    max_depth_grid = [3, 5, 10]

    results = []

    for n in n_estimators_grid:
        for d in max_depth_grid:
            print(f"Running quick backtest: n_estimators={n}, max_depth={d}")
            # update config
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy.setdefault("ml_algorithms", {})
            cfg_copy["ml_algorithms"].setdefault("random_forest", {})
            cfg_copy["ml_algorithms"]["random_forest"]["n_estimators"] = n
            cfg_copy["ml_algorithms"]["random_forest"]["max_depth"] = d
            cfg_copy.setdefault("cluster_strategy", {})
            cfg_copy["cluster_strategy"]["num_clusters"] = 100

            start = time.time()
            pct, portfolio = quick_backtest_for_config(cfg_copy)
            elapsed = time.time() - start
            print(f"Elapsed: {elapsed:.1f}s | Return%: {pct}")
            results.append({"n_estimators": n, "max_depth": d, "return_pct": pct})

    print("\nGrid search complete. Results:")
    for r in results:
        print(r)

    # Save results
    out = Path(__file__).parent / "tuning_results.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f)

    print(f"Results written to {out}")
