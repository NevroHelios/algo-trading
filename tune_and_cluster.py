"""
Lightweight tuning and cluster orchestration script
- Creates a temporary config with num_clusters=100
- Runs the statistical clusters backtest
- Performs a small grid search over Random Forest (if available) by updating config
- Prints best performing parameter set (final portfolio return)

Usage: run within the project virtualenv.
"""
import copy
import yaml
import os
import pprint
from backtester.backtest import run_backtest

BASE_CONFIG_PATH = os.path.join("config", "config.yaml")


def load_config(path=BASE_CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_with_num_clusters(num_clusters: int, config: dict):
    cfg = copy.deepcopy(config)
    cfg.setdefault("cluster_strategy", {})
    cfg["cluster_strategy"]["num_clusters"] = num_clusters
    # Force statistical_clusters strategy
    cfg["strategy"] = "statistical_clusters"

    # Save to temp file
    tmp_path = "tmp_tune_config.yaml"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Run backtest
    print(f"Running backtest with {num_clusters} clusters...")
    portfolio = run_backtest(config_path=tmp_path)

    # Clean up
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return portfolio


def small_rf_grid_search(config: dict):
    # If ml_algorithms.random_forest exists in config, grid search
    ml_cfg = config.get("ml_algorithms", {})
    rf_cfg = ml_cfg.get("random_forest", {})
    if not rf_cfg:
        print("No Random Forest config present in config.yaml; skipping RF grid search.")
        return None

    base_rf = rf_cfg.copy()
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15],
    }

    best = {"params": None, "return": -1e9}
    total = len(param_grid["n_estimators"]) * len(param_grid["max_depth"])
    i = 0

    for n in param_grid["n_estimators"]:
        for d in param_grid["max_depth"]:
            i += 1
            print(f"Grid search {i}/{total}: n_estimators={n}, max_depth={d}")
            cfg = copy.deepcopy(config)
            cfg.setdefault("cluster_strategy", {})
            cfg["cluster_strategy"]["num_clusters"] = 100
            cfg["strategy"] = "statistical_clusters"
            cfg.setdefault("ml_algorithms", {})
            cfg["ml_algorithms"]["random_forest"] = base_rf.copy()
            cfg["ml_algorithms"]["random_forest"]["n_estimators"] = n
            cfg["ml_algorithms"]["random_forest"]["max_depth"] = d

            tmp_path = "tmp_tune_config.yaml"
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)

            portfolio = run_backtest(config_path=tmp_path)
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            # Inspect portfolio final value
            if portfolio is not None:
                if portfolio.history:
                    last_price = portfolio.history[-1][1]
                    final_value = portfolio.get_current_value(last_price)
                else:
                    # No trades executed: use cash
                    final_value = portfolio.cash

                ret = final_value - portfolio.initial_cash
                print(f" -> Final return: {ret:+.2f}")

                if ret > best["return"]:
                    best["return"] = ret
                    best["params"] = {"n_estimators": n, "max_depth": d}

    print("Grid search complete. Best:")
    pprint.pprint(best)
    return best


if __name__ == "__main__":
    cfg = load_config()

    # Step 1: Run a single 100-cluster backtest
    portfolio = run_with_num_clusters(100, cfg)
    if portfolio is not None:
        if portfolio.history:
            last_price = portfolio.history[-1][1]
            print("Single run final portfolio value:", portfolio.get_current_value(last_price))
        else:
            print("Single run executed no trades. Final cash:", portfolio.cash)

    # Step 2: Small Random Forest grid search (uses portfolio return to compare)
    _ = small_rf_grid_search(cfg)
