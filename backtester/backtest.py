import yaml
import importlib
from backtester.data_handler import get_data
from enhanced_portfolio import EnhancedPortfolio
from backtester.executor import Executor


def run_backtest(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data = get_data(config)

    # Initialize strategy based on type
    strategy_name = config["strategy"]

    if strategy_name == "statistical_clusters":
        # Import and initialize statistical clusters strategy
        from strategies.statistical_clusters_strategy import StatisticalClustersStrategy

        strategy = StatisticalClustersStrategy(config)

        # For statistical clusters, we need all timeframe data
        print(f"Starting Statistical Clusters backtest:")
        print(f"Initial Capital: ₹{config['initial_cash']:,.2f}")

        # Run statistical clusters backtest
        run_statistical_clusters_backtest(strategy, data, config)

    else:
        # Original multi-timeframe strategy
        strategy_config = config["strategy_params"].copy()
        strategy_config.update(
            {
                "ml_algorithms": config.get("ml_algorithms", {}),
                "majority_voting": config.get("majority_voting", {}),
            }
        )

        strat_module = importlib.import_module(f"strategies.{config['strategy']}")
        strategy = strat_module.MyStrategy(strategy_config)

        # Run original backtest
        run_original_backtest(strategy, data, config)


def run_statistical_clusters_backtest(strategy, data, config):
    """Run backtest for statistical clusters strategy"""
    executor = Executor(config)
    portfolio = EnhancedPortfolio(config["initial_cash"], config)

    print(f"Data points available: {len(data)}")

    signals_generated = 0
    trades_executed = 0

    # For statistical clusters, we pass the full DataFrame
    for i in range(50, len(data)):  # Start after sufficient data for analysis
        # Get current slice of data up to current point
        current_data = {config.get("primary_timeframe", "1d"): data.iloc[: i + 1]}

        signal = strategy.generate_signal(current_data)

        if signal != "HOLD":
            signals_generated += 1

        # Get current row for portfolio update
        current_row = data.iloc[i].to_dict()

        if config["data_source"] == "yahoo":
            if portfolio.update(signal, current_row):
                trades_executed += 1
        else:
            if signal != "HOLD":
                current_price = data.iloc[i]["Close"]
                executor.execute(signal, current_price, config["ticker"])
                trades_executed += 1

    print(f"\nStatistical Clusters Backtest completed:")
    print(f"Signals generated: {signals_generated}")
    print(f"Trades executed: {trades_executed}")

    if config["data_source"] == "yahoo":
        portfolio.summary()
    else:
        print(f"Live trading session completed")


def run_original_backtest(strategy, data, config):
    """Run backtest for original multi-timeframe strategy"""
    executor = Executor(config)
    portfolio = EnhancedPortfolio(config["initial_cash"], config)

    print(f"Starting backtest with {len(data)} data points")
    print(f"Strategy: {config['strategy']}")
    print(f"Timeframes: {config.get('timeframes', ['1d'])}")
    print(f"ML Algorithms: {config.get('ml_algorithms', {}).get('enabled', False)}")

    signals_generated = 0
    trades_executed = 0

    for i, row in enumerate(data.itertuples()):
        # Create a row dict with timeframe data attached
        row_dict = row._asdict()

        # Attach timeframe data and index to the row for multi-timeframe analysis
        if hasattr(data, "attrs") and "timeframe_data" in data.attrs:
            row_dict["_df_attrs"] = data.attrs
            row_dict["_index"] = i

        signal = strategy.generate_signal(row_dict)

        if signal != "HOLD":
            signals_generated += 1

        if config["data_source"] == "yahoo":
            if portfolio.update(signal, row_dict):
                trades_executed += 1
        else:
            if signal != "HOLD":
                executor.execute(signal, row.Close, config["ticker"])
                trades_executed += 1

    print(f"\nBacktest completed:")
    print(f"Signals generated: {signals_generated}")
    print(f"Trades executed: {trades_executed}")

    if config["data_source"] == "yahoo":
        portfolio.summary()
    else:
        print(f"Live trading session completed:")
        print(f"Orders sent: {trades_executed}")
