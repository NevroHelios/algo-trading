import yaml
import importlib
from backtester.data_handler import get_data
from enhanced_portfolio import EnhancedPortfolio
from backtester.executor import Executor


def run_backtest(config_path="config/config.yaml"):
    config = yaml.safe_load(open(config_path))
    data = get_data(config)

    # Pass full config to strategy, not just strategy_params
    strategy_config = config["strategy_params"].copy()
    strategy_config.update({
        "ml_algorithms": config.get("ml_algorithms", {}),
        "majority_voting": config.get("majority_voting", {})
    })
    
    strat_module = importlib.import_module(f"strategies.{config['strategy']}")
    strategy = strat_module.MyStrategy(strategy_config)

    executor = Executor(config)
    
    # Use enhanced portfolio with tax and fee calculations
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

    if config["data_source"] == "yahoo":
        print(f"\nBacktest completed:")
        print(f"Signals generated: {signals_generated}")
        print(f"Trades executed: {trades_executed}")
        portfolio.summary()
    else:
        print(f"\nLive trading session completed:")
        print(f"Signals generated: {signals_generated}")
        print(f"Orders sent: {trades_executed}")
