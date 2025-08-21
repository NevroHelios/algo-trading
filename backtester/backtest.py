import yaml
import importlib
from backtester.data_handler import get_data
from backtester.portfolio import Portfolio
from backtester.executor import Executor

def run_backtest(config_path="config/config.yaml"):
    config = yaml.safe_load(open(config_path))
    data = get_data(config)

    strat_module = importlib.import_module(f"strategies.{config['strategy']}")
    strategy = strat_module.MyStrategy(config["strategy_params"])

    executor = Executor(config)
    portfolio = Portfolio(config["initial_cash"])

    for row in data.itertuples():
        signal = strategy.generate_signal(row._asdict()) # type: ignore
        if config["data_source"] == "yahoo":
            portfolio.update(signal, row._asdict()) # type: ignore
        else:
            executor.execute(signal, row.Close, config["ticker"])

    if config["data_source"] == "yahoo":
        portfolio.summary()
