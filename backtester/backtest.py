import yaml
import importlib
from backtester.data_handler import get_data
from backtester.portfolio import Portfolio

def run_backtest(config_path="config/config.yaml"):
    config = yaml.safe_load(open(config_path))
    data = get_data(config)

    strat_module = importlib.import_module(f"strategies.{config['strategy']}")
    strategy = strat_module.MyStrategy(config["strategy_params"])
    portfolio = Portfolio(config["initial_cash"])

    for row in data.itertuples():
        row = row._asdict() # type: ignore
        signal = strategy.generate_signal(row)
        portfolio.update(signal, row)

    portfolio.summary()
