import yfinance as yf
import pandas as pd

def get_data(config):
    # ticker = config["ticker"]
    # df = yf.download(
    #     ticker,
    #     start=config["start_date"],
    #     end=config["end_date"],
    #     interval=config.get("interval", "1d")
    # )
    # assert df is not None, "Dataset load failed"
    # df.dropna(inplace=True)

    # # Precompute indicators if needed (like moving averages)
    # df["fast_ma"] = df["Close"].rolling(config["strategy_params"]["fast_ma"]).mean()
    # df["slow_ma"] = df["Close"].rolling(config["strategy_params"]["slow_ma"]).mean()
    # df.to_csv('data.csv', index=False)
    # return df
    df = pd.read_csv('data/data.csv')
    assert df is not None, "Dataset load failed"
    return df
