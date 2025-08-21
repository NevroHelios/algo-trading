import yfinance as yf
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def get_data(config):
    df = None
    if config["data_source"] == "yahoo":
        df = yf.download(
            config["ticker"],
            start=config["start_date"],
            end=config["end_date"],
            interval=config.get("interval", "1d")
        )
        assert df is not None, "Failed to download data from Yahoo Finance"
        # Fix multi-index issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[0] != "Adj Close" else "Close" for col in df.columns]
        df.rename(columns={"Adj Close": "Close"}, inplace=True)
        df.dropna(inplace=True)
        

    elif config["data_source"] == "alpaca":
        client = StockHistoricalDataClient(
            config["alpaca"]["key_id"],
            config["alpaca"]["secret_key"]
        )
        request = StockBarsRequest(
            symbol_or_symbols=config["ticker"],
            timeframe=TimeFrame.Day if config["interval"] == "1Day" else TimeFrame.Minute, # type: ignore
            start=pd.Timestamp(config["start_date"]),
            end=pd.Timestamp(config["end_date"])
        )
        bars = client.get_stock_bars(request)
        df = bars.df
        df = df[df.index.get_level_values("symbol") == config["ticker"]]
        df = df.reset_index(level="symbol", drop=True)
        df.rename(columns={"close": "Close"}, inplace=True)

    assert df is not None, "Failed to retrieve data"
    # Add indicators
    df["fast_ma"] = df["Close"].rolling(config["strategy_params"]["fast_ma"]).mean()
    df["slow_ma"] = df["Close"].rolling(config["strategy_params"]["slow_ma"]).mean()
    
    return df
