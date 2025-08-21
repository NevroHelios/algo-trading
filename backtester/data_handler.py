import yfinance as yf
import pandas as pd


def compute_rsi(series, period=14):
    """Compute Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=14):
    """Compute Average True Range"""
    df = df.copy()
    df["high_low"] = df["High"] - df["Low"]
    df["high_close"] = (df["High"] - df["Close"].shift()).abs()
    df["low_close"] = (df["Low"] - df["Close"].shift()).abs()
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    atr = df["tr"].rolling(window=period).mean()
    return atr


def compute_bollinger_bands(series, period=20, std_dev=2.0):
    """Compute Bollinger Bands"""
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)
    return upper, lower, ma


def compute_ichimoku(df, conversion=9, base=26, span_b=52):
    """Compute Ichimoku Cloud components"""
    high_conversion = df["High"].rolling(window=conversion).max()
    low_conversion = df["Low"].rolling(window=conversion).min()
    conversion_line = (high_conversion + low_conversion) / 2

    high_base = df["High"].rolling(window=base).max()
    low_base = df["Low"].rolling(window=base).min()
    base_line = (high_base + low_base) / 2

    span_a = ((conversion_line + base_line) / 2).shift(base)

    high_span_b = df["High"].rolling(window=span_b).max()
    low_span_b = df["Low"].rolling(window=span_b).min()
    span_b_line = ((high_span_b + low_span_b) / 2).shift(base)

    return conversion_line, base_line, span_a, span_b_line


def add_technical_indicators(df, config):
    """Add all technical indicators with configurable parameters"""
    params = config["strategy_params"]

    # Moving Averages - use multiple windows
    for fast_window in params["fast_ma_windows"]:
        df[f"fast_ma_{fast_window}"] = df["Close"].rolling(fast_window).mean()

    for slow_window in params["slow_ma_windows"]:
        df[f"slow_ma_{slow_window}"] = df["Close"].rolling(slow_window).mean()

    # RSI with multiple periods
    for rsi_period in params["rsi_periods"]:
        df[f"rsi_{rsi_period}"] = compute_rsi(df["Close"], rsi_period)

    # Bollinger Bands with multiple parameters
    for bb_period in params["bb_periods"]:
        for bb_std in params["bb_std_devs"]:
            upper, lower, middle = compute_bollinger_bands(
                df["Close"], bb_period, bb_std
            )
            df[f"bb_upper_{bb_period}_{bb_std}"] = upper
            df[f"bb_lower_{bb_period}_{bb_std}"] = lower
            df[f"bb_middle_{bb_period}_{bb_std}"] = middle

    # ATR with multiple periods
    for atr_period in params["atr_periods"]:
        df[f"atr_{atr_period}"] = compute_atr(df, atr_period)

    # Support and Resistance levels
    for sr_period in params["support_resistance_periods"]:
        df[f"support_{sr_period}"] = df["Low"].rolling(sr_period).min()
        df[f"resistance_{sr_period}"] = df["High"].rolling(sr_period).max()

    # Ichimoku Cloud with multiple parameters
    for conv in params["ichimoku_conversion"]:
        for base in params["ichimoku_base"]:
            for span_b in params["ichimoku_span_b"]:
                conv_line, base_line, span_a, span_b_line = compute_ichimoku(
                    df, conv, base, span_b
                )
                df[f"ichi_conv_{conv}_{base}_{span_b}"] = conv_line
                df[f"ichi_base_{conv}_{base}_{span_b}"] = base_line
                df[f"ichi_span_a_{conv}_{base}_{span_b}"] = span_a
                df[f"ichi_span_b_{conv}_{base}_{span_b}"] = span_b_line

    # MSE (Mean Squared Error) with multiple periods
    for mse_period in params["mse_periods"]:
        df[f"mse_{mse_period}"] = (
            ((df["Close"] - df[f"fast_ma_{params['fast_ma_windows'][0]}"]) ** 2)
            .rolling(mse_period)
            .mean()
        )

    # Volume indicators (if available)
    if "Volume" in df.columns:
        for period in [10, 20, 50]:
            df[f"volume_ma_{period}"] = df["Volume"].rolling(period).mean()
            df[f"volume_ratio_{period}"] = df["Volume"] / df[f"volume_ma_{period}"]

    return df


def get_single_timeframe_data(config, interval):
    """Get data for a single timeframe"""
    # Use Yahoo Finance for all data
    df = yf.download(
        config["ticker"],
        start=config["start_date"],
        end=config["end_date"],
        interval=interval,
    )

    if df is None or df.empty:
        raise ValueError(f"Failed to download data for {interval}")

    # Fix multi-index issue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if col[0] != "Adj Close" else "Close" for col in df.columns
        ]
    df.rename(columns={"Adj Close": "Close"}, inplace=True)
    df.dropna(inplace=True)

    return df


def get_data(config):
    """Get multi-timeframe data with all indicators"""
    timeframes = config.get("timeframes", ["1d"])
    primary_timeframe = config.get("primary_timeframe", timeframes[0])

    print(f"Requested timeframes: {timeframes}")
    print(f"Primary timeframe: {primary_timeframe}")

    # Dictionary to store data for each timeframe
    timeframe_data = {}

    # Get data for each timeframe
    for interval in timeframes:
        print(f"Fetching data for timeframe: {interval}")
        try:
            df = get_single_timeframe_data(config, interval)
            df = add_technical_indicators(df, config)
            timeframe_data[interval] = df
            print(f"Successfully loaded {len(df)} records for {interval}")
        except Exception as e:
            print(f"Warning: Failed to load data for {interval}: {e}")
            continue

    print(f"Available timeframes after fetching: {list(timeframe_data.keys())}")

    if not timeframe_data:
        raise ValueError("Failed to load data for any timeframe")

    # If primary timeframe failed, use the first available timeframe
    if primary_timeframe not in timeframe_data:
        available_timeframes = list(timeframe_data.keys())
        primary_timeframe = available_timeframes[0]
        print(f"Primary timeframe not available, using {primary_timeframe} instead")

    # Return the primary timeframe data with access to all timeframes
    primary_df = timeframe_data[primary_timeframe].copy()
    primary_df.attrs["timeframe_data"] = timeframe_data
    primary_df.attrs["primary_timeframe"] = primary_timeframe

    print(f"Returning primary DataFrame with {len(primary_df)} records")
    print(f"DataFrame attrs keys: {list(primary_df.attrs.keys())}")

    return primary_df
