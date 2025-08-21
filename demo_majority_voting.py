#!/usr/bin/env python3
"""
Demo script to show how the majority voting system works
"""

import pandas as pd
from strategies.multi_timeframe import MyStrategy


def demo_majority_voting():
    """Demonstrate the majority voting system with simulated data"""

    # Configuration for the strategy
    config = {
        "fast_ma_windows": [5, 10],
        "slow_ma_windows": [20, 30],
        "rsi_periods": [14],
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_periods": [20],
        "bb_std_devs": [2.0],
        "atr_periods": [14],
        "support_resistance_periods": [20],
        "ichimoku_conversion": [9],
        "ichimoku_base": [26],
        "ichimoku_span_b": [52],
        "mse_periods": [20],
        "majority_voting": {
            "enabled": True,
            "minimum_timeframes": 2,
            "require_majority": True,
            "weight_by_strength": True,
        },
    }

    strategy = MyStrategy(config)

    print("=" * 60)
    print("MAJORITY VOTING DEMONSTRATION")
    print("=" * 60)

    # Scenario 1: Multiple timeframes agree on BUY
    print("\n📊 SCENARIO 1: Multiple timeframes agree - BUY signal")
    print("-" * 50)

    timeframe_data_scenario1 = {
        "1d": create_mock_df_with_signals("BUY", 0.8),
        "1h": create_mock_df_with_signals("BUY", 0.9),
        "15m": create_mock_df_with_signals("BUY", 0.7),
    }

    row_scenario1 = create_mock_row(timeframe_data_scenario1, 0)
    signal1 = strategy._generate_multi_timeframe_signal(row_scenario1)
    print(f"Final decision: {signal1}")

    # Scenario 2: Conflicting signals - BUY majority
    print("\n📊 SCENARIO 2: Conflicting signals - BUY wins majority")
    print("-" * 50)

    timeframe_data_scenario2 = {
        "1d": create_mock_df_with_signals("BUY", 0.85),
        "1h": create_mock_df_with_signals("BUY", 0.75),
        "15m": create_mock_df_with_signals("SELL", 0.6),
    }

    row_scenario2 = create_mock_row(timeframe_data_scenario2, 0)
    signal2 = strategy._generate_multi_timeframe_signal(row_scenario2)
    print(f"Final decision: {signal2}")

    # Scenario 3: Conflicting signals - SELL majority
    print("\n📊 SCENARIO 3: Conflicting signals - SELL wins majority")
    print("-" * 50)

    timeframe_data_scenario3 = {
        "1d": create_mock_df_with_signals("SELL", 0.9),
        "1h": create_mock_df_with_signals("SELL", 0.8),
        "15m": create_mock_df_with_signals("BUY", 0.7),
    }

    row_scenario3 = create_mock_row(timeframe_data_scenario3, 0)
    signal3 = strategy._generate_multi_timeframe_signal(row_scenario3)
    print(f"Final decision: {signal3}")

    # Scenario 4: Tie situation
    print("\n📊 SCENARIO 4: Tie situation - equal votes")
    print("-" * 50)

    timeframe_data_scenario4 = {
        "1d": create_mock_df_with_signals("BUY", 0.8),
        "1h": create_mock_df_with_signals("SELL", 0.8),
    }

    row_scenario4 = create_mock_row(timeframe_data_scenario4, 0)
    signal4 = strategy._generate_multi_timeframe_signal(row_scenario4)
    print(f"Final decision: {signal4}")

    print("\n" + "=" * 60)
    print("🎯 SUMMARY:")
    print("The majority voting system works as follows:")
    print("1. Each timeframe generates its own signal (BUY/SELL/HOLD)")
    print("2. Timeframes with HOLD signals are ignored in voting")
    print("3. The system counts BUY vs SELL votes")
    print("4. Majority wins, but signal strength must meet threshold")
    print("5. In case of tie, the system returns HOLD")
    print("=" * 60)


def create_mock_df_with_signals(target_signal, strength):
    """Create a mock DataFrame that would generate the target signal"""
    data = {"Close": [100.0], "High": [102.0], "Low": [98.0], "Volume": [1000000]}

    # Add indicators based on desired signal
    if target_signal == "BUY":
        # Set up indicators for BUY signal
        data["fast_ma_5"] = [101.0]  # Fast MA above slow MA (bullish)
        data["slow_ma_20"] = [99.0]
        data["fast_ma_10"] = [100.5]
        data["slow_ma_30"] = [98.0]
        data["rsi_14"] = [25.0]  # RSI oversold (bullish)
    elif target_signal == "SELL":
        # Set up indicators for SELL signal
        data["fast_ma_5"] = [99.0]  # Fast MA below slow MA (bearish)
        data["slow_ma_20"] = [101.0]
        data["fast_ma_10"] = [99.5]
        data["slow_ma_30"] = [102.0]
        data["rsi_14"] = [75.0]  # RSI overbought (bearish)
    else:  # HOLD
        # Neutral indicators
        data["fast_ma_5"] = [100.0]
        data["slow_ma_20"] = [100.0]
        data["fast_ma_10"] = [100.0]
        data["slow_ma_30"] = [100.0]
        data["rsi_14"] = [50.0]  # RSI neutral

    return pd.DataFrame(data)


def create_mock_row(timeframe_data, index):
    """Create a mock row for testing"""
    return {
        "_df_attrs": {"timeframe_data": timeframe_data},
        "_index": index,
        "Close": 100,
    }


if __name__ == "__main__":
    demo_majority_voting()
