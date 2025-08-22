"""
Statistical Clusters Strategy Implementation
Main strategy that uses the statistical method clusters
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from statistical_clusters import StatisticalClusters
from datetime import timezone


class StatisticalClustersStrategy:
    """Strategy that implements statistical method clusters"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clusters = StatisticalClusters(config)
        self.name = "Statistical Clusters Strategy"

        # Get strategy mode and configuration
        self.mode = config.get("cluster_strategy", {}).get("mode", "predefined")
        self.predefined_strategy = config.get("cluster_strategy", {}).get(
            "predefined_strategy", "pairs_trading"
        )

        print("🚀 Statistical Clusters Strategy initialized")
        print(f"   Currency: ₹ (Indian Rupees)")
        print(f"   Mode: {self.mode}")
        if self.mode == "predefined":
            strategy_desc = self._get_strategy_description()
            print(f"   Strategy: {self.predefined_strategy}")
            print(f"   Description: {strategy_desc}")

    def _get_strategy_description(self) -> str:
        """Get description of the current predefined strategy"""
        strategies = self.config.get("predefined_strategies", {})
        strategy_config = strategies.get(self.predefined_strategy, {})
        return strategy_config.get("description", "Statistical trading strategy")

    def generate_signal(self, data: Dict[str, pd.DataFrame]) -> str:
        """Generate trading signal using statistical clusters"""
        # Use the primary timeframe data
        primary_timeframe = self.config.get("primary_timeframe", "1d")
        if primary_timeframe not in data:
            primary_timeframe = list(data.keys())[0]  # Use first available timeframe

        df = data[primary_timeframe]

        if len(df) < 50:  # Need sufficient data
            return "HOLD"

        # Generate signal using clusters
        signal_data = self.clusters.generate_signal(df)
        signal = signal_data.get("signal", "HOLD")

        # Print detailed analysis
        self._print_analysis(signal_data, df)

        return signal

    def _print_analysis(self, signal_data: Dict[str, Any], df: pd.DataFrame):
        """Print detailed analysis of the signal generation"""
        signal = signal_data.get("signal", "HOLD")
        strength = signal_data.get("signal_strength", 0.0)
        cluster_signals = signal_data.get("cluster_signals", {})

        print(f"\n{'=' * 80}")
        print(f"📊 STATISTICAL CLUSTERS ANALYSIS")
        print(f"{'=' * 80}")

        # Current market data
        current_price = df["Close"].iloc[-1]
        prev_price = df["Close"].iloc[-2] if len(df) > 1 else current_price
        price_change = (current_price - prev_price) / prev_price * 100

        print(f"💰 Market Data:")
        print(f"   Current Price: ₹{current_price:.2f}")
        print(f"   Price Change: {price_change:+.2f}%")

        # Optional: Show aggregated news sentiment (does not affect decision)
        try:
            ml_cfg = self.config.get("ml_algorithms", {})
            news_cfg = ml_cfg.get("news_sentiment", {})
            if news_cfg.get("enabled", False):
                from news.news_provider import NewsProvider  # lazy import, optional

                provider = NewsProvider(news_cfg.get("csv_path", "data/news.csv"))
                # Use the last index timestamp as the reference
                ts = pd.to_datetime(df.index[-1])
                if ts.tzinfo is None:
                    ts = ts.tz_localize(timezone.utc)
                else:
                    ts = ts.tz_convert(timezone.utc)
                agg = provider.aggregate_sentiment(
                    ts,
                    news_cfg.get("window_hours", 24),
                    self.config.get("ticker", None),
                )
                print("\n📰 News Sentiment (last {}h):".format(news_cfg.get("window_hours", 24)))
                print("   Average: {:.2f} | Items: {}".format(agg.get("avg", 0.0), agg.get("count", 0)))
        except Exception as _e:
            # Keep resilient; news is optional
            pass

        # Show active clusters and their signals
        print(f"\n🔍 Active Clusters Analysis:")
        for cluster_name, cluster_data in cluster_signals.items():
            cluster_signal = cluster_data.get("signal", "HOLD")
            cluster_strength = cluster_data.get("signal_strength", 0.0)
            reason = cluster_data.get("reason", "No reason provided")

            # Get cluster description
            cluster_desc = self._get_cluster_description(cluster_name)

            print(
                f"   {cluster_name}: {cluster_signal} (strength: {cluster_strength:.2f})"
            )
            print(f"     └─ {cluster_desc}")
            print(f"     └─ Reason: {reason}")

        # Final signal
        print(f"\n🎯 Final Signal:")
        print(f"   Signal: {signal}")
        print(f"   Strength: {strength:.2f}")
        print(f"   Buy Score: {signal_data.get('buy_score', 0.0):.2f}")
        print(f"   Sell Score: {signal_data.get('sell_score', 0.0):.2f}")
        print(f"   Hold Score: {signal_data.get('hold_score', 0.0):.2f}")

        # Signal emoji and description
        if signal == "BUY":
            print(f"   🟢 BUY RECOMMENDATION")
        elif signal == "SELL":
            print(f"   🔴 SELL RECOMMENDATION")
        else:
            print(f"   🟡 HOLD RECOMMENDATION")

    def _get_cluster_description(self, cluster_name: str) -> str:
        """Get description of a cluster"""
        descriptions = {
            "cluster_1": "Mean Reversion / Pairs Trading",
            "cluster_2": "Momentum & Trend Following",
            "cluster_3": "Volatility Trading",
            "cluster_4": "Multi-Factor / Statistical Arbitrage",
            "cluster_5": "Regime Detection & Adaptive Strategies",
            "cluster_6": "Execution & Microstructure-Aware Trading",
            "cluster_7": "Robustness & Validation",
        }
        return descriptions.get(cluster_name, "Unknown cluster")


def get_strategy_info() -> Dict[str, Any]:
    """Return information about available strategies"""
    return {
        "strategy_name": "Statistical Clusters Strategy",
        "description": "Advanced statistical trading using 7 clusters of methods",
        "currency": "INR (₹)",
        "predefined_strategies": {
            "pairs_trading": {
                "name": "Pairs Trading System",
                "description": "Mean reversion with robust validation",
                "clusters": ["Mean Reversion", "Validation"],
                "best_for": "Equities, ETFs, commodities with fundamental linkages",
            },
            "volatility_breakout": {
                "name": "Volatility Breakout System",
                "description": "Volatility trading with regime detection",
                "clusters": ["Volatility Trading", "Regime Detection", "Validation"],
                "best_for": "Options, volatility ETFs, risk hedging",
            },
            "factor_portfolio": {
                "name": "Factor Portfolio",
                "description": "Multi-factor statistical arbitrage",
                "clusters": ["Multi-Factor", "Validation"],
                "best_for": "Equities, cross-asset relative value strategies",
            },
            "adaptive_trend": {
                "name": "Adaptive Trend Following",
                "description": "Momentum with adaptive regime detection",
                "clusters": ["Momentum", "Regime Detection", "Validation"],
                "best_for": "FX, equity indices, liquid futures",
            },
        },
        "clusters": {
            "cluster_1": "Mean Reversion / Pairs Trading - Cointegration, ADF/KPSS, Kalman Filter",
            "cluster_2": "Momentum & Trend Following - ACF/PACF, ARIMA, Granger causality",
            "cluster_3": "Volatility Trading - GARCH, Markov Switching, VaR/CVaR",
            "cluster_4": "Multi-Factor / Statistical Arbitrage - PCA/ICA, Factor models, Bayesian",
            "cluster_5": "Regime Detection & Adaptive - HMM, State-Space, Walk-forward",
            "cluster_6": "Execution & Microstructure - Order book, VWAP/TWAP, Market impact",
            "cluster_7": "Robustness & Validation - Bootstrap, Reality Check, Out-of-sample",
        },
    }
