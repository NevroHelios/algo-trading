"""
Statistical Method Clusters for Algorithmic Trading
Implementation of 7 clusters of statistical methods with predefined combinations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Optional import for news provider (non-fatal if missing)
try:
    from news.news_provider import NewsProvider  # type: ignore

    _NEWS_AVAILABLE = True
except Exception:
    NewsProvider = None  # type: ignore
    _NEWS_AVAILABLE = False

# Statistical libraries
try:
    from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.regime_switching.markov_autoregression import (
        MarkovAutoregression,
    )
    from sklearn.decomposition import PCA, FastICA
    from sklearn.covariance import LedoitWolf
    from scipy import stats
    from arch import arch_model

    ADVANCED_STATS_AVAILABLE = True
except ImportError:
    ADVANCED_STATS_AVAILABLE = False
    print(
        "Advanced statistical libraries not available. Some features will be limited."
    )


class StatisticalClusters:
    """Main class for implementing statistical method clusters"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cluster_config = config.get("cluster_strategy", {})
        self.mode = self.cluster_config.get("mode", "predefined")
        self.predefined_strategy = self.cluster_config.get(
            "predefined_strategy", "pairs_trading"
        )
        self.custom_clusters = self.cluster_config.get("custom_clusters", [])
        self.cluster_weights = self.cluster_config.get("cluster_weights", {})

        # Initialize clusters
        self.clusters = {
            "cluster_1": MeanReversionCluster(
                config.get("cluster_1_mean_reversion", {})
            ),
            "cluster_2": MomentumCluster(config.get("cluster_2_momentum", {})),
            "cluster_3": VolatilityCluster(config.get("cluster_3_volatility", {})),
            "cluster_4": MultiFactorCluster(config.get("cluster_4_multi_factor", {})),
            "cluster_5": RegimeDetectionCluster(
                config.get("cluster_5_regime_detection", {})
            ),
            "cluster_6": ExecutionCluster(config.get("cluster_6_execution", {})),
            "cluster_7": ValidationCluster(config.get("cluster_7_validation", {})),
        }

        # Active clusters based on mode
        self.active_clusters = self._get_active_clusters()

        print("Statistical Clusters initialized:")
        print(f"   Mode: {self.mode}")
        if self.mode == "predefined":
            print(f"   Strategy: {self.predefined_strategy}")
        print(f"   Active clusters: {self.active_clusters}")

    def _get_active_clusters(self) -> List[str]:
        """Get list of active clusters based on mode and configuration"""
        if self.mode == "predefined":
            strategies = self.config.get("predefined_strategies", {})
            strategy_config = strategies.get(self.predefined_strategy, {})
            return strategy_config.get("active_clusters", ["cluster_1", "cluster_7"])
        elif self.mode == "custom":
            return self.custom_clusters
        else:  # multi_timeframe fallback
            return ["cluster_2", "cluster_7"]  # Momentum + Validation

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal using active clusters"""
        signals = {}
        cluster_scores = {}

        # Generate signals from each active cluster
        for cluster_name in self.active_clusters:
            if cluster_name in self.clusters:
                try:
                    cluster = self.clusters[cluster_name]
                    if cluster.enabled:
                        signal_data = cluster.generate_signal(data)
                        signals[cluster_name] = signal_data
                        cluster_scores[cluster_name] = signal_data.get(
                            "signal_strength", 0.0
                        )
                except Exception as e:
                    print(f"Warning: Error in {cluster_name}: {str(e)}")
                    signals[cluster_name] = {"signal": "HOLD", "signal_strength": 0.0}
                    cluster_scores[cluster_name] = 0.0

        # Combine signals using weights
        final_signal = self._combine_signals(signals, cluster_scores)

        return final_signal

    def _combine_signals(
        self, signals: Dict[str, Any], scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Combine signals from multiple clusters using weighted voting"""
        if not signals:
            return {"signal": "HOLD", "signal_strength": 0.0, "cluster_signals": {}}

        # Get weights for active clusters
        if self.mode == "predefined":
            strategies = self.config.get("predefined_strategies", {})
            strategy_config = strategies.get(self.predefined_strategy, {})
            weights = strategy_config.get("cluster_weights", {})
        else:
            weights = self.cluster_weights

    # Calculate weighted scores
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_weight = 0.0

        for cluster_name, signal_data in signals.items():
            weight = weights.get(cluster_name, 1.0 / len(signals))
            strength = signal_data.get("signal_strength", 0.0)
            signal = signal_data.get("signal", "HOLD")

            weighted_strength = weight * abs(strength)
            total_weight += weight

            if signal == "BUY":
                buy_score += weighted_strength
            elif signal == "SELL":
                sell_score += weighted_strength
            else:
                hold_score += weighted_strength

        # Optionally blend in news sentiment before normalization
        buy_score, sell_score, hold_score, total_weight = self._inject_news_adjustment(
            data=signals[self.active_clusters[0]].get("data_frame") if isinstance(signals.get(self.active_clusters[0], {}), dict) else None,
            df_hint=None,
            buy_score=buy_score,
            sell_score=sell_score,
            hold_score=hold_score,
            total_weight=total_weight,
        )

        # Normalize scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
            hold_score /= total_weight

        # Determine final signal
        max_score = max(buy_score, sell_score, hold_score)
        if max_score == buy_score and buy_score > 0.5:
            final_signal = "BUY"
            final_strength = buy_score
        elif max_score == sell_score and sell_score > 0.5:
            final_signal = "SELL"
            final_strength = sell_score
        else:
            final_signal = "HOLD"
            final_strength = hold_score

        return {
            "signal": final_signal,
            "signal_strength": final_strength,
            "cluster_signals": signals,
            "cluster_scores": scores,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "hold_score": hold_score,
        }

    def _inject_news_adjustment(
        self,
        data: Optional[pd.DataFrame],
        df_hint: Optional[pd.DataFrame],
        buy_score: float,
        sell_score: float,
        hold_score: float,
        total_weight: float,
    ) -> tuple[float, float, float, float]:
        """Optionally adjust scores using aggregated news sentiment.

        This only runs when cluster_strategy.news_integration.enabled is true.
        It does not override cluster votes; it adds a small weighted bias based on
        recent average sentiment.
        """
        try:
            ns_cfg = (
                self.config.get("cluster_strategy", {}).get("news_integration", {})
            )
            if not ns_cfg or not ns_cfg.get("enabled", False):
                return buy_score, sell_score, hold_score, total_weight

            if not _NEWS_AVAILABLE:
                return buy_score, sell_score, hold_score, total_weight

            # Determine reference DataFrame and timestamp
            ref_df = data if isinstance(data, pd.DataFrame) else df_hint
            if ref_df is None or len(ref_df.index) == 0:
                return buy_score, sell_score, hold_score, total_weight

            ts = pd.to_datetime(ref_df.index[-1])
            try:
                ts = ts.tz_convert("UTC")  # if tz-aware
            except Exception:
                try:
                    ts = ts.tz_localize("UTC")
                except Exception:
                    pass

            provider = NewsProvider(ns_cfg.get("csv_path", "data/news.csv"))
            agg = provider.aggregate_sentiment(
                ts,
                int(ns_cfg.get("window_hours", 24)),
                self.config.get("ticker"),
            )
            avg = float(agg.get("avg", 0.0))
            n = int(agg.get("count", 0))
            if n < int(ns_cfg.get("min_items", 3)):
                return buy_score, sell_score, hold_score, total_weight

            bull = float(ns_cfg.get("bullish_threshold", 0.25))
            bear = float(ns_cfg.get("bearish_threshold", -0.25))
            w = float(ns_cfg.get("weight", 0.15))  # modest default influence
            # Clamp strength to [0, 1]
            strength = min(abs(avg), 1.0) * w

            # Add a news pseudo-vote into the total weight and respective score
            if avg >= bull:
                buy_score += strength
                total_weight += w
            elif avg <= bear:
                sell_score += strength
                total_weight += w
            else:
                hold_score += strength * 0.5
                total_weight += w * 0.5

            return buy_score, sell_score, hold_score, total_weight
        except Exception:
            # Silent failure to keep robustness
            return buy_score, sell_score, hold_score, total_weight


class MeanReversionCluster:
    """Cluster 1: Mean Reversion / Pairs Trading"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.entry_threshold = config.get("entry_threshold", 2.0)
        self.exit_threshold = config.get("exit_threshold", 0.5)
        self.stop_loss = config.get("stop_loss", -3.0)
        self.lookback = config.get("cointegration", {}).get("lookback_period", 252)

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate mean reversion signal"""
        if len(data) < self.lookback:
            return {
                "signal": "HOLD",
                "signal_strength": 0.0,
                "reason": "Insufficient data",
            }

        try:
            # Calculate rolling mean and std
            prices = data["Close"].values
            rolling_mean = pd.Series(prices).rolling(window=20).mean()
            rolling_std = pd.Series(prices).rolling(window=20).std()

            # Calculate Z-score
            current_price = prices[-1]
            current_mean = rolling_mean.iloc[-1]
            current_std = rolling_std.iloc[-1]

            if current_std == 0 or pd.isna(current_std):
                return {
                    "signal": "HOLD",
                    "signal_strength": 0.0,
                    "reason": "Zero volatility",
                }

            z_score = (current_price - current_mean) / current_std

            # Generate signal based on z-score
            if z_score > self.entry_threshold:
                signal = "SELL"  # Price too high, expect reversion
                strength = min(abs(z_score) / 4.0, 1.0)  # Normalize to 0-1
            elif z_score < -self.entry_threshold:
                signal = "BUY"  # Price too low, expect reversion
                strength = min(abs(z_score) / 4.0, 1.0)
            elif abs(z_score) < self.exit_threshold:
                signal = "HOLD"  # Near mean, no strong signal
                strength = 0.1
            else:
                signal = "HOLD"
                strength = 0.0

            return {
                "signal": signal,
                "signal_strength": strength,
                "z_score": z_score,
                "current_price": current_price,
                "mean": current_mean,
                "std": current_std,
                "reason": f"Z-score: {z_score:.2f}",
            }

        except Exception as e:
            return {
                "signal": "HOLD",
                "signal_strength": 0.0,
                "reason": f"Error: {str(e)}",
            }


class MomentumCluster:
    """Cluster 2: Momentum & Trend Following"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.lookback_period = config.get("lookback_period", 20)
        self.momentum_threshold = config.get("momentum_threshold", 0.02)

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate momentum signal"""
        if len(data) < self.lookback_period + 1:
            return {
                "signal": "HOLD",
                "signal_strength": 0.0,
                "reason": "Insufficient data",
            }

        try:
            prices = data["Close"].values
            returns = np.diff(np.log(prices))

            # Calculate momentum (price change over lookback period)
            current_price = prices[-1]
            past_price = prices[-self.lookback_period - 1]
            momentum = (current_price - past_price) / past_price

            # Calculate trend strength using moving averages
            ma_short = np.mean(prices[-5:])  # 5-period MA
            ma_long = np.mean(prices[-20:])  # 20-period MA
            trend_strength = (ma_short - ma_long) / ma_long

            # Generate signal
            if momentum > self.momentum_threshold and trend_strength > 0:
                signal = "BUY"
                strength = min(abs(momentum) * 10, 1.0)  # Scale momentum
            elif momentum < -self.momentum_threshold and trend_strength < 0:
                signal = "SELL"
                strength = min(abs(momentum) * 10, 1.0)
            else:
                signal = "HOLD"
                strength = 0.0

            return {
                "signal": signal,
                "signal_strength": strength,
                "momentum": momentum,
                "trend_strength": trend_strength,
                "reason": f"Momentum: {momentum:.3f}, Trend: {trend_strength:.3f}",
            }

        except Exception as e:
            return {
                "signal": "HOLD",
                "signal_strength": 0.0,
                "reason": f"Error: {str(e)}",
            }


class VolatilityCluster:
    """Cluster 3: Volatility Trading"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.vol_lookback = config.get("vol_lookback", 30)
        self.vol_threshold_low = config.get("vol_threshold_low", 0.15)
        self.vol_threshold_high = config.get("vol_threshold_high", 0.35)

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate volatility-based signal"""
        if len(data) < self.vol_lookback + 1:
            return {
                "signal": "HOLD",
                "signal_strength": 0.0,
                "reason": "Insufficient data",
            }

        try:
            prices = data["Close"].values
            returns = np.diff(np.log(prices))

            # Calculate rolling volatility (annualized)
            vol_window = returns[-self.vol_lookback :]
            current_vol = np.std(vol_window) * np.sqrt(252)  # Annualized

            # Calculate average volatility
            avg_vol = np.std(returns) * np.sqrt(252)

            # Volatility regime detection
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

            # Generate signal based on volatility regime
            if current_vol < self.vol_threshold_low:
                # Low volatility - expect breakout
                signal = "BUY"  # Bias towards long in low vol
                strength = (
                    self.vol_threshold_low - current_vol
                ) / self.vol_threshold_low
            elif current_vol > self.vol_threshold_high:
                # High volatility - expect mean reversion
                signal = "SELL"  # Bias towards short in high vol
                strength = (current_vol - self.vol_threshold_high) / current_vol
            else:
                signal = "HOLD"
                strength = 0.0

            return {
                "signal": signal,
                "signal_strength": min(strength, 1.0),
                "current_vol": current_vol,
                "avg_vol": avg_vol,
                "vol_ratio": vol_ratio,
                "reason": f"Vol: {current_vol:.2f} (Avg: {avg_vol:.2f})",
            }

        except Exception as e:
            return {
                "signal": "HOLD",
                "signal_strength": 0.0,
                "reason": f"Error: {str(e)}",
            }


class MultiFactorCluster:
    """Cluster 4: Multi-Factor / Statistical Arbitrage"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.n_components = config.get("pca", {}).get("n_components", 5)
        self.lookback = 100

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate multi-factor signal"""
        if len(data) < self.lookback:
            return {
                "signal": "HOLD",
                "signal_strength": 0.0,
                "reason": "Insufficient data",
            }

        try:
            # Create simple factors from price data
            prices = data["Close"].values
            returns = np.diff(np.log(prices))

            # Factor 1: Momentum (20-day return)
            momentum = (prices[-1] / prices[-21] - 1) if len(prices) > 21 else 0

            # Factor 2: Mean reversion (deviation from 50-day MA)
            ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
            mean_reversion = (prices[-1] - ma_50) / ma_50

            # Factor 3: Volatility factor
            vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else 0
            vol_factor = vol_20 * np.sqrt(252)

            # Simple factor combination
            factor_score = (
                0.4 * momentum - 0.3 * mean_reversion + 0.3 * (1 / (1 + vol_factor))
            )

            # Generate signal
            if factor_score > 0.1:
                signal = "BUY"
                strength = min(abs(factor_score) * 2, 1.0)
            elif factor_score < -0.1:
                signal = "SELL"
                strength = min(abs(factor_score) * 2, 1.0)
            else:
                signal = "HOLD"
                strength = 0.0

            return {
                "signal": signal,
                "signal_strength": strength,
                "factor_score": factor_score,
                "momentum": momentum,
                "mean_reversion": mean_reversion,
                "vol_factor": vol_factor,
                "reason": f"Factor score: {factor_score:.3f}",
            }

        except Exception as e:
            return {
                "signal": "HOLD",
                "signal_strength": 0.0,
                "reason": f"Error: {str(e)}",
            }


class RegimeDetectionCluster:
    """Cluster 5: Regime Detection & Adaptive Strategies"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.regime_threshold = config.get("regime_threshold", 0.7)
        self.adaptation_speed = config.get("adaptation_speed", 0.1)

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate regime-aware signal"""
        if len(data) < 50:
            return {
                "signal": "HOLD",
                "signal_strength": 0.0,
                "reason": "Insufficient data",
            }

        try:
            prices = data["Close"].values
            returns = np.diff(np.log(prices))

            # Simple regime detection using volatility and returns
            recent_returns = returns[-20:]
            recent_vol = np.std(recent_returns)
            recent_mean = np.mean(recent_returns)

            # Historical comparison
            hist_vol = np.std(returns[:-20]) if len(returns) > 40 else recent_vol
            hist_mean = np.mean(returns[:-20]) if len(returns) > 40 else recent_mean

            # Regime classification
            vol_regime = "HIGH" if recent_vol > hist_vol * 1.5 else "LOW"
            trend_regime = (
                "BULL"
                if recent_mean > hist_mean * 1.2
                else "BEAR"
                if recent_mean < hist_mean * 0.8
                else "SIDEWAYS"
            )

            # Adaptive signal generation
            if vol_regime == "LOW" and trend_regime == "BULL":
                signal = "BUY"
                strength = 0.8
            elif vol_regime == "LOW" and trend_regime == "BEAR":
                signal = "SELL"
                strength = 0.8
            elif vol_regime == "HIGH":
                signal = "HOLD"  # Avoid trading in high volatility
                strength = 0.2
            else:
                signal = "HOLD"
                strength = 0.0

            return {
                "signal": signal,
                "signal_strength": strength,
                "vol_regime": vol_regime,
                "trend_regime": trend_regime,
                "recent_vol": recent_vol,
                "recent_mean": recent_mean,
                "reason": f"Regime: {vol_regime}-{trend_regime}",
            }

        except Exception as e:
            return {
                "signal": "HOLD",
                "signal_strength": 0.0,
                "reason": f"Error: {str(e)}",
            }


class ExecutionCluster:
    """Cluster 6: Execution & Microstructure-Aware Trading"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.participation_rate = config.get("execution_algos", {}).get(
            "participation_rate", 0.1
        )

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate execution-aware signal"""
        try:
            # Simple execution signal based on volume patterns
            if "Volume" in data.columns:
                volumes = data["Volume"].values
                avg_volume = (
                    np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
                )
                current_volume = volumes[-1]

                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

                # Higher volume suggests better execution conditions
                if volume_ratio > 1.5:
                    signal = "BUY"  # Good liquidity for buying
                    strength = min((volume_ratio - 1) / 2, 1.0)
                elif volume_ratio < 0.5:
                    signal = "HOLD"  # Poor liquidity
                    strength = 0.1
                else:
                    signal = "HOLD"
                    strength = 0.3
            else:
                signal = "HOLD"
                strength = 0.5
                volume_ratio = 1.0

            return {
                "signal": signal,
                "signal_strength": strength,
                "volume_ratio": volume_ratio,
                "reason": f"Volume ratio: {volume_ratio:.2f}",
            }

        except Exception as e:
            return {
                "signal": "HOLD",
                "signal_strength": 0.0,
                "reason": f"Error: {str(e)}",
            }


class ValidationCluster:
    """Cluster 7: Robustness & Validation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.confidence_level = config.get("confidence_level", 0.95)
        self.window_size = config.get("performance_monitoring", {}).get(
            "window_size", 60
        )

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate validation-based signal modifier"""
        if len(data) < self.window_size:
            return {
                "signal": "HOLD",
                "signal_strength": 0.5,
                "reason": "Insufficient data for validation",
            }

        try:
            prices = data["Close"].values
            returns = np.diff(np.log(prices))

            # Calculate rolling Sharpe ratio
            recent_returns = returns[-self.window_size :]
            sharpe = (
                np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252)
                if np.std(recent_returns) > 0
                else 0
            )

            # Calculate maximum drawdown
            peak = np.maximum.accumulate(prices)
            drawdown = (prices - peak) / peak
            max_drawdown = np.min(drawdown)

            # Validation score (higher is better)
            validation_score = (
                0.5 + 0.3 * min(sharpe / 2, 1) + 0.2 * min(-max_drawdown * 10, 1)
            )
            validation_score = max(0, min(validation_score, 1))

            # This cluster doesn't generate its own signal, but provides confidence
            return {
                "signal": "HOLD",  # Validation cluster doesn't generate signals
                "signal_strength": validation_score,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "validation_score": validation_score,
                "reason": f"Validation score: {validation_score:.2f}",
            }

        except Exception as e:
            return {
                "signal": "HOLD",
                "signal_strength": 0.5,
                "reason": f"Error: {str(e)}",
            }
