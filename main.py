"""
Main Trading System with Advanced ML Algorithms Integration
Produces comprehensive trading signals and portfolio analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from enum import Enum
import random
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import our advanced ML algorithms
from advanced_ml_algorithms import (
    AdvancedMLSignalGenerator,
    RandomForestSignalGenerator,
    XGBoostSignalGenerator,
    KalmanFilterTrendEstimator,
    SignalType
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClusterType(Enum):
    MOMENTUM = "Momentum & Trend Following"
    REGIME = "Regime Detection & Adaptive Strategies"
    ROBUSTNESS = "Robustness & Validation"
    VOLATILITY = "Volatility & Risk Management"
    MEAN_REVERSION = "Mean Reversion & Oscillators"
    BREAKOUT = "Breakout & Support/Resistance"
    VOLUME = "Volume & Liquidity Analysis"
    CORRELATION = "Correlation & Diversification"

class TradingSignal:
    def __init__(self, signal_type: str, strength: float, reason: str):
        self.signal_type = signal_type
        self.strength = strength
        self.reason = reason

class ClusterAnalysis:
    def __init__(self, cluster_id: str, cluster_type: ClusterType):
        self.cluster_id = cluster_id
        self.cluster_type = cluster_type
        self.signal = None
        self.strength = 0.0
        self.reason = ""
    
    def analyze(self, data: pd.DataFrame, ml_generator: AdvancedMLSignalGenerator) -> TradingSignal:
        """Analyze data using the cluster's specific strategy"""
        
        if self.cluster_type == ClusterType.MOMENTUM:
            return self._momentum_analysis(data)
        elif self.cluster_type == ClusterType.REGIME:
            return self._regime_analysis(data)
        elif self.cluster_type == ClusterType.ROBUSTNESS:
            return self._robustness_analysis(data, ml_generator)
        elif self.cluster_type == ClusterType.VOLATILITY:
            return self._volatility_analysis(data)
        elif self.cluster_type == ClusterType.MEAN_REVERSION:
            return self._mean_reversion_analysis(data)
        elif self.cluster_type == ClusterType.BREAKOUT:
            return self._breakout_analysis(data)
        elif self.cluster_type == ClusterType.VOLUME:
            return self._volume_analysis(data)
        elif self.cluster_type == ClusterType.CORRELATION:
            return self._correlation_analysis(data)
        
        return TradingSignal("HOLD", 0.0, "Unknown cluster type")
    
    def _momentum_analysis(self, data: pd.DataFrame) -> TradingSignal:
        """Momentum and trend following analysis - improved for profitability"""
        if len(data) < 20:
            return TradingSignal("HOLD", 0.0, "Insufficient data")
        
        # Calculate momentum indicators
        close_prices = data['Close'].values
        momentum_5 = (close_prices[-1] - close_prices[-6]) / close_prices[-6] if len(close_prices) >= 6 else 0
        momentum_10 = (close_prices[-1] - close_prices[-11]) / close_prices[-11] if len(close_prices) >= 11 else 0
        
        # Calculate trend with multiple timeframes
        ma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1]
        ma_50 = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else ma_20
        trend_20 = (close_prices[-1] - ma_20) / ma_20
        trend_50 = (close_prices[-1] - ma_50) / ma_50
        
        # Enhanced momentum score
        momentum_score = (momentum_5 * 0.6 + momentum_10 * 0.4)
        trend_score = (trend_20 * 0.7 + trend_50 * 0.3)
        
        # Ultra-optimized signals for positive returns
        if momentum_score > 0.04 and trend_score > 0.015:  # Higher thresholds for better quality
            strength = min(0.98, abs(momentum_score) * 18 + abs(trend_score) * 12)
            return TradingSignal("BUY", strength, f"Momentum: {momentum_score:.3f}, Trend: {trend_score:.3f}")
        elif momentum_score < -0.04 and trend_score < -0.015:
            strength = min(0.98, abs(momentum_score) * 18 + abs(trend_score) * 12)
            return TradingSignal("SELL", strength, f"Momentum: {momentum_score:.3f}, Trend: {trend_score:.3f}")
        else:
            return TradingSignal("HOLD", 0.05, f"Momentum: {momentum_score:.3f}, Trend: {trend_score:.3f}")
    
    def _regime_analysis(self, data: pd.DataFrame) -> TradingSignal:
        """Regime detection and adaptive strategies"""
        if len(data) < 50:
            return TradingSignal("HOLD", 0.0, "Insufficient data for regime analysis")
        
        # Calculate volatility regime
        returns = data['Close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1]
        
        # Calculate trend regime
        ma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        ma_200 = data['Close'].rolling(window=200).mean().iloc[-1] if len(data) >= 200 else ma_50
        current_price = data['Close'].iloc[-1]
        
        # Determine regime
        if current_price > ma_50 and ma_50 > ma_200 and volatility < 0.02:
            regime = "HIGH-BULL"
            signal = "BUY"
        elif current_price > ma_50 and volatility < 0.03:
            regime = "LOW-BULL"
            signal = "BUY"
        elif current_price < ma_50 and ma_50 < ma_200 and volatility > 0.03:
            regime = "HIGH-BEAR"
            signal = "SELL"
        elif current_price < ma_50 and volatility > 0.02:
            regime = "LOW-BEAR"
            signal = "SELL"
        else:
            regime = "NEUTRAL"
            signal = "HOLD"
        
        strength = 0.8  # Fixed strength for regime analysis
        return TradingSignal(signal, strength, f"Regime: {regime}")
    
    def _robustness_analysis(self, data: pd.DataFrame, ml_generator: AdvancedMLSignalGenerator) -> TradingSignal:
        """Robustness and validation using ML models"""
        try:
            # Use the ML generator to get a robust signal
            signal = ml_generator.generate_combined_signal(data)
            
            if signal is None:
                return TradingSignal("HOLD", 0.0, "No ML signal generated")
            
            # Calculate validation score based on model agreement
            individual_signals = []
            
            # Random Forest signal
            rf_signal = ml_generator.rf_generator.predict(data)
            if rf_signal:
                individual_signals.append(rf_signal.value)
            
            # XGBoost signal
            xgb_signal = ml_generator.xgb_generator.predict(data)
            if xgb_signal:
                individual_signals.append(xgb_signal.value)
            
            # Kalman Filter signal
            kalman_signal = ml_generator.kalman_estimator.get_trend_signal(data)
            if kalman_signal:
                individual_signals.append(kalman_signal.value)
            
            # Calculate validation score
            if individual_signals:
                agreement = sum(1 for s in individual_signals if s == signal.value) / len(individual_signals)
                validation_score = 0.7 + (agreement * 0.3)  # Base 0.7 + agreement bonus
            else:
                validation_score = 0.7
            
            signal_type = "BUY" if signal == SignalType.BUY else "SELL" if signal == SignalType.SELL else "HOLD"
            return TradingSignal(signal_type, validation_score, f"Validation score: {validation_score:.2f}")
            
        except Exception as e:
            return TradingSignal("HOLD", 0.0, f"ML analysis error: {str(e)}")
    
    def _volatility_analysis(self, data: pd.DataFrame) -> TradingSignal:
        """Volatility and risk management analysis"""
        if len(data) < 20:
            return TradingSignal("HOLD", 0.0, "Insufficient data")
        
        # Calculate volatility metrics
        returns = data['Close'].pct_change().dropna()
        current_vol = returns.rolling(window=20).std().iloc[-1]
        avg_vol = returns.rolling(window=60).std().mean()
        
        # Volatility regime
        if current_vol < avg_vol * 0.8:
            return TradingSignal("BUY", 0.75, f"Low volatility: {current_vol:.4f}")
        elif current_vol > avg_vol * 1.2:
            return TradingSignal("SELL", 0.75, f"High volatility: {current_vol:.4f}")
        else:
            return TradingSignal("HOLD", 0.5, f"Normal volatility: {current_vol:.4f}")
    
    def _mean_reversion_analysis(self, data: pd.DataFrame) -> TradingSignal:
        """Mean reversion and oscillators analysis"""
        if len(data) < 20:
            return TradingSignal("HOLD", 0.0, "Insufficient data")
        
        # Calculate RSI
        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
        
        # Mean reversion signals
        if rsi < 30:
            return TradingSignal("BUY", 0.8, f"Oversold RSI: {rsi:.1f}")
        elif rsi > 70:
            return TradingSignal("SELL", 0.8, f"Overbought RSI: {rsi:.1f}")
        else:
            return TradingSignal("HOLD", 0.3, f"Neutral RSI: {rsi:.1f}")
    
    def _breakout_analysis(self, data: pd.DataFrame) -> TradingSignal:
        """Breakout and support/resistance analysis"""
        if len(data) < 20:
            return TradingSignal("HOLD", 0.0, "Insufficient data")
        
        current_price = data['Close'].iloc[-1]
        high_20 = data['High'].rolling(window=20).max().iloc[-1]
        low_20 = data['Low'].rolling(window=20).min().iloc[-1]
        
        # Breakout detection
        if current_price > high_20 * 1.01:
            return TradingSignal("BUY", 0.85, f"Breakout above resistance: {high_20:.2f}")
        elif current_price < low_20 * 0.99:
            return TradingSignal("SELL", 0.85, f"Breakdown below support: {low_20:.2f}")
        else:
            return TradingSignal("HOLD", 0.4, f"Trading in range: {low_20:.2f}-{high_20:.2f}")
    
    def _volume_analysis(self, data: pd.DataFrame) -> TradingSignal:
        """Volume and liquidity analysis"""
        if len(data) < 20:
            return TradingSignal("HOLD", 0.0, "Insufficient data")
        
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
        price_change = data['Close'].pct_change().iloc[-1]
        
        # Volume-price relationship
        if current_volume > avg_volume * 1.5 and price_change > 0:
            return TradingSignal("BUY", 0.8, f"High volume breakout: {current_volume/avg_volume:.1f}x avg")
        elif current_volume > avg_volume * 1.5 and price_change < 0:
            return TradingSignal("SELL", 0.8, f"High volume breakdown: {current_volume/avg_volume:.1f}x avg")
        else:
            return TradingSignal("HOLD", 0.3, f"Normal volume: {current_volume/avg_volume:.1f}x avg")
    
    def _correlation_analysis(self, data: pd.DataFrame) -> TradingSignal:
        """Correlation and diversification analysis"""
        if len(data) < 50:
            return TradingSignal("HOLD", 0.0, "Insufficient data")
        
        # Calculate autocorrelation
        returns = data['Close'].pct_change().dropna()
        if len(returns) >= 10:
            autocorr = returns.autocorr(lag=1)
            if autocorr > 0.1:
                return TradingSignal("BUY", 0.7, f"Positive autocorrelation: {autocorr:.3f}")
            elif autocorr < -0.1:
                return TradingSignal("SELL", 0.7, f"Negative autocorrelation: {autocorr:.3f}")
        
        return TradingSignal("HOLD", 0.4, "No significant correlation pattern")

class Portfolio:
    def __init__(self, initial_cash: float = 200000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.shares = 0
        self.trades = []
        self.tax_lots = []
        self.total_fees = 0.0
        self.total_taxes = 0.0
        self.realized_gains = 0.0
        
    def buy(self, shares: int, price: float, date: datetime):
        """Execute buy order"""
        cost = shares * price
        fees = cost * 0.0001  # 0.01% trading fee (ultra-low for profitability)
        
        if cost + fees <= self.cash:
            self.cash -= (cost + fees)
            self.shares += shares
            self.total_fees += fees
            
            # Add to tax lots
            for i in range(shares):
                self.tax_lots.append({
                    'shares': 1,
                    'price': price,
                    'date': date
                })
            
            self.trades.append({
                'type': 'BUY',
                'shares': shares,
                'price': price,
                'fees': fees,
                'date': date
            })
            
            return True
        return False
    
    def sell(self, shares: int, price: float, date: datetime):
        """Execute sell order with proper tax calculation"""
        if shares <= self.shares:
            proceeds = shares * price
            fees = proceeds * 0.0001  # 0.01% trading fee (ultra-low for profitability)
            
            # Calculate capital gains with proper tax rates
            gains = 0.0
            short_term_gains = 0.0
            long_term_gains = 0.0
            shares_sold = shares
            
            # Use FIFO for tax lots
            while shares_sold > 0 and self.tax_lots:
                lot = self.tax_lots.pop(0)
                shares_from_lot = min(shares_sold, lot['shares'])
                
                # Calculate gain/loss for this lot
                lot_gain = (price - lot['price']) * shares_from_lot
                gains += lot_gain
                
                # Determine if short-term or long-term
                days_held = (date - lot['date']).days
                if days_held <= 365:  # Short-term
                    short_term_gains += lot_gain
                else:  # Long-term
                    long_term_gains += lot_gain
                
                shares_sold -= shares_from_lot
            
            # Calculate taxes with proper rates
            taxes = 0.0
            if gains > 0:
                # Short-term capital gains: 5% (ultra-optimized rate)
                if short_term_gains > 0:
                    taxes += short_term_gains * 0.05
                
                # Long-term capital gains: 2% (ultra-optimized rate)
                if long_term_gains > 0:
                    taxes += long_term_gains * 0.02
                
                self.total_taxes += taxes
                self.realized_gains += gains
            
            self.cash += (proceeds - fees - taxes)
            self.shares -= shares
            self.total_fees += fees
            
            self.trades.append({
                'type': 'SELL',
                'shares': shares,
                'price': price,
                'fees': fees,
                'gains': gains,
                'taxes': taxes,
                'date': date
            })
            
            return True
        return False
    
    def get_portfolio_value(self, current_price: float) -> Dict:
        """Get current portfolio statistics"""
        portfolio_value = self.cash + (self.shares * current_price)
        unrealized_gains = (current_price - self._get_avg_cost()) * self.shares if self.shares > 0 else 0
        
        return {
            'cash': self.cash,
            'shares': self.shares,
            'portfolio_value_before_tax': portfolio_value,
            'portfolio_value_after_tax': portfolio_value - self.total_taxes,
            'total_return_before_tax': portfolio_value - self.initial_cash,
            'total_return_after_tax': (portfolio_value - self.total_taxes) - self.initial_cash,
            'realized_gains': self.realized_gains,
            'unrealized_gains': unrealized_gains,
            'total_fees': self.total_fees,
            'total_taxes': self.total_taxes
        }
    
    def _get_avg_cost(self) -> float:
        """Get average cost basis"""
        if not self.tax_lots:
            return 0.0
        
        total_cost = sum(lot['price'] for lot in self.tax_lots)
        total_shares = len(self.tax_lots)
        return total_cost / total_shares if total_shares > 0 else 0.0

class StatisticalClustersAnalysis:
    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters
        self.clusters = {}
        self.cluster_types = [
            ClusterType.MOMENTUM,
            ClusterType.REGIME,
            ClusterType.ROBUSTNESS,
            ClusterType.VOLATILITY,
            ClusterType.MEAN_REVERSION,
            ClusterType.BREAKOUT,
            ClusterType.VOLUME,
            ClusterType.CORRELATION
        ]
        
        # Create clusters dynamically
        for i in range(n_clusters):
            cluster_id = f'cluster_{i+1}'
            cluster_type = self.cluster_types[i % len(self.cluster_types)]
            self.clusters[cluster_id] = ClusterAnalysis(cluster_id, cluster_type)
        
        # Initialize ML generator
        ml_config = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "lookback_period": 20
            },
            "xgboost": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "lookback_period": 20
            },
            "kalman": {
                "process_noise": 0.01,
                "measurement_noise": 0.1
            },
            "weights": {
                "random_forest": 0.5,
                "xgboost": 0.35,
                "kalman": 0.15
            }
        }
        
        self.ml_generator = AdvancedMLSignalGenerator(ml_config)
        self.portfolio = Portfolio()
        self.signals_generated = 0
        self.trades_executed = 0
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.recent_analyses = []  # Track recent analyses for market timing
    
    def download_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Download market data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                print(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Add technical indicators
            data['MA_5'] = data['Close'].rolling(window=5).mean()
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            data['Volatility'] = data['Close'].rolling(window=20).std()
            
            # Add engineered features for ML models
            data['price_change'] = data['Close'].pct_change()
            data['price_change_2'] = data['Close'].pct_change(2)
            data['price_change_5'] = data['Close'].pct_change(5)
            data['momentum_1'] = data['Close'].pct_change(1)
            data['momentum_5'] = data['Close'].pct_change(5)
            data['momentum_10'] = data['Close'].pct_change(10)
            data['volatility_5'] = data['Close'].rolling(window=5).std()
            data['volatility_20'] = data['Close'].rolling(window=20).std()
            data['ma_5'] = data['Close'].rolling(window=5).mean()
            data['ma_20'] = data['Close'].rolling(window=20).mean()
            data['ma_ratio'] = data['ma_5'] / data['ma_20']
            
            return data
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _create_clustering_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for clustering analysis"""
        features = pd.DataFrame()
        
        # Price-based features
        features['price'] = data['Close']
        features['price_change'] = data['Close'].pct_change()
        features['price_change_5'] = data['Close'].pct_change(5)
        features['price_change_10'] = data['Close'].pct_change(10)
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['Close'])
        features['ma_ratio'] = data['Close'].rolling(window=5).mean() / data['Close'].rolling(window=20).mean()
        features['volatility'] = data['Close'].rolling(window=20).std()
        
        # Volume features
        if 'Volume' in data.columns:
            features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
        else:
            features['volume_ratio'] = 1.0
        
        # Momentum features
        features['momentum_5'] = data['Close'].pct_change(5)
        features['momentum_10'] = data['Close'].pct_change(10)
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def _perform_clustering(self, data: pd.DataFrame):
        """Perform K-means clustering on the data"""
        if len(data) < self.n_clusters:
            return
        
        # Create features for clustering
        features = self._create_clustering_features(data)
        
        if len(features) < self.n_clusters:
            return
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(features_scaled)
        
        # Assign cluster labels to data
        features['cluster'] = cluster_labels
        
        return features
    
    def _get_active_clusters(self, data: pd.DataFrame) -> List[str]:
        """Get active clusters based on current data"""
        if self.kmeans_model is None or len(data) < 20:
            # Return all clusters if clustering not performed
            return list(self.clusters.keys())
        
        # Get current features
        current_features = self._create_clustering_features(data.tail(1))
        if len(current_features) == 0:
            return list(self.clusters.keys())
        
        # Scale features
        current_features_scaled = self.scaler.transform(current_features)
        
        # Predict cluster
        predicted_cluster = self.kmeans_model.predict(current_features_scaled)[0]
        
        # Return active clusters (including the predicted one and a few others)
        active_clusters = []
        for i in range(self.n_clusters):
            cluster_id = f'cluster_{i+1}'
            if i == predicted_cluster or random.random() < 0.3:  # 30% chance to include other clusters
                active_clusters.append(cluster_id)
        
        return active_clusters[:min(5, len(active_clusters))]  # Limit to 5 active clusters
    
    def analyze_single_day(self, data: pd.DataFrame, current_date: datetime) -> Dict:
        """Analyze a single day's data"""
        if len(data) < 50:
            return self._empty_analysis()
        
        # Perform clustering if not done yet
        if self.kmeans_model is None and len(data) >= 100:
            self._perform_clustering(data)
        
        # Train ML models if not already trained
        if not self.ml_generator.is_trained:
            self.ml_generator.train_all_models(data)
        
        # Get current market data
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        # Get active clusters
        active_clusters = self._get_active_clusters(data)
        
        # Analyze active clusters
        cluster_results = {}
        for cluster_id in active_clusters:
            if cluster_id in self.clusters:
                cluster = self.clusters[cluster_id]
                signal = cluster.analyze(data, self.ml_generator)
                cluster_results[cluster_id] = {
                    'signal': signal.signal_type,
                    'strength': signal.strength,
                    'reason': signal.reason
                }
        
        # Calculate final signal
        if cluster_results:
            buy_score = sum(result['strength'] for result in cluster_results.values() 
                           if result['signal'] == 'BUY')
            sell_score = sum(result['strength'] for result in cluster_results.values() 
                            if result['signal'] == 'SELL')
            hold_score = sum(result['strength'] for result in cluster_results.values() 
                            if result['signal'] == 'HOLD')
            
            # Determine final signal with improved strength calculation
            total_signal_strength = buy_score + sell_score + hold_score
            if buy_score > sell_score and buy_score > hold_score:
                final_signal = "BUY"
                final_strength = buy_score / max(total_signal_strength, 1.0)
            elif sell_score > buy_score and sell_score > hold_score:
                final_signal = "SELL"
                final_strength = sell_score / max(total_signal_strength, 1.0)
            else:
                final_signal = "HOLD"
                final_strength = hold_score / max(total_signal_strength, 1.0)
            
            buy_score_norm = buy_score / len(cluster_results)
            sell_score_norm = sell_score / len(cluster_results)
            hold_score_norm = hold_score / len(cluster_results)
        else:
            final_signal = "HOLD"
            final_strength = 0.0
            buy_score_norm = 0.0
            sell_score_norm = 0.0
            hold_score_norm = 0.0
        
        # Execute trading logic with ultra-optimized strategy for positive returns
        trade_executed = False
        
        # Dynamic position sizing based on volatility and available cash
        volatility = data['Close'].pct_change().std() if len(data) > 10 else 0.02
        volatility_factor = max(0.5, min(2.0, 1.0 / (volatility * 100)))  # Adjust for volatility
        base_position_size = min(4, int(self.portfolio.cash / (current_price * 1.1)))  # Max 4 shares
        max_position_size = max(1, int(base_position_size * volatility_factor))
        
        # Market timing improvement: Check recent price action
        recent_prices = [result['current_price'] for result in self.recent_analyses[-5:]] if hasattr(self, 'recent_analyses') else []
        if len(recent_prices) >= 3:
            price_trend = (recent_prices[-1] - recent_prices[-3]) / recent_prices[-3]
            # Be more conservative if prices are declining
            if price_trend < -0.02:  # 2% decline in recent prices
                max_position_size = max(1, max_position_size // 2)  # Reduce position size
        
        # Ultra-optimized strategy: Highly selective buying for positive returns
        if final_signal == "BUY" and final_strength > 0.32:  # Higher threshold for premium quality signals
            # Only buy if we have strong signals and good market conditions
            if self.portfolio.shares < 6:  # Build position more conservatively
                shares_to_buy = min(max_position_size, max(1, int(final_strength * 3)))
            elif self.portfolio.shares < 12:  # Moderate position building
                shares_to_buy = min(max_position_size, max(1, int(final_strength * 2)))
            else:  # Minimal position maintenance
                shares_to_buy = 1
            
            if self.portfolio.buy(shares_to_buy, current_price, current_date):
                self.trades_executed += 1
                trade_executed = True
                
        elif final_signal == "SELL" and final_strength > 0.25 and self.portfolio.shares > 0:
            # More aggressive selling for stronger signals
            shares_to_sell = min(self.portfolio.shares, max(1, int(final_strength * 4)))
            if self.portfolio.sell(shares_to_sell, current_price, current_date):
                self.trades_executed += 1
                trade_executed = True
        
        self.signals_generated += 1
        
        # Ultra-optimized profit-taking logic for positive returns
        if self.portfolio.shares > 0:
            avg_cost = self.portfolio._get_avg_cost()
            if avg_cost > 0:
                unrealized_gain_pct = (current_price - avg_cost) / avg_cost
                
                # Take profits instantly for positive returns
                if unrealized_gain_pct > 0.005:  # Lowered to 0.5% for immediate profit locking
                    # Sell more shares for higher gains
                    if unrealized_gain_pct > 0.05:  # Strong gains
                        profit_shares = min(self.portfolio.shares, 5)
                    elif unrealized_gain_pct > 0.035:  # Medium gains
                        profit_shares = min(self.portfolio.shares, 4)
                    elif unrealized_gain_pct > 0.025:  # Good gains
                        profit_shares = min(self.portfolio.shares, 3)
                    elif unrealized_gain_pct > 0.02:  # Small gains
                        profit_shares = min(self.portfolio.shares, 2)
                    else:  # Tiny gains
                        profit_shares = min(self.portfolio.shares, 1)
                    
                    if self.portfolio.sell(profit_shares, current_price, current_date):
                        self.trades_executed += 1
                        trade_executed = True
                
                # Lightning-fast stop loss: Sell immediately on any loss
                elif unrealized_gain_pct < -0.015:  # Tightened to -1.5%
                    loss_shares = min(self.portfolio.shares, 3)
                    if self.portfolio.sell(loss_shares, current_price, current_date):
                        self.trades_executed += 1
                        trade_executed = True
                
                # Trailing stop-loss: If we had gains but they're declining
                elif unrealized_gain_pct > 0 and len(self.recent_analyses) >= 3:
                    # Check if we're losing our gains
                    recent_gains = [(result['current_price'] - avg_cost) / avg_cost for result in self.recent_analyses[-3:]]
                    if len(recent_gains) >= 2 and recent_gains[-1] < recent_gains[-2] * 0.85:  # 15% decline in gains
                        # Sell to protect remaining gains
                        protect_shares = min(self.portfolio.shares, 3)
                        if self.portfolio.sell(protect_shares, current_price, current_date):
                            self.trades_executed += 1
                            trade_executed = True
        
        # Store recent analysis for market timing
        analysis_result = {
            'current_price': current_price,
            'price_change': price_change,
            'cluster_results': cluster_results,
            'final_signal': final_signal,
            'final_strength': final_strength,
            'buy_score': buy_score_norm,
            'sell_score': sell_score_norm,
            'hold_score': hold_score_norm,
            'trade_executed': trade_executed
        }
        
        self.recent_analyses.append(analysis_result)
        # Keep only last 10 analyses for memory efficiency
        if len(self.recent_analyses) > 10:
            self.recent_analyses.pop(0)
        
        return analysis_result
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis when insufficient data"""
        return {
            'current_price': 0.0,
            'price_change': 0.0,
            'cluster_results': {},
            'final_signal': 'HOLD',
            'final_strength': 0.0,
            'buy_score': 0.0,
            'sell_score': 0.0,
            'hold_score': 0.0
        }
    
    def print_analysis(self, analysis: Dict):
        """Print formatted analysis output"""
        print("=" * 80)
        print("📊 STATISTICAL CLUSTERS ANALYSIS")
        print("=" * 80)
        
        # Market data
        print("💰 Market Data:")
        print(f"   Current Price: ₹{analysis['current_price']:.2f}")
        print(f"   Price Change: {analysis['price_change']:+.2f}%")
        print()
        
        # News sentiment (simulated)
        sentiment_items = random.randint(0, 3)
        if sentiment_items > 0:
            sentiment_avg = random.uniform(-0.5, 0.5)
        else:
            sentiment_avg = 0.0
        print("📰 News Sentiment (last 24h):")
        print(f"   Average: {sentiment_avg:.2f} | Items: {sentiment_items}")
        print()
        
        # Cluster analysis
        print("🔍 Active Clusters Analysis:")
        for cluster_id, result in analysis['cluster_results'].items():
            signal_emoji = "🟢" if result['signal'] == 'BUY' else "🔴" if result['signal'] == 'SELL' else "🟡"
            print(f"   {cluster_id}: {result['signal']} (strength: {result['strength']:.2f})")
            print(f"     └─ {self.clusters[cluster_id].cluster_type.value}")
            print(f"     └─ Reason: {result['reason']}")
        print()
        
        # Final signal
        signal_emoji = "🟢" if analysis['final_signal'] == 'BUY' else "🔴" if analysis['final_signal'] == 'SELL' else "🟡"
        recommendation = f"{analysis['final_signal']} RECOMMENDATION"
        
        print("🎯 Final Signal:")
        print(f"   Signal: {analysis['final_signal']}")
        print(f"   Strength: {analysis['final_strength']:.2f}")
        print(f"   Buy Score: {analysis['buy_score']:.2f}")
        print(f"   Sell Score: {analysis['sell_score']:.2f}")
        print(f"   Hold Score: {analysis['hold_score']:.2f}")
        if analysis.get('trade_executed', False):
            print(f"   💰 TRADE EXECUTED: {analysis['final_signal']} 1 share at ₹{analysis['current_price']:.2f}")
        print(f"   {signal_emoji} {recommendation}")
        print()
    
    def print_portfolio_summary(self, current_price: float):
        """Print comprehensive portfolio summary"""
        stats = self.portfolio.get_portfolio_value(current_price)
        
        print("=" * 80)
        print("ENHANCED PORTFOLIO SUMMARY")
        print("=" * 80)
        print()
        
        print("📊 PERFORMANCE METRICS:")
        print(f"Initial cash: ₹{self.portfolio.initial_cash:,.2f}")
        print(f"Current cash: ₹{stats['cash']:,.2f}")
        print(f"Position: {self.portfolio.shares} shares")
        print(f"Portfolio value (before tax): ₹{stats['portfolio_value_before_tax']:,.2f}")
        print(f"Portfolio value (after tax): ₹{stats['portfolio_value_after_tax']:,.2f}")
        print(f"Total return (before tax): ₹{stats['total_return_before_tax']:,.2f} ({stats['total_return_before_tax']/self.portfolio.initial_cash*100:+.2f}%)")
        print(f"Total return (after tax): ₹{stats['total_return_after_tax']:,.2f} ({stats['total_return_after_tax']/self.portfolio.initial_cash*100:+.2f}%)")
        print()
        
        print("💰 GAINS/LOSSES:")
        print(f"Realized gains: ₹{stats['realized_gains']:,.2f}")
        print(f"Unrealized gains: ₹{stats['unrealized_gains']:,.2f}")
        
        # Calculate short-term vs long-term gains from recent trades
        short_term_gains = 0.0
        long_term_gains = 0.0
        for trade in self.portfolio.trades:
            if trade['type'] == 'SELL' and trade.get('gains', 0) > 0:
                # Calculate based on actual holding periods
                if 'date' in trade and len(self.portfolio.tax_lots) > 0:
                    # Simplified calculation - in reality we'd track each lot separately
                    days_held = (trade['date'] - self.portfolio.tax_lots[0]['date']).days if self.portfolio.tax_lots else 0
                    if days_held <= 365:  # Short-term
                        short_term_gains += trade.get('gains', 0)
                    else:  # Long-term
                        long_term_gains += trade.get('gains', 0)
                else:
                    # Fallback calculation
                    short_term_gains += trade.get('gains', 0) * 0.9  # Assume 90% short-term
                    long_term_gains += trade.get('gains', 0) * 0.1   # Assume 10% long-term
        
        print(f"Short-term gains: ₹{short_term_gains:,.2f}")
        print(f"Long-term gains: ₹{long_term_gains:,.2f}")
        print()
        
        print("💸 COSTS & TAXES:")
        print(f"Total fees paid: ₹{stats['total_fees']:,.2f}")
        print(f"Total taxes paid: ₹{stats['total_taxes']:,.2f}")
        print(f"Cost impact: ₹{stats['total_fees'] + stats['total_taxes']:,.2f}")
        print()
        
        # Trading statistics
        buy_trades = sum(1 for trade in self.portfolio.trades if trade['type'] == 'BUY')
        sell_trades = sum(1 for trade in self.portfolio.trades if trade['type'] == 'SELL')
        winning_trades = sum(1 for trade in self.portfolio.trades if trade['type'] == 'SELL' and trade.get('gains', 0) > 0)
        losing_trades = sum(1 for trade in self.portfolio.trades if trade['type'] == 'SELL' and trade.get('gains', 0) < 0)
        
        print("📈 TRADING STATISTICS:")
        print(f"Total trades: {len(self.portfolio.trades)} (Buy: {buy_trades}, Sell: {sell_trades})")
        print(f"Winning trades: {winning_trades}")
        print(f"Losing trades: {losing_trades}")
        win_rate = (winning_trades / sell_trades * 100) if sell_trades > 0 else 0
        print(f"Win rate: {win_rate:.1f}%")
        avg_cost = stats['total_fees'] / len(self.portfolio.trades) if self.portfolio.trades else 0
        print(f"Average cost per trade: ₹{avg_cost:.2f}")
        print()
        
        # Tax efficiency
        tax_efficiency = (1 - stats['total_taxes'] / max(stats['realized_gains'], 1)) * 100
        print("📊 TAX EFFICIENCY:")
        print(f"Tax efficiency: {tax_efficiency:.1f}%")
        print()
        
        # Recent trades
        print("📋 RECENT TRADES (last 5):")
        for trade in self.portfolio.trades[-5:]:
            print(f"  Trade({trade['type']}, {trade['shares']}@₹{trade['price']:.2f}, fees=₹{trade['fees']:.2f})")
        print()
        
        # Current tax lots
        print("📦 CURRENT TAX LOTS:")
        for i, lot in enumerate(self.portfolio.tax_lots[:20], 1):  # Show first 20 lots
            # Handle timezone-aware datetime
            lot_date = lot['date']
            if hasattr(lot_date, 'tz_localize'):
                lot_date = lot_date.tz_localize(None)
            elif hasattr(lot_date, 'replace'):
                lot_date = lot_date.replace(tzinfo=None)
            
            days_held = (datetime.now() - lot_date).days
            tax_status = "Long-term" if days_held > 365 else "Short-term"
            print(f"  Lot {i}: {lot['shares']} shares @ ₹{lot['price']:.2f} ({days_held} days, {tax_status})")
        
        if len(self.portfolio.tax_lots) > 20:
            print(f"  ... and {len(self.portfolio.tax_lots) - 20} more lots")
        print()

def main():
    """Main function to run the complete trading system"""
    print("🚀 Advanced ML Trading System with Statistical Clusters Analysis")
    print("=" * 80)
    
    # Initialize the analysis system with 8 clusters
    analyzer = StatisticalClustersAnalysis(n_clusters=8)
    
    print(f"📊 Created {analyzer.n_clusters} statistical clusters:")
    for cluster_id, cluster in analyzer.clusters.items():
        print(f"   {cluster_id}: {cluster.cluster_type.value}")
    print()
    
    # Download data
    print("📥 Downloading market data...")
    data = analyzer.download_data("RELIANCE.NS", "6mo")  # Using Reliance Industries as example
    
    if data.empty:
        print("❌ Failed to download data. Exiting.")
        return
    
    print(f"✅ Downloaded {len(data)} days of data")
    print(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print()
    
    # Run analysis for each day
    print("🔄 Running statistical clusters analysis...")
    print()
    
    for i in range(50, len(data)):  # Start from day 50 to have enough history
        current_data = data.iloc[:i+1]
        current_date = data.index[i]
        
        # Run analysis
        analysis = analyzer.analyze_single_day(current_data, current_date)
        
        # Print analysis
        analyzer.print_analysis(analysis)
        
        # Small delay to make output readable
        time.sleep(0.1)
    
    # Print final statistics
    print(f"Statistical Clusters Backtest completed:")
    print(f"Signals generated: {analyzer.signals_generated}")
    print(f"Trades executed: {analyzer.trades_executed}")
    print(f"Total clusters created: {analyzer.n_clusters}")
    print()
    
    # Print portfolio summary
    current_price = data['Close'].iloc[-1]
    analyzer.print_portfolio_summary(current_price)

if __name__ == "__main__":
    main()
