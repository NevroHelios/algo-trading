"""
Advanced ML Algorithms for Trading Signals
Includes PCA, ARIMA, and GARCH implementations
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    ARIMA = None
    print("Warning: statsmodels not available. ARIMA functionality will be limited.")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    arch_model = None
    print("Warning: arch package not available. GARCH functionality will be limited.")

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = None
    StandardScaler = None
    print("Warning: sklearn not available. PCA functionality will be limited.")


class PCASignalGenerator:
    """Principal Component Analysis for dimensionality reduction and signal generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.n_components = config.get('n_components', 5)
        self.lookback_period = config.get('lookback_period', 50)
        self.variance_threshold = config.get('variance_threshold', 0.95)
        self.pca = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_columns = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix from OHLCV data and technical indicators"""
        features = []
        
        # Price-based features
        features.extend(['Open', 'High', 'Low', 'Close'])
        if 'Volume' in df.columns:
            features.append('Volume')
        
        # Technical indicators (if available)
        tech_indicators = [col for col in df.columns if any(indicator in col.lower() 
                          for indicator in ['ma', 'rsi', 'bb', 'atr', 'ichi'])]
        features.extend(tech_indicators)
        
        # Remove duplicates and ensure all columns exist
        features = list(set(features))
        available_features = [f for f in features if f in df.columns]
        
        self.feature_columns = available_features
        return df[available_features].dropna()
    
    def fit_transform(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Fit PCA model and transform data"""
        if not SKLEARN_AVAILABLE:
            return None
            
        try:
            feature_df = self.prepare_features(df)
            if len(feature_df) < self.lookback_period:
                return None
            
            # Take recent data for fitting
            recent_data = feature_df.tail(self.lookback_period)
            
            # Scale features
            scaled_data = self.scaler.fit_transform(recent_data)
            
            # Fit PCA
            self.pca = PCA(n_components=min(self.n_components, scaled_data.shape[1]))
            transformed_data = self.pca.fit_transform(scaled_data)
            
            return transformed_data
            
        except Exception as e:
            print(f"PCA fit_transform error: {e}")
            return None
    
    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """Generate trading signal based on PCA analysis"""
        try:
            if current_index < self.lookback_period:
                return {"signal": "HOLD", "strength": 0.0, "reason": "Insufficient data for PCA"}
            
            # Get recent window of data
            window_data = df.iloc[max(0, current_index - self.lookback_period):current_index + 1]
            transformed_data = self.fit_transform(window_data)
            
            if transformed_data is None:
                return {"signal": "HOLD", "strength": 0.0, "reason": "PCA transformation failed"}
            
            # Analyze the first principal component trend
            pc1 = transformed_data[:, 0]
            
            # Calculate trend and momentum
            recent_trend = np.mean(np.diff(pc1[-10:]))  # Last 10 points trend
            momentum = np.mean(np.diff(pc1[-5:]))       # Last 5 points momentum
            
            # Explained variance ratio for confidence
            explained_variance = self.pca.explained_variance_ratio_[0] if self.pca else 0
            
            # Generate signal based on trend and momentum
            signal_strength = min(float(abs(recent_trend) * explained_variance * 2), 1.0)
            
            if recent_trend > 0 and momentum > 0 and signal_strength > 0.3:
                signal = "BUY"
            elif recent_trend < 0 and momentum < 0 and signal_strength > 0.3:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            return {
                "signal": signal,
                "strength": signal_strength,
                "reason": f"PCA trend: {recent_trend:.4f}, momentum: {momentum:.4f}, var: {explained_variance:.2f}"
            }
            
        except Exception as e:
            return {"signal": "HOLD", "strength": 0.0, "reason": f"PCA error: {str(e)[:50]}"}


class ARIMASignalGenerator:
    """ARIMA model for time series forecasting and signal generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.order = tuple(config.get('order', [1, 1, 1]))
        self.seasonal_order = tuple(config.get('seasonal_order', [1, 1, 1, 12]))
        self.forecast_steps = config.get('forecast_steps', 5)
        self.confidence_level = config.get('confidence_level', 0.95)
        self.min_observations = max(50, sum(self.order) + sum(self.seasonal_order) + 20)
        
    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """Generate trading signal based on ARIMA forecasting"""
        if not STATSMODELS_AVAILABLE:
            return {"signal": "HOLD", "strength": 0.0, "reason": "ARIMA not available"}
        
        try:
            if current_index < self.min_observations:
                return {"signal": "HOLD", "strength": 0.0, "reason": "Insufficient data for ARIMA"}
            
            # Get historical price data
            price_data = df['Close'].iloc[:current_index + 1]
            
            # Convert to returns for better stationarity
            returns = price_data.pct_change().dropna()
            
            if len(returns) < self.min_observations:
                return {"signal": "HOLD", "strength": 0.0, "reason": "Insufficient return data"}
            
            # Fit ARIMA model
            model = ARIMA(returns, order=self.order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=self.forecast_steps)
            forecast_ci = fitted_model.get_forecast(steps=self.forecast_steps).conf_int()
            
            # Calculate signal based on forecast
            avg_forecast = np.mean(forecast)
            
            # Confidence based on forecast interval width
            ci_width = np.mean(forecast_ci.iloc[:, 1] - forecast_ci.iloc[:, 0])
            confidence = max(0, 1 - ci_width * 10)  # Normalize confidence
            
            # Generate signal
            signal_strength = min(abs(avg_forecast) * confidence * 5, 1.0)
            
            if avg_forecast > 0.002 and confidence > 0.3:  # Positive forecast
                signal = "BUY"
            elif avg_forecast < -0.002 and confidence > 0.3:  # Negative forecast
                signal = "SELL"
            else:
                signal = "HOLD"
            
            return {
                "signal": signal,
                "strength": signal_strength,
                "reason": f"ARIMA forecast: {avg_forecast:.4f}, confidence: {confidence:.2f}"
            }
            
        except Exception as e:
            return {"signal": "HOLD", "strength": 0.0, "reason": f"ARIMA error: {str(e)[:50]}"}


class GARCHSignalGenerator:
    """GARCH model for volatility forecasting and signal generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.p = config.get('p', 1)
        self.q = config.get('q', 1)
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
        self.forecast_horizon = config.get('forecast_horizon', 5)
        self.min_observations = max(100, (self.p + self.q) * 10)
        
    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """Generate trading signal based on GARCH volatility forecasting"""
        if not ARCH_AVAILABLE:
            return {"signal": "HOLD", "strength": 0.0, "reason": "GARCH not available"}
        
        try:
            if current_index < self.min_observations:
                return {"signal": "HOLD", "strength": 0.0, "reason": "Insufficient data for GARCH"}
            
            # Get historical price data
            price_data = df['Close'].iloc[:current_index + 1]
            
            # Convert to returns
            returns = price_data.pct_change().dropna() * 100  # Convert to percentage
            
            if len(returns) < self.min_observations:
                return {"signal": "HOLD", "strength": 0.0, "reason": "Insufficient return data"}
            
            # Fit GARCH model
            model = arch_model(returns, vol='Garch', p=self.p, q=self.q)
            fitted_model = model.fit(disp='off')
            
            # Generate volatility forecast
            forecast = fitted_model.forecast(horizon=self.forecast_horizon)
            volatility_forecast = np.sqrt(forecast.variance.values[-1, :])
            
            # Current and forecasted volatility
            current_vol = np.sqrt(fitted_model.conditional_volatility.iloc[-1])
            avg_forecast_vol = np.mean(volatility_forecast)
            
            # Calculate volatility regime change
            vol_change = (avg_forecast_vol - current_vol) / current_vol
            vol_level = avg_forecast_vol / 100  # Convert back to decimal
            
            # Generate signal based on volatility regime
            if vol_level < self.volatility_threshold and vol_change < -0.1:
                # Low volatility regime, good for trend following
                signal = "BUY"
                signal_strength = min(abs(vol_change), 0.8)
                reason = f"Low vol regime: {vol_level:.4f}, decreasing"
            elif vol_level > self.volatility_threshold * 2 and vol_change > 0.1:
                # High volatility regime, risk-off
                signal = "SELL"
                signal_strength = min(abs(vol_change), 0.8)
                reason = f"High vol regime: {vol_level:.4f}, increasing"
            else:
                signal = "HOLD"
                signal_strength = 0.0
                reason = f"Neutral vol: {vol_level:.4f}, change: {vol_change:.2f}"
            
            return {
                "signal": signal,
                "strength": signal_strength,
                "reason": reason
            }
            
        except Exception as e:
            return {"signal": "HOLD", "strength": 0.0, "reason": f"GARCH error: {str(e)[:50]}"}


class MLEnsemble:
    """Ensemble class to combine ML algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.algorithms = {}
        self.weights = {}
        
        # Initialize algorithms
        if config.get('pca', {}).get('enabled', False):
            self.algorithms['pca'] = PCASignalGenerator(config['pca'])
            self.weights['pca'] = config['pca'].get('weight', 0.25)
            
        if config.get('arima', {}).get('enabled', False):
            self.algorithms['arima'] = ARIMASignalGenerator(config['arima'])
            self.weights['arima'] = config['arima'].get('weight', 0.25)
            
        if config.get('garch', {}).get('enabled', False):
            self.algorithms['garch'] = GARCHSignalGenerator(config['garch'])
            self.weights['garch'] = config['garch'].get('weight', 0.25)
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.ensemble_config = config.get('ensemble', {})
        self.combination_method = self.ensemble_config.get('combination_method', 'weighted_average')
        self.minimum_algorithms = self.ensemble_config.get('minimum_algorithms', 2)
        self.confidence_threshold = self.ensemble_config.get('confidence_threshold', 0.6)
    
    def generate_ensemble_signal(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """Generate ensemble signal from all enabled ML algorithms"""
        if not self.algorithms:
            return {"signal": "HOLD", "strength": 0.0, "reason": "No ML algorithms enabled"}
        
        # Collect signals from all algorithms
        algorithm_signals = {}
        for name, algorithm in self.algorithms.items():
            try:
                signal_data = algorithm.generate_signal(df, current_index)
                algorithm_signals[name] = signal_data
            except Exception as e:
                algorithm_signals[name] = {
                    "signal": "HOLD", 
                    "strength": 0.0, 
                    "reason": f"Error: {str(e)[:30]}"
                }
        
        # Filter out algorithms with insufficient data or errors
        valid_signals = {k: v for k, v in algorithm_signals.items() 
                        if v['signal'] != 'HOLD' or v['strength'] > 0}
        
        if len(valid_signals) < self.minimum_algorithms:
            return {
                "signal": "HOLD", 
                "strength": 0.0, 
                "reason": f"Only {len(valid_signals)} algorithms available, need {self.minimum_algorithms}",
                "algorithm_signals": algorithm_signals
            }
        
        # Combine signals based on method
        if self.combination_method == "weighted_average":
            ensemble_signal = self._weighted_average_combination(valid_signals)
        elif self.combination_method == "voting":
            ensemble_signal = self._voting_combination(valid_signals)
        else:
            ensemble_signal = self._weighted_average_combination(valid_signals)
        
        ensemble_signal["algorithm_signals"] = algorithm_signals
        return ensemble_signal
    
    def _weighted_average_combination(self, signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine signals using weighted average"""
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        reasons = []
        
        for name, signal_data in signals.items():
            weight = self.weights.get(name, 1.0)
            strength = signal_data['strength']
            signal = signal_data['signal']
            
            if signal == "BUY":
                buy_score += weight * strength
                reasons.append(f"{name}: BUY({strength:.2f})")
            elif signal == "SELL":
                sell_score += weight * strength
                reasons.append(f"{name}: SELL({strength:.2f})")
            
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine final signal
        if buy_score > sell_score and buy_score >= self.confidence_threshold:
            final_signal = "BUY"
            final_strength = buy_score
        elif sell_score > buy_score and sell_score >= self.confidence_threshold:
            final_signal = "SELL"
            final_strength = sell_score
        else:
            final_signal = "HOLD"
            final_strength = max(buy_score, sell_score)
        
        return {
            "signal": final_signal,
            "strength": final_strength,
            "reason": f"Ensemble: {', '.join(reasons[:3])}"
        }
    
    def _voting_combination(self, signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine signals using simple voting"""
        buy_votes = 0
        sell_votes = 0
        reasons = []
        
        for name, signal_data in signals.items():
            signal = signal_data['signal']
            strength = signal_data['strength']
            
            if signal == "BUY" and strength >= self.confidence_threshold:
                buy_votes += 1
                reasons.append(f"{name}: BUY")
            elif signal == "SELL" and strength >= self.confidence_threshold:
                sell_votes += 1
                reasons.append(f"{name}: SELL")
        
        # Determine final signal
        if buy_votes > sell_votes:
            final_signal = "BUY"
            final_strength = buy_votes / len(signals)
        elif sell_votes > buy_votes:
            final_signal = "SELL"
            final_strength = sell_votes / len(signals)
        else:
            final_signal = "HOLD"
            final_strength = 0.0
        
        return {
            "signal": final_signal,
            "strength": final_strength,
            "reason": f"Voting: {', '.join(reasons)}"
        }
