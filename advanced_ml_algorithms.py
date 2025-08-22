"""
Advanced ML Algorithms for Trading Signals
Implements Random Forest, LSTM, XGBoost, and Kalman Filter algorithms
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Optional imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None
    print("Warning: PyTorch not available. LSTM functionality will be limited.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. XGBoost functionality will be limited.")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Random Forest functionality will be limited.")

try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    print("Warning: filterpy not available. Kalman Filter functionality will be limited.")


class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """LSTM Neural Network for time series prediction"""
        
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
            # Forward propagate LSTM
            out, _ = self.lstm(x, (h0, c0))
            
            # Decode the hidden state of the last time step
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            return out
else:
    # Dummy class when PyTorch is not available
    class LSTMModel:
        def __init__(self, *args, **kwargs):
            pass


class LSTMSignalGenerator:
    """LSTM Neural Network Signal Generator for sequential price data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.sequence_length = config.get("sequence_length", 60)
        self.hidden_size = config.get("hidden_size", 128)
        self.num_layers = config.get("num_layers", 2)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 32)
        
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM"""
        if not TORCH_AVAILABLE:
            return np.array([]), np.array([])
            
        # Use OHLCV data
        features = ['Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns:
            features.append('Volume')
            
        data = df[features].values
        
        # Normalize data
        if self.scaler:
            data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            # Target: 1 if price goes up, 0 if down
            y.append(1 if data[i, 3] > data[i-1, 3] else 0)
            
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame) -> bool:
        """Train the LSTM model"""
        if not TORCH_AVAILABLE:
            return False
            
        try:
            X, y = self.prepare_data(df)
            if len(X) == 0:
                return False
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Convert to tensors
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_test = torch.FloatTensor(X_test).to(self.device)
            y_test = torch.FloatTensor(y_test).to(self.device)
            
            # Create model
            input_size = X_train.shape[2]
            self.model = LSTMModel(input_size, self.hidden_size, self.num_layers, 1).to(self.device)
            
            # Loss and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                outputs = self.model(X_train).squeeze()
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 20 == 0:
                    logging.info(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logging.error(f"Error training LSTM: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Optional[SignalType]:
        """Generate trading signal using LSTM"""
        if not self.is_trained or not TORCH_AVAILABLE:
            return None
            
        try:
            X, _ = self.prepare_data(df)
            if len(X) == 0:
                return None
                
            # Use last sequence for prediction
            last_sequence = torch.FloatTensor(X[-1:]).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                prediction = torch.sigmoid(self.model(last_sequence)).item()
                
            # Convert to signal
            if prediction > 0.6:
                return SignalType.BUY
            elif prediction < 0.4:
                return SignalType.SELL
            else:
                return SignalType.HOLD
                
        except Exception as e:
            logging.error(f"Error predicting with LSTM: {e}")
            return None


class RandomForestSignalGenerator:
    """Random Forest Signal Generator for tabular data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.n_estimators = config.get("n_estimators", 100)
        self.max_depth = config.get("max_depth", 10)
        self.min_samples_split = config.get("min_samples_split", 5)
        self.lookback_period = config.get("lookback_period", 20)
        
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix with technical indicators"""
        if not SKLEARN_AVAILABLE:
            return pd.DataFrame()
            
        features = []
        
        # Price-based features
        features.extend(['Open', 'High', 'Low', 'Close'])
        if 'Volume' in df.columns:
            features.append('Volume')
        
        # Technical indicators
        tech_indicators = [
            col for col in df.columns 
            if any(indicator in col.lower() for indicator in ['ma', 'rsi', 'bb', 'atr', 'ichi'])
        ]
        features.extend(tech_indicators)
        
        # Add price changes
        if 'Close' in df.columns:
            df['price_change'] = df['Close'].pct_change()
            df['price_change_2'] = df['Close'].pct_change(2)
            df['price_change_5'] = df['Close'].pct_change(5)
            features.extend(['price_change', 'price_change_2', 'price_change_5'])
        
        # Add volatility
        if 'Close' in df.columns:
            df['volatility'] = df['Close'].rolling(window=20).std()
            features.append('volatility')
        
        # Remove duplicates and ensure all columns exist
        features = list(set(features))
        available_features = [f for f in features if f in df.columns]
        
        return df[available_features].dropna()
    
    def prepare_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare target labels for classification"""
        if 'Close' not in df.columns:
            return np.array([])
            
        # Create target: 1 for price increase, 0 for decrease
        future_returns = df['Close'].shift(-1) / df['Close'] - 1
        targets = (future_returns > 0).astype(int)
        
        return targets[:-1]  # Remove last row as we don't have future data
    
    def train(self, df: pd.DataFrame) -> bool:
        """Train the Random Forest model"""
        if not SKLEARN_AVAILABLE:
            return False
            
        try:
            feature_df = self.prepare_features(df)
            targets = self.prepare_targets(df)
            
            if len(feature_df) == 0 or len(targets) == 0:
                return False
            
            # Align features and targets
            min_len = min(len(feature_df), len(targets))
            feature_df = feature_df.iloc[:min_len]
            targets = targets[:min_len]
            
            # Remove rows with NaN values
            valid_mask = ~(feature_df.isna().any(axis=1) | pd.isna(targets))
            feature_df = feature_df[valid_mask]
            targets = targets[valid_mask]
            
            if len(feature_df) < 50:  # Need minimum data
                return False
            
            # Scale features
            if self.scaler:
                feature_df = pd.DataFrame(
                    self.scaler.fit_transform(feature_df),
                    columns=feature_df.columns,
                    index=feature_df.index
                )
            
            # Create and train model
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42
            )
            
            self.model.fit(feature_df, targets)
            self.is_trained = True
            return True
            
        except Exception as e:
            logging.error(f"Error training Random Forest: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Optional[SignalType]:
        """Generate trading signal using Random Forest"""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return None
            
        try:
            feature_df = self.prepare_features(df)
            if len(feature_df) == 0:
                return None
            
            # Use last row for prediction
            last_features = feature_df.iloc[-1:].dropna()
            if len(last_features) == 0:
                return None
            
            # Scale features
            if self.scaler:
                last_features = pd.DataFrame(
                    self.scaler.transform(last_features),
                    columns=last_features.columns,
                    index=last_features.index
                )
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(last_features)[0]
            
            # Convert to signal based on probability threshold
            buy_prob = probabilities[1] if len(probabilities) > 1 else 0.5
            
            if buy_prob > 0.6:
                return SignalType.BUY
            elif buy_prob < 0.4:
                return SignalType.SELL
            else:
                return SignalType.HOLD
                
        except Exception as e:
            logging.error(f"Error predicting with Random Forest: {e}")
            return None


class XGBoostSignalGenerator:
    """XGBoost Signal Generator for non-linear relationships"""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_depth = config.get("max_depth", 6)
        self.learning_rate = config.get("learning_rate", 0.1)
        self.n_estimators = config.get("n_estimators", 100)
        self.subsample = config.get("subsample", 0.8)
        self.colsample_bytree = config.get("colsample_bytree", 0.8)
        self.lookback_period = config.get("lookback_period", 20)
        
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for XGBoost"""
        if not SKLEARN_AVAILABLE:
            return pd.DataFrame()
            
        features = []
        
        # Price-based features
        features.extend(['Open', 'High', 'Low', 'Close'])
        if 'Volume' in df.columns:
            features.append('Volume')
        
        # Technical indicators
        tech_indicators = [
            col for col in df.columns 
            if any(indicator in col.lower() for indicator in ['ma', 'rsi', 'bb', 'atr', 'ichi'])
        ]
        features.extend(tech_indicators)
        
        # Add engineered features
        if 'Close' in df.columns:
            # Price momentum
            df['momentum_1'] = df['Close'].pct_change(1)
            df['momentum_5'] = df['Close'].pct_change(5)
            df['momentum_10'] = df['Close'].pct_change(10)
            
            # Volatility
            df['volatility_5'] = df['Close'].rolling(window=5).std()
            df['volatility_20'] = df['Close'].rolling(window=20).std()
            
            # Moving averages
            df['ma_5'] = df['Close'].rolling(window=5).mean()
            df['ma_20'] = df['Close'].rolling(window=20).mean()
            df['ma_ratio'] = df['ma_5'] / df['ma_20']
            
            features.extend([
                'momentum_1', 'momentum_5', 'momentum_10',
                'volatility_5', 'volatility_20',
                'ma_5', 'ma_20', 'ma_ratio'
            ])
        
        # Remove duplicates and ensure all columns exist
        features = list(set(features))
        available_features = [f for f in features if f in df.columns]
        
        return df[available_features].dropna()
    
    def prepare_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare target labels for XGBoost"""
        if 'Close' not in df.columns:
            return np.array([])
            
        # Create target: 1 for price increase, 0 for decrease
        future_returns = df['Close'].shift(-1) / df['Close'] - 1
        targets = (future_returns > 0).astype(int)
        
        return targets[:-1]
    
    def train(self, df: pd.DataFrame) -> bool:
        """Train the XGBoost model"""
        if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
            return False
            
        try:
            feature_df = self.prepare_features(df)
            targets = self.prepare_targets(df)
            
            if len(feature_df) == 0 or len(targets) == 0:
                return False
            
            # Align features and targets
            min_len = min(len(feature_df), len(targets))
            feature_df = feature_df.iloc[:min_len]
            targets = targets[:min_len]
            
            # Remove rows with NaN values
            valid_mask = ~(feature_df.isna().any(axis=1) | pd.isna(targets))
            feature_df = feature_df[valid_mask]
            targets = targets[valid_mask]
            
            if len(feature_df) < 50:
                return False
            
            # Scale features
            if self.scaler:
                feature_df = pd.DataFrame(
                    self.scaler.fit_transform(feature_df),
                    columns=feature_df.columns,
                    index=feature_df.index
                )
            
            # Create and train model
            self.model = xgb.XGBClassifier(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=42
            )
            
            self.model.fit(feature_df, targets)
            self.is_trained = True
            return True
            
        except Exception as e:
            logging.error(f"Error training XGBoost: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Optional[SignalType]:
        """Generate trading signal using XGBoost"""
        if not self.is_trained or not XGBOOST_AVAILABLE:
            return None
            
        try:
            feature_df = self.prepare_features(df)
            if len(feature_df) == 0:
                return None
            
            # Use last row for prediction
            last_features = feature_df.iloc[-1:].dropna()
            if len(last_features) == 0:
                return None
            
            # Scale features
            if self.scaler:
                last_features = pd.DataFrame(
                    self.scaler.transform(last_features),
                    columns=last_features.columns,
                    index=last_features.index
                )
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(last_features)[0]
            
            # Convert to signal
            buy_prob = probabilities[1] if len(probabilities) > 1 else 0.5
            
            if buy_prob > 0.6:
                return SignalType.BUY
            elif buy_prob < 0.4:
                return SignalType.SELL
            else:
                return SignalType.HOLD
                
        except Exception as e:
            logging.error(f"Error predicting with XGBoost: {e}")
            return None


class KalmanFilterTrendEstimator:
    """Kalman Filter for trend estimation and noise reduction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.process_noise = config.get("process_noise", 0.01)
        self.measurement_noise = config.get("measurement_noise", 0.1)
        self.initial_state = config.get("initial_state", [0, 0])  # [position, velocity]
        self.initial_covariance = config.get("initial_covariance", [[1, 0], [0, 1]])
        
        self.kf = None
        self.is_initialized = False
        self.trend_history = []
        
    def initialize_filter(self, initial_price: float):
        """Initialize the Kalman Filter"""
        if not FILTERPY_AVAILABLE:
            return False
            
        try:
            # State: [price, price_velocity]
            # Measurement: [price]
            self.kf = KalmanFilter(dim_x=2, dim_z=1)
            
            # State transition matrix
            self.kf.F = np.array([[1, 1], [0, 1]])
            
            # Measurement matrix
            self.kf.H = np.array([[1, 0]])
            
            # Process noise
            self.kf.Q = np.array([[self.process_noise, 0], [0, self.process_noise]])
            
            # Measurement noise
            self.kf.R = np.array([[self.measurement_noise]])
            
            # Initial state
            self.kf.x = np.array([initial_price, 0])
            
            # Initial covariance
            self.kf.P = np.array(self.initial_covariance)
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logging.error(f"Error initializing Kalman Filter: {e}")
            return False
    
    def update(self, price: float) -> Optional[Dict[str, float]]:
        """Update the Kalman Filter with new price data"""
        if not self.is_initialized or not FILTERPY_AVAILABLE:
            return None
            
        try:
            # Predict
            self.kf.predict()
            
            # Update with measurement
            self.kf.update(price)
            
            # Extract estimated price and velocity
            estimated_price = self.kf.x[0]
            estimated_velocity = self.kf.x[1]
            
            # Store trend information
            trend_info = {
                'price': estimated_price,
                'velocity': estimated_velocity,
                'trend_strength': abs(estimated_velocity),
                'trend_direction': np.sign(estimated_velocity)
            }
            
            self.trend_history.append(trend_info)
            
            return trend_info
            
        except Exception as e:
            logging.error(f"Error updating Kalman Filter: {e}")
            return None
    
    def get_trend_signal(self, df: pd.DataFrame) -> Optional[SignalType]:
        """Generate trading signal based on trend estimation"""
        if 'Close' not in df.columns or len(df) == 0:
            return None
            
        try:
            # Initialize filter with first price if not already done
            if not self.is_initialized:
                if not self.initialize_filter(df['Close'].iloc[0]):
                    return None
            
            # Update filter with all prices
            for price in df['Close']:
                self.update(price)
            
            if len(self.trend_history) == 0:
                return None
            
            # Get latest trend information
            latest_trend = self.trend_history[-1]
            
            # Generate signal based on trend strength and direction
            trend_strength = latest_trend['trend_strength']
            trend_direction = latest_trend['trend_direction']
            
            # Threshold for trend strength
            strength_threshold = 0.01
            
            if trend_strength > strength_threshold:
                if trend_direction > 0:
                    return SignalType.BUY
                else:
                    return SignalType.SELL
            else:
                return SignalType.HOLD
                
        except Exception as e:
            logging.error(f"Error getting trend signal: {e}")
            return None
    
    def get_smoothed_prices(self) -> List[float]:
        """Get smoothed price estimates from Kalman Filter"""
        return [trend['price'] for trend in self.trend_history]


class AdvancedMLSignalGenerator:
    """Combined advanced ML signal generator using multiple algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.lstm_generator = LSTMSignalGenerator(config.get("lstm", {}))
        self.rf_generator = RandomForestSignalGenerator(config.get("random_forest", {}))
        self.xgb_generator = XGBoostSignalGenerator(config.get("xgboost", {}))
        self.kalman_estimator = KalmanFilterTrendEstimator(config.get("kalman", {}))
        
        self.weights = config.get("weights", {
            "lstm": 0.3,
            "random_forest": 0.3,
            "xgboost": 0.3,
            "kalman": 0.1
        })
        
        self.is_trained = False
        
    def train_all_models(self, df: pd.DataFrame) -> bool:
        """Train all ML models"""
        try:
            # Train each model
            lstm_success = self.lstm_generator.train(df)
            rf_success = self.rf_generator.train(df)
            xgb_success = self.xgb_generator.train(df)
            
            # Kalman filter doesn't need training, just initialization
            kalman_success = True
            
            # Consider trained if at least 2 models are successful
            success_count = sum([lstm_success, rf_success, xgb_success, kalman_success])
            self.is_trained = success_count >= 2
            
            logging.info(f"Training results - LSTM: {lstm_success}, RF: {rf_success}, XGB: {xgb_success}, Kalman: {kalman_success}")
            
            return self.is_trained
            
        except Exception as e:
            logging.error(f"Error training models: {e}")
            return False
    
    def generate_combined_signal(self, df: pd.DataFrame) -> Optional[SignalType]:
        """Generate combined trading signal using all models"""
        if not self.is_trained:
            return None
            
        try:
            signals = {}
            scores = {}
            
            # Get signals from each model
            lstm_signal = self.lstm_generator.predict(df)
            rf_signal = self.rf_generator.predict(df)
            xgb_signal = self.xgb_generator.predict(df)
            kalman_signal = self.kalman_estimator.get_trend_signal(df)
            
            # Collect valid signals
            if lstm_signal is not None:
                signals["lstm"] = lstm_signal
                scores["lstm"] = self.weights["lstm"]
            
            if rf_signal is not None:
                signals["random_forest"] = rf_signal
                scores["random_forest"] = self.weights["random_forest"]
            
            if xgb_signal is not None:
                signals["xgboost"] = xgb_signal
                scores["xgboost"] = self.weights["xgboost"]
            
            if kalman_signal is not None:
                signals["kalman"] = kalman_signal
                scores["kalman"] = self.weights["kalman"]
            
            if not signals:
                return None
            
            # Calculate weighted score
            total_score = 0
            total_weight = 0
            
            for model, signal in signals.items():
                weight = scores[model]
                total_score += signal.value * weight
                total_weight += weight
            
            if total_weight == 0:
                return None
            
            # Normalize score
            normalized_score = total_score / total_weight
            
            # Convert to signal
            if normalized_score > 0.3:
                return SignalType.BUY
            elif normalized_score < -0.3:
                return SignalType.SELL
            else:
                return SignalType.HOLD
                
        except Exception as e:
            logging.error(f"Error generating combined signal: {e}")
            return None
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get training status of all models"""
        return {
            "lstm": self.lstm_generator.is_trained,
            "random_forest": self.rf_generator.is_trained,
            "xgboost": self.xgb_generator.is_trained,
            "kalman": self.kalman_estimator.is_initialized
        }
