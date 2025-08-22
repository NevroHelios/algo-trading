"""
Demo script for ML Algorithms (without PyTorch)
Tests Random Forest, XGBoost, and Kalman Filter algorithms
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
from advanced_ml_algorithms import (
    RandomForestSignalGenerator,
    XGBoostSignalGenerator,
    KalmanFilterTrendEstimator,
    SignalType
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_sample_data(symbol="AAPL", period="1y"):
    """Download sample stock data"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            print(f"No data found for {symbol}")
            return None
            
        # Add some basic technical indicators
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = calculate_rsi(data['Close'], window=14)
        data['Volatility'] = data['Close'].rolling(window=20).std()
        
        print(f"Downloaded {len(data)} days of data for {symbol}")
        return data
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def calculate_rsi(prices, window=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def test_random_forest(data):
    """Test Random Forest model"""
    print("\n" + "="*50)
    print("TESTING RANDOM FOREST MODEL")
    print("="*50)
    
    rf_config = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "lookback_period": 20
    }
    
    rf_generator = RandomForestSignalGenerator(rf_config)
    
    print("Training Random Forest model...")
    if rf_generator.train(data):
        print("✅ Random Forest training successful!")
        
        # Generate signal
        signal = rf_generator.predict(data)
        print(f"Random Forest Signal: {signal}")
        
        # Get feature importance if available
        if hasattr(rf_generator.model, 'feature_importances_'):
            print("Top 5 most important features:")
            feature_importance = list(zip(rf_generator.model.feature_importances_, 
                                        rf_generator.model.feature_names_in_))
            feature_importance.sort(reverse=True)
            for importance, feature in feature_importance[:5]:
                print(f"  {feature}: {importance:.4f}")
    else:
        print("❌ Random Forest training failed")

def test_xgboost(data):
    """Test XGBoost model"""
    print("\n" + "="*50)
    print("TESTING XGBOOST MODEL")
    print("="*50)
    
    xgb_config = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lookback_period": 20
    }
    
    xgb_generator = XGBoostSignalGenerator(xgb_config)
    
    print("Training XGBoost model...")
    if xgb_generator.train(data):
        print("✅ XGBoost training successful!")
        
        # Generate signal
        signal = xgb_generator.predict(data)
        print(f"XGBoost Signal: {signal}")
        
        # Get feature importance if available
        if hasattr(xgb_generator.model, 'feature_importances_'):
            print("Top 5 most important features:")
            feature_importance = list(zip(xgb_generator.model.feature_importances_, 
                                        xgb_generator.model.feature_names_in_))
            feature_importance.sort(reverse=True)
            for importance, feature in feature_importance[:5]:
                print(f"  {feature}: {importance:.4f}")
    else:
        print("❌ XGBoost training failed")

def test_kalman_filter(data):
    """Test Kalman Filter"""
    print("\n" + "="*50)
    print("TESTING KALMAN FILTER")
    print("="*50)
    
    kalman_config = {
        "process_noise": 0.01,
        "measurement_noise": 0.1
    }
    
    kalman_estimator = KalmanFilterTrendEstimator(kalman_config)
    
    print("Processing data with Kalman Filter...")
    signal = kalman_estimator.get_trend_signal(data)
    print(f"Kalman Filter Signal: {signal}")
    
    # Get smoothed prices
    smoothed_prices = kalman_estimator.get_smoothed_prices()
    if smoothed_prices:
        print(f"Generated {len(smoothed_prices)} smoothed price estimates")
        print(f"Latest smoothed price: ${smoothed_prices[-1]:.2f}")
        print(f"Original latest price: ${data['Close'].iloc[-1]:.2f}")

def test_ensemble_approach(data):
    """Test ensemble approach combining available models"""
    print("\n" + "="*50)
    print("TESTING ENSEMBLE APPROACH")
    print("="*50)
    
    # Train all available models
    models = {}
    signals = {}
    
    # Random Forest
    rf_config = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5}
    rf_generator = RandomForestSignalGenerator(rf_config)
    if rf_generator.train(data):
        models["Random Forest"] = rf_generator
        signals["Random Forest"] = rf_generator.predict(data)
        print("✅ Random Forest trained")
    else:
        print("❌ Random Forest failed")
    
    # XGBoost
    xgb_config = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100}
    xgb_generator = XGBoostSignalGenerator(xgb_config)
    if xgb_generator.train(data):
        models["XGBoost"] = xgb_generator
        signals["XGBoost"] = xgb_generator.predict(data)
        print("✅ XGBoost trained")
    else:
        print("❌ XGBoost failed")
    
    # Kalman Filter
    kalman_config = {"process_noise": 0.01, "measurement_noise": 0.1}
    kalman_estimator = KalmanFilterTrendEstimator(kalman_config)
    kalman_signal = kalman_estimator.get_trend_signal(data)
    if kalman_signal is not None:
        models["Kalman Filter"] = kalman_estimator
        signals["Kalman Filter"] = kalman_signal
        print("✅ Kalman Filter ready")
    else:
        print("❌ Kalman Filter failed")
    
    # Generate ensemble signal
    if signals:
        print(f"\nIndividual signals: {signals}")
        
        # Simple voting mechanism
        buy_votes = sum(1 for signal in signals.values() if signal == SignalType.BUY)
        sell_votes = sum(1 for signal in signals.values() if signal == SignalType.SELL)
        hold_votes = sum(1 for signal in signals.values() if signal == SignalType.HOLD)
        
        total_models = len(signals)
        print(f"\nVoting Results:")
        print(f"  BUY votes: {buy_votes}/{total_models}")
        print(f"  SELL votes: {sell_votes}/{total_models}")
        print(f"  HOLD votes: {hold_votes}/{total_models}")
        
        # Determine ensemble signal
        if buy_votes > sell_votes and buy_votes > hold_votes:
            ensemble_signal = SignalType.BUY
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            ensemble_signal = SignalType.SELL
        else:
            ensemble_signal = SignalType.HOLD
        
        print(f"\nEnsemble Signal: {ensemble_signal}")
    else:
        print("❌ No models available for ensemble")

def main():
    """Main demo function"""
    print("🚀 ML Algorithms Demo (without PyTorch)")
    print("="*50)
    
    # Download sample data
    print("Downloading sample data...")
    data = download_sample_data("AAPL", "6mo")  # 6 months of data
    
    if data is None:
        print("Failed to download data. Exiting.")
        return
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Latest close price: ${data['Close'].iloc[-1]:.2f}")
    
    # Test individual models
    test_random_forest(data)
    test_xgboost(data)
    test_kalman_filter(data)
    
    # Test ensemble approach
    test_ensemble_approach(data)
    
    print("\n" + "="*50)
    print("Demo completed!")
    print("="*50)

if __name__ == "__main__":
    main()
