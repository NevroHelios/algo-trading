"""
Demo script for Advanced ML Algorithms
Tests LSTM, Random Forest, XGBoost, and Kalman Filter algorithms
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
from advanced_ml_algorithms import (
    AdvancedMLSignalGenerator,
    LSTMSignalGenerator,
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

def test_individual_models(data):
    """Test individual ML models"""
    print("\n" + "="*50)
    print("TESTING INDIVIDUAL MODELS")
    print("="*50)
    
    # Test LSTM
    print("\n1. Testing LSTM Model...")
    lstm_config = {
        "sequence_length": 30,
        "hidden_size": 64,
        "num_layers": 2,
        "learning_rate": 0.001,
        "epochs": 50,  # Reduced for demo
        "batch_size": 16
    }
    
    lstm_generator = LSTMSignalGenerator(lstm_config)
    if lstm_generator.train(data):
        signal = lstm_generator.predict(data)
        print(f"LSTM Signal: {signal}")
    else:
        print("LSTM training failed")
    
    # Test Random Forest
    print("\n2. Testing Random Forest Model...")
    rf_config = {
        "n_estimators": 50,  # Reduced for demo
        "max_depth": 8,
        "min_samples_split": 5,
        "lookback_period": 20
    }
    
    rf_generator = RandomForestSignalGenerator(rf_config)
    if rf_generator.train(data):
        signal = rf_generator.predict(data)
        print(f"Random Forest Signal: {signal}")
    else:
        print("Random Forest training failed")
    
    # Test XGBoost
    print("\n3. Testing XGBoost Model...")
    xgb_config = {
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_estimators": 50,  # Reduced for demo
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lookback_period": 20
    }
    
    xgb_generator = XGBoostSignalGenerator(xgb_config)
    if xgb_generator.train(data):
        signal = xgb_generator.predict(data)
        print(f"XGBoost Signal: {signal}")
    else:
        print("XGBoost training failed")
    
    # Test Kalman Filter
    print("\n4. Testing Kalman Filter...")
    kalman_config = {
        "process_noise": 0.01,
        "measurement_noise": 0.1
    }
    
    kalman_estimator = KalmanFilterTrendEstimator(kalman_config)
    signal = kalman_estimator.get_trend_signal(data)
    print(f"Kalman Filter Signal: {signal}")

def test_combined_model(data):
    """Test the combined advanced ML model"""
    print("\n" + "="*50)
    print("TESTING COMBINED ADVANCED ML MODEL")
    print("="*50)
    
    # Configuration for combined model
    config = {
        "lstm": {
            "sequence_length": 30,
            "hidden_size": 64,
            "num_layers": 2,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 16
        },
        "random_forest": {
            "n_estimators": 50,
            "max_depth": 8,
            "min_samples_split": 5,
            "lookback_period": 20
        },
        "xgboost": {
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_estimators": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lookback_period": 20
        },
        "kalman": {
            "process_noise": 0.01,
            "measurement_noise": 0.1
        },
        "weights": {
            "lstm": 0.3,
            "random_forest": 0.3,
            "xgboost": 0.3,
            "kalman": 0.1
        }
    }
    
    # Create and train combined model
    advanced_ml = AdvancedMLSignalGenerator(config)
    
    print("Training all models...")
    if advanced_ml.train_all_models(data):
        print("✅ All models trained successfully!")
        
        # Get model status
        status = advanced_ml.get_model_status()
        print("\nModel Training Status:")
        for model, trained in status.items():
            print(f"  {model}: {'✅' if trained else '❌'}")
        
        # Generate combined signal
        print("\nGenerating combined signal...")
        combined_signal = advanced_ml.generate_combined_signal(data)
        print(f"Combined Signal: {combined_signal}")
        
        # Test individual signals
        print("\nIndividual Model Signals:")
        lstm_signal = advanced_ml.lstm_generator.predict(data)
        rf_signal = advanced_ml.rf_generator.predict(data)
        xgb_signal = advanced_ml.xgb_generator.predict(data)
        kalman_signal = advanced_ml.kalman_estimator.get_trend_signal(data)
        
        print(f"  LSTM: {lstm_signal}")
        print(f"  Random Forest: {rf_signal}")
        print(f"  XGBoost: {xgb_signal}")
        print(f"  Kalman Filter: {kalman_signal}")
        
    else:
        print("❌ Some models failed to train")

def main():
    """Main demo function"""
    print("🚀 Advanced ML Algorithms Demo")
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
    test_individual_models(data)
    
    # Test combined model
    test_combined_model(data)
    
    print("\n" + "="*50)
    print("Demo completed!")
    print("="*50)

if __name__ == "__main__":
    main()
