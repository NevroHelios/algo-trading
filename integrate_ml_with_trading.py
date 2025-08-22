"""
Integration script showing how to use Advanced ML Algorithms with the trading system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from advanced_ml_algorithms import AdvancedMLSignalGenerator, SignalType

def create_ml_trading_strategy():
    """
    Create a trading strategy that uses the advanced ML algorithms
    """
    
    # Configuration for the ML models
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
            "random_forest": 0.4,
            "xgboost": 0.4,
            "kalman": 0.2
        }
    }
    
    return AdvancedMLSignalGenerator(ml_config)

def ml_trading_signal_generator(data, ml_generator):
    """
    Generate trading signals using the ML algorithms
    
    Args:
        data (pd.DataFrame): OHLCV data with technical indicators
        ml_generator (AdvancedMLSignalGenerator): Trained ML generator
    
    Returns:
        dict: Trading signal information
    """
    
    try:
        # Generate combined signal
        signal = ml_generator.generate_combined_signal(data)
        
        if signal is None:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "No signal generated",
                "timestamp": datetime.now()
            }
        
        # Get individual model signals for confidence calculation
        individual_signals = {}
        
        # Random Forest signal
        rf_signal = ml_generator.rf_generator.predict(data)
        if rf_signal:
            individual_signals["random_forest"] = rf_signal.value
        
        # XGBoost signal
        xgb_signal = ml_generator.xgb_generator.predict(data)
        if xgb_signal:
            individual_signals["xgboost"] = xgb_signal.value
        
        # Kalman Filter signal
        kalman_signal = ml_generator.kalman_estimator.get_trend_signal(data)
        if kalman_signal:
            individual_signals["kalman"] = kalman_signal.value
        
        # Calculate confidence based on agreement
        if individual_signals:
            signal_values = list(individual_signals.values())
            agreement = sum(1 for v in signal_values if v == signal.value) / len(signal_values)
            confidence = agreement * 100
        else:
            confidence = 50.0  # Default confidence
        
        # Determine signal type
        signal_type = "BUY" if signal == SignalType.BUY else "SELL" if signal == SignalType.SELL else "HOLD"
        
        return {
            "signal": signal_type,
            "confidence": confidence,
            "individual_signals": individual_signals,
            "reason": f"ML ensemble signal with {confidence:.1f}% confidence",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "reason": f"Error generating signal: {str(e)}",
            "timestamp": datetime.now()
        }

def integrate_with_existing_system():
    """
    Example of how to integrate ML algorithms with existing trading system
    """
    
    print("🤖 Advanced ML Trading System Integration")
    print("="*50)
    
    # Create ML trading strategy
    ml_strategy = create_ml_trading_strategy()
    
    print("✅ ML Strategy created successfully")
    print(f"Available models: {list(ml_strategy.get_model_status().keys())}")
    
    # Example usage in a trading loop
    def trading_loop_example():
        """
        Example of how to use ML signals in a trading loop
        """
        
        # This would be your existing data fetching logic
        # data = fetch_market_data(symbol="AAPL", period="1d")
        
        # For demonstration, we'll use a placeholder
        print("\n📊 Trading Loop Example:")
        print("1. Fetch market data")
        print("2. Add technical indicators")
        print("3. Generate ML signals")
        print("4. Execute trades based on signals")
        
        # Example signal generation
        # signal_info = ml_trading_signal_generator(data, ml_strategy)
        
        # Example trading logic
        print("\n🎯 Example Trading Logic:")
        print("if signal_info['signal'] == 'BUY' and signal_info['confidence'] > 70:")
        print("    execute_buy_order()")
        print("elif signal_info['signal'] == 'SELL' and signal_info['confidence'] > 70:")
        print("    execute_sell_order()")
        print("else:")
        print("    hold_position()")
    
    trading_loop_example()
    
    return ml_strategy

def backtesting_integration_example():
    """
    Example of how to integrate ML algorithms with backtesting
    """
    
    print("\n📈 Backtesting Integration Example:")
    print("="*50)
    
    # Example backtesting structure
    backtest_config = {
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "initial_capital": 100000,
        "ml_config": {
            "random_forest": {"n_estimators": 100, "max_depth": 10},
            "xgboost": {"max_depth": 6, "learning_rate": 0.1},
            "kalman": {"process_noise": 0.01, "measurement_noise": 0.1},
            "weights": {"random_forest": 0.4, "xgboost": 0.4, "kalman": 0.2}
        }
    }
    
    print("Backtest Configuration:")
    for key, value in backtest_config.items():
        print(f"  {key}: {value}")
    
    print("\nBacktesting Steps:")
    print("1. Load historical data")
    print("2. Train ML models on training period")
    print("3. Generate signals for test period")
    print("4. Execute simulated trades")
    print("5. Calculate performance metrics")
    
    return backtest_config

def risk_management_integration():
    """
    Example of how to integrate ML signals with risk management
    """
    
    print("\n🛡️ Risk Management Integration:")
    print("="*50)
    
    risk_config = {
        "max_position_size": 0.1,  # 10% of portfolio
        "stop_loss": 0.05,         # 5% stop loss
        "take_profit": 0.15,       # 15% take profit
        "max_drawdown": 0.20,      # 20% max drawdown
        "ml_confidence_threshold": 70,  # Minimum confidence for ML signals
    }
    
    print("Risk Management Configuration:")
    for key, value in risk_config.items():
        print(f"  {key}: {value}")
    
    print("\nRisk Management Rules:")
    print("1. Only trade if ML confidence > threshold")
    print("2. Adjust position size based on confidence")
    print("3. Use stop-loss and take-profit orders")
    print("4. Monitor portfolio drawdown")
    print("5. Diversify across multiple ML signals")
    
    return risk_config

def main():
    """
    Main integration demonstration
    """
    
    print("🚀 Advanced ML Trading System Integration Demo")
    print("="*60)
    
    # 1. Create ML strategy
    ml_strategy = integrate_with_existing_system()
    
    # 2. Show backtesting integration
    backtest_config = backtesting_integration_example()
    
    # 3. Show risk management integration
    risk_config = risk_management_integration()
    
    # 4. Summary
    print("\n" + "="*60)
    print("📋 Integration Summary")
    print("="*60)
    print("✅ ML Algorithms: Random Forest, XGBoost, Kalman Filter")
    print("✅ Signal Generation: Ensemble approach with weighted voting")
    print("✅ Risk Management: Configurable thresholds and position sizing")
    print("✅ Backtesting: Ready for historical performance analysis")
    print("✅ Real-time Trading: Can be integrated with live trading systems")
    
    print("\n🎯 Next Steps:")
    print("1. Train models on historical data")
    print("2. Backtest on out-of-sample data")
    print("3. Implement in live trading system")
    print("4. Monitor performance and retrain models")
    print("5. Add PyTorch LSTM when available for Python 3.13")
    
    print("\n" + "="*60)
    print("Integration demo completed!")
    print("="*60)

if __name__ == "__main__":
    main()
