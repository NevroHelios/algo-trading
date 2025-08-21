"""
Statistical Clusters Demo Script
Demonstrates all 4 predefined strategies and custom cluster combinations
"""

import yaml
from strategies.statistical_clusters_strategy import get_strategy_info


def demo_statistical_clusters():
    """Demonstrate the statistical clusters trading system"""

    print("=" * 80)
    print("🎯 STATISTICAL CLUSTERS FOR ALGORITHMIC TRADING DEMO")
    print("=" * 80)
    print("Currency: ₹ (Indian Rupees)")
    print("Market: Indian Equity Market (NSE/BSE)")
    print()

    # Get strategy information
    info = get_strategy_info()

    print("📋 AVAILABLE PREDEFINED STRATEGIES:")
    print()

    for strategy_id, strategy_info in info["predefined_strategies"].items():
        print(f"🔸 {strategy_info['name']}")
        print(f"   ID: {strategy_id}")
        print(f"   Description: {strategy_info['description']}")
        print(f"   Clusters: {', '.join(strategy_info['clusters'])}")
        print(f"   Best for: {strategy_info['best_for']}")
        print()

    print("🔬 AVAILABLE STATISTICAL CLUSTERS:")
    print()

    for cluster_id, cluster_desc in info["clusters"].items():
        print(f"🔸 {cluster_id}: {cluster_desc}")

    print("\n" + "=" * 80)
    print("⚙️ CONFIGURATION EXAMPLES")
    print("=" * 80)

    # Example configurations
    examples = [
        {
            "name": "Pairs Trading System",
            "config": {
                "cluster_strategy": {
                    "mode": "predefined",
                    "predefined_strategy": "pairs_trading",
                }
            },
            "description": "Mean reversion strategy perfect for equity pairs trading",
        },
        {
            "name": "Volatility Breakout System",
            "config": {
                "cluster_strategy": {
                    "mode": "predefined",
                    "predefined_strategy": "volatility_breakout",
                }
            },
            "description": "Volatility-based trading with regime detection",
        },
        {
            "name": "Factor Portfolio",
            "config": {
                "cluster_strategy": {
                    "mode": "predefined",
                    "predefined_strategy": "factor_portfolio",
                }
            },
            "description": "Multi-factor statistical arbitrage strategy",
        },
        {
            "name": "Adaptive Trend Following",
            "config": {
                "cluster_strategy": {
                    "mode": "predefined",
                    "predefined_strategy": "adaptive_trend",
                }
            },
            "description": "Momentum strategy with adaptive regime detection",
        },
        {
            "name": "Custom Combination",
            "config": {
                "cluster_strategy": {
                    "mode": "custom",
                    "custom_clusters": ["cluster_2", "cluster_3", "cluster_7"],
                    "cluster_weights": {
                        "cluster_2": 0.5,  # Momentum
                        "cluster_3": 0.4,  # Volatility
                        "cluster_7": 0.1,  # Validation
                    },
                }
            },
            "description": "Custom momentum + volatility combination",
        },
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n📝 Example {i}: {example['name']}")
        print(f"   {example['description']}")
        print(f"   Configuration snippet for config.yaml:")
        print("   ```yaml")
        for key, value in example["config"].items():
            print(f"   {key}:")
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        print(f"     {subkey}:")
                        for subsubkey, subsubvalue in subvalue.items():
                            print(f"       {subsubkey}: {subsubvalue}")
                    elif isinstance(subvalue, list):
                        print(f"     {subkey}: {subvalue}")
                    else:
                        print(f'     {subkey}: "{subvalue}"')
        print("   ```")

    print("\n" + "=" * 80)
    print("🔧 HOW TO USE")
    print("=" * 80)

    print("""
1. Edit config/config.yaml to set your desired strategy:
   - Change 'strategy' to 'statistical_clusters'
   - Set cluster_strategy.mode to 'predefined' or 'custom'
   - Choose your predefined_strategy or custom_clusters

2. Adjust cluster-specific parameters:
   - Each cluster has detailed configuration options
   - All parameters are documented in the config file
   - Weights can be adjusted for ensemble combinations

3. Run the backtest:
   - Execute: uv run main.py
   - System will show detailed analysis for each trade decision
   - Results include Indian tax calculations and trading costs

4. Indian Market Features:
   - Zero brokerage for equity delivery
   - STT (Securities Transaction Tax): 0.1%
   - Transaction charges: 0.00297% (NSE)
   - SEBI charges: ₹10 per crore
   - Stamp charges: 0.015% on buy side
   - GST: 18% on applicable charges
   - Tax rates: 15% STCG, 10% LTCG (with ₹1 lakh exemption)

5. Performance Analysis:
   - Detailed cluster-by-cluster signal analysis
   - Real-time price and percentage changes in ₹
   - Tax-efficient portfolio tracking
   - FIFO tax lot accounting
   - Comprehensive cost breakdown
    """)

    print("\n" + "=" * 80)
    print("🎯 CURRENT CONFIGURATION")
    print("=" * 80)

    # Show current configuration
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        current_strategy = config.get("cluster_strategy", {})
        mode = current_strategy.get("mode", "not set")

        print(f"Strategy Mode: {mode}")

        if mode == "predefined":
            predefined = current_strategy.get("predefined_strategy", "not set")
            print(f"Predefined Strategy: {predefined}")

            if predefined in info["predefined_strategies"]:
                strategy_info = info["predefined_strategies"][predefined]
                print(f"Description: {strategy_info['description']}")
                print(f"Active Clusters: {', '.join(strategy_info['clusters'])}")

        elif mode == "custom":
            custom_clusters = current_strategy.get("custom_clusters", [])
            print(f"Custom Clusters: {', '.join(custom_clusters)}")

        print(f"Initial Capital: ₹{config.get('initial_cash', 0):,.2f}")
        print(f"Ticker: {config.get('ticker', 'not set')}")
        print(
            f"Date Range: {config.get('start_date', 'not set')} to {config.get('end_date', 'not set')}"
        )

    except Exception as e:
        print(f"Could not read current configuration: {e}")

    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE - Ready for Trading!")
    print("=" * 80)


if __name__ == "__main__":
    demo_statistical_clusters()
