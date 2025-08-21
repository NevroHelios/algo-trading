"""
Strategy Switcher - Quick Configuration Tool
Easily switch between different statistical cluster strategies
"""

import yaml
import os


def load_config():
    """Load current configuration"""
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def save_config(config):
    """Save configuration"""
    try:
        with open("config/config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                width=120,
                indent=2,
            )
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def switch_strategy():
    """Interactive strategy switcher"""

    print("=" * 60)
    print("🔄 STATISTICAL CLUSTERS STRATEGY SWITCHER")
    print("=" * 60)

    config = load_config()
    if not config:
        return

    # Strategy options
    strategies = {
        "1": {
            "name": "Pairs Trading System",
            "description": "Mean reversion with validation",
            "config": {"mode": "predefined", "predefined_strategy": "pairs_trading"},
        },
        "2": {
            "name": "Volatility Breakout System",
            "description": "Volatility trading with regime detection",
            "config": {
                "mode": "predefined",
                "predefined_strategy": "volatility_breakout",
            },
        },
        "3": {
            "name": "Factor Portfolio",
            "description": "Multi-factor statistical arbitrage",
            "config": {"mode": "predefined", "predefined_strategy": "factor_portfolio"},
        },
        "4": {
            "name": "Adaptive Trend Following",
            "description": "Momentum with adaptive regime detection",
            "config": {"mode": "predefined", "predefined_strategy": "adaptive_trend"},
        },
        "5": {
            "name": "Custom: Momentum + Volatility",
            "description": "Custom blend of momentum and volatility clusters",
            "config": {
                "mode": "custom",
                "custom_clusters": ["cluster_2", "cluster_3", "cluster_7"],
                "cluster_weights": {
                    "cluster_2": 0.5,
                    "cluster_3": 0.4,
                    "cluster_7": 0.1,
                },
            },
        },
        "6": {
            "name": "Custom: All Statistical Clusters",
            "description": "Equal weight ensemble of all 7 clusters",
            "config": {
                "mode": "custom",
                "custom_clusters": [
                    "cluster_1",
                    "cluster_2",
                    "cluster_3",
                    "cluster_4",
                    "cluster_5",
                    "cluster_6",
                    "cluster_7",
                ],
                "cluster_weights": {
                    "cluster_1": 0.143,
                    "cluster_2": 0.143,
                    "cluster_3": 0.143,
                    "cluster_4": 0.143,
                    "cluster_5": 0.143,
                    "cluster_6": 0.143,
                    "cluster_7": 0.141,
                },
            },
        },
    }

    # Show current configuration
    current_mode = config.get("cluster_strategy", {}).get("mode", "unknown")
    current_predefined = config.get("cluster_strategy", {}).get(
        "predefined_strategy", "none"
    )
    current_custom = config.get("cluster_strategy", {}).get("custom_clusters", [])

    print(f"📊 Current Configuration:")
    print(f"   Mode: {current_mode}")
    if current_mode == "predefined":
        print(f"   Strategy: {current_predefined}")
    elif current_mode == "custom":
        print(f"   Clusters: {', '.join(current_custom)}")
    print()

    # Show options
    print("📋 Available Strategies:")
    for key, strategy in strategies.items():
        print(f"   {key}. {strategy['name']}")
        print(f"      {strategy['description']}")

    print("   0. Exit without changes")
    print()

    # Get user choice
    choice = input("🎯 Select strategy (0-6): ").strip()

    if choice == "0":
        print("No changes made.")
        return

    if choice not in strategies:
        print("❌ Invalid choice!")
        return

    # Update configuration
    selected = strategies[choice]

    # Update cluster strategy
    config["strategy"] = "statistical_clusters"
    config["cluster_strategy"] = selected["config"]

    # Save configuration
    if save_config(config):
        print(f"✅ Successfully switched to: {selected['name']}")
        print(f"📝 {selected['description']}")
        print()
        print("🚀 Run 'uv run main.py' to test the new configuration!")
    else:
        print("❌ Failed to save configuration!")


def quick_test():
    """Quick test of current configuration"""
    print("🧪 Quick Test - Current Configuration")
    print("=" * 50)

    # Load and display current config
    config = load_config()
    if not config:
        return

    strategy_mode = config.get("cluster_strategy", {}).get("mode", "unknown")
    print(f"Strategy Mode: {strategy_mode}")

    if strategy_mode == "predefined":
        predefined = config.get("cluster_strategy", {}).get(
            "predefined_strategy", "none"
        )
        print(f"Predefined Strategy: {predefined}")
    elif strategy_mode == "custom":
        clusters = config.get("cluster_strategy", {}).get("custom_clusters", [])
        weights = config.get("cluster_strategy", {}).get("cluster_weights", {})
        print(f"Custom Clusters: {', '.join(clusters)}")
        print("Weights:")
        for cluster, weight in weights.items():
            print(f"  {cluster}: {weight}")

    print(f"Ticker: {config.get('ticker', 'N/A')}")
    print(f"Initial Cash: ₹{config.get('initial_cash', 0):,.2f}")
    print()

    # Ask if user wants to run test
    run_test = input("🚀 Run backtest now? (y/n): ").strip().lower()
    if run_test in ["y", "yes"]:
        print("Starting backtest...")
        os.system("uv run main.py")


def main():
    """Main menu"""
    while True:
        print("\n" + "=" * 60)
        print("🎯 STATISTICAL CLUSTERS MANAGEMENT")
        print("=" * 60)
        print("1. Switch Strategy")
        print("2. Quick Test Current Configuration")
        print("3. Show Demo")
        print("4. Exit")
        print()

        choice = input("Select option (1-4): ").strip()

        if choice == "1":
            switch_strategy()
        elif choice == "2":
            quick_test()
        elif choice == "3":
            os.system("uv run demo_clusters.py")
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
