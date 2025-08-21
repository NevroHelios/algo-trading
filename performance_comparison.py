"""
Performance Comparison Tool
Compare different statistical cluster strategies
"""

import yaml
import pandas as pd
from datetime import datetime
import os


def run_strategy_comparison():
    """Run comparison of all predefined strategies"""

    print("=" * 80)
    print("📊 STATISTICAL CLUSTERS PERFORMANCE COMPARISON")
    print("=" * 80)

    # Strategies to test
    strategies = {
        "pairs_trading": "Pairs Trading System",
        "volatility_breakout": "Volatility Breakout System",
        "factor_portfolio": "Factor Portfolio",
        "adaptive_trend": "Adaptive Trend Following",
    }

    # Load original config
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            original_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    results = {}

    print("🚀 Running backtests for all strategies...")
    print("This may take a few minutes...")
    print()

    for strategy_id, strategy_name in strategies.items():
        print(f"📈 Testing: {strategy_name}")

        # Update config for this strategy
        config = original_config.copy()
        config["strategy"] = "statistical_clusters"
        config["cluster_strategy"] = {
            "mode": "predefined",
            "predefined_strategy": strategy_id,
        }

        # Save temporary config
        try:
            with open("config/config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving config: {e}")
            continue

        # Run backtest (capture output)
        import subprocess

        try:
            result = subprocess.run(
                ["uv", "run", "main.py"], capture_output=True, text=True, timeout=120
            )

            # Parse basic results from output
            output = result.stdout
            results[strategy_id] = {
                "name": strategy_name,
                "success": result.returncode == 0,
                "output": output,
            }

            if result.returncode == 0:
                print(f"   ✅ Completed successfully")
            else:
                print(f"   ❌ Failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout (>2 minutes)")
            results[strategy_id] = {
                "name": strategy_name,
                "success": False,
                "output": "Timeout",
            }
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[strategy_id] = {
                "name": strategy_name,
                "success": False,
                "output": str(e),
            }

    # Restore original config
    try:
        with open("config/config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(original_config, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        print(f"Warning: Could not restore original config: {e}")

    print("\n" + "=" * 80)
    print("📋 COMPARISON RESULTS")
    print("=" * 80)

    # Display results
    for strategy_id, result in results.items():
        print(f"\n🔸 {result['name']} ({strategy_id})")
        print(f"   Status: {'✅ Success' if result['success'] else '❌ Failed'}")

        if result["success"] and result["output"]:
            # Try to extract key metrics from output
            lines = result["output"].split("\n")
            for line in lines:
                if "Final Portfolio Value:" in line:
                    print(f"   {line.strip()}")
                elif "Total Return:" in line:
                    print(f"   {line.strip()}")
                elif "Signals generated:" in line:
                    print(f"   {line.strip()}")
                elif "Trades executed:" in line:
                    print(f"   {line.strip()}")
        elif not result["success"]:
            print(f"   Error: {result['output'][:100]}...")

    print("\n" + "=" * 80)
    print("✅ COMPARISON COMPLETE")
    print("=" * 80)

    return results


def generate_comparison_report():
    """Generate a detailed comparison report"""

    print("📊 Generating Detailed Comparison Report...")

    # Strategy descriptions
    strategy_details = {
        "pairs_trading": {
            "name": "Pairs Trading System",
            "clusters": ["Mean Reversion", "Validation"],
            "best_for": "Equity pairs, ETFs, commodities",
            "risk_profile": "Medium",
            "holding_period": "Days to weeks",
            "market_conditions": "Range-bound markets",
        },
        "volatility_breakout": {
            "name": "Volatility Breakout System",
            "clusters": ["Volatility Trading", "Regime Detection", "Validation"],
            "best_for": "Options, volatility ETFs",
            "risk_profile": "High",
            "holding_period": "Hours to days",
            "market_conditions": "High volatility periods",
        },
        "factor_portfolio": {
            "name": "Factor Portfolio",
            "clusters": ["Multi-Factor", "Validation"],
            "best_for": "Equity portfolios, cross-asset strategies",
            "risk_profile": "Medium-Low",
            "holding_period": "Weeks to months",
            "market_conditions": "All market conditions",
        },
        "adaptive_trend": {
            "name": "Adaptive Trend Following",
            "clusters": ["Momentum", "Regime Detection", "Validation"],
            "best_for": "FX, indices, futures",
            "risk_profile": "Medium-High",
            "holding_period": "Days to weeks",
            "market_conditions": "Trending markets",
        },
    }

    print("\n📋 STRATEGY COMPARISON MATRIX")
    print("=" * 100)

    # Create comparison table
    print(
        f"{'Strategy':<25} {'Clusters':<30} {'Risk':<12} {'Period':<15} {'Best Markets':<15}"
    )
    print("-" * 100)

    for strategy_id, details in strategy_details.items():
        clusters_str = ", ".join(details["clusters"])
        if len(clusters_str) > 28:
            clusters_str = clusters_str[:25] + "..."

        print(
            f"{details['name']:<25} {clusters_str:<30} {details['risk_profile']:<12} "
            f"{details['holding_period']:<15} {details['market_conditions']:<15}"
        )

    print("\n📈 STRATEGY RECOMMENDATIONS")
    print("=" * 80)

    recommendations = [
        {
            "scenario": "New to Algorithmic Trading",
            "strategy": "factor_portfolio",
            "reason": "Lower risk, diversified approach with robust validation",
        },
        {
            "scenario": "Equity Pairs Trading",
            "strategy": "pairs_trading",
            "reason": "Specialized mean reversion designed for correlated assets",
        },
        {
            "scenario": "High Volatility Markets",
            "strategy": "volatility_breakout",
            "reason": "Capitalizes on volatility spikes and regime changes",
        },
        {
            "scenario": "Trending Markets",
            "strategy": "adaptive_trend",
            "reason": "Momentum strategies with adaptive regime detection",
        },
        {
            "scenario": "Conservative Approach",
            "strategy": "factor_portfolio",
            "reason": "Multi-factor diversification with lower drawdowns",
        },
        {
            "scenario": "Day Trading",
            "strategy": "volatility_breakout",
            "reason": "Short-term signals and microstructure analysis",
        },
    ]

    for rec in recommendations:
        strategy_name = strategy_details[rec["strategy"]]["name"]
        print(f"🎯 {rec['scenario']}:")
        print(f"   Recommended: {strategy_name}")
        print(f"   Reason: {rec['reason']}")
        print()

    print("💡 CUSTOM COMBINATIONS")
    print("=" * 50)

    custom_examples = [
        {
            "name": "Conservative Blend",
            "clusters": ["cluster_1", "cluster_4", "cluster_7"],
            "weights": {"cluster_1": 0.4, "cluster_4": 0.4, "cluster_7": 0.2},
            "description": "Mean reversion + multi-factor with heavy validation",
        },
        {
            "name": "Aggressive Momentum",
            "clusters": ["cluster_2", "cluster_3", "cluster_5"],
            "weights": {"cluster_2": 0.5, "cluster_3": 0.3, "cluster_5": 0.2},
            "description": "Momentum + volatility with regime detection",
        },
        {
            "name": "All-Weather Portfolio",
            "clusters": [
                "cluster_1",
                "cluster_2",
                "cluster_4",
                "cluster_5",
                "cluster_7",
            ],
            "weights": {
                "cluster_1": 0.25,
                "cluster_2": 0.25,
                "cluster_4": 0.25,
                "cluster_5": 0.15,
                "cluster_7": 0.1,
            },
            "description": "Diversified across multiple statistical approaches",
        },
    ]

    for example in custom_examples:
        print(f"🔸 {example['name']}")
        print(f"   Clusters: {', '.join(example['clusters'])}")
        print(f"   Description: {example['description']}")
        clusters_display = ", ".join(
            [f"{k}:{v}" for k, v in example["weights"].items()]
        )
        print(f"   Weights: {clusters_display}")
        print()


if __name__ == "__main__":
    print("🎯 Statistical Clusters Performance Analysis")
    print("=" * 60)
    print("1. Quick Strategy Comparison")
    print("2. Detailed Analysis Report")
    print("3. Full Performance Comparison (runs all strategies)")
    print()

    choice = input("Select option (1-3): ").strip()

    if choice == "1":
        generate_comparison_report()
    elif choice == "2":
        generate_comparison_report()
        print("\n" + "=" * 60)
        print("For live performance data, run option 3")
    elif choice == "3":
        run_strategy_comparison()
    else:
        print("Invalid choice!")
