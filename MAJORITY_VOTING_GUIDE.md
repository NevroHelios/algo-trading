# Multi-Timeframe Majority Voting System

## 📊 What is `primary_timeframe`?

The `primary_timeframe` parameter in your configuration determines which timeframe's data structure is used as the base for the backtesting loop. However, with the new majority voting system, it's no longer used for signal generation.

**Before (old system):**
- Only the `primary_timeframe` data was used for generating signals
- Other timeframes were fetched but ignored

**Now (new majority voting system):**
- ALL timeframes are analyzed for signals
- `primary_timeframe` only determines the data structure length for iteration
- Final trading decision is based on majority vote across all timeframes

## 🗳️ How Majority Voting Works

### Basic Process:
1. **Fetch Data**: Download data for all configured timeframes (`1d`, `15m`, `1h`)
2. **Analyze Each**: Generate signals for each timeframe independently using technical indicators
3. **Count Votes**: Tally BUY and SELL signals from each timeframe
4. **Make Decision**: Choose the majority vote or handle ties appropriately

### Example Scenarios:

#### Scenario 1: Clear Majority BUY
```
30m: BUY (strength: 0.8)
1d:  BUY (strength: 0.9) 
15m: SELL (strength: 0.6)

Result: BUY (2 votes vs 1 vote)
```

#### Scenario 2: Clear Majority SELL
```
30m: SELL (strength: 0.85)
1d:  SELL (strength: 0.75)
15m: BUY (strength: 0.70)

Result: SELL (2 votes vs 1 vote)
```

#### Scenario 3: Tie Situation
```
1d:  BUY (strength: 0.8)
15m: SELL (strength: 0.8)

Result: HOLD (tie situation)
```

## ⚙️ Configuration Options

```yaml
# Multi-timeframe configuration
timeframes: ["1d", "15m", "1h"]  # All timeframes to analyze
primary_timeframe: "1d"  # Used for data structure (not for signals anymore)

# Majority voting configuration
majority_voting:
  enabled: true
  minimum_timeframes: 2        # Minimum timeframes needed for a signal
  require_majority: true       # If false, any timeframe agreement counts
  weight_by_strength: true     # Weight votes by signal strength
```

### Parameter Explanations:

- **`minimum_timeframes`**: Minimum number of timeframes that must have signals before making a decision
- **`require_majority`**: 
  - `true`: Strict majority required (2 out of 3, 3 out of 5, etc.)
  - `false`: Any agreement counts, strongest signal wins
- **`weight_by_strength`**: 
  - `true`: Average signal strength must meet threshold
  - `false`: All votes count equally regardless of strength

## 🎯 Signal Strength Threshold

Each timeframe generates a signal with a strength between 0.0 and 1.0:
- **Strength 0.6+**: Strong signal, likely to execute
- **Strength 0.4-0.6**: Moderate signal
- **Strength <0.4**: Weak signal, likely to be ignored

The system requires average strength ≥ `signal_strength_threshold` (default 0.6) to execute trades.

## 📈 Technical Indicators Used

Each timeframe analyzes multiple indicators:
- **Moving Averages**: Multiple fast/slow combinations
- **RSI**: Relative Strength Index for overbought/oversold conditions
- **Bollinger Bands**: Price volatility and mean reversion
- **Ichimoku Cloud**: Momentum and trend analysis
- **Support/Resistance**: Key price levels

## 🚀 Example Output

```
=== MULTI-TIMEFRAME ANALYSIS ===
1d: BUY (strength: 0.85)
  Reasons: MA5 > MA20, MA5 > MA30, RSI14 oversold (28.5)
1h: BUY (strength: 0.75)
  Reasons: MA5 > MA20, Price near support (20)
15m: SELL (strength: 0.60)
  Reasons: MA5 < MA20, RSI14 overbought (72.1)

VOTING RESULTS:
BUY votes: 2
SELL votes: 1

🟢 BUY SIGNAL (strength: 0.80)
Buy timeframes:
  - 1d: 0.85 (MA5 > MA20, MA5 > MA30, RSI14 oversold (28.5))
  - 1h: 0.75 (MA5 > MA20, Price near support (20))
```

## ⚠️ Important Notes

1. **Intraday Data Limitations**: Yahoo Finance only provides 60 days of intraday data (15m, 1h)
2. **Minimum Timeframes**: Set appropriately - too high and you'll get no signals
3. **Signal Quality**: The system prioritizes signal strength over quantity
4. **Tie Handling**: Ties result in HOLD to avoid uncertain trades

## 🔧 Customization Tips

- **Conservative Trading**: Set `minimum_timeframes: 3` and `require_majority: true`
- **Aggressive Trading**: Set `minimum_timeframes: 1` and `require_majority: false`
- **Quality Focus**: Keep `weight_by_strength: true` and high `signal_strength_threshold`
- **Quantity Focus**: Set `weight_by_strength: false` for equal vote weighting

This system gives you sophisticated multi-timeframe analysis with democratic decision-making across different time horizons!
