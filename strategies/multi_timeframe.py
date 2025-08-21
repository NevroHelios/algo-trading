import pandas as pd
from ml_algorithms import MLEnsemble


class MyStrategy:
    def __init__(self, params):
        self.params = params
        self.min_timeframes_agree = params.get("min_timeframes_agree", 2)
        self.signal_strength_threshold = params.get("signal_strength_threshold", 0.6)

        # RSI thresholds
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.rsi_overbought = params.get("rsi_overbought", 70)

        # Majority voting configuration
        self.majority_voting = params.get(
            "majority_voting",
            {
                "enabled": True,
                "minimum_timeframes": 1,  # Default changed to 1
                "require_majority": True,
                "weight_by_strength": True,
            },
        )
        
        # ML Algorithms configuration
        self.ml_config = params.get("ml_algorithms", {})
        self.ml_ensemble = None
        if self.ml_config.get("enabled", False):
            self.ml_ensemble = MLEnsemble(self.ml_config)
        
        # Weight for traditional vs ML signals
        self.traditional_weight = self.ml_config.get("ensemble", {}).get("weight", 0.75)

    def analyze_single_timeframe(self, df, timeframe, current_index):
        """Analyze signals for a single timeframe"""
        if current_index >= len(df):
            return {"signal": "HOLD", "strength": 0.0, "reasons": []}

        row = df.iloc[current_index]
        signals = []
        reasons = []

        # Moving Average signals
        ma_signals = self._analyze_moving_averages(row)
        signals.extend(ma_signals["signals"])
        reasons.extend(ma_signals["reasons"])

        # RSI signals
        rsi_signals = self._analyze_rsi(row)
        signals.extend(rsi_signals["signals"])
        reasons.extend(rsi_signals["reasons"])

        # Bollinger Bands signals
        bb_signals = self._analyze_bollinger_bands(row)
        signals.extend(bb_signals["signals"])
        reasons.extend(bb_signals["reasons"])

        # Ichimoku signals
        ichi_signals = self._analyze_ichimoku(row)
        signals.extend(ichi_signals["signals"])
        reasons.extend(ichi_signals["reasons"])

        # Support/Resistance signals
        sr_signals = self._analyze_support_resistance(row)
        signals.extend(sr_signals["signals"])
        reasons.extend(sr_signals["reasons"])

        # Calculate overall signal and strength
        if not signals:
            return {"signal": "HOLD", "strength": 0.0, "reasons": reasons}

        buy_signals = sum(1 for s in signals if s == "BUY")
        sell_signals = sum(1 for s in signals if s == "SELL")
        total_signals = len(signals)

        if buy_signals > sell_signals:
            signal = "BUY"
            strength = buy_signals / total_signals
        elif sell_signals > buy_signals:
            signal = "SELL"
            strength = sell_signals / total_signals
        else:
            signal = "HOLD"
            strength = 0.0

        return {
            "signal": signal,
            "strength": strength,
            "reasons": reasons,
            "timeframe": timeframe,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "total_signals": total_signals,
        }

    def _analyze_moving_averages(self, row):
        """Analyze moving average crossovers"""
        signals = []
        reasons = []

        # Check all MA combinations
        for fast_window in self.params["fast_ma_windows"]:
            for slow_window in self.params["slow_ma_windows"]:
                if fast_window >= slow_window:
                    continue

                fast_ma_col = f"fast_ma_{fast_window}"
                slow_ma_col = f"slow_ma_{slow_window}"

                if fast_ma_col in row and slow_ma_col in row:
                    if pd.notna(row[fast_ma_col]) and pd.notna(row[slow_ma_col]):
                        if row[fast_ma_col] > row[slow_ma_col]:
                            signals.append("BUY")
                            reasons.append(f"MA{fast_window} > MA{slow_window}")
                        else:
                            signals.append("SELL")
                            reasons.append(f"MA{fast_window} < MA{slow_window}")

        return {"signals": signals, "reasons": reasons}

    def _analyze_rsi(self, row):
        """Analyze RSI signals"""
        signals = []
        reasons = []

        for rsi_period in self.params["rsi_periods"]:
            rsi_col = f"rsi_{rsi_period}"
            if rsi_col in row and pd.notna(row[rsi_col]):
                rsi_value = row[rsi_col]
                if rsi_value < self.rsi_oversold:
                    signals.append("BUY")
                    reasons.append(f"RSI{rsi_period} oversold ({rsi_value:.1f})")
                elif rsi_value > self.rsi_overbought:
                    signals.append("SELL")
                    reasons.append(f"RSI{rsi_period} overbought ({rsi_value:.1f})")

        return {"signals": signals, "reasons": reasons}

    def _analyze_bollinger_bands(self, row):
        """Analyze Bollinger Bands signals"""
        signals = []
        reasons = []

        for bb_period in self.params["bb_periods"]:
            for bb_std in self.params["bb_std_devs"]:
                upper_col = f"bb_upper_{bb_period}_{bb_std}"
                lower_col = f"bb_lower_{bb_period}_{bb_std}"

                if (
                    upper_col in row
                    and lower_col in row
                    and pd.notna(row[upper_col])
                    and pd.notna(row[lower_col])
                ):
                    if row["Close"] <= row[lower_col]:
                        signals.append("BUY")
                        reasons.append(f"Price below BB lower ({bb_period},{bb_std})")
                    elif row["Close"] >= row[upper_col]:
                        signals.append("SELL")
                        reasons.append(f"Price above BB upper ({bb_period},{bb_std})")

        return {"signals": signals, "reasons": reasons}

    def _analyze_atr(self, row):
        """Analyze ATR for volatility signals"""
        signals = []
        reasons = []

        for atr_period in self.params["atr_periods"]:
            atr_col = f"atr_{atr_period}"
            if atr_col in row and pd.notna(row[atr_col]):
                # ATR itself doesn't generate buy/sell signals
                # but high volatility might suggest caution
                pass  # For now, no specific signals from ATR

        return {"signals": signals, "reasons": reasons}

    def _analyze_ichimoku(self, row):
        """Analyze Ichimoku Cloud signals"""
        signals = []
        reasons = []

        for conv in self.params["ichimoku_conversion"]:
            for base in self.params["ichimoku_base"]:
                for span_b in self.params["ichimoku_span_b"]:
                    conv_col = f"ichi_conv_{conv}_{base}_{span_b}"
                    base_col = f"ichi_base_{conv}_{base}_{span_b}"
                    span_a_col = f"ichi_span_a_{conv}_{base}_{span_b}"
                    span_b_col = f"ichi_span_b_{conv}_{base}_{span_b}"

                    if (
                        conv_col in row
                        and base_col in row
                        and pd.notna(row[conv_col])
                        and pd.notna(row[base_col])
                    ):
                        # Tenkan-Kijun cross
                        if row[conv_col] > row[base_col]:
                            signals.append("BUY")
                            reasons.append(
                                f"Ichimoku TK cross bullish ({conv},{base},{span_b})"
                            )
                        else:
                            signals.append("SELL")
                            reasons.append(
                                f"Ichimoku TK cross bearish ({conv},{base},{span_b})"
                            )

                    # Price vs Cloud
                    if (
                        span_a_col in row
                        and span_b_col in row
                        and pd.notna(row[span_a_col])
                        and pd.notna(row[span_b_col])
                    ):
                        cloud_top = max(row[span_a_col], row[span_b_col])
                        cloud_bottom = min(row[span_a_col], row[span_b_col])

                        if row["Close"] > cloud_top:
                            signals.append("BUY")
                            reasons.append(
                                f"Price above Ichimoku cloud ({conv},{base},{span_b})"
                            )
                        elif row["Close"] < cloud_bottom:
                            signals.append("SELL")
                            reasons.append(
                                f"Price below Ichimoku cloud ({conv},{base},{span_b})"
                            )

        return {"signals": signals, "reasons": reasons}

    def _analyze_support_resistance(self, row):
        """Analyze support and resistance levels"""
        signals = []
        reasons = []

        for sr_period in self.params["support_resistance_periods"]:
            support_col = f"support_{sr_period}"
            resistance_col = f"resistance_{sr_period}"

            if (
                support_col in row
                and resistance_col in row
                and pd.notna(row[support_col])
                and pd.notna(row[resistance_col])
            ):
                # Price near support
                support_distance = (row["Close"] - row[support_col]) / row["Close"]
                if support_distance < 0.02:  # Within 2% of support
                    signals.append("BUY")
                    reasons.append(f"Price near support ({sr_period})")

                # Price near resistance
                resistance_distance = (row[resistance_col] - row["Close"]) / row[
                    "Close"
                ]
                if resistance_distance < 0.02:  # Within 2% of resistance
                    signals.append("SELL")
                    reasons.append(f"Price near resistance ({sr_period})")

        return {"signals": signals, "reasons": reasons}

    def generate_signal(self, row):
        """Generate signal using multi-timeframe majority voting with ML integration"""
        # Check if we have multi-timeframe data available
        if "_df_attrs" in row and "timeframe_data" in row["_df_attrs"]:
            return self._generate_multi_timeframe_signal(row)
        else:
            # Fallback to single timeframe analysis
            return self._generate_single_timeframe_signal(row)

    def _generate_multi_timeframe_signal(self, row):
        """Generate signal using majority voting across timeframes with ML integration"""
        timeframe_data = row["_df_attrs"]["timeframe_data"]
        current_index = row["_index"]

        # Get traditional timeframe signals
        timeframe_signals = []
        timeframe_analyses = {}

        print("\n=== MULTI-TIMEFRAME ANALYSIS ===")

        # Analyze each timeframe with traditional indicators
        for timeframe, df in timeframe_data.items():
            try:
                # Find the closest index for this timeframe
                tf_index = min(current_index, len(df) - 1)
                if tf_index < 0:
                    continue

                analysis = self.analyze_single_timeframe(df, timeframe, tf_index)
                timeframe_analyses[timeframe] = analysis

                if analysis["signal"] != "HOLD":
                    timeframe_signals.append(
                        {
                            "timeframe": timeframe,
                            "signal": analysis["signal"],
                            "strength": analysis["strength"],
                            "reasons": analysis["reasons"][:3],  # Top 3 reasons
                        }
                    )

                print(
                    f"{timeframe}: {analysis['signal']} (strength: {analysis['strength']:.2f})"
                )
                if analysis["reasons"]:
                    print(f"  Reasons: {', '.join(analysis['reasons'][:2])}")

            except Exception as e:
                print(f"Error analyzing {timeframe}: {e}")
                continue

        # Get traditional signal
        traditional_signal = self._process_timeframe_signals(timeframe_signals)
        
        # Get ML ensemble signal if enabled
        ml_signal = "HOLD"
        ml_strength = 0.0
        ml_reason = "ML disabled"
        
        if self.ml_ensemble:
            try:
                # Use primary timeframe data for ML analysis
                primary_timeframe = row["_df_attrs"].get("primary_timeframe", "1d")
                primary_df = timeframe_data.get(primary_timeframe)
                
                if primary_df is not None:
                    ml_result = self.ml_ensemble.generate_ensemble_signal(primary_df, current_index)
                    ml_signal = ml_result.get("signal", "HOLD")
                    ml_strength = ml_result.get("strength", 0.0)
                    ml_reason = ml_result.get("reason", "ML analysis")
                    
                    print(f"\n🤖 ML ENSEMBLE ANALYSIS:")
                    print(f"ML Signal: {ml_signal} (strength: {ml_strength:.2f})")
                    print(f"ML Reason: {ml_reason}")
                    
                    # Show individual algorithm results
                    if "algorithm_signals" in ml_result:
                        for algo_name, algo_result in ml_result["algorithm_signals"].items():
                            print(f"  {algo_name}: {algo_result['signal']} ({algo_result['strength']:.2f}) - {algo_result['reason']}")
                            
            except Exception as e:
                print(f"ML analysis error: {e}")
        
        # Combine traditional and ML signals
        final_signal = self._combine_traditional_and_ml_signals(
            traditional_signal, ml_signal, ml_strength, ml_reason
        )
        
        return final_signal
    
    def _process_timeframe_signals(self, timeframe_signals):
        """Process traditional timeframe signals"""
        if not timeframe_signals:
            return "HOLD"

        # Check minimum timeframes requirement
        if len(timeframe_signals) < self.majority_voting["minimum_timeframes"]:
            return "HOLD"

        # Count votes by signal type
        buy_votes = []
        sell_votes = []

        for signal_data in timeframe_signals:
            if signal_data["signal"] == "BUY":
                buy_votes.append(signal_data)
            elif signal_data["signal"] == "SELL":
                sell_votes.append(signal_data)

        buy_count = len(buy_votes)
        sell_count = len(sell_votes)

        print("\nTRADITIONAL TIMEFRAME VOTING:")
        print(f"BUY votes: {buy_count}")
        print(f"SELL votes: {sell_count}")

        # Determine signal based on configuration
        if self.majority_voting["require_majority"]:
            if buy_count > sell_count:
                return self._evaluate_signal_strength("BUY", buy_votes)
            elif sell_count > buy_count:
                return self._evaluate_signal_strength("SELL", sell_votes)
            else:
                return "HOLD"
        else:
            if buy_count > 0 and sell_count > 0:
                avg_buy_strength = sum(vote["strength"] for vote in buy_votes) / buy_count
                avg_sell_strength = sum(vote["strength"] for vote in sell_votes) / sell_count
                
                if avg_buy_strength > avg_sell_strength:
                    return self._evaluate_signal_strength("BUY", buy_votes)
                else:
                    return self._evaluate_signal_strength("SELL", sell_votes)
            elif buy_count > 0:
                return self._evaluate_signal_strength("BUY", buy_votes)
            elif sell_count > 0:
                return self._evaluate_signal_strength("SELL", sell_votes)
            else:
                return "HOLD"
    
    def _evaluate_signal_strength(self, signal_type, votes):
        """Evaluate if signal meets strength threshold"""
        if self.majority_voting["weight_by_strength"]:
            avg_strength = sum(vote["strength"] for vote in votes) / len(votes)
        else:
            avg_strength = 1.0
        
        if avg_strength >= self.signal_strength_threshold:
            return signal_type
        else:
            return "HOLD"
    
    def _combine_traditional_and_ml_signals(self, traditional_signal, ml_signal, ml_strength, ml_reason):
        """Combine traditional timeframe analysis with ML ensemble signals"""
        print(f"\n🔄 SIGNAL COMBINATION:")
        print(f"Traditional: {traditional_signal}")
        print(f"ML Ensemble: {ml_signal} (strength: {ml_strength:.2f})")
        
        # Weight configuration
        traditional_weight = self.traditional_weight
        ml_weight = 1.0 - traditional_weight
        
        # Simple voting combination
        signals = []
        if traditional_signal != "HOLD":
            signals.append(("traditional", traditional_signal, traditional_weight))
        if ml_signal != "HOLD" and ml_strength >= 0.3:  # Minimum ML confidence
            signals.append(("ml", ml_signal, ml_weight * ml_strength))
        
        if not signals:
            print("Final Decision: HOLD (no strong signals)")
            return "HOLD"
        
        # Calculate weighted scores
        buy_score = sum(weight for source, signal, weight in signals if signal == "BUY")
        sell_score = sum(weight for source, signal, weight in signals if signal == "SELL")
        
        # Make final decision
        if buy_score > sell_score and buy_score >= 0.4:
            final_signal = "BUY"
            print(f"Final Decision: BUY (score: {buy_score:.2f})")
        elif sell_score > buy_score and sell_score >= 0.4:
            final_signal = "SELL"
            print(f"Final Decision: SELL (score: {sell_score:.2f})")
        else:
            final_signal = "HOLD"
            print(f"Final Decision: HOLD (buy: {buy_score:.2f}, sell: {sell_score:.2f})")
        
        return final_signal

    def _generate_multi_timeframe_signal(self, row):
        """Generate signal using majority voting across timeframes"""
        timeframe_data = row["_df_attrs"]["timeframe_data"]
        current_index = row["_index"]

        timeframe_signals = []
        timeframe_analyses = {}

        print("\n=== MULTI-TIMEFRAME ANALYSIS ===")

        # Analyze each timeframe
        for timeframe, df in timeframe_data.items():
            try:
                # Find the closest index for this timeframe
                tf_index = min(current_index, len(df) - 1)
                if tf_index < 0:
                    continue

                analysis = self.analyze_single_timeframe(df, timeframe, tf_index)
                timeframe_analyses[timeframe] = analysis

                if analysis["signal"] != "HOLD":
                    timeframe_signals.append(
                        {
                            "timeframe": timeframe,
                            "signal": analysis["signal"],
                            "strength": analysis["strength"],
                            "reasons": analysis["reasons"][:3],  # Top 3 reasons
                        }
                    )

                print(
                    f"{timeframe}: {analysis['signal']} (strength: {analysis['strength']:.2f})"
                )
                if analysis["reasons"]:
                    print(f"  Reasons: {', '.join(analysis['reasons'][:2])}")

            except Exception as e:
                print(f"Error analyzing {timeframe}: {e}")
                continue

        # Implement majority voting with configuration
        if not timeframe_signals:
            print("No clear signals from any timeframe - HOLD")
            return "HOLD"

        # Check minimum timeframes requirement
        if len(timeframe_signals) < self.majority_voting["minimum_timeframes"]:
            print(
                f"Only {len(timeframe_signals)} timeframes have signals, need at least {self.majority_voting['minimum_timeframes']} - HOLD"
            )
            return "HOLD"

        # Count votes by signal type
        buy_votes = []
        sell_votes = []

        for signal_data in timeframe_signals:
            if signal_data["signal"] == "BUY":
                buy_votes.append(signal_data)
            elif signal_data["signal"] == "SELL":
                sell_votes.append(signal_data)

        buy_count = len(buy_votes)
        sell_count = len(sell_votes)

        print("\nVOTING RESULTS:")
        print(f"BUY votes: {buy_count}")
        print(f"SELL votes: {sell_count}")

        # Determine final signal based on configuration
        if self.majority_voting["require_majority"]:
            # Strict majority required
            if buy_count > sell_count:
                return self._process_buy_decision(buy_votes)
            elif sell_count > buy_count:
                return self._process_sell_decision(sell_votes)
            else:
                print("TIE - No clear majority - HOLD")
                return "HOLD"
        else:
            # Any agreement counts, but prioritize stronger signals
            if buy_count > 0 and sell_count > 0:
                # Both buy and sell signals exist, choose stronger one
                avg_buy_strength = (
                    sum(vote["strength"] for vote in buy_votes) / buy_count
                    if buy_count > 0
                    else 0
                )
                avg_sell_strength = (
                    sum(vote["strength"] for vote in sell_votes) / sell_count
                    if sell_count > 0
                    else 0
                )

                if avg_buy_strength > avg_sell_strength:
                    return self._process_buy_decision(buy_votes)
                else:
                    return self._process_sell_decision(sell_votes)
            elif buy_count > 0:
                return self._process_buy_decision(buy_votes)
            elif sell_count > 0:
                return self._process_sell_decision(sell_votes)
            else:
                return "HOLD"

    def _process_buy_decision(self, buy_votes):
        """Process buy decision with strength weighting"""
        if self.majority_voting["weight_by_strength"]:
            avg_strength = sum(vote["strength"] for vote in buy_votes) / len(buy_votes)
        else:
            avg_strength = 1.0  # Treat all votes equally

        if avg_strength >= self.signal_strength_threshold:
            print(f"\n🟢 BUY SIGNAL (strength: {avg_strength:.2f})")
            print("Buy timeframes:")
            for vote in buy_votes:
                print(
                    f"  - {vote['timeframe']}: {vote['strength']:.2f} ({', '.join(vote['reasons'])})"
                )
            return "BUY"
        else:
            print(f"Buy signal but weak strength ({avg_strength:.2f}) - HOLD")
            return "HOLD"

    def _process_sell_decision(self, sell_votes):
        """Process sell decision with strength weighting"""
        if self.majority_voting["weight_by_strength"]:
            avg_strength = sum(vote["strength"] for vote in sell_votes) / len(
                sell_votes
            )
        else:
            avg_strength = 1.0  # Treat all votes equally

        if avg_strength >= self.signal_strength_threshold:
            print(f"\n🔴 SELL SIGNAL (strength: {avg_strength:.2f})")
            print("Sell timeframes:")
            for vote in sell_votes:
                print(
                    f"  - {vote['timeframe']}: {vote['strength']:.2f} ({', '.join(vote['reasons'])})"
                )
            return "SELL"
        else:
            print(f"Sell signal but weak strength ({avg_strength:.2f}) - HOLD")
            return "HOLD"

    def _generate_single_timeframe_signal(self, row):
        """Generate signal using single timeframe with multiple indicators"""
        all_buy_signals = []
        all_sell_signals = []
        all_reasons = []

        # Analyze all indicators
        ma_analysis = self._analyze_moving_averages(row)
        all_buy_signals.extend([s for s in ma_analysis["signals"] if s == "BUY"])
        all_sell_signals.extend([s for s in ma_analysis["signals"] if s == "SELL"])
        all_reasons.extend(ma_analysis["reasons"])

        rsi_analysis = self._analyze_rsi(row)
        all_buy_signals.extend([s for s in rsi_analysis["signals"] if s == "BUY"])
        all_sell_signals.extend([s for s in rsi_analysis["signals"] if s == "SELL"])
        all_reasons.extend(rsi_analysis["reasons"])

        bb_analysis = self._analyze_bollinger_bands(row)
        all_buy_signals.extend([s for s in bb_analysis["signals"] if s == "BUY"])
        all_sell_signals.extend([s for s in bb_analysis["signals"] if s == "SELL"])
        all_reasons.extend(bb_analysis["reasons"])

        atr_analysis = self._analyze_atr(row)
        all_buy_signals.extend([s for s in atr_analysis["signals"] if s == "BUY"])
        all_sell_signals.extend([s for s in atr_analysis["signals"] if s == "SELL"])
        all_reasons.extend(atr_analysis["reasons"])

        ichimoku_analysis = self._analyze_ichimoku(row)
        all_buy_signals.extend([s for s in ichimoku_analysis["signals"] if s == "BUY"])
        all_sell_signals.extend(
            [s for s in ichimoku_analysis["signals"] if s == "SELL"]
        )
        all_reasons.extend(ichimoku_analysis["reasons"])

        support_resistance_analysis = self._analyze_support_resistance(row)
        all_buy_signals.extend(
            [s for s in support_resistance_analysis["signals"] if s == "BUY"]
        )
        all_sell_signals.extend(
            [s for s in support_resistance_analysis["signals"] if s == "SELL"]
        )
        all_reasons.extend(support_resistance_analysis["reasons"])

        # Count signals
        buy_count = len(all_buy_signals)
        sell_count = len(all_sell_signals)

        # Calculate signal strength (percentage of indicators agreeing)
        total_indicators = buy_count + sell_count
        if total_indicators == 0:
            return "HOLD"

        buy_strength = buy_count / total_indicators if total_indicators > 0 else 0
        sell_strength = sell_count / total_indicators if total_indicators > 0 else 0

        # Decision logic - require minimum agreement
        min_signals_needed = 2  # At least 2 indicators must agree

        if buy_count >= min_signals_needed and buy_strength >= 0.6:  # 60% agreement
            signal = "BUY"
            print("\n=== BUY SIGNAL ===")
            print(f"Buy signals: {buy_count}, Sell signals: {sell_count}")
            print(f"Buy strength: {buy_strength:.2f}")
            print(f"Reasons: {', '.join(all_reasons[:5])}")
        elif sell_count >= min_signals_needed and sell_strength >= 0.6:
            signal = "SELL"
            print("\n=== SELL SIGNAL ===")
            print(f"Buy signals: {buy_count}, Sell signals: {sell_count}")
            print(f"Sell strength: {sell_strength:.2f}")
            print(f"Reasons: {', '.join(all_reasons[:5])}")
        else:
            signal = "HOLD"

        return signal

    def _fallback_signal(self, row):
        """Fallback signal generation for single timeframe"""
        # Simple moving average crossover as fallback
        fast_ma_col = f"fast_ma_{self.params['fast_ma_windows'][0]}"
        slow_ma_col = f"slow_ma_{self.params['slow_ma_windows'][0]}"

        if (
            fast_ma_col in row
            and slow_ma_col in row
            and pd.notna(row[fast_ma_col])
            and pd.notna(row[slow_ma_col])
        ):
            if row[fast_ma_col] > row[slow_ma_col]:
                return "BUY"
            elif row[fast_ma_col] < row[slow_ma_col]:
                return "SELL"

        return "HOLD"
