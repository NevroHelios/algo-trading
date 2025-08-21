"""
Enhanced Portfolio class with real-life trading costs and tax calculations
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


class Trade:
    """Represents a single trade with all relevant information"""

    def __init__(
        self,
        action: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        fees: float = 0.0,
    ):
        self.action = action  # 'BUY' or 'SELL'
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.fees = fees
        self.gross_amount = quantity * price
        self.net_amount = (
            self.gross_amount + fees if action == "BUY" else self.gross_amount - fees
        )

    def __repr__(self):
        return f"Trade({self.action}, {self.quantity}@₹{self.price:.2f}, fees=₹{self.fees:.2f})"


class TaxLot:
    """Represents a tax lot for tracking cost basis and holding periods"""

    def __init__(self, quantity: int, cost_basis: float, purchase_date: datetime):
        self.quantity = quantity
        self.cost_basis = cost_basis  # Per share cost basis including fees
        self.purchase_date = purchase_date

    def is_long_term(self, sale_date: datetime, holding_period_days: int = 365) -> bool:
        """Check if this lot qualifies for long-term capital gains"""
        return (sale_date - self.purchase_date).days >= holding_period_days


class EnhancedPortfolio:
    """Enhanced portfolio with realistic trading costs and tax calculations"""

    def __init__(self, initial_cash: float, config: Dict[str, Any]):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.trades: List[Trade] = []
        self.tax_lots: List[TaxLot] = []

        # Trading costs configuration
        self.trading_costs = config.get("trading_costs", {})
        self.commission_per_trade = self.trading_costs.get("commission_per_trade", 0.0)
        self.commission_rate = self.trading_costs.get("commission_rate", 0.0)
        self.bid_ask_spread = self.trading_costs.get("bid_ask_spread", 0.0)
        self.slippage = self.trading_costs.get("slippage", 0.0)
        self.margin_rate = self.trading_costs.get("margin_rate", 0.0)

        # Tax configuration
        self.tax_config = config.get("tax_config", {})
        self.short_term_rate = self.tax_config.get(
            "short_term_rate", 0.15
        )  # 15% for India STCG
        self.long_term_rate = self.tax_config.get(
            "long_term_rate", 0.10
        )  # 10% for India LTCG
        self.ltcg_exemption_limit = self.tax_config.get(
            "ltcg_exemption_limit", 100000
        )  # ₹1 lakh exemption
        self.holding_period_days = self.tax_config.get("holding_period_days", 365)
        self.tax_loss_harvesting = self.tax_config.get("tax_loss_harvesting", False)

        # Performance tracking
        self.realized_gains = 0.0
        self.unrealized_gains = 0.0
        self.total_fees_paid = 0.0
        self.total_taxes_paid = 0.0
        self.short_term_gains = 0.0
        self.long_term_gains = 0.0

        # Statistics
        self.buy_count = 0
        self.sell_count = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def calculate_trading_costs(
        self, action: str, quantity: int, price: float
    ) -> float:
        """Calculate total trading costs for Indian equity delivery trading"""
        gross_amount = quantity * price

        # 1. Brokerage - Zero for equity delivery
        brokerage = self.commission_per_trade + (gross_amount * self.commission_rate)

        # 2. STT (Securities Transaction Tax)
        if action == "BUY":
            stt = gross_amount * self.trading_costs.get("stt_rate_buy", 0.001)
        else:  # SELL
            stt = gross_amount * self.trading_costs.get("stt_rate_sell", 0.001)

        # 3. Transaction charges (NSE: 0.00297%)
        transaction_charges = gross_amount * self.trading_costs.get(
            "transaction_charges", 0.0000297
        )

        # 4. SEBI charges (₹10 per crore)
        sebi_charges_per_crore = self.trading_costs.get("sebi_charges_per_crore", 10.0)
        sebi_charges = (
            gross_amount / 10000000
        ) * sebi_charges_per_crore  # Convert to crores

        # 5. Stamp charges (0.015% on buy side only)
        if action == "BUY":
            stamp_charges = gross_amount * self.trading_costs.get(
                "stamp_charges_buy", 0.00015
            )
        else:
            stamp_charges = 0.0

        # 6. GST (18% on brokerage + SEBI charges + transaction charges)
        taxable_amount = brokerage + sebi_charges + transaction_charges
        gst_rate = self.trading_costs.get("gst_rate", 0.18)
        gst = taxable_amount * gst_rate

        # 7. Market impact costs
        spread_cost = gross_amount * self.bid_ask_spread / 2
        slippage_cost = gross_amount * self.slippage / 2

        # Total regulatory costs
        regulatory_costs = (
            stt + transaction_charges + sebi_charges + stamp_charges + gst + brokerage
        )

        # Total costs including market impact
        total_costs = regulatory_costs + spread_cost + slippage_cost

        return total_costs

    def calculate_effective_price(self, action: str, market_price: float) -> float:
        """Calculate effective price after spread and slippage"""
        if action == "BUY":
            # Pay higher price when buying
            effective_price = market_price * (
                1 + self.bid_ask_spread / 2 + self.slippage / 2
            )
        else:  # SELL
            # Receive lower price when selling
            effective_price = market_price * (
                1 - self.bid_ask_spread / 2 - self.slippage / 2
            )

        return effective_price

    def update(self, signal: str, row: Dict[str, Any]) -> bool:
        """Update portfolio based on signal and return True if trade was executed"""
        if signal == "HOLD":
            return False

        market_price = row["Close"]
        current_time = datetime.now()  # In real implementation, use row timestamp

        trade_executed = False

        if signal == "BUY":
            trade_executed = self._execute_buy(market_price, current_time)
        elif signal == "SELL":
            trade_executed = self._execute_sell(market_price, current_time)

        return trade_executed

    def _execute_buy(self, market_price: float, timestamp: datetime) -> bool:
        """Execute buy order with all costs and tax lot tracking"""
        effective_price = self.calculate_effective_price("BUY", market_price)

        # Calculate maximum quantity we can afford
        estimated_costs = self.calculate_trading_costs("BUY", 1, effective_price)
        total_cost_per_share = effective_price + estimated_costs

        if self.cash < total_cost_per_share:
            return False  # Not enough cash

        # For simplicity, buy 1 share at a time
        quantity = 1
        actual_costs = self.calculate_trading_costs("BUY", quantity, effective_price)
        total_cost = (effective_price * quantity) + actual_costs

        if self.cash >= total_cost:
            # Execute trade
            self.cash -= total_cost
            self.position += quantity
            self.total_fees_paid += actual_costs
            self.buy_count += 1

            # Record trade
            trade = Trade("BUY", quantity, effective_price, timestamp, actual_costs)
            self.trades.append(trade)

            # Create tax lot
            cost_basis_per_share = (
                effective_price * quantity + actual_costs
            ) / quantity
            tax_lot = TaxLot(quantity, cost_basis_per_share, timestamp)
            self.tax_lots.append(tax_lot)

            return True

        return False

    def _execute_sell(self, market_price: float, timestamp: datetime) -> bool:
        """Execute sell order with FIFO tax lot accounting"""
        if self.position <= 0:
            return False

        effective_price = self.calculate_effective_price("SELL", market_price)
        quantity = 1  # Sell 1 share at a time

        if self.position < quantity:
            return False

        trading_costs = self.calculate_trading_costs("SELL", quantity, effective_price)
        gross_proceeds = effective_price * quantity
        net_proceeds = gross_proceeds - trading_costs

        # Execute trade
        self.cash += net_proceeds
        self.position -= quantity
        self.total_fees_paid += trading_costs
        self.sell_count += 1

        # Record trade
        trade = Trade("SELL", quantity, effective_price, timestamp, trading_costs)
        self.trades.append(trade)

        # Calculate gains/losses using FIFO
        self._calculate_realized_gains(
            quantity, effective_price, timestamp, trading_costs
        )

        return True

    def _calculate_realized_gains(
        self,
        quantity_sold: int,
        sale_price: float,
        sale_date: datetime,
        sale_costs: float,
    ):
        """Calculate realized gains using FIFO and apply appropriate tax rates"""
        remaining_to_sell = quantity_sold
        gross_proceeds = sale_price * quantity_sold
        net_proceeds = gross_proceeds - sale_costs
        total_cost_basis = 0.0

        # Use FIFO to determine which tax lots are sold
        lots_to_remove = []
        for i, lot in enumerate(self.tax_lots):
            if remaining_to_sell <= 0:
                break

            if lot.quantity <= remaining_to_sell:
                # Sell entire lot
                total_cost_basis += lot.quantity * lot.cost_basis
                remaining_to_sell -= lot.quantity
                lots_to_remove.append(i)
            else:
                # Partially sell lot
                total_cost_basis += remaining_to_sell * lot.cost_basis
                lot.quantity -= remaining_to_sell
                remaining_to_sell = 0

        # Remove fully sold lots
        for i in reversed(lots_to_remove):
            sold_lot = self.tax_lots.pop(i)

            # Calculate gain/loss for this lot
            lot_proceeds = (sold_lot.quantity / quantity_sold) * net_proceeds
            lot_cost_basis = sold_lot.quantity * sold_lot.cost_basis
            lot_gain = lot_proceeds - lot_cost_basis

            # Determine if long-term or short-term
            if sold_lot.is_long_term(sale_date, self.holding_period_days):
                self.long_term_gains += lot_gain
            else:
                self.short_term_gains += lot_gain

        # Total realized gain for this sale
        total_gain = net_proceeds - total_cost_basis
        self.realized_gains += total_gain

        if total_gain > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Calculate taxes owed
        self._calculate_taxes()

    def _calculate_taxes(self):
        """Calculate taxes owed on realized gains (India tax structure)"""
        # Short-term capital gains tax (15% in India)
        short_term_tax = max(0, self.short_term_gains * self.short_term_rate)

        # Long-term capital gains tax (10% in India, with ₹1 lakh exemption)
        taxable_ltcg = max(0, self.long_term_gains - self.ltcg_exemption_limit)
        long_term_tax = max(0, taxable_ltcg * self.long_term_rate)

        self.total_taxes_paid = short_term_tax + long_term_tax

    def get_current_value(self, current_price: float) -> float:
        """Get current portfolio value including unrealized gains"""
        position_value = self.position * current_price
        total_value = self.cash + position_value

        # Calculate unrealized gains
        if self.position > 0:
            current_cost_basis = sum(
                lot.quantity * lot.cost_basis for lot in self.tax_lots
            )
            self.unrealized_gains = position_value - current_cost_basis

        return total_value

    def get_after_tax_value(self, current_price: float) -> float:
        """Get portfolio value after accounting for potential taxes on unrealized gains"""
        current_value = self.get_current_value(current_price)

        # Estimate taxes on unrealized gains if positions were closed
        potential_tax = 0.0
        if self.unrealized_gains > 0:
            # Assume all unrealized gains would be short-term for conservative estimate
            potential_tax = self.unrealized_gains * self.short_term_rate

        return current_value - self.total_taxes_paid - potential_tax

    def summary(self):
        """Print comprehensive portfolio summary with tax and cost analysis"""
        if not self.trades:
            print("No trades executed")
            return

        last_price = self.trades[-1].price
        current_value = self.get_current_value(last_price)
        after_tax_value = self.get_after_tax_value(last_price)
        total_return = current_value - self.initial_cash
        after_tax_return = after_tax_value - self.initial_cash
        return_pct = (total_return / self.initial_cash) * 100
        after_tax_return_pct = (after_tax_return / self.initial_cash) * 100

        print(f"\n{'=' * 60}")
        print(f"ENHANCED PORTFOLIO SUMMARY")
        print(f"{'=' * 60}")

        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"Initial cash: ₹{self.initial_cash:,.2f}")
        print(f"Current cash: ₹{self.cash:,.2f}")
        print(f"Position: {self.position} shares")
        print(f"Portfolio value (before tax): ₹{current_value:,.2f}")
        print(f"Portfolio value (after tax): ₹{after_tax_value:,.2f}")
        print(f"Total return (before tax): ₹{total_return:,.2f} ({return_pct:+.2f}%)")
        print(
            f"Total return (after tax): ₹{after_tax_return:,.2f} ({after_tax_return_pct:+.2f}%)"
        )

        print(f"\n💰 GAINS/LOSSES:")
        print(f"Realized gains: ₹{self.realized_gains:,.2f}")
        print(f"Unrealized gains: ₹{self.unrealized_gains:,.2f}")
        print(f"Short-term gains: ₹{self.short_term_gains:,.2f}")
        print(f"Long-term gains: ₹{self.long_term_gains:,.2f}")

        print(f"\n💸 COSTS & TAXES:")
        print(f"Total fees paid: ₹{self.total_fees_paid:,.2f}")
        print(f"Total taxes paid: ₹{self.total_taxes_paid:,.2f}")
        print(f"Cost impact: ₹{self.total_fees_paid + self.total_taxes_paid:,.2f}")

        print(f"\n📈 TRADING STATISTICS:")
        print(
            f"Total trades: {len(self.trades)} (Buy: {self.buy_count}, Sell: {self.sell_count})"
        )
        print(f"Winning trades: {self.winning_trades}")
        print(f"Losing trades: {self.losing_trades}")

        if self.winning_trades + self.losing_trades > 0:
            win_rate = (
                self.winning_trades / (self.winning_trades + self.losing_trades) * 100
            )
            print(f"Win rate: {win_rate:.1f}%")

        # Cost analysis
        if len(self.trades) > 0:
            avg_cost_per_trade = self.total_fees_paid / len(self.trades)
            print(f"Average cost per trade: ₹{avg_cost_per_trade:.2f}")

        print(f"\n📊 TAX EFFICIENCY:")
        if self.realized_gains != 0:
            tax_efficiency = (
                1 - self.total_taxes_paid / abs(self.realized_gains)
            ) * 100
            print(f"Tax efficiency: {tax_efficiency:.1f}%")

        # Recent trades
        if len(self.trades) > 1:
            print(f"\n📋 RECENT TRADES (last 5):")
            for trade in self.trades[-5:]:
                print(f"  {trade}")

        # Current tax lots
        if self.tax_lots:
            print(f"\n📦 CURRENT TAX LOTS:")
            for i, lot in enumerate(self.tax_lots):
                days_held = (datetime.now() - lot.purchase_date).days
                status = (
                    "Long-term"
                    if days_held >= self.holding_period_days
                    else "Short-term"
                )
                print(
                    f"  Lot {i + 1}: {lot.quantity} shares @ ₹{lot.cost_basis:.2f} ({days_held} days, {status})"
                )

        print(f"\n{'=' * 60}")
