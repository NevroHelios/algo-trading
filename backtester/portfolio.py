class Portfolio:
    def __init__(self, cash):
        self.initial_cash = cash
        self.cash = cash
        self.position = 0
        self.history = []
        self.buy_count = 0
        self.sell_count = 0

    def update(self, signal, row):
        """Update portfolio based on signal and return True if trade was executed"""
        price = row["Close"]
        trade_executed = False

        if signal == "BUY" and self.cash >= price:
            self.position += 1
            self.cash -= price
            self.history.append(("BUY", price, row.get("Index", len(self.history))))
            self.buy_count += 1
            trade_executed = True
        elif signal == "SELL" and self.position > 0:
            self.position -= 1
            self.cash += price
            self.history.append(("SELL", price, row.get("Index", len(self.history))))
            self.sell_count += 1
            trade_executed = True

        return trade_executed

    def get_current_value(self, current_price):
        """Get current portfolio value"""
        return self.cash + self.position * current_price

    def summary(self):
        """Print detailed portfolio summary"""
        if not self.history:
            print("No trades executed")
            return

        last_price = self.history[-1][1]
        current_value = self.get_current_value(last_price)
        total_return = current_value - self.initial_cash
        return_pct = (total_return / self.initial_cash) * 100

        print(f"\n=== PORTFOLIO SUMMARY ===")
        print(f"Initial cash: ₹{self.initial_cash:,.2f}")
        print(f"Final cash: ₹{self.cash:,.2f}")
        print(f"Final position: {self.position} shares")
        print(f"Portfolio value: ₹{current_value:,.2f}")
        print(f"Total return: ₹{total_return:,.2f} ({return_pct:+.2f}%)")
        print(
            f"Total trades: {len(self.history)} (Buy: {self.buy_count}, Sell: {self.sell_count})"
        )

        if len(self.history) > 1:
            print(f"\nTrade History (last 10):")
            for i, (action, price, index) in enumerate(self.history[-10:]):
                print(f"  {i + 1}: {action} at ₹{price:.2f}")

        # Calculate some basic metrics
        if self.buy_count > 0 and self.sell_count > 0:
            buy_prices = [price for action, price, _ in self.history if action == "BUY"]
            sell_prices = [
                price for action, price, _ in self.history if action == "SELL"
            ]

            if buy_prices and sell_prices:
                avg_buy = sum(buy_prices) / len(buy_prices)
                avg_sell = sum(sell_prices) / len(sell_prices)
                print(f"Average buy price: ₹{avg_buy:.2f}")
                print(f"Average sell price: ₹{avg_sell:.2f}")
                if avg_buy > 0:
                    avg_trade_return = ((avg_sell - avg_buy) / avg_buy) * 100
                    print(f"Average trade return: {avg_trade_return:+.2f}%")
