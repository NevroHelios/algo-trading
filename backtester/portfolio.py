class Portfolio:
    def __init__(self, cash):
        self.cash = cash
        self.position = 0
        self.history = []

    def update(self, signal, row):
        price = row["Close"]

        if signal == "BUY" and self.cash >= price:
            self.position += 1
            self.cash -= price
            self.history.append(("BUY", price))
        elif signal == "SELL" and self.position > 0:
            self.position -= 1
            self.cash += price
            self.history.append(("SELL", price))

    def summary(self):
        print("Final cash:", self.cash)
        print("Final position:", self.position)
        print("Portfolio value:", self.cash + self.position * self.history[-1][1] if self.history else self.cash)
