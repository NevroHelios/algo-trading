class MyStrategy:
    def __init__(self, params):
        self.fast = params.get("fast_ma", 10)
        self.slow = params.get("slow_ma", 50)

    def generate_signal(self, row):
        if row["fast_ma"] > row["slow_ma"]:
            return "BUY"
        elif row["fast_ma"] < row["slow_ma"]:
            return "SELL"
        return "HOLD"
