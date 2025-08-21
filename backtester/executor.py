import requests


class Executor:
    def __init__(self, config):
        self.config = config
        self.live = config["data_source"] == "alpaca"

        if self.live:
            self.key = config["alpaca"]["key_id"]
            self.secret = config["alpaca"]["secret_key"]
            # Allow base_url to be provided with or without trailing /v2
            base_url = config["alpaca"]["base_url"].rstrip("/")
            self.api_v2_base = (
                base_url if base_url.endswith("/v2") else f"{base_url}/v2"
            )
            self.headers = {
                "APCA-API-KEY-ID": self.key,
                "APCA-API-SECRET-KEY": self.secret,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

    def execute(self, signal, price, ticker):
        if not self.live:
            return  # in backtest, Portfolio handles trades

        side = "buy" if signal == "BUY" else "sell" if signal == "SELL" else None
        if side:
            order = {
                "symbol": ticker,
                "qty": 1,
                "side": side,
                "type": "market",
                "time_in_force": "gtc",
            }
            url = f"{self.api_v2_base}/orders"
            try:
                resp = requests.post(url, json=order, headers=self.headers, timeout=30)
                resp.raise_for_status()
                try:
                    print("ORDER RESPONSE:", resp.json())
                except ValueError:
                    # Non-JSON response (e.g., HTML error page) — print raw text for diagnostics
                    print(
                        "ORDER RESPONSE (non-JSON):", resp.status_code, resp.text[:500]
                    )
            except requests.HTTPError as e:
                # Print useful diagnostics instead of crashing
                r = getattr(e, "response", None)
                status = r.status_code if r is not None else "?"
                body = None
                if r is not None:
                    try:
                        body = r.json()
                    except Exception:
                        body = r.text
                print("ORDER ERROR:", status, body)
            except requests.RequestException as e:
                print("ORDER REQUEST FAILED:", str(e))
