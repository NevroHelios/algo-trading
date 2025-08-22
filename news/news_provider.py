from __future__ import annotations

import os
from typing import Any, Dict, Optional, List

import pandas as pd

try:
    # Optional dependency; we handle absence gracefully
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

    _VADER_AVAILABLE = True
except Exception:
    SentimentIntensityAnalyzer = None  # type: ignore
    _VADER_AVAILABLE = False


class NewsProvider:
    """Lightweight news loader + sentiment aggregator.

    Backtests load from a CSV file so runs are reproducible. Live fetch can be
    added later by extending this class (e.g., NewsAPI/Polygon/Finnhub) and
    caching into the same CSV.

    CSV schema (headers):
      - datetime (UTC ISO8601, e.g., 2025-08-20T09:15:00Z)
      - title
      - summary
      - source
      - ticker (optional; empty allowed)
    """

    def __init__(self, csv_path: str = "data/news.csv"):
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None
        self._analyzer = SentimentIntensityAnalyzer() if _VADER_AVAILABLE else None

    # ---------------------------- Data loading ---------------------------------
    def _ensure_loaded(self) -> None:
        if self._df is not None:
            return

        if not os.path.exists(self.csv_path):
            # Create empty dataframe if missing, keeps flow working
            self._df = pd.DataFrame(
                columns=["datetime", "title", "summary", "source", "ticker"]
            )
            return

        df = pd.read_csv(self.csv_path)
        if "datetime" not in df.columns:
            raise ValueError("news.csv must contain a 'datetime' column")

        # Parse datetimes; coerce invalid rows to NaT, drop them
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        for col in ["title", "summary", "source", "ticker"]:
            if col not in df.columns:
                df[col] = ""

        self._df = df.dropna(subset=["datetime"]).sort_values("datetime")

    # ---------------------------- Query window ---------------------------------
    def get_window(
        self, end_dt: pd.Timestamp, hrs: int = 24, ticker: Optional[str] = None
    ) -> pd.DataFrame:
        self._ensure_loaded()
        if self._df is None or self._df.empty:
            return pd.DataFrame(columns=["datetime", "title", "summary", "source", "ticker"])

        start_dt = end_dt - pd.Timedelta(hours=hrs)
        df = self._df
        mask = (df["datetime"] > start_dt) & (df["datetime"] <= end_dt)
        window = df.loc[mask]

        if ticker and "ticker" in window.columns and window["ticker"].notna().any():
            window = window[
                (window["ticker"].astype(str) == str(ticker))
                | (window["ticker"].astype(str).eq("") )
            ]

        return window

    # ---------------------------- Sentiment ------------------------------------
    def _score_text(self, text: str) -> float:
        if not text:
            return 0.0
        if self._analyzer is not None:
            s = self._analyzer.polarity_scores(text)
            return float(s.get("compound", 0.0))

        # Small fallback lexicon if VADER isn't installed
        positives = ["beat", "surge", "record", "growth", "upgrade", "bullish", "profit", "rally", "strong"]
        negatives = ["miss", "drop", "fraud", "downgrade", "bearish", "loss", "probe", "weak", "decline"]
        score = 0.0
        lt = text.lower()
        score += sum(term in lt for term in positives) * 0.2
        score -= sum(term in lt for term in negatives) * 0.2
        return max(min(score, 1.0), -1.0)

    def score_row(self, row: pd.Series) -> float:
        title = str(row.get("title", "") or "")
        summary = str(row.get("summary", "") or "")
        return self._score_text(f"{title} {summary}".strip())

    def aggregate_sentiment(
        self, end_dt: pd.Timestamp, window_hours: int = 24, ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        window = self.get_window(end_dt, window_hours, ticker)
        if window.empty:
            return {"avg": 0.0, "count": 0, "details": []}

        scores: List[float] = []
        details: List[Dict[str, Any]] = []
        for _, r in window.iterrows():
            sc = self.score_row(r)
            scores.append(sc)
            details.append({
                "dt": r["datetime"],
                "score": sc,
                "title": str(r.get("title", ""))[:160],
            })

        avg = float(pd.Series(scores).mean()) if scores else 0.0
        return {"avg": avg, "count": len(scores), "details": details}
