import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import os
import sys


# ---------------------------------------------------------------------------
# Pair configuration — must match the notebook
# TARGET  : TXN  (Texas Instruments)
# PAIR    : A    (Agilent Technologies) — cointegrated partner found in notebook
# Features: Zscore, Zscore_Lag1, Zscore_MA5,
#           Spread, Spread_Lag1, Spread_Lag2, Spread_Change,
#           Return_TXN, Return_Pair, Vol_Ratio
# ---------------------------------------------------------------------------

TARGET     = 'TXN'
PAIR_STOCK = 'A'

FEATURE_COLS = [
    'Zscore',
    'Zscore_Lag1',
    'Zscore_MA5',
    'Spread',
    'Spread_Lag1',
    'Spread_Lag2',
    'Spread_Change',
    'Return_TXN',
    'Return_Pair',
    'Vol_Ratio',
]


def extract_features_pair(
    target: str = TARGET,
    pair_stock: str = PAIR_STOCK,
    lookback_days: int = 365,
    zscore_window: int = 30,
    vol_window: int = 10,
) -> pd.DataFrame:
    """
    Download daily adjusted-close prices for *target* and *pair_stock*,
    compute the spread-based features used in the TXN pair-trading notebook,
    and return a tidy DataFrame whose columns match FEATURE_COLS exactly.

    Parameters
    ----------
    target        : primary ticker  (default 'TXN')
    pair_stock    : paired ticker   (default 'A')
    lookback_days : calendar days of history to download
    zscore_window : rolling window (days) for spread mean/std → z-score
    vol_window    : rolling window (days) for return volatility ratio

    Returns
    -------
    pd.DataFrame  shape (n_rows, 10), columns == FEATURE_COLS
                  index reset to integers, NaN rows dropped
    """
    START_DATE = (
        datetime.date.today() - datetime.timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")

    tickers = [target, pair_stock]
    raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=False)

    # ── Extract adjusted-close prices ────────────────────────────────────────
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Adj Close"][[target, pair_stock]].copy()
    else:
        # Single-ticker fallback (shouldn't happen with two tickers)
        prices = raw[["Adj Close"]].copy()
        prices.columns = tickers

    prices.dropna(inplace=True)
    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)

    # ── Spread & z-score ─────────────────────────────────────────────────────
    prices["Spread"]       = prices[target] - prices[pair_stock]
    prices["Spread_Mean"]  = prices["Spread"].rolling(zscore_window).mean()
    prices["Spread_Std"]   = prices["Spread"].rolling(zscore_window).std()
    prices["Zscore"]       = (
        (prices["Spread"] - prices["Spread_Mean"]) / prices["Spread_Std"]
    )

    # ── Lag / change features ────────────────────────────────────────────────
    prices["Zscore_Lag1"]    = prices["Zscore"].shift(1)
    prices["Zscore_MA5"]     = prices["Zscore"].rolling(5).mean()
    prices["Spread_Lag1"]    = prices["Spread"].shift(1)
    prices["Spread_Lag2"]    = prices["Spread"].shift(2)
    prices["Spread_Change"]  = prices["Spread"].diff()

    # ── Return features ───────────────────────────────────────────────────────
    prices["Return_TXN"]  = prices[target].pct_change()
    prices["Return_Pair"] = prices[pair_stock].pct_change()

    # ── Volatility ratio ──────────────────────────────────────────────────────
    prices["Vol_Ratio"] = (
        prices["Return_TXN"].rolling(vol_window).std()
        / prices["Return_Pair"].rolling(vol_window).std()
    )

    # ── Select, clean, reset index ────────────────────────────────────────────
    features = prices[FEATURE_COLS].dropna().reset_index(drop=True)

    return features
