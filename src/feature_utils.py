import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import requests
import os
import sys

# pandas_datareader uses distutils internally, which was removed in Python 3.12.
# The import is moved inside extract_features() so it only loads when that
# function is actually called. extract_features_pair() (used by the Streamlit
# app) does not need it and will import cleanly.


def extract_features():
    """Original feature extraction for AOS/EXPD regression model."""
    # Local import — keeps the module importable on Python 3.12 even if
    # pandas_datareader is not installed or broken in the current environment.
    import pandas_datareader.data as web

    return_period = 5

    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    stk_tickers = ['AOS', 'EXPD']
    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']

    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)
    ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = web.DataReader(idx_tickers, 'fred', start=START_DATE, end=END_DATE)

    Y = np.log(stk_data.loc[:, ('Adj Close', 'AOS')]).diff(return_period).shift(-return_period)
    Y.name = Y.name[-1] + '_Future'

    X1 = np.log(stk_data.loc[:, ('Adj Close', ('AOS', 'EXPD'))]).diff(return_period)
    X1.columns = X1.columns.droplevel()
    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)

    X = pd.concat([X1, X2, X3], axis=1)

    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.columns]
    dataset.index.name = 'Date'
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    features = features.iloc[:, 1:]
    return features


def extract_features_pair():
    """
    Download the last 365 days of Adj Close prices for the TXN pairs
    trading model. Returns a DataFrame with two columns:
      [valid_partner, 'TXN']
    Column names are placeholder values — the Streamlit app renames them
    to the real ticker names loaded from model_metadata.json.
    """
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")

    # TXN = target ticker. NVDA is a placeholder for the cointegrated partner;
    # the app overwrites column names from model_metadata.json at runtime.
    stk_tickers = ['TXN', 'NVDA']

    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)

    partner_prices = stk_data.loc[:, ('Adj Close', 'NVDA')]
    txn_prices     = stk_data.loc[:, ('Adj Close', 'TXN')]

    dataset = pd.concat([partner_prices, txn_prices], axis=1)
    dataset.columns = ['valid_partner', 'TXN']
    dataset = dataset.dropna()
    dataset.index.name = 'Date'
    features = dataset.sort_index().reset_index(drop=True)
    return features


def get_bitcoin_historical_prices(days=60):
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')
    return df
