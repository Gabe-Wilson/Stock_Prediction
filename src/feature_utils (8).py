import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
import os
import sys


def extract_features():
    """Original feature extraction for AOS/EXPD regression model (unchanged)."""
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
    Download the last 365 days of price data for the cointegrated pair
    used in the TXN pairs trading model: valid_partner and TXN.

    The notebook trains on X = data_prediction[[valid_partner, target_ticker]]
    where target_ticker = 'TXN'. The first column is the cointegrated partner.
    Column order must match exactly what the pipeline was fitted on:
      column 0 = valid_partner (e.g. the stock cointegrated with TXN)
                 — stored at runtime in model_metadata.json
      column 1 = TXN

    The Streamlit app reads model_metadata.json to get the exact column names,
    so this function just returns the raw Adj Close prices in that same order.
    """
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")

    # TXN = target ticker (Texas Instruments)
    # NVDA is used as the placeholder partner here; the app overwrites column
    # names from model_metadata.json so the actual partner ticker is flexible.
    stk_tickers = ['TXN', 'NVDA']

    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)

    # Pull Adj Close for both tickers
    # Column order: [partner, TXN] — matching X_train column order from notebook
    partner_prices = stk_data.loc[:, ('Adj Close', 'NVDA')]
    txn_prices = stk_data.loc[:, ('Adj Close', 'TXN')]

    dataset = pd.concat([partner_prices, txn_prices], axis=1)
    dataset.columns = ['valid_partner', 'TXN']  # placeholder names; app renames from metadata
    dataset = dataset.dropna()
    dataset.index.name = 'Date'
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
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
