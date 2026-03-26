import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
import os
import sys


def extract_features():

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


def extract_features_pair(partner_ticker='EXPD', target_ticker='AOS'):
    """
    Download price data for the cointegrated stock pair identified in the notebook.

    The notebook identifies a target stock (AOS) and its best cointegrated
    partner (EXPD). This function downloads their Adjusted Close prices and
    returns a DataFrame with two columns: [partner_ticker, target_ticker].

    Parameters
    ----------
    partner_ticker : str
        The partner/spread stock symbol (default 'EXPD').
    target_ticker : str
        The target stock symbol (default 'AOS').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [partner_ticker, target_ticker], indexed 0..N,
        containing Adjusted Close prices for the past 365 days.
    """
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")

    tickers = [partner_ticker, target_ticker]
    stk_data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=False)

    # Extract Adjusted Close for each ticker
    partner_prices = stk_data.loc[:, ('Adj Close', partner_ticker)]
    partner_prices.name = partner_ticker

    target_prices = stk_data.loc[:, ('Adj Close', target_ticker)]
    target_prices.name = target_ticker

    dataset = pd.concat([partner_prices, target_prices], axis=1).dropna()
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
