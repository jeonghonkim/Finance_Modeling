# install packages
#pip install yfinance
#pip install scikit-optimize
#pip install ta

# Importing the required libraries
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from skopt import gp_minimize
from skopt.space import Integer
from statsmodels.tsa.stattools import coint

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from ta.momentum import RSIIndicator

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import date, datetime

import warnings
warnings.filterwarnings("ignore", message="The objective has been evaluated at point")

# Define function to download stock data
def download_stock_data(stock_ticker, start_date):
    todays_date = date.today()
    stock_data = yf.download(stock_ticker, start=start_date, end=todays_date)
    return stock_data

# Define function to evaluate model (using Moving Averages)
def evaluate_model(ma_short, ma_long, stock_data):
    stock_data['MA_Short'] = stock_data['Close'].rolling(window=ma_short).mean()
    stock_data['MA_Long'] = stock_data['Close'].rolling(window=ma_long).mean()
    
    # Drop rows with NaN values after calculating the moving averages
    stock_data.dropna(subset=['MA_Short', 'MA_Long'], inplace=True)
    
    # Generate Buy and Sell Signals
    stock_data['Signal'] = np.where(stock_data['MA_Short'] > stock_data['MA_Long'], 1, -1)
    # Calculate Mean Absolute Error (MAE) based on signals for simplicity (customize as needed)
    mae = np.mean(np.abs(stock_data['Signal'].diff()))  # Example metric
    return mae, 0  # Return dummy RMSE (not calculated here)

# Define function for Bayesian Optimization
def objective(params, stock_data):
    ma_short, ma_long = params
    mae, _ = evaluate_model(ma_short, ma_long, stock_data)
    return mae  # We aim to minimize MAE

# Append new stock price to the data
def append_current_stock_price(stock_data, new_price, new_date):
    new_row = pd.DataFrame({'Open': [np.nan], 'High': [np.nan], 'Low': [np.nan], 'Close': [new_price],
                            'Adj Close': [np.nan], 'Volume': [np.nan]}, index=[new_date])
    stock_data = pd.concat([stock_data, new_row])
    return stock_data

# Define trading performance
def evaluate_trading_performance(ma_short, ma_long, stock_data):
    # Calculate the moving averages
    stock_data['MA_Short'] = stock_data['Close'].rolling(window=ma_short).mean()
    stock_data['MA_Long'] = stock_data['Close'].rolling(window=ma_long).mean()
    
    # Drop rows with NaN values
    stock_data.dropna(subset=['MA_Short', 'MA_Long'], inplace=True)
    
    # Generate Buy/Sell signals
    stock_data['Signal'] = np.where(stock_data['MA_Short'] > stock_data['MA_Long'], 1, -1)
    stock_data['Position'] = stock_data['Signal'].diff()
    
    # Simulate the trading performance
    initial_cash = 100000  # Starting capital
    shares = 0
    cash = initial_cash
    portfolio_value = []
    
    for i in range(len(stock_data)):
        if stock_data['Position'].iloc[i] == 2:  # Buy signal
           shares = cash // stock_data['Close'].iloc[i]
           cash -= shares * stock_data['Close'].iloc[i]
        
        elif stock_data['Position'].iloc[i] == -2:  # Sell signal
           cash += shares * stock_data['Close'].iloc[i]
           shares = 0
        
        # Track portfolio value (cash + shares value)
        portfolio_value.append(cash + shares * stock_data['Close'].iloc[i])
    
    # Calculate total return
    final_portfolio_value = portfolio_value[-1]
    total_return = final_portfolio_value - initial_cash
    
    # We return the negative of the total return since gp_minimize performs minimization
    return -total_return

# Define the Bayesian Optimization function
def perform_bayesian_optimization(stock_data, n_calls=50, random_state=42):
    # Define the search space
    search_space = [
        Integer(5, 25, name='ma_short'),  # Short-term moving average range
        Integer(50, 100, name='ma_long')  # Long-term moving average range
    ]

    # Perform Bayesian Optimization
    res = gp_minimize(lambda params: objective(params, stock_data.copy()), search_space, n_calls=n_calls, random_state=random_state)
   
    # Output the best parameters
    short_window = res.x[0]
    long_window = res.x[1]
    print(f"Best MA_Short: {short_window}, Best MA_Long: {long_window}, Best MAE: {res.fun}")
    return short_window, long_window, res

# Define the Bayesian Optimization function for returns
def perform_bayesian_optimization_for_returns(stock_data, n_calls=50, random_state=42):
    # Define the search space
    search_space = [
        Integer(5, 25, name='ma_short'),  # Short-term moving average range
        Integer(50, 100, name='ma_long')  # Long-term moving average range
    ]
   
    # Perform Bayesian Optimization
    res = gp_minimize(lambda params: evaluate_trading_performance(params[0], params[1], stock_data.copy()),
                      search_space, n_calls=n_calls, random_state=random_state)
   
    # Output the best parameters
    short_window = res.x[0]
    long_window = res.x[1]
    print(f"Best MA_Short: {short_window}, Best MA_Long: {long_window}, Best Total Return: {-res.fun}")
   
    return short_window, long_window, res

# Define find best performance windows
def find_best_performance_windows(stock_ticker, start_date):
    
    # Download stock data for any stock
    stock_data = download_stock_data(stock_ticker, start_date)
    
    # Perform Bayesian Optimization to maximize trading returns
    short_window, long_window, optimization_result = perform_bayesian_optimization_for_returns(stock_data, n_calls=50, random_state=42)
    
    return stock_data, short_window, long_window, optimization_result
