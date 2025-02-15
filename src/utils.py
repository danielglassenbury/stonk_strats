import os
import pandas as pd
import yfinance as yf

def download_and_cache_data(ticker_list: list, period: str, cache_filename: str):
    # Check if cached file exists
    if os.path.exists(cache_filename):
        print(f"Loading cached data from {cache_filename}...")
        df = pd.read_csv(cache_filename, index_col=0, parse_dates=True)
    else:
        # Download data for each ticker and combine them
        print("Downloading data from Yahoo Finance...")
        all_data = []
        for ticker in ticker_list:
            print(f"Downloading {ticker}...")
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period=period, actions=True)
            # Add a column for ticker
            hist_data['Ticker'] = ticker
            all_data.append(hist_data)
        df = pd.concat(all_data)
        # Save the combined DataFrame to CSV for later use
        df.to_csv(cache_filename)
        print(f"Data saved to {cache_filename}.")
    return df