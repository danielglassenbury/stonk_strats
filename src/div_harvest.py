import yfinance as yf
import pandas as pd
import time

def cal_div_harvest(ticker_list: list, period: str, bssr: float = 0.5, data: 'pd.DataFrame' = None):
    """
    Simulate dividend harvesting trades over a specified period.
    
    Parameters:
        ticker_list (list): List of ticker symbols.
        period (str): The period over which to retrieve historical data (e.g., '20y').
        bssr (float): Buy Sell Success Rate, a value between 0 and 1.
                      1 means best luck (buy at low, sell at high),
                      0 means worst luck (buy at high, sell at low),
                      and 0.5 means neutral (buy/sell at midpoint).
        data (pd.DataFrame, optional): Preloaded historical data for all tickers.
                                       If provided, should include a 'Ticker' column.
    
    Returns:
        results_profit (pd.DataFrame): Aggregated profit per ticker by year.
        results_pct (pd.DataFrame): Aggregated percentage returns per ticker by year.
        results_trades (dict): Detailed trade information for each ticker.
    """
    import yfinance as yf
    import pandas as pd
    import time

    # Validate that bssr is between 0 and 1.
    assert 0 <= bssr <= 1, "bssr must be between 0 and 1."

    # Dictionaries to hold the yearly profit and percentage returns for each ticker
    results_yearly = {}
    results_yearly_pct = {}

    # Dictionary to hold the individual trade details for each ticker
    results_trades = {}

    # Loop over each ticker
    for ticker in ticker_list:
        print(f"Processing {ticker}...")
        try:
            # If preloaded data is provided, filter it for the current ticker.
            if data is not None:
                hist_data = data[data['Ticker'] == ticker].copy()
            else:
                stock = yf.Ticker(ticker)
                hist_data = stock.history(period=period, actions=True)

            # Skip if no data is returned
            if hist_data.empty:
                print(f"  No historical data for {ticker}.")
                continue

            # Ensure the index is datetime and tz-naive
            if not pd.api.types.is_datetime64_any_dtype(hist_data.index):
                hist_data.index = pd.to_datetime(hist_data.index)
            if hist_data.index.tz is not None:
                hist_data.index = hist_data.index.tz_convert('UTC').tz_localize(None)

            # Fill missing dividend values with 0
            hist_data['Dividends'] = hist_data['Dividends'].fillna(0)

            # Identify dividend events (rows where dividends were paid)
            dividend_events = hist_data[hist_data['Dividends'] > 0]
            if dividend_events.empty:
                print(f"  No dividend events for {ticker}.")
                continue

            # Dictionaries to accumulate yearly profit and percentage returns
            yearly_profits = {}
            yearly_pct = {}

            # List to accumulate individual trade details for this ticker
            trades_details = []

            # Flag to mark if a negative buy price is encountered for this ticker (data issue).
            bad_stock = False

            # Loop through each dividend event
            for ex_div_date in dividend_events.index:
                try:
                    idx = hist_data.index.get_loc(ex_div_date)
                except Exception as e:
                    print(f"  Could not locate index for {ex_div_date} in {ticker}: {e}")
                    continue

                # Need to have a prior day (to buy) and a following day (to sell)
                if idx == 0 or idx >= len(hist_data) - 1:
                    continue

                # Get the daily high and low for the buy day (day before ex-dividend)
                buy_date = hist_data.index[idx - 1]
                daily_high_buy = hist_data.loc[buy_date, 'High']
                daily_low_buy = hist_data.loc[buy_date, 'Low']

                # Calculate buy price based on bssr
                buy_price = (1 - bssr) * daily_high_buy + bssr * daily_low_buy

                # Check for negative buy price. If found, skip this ticker.
                if buy_price < 0:
                    print(f"  Negative buy price encountered for {ticker} on {buy_date} ({buy_price}). Removing ticker from results.")
                    bad_stock = True
                    break

                # Get the daily high and low for the sell day (day after ex-dividend)
                sell_date = hist_data.index[idx + 1]
                daily_high_sell = hist_data.loc[sell_date, 'High']
                daily_low_sell = hist_data.loc[sell_date, 'Low']

                # Calculate sell price based on bssr
                sell_price = bssr * daily_high_sell + (1 - bssr) * daily_low_sell

                # Dividend received on the ex-dividend date
                dividend = hist_data.loc[ex_div_date, 'Dividends']

                # Calculate profit and percentage return
                profit = (sell_price + dividend) - buy_price
                pct_return = profit / buy_price * 100

                # Use the ex-dividend date's year for grouping
                year = ex_div_date.year
                yearly_profits[year] = yearly_profits.get(year, 0.0) + profit
                yearly_pct[year] = yearly_pct.get(year, 0.0) + pct_return

                # Save the individual trade details for later investigation
                trades_details.append({
                    "ex_div_date": ex_div_date,
                    "buy_date": buy_date,
                    "sell_date": sell_date,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "dividend": dividend,
                    "profit": profit,
                    "pct_return": pct_return,
                    "year": year
                })

            # If a negative buy price was encountered, skip this ticker entirely.
            if bad_stock:
                continue

            # Store the aggregate results if any trades were found
            if yearly_profits:
                results_yearly[ticker] = yearly_profits
            else:
                print(f"  No valid trades for {ticker}.")

            if yearly_pct:
                results_yearly_pct[ticker] = yearly_pct

            # Store the individual trade details for this ticker
            results_trades[ticker] = trades_details

        except Exception as e:
            print(f"  Error processing {ticker}: {e}")

        # Pause to be kind to Yahoo Finance's servers if we're fetching live data.
        if data is None:
            time.sleep(1)

    results_pct = pd.DataFrame.from_dict(results_yearly_pct, orient='index').fillna(0)
    results_pct = results_pct[sorted(results_pct.columns, key=int)]
    results_profit = pd.DataFrame.from_dict(results_yearly, orient='index').fillna(0)
    results_profit = results_profit[sorted(results_profit.columns, key=int)]

    return results_profit, results_pct, results_trades
