import yfinance as yf
import pandas as pd
import time

def cal_div_harvest(ticker_list: list, period: str, data: pd.DataFrame = None):
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
            # It is assumed that the preloaded dataframe has a column named "Ticker".
            if data is not None:
                hist_data = data[data['Ticker'] == ticker].copy()
            else:
                print(f'Downloading {ticker} data from Yahoo Finance')
                stock = yf.Ticker(ticker)
                hist_data = stock.history(period=period, actions=True)

            # Ensure the index is datetime (if not already)
            if not pd.api.types.is_datetime64_any_dtype(hist_data.index):
                hist_data.index = pd.to_datetime(hist_data.index)

            # Convert tz-aware datetimes to tz-naive via UTC conversion
            if hist_data.index.tz is not None:
                hist_data.index = hist_data.index.tz_convert('UTC').tz_localize(None)


            # Skip if no data is returned
            if hist_data.empty:
                print(f"  No historical data for {ticker}.")
                continue

            # Ensure the index is datetime (if not already)
            if not pd.api.types.is_datetime64_any_dtype(hist_data.index):
                hist_data.index = pd.to_datetime(hist_data.index)

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

                # Simulate the trade:
                # Buy the day before the ex-dividend date
                buy_date = hist_data.index[idx - 1]
                # Use average of High and Low for the buy price
                buy_price = (hist_data.loc[buy_date, 'High'] + hist_data.loc[buy_date, 'Low']) / 2.0

                # Check for negative buy price. If found, skip this ticker.
                if buy_price < 0:
                    print(f"  Negative buy price encountered for {ticker} on {buy_date} ({buy_price}). Removing ticker from results.")
                    bad_stock = True
                    break

                # Sell the day after the ex-dividend date
                sell_date = hist_data.index[idx + 1]
                # Use average of High and Low for the sell price
                sell_price = (hist_data.loc[sell_date, 'High'] + hist_data.loc[sell_date, 'Low']) / 2.0
                # Dividend received on the ex-dividend date
                dividend = hist_data.loc[ex_div_date, 'Dividends']

                # Calculate profit and percentage return
                profit = (sell_price + dividend) - buy_price
                pct_return = profit / buy_price * 100

                # Use the ex-dividend date's year for grouping
                year = ex_div_date.year
                if year not in yearly_profits:
                    yearly_profits[year] = 0.0
                yearly_profits[year] += profit

                if year not in yearly_pct:
                    yearly_pct[year] = 0.0
                yearly_pct[year] += pct_return

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

        # Pause to be kind to Yahoo Finance's servers (only when downloading live data)
        if data is None:
            time.sleep(0.5)

    results_pct = pd.DataFrame.from_dict(results_yearly_pct, orient='index')
    results_pct = results_pct.fillna(0)
    results_pct = results_pct[sorted(results_pct.columns, key=int)]

    results_profit = pd.DataFrame.from_dict(results_yearly, orient='index')
    results_profit = results_profit.fillna(0)
    results_profit = results_profit[sorted(results_profit.columns, key=int)]

    return results_profit, results_pct, results_trades
