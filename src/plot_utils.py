import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import calendar
import plotly.graph_objects as go
import plotly.express as px


def plot_heatmap(df: pd.DataFrame, 
                 figsize: tuple = (12, 8),
                 colour_min_max: tuple = (-10, 10),
                 cbar_fixed_height_in: float = 6.0):
    """
    Plots a heatmap with a colorbar of fixed physical height.

    Parameters:
      df: DataFrame to plot.
      figsize: Figure size in inches.
      colour_min_max: Tuple with (min, max) color limits.
      cbar_fixed_height_in: colourbar legend height in inches.
    """
    # Create the figure and main axes.
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Calculate the normalized colourbar height.
    # The figure's height in inches is figsize[1].
    normalized_cbar_height = cbar_fixed_height_in / figsize[1]
    
    # Position the colorbar (relative positioning):
    cbar_left = 0.92
    cbar_top = 0.85
    cbar_ax = fig.add_axes([cbar_left, cbar_top - normalized_cbar_height, 0.02, normalized_cbar_height])
    
    # Plot heatmap
    hm = sns.heatmap(df,
                     annot=True,
                     fmt=".2f",
                     cmap='RdYlGn',
                     center=0,
                     vmin=colour_min_max[0],
                     vmax=colour_min_max[1],
                     linewidths=0.5,
                     linecolor='gray',
                     ax=ax,
                     cbar_ax=cbar_ax,
                     annot_kws={"fontsize": 8})
    
    # Formatting
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_title("Dividend Harvesting % Returns", fontsize=16)
    ax.set_ylabel("Stonk", fontsize=14)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    # Add % to colourbar
    cbar_ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x:.1f}%'))
    plt.show()


def plot_yearly_boxplot(data:pd.DataFrame, 
                        ylim:tuple=(-5, 15), 
                        bssr:float=0.5, 
                        trading_fee:float=0, 
                        investment:float=1000):
    
    # calculate the number of trades
    n_trades = data.shape[0]

    # plot
    plt.figure(figsize=(12, 5))
    sns.boxplot(x='year', y='pct_return', data=data)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)

    # formatting
    plt.xlabel("Year")
    plt.ylabel("% Profit")
    plt.title(f"Return (%) of individual trades, grouped by year")
    plt.xticks(rotation=0)
    plt.ylim(ylim)

    # Add config text
    config_text = f"BSSR: {bssr}\nTransaction Fee: ${trading_fee}\nInvestment: ${investment}\n# Trades: {n_trades}"
    plt.gca().text(
        0.99, 0.975, config_text, 
        transform=plt.gca().transAxes, 
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0, edgecolor='black')
    )
    plt.tight_layout()
    plt.show()


def plot_trade_scatter(data: pd.DataFrame, 
                       ylim: tuple = (-5, 15), 
                       bssr: float = 0.5, 
                       trading_fee: float = 0, 
                       investment: float = 1000,
                       marker_size: int = 30,
                       alpha: float = 1):
    """
    Plots a scatter chart of individual trade percentage returns over time.

    The x-axis represents the ex-dividend date of each trade (with one tick per year),
    and the y-axis represents the percentage return of that trade. Additionally, configuration
    details such as the BSSR, transaction fee, investment amount, and the number of trades 
    are displayed in the top-right corner of the plot.

    Parameters:
        data (pd.DataFrame): DataFrame containing trade data. Expected to have the following columns:
                             - 'ex_div_date': Date of the ex-dividend (should be datetime type).
                             - 'pct_return': Percentage return of the trade.
        ylim (tuple): Y-axis limits for the plot (default is (-5, 15)).
        bssr (float): Buy Sell Success Rate (between 0 and 1) used in simulation (default 0.5).
        trading_fee (float): Transaction fee applied on both buy and sell (default 0).
        investment (float): Investment amount per trade (default 1000).
        marker_size (int): scatter plot marker size
        alpha (float): marker transparancy, takes values between 0 and 1.

    Returns:
        None. Displays the plot.
    """
    # Calculate the number of trades
    n_trades = data.shape[0]

    plt.figure(figsize=(12, 5))
    plt.scatter(data['ex_div_date'], data['pct_return'], alpha=alpha, s=marker_size)
    plt.xlabel("Ex-Dividend Date")
    plt.ylabel("% Return")
    plt.title("Return (%) of individual trades")
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.ylim(ylim)

    # Set x-axis ticks to be at each year using Matplotlib's date locators/formatters
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Add configuration text in the top-right corner
    config_text = (f"BSSR: {bssr}\n"
                   f"Transaction Fee: ${trading_fee}\n"
                   f"Investment: ${investment}\n"
                   f"# Trades: {n_trades}")
    plt.gca().text(
        0.99, 0.975, config_text, 
        transform=plt.gca().transAxes, 
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0, edgecolor='black')
    )

    plt.tight_layout()
    plt.show()



def plot_vintage_analysis_old(trade_df: pd.DataFrame, 
                                initial_investment: float = 3000, 
                                min_year: int = None):
    """
    Creates a vintage analysis plot with daily granularity for the dividend harvesting strategy.

    For each year (vintage), this function simulates the running total of capital,
    starting at the specified initial investment (default $3000). The total capital
    is updated at each dividend event (using the 'profit' from that trade), and the 
    running total is recorded on a daily basis (by forward filling between trade dates).
    The x-axis shows days (with month abbreviations) and each vintage (year) is plotted
    as a separate line.

    Optionally, if a minimum year is provided, only trades from years greater than or equal
    to min_year are included.

    Parameters:
        trade_df (pd.DataFrame): DataFrame containing individual trade data with at least:
            - 'ex_div_date': The datetime when the dividend event occurred.
            - 'profit': The profit (or loss) from that trade.
            - 'year': (Optional) The year of the dividend event; if missing, it will be extracted 
                      from 'ex_div_date'.
        initial_investment (float): The starting capital for each year (default is 3000).
        min_year (int, optional): Only include trades for years >= min_year.
    
    Returns:
        None. Displays the vintage analysis plot.
    """
    # Ensure that ex_div_date is a datetime type.
    trade_df['ex_div_date'] = pd.to_datetime(trade_df['ex_div_date'])
    
    # Create a 'year' column if it doesn't exist.
    if 'year' not in trade_df.columns:
        trade_df['year'] = trade_df['ex_div_date'].dt.year

    # Filter by min_year if provided.
    if min_year is not None:
        trade_df = trade_df[trade_df['year'] >= min_year]
    
    # Group the trades by year.
    grouped = trade_df.groupby('year')
    
    plt.figure(figsize=(12, 8))
    
    for year, group in grouped:
        # Sort the group's trades by date.
        group = group.sort_values(by='ex_div_date')
        
        # Create a daily date range for the entire year.
        daily_index = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        # Start with the initial investment on January 1st.
        event_dates = [pd.Timestamp(f"{year}-01-01")]
        cumulative_capital = [initial_investment]
        current_capital = initial_investment
        
        # Update the running total at each trade event.
        for _, row in group.iterrows():
            trade_date = row['ex_div_date'].normalize()  # remove the time component
            current_capital += row['profit']
            event_dates.append(trade_date)
            cumulative_capital.append(current_capital)
        
        # Create a Series of the event dates and cumulative capital.
        s = pd.Series(data=cumulative_capital, index=event_dates)
        # Group by date (in case there are duplicates) and take the last value for each day.
        s = s.groupby(s.index).last()
        # Reindex the series to include every day of the year and forward-fill missing values.
        daily_capital = s.reindex(daily_index, method='ffill')
        
        # Plot the daily running total for this year.
        plt.plot(daily_index, daily_capital, label=str(year))
    
    # Format the x-axis to show month abbreviations.
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xticks(rotation=90)
    plt.xlabel("Month")
    plt.ylabel("Total Capital")
    plt.title("Vintage Analysis: Daily Running Total Capital by Year")
    plt.legend(title="Year")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_vintage_hold_strategy(data: pd.DataFrame, 
                               initial_investment: float = 2000, 
                               min_year: int = None,
                               max_year: int = None,
                               dummy_year: int = 2000):
    """
    Creates an interactive vintage analysis plot for a passive hold strategy using Plotly Express.
    
    For each year in the provided dataframe, the function simulates investing a fixed amount
    (default $2000) on the first trading day of that year. It calculates the number of shares purchased
    based on the first day's closing price, then computes the daily portfolio value as the product of the 
    number of shares and the daily closing price. The daily values are reindexed over the full year and 
    forward-filled. Finally, the dates are rebased to a common dummy year (default 2000) so that all vintages 
    begin on January 1, allowing for direct intra-year comparisons.
    
    Parameters:
        data (pd.DataFrame): DataFrame with daily index data. It should have a 'Date' column
                                         (or a datetime index) and a 'Close' column.
        initial_investment (float): The amount invested on the first trading day of each year (default $2000).
        min_year (int, optional): Only include years greater than or equal to this value.
        max_year (int, optional): Only include years less than or equal to this value.
        dummy_year (int): The year to which all dates are rebased for plotting (default is 2000).
    
    Returns:
        None. Displays an interactive Plotly Express plot.
    """
    # Work on a copy.
    df = data.copy()
    
    # If a 'Date' column exists, convert to datetime and set as index.
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
    
    # Create a 'year' column from the index.
    df['year'] = df.index.year
    if min_year is not None:
        df = df[df['year'] >= min_year]
    if max_year is not None:
        df = df[df['year'] <= max_year]
    
    all_data = []
    
    # Group by year.
    grouped = df.groupby('year')
    
    for year, group in grouped:
        group = group.sort_index()
        if group.empty:
            continue
        
        # Use the first trading day's close price.
        first_day = group.iloc[0]
        first_close = first_day['Close']
        shares = initial_investment / first_close
        
        # Create a full daily date range for the current year.
        daily_index = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        # Compute daily portfolio value: shares * daily Close.
        # Reindex the group's 'Close' series to the full daily date range, using forward-fill.
        s = group['Close'] * shares
        daily_value = s.reindex(daily_index, method='ffill')
        
        # Rebase the daily dates to the dummy year.
        dummy_index = daily_index.map(lambda d: d.replace(year=dummy_year))
        
        # Create a temporary DataFrame for this vintage.
        temp_df = pd.DataFrame({
            "dummy_date": dummy_index,
            "portfolio_value": daily_value.values,
            "year": year
        })
        all_data.append(temp_df)
    
    # Concatenate all vintage data.
    final_df = pd.concat(all_data, axis=0)
    
    # Create a Plotly Express line plot.
    fig = px.line(final_df, 
                  x="dummy_date", 
                  y="portfolio_value", 
                  color="year",
                  title="Vintage Analysis: Passive Hold Strategy Performance by Year",
                  labels={"dummy_date": "Month", "portfolio_value": "Portfolio Value ($)", "year": "Year"},
                  width=1200,
                  height=600,
                  template="plotly_white")
    
    # Update x-axis ticks to display month abbreviations.
    fig.update_xaxes(tickformat="%b", dtick="M1")

    # add horizontal reference line
    fig.add_shape(
        type="line",
        x0=final_df['dummy_date'].min(),
        x1=final_df['dummy_date'].max(),
        y0=initial_investment,
        y1=initial_investment,
        line=dict(color="red", dash="dash"),
        xref="x",
        yref="y"
    )
    
    fig.show()


import pandas as pd
import plotly.express as px

def plot_vintage_ratio(trade_df: pd.DataFrame,
                       hold_strategy_df: pd.DataFrame,
                       initial_investment: float,
                       min_year: int = None,
                       max_year: int = None,
                       dummy_year: int = 2000):
    """
    Creates an interactive vintage analysis plot using Plotly Express comparing the ratio of
    a dividend strategy to a passive hold strategy.

    For the dividend strategy (from trade_df), the function simulates a running total portfolio
    value by starting at initial_investment on January 1 and then updating that value with the 
    'profit' from each trade (based on 'ex_div_date'). For the hold strategy (from hold_strategy_df),
    the function simulates investing initial_investment_hold on the first trading day of the year by 
    computing the number of shares bought at that day's 'Close' and then calculating the daily 
    portfolio value (shares × Close).

    The daily ratio is computed as:
        ratio = (Dividend Strategy Portfolio Value) / (Hold Strategy Portfolio Value)

    For each common year, the daily series is reindexed to include every day in the year, forward-filled,
    and the dates are rebased to a dummy year (default 2000) so that all lines begin at January 1, allowing 
    for intra-year comparisons.

    Parameters:
        trade_df (pd.DataFrame): DataFrame containing dividend strategy trade data with at least:
            - 'ex_div_date': The datetime when the dividend event occurred.
            - 'profit': The profit (or loss) from that trade.
            Optionally, a 'year' column; if missing, it will be created.
        hold_strategy_df (pd.DataFrame): DataFrame containing daily hold strategy data. It should have either:
            - a 'Date' column (which will be converted to datetime and set as index), or
            - a datetime index, and must include a 'Close' column.
        initial_investment (float): The starting portfolio value
        min_year (int, optional): Only include years >= min_year.
        max_year (int, optional): Only include years <= max_year.
        dummy_year (int): The year to which all dates are rebased for plotting (default 2000).

    Returns:
        None. Displays an interactive Plotly Express plot of the ratio by year.
    """
    # Process dividend strategy data.
    div_df = trade_df.copy()
    div_df['ex_div_date'] = pd.to_datetime(div_df['ex_div_date'])
    if 'year' not in div_df.columns:
        div_df['year'] = div_df['ex_div_date'].dt.year
    if min_year is not None:
        div_df = div_df[div_df['year'] >= min_year]
    if max_year is not None:
        div_df = div_df[div_df['year'] <= max_year]
    
    # Process hold strategy data.
    hold_df = hold_strategy_df.copy()
    if 'Date' in hold_df.columns:
        hold_df['Date'] = pd.to_datetime(hold_df['Date'])
        hold_df.set_index('Date', inplace=True)
    else:
        if not pd.api.types.is_datetime64_any_dtype(hold_df.index):
            hold_df.index = pd.to_datetime(hold_df.index)
    hold_df['year'] = hold_df.index.year
    if min_year is not None:
        hold_df = hold_df[hold_df['year'] >= min_year]
    if max_year is not None:
        hold_df = hold_df[hold_df['year'] <= max_year]
    
    # Prepare dictionaries to hold daily portfolio values for each strategy.
    div_series_dict = {}
    hold_series_dict = {}
    
    # Process dividend strategy per vintage.
    for year, group in div_df.groupby('year'):
        group = group.sort_values(by='ex_div_date')
        daily_index = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        event_dates = [pd.Timestamp(f"{year}-01-01")]
        portfolio_value_div = [initial_investment]
        current_value_div = initial_investment
        
        for _, row in group.iterrows():
            trade_date = row['ex_div_date'].normalize()  # drop time component
            current_value_div += row['profit']
            event_dates.append(trade_date)
            portfolio_value_div.append(current_value_div)
        
        s_div = pd.Series(portfolio_value_div, index=event_dates).groupby(level=0).last()
        daily_value_div = s_div.reindex(daily_index, method='ffill')
        div_series_dict[year] = daily_value_div
    
    # Process hold strategy per vintage.
    for year, group in hold_df.groupby('year'):
        group = group.sort_index()
        daily_index = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        if group.empty:
            continue
        first_day = group.iloc[0]
        first_close = first_day['Close']
        shares = initial_investment / first_close
        s_hold = group['Close'] * shares
        daily_value_hold = s_hold.reindex(daily_index, method='ffill')
        hold_series_dict[year] = daily_value_hold
    
    # Compute the daily ratio for each year that is common to both strategies.
    all_data = []
    common_years = sorted(set(div_series_dict.keys()).intersection(hold_series_dict.keys()))
    for year in common_years:
        daily_index = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        div_series = div_series_dict[year].reindex(daily_index, method='ffill')
        hold_series = hold_series_dict[year].reindex(daily_index, method='ffill')
        ratio = div_series / hold_series
        
        # Rebase the dates to the dummy year.
        dummy_index = daily_index.map(lambda d: d.replace(year=dummy_year))
        
        temp_df = pd.DataFrame({
            "dummy_date": dummy_index,
            "ratio": ratio.values,
            "year": year
        })
        all_data.append(temp_df)
    
    final_df = pd.concat(all_data, axis=0)
    
    # Create the Plotly Express line plot.
    fig = px.line(final_df, 
                  x="dummy_date", 
                  y="ratio", 
                  color="year",
                  title="Vintage Analysis Ratio: Dividend Strategy vs. Hold Strategy",
                  labels={"dummy_date": "Month", "ratio": "Portfolio Value Ratio", "year": "Year"},
                  width=1200,
                  height=600,
                  template="plotly_white")
    
    # Update x-axis to show month abbreviations.
    fig.update_xaxes(tickformat="%b", dtick="M1")

    # add horizontal reference line
    fig.add_shape(
        type="line",
        x0=final_df['dummy_date'].min(),
        x1=final_df['dummy_date'].max(),
        y0=1,
        y1=1,
        line=dict(color="red", dash="dash"),
        xref="x",
        yref="y"
    )

    
    fig.show()





def plot_vintage_analysis(trade_df: pd.DataFrame, 
                          initial_investment: float = 3000, 
                          min_year: int = None,
                          max_year: int = None,
                          dummy_year: int = 2000):
    """
    Creates an interactive vintage analysis plot with daily granularity for the dividend harvesting strategy using Plotly Express.
    Each year's running total portfolio value is simulated (starting at initial_investment) and the dates are rebased
    to a common dummy year (default 2000) so that all vintages begin at January 1. This allows direct intra-year comparisons.
    
    For each year (vintage), the function:
      - Starts with the initial investment on January 1.
      - Updates the running total at each dividend event (using the 'profit' from that trade).
      - Fills forward to provide a daily series.
      - Rebases the dates to the dummy year.
    
    Optionally, only trades for years >= min_year and <= max_year are included.
    
    Parameters:
        trade_df (pd.DataFrame): DataFrame containing trade data with at least:
            - 'ex_div_date': The datetime when the dividend event occurred.
            - 'profit': The profit (or loss) from that trade.
            - 'year': (Optional) The year of the dividend event; if missing, it will be extracted from 'ex_div_date'.
        initial_investment (float): The starting portfolio value for each year.
        min_year (int, optional): Only include trades for years >= min_year.
        max_year (int, optional): Only include trades for years <= max_year.
        dummy_year (int): The year to which all dates are rebased for plotting (default is 2000).
    
    Returns:
        None. Displays an interactive Plotly Express plot.
    """
    # Make a copy and ensure ex_div_date is datetime.
    df = trade_df.copy()
    df['ex_div_date'] = pd.to_datetime(df['ex_div_date'])
    
    # Create a 'year' column if it doesn't exist.
    if 'year' not in df.columns:
        df['year'] = df['ex_div_date'].dt.year
        
    # Filter by min_year and max_year if provided.
    if min_year is not None:
        df = df[df['year'] >= min_year]
    if max_year is not None:
        df = df[df['year'] <= max_year]
    
    # Prepare a list to collect daily data for each year.
    all_data = []
    # Group by year.
    grouped = df.groupby('year')
    
    for year, group in grouped:
        group = group.sort_values(by='ex_div_date')
        # Create a full daily date range for the current year.
        daily_index = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        # Start with the initial investment on January 1.
        event_dates = [pd.Timestamp(f"{year}-01-01")]
        cumulative_value = [initial_investment]
        current_value = initial_investment
        
        # Update running total portfolio value at each trade event.
        for _, row in group.iterrows():
            trade_date = row['ex_div_date'].normalize()  # drop time component
            current_value += row['profit']
            event_dates.append(trade_date)
            cumulative_value.append(current_value)
        
        # Create a Series from event_dates and cumulative_value.
        s = pd.Series(cumulative_value, index=event_dates)
        # For days with multiple trades, keep the last value.
        s = s.groupby(s.index).last()
        # Reindex the series to the full daily range and forward-fill missing days.
        daily_value = s.reindex(daily_index, method='ffill')
        
        # Rebase the daily_index to the dummy year.
        dummy_index = daily_index.map(lambda d: d.replace(year=dummy_year))
        
        # Create a temporary DataFrame with the rebased dates, portfolio values, and the vintage year.
        temp_df = pd.DataFrame({
            "dummy_date": dummy_index,
            "portfolio_value": daily_value.values,
            "year": year
        })
        all_data.append(temp_df)
    
    # Concatenate data for all years.
    final_df = pd.concat(all_data, axis=0)
    
    # Create a Plotly Express line plot.
    fig = px.line(final_df, 
                  x="dummy_date", 
                  y="portfolio_value", 
                  color="year",
                  title="Vintage Analysis: Daily Running Portfolio Value by Year",
                  labels={"dummy_date": "Month", "portfolio_value": "Portfolio Value ($)", "year": "Year"},
                  width=1200,
                  height=600,
                  template="plotly_white")
    
    # Update x-axis to display month abbreviations.
    fig.update_xaxes(tickformat="%b", dtick="M1")

    # add horizontal reference line
    fig.add_shape(
        type="line",
        x0=final_df['dummy_date'].min(),
        x1=final_df['dummy_date'].max(),
        y0=initial_investment,
        y1=initial_investment,
        line=dict(color="red", dash="dash"),
        xref="x",
        yref="y"
    )
    
    fig.show()


def plot_event_overlap_with_positions(trade_df: pd.DataFrame):
    """
    Creates an interactive Plotly chart showing:
      - A bar chart for the number of dividend events per day.
      - An overlaid line chart for the number of open positions on each day,
        calculated using the 'buy_date' and 'sell_date' columns.
    
    For each trade, it is assumed that the position is open from the buy_date through the sell_date (inclusive).
    The function counts dividend events by converting 'ex_div_date' to a date (dropping the time component)
    and grouping. Then it iterates over each trade and, for each day between buy_date and sell_date, increments
    a counter to compute the number of open positions.
    
    Parameters:
        trade_df (pd.DataFrame): DataFrame containing trade data. Must include:
            - 'ex_div_date': The datetime when the dividend event occurred.
            - 'buy_date': The date when the position was opened.
            - 'sell_date': The date when the position was closed.
    
    Returns:
        None. Displays an interactive Plotly chart.
    """
    # Make a copy and ensure the relevant columns are datetime.
    df = trade_df.copy()
    df['ex_div_date'] = pd.to_datetime(df['ex_div_date'])
    df['buy_date'] = pd.to_datetime(df['buy_date'])
    df['sell_date'] = pd.to_datetime(df['sell_date'])
    
    # Create a column with just the date (drop time) for ex_div_date.
    df['date_only'] = df['ex_div_date'].dt.date
    
    # Group by the date_only to count dividend events per day.
    event_counts = df.groupby('date_only').size().reset_index(name='count')
    event_counts['date_only'] = pd.to_datetime(event_counts['date_only'])
    
    # Compute open positions per day.
    open_positions = {}
    for _, row in df.iterrows():
        # Normalize buy_date and sell_date (drop time).
        start_date = row['buy_date'].normalize()
        end_date = row['sell_date'].normalize()
        for d in pd.date_range(start=start_date, end=end_date, freq='D'):
            open_positions[d] = open_positions.get(d, 0) + 1
            
    # Convert the open_positions dictionary to a DataFrame.
    open_positions_df = pd.DataFrame(list(open_positions.items()), columns=['date', 'open_positions'])
    open_positions_df = open_positions_df.sort_values(by='date')
    
    # Create a Plotly Express bar chart for dividend events.
    fig = px.bar(event_counts, x='date_only', y='count', 
                 title="Event Overlap Analysis: Dividend Events and Open Positions",
                 labels={"date_only": "Date", "count": "Number of Dividend Events"})
    
    # Add the open positions as an overlaid line.
    fig.add_trace(go.Scatter(
        x=open_positions_df['date'],
        y=open_positions_df['open_positions'],
        mode='lines',
        name="Open Positions",
        line=dict(color='grey', width=2)
    ))
    
    # Update the layout.
    fig.update_layout(
        template="plotly_white",
        width=1200,
        height=600,
        xaxis_title="Date",
        yaxis_title="Count"
    )
    
    # Optionally, format the x-axis ticks.
    fig.update_xaxes(tickformat="%Y-%m-%d")
    
    fig.show()

