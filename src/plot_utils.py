import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import calendar


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
    cbar_top = 0.9
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
                     cbar_ax=cbar_ax)
    
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


def plot_vintage_analysis(trade_df: pd.DataFrame, 
                            initial_investment: float = 3000, 
                            min_year: int = None,
                            max_year: int = None):
    """
    Creates a vintage analysis plot with daily granularity for the dividend harvesting strategy,
    rebasing each year's dates to a common dummy year so that all years start at the same x-axis position.
    
    For each year (vintage), this function simulates the running total of capital,
    starting at the specified initial investment (default $3000). The total capital
    is updated at each dividend event (using the 'profit' from that trade), and the 
    running total is recorded on a daily basis (by forward filling between trade dates).
    
    The x-axis shows the months (rebased to the dummy year) and each vintage (year) is plotted
    as a separate line. Optionally, if a minimum year is provided, only trades from years greater 
    than or equal to min_year are included.
    
    Parameters:
        trade_df (pd.DataFrame): DataFrame containing individual trade data with at least:
            - 'ex_div_date': The datetime when the dividend event occurred.
            - 'profit': The profit (or loss) from that trade.
            - 'year': (Optional) The year of the dividend event; if missing, it will be extracted 
                      from 'ex_div_date'.
        initial_investment (float): The starting capital for each year.
        min_year (int, optional): Only include trades for years >= min_year.
    
    Returns:
        None. Displays the vintage analysis plot.
    """
    dummy_year = 2000
    # Ensure that ex_div_date is a datetime type.
    trade_df['ex_div_date'] = pd.to_datetime(trade_df['ex_div_date'])
    
    # Create a 'year' column if it doesn't exist.
    if 'year' not in trade_df.columns:
        trade_df['year'] = trade_df['ex_div_date'].dt.year

    # Filter by min_year if provided.
    if min_year is not None:
        trade_df = trade_df[trade_df['year'] >= min_year]
    if max_year is not None:
        trade_df = trade_df[trade_df['year'] <= max_year]
    
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
            # Normalize the ex_div_date (drop time component)
            trade_date = row['ex_div_date'].normalize()
            current_capital += row['profit']
            event_dates.append(trade_date)
            cumulative_capital.append(current_capital)
        
        # Create a Series of the event dates and cumulative capital.
        s = pd.Series(data=cumulative_capital, index=event_dates)
        # In case of multiple trades on the same day, keep the last value.
        s = s.groupby(s.index).last()
        # Reindex the series to include every day of the year and forward-fill missing values.
        daily_capital = s.reindex(daily_index, method='ffill')
        
        # Rebase the daily_index to a common dummy year so that all lines start at the same x-axis.
        dummy_index = daily_index.map(lambda d: d.replace(year=dummy_year))
        
        # Plot the daily running total for this year using the dummy_index.
        plt.plot(dummy_index, daily_capital, label=str(year))
    
    # Format the x-axis to show month abbreviations.
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.axhline(y=initial_investment, color='red', linestyle='--', linewidth=1)
    
    plt.xlabel("Month")
    plt.ylabel("Total Capital")
    plt.title("Vintage Analysis: Daily Running Total Capital by Year")
    plt.legend(title="Year")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def plot_axjo_vintage(axjo_df: pd.DataFrame, 
                       initial_investment: float = 2000, 
                       min_year: int = None,
                       max_year: int = None,
                       dummy_year: int = 2000):
    """
    Creates a vintage analysis plot for an AXJO index investment strategy.
    
    For each year in the provided AXJO dataset, the function simulates investing a fixed amount
    (default $2000) on the first trading day of that year. It calculates the number of shares purchased
    based on the first day's closing price, then computes the daily portfolio value as the product of the 
    number of shares and the daily closing price. The daily values are reindexed over the full year and 
    forward-filled. Finally, the dates are rebased to a dummy year (default 2000) so that each vintage
    starts at January 1, allowing for a direct comparison across years.
    
    Parameters:
        axjo_df (pd.DataFrame): DataFrame with daily AXJO index data. It should have a 'Date' column
                                (or a datetime index) and a 'Close' column.
        initial_investment (float): The amount invested on the first trading day of each year (default $2000).
        min_year (int, optional): Only include years greater than or equal to this value.
        dummy_year (int): The year to which all dates are rebased for plotting (default 2000).
    
    Returns:
        None. Displays the vintage analysis plot.
    """
    # Ensure the Date column is datetime and set as index if needed.
    if 'Date' in axjo_df.columns:
        axjo_df['Date'] = pd.to_datetime(axjo_df['Date'])
        axjo_df.set_index('Date', inplace=True)
    else:
        # If already indexed by date, ensure it's datetime.
        if not pd.api.types.is_datetime64_any_dtype(axjo_df.index):
            axjo_df.index = pd.to_datetime(axjo_df.index)
    
    # Create a 'year' column from the index.
    axjo_df = axjo_df.copy()
    axjo_df['year'] = axjo_df.index.year
    if min_year is not None:
        axjo_df = axjo_df[axjo_df['year'] >= min_year]
    if max_year is not None:
        axjo_df = axjo_df[axjo_df['year'] <= max_year]
    
    plt.figure(figsize=(12, 8))
    
    # Group by year.
    grouped = axjo_df.groupby('year')
    
    for year, group in grouped:
        group = group.sort_index()
        if group.empty:
            continue
        
        # Use the first trading day's close price to compute the number of shares.
        first_day = group.iloc[0]
        first_close = first_day['Close']
        shares = initial_investment / first_close
        
        # Create a full daily date range for the entire year.
        daily_index = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        # Compute daily portfolio value: shares * daily Close.
        # Reindex the group's Close series to the full daily index using forward-fill.
        s = group['Close'] * shares
        daily_value = s.reindex(daily_index, method='ffill')
        
        # Rebase the daily_index to the dummy year.
        dummy_index = daily_index.map(lambda d: d.replace(year=dummy_year))
        
        # Plot the daily portfolio value for the year.
        plt.plot(dummy_index, daily_value, label=str(year))
    
    # Format the x-axis to show month abbreviations.
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.axhline(y=initial_investment, color='red', linestyle='--', linewidth=1)
    
    plt.xlabel("Month")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Vintage Analysis: AXJO Investment Performance by Year")
    plt.legend(title="Year")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_vintage_ratio(trade_df: pd.DataFrame,
                       axjo_df: pd.DataFrame,
                       initial_investment_div: float = 3000,
                       initial_investment_ax: float = 2000,
                       min_year: int = None,
                       max_year: int = None,
                       dummy_year: int = 2000):
    """
    Creates a vintage analysis plot for the ratio of a dividend investing strategy 
    (based on trade_df) over a passive AXJO index strategy (based on axjo_df).

    For each year present in both datasets, the function simulates:
      - A dividend strategy: Starting at initial_investment_div, updated daily 
        by adding the 'profit' from each dividend trade.
      - An AXJO strategy: Investing initial_investment_ax on the first trading day 
        of the year, computing the daily portfolio value (shares * daily Close).
    
    The ratio is computed daily as:
        ratio = (Dividend Strategy Running Total) / (AXJO Daily Portfolio Value)
    
    Each year's daily values are reindexed over the full year and forward-filled, then 
    the dates are rebased to a dummy year (default 2000) so that all lines start at January 1.
    
    Parameters:
        trade_df (pd.DataFrame): DataFrame with dividend strategy trade data. It must include:
                                 - 'ex_div_date': The datetime when the dividend event occurred.
                                 - 'profit': The profit (or loss) from that trade.
        axjo_df (pd.DataFrame): DataFrame with daily AXJO index data. It should have either:
                                - a 'Date' column (which will be parsed to datetime and set as index), or
                                - a datetime index, and it must include a 'Close' column.
        initial_investment_div (float): Starting capital for the dividend strategy per year.
        initial_investment_ax (float): Investment amount for the AXJO strategy per year.
        min_year (int, optional): Only include years greater than or equal to this value.
        max_year (int, optional): Only include years less than or equal to this value.
        dummy_year (int): The year to which all dates are rebased for plotting (default 2000).
    
    Returns:
        None. Displays the vintage ratio plot.
    """
    # Process dividend trade data.
    trade_df = trade_df.copy()
    trade_df['ex_div_date'] = pd.to_datetime(trade_df['ex_div_date'])
    if 'year' not in trade_df.columns:
        trade_df['year'] = trade_df['ex_div_date'].dt.year
    if min_year is not None:
        trade_df = trade_df[trade_df['year'] >= min_year]
    if max_year is not None:
        trade_df = trade_df[trade_df['year'] <= max_year]
        
    # Process AXJO data.
    axjo_df = axjo_df.copy()
    if 'Date' in axjo_df.columns:
        axjo_df['Date'] = pd.to_datetime(axjo_df['Date'])
        axjo_df.set_index('Date', inplace=True)
    else:
        if not pd.api.types.is_datetime64_any_dtype(axjo_df.index):
            axjo_df.index = pd.to_datetime(axjo_df.index)
    axjo_df['year'] = axjo_df.index.year
    if min_year is not None:
        axjo_df = axjo_df[axjo_df['year'] >= min_year]
    if max_year is not None:
        axjo_df = axjo_df[axjo_df['year'] <= max_year]
    
    plt.figure(figsize=(12, 8))
    
    # Group both datasets by year.
    div_grouped = trade_df.groupby('year')
    axjo_grouped = axjo_df.groupby('year')
    
    # Process only years that are common to both groups.
    common_years = sorted(set(div_grouped.groups.keys()).intersection(axjo_grouped.groups.keys()))
    
    for year in common_years:
        # --- Dividend Strategy ---
        group_div = div_grouped.get_group(year).sort_values(by='ex_div_date')
        daily_index = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        event_dates = [pd.Timestamp(f"{year}-01-01")]
        capital_div = [initial_investment_div]
        current_capital_div = initial_investment_div
        
        for _, row in group_div.iterrows():
            trade_date = row['ex_div_date'].normalize()
            current_capital_div += row['profit']
            event_dates.append(trade_date)
            capital_div.append(current_capital_div)
            
        s_div = pd.Series(data=capital_div, index=event_dates).groupby(level=0).last()
        daily_capital_div = s_div.reindex(daily_index, method='ffill')
        
        # --- AXJO Strategy ---
        group_ax = axjo_grouped.get_group(year).sort_index()
        first_day_ax = group_ax.iloc[0]
        first_close_ax = first_day_ax['Close']
        shares = initial_investment_ax / first_close_ax
        s_ax = group_ax['Close'] * shares
        daily_value_ax = s_ax.reindex(daily_index, method='ffill')
        
        # Compute the ratio: dividend strategy running total divided by AXJO running total.
        ratio = daily_capital_div / daily_value_ax
        
        # Rebase the daily_index to dummy_year.
        dummy_index = daily_index.map(lambda d: d.replace(year=dummy_year))
        plt.plot(dummy_index, ratio, label=str(year))
    
    # Format x-axis with month abbreviations.
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.axhline(y=1, color='red', linestyle='--', linewidth=1)
    
    plt.xlabel("Month")
    plt.ylabel("Dividend / AXJO Ratio")
    plt.title("Vintage Analysis Ratio: Dividend Strategy vs. AXJO Index")
    plt.legend(title="Year")
    plt.grid(True)
    plt.tight_layout()
    plt.show()