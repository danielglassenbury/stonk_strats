# Dividend Harvest Strategy Analysis

This project is a collection of tools for analysing dividend harvesting strategies and comparing them to passive index (hold) strategies. The project fetches historical data using [yfinance](https://github.com/ranaroussi/yfinance), performs backtesting, and generates interactive vintage analysis plots with [Plotly](https://plotly.com/python/) and [Plotly Express](https://plotly.com/python/plotly-express/).

## Overview

- **Dividend Harvest Strategy:**  
  Buying shares before the ex-dividend date, collect the dividend, sell post-ex-dividend date. Transaction fees and investment amount at each dividend event are considered, as well as custom buy-sell-success-ratio which considers intra-day trade execution sucess on the daily high/low price. The performance is computed based on the profit (or loss) from each dividend event.


- **Trade Performance:**
  - **Heatmap:** Visualises historical dividend harvesting % returns, aggregated by stock and year.
  - **Scatter plot:**  Visulaised individual trade % returns over time.
  - **Boxplot:** Summarises the scatter plot data, grouping by year. Provides insights into the central tendency and variability of returns, highlighting the median, quartiles, and potential outliers.

- **Strategy Performance:**  
  Generate backtesting vintage analysis plots that compare the performance of the dividend harvesting strategy across different years.  
  - **Active Strategy Vintage Plot:** Shows the daily running portfolio value for each vintage.
  - **Passive Hold Strategy Vintage Plot:** Simulates a passive investment (e.g., into the AXJO index or a similar hold strategy) with a fixed investment on the first trading day of each year.
  - **Ratio Analysis:** Compares the active dividend strategy to the passive hold strategy by plotting the ratio of the portfolio values over time.

- **Event Overlap Analysis:**  
  Analyse and visualise the frequency of dividend events and the number of overlapping open positions on each day.