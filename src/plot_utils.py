import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

def plot_heatmap(df: pd.DataFrame, figsize: tuple = (12, 8),
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


def plot_yearly_boxplot(data:pd.DataFrame, bssr_label:float, ylim:tuple=(-5, 15)):
    plt.figure(figsize=(12, 5))
    sns.boxplot(x='year', y='pct_return', data=data)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)

    # formatting
    plt.xlabel("Year")
    plt.ylabel("% Profit")
    plt.title(f"Individual trade % return, grouped by year. BSSR = {bssr_label}")
    plt.xticks(rotation=0)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.show()