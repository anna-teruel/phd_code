import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def interpolation_plot(ax, data, title, style='light'):
    """
    Plot the provided data on a given axis, highlighting segments based on likelihood values.

    This function visualizes animal tracking data as a line plot where each segment of the line 
    is color-coded based on likelihood values. The segments with a higher likelihood value will 
    appear more prominently (i.e., greener in the RdYlGn colormap) than those with lower likelihood values.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes on which to plot the data.
        data (pandas.DataFrame): A dataframe containing columns for 'x', 'y', and 'likelihood'. 
                                 Each row represents a point in the tracking data.
        title (str): Title to set for the provided axes.
        style (str): Plot style. Default "light", but user can set it to "dark"

    Returns:
        matplotlib.collections.LineCollection: The LineCollection object that was added to the axes.
    """
    # Save the current style
    current_style = plt.rcParams.copy()

    # Apply the desired style
    if style == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    try:
        x = data.loc[:, (slice(None), slice(None), 'x')].values
        y = data.loc[:, (slice(None), slice(None), 'y')].values
        likelihood = data.loc[:, (slice(None), slice(None), 'likelihood')].values.flatten()

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        colormap = plt.cm.RdYlGn
        lc = LineCollection(segments, cmap=colormap, norm=plt.Normalize(likelihood.min(), likelihood.max()), linewidth=0.5, alpha=0.5)
        lc.set_array(likelihood)

        ax.add_collection(lc)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_title(title)

        return lc
    finally:
        # Restore the original style
        plt.rcParams.update(current_style)

def density_plot(bodypart, cmap='viridis', cbar=True, title="", style='light'):
    """
    Generate a 2D kernel density estimate (KDE) plot for the provided data points.

    Args:
        bodypart (pandas.DataFrame): A dataframe containing columns 'x' and 'y'.
        cmap (str, optional): Colormap for the KDE plot. Defaults to 'viridis'.
        cbar (bool, optional): Whether to display a colorbar. Defaults to True.
        title (str, optional): Title for the plot. Defaults to an empty string.
        style (str): Plot style. Default "light", user can set to "dark".

    Returns:
        None: Saves the plot as a PDF file.
    """ 
    if style == 'dark': 
        plt.style.use('dark_background') 
    elif style == "light":
        plt.style.use('default') 

    plt.subplots(figsize=(10, 10))
    sns.set(style="dark")

    x_data = bodypart.loc[:,(slice(None), slice(None), 'x')].values.flatten()
    y_data = bodypart.loc[:,(slice(None), slice(None), 'y')].values.flatten()

    ax = sns.kdeplot(
            x=x_data,
            y=y_data,
            fill=True,
            cbar=cbar,
            cmap=cmap,
            levels=10,
            alpha=0.9,
        )    

    plt.title(title)
    ax.set_aspect("equal", adjustable="box") 
    plt.savefig("Density_plot_" + title + ".pdf", format="pdf", dpi=100)
