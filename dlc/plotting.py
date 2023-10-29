import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_interpolation(ax, data, title):
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

    Returns:
        matplotlib.collections.LineCollection: The LineCollection object that was added to the axes.
    """        
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