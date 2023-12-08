import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from dlc.data import Interpolation
from dlc.data import Centroid
from dlc.load_data import DataLoader

class InterpolationPlot:
    def __init__(self, style='light', colormap='RdYlGn', linewidth=0.5, alpha=1):
        """
        Initialize an InterpolationPlot object for visualizing animal tracking data.

        Args:
            style (str, optional): Specifies the plot style ('light' or 'dark'). Defaults to 'light'.
            colormap (matplotlib.colors.Colormap, optional): Colormap to be used for plotting. Defaults to plt.cm.RdYlGn.
            linewidth (float, optional): Width of the plot lines. Defaults to 0.5.
            alpha (float, optional): Transparency level of the plot lines. Defaults to 0.5.
        """
        self.style = style
        self.interpolator = Interpolation()
        self.colormap = plt.get_cmap(colormap)
        self.linewidth = linewidth
        self.alpha = alpha

    def interpolation_plot(self, ax, data, title, cbar = True):
        """
        Plot data on a given axes object with specified style, colormap, linewidth, and alpha.

        This function plots data on the provided axes with customizable styling parameters. It creates a line
        plot connecting points in the 'x' and 'y' columns of the DataFrame. The color and transparency of the
        lines are determined by the 'likelihood' column of the DataFrame. A colorbar is added to the plot to
        indicate the likelihood values.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes object to plot on.
            data (pandas.DataFrame): DataFrame containing 'x', 'y', and 'likelihood' values.
            title (str): Title of the plot.

        Returns:
            matplotlib.collections.LineCollection: The LineCollection object representing the plotted data.
        """        
        current_style = plt.rcParams.copy()  # Save the current style
        plt.style.use('dark_background' if self.style == 'dark' else 'default')

        try:
            x = data['x'].values
            y = data['y'].values
            likelihood = data['likelihood'].values

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = plt.Normalize(likelihood.min(), likelihood.max())
                
            lc = LineCollection(segments, cmap=self.colormap, norm=norm, array=likelihood, linewidth=self.linewidth, alpha=self.alpha)
            lc.set_array(likelihood)

            ax.add_collection(lc)
            ax.autoscale()

            # Adding colorbar
            if cbar:
                cbar = plt.colorbar(lc, ax=ax)
                cbar.set_label('Likelihood Value')


            return lc
        finally:
            plt.rcParams.update(current_style)

    def plot_bodyparts(self, file_path, bodyparts, title, save=False, save_directory='.', save_filename='plot.png'):
        """
        Plot data for multiple body parts from a single file.

        This function reads data from a single file specified by 'file_path' and plots the trajectories of
        multiple body parts over time. The 'bodyparts' argument should be a list of body part names to be
        included in the plot. Each body part will be plotted in a separate subplot.

        Args:
            file_path (str): Path to the .h5 file containing the data.
            bodyparts (list of str): List of body parts to include in the plot.
            title_base (str): Base title for the plots, each subplot will have this title followed by the body part name.
            save (bool, optional): If True, save the plot to the specified directory. If False, display the plot. Defaults to False.

        """        
        data = self.interpolator.interpolate_data(file_path, bodyparts)
        num_bodyparts = len(bodyparts)
        fig, axes = plt.subplots(1, num_bodyparts, figsize=(5 * num_bodyparts, 5), squeeze=False)

        for i, bp in enumerate(bodyparts):
            ax = axes[0, i]
            title = f"{title} - {bp}"

            scorer = data.columns.levels[0][0]
            # Constructing a DataFrame for each body part
            bp_data = pd.DataFrame({
                'x': data[(scorer, bp, 'x')].values,
                'y': data[(scorer, bp, 'y')].values,
                'likelihood': data[(scorer, bp, 'likelihood')].values
            })

            self.interpolation_plot(ax, bp_data, title)

        plt.tight_layout()
        if save:
            save_path = Path(save_directory) / save_filename
            Path(save_directory).mkdir(parents=True, exist_ok=True)  # Create directory if it does not exist
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_file(self, file_path, bodyparts, title_base):
        """
        Creates a figure with subplots for each body part from a single file.

        Args:
            file_path (str): Path to the .h5 file.
            bodyparts (list of str): Body parts to include in the plot.
            title_base (str): Base title for the plots, each subplot will have this title followed by the body part name.
        """
        self.plot_bodyparts(file_path, bodyparts, title_base)

    def plot_directory(self, directory_path, bodyparts, title_base):
        """
        Creates figures with subplots for each body part from all .h5 files in a directory.

        Args:
            directory_path (str): Path to the directory containing .h5 files.
            bodyparts (list of str): Body parts to include in the plot.
            title_base (str): Base title for the plots, each subplot will have this title followed by the body part name.
        """
        directory = Path(directory_path)
        h5_files = list(directory.glob('*filtered.h5'))  # Adjust pattern if needed

        for file_path in h5_files:
            self.plot_file(str(file_path), bodyparts, title_base.format(file_path.stem))

    def plot_comparison(self, file_path, bodyparts, title):
        """
        Compares data before and after interpolation in side-by-side plots.

        Args:
            file_path (str): Path to the .h5 file.
            bodyparts (list of str): Body parts to include in the plot.
            title (str): Title of the plot.
            colormap, linewidth, alpha: Styling parameters for the plot.
        """
        loader = DataLoader(file_path)
        raw_data = loader.read_data(file_path)  
        print("File path:", file_path)

        int_data = self.interpolator.interpolate_data(input_data=file_path, bodyparts=bodyparts)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        self.plot_bodyparts(ax1, raw_data, f"{title} - Before Interpolation")
        self.plot_bodyparts(ax2, int_data, f"{title} - After Interpolation")

        # Add colorbar and show plot
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(ax2.collections[0], cax=cbar_ax)  # Assuming ax2 has the LineCollection
        cbar.set_label('Likelihood Value')
        plt.tight_layout()
        plt.show()

class TrackingPlot:
    def __init__(self, style='light'):
        self.style = style
        if self.style == 'dark':
            plt.style.use('dark_background')
        elif self.style == 'light':
            plt.style.use('default')

    def density_plot(self, bodypart, cmap='viridis', cbar=True, title=""):
        """
        Generate a 2D kernel density estimate (KDE) plot for the provided data points.

        Args:
            bodypart (pandas.DataFrame): A dataframe containing columns 'x' and 'y'.
            cmap (str, optional): Colormap for the KDE plot. Defaults to 'viridis'.
            cbar (bool, optional): Whether to display a colorbar. Defaults to True.
            title (str, optional): Title for the plot. Defaults to an empty string.

        Returns:
            None: Saves the plot as a PDF file.
        """
        plt.figure(figsize=(10, 10))
        sns.set(style=self.style)

        x_data = bodypart.loc[:, (slice(None), slice(None), 'x')].values.flatten()
        y_data = bodypart.loc[:, (slice(None), slice(None), 'y')].values.flatten()

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

    def line_plot(self):
        pass  # Implement the line plot function here

    def plot_file(self, file_path, bodyparts, title):
        """
        Plot data from a single file.

        Args:
            file_path (str): Path to the .h5 file.
            bodyparts (list): List of bodyparts to include in the plot.
            title (str): Title of the plot.

        Returns:
            None: Saves the plot as a PDF file.
        """
        # Load data from file (you may need to adapt this part)
        data = pd.read_hdf(file_path)

        # Perform necessary operations on data and call plot functions
        # Example: self.density_plot(data, cmap='viridis', cbar=True, title=title)

    def plot_directory(self, directory_path, bodyparts, title):
        """
        Plot data from all files in a directory.

        Args:
            directory_path (str): Path to the directory containing .h5 files.
            bodyparts (list): List of bodyparts to include in the plot.
            title (str): Title of the plot.

        Returns:
            None: Saves the plots for all files in the directory.
        """
        # List all .h5 files in the directory
        h5_files = [f for f in os.listdir(directory_path) if f.endswith(".h5")]

        for file in h5_files:
            file_path = os.path.join(directory_path, file)
            self.plot_file(file_path, bodyparts, title + "_" + os.path.splitext(file)[0])

