import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os

from dlc.data import Interpolation
from dlc.data import Centroid
from dlc.load_data import DataLoader


class InterpolationPlot:
    def __init__(self, style="light", colormap="RdYlGn", linewidth=0.5, alpha=1):
        """
        Initialize an InterpolationPlot object for visualizing animal tracking data.

        Args:
            style (str, optional): Specifies the plot style ('light' or 'dark'). Defaults to 'light'.
            colormap (matplotlib.colors.Colormap, optional): Colormap to be used for plotting. Defaults to plt.cm.RdYlGn.
            linewidth (float, optional): Width of the plot lines. Defaults to 0.5.
            alpha (float, optional): Transparency level of the plot lines. Defaults to 0.5.
        """
        self.style = style
        if style == "dark":
            plt.style.use("dark_background")
        elif style == "light":
            plt.style.use("default")

        self.interpolator = Interpolation()
        self.colormap = plt.get_cmap(colormap)
        self.linewidth = linewidth
        self.alpha = alpha

    def interpolation_plot(self, ax, data, title, cbar=True):
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
            cbar (bool, optional): Whether to include a colorbar in the plot. Defaults to True.

        Returns:
            matplotlib.collections.LineCollection: The LineCollection object representing the plotted data.
        """
        x = data["x"].values
        y = data["y"].values
        likelihood = data["likelihood"].values

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(likelihood.min(), likelihood.max())

        lc = LineCollection(
            segments,
            cmap=self.colormap,
            norm=norm,
            array=likelihood,
            linewidth=self.linewidth,
            alpha=self.alpha,
        )
        lc.set_array(likelihood)

        ax.add_collection(lc)
        ax.autoscale()

        # Adding colorbar
        if cbar:
            cbar = plt.colorbar(lc, ax=ax)
            cbar.set_label("Likelihood")

        return lc

    def plot_comparison(self, axes, raw_data, int_data, bodypart, title):
        """
        Compares data before and after interpolation in side-by-side plots.

        Args:
            axes (list of matplotlib.axes.Axes): List of two axes for plotting before and after data.
            raw_data (pandas.DataFrame): Raw data DataFrame before interpolation.
            int_data (pandas.DataFrame): DataFrame after interpolation.
            bodypart (str): Name of the body part being plotted.
            title (str): Title of the plot.
        """
        bp_raw_data = pd.DataFrame(
            {
                "x": raw_data.loc[:, (slice(None), bodypart, "x")].values.flatten(),
                "y": raw_data.loc[:, (slice(None), bodypart, "y")].values.flatten(),
                "likelihood": raw_data.loc[
                    :, (slice(None), bodypart, "likelihood")
                ].values.flatten(),
            }
        )
        self.interpolation_plot(axes[0], bp_raw_data, "Before Interpolation")

        bp_int_data = pd.DataFrame(
            {
                "x": int_data.loc[:, (slice(None), bodypart, "x")].values.flatten(),
                "y": int_data.loc[:, (slice(None), bodypart, "y")].values.flatten(),
                "likelihood": int_data.loc[
                    :, (slice(None), bodypart, "likelihood")
                ].values.flatten(),
            }
        )
        self.interpolation_plot(axes[1], bp_int_data, "After Interpolation")

    def plot_bodyparts(self, file_path, bodyparts, title):
        """Furth
        Plot data for multiple body parts from a single file.

        This function reads data from a single file specified by 'file_path' and plots the trajectories of
        multiple body parts over time. The 'bodyparts' argument should be a list of body part names to be
        included in the plot. Each body part will be plotted in a separate subplot.

        Args:
            file_path (str): Path to the .h5 file containing the data.
            bodyparts (list of str): List of body parts to include in the plot.
            title (str): Base title for the plots, each subplot will have this title followed by the body part name.
        """
        loader = DataLoader(file_path)
        raw_data = loader.read_data(file_path)  # data before inteprolation
        int_data = self.interpolator.interpolate_data(
            file_path, bodyparts
        )  # data after interpolation

        num_bodyparts = len(bodyparts)
        fig, all_axes = plt.subplots(
            num_bodyparts, 2, figsize=(10, 5 * num_bodyparts)
        )  # 2 columns for before and after
        fig.suptitle(title)

        if (
            num_bodyparts == 1
        ):  # If there's only one body part, we make sure all_axes is a 2D array
            all_axes = [all_axes]

        for i, bp in enumerate(bodyparts):
            axes = all_axes[i]  # This is a pair of axes (one row for each body part)
            # Now pass the correct number of arguments to plot_comparison
            self.plot_comparison(axes, raw_data, int_data, bp, title)

            axes[0].set_title(f"{bp} - Before")
            axes[1].set_title(f"{bp} - After")

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust layout to make room for suptitle

    def plot_file(
        self,
        file_path,
        bodyparts,
        title,
        save=False,
        save_directory=".",
        save_filename="plot.png",
    ):
        """
        Creates a figure with subplots for each body part from a single file.

        Args:
            file_path (str): Path to the .h5 file.
            bodyparts (list of str): Body parts to include in the plot.
            title (str): Base title for the plots, each subplot will have this title followed by the body part name.
            save (bool, optional): If True, save the plot to the specified directory. If False, display the plot. Defaults to False.
            save_directory (str, optional): Directory where the plot should be saved if 'save' is True. Defaults to '.'.
            save_filename (str, optional): Filename for the saved plot if 'save' is True. Defaults to 'plot.png'.
        """
        name = self.data_loader.get_file_name(file_path)
        full_title = f"{title} - {name}"

        title_plot = f"{title} - {full_title}"
        
        self.plot_bodyparts(file_path, bodyparts, title_plot)

        if save:
            save_path = Path(save_directory) / save_filename
            Path(save_directory).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_directory(self, directory_path, bodyparts, title):
        """
        Creates figures with subplots for each body part from all .h5 files in a directory.

        Args:
            directory_path (str): Path to the directory containing .h5 files.
            bodyparts (list of str): Body parts to include in the plot.
            title (str): Base title for the plots, each subplot will have this title followed by the body part name.
        """
        directory = Path(directory_path)
        h5_files = list(directory.glob("*filtered.h5"))  # Adjust pattern if needed

        for file_path in h5_files:
            self.plot_file(str(file_path), bodyparts, title.format(file_path.stem))


class TrackingPlot:
    def __init__(self, style="light"):
        """
        Initialize a TrackingPlot object.

        Args:
            style (str, optional): Plot style ('light' or 'dark'). Defaults to 'light'.
        """
        self.centroid = Centroid()
        self.data_loader = DataLoader()

        self.style = style
        if self.style == "dark":
            plt.style.use("dark_background")
        elif self.style == "light":
            plt.style.use("default")

    def load_data(self, file_path, bodyparts):
        """
        Load and calculate centroids for data from a single file or directory.

        Args:
            file_path (str): Path to the .h5 file or directory containing .h5 files.
            bodyparts (list): List of bodyparts to include in centroid calculation.

        Returns:
            dict or pd.DataFrame: Dictionary of centroids for multiple files or single DataFrame for a single file.
        """
        return self.centroid.get_centroid(file_path, bodyparts)

    def density_plot(self, bodypart, cmap="viridis", cbar=True, title=""):
        """
        Generate a 2D kernel density estimate (KDE) plot for the provided data points.

        Args:
            bodypart (pandas.DataFrame): A dataframe containing columns 'x' and 'y'.
            cmap (str, optional): Colormap for the KDE plot. Defaults to 'viridis'.
            cbar (bool, optional): Whether to display a colorbar. Defaults to True.
            title (str, optional): Title for the plot. Defaults to an empty string.

        Returns:
            None: Displays the KDE plot.
        """
        plt.figure(figsize=(10, 10))

        x = bodypart.loc[:, (slice(None), slice(None), "x")].values.flatten()
        y = bodypart.loc[:, (slice(None), slice(None), "y")].values.flatten()
        ax = sns.kdeplot(x=x, y=y, fill=True, cmap=cmap, levels=10, alpha=0.9)
        if cbar:
            cbar = plt.colorbar(ax.collections[0])
            cbar_values = cbar.get_ticks()
            if cbar_values.size > 0:
                cbar_labels = (
                    ["-"] + [""] * (len(cbar_values) - 2) + ["+"]
                )  # replace first and last values with labels
                cbar.set_ticks(cbar_values)
                cbar.set_ticklabels(cbar_labels)

        plt.title(title)

    def line_plot(self, bodypart, title, color="red"):
        """
        Generate a line plot for the provided data points.

        Args:
            bodypart (pandas.DataFrame): A dataframe containing columns 'x' and 'y'.
            title (str): Title for the plot.
            color (str, optional): Color of the line plot. Defaults to 'red'.

        Returns:
            None: Displays the line plot.
        """
        plt.figure(figsize=(10, 10))

        x = bodypart.loc[:, (slice(None), slice(None), "x")].values.flatten()
        y = bodypart.loc[:, (slice(None), slice(None), "y")].values.flatten()

        plt.plot(x, y, marker="o", markersize=2, linestyle="-", color=color)
        plt.title(title)

    def plot_file(
        self,
        file_path,
        bodyparts,
        title,
        plot_type="density",
        save=False,
        save_directory=".",
        save_filename="plot.png",
    ):
        """
        Plot data from a single file.

        Args:
            file_path (str): Path to the .h5 file.
            bodyparts (list): List of bodyparts to include in the plot.
            title (str): Title of the plot.
            plot_type (str, optional): Type of plot ('density' or 'line'). Defaults to 'density'.
            save (bool, optional): Whether to save the plot as a file. Defaults to False.
            save_directory (str, optional): Directory to save the plot file. Defaults to '.'.
            save_filename (str, optional): Filename to save the plot. Defaults to 'plot.png'.

        Returns:
            None: Displays or saves the plot.
        """
        data = self.load_data(file_path, bodyparts)
        centroid_df = data.loc[:, (slice(None), "centroid", slice(None))]
        name = self.data_loader.get_file_name(file_path)
        full_title = f"{title} - {name}"

        if plot_type == "density":
            self.density_plot(centroid_df, title=f"{full_title}")
        elif plot_type == "line":
            self.line_plot(centroid_df, title=f"{full_title}")

        if save:
            save_path = Path(save_directory) / save_filename
            Path(save_directory).mkdir(
                parents=True, exist_ok=True
            )  # Create directory if it does not exist
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_directory(
        self,
        dir_path,
        bodyparts,
        title,
        plot_type="density",
        save=False,
        save_directory=".",
        save_filename="plot.png",
    ):
        """
        Plot data from all files in a directory.

        Args:
            dir_path (str): Path to the directory containing .h5 files.
            bodyparts (list): List of bodyparts to include in the plot.
            title (str): Title of the plot.
            plot_type (str, optional): Type of plot ('density' or 'line'). Defaults to 'density'.
            save (bool, optional): Whether to save the plots as files. Defaults to False.
            save_directory (str, optional): Directory to save the plot files. Defaults to '.'.
            save_filename (str, optional): Filename to save the plots. Defaults to 'plot.png'.

        Returns:
            None: Displays or saves the plots for all files in the directory.
        """
        data_dict = self.data_loader.read_directory(dir_path)

        for file_name in data_dict.keys():
            file_path = os.path.join(dir_path, file_name)
            
            name = self.data_loader.get_file_name(file_path)
            full_title = f"{title} - {name}"

            self.plot_file(
                file_path=file_path,
                bodyparts=bodyparts,
                title=full_title,
                plot_type=plot_type,
                save=save,
                save_directory=save_directory,
                save_filename=save_filename
            )
