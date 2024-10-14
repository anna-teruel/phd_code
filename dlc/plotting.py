"""
Plotting module for visualizing animal tracking data.
@author Anna Teruel-Sanchis, 2023
"""

import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import pandas as pd
from pathlib import Path
import os

from dlc.data import Interpolation
from dlc.data import Centroid
from dlc.load_data import DataLoader


class InterpolationPlot:
    def __init__(self, interpolator, style="light", colormap="RdYlGn", linewidth=0.5, alpha=1):
        """
        Initialize an InterpolationPlot object with specified parameters.

        Args:
            interpolator (Interpolation): An instance of the Interpolation class.
            style (str, optional): Plotting style, either 'light' or 'dark'. Defaults to 'light'.
            colormap (str, optional): Colormap to use for the scatter plot. Defaults to 'RdYlGn'.
            linewidth (float, optional): Line width for the scatter plot. Defaults to 0.5.
            alpha (float, optional): Alpha value for the scatter plot. Defaults to 1.
        """
        self.interpolator = interpolator
        self.style = style
        if style == "dark":
            plt.style.use("dark_background")
        elif style == "light":
            plt.style.use("default")

        self.colormap = plt.get_cmap(colormap)
        self.linewidth = linewidth
        self.alpha = alpha

    def interpolation_plot(self, ax, data, title, cbar=True):
        """
        Create a line plot for the given data.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to plot.
            data (pd.DataFrame): DataFrame containing 'x', 'y', and 'likelihood' columns.
            title (str): Title of the plot.
            cbar (bool, optional): Whether to include a colorbar. Defaults to True.
        """
        points = data[['x', 'y']].values.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=self.colormap, linewidth=self.linewidth, alpha=self.alpha)
        lc.set_array(data['likelihood'])
        ax.add_collection(lc)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.autoscale()  # Automatically scale the axis to fit the data
        if cbar:
            plt.colorbar(lc, ax=ax, label='Likelihood')

    def plot_comparison(self, raw_data, int_data, bodypart, axes):
        """
        Plot a comparison of raw and interpolated data for a specific body part.

        Args:
            raw_data (pd.DataFrame): Raw data DataFrame.
            int_data (pd.DataFrame): Interpolated data DataFrame.
            bodypart (str): The body part to plot.
            axes (list of matplotlib.axes.Axes): List of axes for plotting.
        """
        bp_raw_data = pd.DataFrame(
            {
                "x": raw_data.loc[:, (slice(None), bodypart, "x")].values.flatten(),
                "y": raw_data.loc[:, (slice(None), bodypart, "y")].values.flatten(),
                "likelihood": raw_data.loc[:, (slice(None), bodypart, "likelihood")].values.flatten(),
            }
        )
        self.interpolation_plot(axes[0], bp_raw_data, f"Before Interpolation - {bodypart}")

        bp_int_data = pd.DataFrame(
            {
                "x": int_data.loc[:, (slice(None), bodypart, "x")].values.flatten(),
                "y": int_data.loc[:, (slice(None), bodypart, "y")].values.flatten(),
                "likelihood": int_data.loc[:, (slice(None), bodypart, "likelihood")].values.flatten(),
            }
        )
        self.interpolation_plot(axes[1], bp_int_data, f"After Interpolation - {bodypart}")

    def plot_bodyparts(self, raw_data, int_data, bodyparts, title, output_dir):
        """
        Plot comparisons for multiple body parts and save the figure.

        Args:
            raw_data (pd.DataFrame): Raw data DataFrame.
            int_data (pd.DataFrame): Interpolated data DataFrame.
            bodyparts (list): List of body parts to plot.
            title (str): Title of the figure.
            output_dir (str): Directory to save the figure.
        """
        num_bodyparts = len(bodyparts)
        fig, all_axes = plt.subplots(num_bodyparts, 2, figsize=(10, 5 * num_bodyparts))  # 2 columns for before and after
        fig.suptitle(title)

        for i, bodypart in enumerate(bodyparts):
            self.plot_comparison(raw_data, int_data, bodypart, all_axes[i])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = os.path.join(output_dir, f"{title}.png")
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Figure saved to {output_path}")

    def plot_file(self, file_path, bodyparts, title, output_dir):
        """
        Plot comparisons for a single file and save the figure.

        Args:
            file_path (str): Path to the file.
            bodyparts (list): List of body parts to plot.
            title (str): Title of the figure.
            output_dir (str): Directory to save the figure.
        """
        raw_data = self.interpolator.loader.read_data(file_path)  # Use the loader from the interpolator
        int_data = self.interpolator.get_interpolation(raw_data, bodyparts)  # Interpolate the data
        self.plot_bodyparts(raw_data, int_data, bodyparts, title, output_dir)

    def plot_directory(self, directory_path, bodyparts, title, output_dir):
        """
        Plot comparisons for all files in a directory and save the figures.

        Args:
            directory_path (str): Path to the directory.
            bodyparts (list): List of body parts to plot.
            title (str): Title of the figure.
            output_dir (str): Directory to save the figures.
        """
        data_dict = self.interpolator.loader.read_directory(directory_path)  # Use the loader from the interpolator
        for filename, raw_data in data_dict.items():
            print(f"Processing file: {filename}")
            int_data = self.interpolator.get_interpolation(raw_data, bodyparts)
            self.plot_bodyparts(raw_data, int_data, bodyparts, f"{title} - {filename}", output_dir)

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

    def density_plot(self, 
                 bodypart, 
                 cmap="viridis", 
                 cbar=True, 
                 title="", 
                 width=11.34, 
                 height=3.35, 
    ):
        """
        Generate a 2D kernel density estimate (KDE) plot for the provided data points.

        Args:
            bodypart (pandas.DataFrame): A dataframe containing columns 'x' and 'y'.
            cmap (str, optional): Colormap for the KDE plot. Defaults to 'viridis'.
            cbar (bool, optional): Whether to display a colorbar. Defaults to True.
            title (str, optional): Title for the plot. Defaults to an empty string.
            width (int, optional): Width of the plot in inches. Defaults to 10.
            height (int, optional): Height of the plot in inches. Defaults to 10.
            xlim (tuple, optional): X-axis limits. Defaults to (0, 1000).
            ylim (tuple, optional): Y-axis limits. Defaults to (0, 300).

        Returns:
            None: Displays the KDE plot.
        """
        plt.figure(figsize=(width, height))

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
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

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
        cmap='viridis',
        width=10,
        height=10,
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
            self.density_plot(centroid_df, title=f"{full_title}", cmap = cmap, width=width, height=height)
        elif plot_type == "line":
            self.line_plot(centroid_df, title=f"{full_title}")

        if save:
            save_path = Path(save_directory) / save_filename
            Path(save_directory).mkdir(
                parents=True, exist_ok=True
            )  # Create directory if it does not exist
            plt.savefig(save_path, dpi=300, format="svg")
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
        cmap='viridis',
        width=10,
        height=10,
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
                save_filename=name + '.svg',
                cmap=cmap,
                width=width,
                height=width,
            )

        
