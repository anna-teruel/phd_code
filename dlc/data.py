import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dlc.load_data import DataLoader

class Centroid:
    def __init__(self, minutes=None, fps=None):
        self.minutes = minutes
        self.fps = fps

    def calculate_centroid(self, df, bodyparts):
        """
        Calculate the centroid for a given DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing bodypart coordinates.
            bodyparts (list): List of bodyparts to include in the centroid calculation.

        Returns:
            pd.DataFrame: DataFrame with added centroid coordinates.
        """
        x_columns = [bp for bp in bodyparts]
        y_columns = [bp for bp in bodyparts]

        centroid_x = np.mean(df.loc[:, (slice(None), x_columns, 'x')].mean(axis=1))
        centroid_y = np.mean(df.loc[:, (slice(None), y_columns, 'y')].mean(axis=1))

        DLCscorer = df.columns[0][0]

        df.loc[:, (DLCscorer, "centroid", "x")] = centroid_x
        df.loc[:, (DLCscorer, "centroid", "y")] = centroid_y
        return df

    def get_centroid(self, input_data, bodyparts):
        """
        Get centroid for single or multiple DataFrames.

        Args:
            input_data (str): Path to .h5 file/s.
            bodyparts (list): List of bodyparts for centroid calculation.

        Returns:
            dict or pd.DataFrame: Dictionary with centroids or single DataFrame with centroid.
        """
        if isinstance(input_data, str):
            if os.path.isfile(input_data):
                # If it's a file; read the single file
                loader = DataLoader(input_data, self.minutes, self.fps)
                df = loader.read_data(input_data)
                return self.calculate_centroid(df, bodyparts)
            elif os.path.isdir(input_data):
                # If it's a directory; read all files in the directory
                loader = DataLoader(input_data, self.minutes, self.fps)
                data_dict = loader.read_all_data()
                return {key: self.calculate_centroid(df, bodyparts) for key, df in data_dict.items()}
            else:
                raise ValueError("Provided path does not exist.")
        else:
            raise ValueError("Input data must be a file path or directory path")



class Interpolation:
    def __init__(self, threshold=0.90, interpolation_method='linear'):
        """
        Initialize an Interpolator object with specified threshold and interpolation method.

        The Interpolator class is designed to interpolate x, y coordinates of specified body parts 
        in a DataFrame or a collection of DataFrames. The interpolation is applied where the likelihood 
        of body part coordinates is below a given threshold.

        Args:
            threshold (float, optional): The likelihood threshold below which interpolation should be performed.
                                         Coordinates with a likelihood value lower than this threshold will be interpolated.
                                         Defaults to 0.90.
            
            interpolation_method (str, optional): The method of interpolation to use. It can be any method supported
                                                  by pandas.DataFrame.interpolate. Common methods include:
                                                  - 'linear': Linear interpolation; default method that usually works well.
                                                  - 'time': Interpolation over time (useful if your data is time-indexed).
                                                  - 'index', 'values': Use the actual numerical values of the index.
                                                  - 'pad', 'ffill': Forward fill.
                                                  - 'bfill', 'backfill': Backward fill.
                                                  - 'nearest': Nearest interpolation.
                                                  More advanced methods like 'polynomial' and 'spline' require additional 
                                                  parameters. Defaults to 'linear'.
        """        
        self.threshold = threshold
        self.interpolation_method = interpolation_method

    def get_interpolation(self, df, bodyparts):
        """
        Interpolate x,y coordinates for specified body parts in a single DataFrame.

        Args:
            df (pd.DataFrame): DataFrame of bodypart coordinates.
            bodyparts (list): A list of bodyparts to interpolate.

        Returns:
            pd.DataFrame: DataFrame with interpolated values for the specified body parts.
        """
        scorer = df.columns.get_level_values('scorer')[0]  # Assuming the scorer is consistent
        for bp in bodyparts:
            if bp in df.columns.get_level_values('bodyparts').unique():
                mask = df.loc[:, (scorer, bp, 'likelihood')] < self.threshold
                df.loc[mask, (scorer, bp, 'x')] = df[(scorer, bp, 'x')].interpolate(method=self.interpolation_method)
                df.loc[mask, (scorer, bp, 'y')] = df[(scorer, bp, 'y')].interpolate(method=self.interpolation_method)
            else:
                print(f"{bp} bodypart is not found in the DataFrame.")
        return df

    def interpolate_data(self, input_data, bodyparts):
        """
        Interpolate data for either a single DataFrame or a dictionary of DataFrames.

        Args:
            input_data (str): Path to .h5 file/s.
            bodyparts (list): List of bodyparts for interpolation.

        Returns:
            pd.DataFrame or dict: Interpolated DataFrame or dictionary of DataFrames.
        """
        if isinstance(input_data, str):
            if os.path.isfile(input_data):
                # If it's a file; read the single file
                loader = DataLoader(h5_path=os.path.dirname(input_data))
                df = loader.read_data(input_data)
                return self.get_interpolation(df, bodyparts)
            elif os.path.isdir(input_data):
                # If it's a directory; read all files in the directory
                loader = DataLoader(h5_path=input_data)
                data_dict = loader.read_all_data()
                return {key: self.get_interpolation(df, bodyparts) for key, df in data_dict.items()}
            else:
                raise ValueError("Provided path does not exist.")

        else:
            raise ValueError("Input data must be a file path or directory path")




