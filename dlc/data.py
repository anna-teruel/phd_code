import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection



def get_centroid(data_dict, bodyparts):
    """
    Calculate the centroid of specified bodyparts in each DataFrame in the dictionary.

    Args:
        data_dict (dict): A dictionary where keys are identifiers and values are DataFrames of bodypart coordinates.
        bodyparts (list): A list of bodyparts to include in the centroid calculation.

    Returns:
        centroids (dict): A dictionary with the same keys, where values are the centroids of the specified bodyparts.
    """
    centroids = {}
    for key, df in data_dict.items():
        x_columns = [bp for bp in bodyparts]
        y_columns = [bp for bp in bodyparts]

        centroid_x = np.mean(df.loc[:, (slice(None), x_columns, 'x')].mean(axis=1))
        centroid_y = np.mean(df.loc[:, (slice(None), y_columns, 'y')].mean(axis=1))

        DLCscorer = df.columns[0][0]

        df.loc[:, (DLCscorer,"centroid", "x")] = centroid_x
        df.loc[:, (DLCscorer, "centroid", "y")] = centroid_y

        centroids[key] = df

    return centroids


def get_interpolation(data_dict, bodyparts, threshold=0.95, interpolation_method='linear'):
    """
    Interpolate x,y coordinates for specified body parts in a DataFrame when their likelihood 
    is below a given threshold.

    Args:
        data_dict (dict): A dictionary where keys are identifiers and values are DataFrames of bodypart coordinates.
        bodyparts (list): A list of bodyparts to interpolate.
        threshold (float): Likelihood threshold below which interpolation should be performed. Defaults to 0.95.
        interpolation_method (string): Interpolation method used. Default linear interpolation.
    Returns:
        dict: Dictionary with DataFrames that have interpolated values for the specified body parts.
    """   
    for key, df in data_dict.items():
        for bp in bodyparts:
            if bp in df.columns.get_level_values('bodyparts').unique():
            
                mask = df.loc[:, (slice(None), bp, 'likelihood')] < threshold

                df.loc[mask, (bp, 'x')] = df[bp]['x'].interpolate(method=interpolation_method)
                df.loc[mask, (bp, 'y')] = df[bp]['y'].interpolate(method=interpolation_method)

        data_dict[key] = df
    return data_dict



