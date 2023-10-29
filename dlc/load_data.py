import os
import os.path
import pandas as pd

def get_h5List(h5_path):
    """
    This code is used to get list of .h5 files to be processed within a directory

    Args:
        h5_path (str): path to .h5 files to be processed

    Returns:
        h5List (list): list of paths of .h5 files to be processed
    """    
    h5_path = os.chdir(h5_path)
    h5List = []
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith("filtered.h5")]:
            h5List.append(os.path.join(dirpath, filename))
    h5List.sort()
    return h5List

def read_data(h5_file, minutes, fps):
    """
    Reads data from a specified .h5 file for a given duration.

    Args:
        h5_file (str): Path to the .h5 file to be read.
        minutes (int): Number of minutes of data to be read from the file.
        fps (int): Frames per second of the data in the file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the specified duration of data from the .h5 file.
    """   
    df = pd.read_hdf(h5_file)

    frames = fps * 60 * minutes  
    df = df[:frames]
    return df

def read_all_data(h5_path, minutes, fps):
    """
    Reads data from all .h5 files in the provided path for batch processing

    Args:
        h5_path (str): Path to .h5 files to be processed.
        minutes (int): Number of minutes to be read from each file.
        fps (int): Frames per second of the data.

    Returns:
        dict: A dictionary with file names as keys and dataframes as values.
    """
    h5_list = get_h5List(h5_path)
    data_dict = {}

    for h5_file in h5_list:
        df = read_data(h5_file, minutes, fps)
        file_name = os.path.basename(h5_file)
        data_dict[file_name] = df

    return data_dict

