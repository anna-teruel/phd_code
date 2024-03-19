"""
Loading data functions from .h5 files.
@author Anna Teruel-Sanchis, 2023
"""

import os
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, minutes=None, fps=None):
        """
        Initialize a DataLoader object.

        Args:
            minutes (int, optional): The duration of the video in minutes. Defaults to None.
            fps (int, optional): The frames per second (fps) of the video. Defaults to None.
        """
        self.minutes = minutes
        self.fps = fps

    def read_data(self, input_path):
        """
        Read data from either a single file or a directory.

        Args:
            input_path (str): The path to the file or directory containing the data.

        Raises:
            ValueError: If the provided path does not exist.

        Returns:
            pandas.DataFrame or dict: A DataFrame if a single file is read, or a dictionary of
                                      DataFrames if multiple files are read.
        """
        if os.path.isfile(input_path):  # Check if it's a file
            return self.read_file(input_path)

        elif os.path.isdir(input_path):  # Check if it's a directory
            return self.read_directory(input_path)

        else:
            raise ValueError("Provided path does not exist.")

    def read_file(self, file_path):
        """
        Read data from a single file.

        Args:
            file_path (str): The path to the .h5 file.

        Returns:
            pandas.DataFrame: A DataFrame containing the data from the file.
        """
        df = pd.read_hdf(file_path)
        return df

    def read_directory(self, dir_path, suffix=("filtered.h5",)):
        """
        Reads data from all files in a directory that end with a specified suffix.

        Args:
            directory_path (str): The path to the directory containing the .h5 files.
            suffix (tuple, optional): The suffix that the files should end with. Defaults to ('filtered.h5',).

        Returns:
            dict: A dictionary where keys are file names and values are DataFrames containing the data from each file.
        """
        data_dict = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(suffix):
                file_path = os.path.join(dir_path, filename)
                data_dict[filename] = self.read_file(file_path)
        return data_dict
    
    def get_file_name(self, file_path): 
        """_summary_

        Args:
            file_path (str): The path to the .h5 file.

        Returns:
            _type_: _description_
        """        
        file_name_long = Path(file_path).name  # complete file name with extension
        dlc_index = file_name_long.find(
            "DLC"
        )  # all deeplabcut files include project info starts with "DLC"
        file_name = (
            file_name_long[:dlc_index] if dlc_index != -1 else Path(file_path).stem
        )  # remove extra info from title
        return file_name
