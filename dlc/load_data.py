import os
import pandas as pd

class DataLoader:
    def __init__(self, h5_path=None, minutes=None, fps=None):
        """
        Initialize the data loader with the path to the .h5 files, the number of minutes to read, and the fps.

        Args:
            h5_path (str): Path to .h5 files to be processed.
            minutes (int): Number of minutes of data to be read from each file.
            fps (int): Frames per second of the data in the files.
        """
        self.h5_path = h5_path
        self.minutes = minutes
        self.fps = fps

    def read_data(self, h5_file):
        """
        Reads data from a specified .h5 file for a given duration.

        Args:
            h5_file (str): Path to the .h5 file to be read.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the specified duration of data from the .h5 file.
        """
        df = pd.read_hdf(h5_file)
        
        if self.minutes is not None and self.fps is not None:
            frames = self.fps * 60 * self.minutes  
            df = pd.read_hdf(h5_file)[:frames]
        return df

    def get_h5_list(self):
        """
        Get a list of .h5 files to be processed within a directory.

        Returns:
            h5List (list): List of paths of .h5 files to be processed.
        """
        h5_list = []
        for dirpath, dirnames, filenames in os.walk(self.h5_path):
            for filename in [f for f in filenames if f.endswith("filtered.h5")]:
                h5_list.append(os.path.join(dirpath, filename))
        h5_list.sort()
        return h5_list

    def read_all_data(self):
        """
        Reads data from all .h5 files in the provided path for batch processing.

        Returns:
            dict: A dictionary with file names as keys and dataframes as values.
        """
        h5_list = self.get_h5_list()
        data_dict = {}
        for h5_file in h5_list:
            df = self.read_data(h5_file)
            file_name = os.path.basename(h5_file)
            data_dict[file_name] = df
        return data_dict

