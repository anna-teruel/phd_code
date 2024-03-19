"""
This script merges multiple '.h5' files from each session's DeepLabCut subdirectory within 
a given root directory into a single DataFrame per session. Data from minicam are divided
in different subdirectories, and thus we need to merge the files to have a single DataFrame.
@author Anna Teruel-Sanchis, 2023
"""

import os
import pandas as pd

def merge_h5_files(h5_dir):
    """
    Merge all 'filtered.h5' files within each session's DeepLabCut subdirectory 
    in the given root directory.

    Minicam recordings save a video file every 1000 frames. That is why we 
    have multiple files we need to merge. 

    Args:
        h5_dir (str): Path to the root directory containing session directories.
                      Each session directory should include a subdirectory, named
                      "dlc" that contains the multiple .h5 files to be merged. 

    Returns:
        dict: Dictionary where keys are session directory names and values are merged DataFrames.

    """
    merged_dfs = {} 
    for session_dir in os.listdir(h5_dir):
        session_path = os.path.join(h5_dir, session_dir)

        if os.path.isdir(session_path):
            dlc_output_dir = os.path.join(session_path, 'dlc')

            if os.path.exists(dlc_output_dir) and os.path.isdir(dlc_output_dir):
                dataframes = []

                for h5_file in os.listdir(dlc_output_dir):
                    if h5_file.startswith("._"):
                        file_path = os.path.join(dlc_output_dir, h5_file)
                        os.remove(file_path)
                    elif h5_file.endswith('filtered.h5'):
                        h5_file_path = os.path.join(dlc_output_dir, h5_file)

                        df = pd.read_hdf(h5_file_path)
                        dataframes.append(df)

                if dataframes:
                    merged_df = pd.concat(dataframes, axis=0)
                    merged_dfs[session_dir] = merged_df
                    
                    print(f"Merged data for session '{session_dir}'")
    
    return merged_dfs