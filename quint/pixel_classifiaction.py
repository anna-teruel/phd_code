"""
This code uses the ilastik API for pixel classification and gets probability maps
for all images organized in different subdirectories.
@author Anna Teruel-Sanchis, march 2024
"""

import imageio
import matplotlib.pyplot as plt
import h5py
import os
from pathlib import Path
from ilastik.experimental.api import PixelClassificationPipeline
import numpy
from xarray import DataArray
import imageio.v3 as iio
from matplotlib import pyplot as plt

def batch_probabilities(dir, project):
    """
    Process images in the given base directory using a pipeline initialized from the specified project file,
    and save the probabilities to HDF5 files with names ending in "_Probabilities.h5".

    Args:
        dir (str): Path to the directory containing subdirectories with images.
        project (str): Path to the .ilp project file for initializing the pipeline.
    """
    pipeline = PixelClassificationPipeline.from_ilp_file(project)
    
    for subdir, files in os.walk(dir):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(subdir, file)
                file_name = Path(file_path).stem
                h5_path = Path(file_path).with_name(f"{file_name}_Probabilities.h5")
                
                # Load the image and provide appropriate dimension names for an RGB picture
                image = DataArray(imageio.imread(file_path), dims=("y", "x", "c"))
                
                # Call the pipeline to get the probabilities for the image and save them to an HDF5 file
                prediction = pipeline.get_probabilities(image)
                data_to_save = prediction.values
                
                with h5py.File(h5_path, 'w') as hdf:
                    hdf.create_dataset('probabilities', data=data_to_save)
                print(f"Probabilities saved to {h5_path}")
