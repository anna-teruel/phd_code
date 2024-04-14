"""
This code is a modified version of the original minian pipeline notebook from the minian package. It is modified 
and defined for the particular use case of our dCA1 data. The original code and package can be found here: 
https://github.com/denisecailab/minian
"""

minian_path = "/Users/annateruel/minian"

import itertools as itt
import os
import sys

import holoviews as hv
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from holoviews.operation.datashader import datashade, regrid
from holoviews.util import Dynamic
from IPython.core.display import display

sys.path.append(minian_path)
from minian.cnmf import (
    compute_AtC,
    compute_trace,
    get_noise_fft,
    smooth_sig,
    unit_merge,
    update_spatial,
    update_temporal,
    update_background,
)
from minian.initialization import (
    gmm_refine,
    initA,
    initC,
    intensity_refine,
    ks_refine,
    pnr_refine,
    seeds_init,
    seeds_merge,
)
from minian.motion_correction import apply_transform, estimate_motion
from minian.preprocessing import denoise, remove_background
from minian.preprocessing import glow_removal, remove_background
from minian.utilities import (
    TaskAnnotation,
    get_optimal_chk,
    load_videos,
    open_minian,
    save_minian,
)
from minian.visualization import (
    CNMFViewer,
    VArrayViewer,
    generate_videos,
    visualize_gmm_fit,
    visualize_motion,
    visualize_preprocess,
    visualize_seeds,
    visualize_spatial_update,
    visualize_temporal_update,
    write_video,
)

def preprocessing(intpath, dpath, 
                  param_load_videos={"pattern": "[0-10]+\.avi$", "dtype": np.uint8, "downsample": dict(frame=1, height=1, width=1), "downsample_strategy": "subset"}, 
                  param_denoise={"method": "median", "ksize": 4}, 
                  param_background_removal={"method": "tophat", "wnd": 4}):
    
    # Load video data
    video_data = load_videos(dpath, **param_load_videos)
    
    # Get optimal chunk size
    chk, _ = get_optimal_chk(video_data, dtype=float)
    
    # Apply glow_removal
    video_data = glow_removal(video_data)
    
    # Apply denoise
    video_data = denoise(video_data, **param_denoise)
    
    # Apply background_removal
    video_data = remove_background(video_data, **param_background_removal)
    
    # Save the processed video data
    save_minian(video_data.rename("varr_ref"), dpath=intpath, overwrite=True)

# # Pre-processing Parameters#
# param_load_videos = {
#     "pattern": "[0-10]+\.avi$",
#     "dtype": np.uint8,
#     "downsample": dict(frame=1, height=1, width=1),
#     "downsample_strategy": "subset",
# }
# param_denoise = {"method": "median", "ksize": 4}
# param_background_removal = {"method": "tophat", "wnd": 4}

# # Motion Correction Parameters#
# subset_mc = None
# param_estimate_motion = {"dim": "frame"}

# # Initialization Parameters#
# param_seeds_init = {
#     "wnd_size": 1000,
#     "method": "rolling",
#     "stp_size": 500,
#     "max_wnd": 15,
#     "diff_thres": 3,
# }
# param_pnr_refine = {"noise_freq": 0.06, "thres": 1}
# param_ks_refine = {"sig": 0.05}
# param_seeds_merge = {"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.06}
# param_initialize = {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06}
# param_init_merge = {"thres_corr": 0.8}

# # CNMF Parameters#
# param_get_noise = {"noise_range": (0.06, 0.5)}
# param_first_spatial = {
#     "dl_wnd": 10,
#     "sparse_penal": 0.01,
#     "size_thres": (25, None),
# }
# param_first_temporal = {
#     "noise_freq": 0.06,
#     "sparse_penal": 1,
#     "p": 1,
#     "add_lag": 20,
#     "jac_thres": 0.2,
# }
# param_first_merge = {"thres_corr": 0.8}
# param_second_spatial = {
#     "dl_wnd": 10,
#     "sparse_penal": 0.01,
#     "size_thres": (25, None),
# }
# param_second_temporal = {
#     "noise_freq": 0.06,
#     "sparse_penal": 1,
#     "p": 1,
#     "add_lag": 20,
#     "jac_thres": 0.4,
# }

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MINIAN_INTERMEDIATE"] = intpath


