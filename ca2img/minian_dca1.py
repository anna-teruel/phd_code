"""
This code is a modified version of the original minian pipeline notebook from the minian package. It is modified 
and defined for the particular use case of our data. The original code and package can be found here: 
https://github.com/denisecailab/minian
"""

minian_path = "/Users/annateruel/minian"

import itertools as itt
import os
import sys
from time import time

import holoviews as hv
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from holoviews.operation.datashader import datashade, regrid
from holoviews.util import Dynamic
from IPython.core.display import display
import zarr

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

def start_cluster(n_workers=8, 
                    memory_limit="3GB", 
                    resources={"MEM": 1}, 
                    threads_per_worker=2, 
                    dashboard_address=":8787"):
    """
    Starts a Dask LocalCluster and returns a Client connected to this cluster.

    Args:
        n_workers (int, optional): The number of workers in the cluster. Defaults to 8.
        memory_limit (str, optional): The maximum amount of memory that each worker can use. Defaults to "3GB".
        resources (dict, optional): Resources that each worker provides. Defaults to {"MEM": 1}.
        threads_per_worker (int, optional): The number of threads that each worker can use. Defaults to 2.
        dashboard_address (str, optional): The address where the Dask dashboard will be hosted. Defaults to ":8787".

    Returns:
        dask.distributed.Client: A client connected to the cluster.
    """       
    
    cluster = LocalCluster(
        n_workers=n_workers, 
        memory_limit=memory_limit,
        resources=resources,
        threads_per_worker=threads_per_worker,
        dashboard_address=dashboard_address,
    )
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)
    return client


def preprocessing(intpath, dpath, 
                  param_load_videos={"pattern": "[0-10]+\.avi$", "dtype": np.uint8, "downsample": dict(frame=1, height=1, width=1), "downsample_strategy": "subset"}, 
                  param_denoise={"method": "median", "ksize": 4}, 
                  param_background_removal={"method": "tophat", "wnd": 4}):
    """
    Preprocesses video data by loading the videos, removing glow, denoising, and removing the background.

    Args:
        intpath (str): The path to the directory where the processed video data will be saved.
        dpath (str): The path to the directory where the raw video data is stored.
        param_load_videos (dict, optional): Parameters for loading the videos. Defaults to {"pattern": "[0-10]+\.avi$", "dtype": np.uint8, "downsample": dict(frame=1, height=1, width=1), "downsample_strategy": "subset"}.
        param_denoise (dict, optional): Parameters for the denoising process. Defaults to {"method": "median", "ksize": 4}.
        param_background_removal (dict, optional): Parameters for the background removal process. Defaults to {"method": "tophat", "wnd": 4}.
    """                 
    zarr_path = os.path.join(intpath, 'varr_ref.zarr')
    zarr.open(zarr_path, mode='w')
    #print(f'Checking if {zarr_path} exists...')
    # try:
    #     zarr.open(zarr_path, mode='w')
    #     print('varr_ref.zarr already exists, overwriting')
    #     return
    # except ValueError:
    #     print('varr_ref.zarr does not exist, proceeding with process.')

    # Load video data
    video_data = load_videos(dpath, **param_load_videos)
    print(f'Number of videos loaded: {len(video_data)}')
    
    # Set chunk size
    chk, _ = get_optimal_chk(video_data, dtype=float)
    
    # Glow removal
    video_data = glow_removal(video_data)
    
    # Denoise
    video_data = denoise(video_data, **param_denoise)
    
    # Background removal
    video_data = remove_background(video_data, **param_background_removal)

    save_minian(video_data.rename("varr_ref"), dpath=intpath, overwrite=True)


def motion_correction(int_dir, varr_ref, param_estimate_motion, param_save_minian, chk):
    """
    Performs motion correction on a given video array reference.

    Args:
        int_dir (str): The directory where the intermediate results will be saved.
        varr_ref (xarray.DataArray): The video array reference to be corrected.
        param_estimate_motion (dict): Parameters for the `estimate_motion` function.
        param_save_minian (dict): Parameters for the `save_minian` function.
        chk (dict): Chunk sizes for different dimensions.

    Returns:
        None. The function saves the motion corrected video array and the motion estimation to disk.
    """
    motion = estimate_motion(varr_ref, **param_estimate_motion)
    motion = save_minian(
        motion.rename("motion").chunk({"frame": chk["frame"]}), **param_save_minian
    )

    Y = apply_transform(varr_ref, motion, fill=0)
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), int_dir, overwrite=True)
    Y_hw_chk = save_minian(
        Y_fm_chk.rename("Y_hw_chk"),
        int_dir,
        overwrite=True,
        chunks={"frame": -1, "height": chk["height"], "width": chk["width"]})


def initialize(Y_fm_chk, 
               Y_hw_chk, 
               param_save_minian, 
               param_seeds_init, 
               param_pnr_refine, 
               param_ks_refine, 
               param_seeds_merge, 
               param_initialize, 
               param_init_merge, 
               intpath, 
               chk):
    """
    Performs initialization process including max projection, peak-to-noise ratio refinement, 
    Kolmogorov-Smirnov test, merging seeds, initializing spatial and temporal matrices, 
    and updating background.

    Args:
        Y_fm_chk (xarray.DataArray): The motion corrected video array.
        Y_hw_chk (xarray.DataArray): The motion corrected video array with different chunking.
        param_save_minian (dict): Parameters for the `save_minian` function.
        param_seeds_init (dict): Parameters for the `seeds_init` function.
        param_pnr_refine (dict): Parameters for the `pnr_refine` function.
        param_ks_refine (dict): Parameters for the `ks_refine` function.
        param_seeds_merge (dict): Parameters for the `seeds_merge` function.
        param_initialize (dict): Parameters for the `initA` and `initC` functions.
        param_init_merge (dict): Parameters for the `unit_merge` function.
        intpath (str): The directory where the intermediate results will be saved.
        chk (dict): Chunk sizes for different dimensions.

    Returns:
        None. The function saves the results to disk.
    """
    print(f'Processing directory: {intpath}')

    print('Starting max projection')
    max_proj = save_minian(
        Y_fm_chk.max("frame").rename("max_proj"), **param_save_minian
    ).compute()

    print('Initializing seeds')
    seeds = seeds_init(Y_fm_chk, **param_seeds_init)
    print('Number of seeds detected is ' + str(len(seeds)))

    print('Refining seeds with PNR')
    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param_pnr_refine)

    print('Refining seeds with KS test')
    seeds = ks_refine(Y_hw_chk, seeds, **param_ks_refine)

    print('Merging seeds')
    seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
    print('Number of seeds_final detected is ' + str(len(seeds_final)))
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param_seeds_merge)
    print('Number of seeds_final detected is ' + str(len(seeds_final)))

    print('Initializing spatial matrix A')
    A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param_initialize)
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)
    print("Does A_init contain NaN values?", A_init.isnull().any().compute())

    print('Initializing temporal matrix C')
    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(
        C_init.rename("C_init"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1}
    )
    print("Does C_init contain NaN values?", C_init.isnull().any().compute())


    if not (A_init == 0).all():
        print('Merging units in A and C')
        try:
            A, C = unit_merge(A_init, C_init, **param_init_merge)
            assert not np.isnan(A.values).any(), "A contains NaN values"
            assert not np.isnan(C.values).any(), "C contains NaN values"

            A = save_minian(A.rename("A"), intpath, overwrite=True)
            C = save_minian(C.rename("C"), intpath, overwrite=True)
            print("Does A contain NaN values?", A.isnull().any().compute())
            print("Does C contain NaN values?", C.isnull().any().compute())
        except ValueError as e:
            print(f"Error during unit_merge: {e}")
            print("Skipping unit merge due to empty inputs.")
            A = A_init
            C = C_init
    else:
        print('No overlap between units, skipping unit_merge')
        A = A_init
        C = C_init
        A = save_minian(A.rename("A"), intpath, overwrite=True)
        C = save_minian(C.rename("C"), intpath, overwrite=True)
        print("Does A contain NaN values?", A.isnull().any().compute())
        print("Does C contain NaN values?", C.isnull().any().compute())

    print('Saving C_chk')
    print(f"chk type: {type(chk)}, value: {chk}")
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk[0]["frame"]},
    )

    print('Updating background')
    b, f = update_background(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)