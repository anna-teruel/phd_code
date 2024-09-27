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
from minian.preprocessing import remove_background
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
    varr_min = video_data.min("frame").compute()
    video_data = video_data - varr_min
    
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

def check_nans_and_zeros(data, name):
    has_nans = np.isnan(data).any().compute()
    all_zeros = np.all(data == 0).compute()
    print(f"Array {name} has NaNs: {has_nans}")
    print(f"All values in array {name} are zeros: {all_zeros}")


def run_cnmf(Y_hw_chk, 
             Y_fm_chk, 
             A, 
             C, 
             C_chk, 
             param_get_noise, 
             param_first_spatial, 
             param_first_temporal, 
             param_first_merge, 
             param_second_spatial, 
             param_second_temporal, 
             intpath, 
             chk, 
             interactive=False):
    """
    This function performs the CNMF (Constrained Nonnegative Matrix Factorization) algorithm on the given data.
    Minian uses CNMF with CVXPY as deconvolution backend. 
    Args:
        Y_hw_chk (xarray.DataArray): Input data in the form of a 3D array (height x width x time).
        Y_fm_chk (xarray.DataArray): Input data in the form of a 3D array (frame x height x width).
        A (xarray.DataArray): Initial spatial footprints of the identified components.
        C (xarray.DataArray): Initial temporal components.
        C_chk (xarray.DataArray): Checkpoint for temporal components.
        param_get_noise (dict): Parameters for the noise estimation function.
        param_first_spatial (dict): Parameters for the first spatial update function.
        param_first_temporal (dict): Parameters for the first temporal update function.
        param_first_merge (dict): Parameters for the first merge function.
        param_second_spatial (dict): Parameters for the second spatial update function.
        param_second_temporal (dict): Parameters for the second temporal update function.
        intpath (str): Path to the directory where intermediate results will be saved.
        chk (dict): Dictionary containing chunking information for the data.
        interactive (bool, optional): If True, interactive plots will be displayed during the process. Defaults to False.
    Returns:
        tuple: Returns a tuple containing updated spatial footprints (A), temporal components (C), checkpoint for temporal components (C_chk), background spatial component (b), background temporal component (f), spike trains (S), baseline fluorescence for each neuron (b0), initial concentration for each neuron (c0), and noise level for each pixel (sig).
    """     
    print(f'Processing directory: {intpath}')
    # 1. Estimate spatial noise
    print("Estimate spatial noise")
    sn_spatial = get_noise_fft(Y_hw_chk, **param_get_noise)
    check_nans_and_zeros(sn_spatial, "sn_spatial")
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)
    # 2. First spatial update
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **param_first_spatial)
    check_nans_and_zeros(A_new, "A_new")
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
    check_nans_and_zeros(C_new, "C_new")
    check_nans_and_zeros(C_chk_new, "C_chk_new")
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    print("Saving results from spatial update")
    A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1})
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
    
    check_nans_and_zeros(A, "A")
    check_nans_and_zeros(C, "C")
    check_nans_and_zeros(b, "b")
    check_nans_and_zeros(f, "f")
    check_nans_and_zeros(C_chk, "C_chk")
    
    # 3. First temporal update
    print("First temporal update")
    YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1})
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **param_first_temporal)
    
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]})
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    A = A.sel(unit_id=C.coords["unit_id"].values)
    check_nans_and_zeros(A, "A")
    check_nans_and_zeros(C, "C")
    check_nans_and_zeros(S, "S")
    check_nans_and_zeros(b0, "b0")
    check_nans_and_zeros(c0, "c0")
    check_nans_and_zeros(C_chk, "C_chk")
    # Merge units 
    try: 
        A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **param_first_merge)
    except Exception as e:
        print(f"Failed to merge units due to error: {e}")
        A_mrg = A 
        C_mrg = C
    finally:
        A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
        C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_mrg_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
        check_nans_and_zeros(A, "A")
        check_nans_and_zeros(C, "C")
        check_nans_and_zeros(C_chk, "C_chk")
    # 4, Second spatial update
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **param_second_spatial)
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1})
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
    check_nans_and_zeros(A, "A")
    check_nans_and_zeros(C, "C")
    check_nans_and_zeros(b, "b")
    check_nans_and_zeros(f, "f")
    check_nans_and_zeros(C_chk, "C_chk")
    # 5. Second temporal update
    YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1})
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **param_second_temporal)
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    A = A.sel(unit_id=C.coords["unit_id"].values)
    check_nans_and_zeros(A, "A")
    check_nans_and_zeros(C, "C")
    check_nans_and_zeros(S, "S")
    check_nans_and_zeros(b0, "b0")
    check_nans_and_zeros(c0, "c0")
    check_nans_and_zeros(C_chk, "C_chk")
    return A, C, C_chk, b, f, S, b0, c0