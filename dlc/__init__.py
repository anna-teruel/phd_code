
"""
This code is written to analyse output data from DeepLabCut
"""
#init file

import dlc.data as data
import dlc.plotting as plotting
import dlc.load_data as load_data
import dlc.analysis.kinematics as kinematics
import dlc.analysis.time_roi as time_roin
import dlc.minicam_data as minicam_data

__all__ = ["data", "plotting", "load_data", "kinematics", "time_roi"]
