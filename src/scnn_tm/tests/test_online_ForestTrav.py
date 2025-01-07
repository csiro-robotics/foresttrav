# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

from pathlib import Path

import h5py
import sklearn.model_selection

from scnn_tm.models.OnlineForestTrav import OnlineDataSetLoader

HDF5_DEBUG_FILE = Path("/data/debug/test_data_fuser/QCATF_FOREST_1.hdf5")

def test_data_parsing():
    
    data_loader  = OnlineDataSetLoader(voxel_size=0.1, target_feature_set=[ "mean_count"], min_pose_dist=0.5)
    data_loader.load_online_data_set_raw(HDF5_DEBUG_FILE, 0)
    a =1 


test_data_parsing()
