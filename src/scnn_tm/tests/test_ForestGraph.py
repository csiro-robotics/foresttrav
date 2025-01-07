# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

import time
from pathlib import Path

import numpy as np
import pandas as pd

import scnn_tm.utils as utils
from scnn_tm.models.ForestTravGraph import ForestTravGraph, voxel_wise_submap_fusion
from scnn_tm.models.OnlineForestTrav import (
    OnlineDataSetLoader,
)

from copy import deepcopy

# Test the unique fusion of two submaps based on the ordering
def test_seperate_submap_fusion():
    """Test the submap fusion.
    Given to non-overlappping pcl
    - All the coords should be maintained
    - All the features should be maintained
    - All elements should be unique
    """
    bounds_0 = [-1.0, 0, -1.0, 0, 1.0,0.0]
    bounds_1 = [ 0.0, 0, -1.0, 1.0, 1.0,0.0]
    voxel_size = 0.1
    
    cloud_0 = generate_valid_pcl(voxel_size=voxel_size, bounds=bounds_0, feature_offset=0.0)
    cloud_1 = generate_valid_pcl(voxel_size=voxel_size, bounds=bounds_1, feature_offset=1.0)
    
    cloud_0_org = deepcopy(cloud_0)
    
    fused_map = voxel_wise_submap_fusion([cloud_0, cloud_1], voxel_size=voxel_size)
    
    # Test shape is the same and all elements are the sampe

    assert np.allclose(fused_map, np.vstack([cloud_0, cloud_1]))
    assert np.allclose(cloud_0,cloud_0_org)
    
    # Test unique
    point_ids = utils.points_to_voxel_ids(fused_map[:, :3], voxel_size)
    uniqeu_rows, _ = np.unique(point_ids, axis=0, return_index=True)
    assert len(uniqeu_rows) == fused_map.shape[0]

def test_overlapping_submap_fusion():
    """Test the submap fusion.
    Given to non-overlappping pcl
    - All the coords should be maintained
    - All the features should be maintained
    - All elements should be unique
    """
    
    bounds_0 = [-1.0, 0, -1.0, 0, 1.0,0.0]
    bounds_1 = [ 0.0, 0, -1.0, 1.0, 1.0,0.0]
    bounds_2 = [ -0.5, 0, -1.0, 0.5, 1.0,0.0] #The overlapping bounds
    voxel_size = 0.1
    
    cloud_0 = generate_valid_pcl(voxel_size=voxel_size, bounds=bounds_0, feature_offset=0.0)
    cloud_1 = generate_valid_pcl(voxel_size=voxel_size, bounds=bounds_1, feature_offset=1.0)
    cloud_2 = generate_valid_pcl(voxel_size=voxel_size, bounds=bounds_2, feature_offset=2.0)
    cloud_0_org = deepcopy(cloud_0)
    
    fused_map = voxel_wise_submap_fusion([cloud_0, cloud_1, cloud_2], voxel_size=voxel_size)
    
    # Test shape is the same and all elements are the sampe
    assert np.allclose(fused_map, np.vstack([cloud_0, cloud_1]))
    assert np.allclose(cloud_0,cloud_0_org)
    
    # Test unique
    point_ids = utils.points_to_voxel_ids(fused_map[:, :3], voxel_size)
    uniqeu_rows, _ = np.unique(point_ids, axis=0, return_index=True)
    assert len(uniqeu_rows) == fused_map.shape[0]

def generate_valid_pcl(voxel_size, bounds, feature_offset):

    # Generate a plane with random z values
    coord = np.array(
        [
            [x, y, z]
            for x in np.arange(bounds[0]+voxel_size/2.0, bounds[3], voxel_size)
            for y in np.arange(bounds[1]+voxel_size/2.0, bounds[4], voxel_size)
            for z in np.arange(bounds[2]+voxel_size/2.0, bounds[5], voxel_size)
        ]
    )

    # Generate random features with feature_offset
    feature_cloud = np.random.random((coord.shape[0], 7))  + feature_offset
    
    return np.hstack([coord, feature_cloud])
    

# Test the collision map     

if __name__ == "__main__":
    test_seperate_submap_fusion()
    test_overlapping_submap_fusion()
