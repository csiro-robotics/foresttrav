# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

import numpy as np
import pandas as pd
import argparse
import yaml
import open3d as o3d

from scnn_tm import models
from scnn_tm import utils


DATA_SETS = utils.sweep_utils.data_set_by_key("lfe_hl")
FEATURE_SET = utils.scnn_io.generate_feature_set_from_key("occ_perm")
CONFIG_FILE = Path(__file__).parent / "config" / "default_test_configs.yaml"
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

##### CONFIG LOADING COMPLETE ############

def visualize_patches_for_a_scene():
    """ Visualizes the patches of a scene and colours them randomly. 

    """
    test_map = "/data/processed/scene_feature_sets/scenes/2021_12_14_00_14_53Z_full_map.csv"
    df = pd.read_csv(test_map)

    scene_coords = df[["x", "y", "z"]].to_numpy()
    i = 0
    voxel_size = 0.1
    nvoxel_leaf = 32
    map_bound_items = models.ForestTravDataSet.generate_map_bounds_for_static_map(
        scene_coords, map_id=i, voxel_size=voxel_size, nvoxel_leaf=nvoxel_leaf)

    cloud_coords = np.empty((0, 3))
    colour_prob = np.empty((0, 1))
    for map_id_pair in map_bound_items:

        cloud_mask = models.ForestTravDataSet.mask_for_points_in_patch(scene_coords, map_id_pair)
        new_cloud = scene_coords[cloud_mask]
        new_prob_cloud = np.ones((new_cloud.shape[0], 1)) * np.random.uniform(0, 1.0, size=(1, 1))

        colour_prob = np.vstack([colour_prob, new_prob_cloud])

        cloud_coords = np.vstack((cloud_coords, new_cloud))

    utils.visualize_probability(cloud_coords, colour_prob.squeeze())


def visualize_cv_patch_split_for_a_scene(data_sets, feature_set, params):

    params.current_fold = 3
    params.num_splits = len(data_sets)
    data_set_factory = models.ForestTravDataSet.PatchCvDataSetFactory(data_set=data_sets,
                                                                         feature_set=feature_set,
                                                                         params=params)

    train_dl = data_set_factory.train_data_set
    val_dl = data_set_factory.val_data_set
    test_dl = data_set_factory.test_data_set
    
    # Go over all the items and
    for map_id in range(len(data_sets)):
        train_coords = coords_for_data_loader_and_map_id(train_dl, map_id)
        val_coords = coords_for_data_loader_and_map_id(val_dl, map_id)
        test_coords = coords_for_data_loader_and_map_id(test_dl, map_id)

        cloud_train = o3d.geometry.PointCloud()
        cloud_train.points = o3d.utility.Vector3dVector(train_coords)
        cloud_train.paint_uniform_color([0, 1.0, 0])
        
        cloud_val = o3d.geometry.PointCloud()
        cloud_val.points = o3d.utility.Vector3dVector(val_coords)
        cloud_val.paint_uniform_color([0, 0.0, 1.0])
        
        cloud_test = o3d.geometry.PointCloud()
        cloud_test.points = o3d.utility.Vector3dVector(test_coords)
        cloud_test.paint_uniform_color([1, 0.0, 0])
        
        o3d.visualization.draw_geometries([cloud_train, cloud_val, cloud_test])
        

def coords_for_data_loader_and_map_id(data_loader, query_map_id, voxel_size=0.1):

    coords = np.empty((0, 3), dtype=np.float64)
    for k in range(len(data_loader)):

        # Check if it is the desired map_id and load data
        if data_loader.patches[k].map_id != query_map_id:
            continue

        batch = data_loader.__getitem__(k)
        batch_coords = batch["coordinates"] * voxel_size
        coords = np.vstack([coords, batch_coords])

    return coords

def check_test_fold_coverage():
    """ Visualises the test fold coverage"""
    a = 1
    

if __name__ == '__main__':
    visualize_cv_patch_split_for_a_scene(DATA_SETS, FEATURE_SET, args)
