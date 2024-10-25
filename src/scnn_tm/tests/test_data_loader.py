import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import open3d as o3d
import copy

from scnn_tm import models
from scnn_tm import utils

from scnn_tm.models.ForestTravDataSet import ForestTravDataReader,  generate_map_patches
from scnn_tm.models.MapPatch import MapPatch, mask_for_points_within_patch

# parser = argparse.ArgumentParser()
CONFIG_FILE = Path(__file__).parent / "config" / "default_test_configs.yaml"
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

args = utils.scnn_io.Dict2Class(config)
FEATURE_SET = utils.scnn_io.generate_feature_set_from_key("occ_perm")


def test_correct_map_patch_to_feature_and_labels():
    '''Correctly assigns for each map_patch the id and labels

    Test is semi relevant in its current form but unclear how we can mock up a better one
    '''
    
    # Associated to the correct map
    # Associated to the corret patch
    test_args = copy.deepcopy(args)
    sample_number = 1000

    # Generate patches
    patches = [
        MapPatch(map_id=0, bounds=np.array([-1.0, -1.0, -0.01, 0.0, 0.0, 0.99]), global_patch_id=0),
        MapPatch(map_id=0, bounds=np.array([-1.0, -0.0, -0.01, 0.0, 1.0, 0.99]), global_patch_id=1),
        MapPatch(map_id=0, bounds=np.array([0.0, -1.0, -0.01, 1.0, 0.0, 0.99]), global_patch_id=2),
        MapPatch(map_id=0, bounds=np.array([0.0, 0.0, -0.01, 1.0, 1.0, 0.99]), global_patch_id=3),
        MapPatch(map_id=1, bounds=np.array([-1.0, -1.0, -0.01, 0.0, 0.0, 0.99]), global_patch_id=4),
        MapPatch(map_id=1, bounds=np.array([-1.0, -0.0, -0.01, 0.0, 1.0, 0.99]), global_patch_id=5),
        MapPatch(map_id=1, bounds=np.array([0.0, -1.0, -0.01, 1.0, 0.0, 0.99]), global_patch_id=6),
        MapPatch(map_id=1, bounds=np.array([0.0, 0.0, -0.01, 1.0, 1.0, 0.99]), global_patch_id=7),
    ]

    # Map 0 :
    coord_0 = np.array([
        [-1.0, -1.0, -0.01],
        # [-0.5, -0.5, 0.0],
        [-0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5,  0.5, 0.0], ])

    coord_1 = np.array([
        [-1.0, -1.0, -0.01],
        [-0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5,  0.5, 0.0], ])
    coords = [coord_0, coord_1]

    feature_0 = np.array([[0], [1], [2], [3]])
    feature_1 = np.array([[4], [5], [6], [7]])
    features = [feature_0, feature_1]

    labels_0 = np.array([[0], [1], [2], [3]])
    labels_1 = np.array([[4], [5], [6], [7]])
    labels = [labels_0, labels_1]

    coords_org = np.vstack(copy.deepcopy(coords))
    features_org =  np.vstack(copy.deepcopy(features))
    labels_org =  np.vstack(copy.deepcopy(labels))

    test_patches = generate_map_patches(coords, 0.1, 10, 0)

    # Generate the same patches as expected (more inconsequential sanity check so the features etc should align)
    for i in range(len(test_patches)):
        assert test_patches[i].map_id == patches[i].map_id
        assert test_patches[i].global_patch_id == patches[i].global_patch_id
        assert (test_patches[i].bounds == patches[i].bounds).all()

    for patch_i in test_patches:
        mask_i = mask_for_points_within_patch(data_coords=coords[patch_i.map_id], map_patch=patch_i)
        assert (coords[patch_i.map_id][mask_i] == coords_org[patch_i.global_patch_id]).all()
        assert features[patch_i.map_id][mask_i] == features_org[patch_i.global_patch_id]
        assert labels[patch_i.map_id][mask_i] == labels_org[patch_i.global_patch_id]