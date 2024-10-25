from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import open3d as o3d
import copy

from scnn_tm import models
from scnn_tm import utils


CONFIG_FILE = Path(__file__).parent / "config" / "default_test_configs.yaml"
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

args = utils.scnn_io.Dict2Class(config)

FEATURE_SET = utils.scnn_io.generate_feature_set_from_key("occ_perm")

def test_data_augmenter_no_augmentation():
    """This is the case when we only want geometric augmentation"""

    data_augmenter = models.DataAugmenter.DataPatchAugmenter(args)

    input_coords = np.random.uniform(size=(30, 3), low=-20.0, high=20.0)
    input_features = np.random.uniform(size=(30, 15), low=0, high=0.989)
    input_labels = np.random.uniform(size=(30, 1))

    original_coords = copy.deepcopy(input_coords)
    original_features = copy.deepcopy(input_features)
    original_labels = copy.deepcopy(input_labels)

    scaled_coord, scaled_labels, scaled_features = data_augmenter.augment_data(
        coord=input_coords, labels=input_labels, features=input_features
    )

    assert (original_coords == scaled_coord).all()
    assert (original_features == scaled_features).all()
    assert (original_labels == scaled_labels).all()


def test_data_augmenter_geometric_augmentation():
    """This is the case when we only want geometric augmentation
    a) The coordinates are shifted
    b) Scaled and input should not be the same for the coordinates
    """
    params = copy.deepcopy(args)
    params.data_augmenter_batch_nvoxel_displacement = 5.0
    params.data_augmenter_batch_translation_chance = 1.0  
    # Note: Less than 1 makes no sense since the whole translation is for the batch
    data_augmenter = models.DataAugmenter.DataPatchAugmenter(params)

    input_coords = np.random.uniform(size=(30, 3), low=-20.0, high=20.0)
    input_features = np.random.uniform(size=(30, 15), low=0, high=0.989)
    input_labels = np.random.uniform(size=(30, 1))

    original_coords = copy.deepcopy(input_coords)
    original_features = copy.deepcopy(input_features)
    original_labels = copy.deepcopy(input_labels)

    scaled_coord, scaled_labels, scaled_features = data_augmenter.augment_data(
        coord=input_coords, labels=input_labels, features=input_features
    )

    # We only aim to translate the scene. Hence the z-axis is excluded from this test
    assert not np.isclose(original_coords[:, 0:2], scaled_coord[:, 0:2]).all()
    assert (np.isclose(original_features, scaled_features)).all()
    assert (np.isclose(original_labels, scaled_labels)).all()

    # Sanity check so we don't modify the input coords
    assert (np.isclose(original_coords, input_coords)).all()
    assert (np.isclose(original_features, input_features)).all()
    assert (np.isclose(original_labels, input_labels)).all()


test_data_augmenter_geometric_augmentation()


def test_data_augmenter_feature_augmentation():
    """This is the case when we only want augmentation on the features
    a) The features are augmented
    b)  and input should not be the same for the coordinates
    """
    params = copy.deepcopy(args)
    params.data_augmenter_noise_chance = 1.0
    params.data_augmenter_noise_mean = 0.5
    params.data_augmenter_noise_std = 0.05
    data_augmenter = models.DataAugmenter.DataPatchAugmenter(params)

    input_coords = np.random.uniform(size=(30, 3), low=-20.0, high=20.0)
    input_features = np.random.uniform(size=(30, 15), low=0, high=0.989)
    input_labels = np.random.uniform(size=(30, 1))

    original_coords = copy.deepcopy(input_coords)
    original_features = copy.deepcopy(input_features)
    original_labels = copy.deepcopy(input_labels)

    scaled_coord, scaled_labels, scaled_features = data_augmenter.augment_data(
        coord=input_coords, labels=input_labels, features=input_features
    )

    # We only aim to translate the scene. Hence the z-axis is excluded from this test
    assert (np.isclose(original_features, input_features)).all()
    assert not (np.isclose(original_features, scaled_features)).all()

    # Test that we dont modify anything else by accident
    assert np.isclose(original_coords , scaled_coord).all()
    assert np.isclose(original_labels ,scaled_labels).all()
    assert np.isclose(original_coords ,input_coords).all()
    assert np.isclose(original_labels ,input_labels).all()


test_data_augmenter_feature_augmentation()
