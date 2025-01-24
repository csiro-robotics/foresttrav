# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import open3d as o3d
import copy

from scnn_tm import models
from scnn_tm import utils

from scnn_tm.models.ForestTravDataSet import ForestTravDataSet, ForestTravDataReader
from scnn_tm.models.ForestTravDataSetFactory import ForestTravDataSetFactory


# parser = argparse.ArgumentParser()
CONFIG_FILE = Path(__file__).parent / "config" / "default_test_configs.yaml"
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

args = utils.scnn_io.Dict2Class(config)

# Data set files
TRAIN_DATA_SETS = [Path(__file__).parent / "config" / "train_data_set.csv"]
TEST_DATA_SETS = [Path(__file__).parent / "config" / "train_data_set.csv"]


def test_basic_patch_factory():
    """Test the split of the data into the correct elements and nothing mutates the data per-se
    - No  coords augmentation
    - No feature scaling or augmentation
    - No labels augmentation
    """
    test_args = copy.deepcopy(args)

    # Test case arguments
    test_args.feature_set = utils.scnn_io.generate_feature_set_from_key("ftm")
    test_args.train_data_sets = TRAIN_DATA_SETS
    test_args.test_data_sets = TEST_DATA_SETS
    test_args.use_cv = False
    test_args.num_splits = 3
    test_args.use_feature_scaling = True

    data_set_factory = ForestTravDataSetFactory(params=test_args, scaler=None)

    # Load the data again so we can compare
    org_data_set_loader = ForestTravDataReader(
        data_sets_files=TRAIN_DATA_SETS,
        feature_set=test_args.feature_set,
    )
    raw_train_data  = data_set_factory.train_data_reader.raw_data_set
    org_train_data  = org_data_set_loader.train_data_reader.raw_data_set
    # Check that the base data has not been modified
    for i in range(len(raw_train_data)):
        assert (
            raw_train_data[i]["coords"]
            == org_train_data[i]["coords"]
        ).all()
        assert (
            raw_train_data[i]["label"]
            == org_train_data[i]["label"]
        ).all()
        assert (
            raw_train_data[i]["features"]
            == org_train_data[i]["features"]
        ).all()



# def test_patch_factory_with_feature_scaling():
#     """Test the split of the data into the correct elements and nothing mutates the data per-se
#     - No  coords augmentation
#     - No feature scaling or augmentation
#     - No labels augmentation
#     """
#     test_args = copy.deepcopy(args)

#     # Test case arguments
#     test_args.data_set_key = "debug"
#     feature_set = utils.scnn_io.generate_feature_set_from_key("occ_ev")

#     # Set the scaling intervals such that there should be no overlap of the original features and the scaled features
#     # These are nonsensical bound
#     for feature in feature_set:
#         test_args.scaler_params["predefined_scaling_bounds"][feature] = [
#             -1000.0,
#             -100.0,
#         ]

#     test_args.num_splits = len(TRAIN_DATA_SETS)
#     test_args.use_feature_scaling = True
#     data_set_factory = PatchCvDataSetFactory(
#         data_set=TRAIN_DATA_SETS, feature_set=feature_set, params=test_args
#     )

#     # Load the data again so we can compare
#     org_data_set_loader = VoxelPatchForestDataSetLoader(
#         data_sets=TRAIN_DATA_SETS,
#         feature_set=feature_set,
#         voxel_size=test_args.voxel_size,
#         nvoxel_leaf=test_args.nvoxel_leaf,
#     )

#     # Check that the base data has not been modified
#     for i in range(len(org_data_set_loader.data_set_coords)):
#         assert (
#             org_data_set_loader.data_set_coords[i]
#             == data_set_factory.loader_data_set.data_set_coords[i]
#         ).all()
#         assert (
#             org_data_set_loader.data_set_features[i]
#             == data_set_factory.loader_data_set.data_set_features[i]
#         ).all()
#         assert (
#             org_data_set_loader.data_set_labels[i]
#             == data_set_factory.loader_data_set.data_set_labels[i]
#         ).all()

#         assert (
#             org_data_set_loader.data_set_coords[i]
#             == data_set_factory.data_sets_train_val_test_coords[i]
#         ).all()
#         assert (
#             org_data_set_loader.data_set_features[i]
#             != data_set_factory.data_sets_train_val_test_features[i]
#         ).all()
#         assert (
#             org_data_set_loader.data_set_labels[i]
#             == data_set_factory.data_sets_train_val_test_labels[i]
#         ).all()


# def test_patch_factory_full():
#     """Test the split of the data into the correct elements and nothing mutates the data per-se
#     - Coords augmentation thought the FtmDataAugmentor
#     - Feature scaling and feature augmentation
#     - No labels augmentation
#     """
#     test_args = copy.deepcopy(args)

#     # Test case arguments
#     test_args.data_set_key = "debug"
#     feature_set = utils.scnn_io.generate_feature_set_from_key("occ_ev")

#     # Set the scaling intervals such that there should be no overlap of the original features and the scaled features
#     # These are nonsensical bound
#     for feature in feature_set:
#         test_args.scaler_params["predefined_scaling_bounds"][feature] = [
#             -1000.0,
#             -100.0,
#         ]

#     # Data Augmentation
#     test_args.num_splits = len(TRAIN_DATA_SETS)
#     test_args.use_feature_scaling = True
#     test_args.use_data_augmentation = True
#     test_args.data_augmenter_noise_chance = 1.0
#     test_args.data_augmenter_noise_mean = 1.0
#     test_args.data_augmenter_noise_std = 0.05
#     test_args.data_augmenter_batch_translation_chance = 1.0
#     test_args.data_augmenter_batch_nvoxel_displacement = 5.0

#     # Setup the factory
#     data_set_factory = PatchCvDataSetFactory(
#         data_set=TRAIN_DATA_SETS, feature_set=feature_set, params=test_args
#     )

#     # G
#     for n in range(6):
#         data_set_factory.train_data_set[n]
#         data_set_factory.val_data_set[n]
#         data_set_factory.test_data_set[n]

#     # Load the data again so we can compare
#     org_data_set_loader = VoxelPatchForestDataSetLoader(
#         data_sets=TRAIN_DATA_SETS,
#         feature_set=feature_set,
#         voxel_size=test_args.voxel_size,
#         nvoxel_leaf=test_args.nvoxel_leaf,
#     )

#     # Check that the base data has not been modified
#     for i in range(len(org_data_set_loader.data_set_coords)):
#         assert (
#             org_data_set_loader.data_set_coords[i]
#             == data_set_factory.loader_data_set.data_set_coords[i]
#         ).all()
#         assert (
#             org_data_set_loader.data_set_features[i]
#             == data_set_factory.loader_data_set.data_set_features[i]
#         ).all()
#         assert (
#             org_data_set_loader.data_set_labels[i]
#             == data_set_factory.loader_data_set.data_set_labels[i]
#         ).all()

#         assert (
#             org_data_set_loader.data_set_coords[i]
#             == data_set_factory.data_sets_train_val_test_coords[i]
#         ).all()
#         assert (
#             org_data_set_loader.data_set_features[i]
#             != data_set_factory.data_sets_train_val_test_features[i]
#         ).all()
#         assert (
#             org_data_set_loader.data_set_labels[i]
#             == data_set_factory.data_sets_train_val_test_labels[i]
#         ).all()


# #################################### TEST_TRAIN FACTORY ####################################


# def test_basic_train_factory():
#     """Test the split of the data into the correct elements and nothing mutates the data per-se
#     - No  coords augmentation
#     - No feature scaling or augmentation
#     - No labels augmentation
#     """
#     test_args = copy.deepcopy(args)

#     # Test case arguments
#     feature_set = utils.scnn_io.generate_feature_set_from_key("ftm")

#     for feature in feature_set:
#         test_args.scaler_params["predefined_scaling_bounds"][feature] = [
#             -1000.0,
#             -100.0,
#         ]

#     test_args.num_splits = len(TRAIN_DATA_SETS)
#     test_args.use_feature_scaling = False
#     data_set_factory = ClassicalTrainTestDataSetFactory(
#         train_data_sets=TRAIN_DATA_SETS,
#         test_data_sets=TEST_DATA_SETS,
#         feature_set=feature_set,
#         params=test_args,
#     )

#     # Load the data again so we can compare
#     org_train_data_set_loader = VoxelPatchForestDataSetLoader(
#         data_sets=TRAIN_DATA_SETS,
#         feature_set=feature_set,
#         voxel_size=test_args.voxel_size,
#         nvoxel_leaf=test_args.nvoxel_leaf,
#     )

#     # Check that the base data has not been modified
#     for i in range(len(org_train_data_set_loader.data_set_coords)):
#         assert (
#             org_train_data_set_loader.data_set_coords[i]
#             == data_set_factory.train_data_loader.data_set_coords[i]
#         ).all()
#         assert (
#             org_train_data_set_loader.data_set_features[i]
#             == data_set_factory.train_data_loader.data_set_features[i]
#         ).all()
#         assert (
#             org_train_data_set_loader.data_set_labels[i]
#             == data_set_factory.train_data_loader.data_set_labels[i]
#         ).all()

#         assert (
#             org_train_data_set_loader.data_set_coords[i]
#             == data_set_factory.data_sets_train_val_coords[i]
#         ).all()
#         assert (
#             org_train_data_set_loader.data_set_features[i]
#             == data_set_factory.data_sets_train_val_features[i]
#         ).all()
#         assert (
#             org_train_data_set_loader.data_set_labels[i]
#             == data_set_factory.data_sets_train_val_labels[i]
#         ).all()

#     org_test_data_set_loader = VoxelPatchForestDataSetLoader(
#         data_sets=TEST_DATA_SETS,
#         feature_set=feature_set,
#         voxel_size=test_args.voxel_size,
#         nvoxel_leaf=test_args.nvoxel_leaf,
#     )

#     # Check that the base test data has not been modified
#     for i in range(len(org_test_data_set_loader.data_set_coords)):
#         assert (
#             org_test_data_set_loader.data_set_coords[i]
#             == data_set_factory.test_loader.data_set_coords[i]
#         ).all()
#         assert (
#             org_test_data_set_loader.data_set_features[i]
#             == data_set_factory.test_loader.data_set_features[i]
#         ).all()
#         assert (
#             org_test_data_set_loader.data_set_labels[i]
#             == data_set_factory.test_loader.data_set_labels[i]
#         ).all()

#         assert (
#             org_test_data_set_loader.data_set_features[i]
#             == data_set_factory.data_set_test_features[i]
#         ).all()


# def test_test_train_factory_all_augmentation():
#     """Test the split of the data into the correct elements and nothing mutates the data per-se
#     - No  coords augmentation
#     - No feature scaling or augmentation
#     - No labels augmentation
#     """
#     test_args = copy.deepcopy(args)

#     # Test case arguments
#     test_args.data_set_key = "debug"

#     feature_set = utils.scnn_io.generate_feature_set_from_key("occ_ev")

#     # Set the scaling intervals such that there should be no overlap of the original features and the scaled features
#     # These are nonsensical bound
#     for feature in feature_set:
#         test_args.scaler_params["predefined_scaling_bounds"][feature] = [
#             -1000.0,
#             -100.0,
#         ]

#     # Data Augmentation
#     test_args.num_splits = len(TRAIN_DATA_SETS)
#     test_args.use_feature_scaling = True
#     test_args.use_data_augmentation = True
#     test_args.data_augmenter_noise_chance = 1.0
#     test_args.data_augmenter_noise_mean = 1.0
#     test_args.data_augmenter_noise_std = 0.05
#     test_args.data_augmenter_batch_translation_chance = 1.0
#     test_args.data_augmenter_batch_nvoxel_displacement = 5.0

#     test_args.num_splits = len(TRAIN_DATA_SETS)
#     test_args.use_feature_scaling = True

#     data_set_factory = ClassicalTrainTestDataSetFactory(
#         train_data_sets=TRAIN_DATA_SETS,
#         test_data_sets=TEST_DATA_SETS,
#         feature_set=feature_set,
#         params=test_args,
#     )

#     # Load the data again so we can compare
#     org_train_data_set_loader = VoxelPatchForestDataSetLoader(
#         data_sets=TRAIN_DATA_SETS,
#         feature_set=feature_set,
#         voxel_size=test_args.voxel_size,
#         nvoxel_leaf=test_args.nvoxel_leaf,
#     )

#     for n in range(6):
#         data_set_factory.train_data_set[n]
#         data_set_factory.val_data_set[n]
#         data_set_factory.test_data_set[n]

#     # Check that the base data has not been modified
#     for i in range(len(org_train_data_set_loader.data_set_coords)):
#         assert (
#             org_train_data_set_loader.data_set_coords[i]
#             == data_set_factory.train_data_loader.data_set_coords[i]
#         ).all()
#         assert (
#             org_train_data_set_loader.data_set_features[i]
#             == data_set_factory.train_data_loader.data_set_features[i]
#         ).all()
#         assert (
#             org_train_data_set_loader.data_set_labels[i]
#             == data_set_factory.train_data_loader.data_set_labels[i]
#         ).all()

#         assert (
#             org_train_data_set_loader.data_set_coords[i]
#             == data_set_factory.data_sets_train_val_coords[i]
#         ).all()
#         assert (
#             org_train_data_set_loader.data_set_features[i]
#             != data_set_factory.data_sets_train_val_features[i]
#         ).all()
#         assert (
#             org_train_data_set_loader.data_set_labels[i]
#             == data_set_factory.data_sets_train_val_labels[i]
#         ).all()

#     org_test_data_set_loader = VoxelPatchForestDataSetLoader(
#         data_sets=TEST_DATA_SETS,
#         feature_set=feature_set,
#         voxel_size=test_args.voxel_size,
#         nvoxel_leaf=test_args.nvoxel_leaf,
#     )

#     # Check that the base test data has not been modified
#     for i in range(len(org_test_data_set_loader.data_set_coords)):
#         assert (
#             org_test_data_set_loader.data_set_coords[i]
#             == data_set_factory.test_loader.data_set_coords[i]
#         ).all()
#         assert (
#             org_test_data_set_loader.data_set_features[i]
#             == data_set_factory.test_loader.data_set_features[i]
#         ).all()
#         assert (
#             org_test_data_set_loader.data_set_labels[i]
#             == data_set_factory.test_loader.data_set_labels[i]
#         ).all()

#         # Note: Only the features should change!
#         assert (
#             org_test_data_set_loader.data_set_features[i]
#             != data_set_factory.data_set_test_features[i]
#         ).all()
