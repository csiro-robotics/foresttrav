# Copyright (c) 2023
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

from pathlib import Path

import numpy as np

from scnn_tm.models.ForestTravDataSet import ForestTravDataReader, ForestTravDataSet
from scnn_tm.models.MapPatch import MapPatch, convert_patches_to_data_set


def full_scene_data_set(data_files, voxel_size, feature_set):
    """Loads all data files into the a single

    Args:
        files (_type_): _description_
        voxel_size (float): Size of the voxel representation
        feature_set (list): Feature set defined as a list of strings
    """

    if not data_files:
        msg = "data files is empty"
        raise ValueError(msg)

    return ForestTravDataReader(
        data_sets_files=data_files,
        feature_set=feature_set,
    ).raw_data_set


def setup_and_preprocess_data_sets(
    data_sets,
    params,
):
    if not (
        hasattr(params, "test_data_sets")
        and hasattr(params, "train_data_sets")
        and hasattr(params, "feature_set")
    ):
        msg = "Parameters are missing definition of data sets (train or test) or feature ses."
        raise ValueError(msg)

    # Loading all the data into the data set format
    # Data source needs to be a) type and b) file location

    # Case 1: Training data only

    # Case 2: Training and testing data

    # Case 3: Training, testing and validation data -> not implemented

    # Modification of raw data, i.e sub-sampling, split into evenly small cubes, etc

    # Split the data based on these cases

    # Case kFold

    # Case Random split

    # Case 3: No split -> not implemented

    # Scaling of data

    # return train, test and val data sets
