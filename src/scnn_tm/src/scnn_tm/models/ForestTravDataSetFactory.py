# Copyright (c) 2024
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz


import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from scnn_tm.models.DataAugmenter import DataPatchAugmenter
from scnn_tm.models.ForestTravDataSet import ForestTravDataReader, ForestTravDataSet
from scnn_tm.models.MapPatch import (
    convert_patches_to_data_set,
    generate_map_patches,
)


class ForestTravDataSetFactory:
    """The ForestTravDataSetFacory generates the trainning and test set based on the config files
    and variations. It assumes to load a train, validation and test set, where the test set is held out.

    """

    def __init__(self, params, scaler, data_augmentor=None) -> None:
        # Assumption is made that the cv_scene is handeled in the parameter side
        self.params = params

        # Check that none of the data sets overlap
        # self.check_data_sets(params.train_data_sets, params.test_data_sets)
        raw_train_data_set, raw_test_data_set = self.read_data_set()

        raw_train_data = patchify_data_set(
            raw_data=raw_train_data_set,
            patch_strategy=params.patch_strategy,
            voxel_size=params.voxel_size,
            nvoxel_leaf=params.nvoxel_leaf,
            min_patch_sample_nbr=params.min_patch_sample_nbr,
        )

        train_data, val_data = split_data_sets_based_on_strategy(params, raw_train_data)

        # Only want to do this when we need to
        if scaler is None and params.use_feature_scaling:
            self.scaler = initalize_and_fit_scaler(train_data)
        else:
            self.scaler = scaler

        if params.use_data_augmentation:
            if data_augmentor is None:
                self.data_augmentor = DataPatchAugmenter(
                    voxel_size=params.voxel_size,
                    noise_chance=params.data_augmenter_noise_chance,
                    noise_mean=params.data_augmenter_noise_mean,
                    noise_std=params.data_augmenter_noise_std,
                    sample_pruning_chance=params.data_augmenter_sample_pruning_chance,
                    rotation_chance=params.data_augmenter_batch_rotation_chance,
                    translation_chance=params.data_augmenter_batch_translation_chance,
                    n_voxel_displacement=params.data_augmenter_batch_nvoxel_displacement,
                    mirror_chance=params.data_augmenter_mirror_chance,
                )
            else:
                self.data_augmentor = data_augmentor
        else:
            self.data_augmentor = None

        # These are the data itself
        train_data_scaled = scale_data(data_set=train_data, scaler=self.scaler)
        val_data_scaled = scale_data(data_set=val_data, scaler=self.scaler)
        test_data_scaled = scale_data(data_set=raw_test_data_set, scaler=self.scaler)

        self.generate_data_sets(
            train_data=train_data_scaled,
            val_data=val_data_scaled,
            test_data=test_data_scaled,
        )

    def read_data_set(self):
        """reads the data sets"""
        self.train_data_reader = ForestTravDataReader(
            self.params.train_data_sets, feature_set=self.params.feature_set
        )
        self.test_data_reader = ForestTravDataReader(
            data_sets_files=self.params.test_data_sets,
            feature_set=self.params.feature_set,
        )

        return self.train_data_reader.raw_data_set, self.test_data_reader.raw_data_set

    def generate_data_sets(self, train_data, val_data, test_data):
        """Generates the DataSet used by pytorch

        Args:
            train_data (_type_): _description_
            val_data (_type_): _description_
            test_data (_type_): _description_
        """
        self.train_data_set = ForestTravDataSet(
            data_set=train_data,
            voxel_size=self.params.voxel_size,
            data_augmentor=self.data_augmentor,
            use_data_augmentation=self.params.use_data_augmentation,
        )

        self.val_data_set = ForestTravDataSet(
            data_set=val_data,
            voxel_size=self.params.voxel_size,
            data_augmentor=None,
            use_data_augmentation=False,
        )

        self.test_data_set = ForestTravDataSet(
            data_set=test_data,
            voxel_size=self.params.voxel_size,
            data_augmentor=None,
            use_data_augmentation=False,
        )


def initalize_and_fit_scaler(data_train):
    # Initialize a default scaler if we want to use scaling put dont have a scaler
    scaler = StandardScaler()
    stacked_features = [np.array(data["features"]) for data in data_train]
    stacked_features = np.vstack(stacked_features)
    scaler.fit(stacked_features)

    return scaler


def scale_data(scaler, data_set):
    if scaler == None:
        raise ValueError("Scaler is not defined")

    if not hasattr(scaler, "n_features_in_"):
        raise AttributeError(
            "Sacler is not fitted. Please fit scaler before scaling data"
        )

    for data in data_set:
        data["features"] = scaler.transform(data["features"])

    return data_set


def patchify_data_set(
    raw_data: list,
    patch_strategy,
    voxel_size,
    nvoxel_leaf,
    min_patch_sample_nbr,
):
    data_patches = []

    # Sampling
    if patch_strategy == "sampling":
        msg = "Patch sampling strategy is not implemented"
        raise NotImplemented(msg)

    # Grid
    elif patch_strategy == "grid":
        data_set_coords = [data_set["coords"] for data_set in raw_data]

        data_patches = generate_map_patches(
            data_set_coords=data_set_coords,
            voxel_size=voxel_size,
            nvoxel_leaf=nvoxel_leaf,
            min_number_of_samples=min_patch_sample_nbr,
        )

    # Default grid
    else:
        msg = f'Could not find correct patchify_strategt {patch_strategy}  \n Options: [ "grid", "sampling"]'
        raise ValueError(msg)

    # MapPatches to data_set using fields
    data_set = convert_patches_to_data_set(raw_data_sets=raw_data, patches=data_patches)
    return data_set


################## SPLITING DATA SET ####################


def kfold_train_val_split(data_set_patches, params, fold_k):
    """Generates a k-fold split from data_set patches and return a train/val split"""
    kf = KFold(
        n_splits=params.num_splits,
        random_state=params.random_seed,
        shuffle=params.shuffle,
    )
    all_splits = [k for k in kf.split(data_set_patches)]

    # Do a train/val split still?
    train_indexes, val_index = all_splits[fold_k]
    train_patches = np.array(data_set_patches)[train_indexes]
    val_patches = np.array(data_set_patches)[val_index]

    assert params.use_cv

    return train_patches, val_patches


def scale_patch_data_set(data_set, scaler):
    if scaler == None:
        msg = "scaler is not defined"
        raise ValueError(msg)

    # Scale data set
    for data in data_set:
        data["features"] = scaler.transform(data["features"])

    return data_set


def generate_and_fit_scaler(params, data_train, scaler):
    # Initialize a default scaler if we want to use scaling put dont have a scaler
    if scaler is None and params.use_feature_scaling:
        scaler = StandardScaler()

    stacked_features = [np.array(data["features"]) for data in data_train]
    stacked_features = np.vstack(stacked_features)
    scaler.fit(stacked_features)

    return scaler


def split_data_sets_based_on_strategy(params, raw_train_data):
    train_data = []
    val_data = []

    if params.use_cv:
        train_data, val_data = kfold_train_val_split(
            raw_train_data, params, params.current_fold
        )
    else:
        train_data, val_data = train_test_split(
            raw_train_data,
            train_size=params.train_ratio,
            random_state=params.random_seed,
            shuffle=params.shuffle,
        )

    assert len(raw_train_data) == (len(train_data) + len(val_data))
    return train_data, val_data
