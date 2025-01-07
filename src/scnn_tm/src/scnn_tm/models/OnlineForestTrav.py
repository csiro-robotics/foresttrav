# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz


from pathlib import Path

from scnn_tm.models.DataAugmenter import DataPatchAugmenter
from scnn_tm.models.ForestTravGraph import ForestTravGraph
from scnn_tm.models.ForestTravDataSet import ForestTravDataReader, ForestTravDataSet
from scnn_tm.models.ForestTravDataSetFactory import (
    generate_and_fit_scaler,
    scale_patch_data_set,
    split_data_sets_based_on_strategy,
)
from scnn_tm.models.OnlineFotrestTravDataLoader import OnlineDataSetLoader


class OnlineForestTravDataFactory:
    REQUIRED_PARAMS = [
        "voxel_size",
        "feature_set",
        "min_pose_dist",
        "patch_width",
    ]

    def __init__(self, params, scaler, data_augmentor=None) -> None:
        self.params = params
        self.scaler = scaler

        self.check_params()

        # Need a separate graph for each data set
        raw_train_patch_data_sets = self.generate_data_graphs_from_file(
            params, params.train_data_sets
        )

        # Perform the splits into train/val if required
        train_data, val_data = split_data_sets_based_on_strategy(
            params=params, raw_train_data=raw_train_patch_data_sets
        )
        # Setup scaler
        if scaler is None and params.use_feature_scaling:
            scaler = generate_and_fit_scaler(
                params=params, scaler=scaler, data_train=train_data
            )
        self.scaler = scaler
        # Data augmentation
        if params.use_data_augmentation:
            if data_augmentor is None:
                self.data_augmentor = DataPatchAugmenter(params)
            else:
                self.data_augmentor = data_augmentor
        else:
            self.data_augmentor = None
        # Load test data set
        raw_test_set = self.load_test_data(params.test_data_sets)

        # Return the two or three data sets
        train_data_scaled = scale_patch_data_set(train_data, scaler)
        val_data_scaled = scale_patch_data_set(val_data, scaler)
        test_data_scaled = scale_patch_data_set(raw_test_set, scaler)

        self.generate_data_sets(
            train_data=train_data_scaled,
            val_data=val_data_scaled,
            test_data=test_data_scaled,
        )

    def generate_data_graphs_from_file(self, params, data_set_files) -> list:
        # Setup the data loader
        data_set_loader = OnlineDataSetLoader(
            params.voxel_size, params.feature_set, params.min_pose_dist
        )

        # Generate the graphs for each data set
        patch_data_set = []
        for data_file in data_set_files:
            graph_i = ForestTravGraph(
                params.voxel_size, params.patch_width, params.min_pose_dist
            )
            new_data_batch = data_set_loader.load_online_data_set_filtered(
                data_file, tk_min=0, tk_max=float("inf")
            )
            graph_i.add_new_data(new_data_batch=new_data_batch)

            # TODO: Do we want the joint data set as default?
            # TODO: This should be the point where a) traversability threshold b) use the lfe vs full data...
            patch_data_set += graph_i.get_patch_data_set_copy()

        return patch_data_set

    def load_test_data(self, test_data_files: list):

        test_csv_files = [
            file_i for file_i in test_data_files if "csv" in Path(file_i).suffix
        ]

        # Read and patchify the data?
        return ForestTravDataReader(
            data_sets_files=test_csv_files, feature_set=self.params.feature_set
        ).raw_data_set

        # test_graph_files = [file_i for file_i in test_data_files if "hf5" in Path(file_i).suffix]

        # # No other options than generate the graphs from files
        # raw_graph_data= self.generate_data_graphs_from_file(data_set_files=test_graph_files, params=self.params)

        # # Convert test data set into the data laoder
        # self.test_data_set = ForestTravDataSet(data_set=raw_test_data_reader.raw_data_set, voxel_size=self.params.voxel_size)

    def check_params(self):
        for param_key in self.REQUIRED_PARAMS:
            if not hasattr(self.params, param_key):
                msg = f" Could not find parameter {param_key} "
                raise ValueError(msg)

    def generate_data_sets(self, train_data, val_data, test_data=None):
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
