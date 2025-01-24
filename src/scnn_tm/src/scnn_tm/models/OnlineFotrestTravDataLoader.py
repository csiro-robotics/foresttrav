# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

from pathlib import Path
import h5py
import numpy as np


class OnlineDataSetLoader:
    """Deals with all the logic to load the data sets"""

    def __init__(
        self,
        voxel_size: float,
        target_feature_set: list,
        min_pose_dist=1.0,
    ):
        self.target_feature_set = target_feature_set
        self.min_pose_distance = min_pose_dist
        self.voxel_size = voxel_size
        
        if not self.target_feature_set:
            raise ValueError("Target feature_set is not defined corretly")

    def load_online_data_set_raw(
        self, file_path: Path, tk_min: float = 0.0, tk_max: float = float("inf")
    ) -> dict:
        """Parse the hdf5 file"""
        assert Path(file_path).is_file()
        file_ds = h5py.File(file_path, "r")

        # Get target_to_source_feature_ids
        # HDF5 uses a byte representation to store string,
        # see https://stackoverflow.com/questions/51480834/matlab-hdf5-written-string-attributes-have-bstring-format
        self.source_feature_set = [
            feature.decode("ascii")
            for feature in list(file_ds["meta_data/feature_cloud_keys"])
        ]

        target_source_feature_ids = find_feature_ids(
            source_features=self.source_feature_set,
            target_features=self.target_feature_set,
        )

        # Check if all the features are present
        if (
            len(self.target_feature_set) != len(target_source_feature_ids)
            or not self.target_feature_set
        ):
            missing_features = [
                f for f in self.target_feature_set if f not in self.source_feature_set
            ]
            raise ValueError(
                f"Not all features found for target_feature set. {missing_features}"
            )

        # Poses
        poses_ds = file_ds["poses"]
        latest_poses = new_data_as_nparray(
            data_ds=poses_ds,
            tk_min=tk_min,
            tk_max=tk_max,
        )

        # Collsion cloud
        collision_cloud_ds = file_ds["collision_clouds"]
        latest_collision_cloud = newest_data_as_array(
            data_ds=collision_cloud_ds,
            tk_min=tk_min,
            tk_max=tk_max,
        )

        # Feature cloud
        feature_clouds = new_data_as_nparray(
            data_ds=file_ds["feature_clouds"],
            tk_min=tk_min,
            tk_max=tk_max,
        )

        file_ds.close()

        return {
            "poses": latest_poses,
            "feature_clouds": feature_clouds,
            "target_source_feature_ids": target_source_feature_ids,
            "coord_ids": [0, 1, 2],
            "collision_map": latest_collision_cloud,
        }

    def load_online_data_set_filtered(
        self, file_path: Path, tk_min: float = 0.0, tk_max: float = float("inf")
    ) -> dict:
        """Similar to load_online_data raw but filters with the target data"""
        # Valid ids of coords and feature set
        # TODO: Why dont we split it up now?
        data = self.load_online_data_set_raw(file_path, tk_min, tk_max)
        valid_ids = data["coord_ids"] + data["target_source_feature_ids"]
        filtered_feature_cloud = [
            cloud[:, valid_ids] for cloud in data["feature_clouds"]
        ]
        data["feature_clouds"] = filtered_feature_cloud
        return data


def sorted_timestamps_from_data(
    data_ds: h5py.Dataset,
    tk_min: float = float(0),
    tk_max: float = float("inf"),
):
    """Sorts the timestamps from the data sets ( ascending order). This is done in str form, since casting into numerical float values
        induces rounding errors.

    Args:
        data_ds (h5py.Dataset): Data set to load
        tk_min (float): Lower bound timestamp in systemtime
        tk_max (float): Upper timestamp in systemtime
    """

    data_timestamps = [
        data_stamp
        for data_stamp in data_ds.keys()
        if ((float(data_stamp) - tk_min) > 0.0 and (tk_max - float(data_stamp)) > 0.0)
    ]
    data_timestamps.sort(reverse=True)
    (f"timestamps sorted in str {data_timestamps}")

    # The timestamps should be sorted in "descending" order
    if float(data_timestamps[0]) < float(data_timestamps[-1]):
        msg = "[OnlineDataLoader] Timestamps are not sorted correctly."
        raise ValueError(msg)

    return data_timestamps


def new_data_as_nparray(
    data_ds: h5py.Dataset,
    tk_min: float = float("-inf"),
    tk_max: float = float("inf"),
) -> list:
    """Generates

    Args:
        data_ds (h5py.Dataset):     Data set containing the online data
        tk_min (float, optional):   Lower time bound for data to be extracted. Defaults to float("-inf").
        tk_max (float, optional):   Upper time boound for data to be extracted. Defaults to float("inf").

    Returns:
        list: _description_
    """
    sorted_timestamps = sorted_timestamps_from_data(
        data_ds=data_ds, tk_min=tk_min, tk_max=tk_max
    )

    new_data = [np.array(data_ds[str(data_stamp)]) for data_stamp in sorted_timestamps]

    return new_data


def newest_data_as_array(
    data_ds,
    tk_min: float = float("-inf"),
    tk_max: float = float("inf"),
) -> np.array:
    """Returns the newest/latest data for the current model"""

    sorted_timestamps = sorted_timestamps_from_data(
        data_ds=data_ds, tk_min=tk_min, tk_max=tk_max
    )
    
    # Make sure we are not doing somehting stupid
    assert float(sorted_timestamps[0]) - float(sorted_timestamps[-1]) >= 0.0

    return np.array(data_ds[sorted_timestamps[0]])


def find_feature_ids(source_features, target_features):
    """Returns an list of indexes matching the desired features to the source features.

    param: source_features  List of feature names for the feature of the data set (Data set stored)
    param: target_features  List of target feature names which is required  (Feature set wanted for model)
    """
    return [
        source_features.index(f_i)
        for f_i in (target_features)
        if f_i in source_features
    ]
