# Copyright (c) 2024
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

from typing import Any, Dict
from pathlib import Path

from scnn_tm.utils.file_io import load_data_set

import torch
from torch.utils.data import Dataset
from torchsparse import SparseTensor

"""
 ForestTrav data set classes
 The @c ForestTravDataReader parses the data files and ensures the correct format. This class should be used to load the data

ForestTravDataSet' is the data set class used for the PyTorch learning modules. The data set class used to load the 

"""


# TODO:
#   - [ ] Should remove the additional fields for the label_obs and label_prob, they are used for a special case only... 

class ForestTravDataReader:
    """  Reads the data set files and parses into the raw data set

    Raises:
        FileNotFoundError: _description_
    
    """ 

    # Base fields that define a data set i with the keys beeing the required fields
    BASE_FIELDS = {
        "coords": ["x", "y", "z"],
        "feature_set": [],
        "label": ["label"],
        "label_obs": ["label_obs"],
        "label_prob": ["label_prob"],
    }

    def __init__(
        self,
        data_sets_files: list,
        feature_set: list,
        base_fields: dict = {},
        additional_fields: dict = {},
    ) -> None:
        # Data set that contains
        self.raw_data_set = []

        # Initialize base fields if not provided
        if not base_fields:
            base_fields = self.BASE_FIELDS

        # Complain if we have no data set
        if len(data_sets_files) < 1:
            msg = "No data set files past to data reader"
            raise ValueError(msg)
        elif len(feature_set) < 1:
            msg = "No feature set provided for the data reader"
            raise ValueError(msg)

        base_fields["feature_set"] = feature_set
        base_fields.update(additional_fields)

        # Check fields
        for key, data in base_fields.items():
            if not len(data) > 0:
                raise ValueError(f"Filed for [ {key} ] contains no data")

        self.load_data_set_to_df(data_sets_files, fields=base_fields)

    def load_data_set_to_df(self, data_sets: list, fields: list) -> list:
        """ """
        print("Features [{}]".format(fields["feature_set"]))

        self.raw_data_set = []
        for data_set in data_sets:
            df_total = load_data_set(data_set)

            # Seperate the position and features into numpy arrays
            data_dict = {}
            data_dict["coords"] = df_total[fields["coords"]].to_numpy()
            data_dict["features"] = df_total[fields["feature_set"]].to_numpy()
            data_dict["label"] = df_total[["label"]].to_numpy()
            data_dict["label_obs"] = df_total[fields["label_obs"]].to_numpy()
            data_dict["label_prob"] = df_total[fields["label_prob"]].to_numpy()

            # Additional files:
            data_dict["id"] = len(self.raw_data_set)
            data_dict["name"] = Path(data_set).stem

            self.raw_data_set.append(data_dict)

        if not self.raw_data_set:
            raise FileNotFoundError(
                f" No data could be loader for the data sets: {data_sets}"
            )
            
    def __len__(self):
        return len(self.raw_data_set)

class ForestTravDataSet(Dataset):
    """Generates patches of data for ME engine based on map_id_pairs."""

    def __init__(
        self,
        data_set,
        voxel_size: float,
        data_augmentor=None,
        use_data_augmentation=False,
    ) -> None:
        super().__init__()

        self.data_set = data_set
        self.voxel_size = voxel_size
        self.data_augmentor = data_augmentor
        self.use_data_augmentation = use_data_augmentation

    def __getitem__(self, i):
        """Return a patch of the point-cloud based on region"""

        # Batching
        batch_coords = self.data_set[i]["coords"]
        batch_features = self.data_set[i]["features"]
        batch_labels = self.data_set[i]["label"]
        batch_labels_obs = self.data_set[i]["label_obs"]
        batch_labels_prob = self.data_set[i]["label_prob"]

        if self.data_augmentor != None and self.use_data_augmentation:
            (
                batch_coords,
                batch_labels,
                batch_features,
            ) = self.data_augmentor.augment_data(
                batch_coords, batch_labels, batch_features
            )
            batch_labels_prob = batch_labels_prob[self.data_augmentor.pruning_mask]
            batch_labels_obs = batch_labels_obs[self.data_augmentor.pruning_mask]
        # May need to augment additional features
        batch_coords = batch_coords // self.voxel_size

        coords = torch.tensor(batch_coords, dtype=torch.int)
        features = torch.tensor(batch_features, dtype=torch.float)
        labels = torch.tensor(batch_labels, dtype=torch.long)

        assert coords.shape[0] == features.shape[0]
        assert features.shape[0] == labels.shape[0]

        input = SparseTensor(coords=coords, feats=features)
        label = SparseTensor(coords=coords, feats=labels)

        # Additional fields
        label_obs = torch.tensor(batch_labels_obs, dtype=torch.long)
        label_prob = torch.tensor(batch_labels_prob, dtype=torch.float)
        
        # TODO(fab): Should the filtering od the lfe/full data happen here?

        return {
            "input": input,
            "label": label,
            "label_prob": SparseTensor(coords=coords, feats=label_prob),
            "label_obs": SparseTensor(coords=coords, feats=label_obs),
        }

    def __len__(self):
        return len(self.data_set)
