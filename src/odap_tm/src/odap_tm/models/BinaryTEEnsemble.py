# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

from pathlib import Path
import numpy as np

import torch
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate


from odap_tm.models import io_model_parsing as oadap_utils


class BinaryFtmEnsemble:
    """Ensemble that has a binary classifier that can provide
    a) Traversability probability

    """

    def __init__(
        self, model_config_files: list, device: str, input_feature_set: list
    ) -> None:
        self.models = []
        self.org_model_feature_sets = []
        self.scalers = []
        self.feature_set_indexs = []

        self.device = device

        # Rethink what model_file is: (model_yaml_file, cv_nmbr)
        if not model_config_files:
            raise ValueError("No models provided found!")

        for model_config_file, cv_number in model_config_files:
            if not Path(model_config_file).is_file():
                print(f"Could not load model file {model_config_file}")

            (
                model,
                scaler,
                model_feature_set,
            ) = oadap_utils.load_from_from_yaml(params=None, config_file=model_config_file)
            model.to(device)
            model.eval()
            self.models.append(model)
            self.scalers.append(scaler)
            self.org_model_feature_sets.append(model_feature_set)

        # We need to keep the indexes we need for the feature set based on the ftm_feature set structure
        self.map_from_input_to_model_feature_sets(input_feature_set)

    def find_feature_ids(self, source_features, target_features):
        """Returns an list of indexes matching the desired features to the source features.

        param: source_features  List of feature names for the feature of the data set (Data set stored)
        param: target_features  List of target feature names which is required  (Feature set wanted for model)
        """
        return [
            source_features.index(f_i)
            for f_i in target_features
            if f_i in source_features
        ]

    def map_from_input_to_model_feature_sets(self, source_feature_set: list) -> None:
        """
        source_feature_set: Feature set given by the data set or such
        """

        # No generate the an ordered mapping from the input (source) feature set to each model's feature set
        for target_feature_set in self.org_model_feature_sets:
            self.feature_set_indexs.append(
                self.find_feature_ids(source_feature_set, target_feature_set)
            )

            # Check all features for each model.
            for f_i in target_feature_set:
                assert f_i in source_feature_set

        # Blow up if empty
        assert self.feature_set_indexs

    def predict(self, X_coords, X_features, voxel_size):

        # Preditions with probability
        preds_arr = np.empty((0, X_coords.shape[0]))

        # Prescale the data and dont have to deal with it

        for model, scaler, feature_ids in zip(
            self.models, self.scalers, self.feature_set_indexs
        ):
            assert self.models

            y_pred_binary, y_pred_prob = model.predict_te_classification(
                X_coords=X_coords,
                X_features=scaler.transform(X_features[:, feature_ids]).astype(
                    np.float32
                ),
                voxel_size=voxel_size,
                device=self.device,
            )

            preds_arr = np.vstack((preds_arr, y_pred_prob[np.newaxis, :]))
        return np.mean(preds_arr, axis=0), np.var(preds_arr, axis=0)

    def fast_predict(self, X_coords, X_features, voxel_size):
        """ Fast prediction of ensemble. Uses a single scaler rather than multiple-different scalers as predicts 

        Args:
        X_coords (np.array): _description_
        X_features (np.array): _description_
        voxel_size (float23): Voxel size [m]
        """

        # Conversion to SparseTensor used for all models. This should speedup the inference time
        s_coords = X_coords // voxel_size
        coords = torch.tensor(s_coords, dtype=torch.int)
        feats = torch.tensor(self.scalers[0].transform(
            X_features[:, self.feature_set_indexs[0]]).astype(np.float32), dtype=torch.float)
        stensor = SparseTensor(coords=coords, feats=feats)
        stensor = sparse_collate([stensor]).to(self.device)

        pred_arr = np.vstack([self.mpredict(model=model, stensor=stensor)
                              for model in self.models])

        return np.mean(pred_arr, axis=0), np.var(pred_arr, axis=0)

    def mpredict(self,
                model,
                stensor: SparseTensor,
                ):
        """ Fast prediction without casting and depending on model type. Hack to test speedup if model is
            copied onto the memory directly. May not be an issue with an orin/nvidia device with shared memory
        Args:
            model (nn.Module): Deep learning model
            stensor (SparseTensor): Sparse tensor, already passed to deviceS

        Returns:
            Returns the binary and logits: _description_
        """
        y_pred = None
        if "THM" in model.__class__.__name__:
            logits = model(stensor)[0]
        elif "LMCD" in model.__class__.__name__:
            logits = model(stensor)
        else:
            msg = "Could not find a valid model type"
            raise ValueError(msg)

        return logits.F.softmax(dim=1)[:, 1].cpu().detach().numpy()


def parse_iter_model_from(dir: Path, cv_start: int, cv_number: int):
    """Loads the incremental model number starting at cv start and for n models"""
    dir_path = Path(dir)
    # Force nv_number to be at least 1
    cv_number = max(cv_number, 1)

    model_files = []
    for cv_nbr in range(cv_start, cv_start + cv_number):
        model_config_file = dir_path / f"model_config_cv_{cv_nbr}.yaml"

        if not model_config_file.exists():
            continue

        model_files.append((model_config_file, cv_nbr))

    return model_files
