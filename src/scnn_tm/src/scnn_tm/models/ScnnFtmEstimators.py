# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

from pathlib import Path
import numpy as np
import torch

from scnn_tm.utils.scnn_io import (
    load_cv_unet_model_from_yaml,
    load_unet_model_from_yaml,
)
from scnn_tm.models.UnetPlMCD import predict_unet_device


class ScnnFtmEstimator:
    def __init__(self, model_file: Path, device="cuda", cv_nbr = None) -> None:
        self.device = device
        
        if not model_file:
            raise ValueError(model_file)
        
        if cv_nbr == None: 
            self.model, self.scaler, self.feature_set = load_unet_model_from_yaml(
            model_file)
        else:
            self.model, self.scaler, self.feature_set = load_cv_unet_model_from_yaml(
            model_file, cv_nbr=cv_nbr)
            
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu"
        )
        # print(f"Setting model up for {self.device} inference")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, X_coords, X_features, voxel_size):
        X_features_scaled = self.scaler.transform(X_features)
        return predict_unet_device(
            model=self.model,
            X_coord=X_coords,
            X_features=X_features_scaled,
            voxel_size=voxel_size,
            device=self.device,
        ), np.empty((0, 1))


class ScnnFtmEnsemble:
    """ Ensemble method to use n time the same feature set for n different estimators.
    
        Note: The ensemble is responsible to address the 
    """

    def __init__(self, model_config_files: list, device: str, input_feature_set: list) -> None:

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

            model, scaler, model_feature_set = load_cv_unet_model_from_yaml(model_config_file, cv_number)
            model.to(device)
            model.eval()
            self.models.append(model)
            self.scalers.append(scaler)
            self.org_model_feature_sets.append(model_feature_set)
        
        # We need to keep the indexes we need for the feature set based on the ftm_feature set structure
        self.map_from_input_to_model_feature_sets(input_feature_set)
        
    def find_feature_ids(self, source_features, target_features):
        """ Returns an list of indexes matching the desired features to the source features.
        
        param: source_features  List of feature names for the feature of the data set (Data set stored)
        param: target_features  List of target feature names which is required  (Feature set wanted for model)
        """
        return [source_features.index(f_i) for f_i in target_features if f_i in source_features]

    def map_from_input_to_model_feature_sets(self, source_feature_set:list )->None:
        """
            source_feature_set: Feature set given by the data set or such
        """
              
        # No generate the an ordered mapping from the input (source) feature set to each model's feature set 
        for target_feature_set in self.org_model_feature_sets:
            self.feature_set_indexs.append(self.find_feature_ids(source_feature_set, target_feature_set))
            
            # Check all features for each model. 
            for f_i in target_feature_set: 
                assert f_i in source_feature_set
        
        # Blow up if empty
        assert self.feature_set_indexs

    
    def predict(self, X_coords, X_features, voxel_size):

        preds_arr = np.empty((0, X_coords.shape[0]))
        for model, scaler, feature_ids in zip(
            self.models, self.scalers, self.feature_set_indexs
        ):
            assert self.models
            
            predicitons = predict_unet_device(
                model=model,
                X_coord=X_coords,
                X_features=scaler.transform(X_features[:, feature_ids]).astype(np.float32),
                voxel_size=voxel_size,
                device=self.device,
            )
            preds_arr = np.vstack((preds_arr, predicitons[np.newaxis, :]))

        return np.mean(preds_arr, axis=0), np.var(preds_arr, axis=0)


#TODO: Generate a mixed ScnnForestTravEnsemble method

def parse_cv_models_from_dir(dir: Path, cv_total):
    """  For an experiment dir parse all the models that have been generated. Genearates the format to load the cv_model.
    
    output: list of model_config.yaml and cv_nbrs. 
    """
    dir_path = Path(dir)
    
    model_files = []
    for cv_nbr in range(cv_total):
        model_file = dir_path / "model_config.yaml"
        
        scaler_test = dir_path / f"scaler_cv_{cv_nbr}.joblib"
        
        if not scaler_test.exists():
            continue
        
        model_files.append((model_file,cv_nbr ))
    
    #  
    return model_files

def parse_iter_model_from( dir: Path, cv_start: int, cv_number: int):
        """ Loads the incremental model number starting at cv start and for n models
            
        """
        dir_path = Path(dir)
    
        model_files = []
        for cv_nbr in range(cv_start, cv_start+cv_number):
            model_file = dir_path / "model_config.yaml"

            scaler_test = dir_path / f"scaler_cv_{cv_nbr}.joblib"

            if not scaler_test.exists():
                continue
            
            model_files.append((model_file,cv_nbr ))

        #  
        return model_files
    
    