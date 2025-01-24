# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

import argparse
import copy
import yaml 
from pathlib import Path

from odap_tm.train_model import train_cv

# TODO: The model configs need the loss function for fine tuning
def finetune_cv_models(_params, ft_data_dir):
    
    # Make the tag combinations 
    tags = ["indust", "dense", "sparse"]
    tag_combinations = [[tag, ft_tag] for tag in tags for ft_tag in tags if tag != ft_tag]
    
    ft_params = load_cv_params(ft_data_dir)
    for tag_comb in tag_combinations: 
        # Set the out dir 
        base_models = ft_params[f"{tag_comb[0]}_base"]
        for bmodel in base_models:
            
            if "THM" in Path(bmodel).name:
                continue
        
            # Copy the params so there is no leaking, set experiment
            params = copy.deepcopy(_params)    
            params.experiment_root_dir = Path(params.experiment_root_dir) / f"bm_{tag_comb[0]}_ft_{tag_comb[1]}" 
        
            # Set the model_config, train data and test data for th
            params.base_model_config = Path(bmodel) / "model_config_cv_0.yaml"
            params.train_data_sets = ft_params[f"{tag_comb[1]}_data"]["train_data_sets"]
            params.test_data_sets = ft_params[f"{tag_comb[1]}_data"]["test_data_sets"]
            
            train_cv(params)
  
        
def base_cv_models(_params):
    tags = ["full"]
    models = ["UNet4THM", "UNet4LMCD", ]
    feature_set_keys = ["ohm_mr", "occ_int_hm_mr"]

    ft_params = load_cv_params(DATA_SET_DEF)
    for tag in tags: 
        
        dir_tag =f"{tag}_base"

        for model_tag in models:
            
            for feature_set_key in feature_set_keys:
                    
                # Copy the params so there is no leaking, set experiment
                params = copy.deepcopy(_params)    
                params.experiment_root_dir = Path(params.experiment_root_dir) / dir_tag 
                params.model_name = model_tag
                params.feature_set_key = feature_set_key

                # Set the model_config, train data and test data for th
                params.train_data_sets = ft_params[f"{tag}_data"]["train_data_sets"]
                params.test_data_sets = ft_params[f"{tag}_data"]["test_data_sets"]

                # TH
                train_cv(params)
 
        
def load_cv_params(file_path: Path):
    config = None
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config      
        

###################################################
#        Main Loop and Arguments
###################################################


## Training of models HL
CONFIG_FILE = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_05_02/base_config.yaml"
DATA_SET_DEF = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_05_02/hl_data_set_definition.yaml"

## Training of models LFE
# CONFIG_FILE = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_05_02/lfe_base_config.yaml"
# DATA_SET_DEF = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_05_02/lfe_data_set_definition.yaml"


##### FINETUNE CONFIG
## Fintune HL->HL
# CONFIG_FILE = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_05_02/finetune/2024_5_5_hl_hl_base.yaml"
# DATA_SET_FILES = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_05_02/finetune/2024_5_4_hl_hl_cv_finetuning.yaml" 

## Fintune HL->LFE
# CONFIG_FILE = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_05_02/finetune/2024_5_5_hl_lfe_base.yaml"
# DATA_SET_FILES = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_05_02/finetune/2024_5_4_hl_lfe_cv_finetuning.yaml" 

## Fintune LFE->LFE
# CONFIG_FILE = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_05_02/finetune/2024_5_5_lfe_lfe_base.yaml"
# DATA_SET_FILES = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_05_02/finetune/2024_5_4_lfe_lfe_cv_finetuning.yaml" 

## Finetunr HL->HL ProbLoss
# CONFIG_FILE = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_05_02/finetune/2024_5_5_hl_hl_base.yaml"
# DATA_SET_FILES = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_05_02/finetune/2024_5_4_hl_hl_cv_finetuning.yaml" 

parser = argparse.ArgumentParser()


if __name__ == "__main__":
    # This loads the yaml file and maintains them in global scope. This allows wandb to change them
    # (CHECK): wand can change the files and this is
    # TODO: This is a dangerous parsing operation that wont allow much flexibility
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
        for key, value in config.items():
            parser.add_argument(f"--{key}", type=type(value), dest=key, default=value)

    # Parse the command line arguments
    args = parser.parse_args()
    args.cv_exp_dir = None

    base_cv_models(args)
    # finetune_cv_models(args, DATA_SET_FILES)
