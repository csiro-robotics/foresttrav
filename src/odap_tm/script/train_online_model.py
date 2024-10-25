# Copyright (c) 2023
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

from pathlib import Path

from odap_tm.models.OnlineLearingModule import OnlineActiveLearingModule
from scnn_tm.utils import load_yamls_as_struct
from odap_tm.setup_training import set_cv_exp_dir
from odap_tm.models.io_model_parsing import simple_model_save
import re
import numpy as np
import pandas as pd
import argparse
import yaml
from copy import deepcopy

def load_timestamp_data_ordered(root_dir: Path, ext="hdf5"):
    """Load the hdf5 data in ascending order
    Args:
        root_dir (Path): Root of the data to be loaded
        ext (str): File extension to be loaded in ascending order

    Return:
        timestamps: (list(float)): Timestamps in ascending order (system time)
        data_files (list): Files to be loaded in ascending order
    """

    data_path = Path(root_dir)

    data_files = sorted(data_path.glob(f"*.{ext}"))

    if not data_files:
        msg = f"Could not find files in {root_dir}"
        raise FileNotFoundError(msg)

    # Regular expression pattern to match numbers
    pattern = r"\d+\.?\d*"
    timestamps = [
        float(re.findall(pattern, Path(file_name).stem)[-1]) for file_name in data_files
    ]

    if float(timestamps[-1]) - float(timestamps[0]) < 0:
        msg = "The files are not in ascending order."
        raise ValueError(msg)

    return {"timestamps": timestamps, "data_files": data_files}


def set_odap_exp_dir(params):
    """Sets up a directory with the current timestamp and model name, for an unique identifier. YYYY_MM_DD_HH_MM_MODEL_FSET_CV"""
    now = pd.to_datetime("today").strftime("%Y_%m_%d_%H_%M")[2::]

    tag = "0"
    if params.use_pretrained_model:
        if "dense" in params.base_model_config:
            tag = "dense"
        elif "sparse" in params.base_model_config:
            tag = "sparse"
        elif "indust" in params.base_model_config:
            tag = "indust"
        else:
            tag = "1"  # Default value to just denote a model has been used



    model_cv_name = f"{now}_{params.model_name}_{params.feature_set_key}_cl_{params.cont_learning}_bm_{tag}"
    
    # Add the prob loss tag if available
    if "THM" in params.model_name:
        loss_tag = 'THL'
        if "Prob" in params.loss_function_tag: 
            loss_tag = "PL"
        model_cv_name += loss_tag

    params.cv_exp_dir = Path(params.experiment_root_dir) / Path(model_cv_name)
    return Path(params.experiment_root_dir) / Path(model_cv_name)


def train_online_models(
    learning_params: object,
    online_data_root_dir: Path,
    cont_learning: bool,
):
    # Load the data
    loaded_online_data = load_timestamp_data_ordered(online_data_root_dir)

    online_learner = OnlineActiveLearingModule(learning_params)

    # Update the online learner
    mcc_scores = []
    is_first_iteration = True and cont_learning

    # Setting the model up for contioual learning and flag for the first iteration
    out_dir = set_odap_exp_dir(params=learning_params)

    for time_stamp, new_online_data in zip(
        loaded_online_data["timestamps"], loaded_online_data["data_files"]
    ):
        # Update data set and
        mcc_scores_i = []
        online_learner.update_online_dataset([new_online_data])
        for cv_i in range(learning_params.max_cv):

            # Train the new model
            model_files_i, mcc_i = online_learner.train_new_model(cv_i)
            
            if model_files_i is None:
                continue

            # If the previous mcc score where better, abandon model:
            mcc_scores_i.append(mcc_i)

            # Override the previous model if we want to
            # TODO: Should we override the model if it is worse? Or just ignore and keep going?
            if cont_learning:
                online_learner.base_models[cv_i] = model_files_i

            # Setup the outdir after loading the models in case pre-trained models are used
            model_i_dir = Path(out_dir) / str(time_stamp)
            simple_model_save(
                model_i_dir, params=model_files_i["params"], model_file=model_files_i
            )
        # Add all the scores to the
        mcc_scores.append(mcc_scores_i)
        print( f"Latest MCC-scores: {mcc_scores[-1]}")

        # Hack to update learning rate: initial rate and finetune rate:
        if is_first_iteration and cont_learning:
            is_first_iteration = False

            # Adopt the learning rate for all the models, since the params file is laxely handled in the initialisation
            learning_params.lr = learning_params.ft_lr
            for cv_i in range(learning_params.max_cv):
                online_learner.base_models[cv_i]["params"].lr = (
                    online_learner.base_models[cv_i]["params"].ft_lr
                )

    ## Generate the mean
    mcc_arr = np.array(mcc_scores)
    mcc_mean = np.mean(mcc_arr, axis=1)
    mcc_std = np.std(mcc_arr, axis=1)
    ts_arr = np.array(loaded_online_data["timestamps"])

    result = np.hstack(
        (
            ts_arr[:, np.newaxis],
            mcc_arr,
            mcc_mean[:, np.newaxis],
            mcc_std[:, np.newaxis],
        )
    )

    header = (
        ["timestamp"]
        + [f"mcc_model_{i}" for i in range(learning_params.max_cv)]
        + ["mcc_mean", "mcc_var"]
    )
    file_name = out_dir / "results.csv"
    pd.DataFrame(data=result, columns=header).to_csv(file_name)
    
    # Save the params
    file_name = out_dir / "learning_params.yaml"
    with open(file_name, 'w') as yaml_file:
        yaml.dump(learning_params, yaml_file, default_flow_style=False)
        


def main(args):

    # Setup the ase model based on tags
    if args.use_pretrained_model:
        args.lr = args.base_lr  # Ensure that we are not scaling the learning rate

        # Parse model_config_file name so we can save correctly (kinda ugly)
        if "THM" in args.base_model_config:
            args.model_name = "UNet4THM"
        elif "LMCD" in args.base_model_config:
            args.model_name = "UNet4LMCD"
        else:
            args.model_name = "Unknown"

    # Train the online model
    train_online_models(
        online_data_root_dir=args.online_data_root_dir,
        learning_params=args,
        cont_learning=args.cont_learning,
    )


CONFIG_FILE = "/nve_ws/src/odap_tm/config/default_online_learning.yaml"
# RUN_FILE = "/nve_ws/src/odap_tm/config/experiment_odap.yaml"

parser = argparse.ArgumentParser()


if __name__ == "__main__":
    
    # This loads the yaml file and maintains them in global scope/global name space. This allows wandb to change them
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
        for key, value in config.items():
            parser.add_argument(f"--{key}", type=type(value), dest=key, default=value)

    # Parse the command line arguments
    args = parser.parse_args()
    args.cv_exp_dir = None
    
    main(args)
    
    # run_configs = None
    # with open(RUN_FILE, "r") as f:
    #     run_configs = yaml.safe_load(f)
        
    # for key, run_config in run_configs.items():
        
    #     # Magic loop to train all experiments
    #     params = deepcopy(args)
    #     for key, value in run_config.items():
    #         setattr(params, key, value)
    #         print(f"{key}: {value}")
        
    #     main(params)
