# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

from pathlib import Path
import numpy as np
import pandas as pd
import torch as torch
from tabulate import tabulate

from pytorch_lightning.loggers import WandbLogger
from odap_tm.models.PUVoxTrav import PUVoxTravPLM
from odap_tm.models.UnetMultiHeadPLModule import UnetMultiHeadPLModule
from scnn_tm import utils
from scnn_tm.models.FtmMePlModule import setup_data_set_factory
from scnn_tm.utils import test_data_set_by_key, train_data_set_by_key
from torchsparse.utils.collate import sparse_collate_fn


def process_data_set_key(params: dict) -> None:
    if not hasattr(params, "data_set_key"):
        return

    params["train_data_sets"] = train_data_set_by_key(
        params["data_set_key"], params.voxel_size
    )
    params["test_data_sets"] = test_data_set_by_key(
        params["data_set_key"], params.voxel_size
    )


def setup_data_factory(params, scaler):
    """Loads the data_sets_factory for the training

    Args:
        params (obj): _description_
        scaler (_type_): _description_

    Raises:
        Exception: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    data_set_factory = None
    # Parse the options data set keys and valid data set configureations
    if params.use_data_set_key:
        if "lfe" in params.data_set_key:
            a = 1

    # Check if we have loaded the data files
    if "train" in params.training_strategy_key:
        assert params.train_data_sets
    if "test" in params.training_strategy_key:
        assert params.test_data_sets
    else:
        assert params.train_data_sets

    check_n_splits(params)

    # TODO: Split out the dependencies
    data_set_factory = setup_data_set_factory(
        train_strategy=params.training_strategy_key, params=params, scaler=scaler
    )

    # Check the valid values

    return data_set_factory, data_set_factory.scaler


def check_n_splits(params):
    # Check for the number of splits
    if "scene" in params.training_strategy_key:
        params.num_splits = len(params.data_sets) - len(
            utils.data_set_name_without_nte()
        )
    else:
        params.num_splits = params.max_cv


def set_cv_exp_dir(params):
    """Sets up a directory with the current timestamp and model name, for an unique identifier. YYYY_MM_DD_HH_MM_MODEL_FSET_CV
    #TODO: This should have a return and not set the param inside.
    #TODO: Thid should be parametrised and not "params" pass
    """
    now = pd.to_datetime("today").strftime("%Y_%m_%d_%H_%M")[2::]
    model_cv_name = f"{now}_{params.model_name}_{params.feature_set_key}_{params.training_strategy_key}_epoch_{params.max_epochs}_fn_{params.model_nfeature_enc_ch_out}"
    params.cv_exp_dir = Path(params.experiment_root_dir) / Path(model_cv_name)
    return Path(params.experiment_root_dir) / Path(model_cv_name)


def setup_wandb_logger(params):
    now = pd.to_datetime("today").strftime("%Y_%m_%d_%H_%M")[2::]
    params.model_file_name_cv = f"{now}_{params.model_name}_{params.feature_set_key}_{params.training_strategy_key}_{params.cv_nbr}"
    print(params.model_file_name_cv)

    wandb_logger = None
    if params.debug or not params.use_wandb:
        wandb_logger = WandbLogger(
            name=params.model_file_name_cv,
            log_model=False,
            project=params.experiment_name,
            id=params.model_file_name_cv,
            dir=Path(params.experiment_root_dir),
            save_dir=Path(params.experiment_root_dir),
            mode="disabled",
        )
    else:
        wandb_logger = WandbLogger(
            name=params.model_file_name_cv,
            log_model=False,
            project=params.experiment_name,
            id=params.model_file_name_cv,
            dir=Path(params.experiment_root_dir),
            save_dir=Path(params.experiment_root_dir),
        )
    return wandb_logger


TWO_HEAD_MODELS = [
    "UNet4THM",
    "UNet3THM",
]
ONE_HEAD_MODELS = ["UNet3LMCD", "UNet4LMCD", "UNet5LMCD"]

# TODO: Different function calls and models are confusing the flow, explicit types and function signatures!


def evaluate_cv_run(params, model_files: dict):
    """_summary_

    Args:
        params (_type_): _description_
        model_files (dict): _description_
    """
    # Set model to evaluation
    model = model_files["model"]

    # Batch data as sparse tensor
    test_data_set = model_files["data_loader"].test_data_set
    return evaluate_cv_model(
        model=model, test_data_set=test_data_set, device=params.accelerator
    )


def evaluate_cv_model(model, test_data_set, device="cuda"):

    model.to(device)
    model.eval()
    dl_batched = [test_data_set[i] for i in range(len(test_data_set))]
    batched_data = sparse_collate_fn(dl_batched)

    # Easy(dirty) way to deal with the two types of models.
    y_pred = None
    if model.__class__.__name__ in TWO_HEAD_MODELS:
        logits = model(batched_data["input"].to(device))[0]
    elif model.__class__.__name__ in ONE_HEAD_MODELS:
        logits = model(batched_data["input"].to(device))
    else:
        msg = "Could not find a valid model type"
        raise ValueError(msg)

    _, y_pred = logits.F.max(1)

    y_target = batched_data["label"].F.squeeze().cpu().numpy()
    return y_pred.cpu().numpy(), y_target


def evaluate_cv_experiment(params, model_files):
    """_summary_

    Args:
        params (_type_): _description_
        model_files (_type_): _description_
    """
    exp_dir = params.cv_exp_dir
    result_file = params.cv_exp_dir / Path("exp_results.txt")
    header = ["Metric"] + [key for key in model_files.keys()] + [" mean", "std"]
    mcc_score = [
        round(values["mcc_score"], params.metric_rounding)
        for key, values in model_files.items()
    ]
    mcc_score = (
        ["mcc-score"]
        + mcc_score
        + [
            round(np.mean(np.array(mcc_score)), 2),
            round(np.std(np.array(mcc_score)), params.metric_rounding),
        ]
    )
    f1_score = [
        round(values["f1_score"], params.metric_rounding)
        for key, values in model_files.items()
    ]
    f1_score = (
        ["f1-score"]
        + f1_score
        + [
            round(np.mean(np.array(f1_score)), params.metric_rounding),
            round(np.std(np.array(f1_score)), params.metric_rounding),
        ]
    )

    scores = [mcc_score, f1_score]
    f = open(result_file, "a+")
    f.write(f"\n \n Final report over for stacked cv's \n")
    f.write(tabulate(scores, headers=header))

    f.close()
