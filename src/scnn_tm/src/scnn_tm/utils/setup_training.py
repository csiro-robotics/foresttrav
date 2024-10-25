from pathlib import Path

import numpy as np
import pandas as pd
import torch as torch
import yaml
from joblib import dump
from pytorch_lightning.loggers import WandbLogger
from tabulate import tabulate

from scnn_tm import utils
from scnn_tm.models.DataAugmenter import DataPatchAugmenter
from scnn_tm.models.FtmMePlModule import (
    ForestTravPostProcessdDataPLModule,
    setup_data_set_factory,
)
from scnn_tm.utils import (
    select_foresttrav_model,
    test_data_set_by_key,
    train_data_set_by_key,
)
from torchsparse.utils.collate import sparse_collate_fn


def setup_data_factory_scnn(params, scaler):
    """ Loads the data_sets_factory for the training

    Args:
        params (_type_): _description_
        scaler (_type_): _description_

    Raises:
        Exception: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    data_set_factory = None
    params.train_data_sets = train_data_set_by_key(params.data_set_key, params.voxel_size)
    params.test_data_sets = test_data_set_by_key(params.data_set_key, params.voxel_size)
    
    # Check the 
    check_n_splits(params)

    # TODO: Split out the dependencies
    data_set_factory = setup_data_set_factory(train_strategy=params.training_strategy_key, scaler = scaler,  params=params)

    return data_set_factory

def check_n_splits(params):
    
    # Check for the number of splits
    if "scene" in params.training_strategy_key:
        params.num_splits = len(params.data_sets) - len( utils.data_set_name_without_nte())

    else:
        params.num_splits = params.max_cv


def setup_model(params):
    """Setups the model and scaler if pretrained model

    Args:
        params (_type_): _description_

    Returns:
        dict: Contains "model" and "scaler"
    """

    model = None
    scaler = None
    feature_set = None

    # Return if found a model
    if hasattr(params, "pre_trained_model_file"):
        raise ValueError("PreTrained model not implemented")
        # print(f"[Setup Model] Using model to finetune")
        # model, scaler, feature_set = utils.load_unet_model_from_yaml(
        #     params.pre_trained_model_file
        # )
        # params.set_new_attribute("feature_set", feature_set)
        return {"model": model, "scaler": scaler}

    # Setup the feature sets used for the model
    print("[Setup Model]: Setting up a new model to be trained")
    if hasattr(params, "feature_set_key"):
        feature_set = utils.generate_feature_set_from_key(params.feature_set_key)
    else:
        print("[Setup Model]: Using default feature set: [ohm]")
        feature_set = utils.generate_feature_set_from_key("ohm")
    params.feature_set = feature_set
    params.nfeatures = len(feature_set)

    # Setup the skip connections
    model_skip_connection = []
    if hasattr(params, "model_skip_connection_key"):
        model_skip_connection = utils.parse_skip_connection_from_key(
            params.model_skip_connection_key
        )
    else:
        print("[Setup Model]: Setting up default skip connections")
        model_skip_connection = [1, 1, 1, 1, 1]
    params.model_skip_connection = model_skip_connection

    # Setup model
    model = select_foresttrav_model(params)

    # Setup Data Loader
    return {"model": model, "scaler": scaler,}


def setup_data_augmentor(params):
    if not params.use_data_augmentation:
        return None
    
    
    

def setup_pl_module(params, model_files):
    
    pl_module = None
    if ("UNet" in params.model_name) and ("MCD" in params.model_name):
        
        pl_module = None
        return ForestTravPostProcessdDataPLModule( 
            model = model_files["model"],
            data_loader = model_files["data_loader"],
            lr =  params.lr, 
            weight_decay = params.weight_decay,
            voxel_size=params.voxel_size,
            batch_size = params.batch_size,
            params = params,
        )

    else:
        msg = "No PL module could be found with the given name"
        raise ValueError(msg)

def set_cv_exp_dir(params):
    now = pd.to_datetime("today").strftime("%Y_%m_%d_%H_%M")[2::]
    model_cv_name = f"{now}_{params.model_name}_{params.feature_set_key}_{params.training_strategy_key}"
    params.cv_exp_dir = Path(params.experiment_root_dir)/ Path(model_cv_name)

def setup_wandb_logger(params):
    
    now = pd.to_datetime("today").strftime("%Y_%m_%d_%H_%M")[2::]
    params.model_file_name_cv = f"{now}_{params.model_name}_{params.feature_set_key}_{params.training_strategy_key}_{params.cv_nbr}"
    print(params.model_file_name_cv)
    
    wandb_logger = None
    if params.debug:
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


def evaluate_cv_run(params, model_files: dict):
    """_summary_

    Args:
        params (_type_): _description_
        model_files (dict): _description_
    """
    # Set model to evaluation
    model =model_files["model"]
    model.to(params.accelerator)
    model.eval()
    
    # Batch data as sparse tensor
    test_data_set = model_files["data_loader"].test_data_set
    dl_batched = [ test_data_set[i] for i in range(len(test_data_set))]
    batched_data = sparse_collate_fn(dl_batched)

    # TODO: Make this agnostic for TE classification
    logits = model(batched_data["input"].to(params.accelerator))
    _, y_pred = logits.F.max(1)
    y_pred = y_pred.cpu().numpy()
    
    y_target = batched_data["label"].F.squeeze().cpu().numpy()
    return y_pred, y_target

def simple_model_save(params, model_file):
    """Simple function to write out a model compatible with ME learning"""

    model_out_dir = params.cv_exp_dir
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model = model_file["model"]
    model_pl_file = model_out_dir / f"{model.__class__.__name__}_cv_{params.cv_nbr}.pl"
    torch.save(model.state_dict(), model_pl_file)

    # Save the data scaler
    scaler_file = model_out_dir / f"scaler_cv_{params.cv_nbr}.joblib"
    dump(model_file["scaler"], scaler_file)

    # Save the config file
    config_file = model_out_dir / f"model_config_{params.cv_nbr}.yaml"


    config_file.touch()
    with open(config_file, "w") as file_descriptor:
        yaml_params = {}
        yaml_params["model_name"] = model.__class__.__name__
        yaml_params["model_nfeature_enc_ch_out"] = model.ENCODER_CH_OUT[0]
        yaml_params["model_skip_connection_key"] = f"s{model.SKIP_CONNECTION.count(1)}"
        yaml_params["feature_set_key"] = params.feature_set_key

        # We make the assumption that the model is in the same dir
        yaml_params["torch_model_path"] = str(model_pl_file)
        yaml_params["scaler_path"] = str(scaler_file)
        yaml.safe_dump(yaml_params, file_descriptor)

def evaluate_cv_experiment(params, model_files):
    """_summary_

    Args:
        params (_type_): _description_
        model_files (_type_): _description_
    """
    exp_dir = params.cv_exp_dir
    result_file = params.cv_exp_dir / Path("exp_results.txt")
    header = ["Metric"] + [key for key in model_files.keys()] + [" mean", "std"]
    mcc_score = [round(values["mcc_score"],params.metric_rounding) for key, values  in model_files.items()] 
    mcc_score = ["mcc-score"] + mcc_score + [round(np.mean(np.array(mcc_score)),2), round(np.std(np.array(mcc_score)),params.metric_rounding)]
    f1_score =  [ round(values["f1_score"],params.metric_rounding) for key, values in model_files.items()]
    f1_score = ["f1-score"] + f1_score + [ round(np.mean(np.array(f1_score)),params.metric_rounding), round(np.std(np.array(f1_score)),params.metric_rounding)]
    
    scores = [mcc_score, f1_score]
    f = open(result_file, "a+")
    f.write(f"\n \n Final report over for stacked cv's \n")
    f.write(tabulate(scores, headers=header))
    
    f.close()

    