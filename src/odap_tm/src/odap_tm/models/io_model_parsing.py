# Copyright (c) 2023
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

from pathlib import Path

import torch
import torch as torch
import yaml
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

from odap_tm.models import UnetMultiHeadMid, io_model_parsing
from scnn_tm import utils as scnn_utils, utils
from scnn_tm.models import UnetPlMCD
from dataclasses import dataclass, fields

MODEL_MAP = {
    "UNet4THM": UnetMultiHeadMid.UNet4THM,
    "UNet3THM": UnetMultiHeadMid.UNet3THM,
    # Old models
    "UNet3LMCD": UnetPlMCD.UNet3LMCD,
    "UNet4LMCD": UnetPlMCD.UNet4LMCD,
    "UNet5LMCD": UnetPlMCD.UNet5LMCD,
    # PU learning model
    # "PUVoxTravUnet4L": PUVoxTrav.PUVoxTravUnet4L,
}

MODEL_MAP_PARAMS = {
    "UNet4THM": UnetMultiHeadMid.UNetTHMParams,
    "UNet3THM": UnetMultiHeadMid.UNetTHMParams,
    # Old models
    "UNet3LMCD": UnetPlMCD.UNetMCDParams,
    "UNet4LMCD": UnetPlMCD.UNetMCDParams,
    "UNet5LMCD": UnetPlMCD.UNetMCDParams,
    # PU learning model
    # "PUVoxTravUnet4L": PUVoxTrav.PUVoxTravUnet4L,
}


# TODO: The load model from yaml is to complicated, remove dependencies on the param file...
# TODO: Differentiate the two loading files for the models,
#       - load_model_from_yaml
#       - load_cv_model...

def default_model_params():

    default_model_params_ = {
        "model_name": " ",
        "model_nfeature_enc_ch_out": 0,
        "feature_set": 0,
        "nfeatures": 0,
        "model_skip_connection": 0,
        "model_stride": 0,
        "scaler_path": 0,
        "torch_model_path": 0,
        "feature_set_key": None,
        "loss_function_tag": None,
        # model_skip_connection_key
    }
    return scnn_utils.Dict2Class(default_model_params_)


#################  HELPER FUNCTIONS ####################


def dict_to_model_params(dparams: dict, model_name) -> dataclass:
    """Generates the paramers for a data class from a dict

    Args:
        dparams (dict): Dictionary with all the needed parameters
        model_name (_type_): Name/Tag of the model, corresponds to class name

    Returns:
        dataclass: _description_
    """
    relevant_data = {
        k: v
        for k, v in dparams.items()
        if k in MODEL_MAP_PARAMS[model_name].__annotations__
    }

    return MODEL_MAP_PARAMS[model_name](**relevant_data)


def dict_to_params(dparams: dict, map_to_params: dict, tag: str) -> dataclass:

    relevant_data = {
        k: v for k, v in dparams.items() if k in map_to_params[tag].__annotations__
    }

    return map_to_params[tag](**relevant_data)


def obj_to_params(dparams: dict, class_ref) -> dataclass:
    cls_field_names = {f.name for f in fields(class_ref)}
    init_params = {name: getattr(dparams, name, None) for name in cls_field_names}
    return class_ref(**init_params)


def model_selection(params: object):
    """Selects the correct model based on the model name

    Args:
        params (obj): Contains all the parameters

    Raises:
        ValueError: If non-existing value is called

    Returns:
        dict:   Returns a dictionary with the "model" and "scaler"
    """
    # Auto conversion if in the wrong format
    if type(params) is dict:
        params = dict_to_model_params(params, params["model_name"])

    model_name = params.model_name

    if model_name in MODEL_MAP.keys():
        return MODEL_MAP[model_name](
            nfeatures=params.nfeatures,
            encoder_nchannels=params.model_nfeature_enc_ch_out,
            skip_conection=params.model_skip_connection,
            stride=params.model_stride,
        )
    else:
        msg = f"Could not find {model_name}"
        raise KeyError(msg)


def find_in_dir(file_path: Path, dir):
    # If the file does not exist try to load it from the local dict
    new_file = Path(dir) / Path(file_path).name
    return new_file if new_file.exists() else Path(" ")


#################  PARSING YAML AND CONFIG FILES ####################

# This loads the model params
def model_config_from_file(
    config_file: Path,
    cv_num: int = -1,
    convert_dic_to_obj: bool = True,
):
    """Loads the model config from a yaml file. The file needs to be a valid model confiurations,

    Args:
        yaml_file (Path):           Path to file (absolute)
        cv_num (int):               If a specific version/cv number should be loaded.
        convert_dic_to_obj (bool):  Flag to force conversion to object

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    #
    config_file = Path(config_file)
    f = open(config_file)
    params = yaml.safe_load(f)

    # Check if all fields are available
    required_fields = [
        "model_name",
        "cv_num",
        "feature_set_key",
        "model_skip_connection_key",
        "model_stride",
        "model_nfeature_enc_ch_out",
    ]
    missing_attr = [attr for attr in required_fields if not hasattr(params, attr)]

    if not missing_attr:
        msg = f"Could'nt find the follwing attritures in the config file {missing_attr}"
        raise ValueError(msg)

    #  Load all the features needed
    params["feature_set"] = scnn_utils.generate_feature_set_from_key(
        params["feature_set_key"]
    )
    params["nfeatures"] = len(params["feature_set"])
    params["model_skip_connection"] = scnn_utils.parse_skip_connection_from_key(
        params["model_skip_connection_key"]
    )
    params["model_stride"] = [1, 2, 2, 2, 2, 2]

    # Parse the model name and scaler path, assuming that they are in the same directry
    if cv_num < 0:
        cv_num = params["cv_num"]

    model_path = config_file.parent / Path(params["model_name"] + f"_cv_{cv_num}.pl")
    scaler_path = config_file.parent / Path(f"scaler_cv_{cv_num}.joblib")

    # Check if the files exist
    if not (model_path.exists() and scaler_path.exists()):
        msg = f"Files do not exist in directory   {config_file.parent} \n Scale file {scaler_path} \n model file{model_path}"
        raise FileNotFoundError(msg)

    # Update the scaler and model path
    params["model_file"] = model_path
    params["scaler_path"] = scaler_path

    f.close()

    # Should convert dic to cobject with members?
    if convert_dic_to_obj:
        return scnn_utils.Dict2Class(params)

    return params


def load_from_from_yaml(params, config_file, as_dict=False):
    """Loads the model from a specific yaml config file.

    Args:
        params (obj): Object containing all the parameters
        config_yaml (dict): Contains all the objects used for training

    Return:

    """
    f = open(config_file)
    yaml_params = yaml.safe_load(f)

    # Setup the default params if not initialised
    if params is None:
        params = default_model_params()

    # Load all the params into the param file
    params.model_name = yaml_params["model_name"]
    params.model_nfeature_enc_ch_out = yaml_params["model_nfeature_enc_ch_out"]
    params.feature_set_key = yaml_params["feature_set_key"]
    params.feature_set = scnn_utils.generate_feature_set_from_key(
        yaml_params["feature_set_key"]
    )
    params.nfeatures = len(params.feature_set)
    params.model_skip_connection = scnn_utils.parse_skip_connection_from_key(
        yaml_params["model_skip_connection_key"]
    )
    params.model_stride = yaml_params["model_stride"] = [1, 2, 2, 2, 2, 2]
    params.scaler_path = yaml_params["scaler_path"]
    params.torch_model_path = yaml_params["torch_model_path"]
    
    # Load loss function tag if avaialble:
    if hasattr(params,'loss_function_tag') and  hasattr(yaml_params,'loss_function_tag'): 
        params.loss_function_tag = yaml_params['loss_function_tag']

    f.close()

    # Find the torch path if the model does not exist as in the origial location,
    if not Path(params.torch_model_path).exists():
        params.torch_model_path = find_in_dir(
            params.torch_model_path, Path(config_file).parent
        )
    model = model_selection(params=params)
    model.load_state_dict(torch.load(params.torch_model_path))

    if not Path(params.scaler_path).exists():
        params.scaler_path = find_in_dir(params.scaler_path, Path(config_file).parent)

    scaler = load(params.scaler_path)

    # Return in the desired format
    if as_dict:
        return {"model": model, "scaler": scaler, "feature_set": params.feature_set}
    return model, scaler, params.feature_set


################# LOADING NN ####################


def load_model_from_config(model_config: object):
    """Loads a TE calssification model from config_config (obj)

    Args:
        model_config(obj):  Object with the requiered fields:
            -

    """

    # Load the base model with the correct architecture
    model = model_selection(model_config)

    # Load state dict and set wrights
    model.load_state_dict(torch.load(model_config.model_file))
    model.eval()

    # Load the
    scaler = load(model_config.scaler_path)

    if not isinstance(scaler, StandardScaler):
        raise TypeError("Wrong type of StandartSaler")

    return model, scaler, model_config.feature_set


def setup_model(params):
    """Setups the model for training, either using pre_trained or completly new model

    Args:
        params (_type_): _description_

    Returns:
        dict: Contains "model" and "scaler"
    """

    model = None
    scaler = None
    feature_set = None

    # Return if found a model
    if params.use_pretrained_model:
        return setup_pre_trained_model(params, as_dict=True)

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
    model = io_model_parsing.model_selection(params)

    return {"model": model, "scaler": None}


def simple_model_save(out_dir: Path, params, model_file):
    """Simple function to write out a model compatible with ME learning"""
    # def simple_model_save(model_file, out_dir, cv_number, fields = [] ):
    model_out_dir = Path(out_dir)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model = model_file["model"]
    model_pl_file = model_out_dir / f"{model.__class__.__name__}_cv_{params.cv_nbr}.pl"
    torch.save(model.state_dict(), model_pl_file)

    # Save the data scaler
    scaler_file = model_out_dir / f"scaler_cv_{params.cv_nbr}.joblib"
    if not isinstance(model_file["scaler"], StandardScaler):
        msg = "Scaler is not valid"
        raise TypeError(msg)
    dump(model_file["scaler"], scaler_file)

    # Save the config file
    config_file = model_out_dir / f"model_config_cv_{params.cv_nbr}.yaml"

    config_file.touch()
    with open(config_file, "w") as file_descriptor:
        yaml_params = {}
        yaml_params["model_name"] = model.__class__.__name__
        yaml_params["model_nfeature_enc_ch_out"] = model.ENCODER_CH_OUT[0]
        yaml_params["model_skip_connection_key"] = f"s{model.SKIP_CONNECTION.count(1)}"
        yaml_params["feature_set_key"] = params.feature_set_key
        yaml_params["cv_num"] = params.cv_nbr

        # We make the assumption that the model is in the same dir
        yaml_params["torch_model_path"] = str(model_pl_file)
        yaml_params["scaler_path"] = str(scaler_file)
        
        # Check the config for the pre-trained models
        if hasattr(params,'use_pretrained_model') and params.use_pretrained_model:
            yaml_params['base_model_config'] = str(params.base_model_config)
        
        # Check for loss function tag
        if hasattr(params,'loss_function_tag'): 
            yaml_params['loss_function_tag'] = params.loss_function_tag
        
        
        yaml.safe_dump(yaml_params, file_descriptor)


# TODO: Override the params model names and other parameters
def setup_pre_trained_model(params: object, as_dict=False):
    """ Setup the pre-trained model and different strategies.
    
    
    Args:
        params (obj): Parameters to load all the models and finetune
        as_dict (bool): Flat to returnt the output as dict
    
    Return:
        model (obj): Model
        scaler (obj): Scaler for the model
        feature_set: Feature set used as list of feature names
    
    """
    print(f"[Setup Model] Using model to finetune")
    model = None 
    scaler = None
    feature_set = None
    
    # cv_finetune 
    if  hasattr(params,"cv_finetune") and params.cv_finetune:
        cv_num = params.cv_nbr
        
        # For the cv_finetune assume the parent of the model_file is in the parent "model_base_file"
        cv_model_file = f"model_config_cv_{cv_num}.yaml"
        model_file_path = Path(params.base_model_config).parent / cv_model_file
        print(f"[Setup CV Finetune] Setting up finetune model for {cv_num}")
        
        # Load the model
        model, scaler, feature_set = io_model_parsing.load_from_from_yaml(params, model_file_path)
        params.feature_set = feature_set

    else:
        print(f"[Setup Finetune]: Using model single base model to finetune")
        model, scaler, feature_set = io_model_parsing.load_from_from_yaml(params, params.base_model_config)
        params.feature_set = feature_set
    
        
    if as_dict:
        return {"model": model, "scaler": scaler, "feature_set": feature_set}
    return model, scaler, feature_set
