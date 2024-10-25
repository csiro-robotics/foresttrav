from pathlib import Path

import torch

import yaml
from joblib import load

from scnn_tm.models import UnetPlMCD as UnetModels
import pandas as pd


# Turns a dictionary into a class
class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

    def set_new_attribute(self, key: str, value: any):
        setattr(self, key, value)


def load_yamls_as_struct(learnig_param_file: Path):
    """Load the learning param file as as struct. Each first level yaml key will be
    a attribute field of a class

    """
    learning_params = None
    if not (learnig_param_file.exists() and learnig_param_file.is_file()):
        raise ValueError(
            f"Could not find valid learning param file: {str(learnig_param_file)}"
        )

    with open(learnig_param_file, "r") as f:
        data_loaded = yaml.safe_load(f)
        learning_params = Dict2Class(data_loaded)
    return learning_params


def load_te_data_set(
    data_set_dir_path: Path,
    random_seed=12354,
):
    df_class1 = pd.read_csv(Path(data_set_dir_path) / "semantic_cloud_class_1.csv")
    df_class1["label"] = 0
    df_class2 = pd.read_csv(Path(data_set_dir_path) / "semantic_cloud_class_2.csv")
    df_class2["label"] = 1

    return pd.concat([df_class1, df_class2])


def load_cv_unet_model_from_yaml(
    yaml_file_path: Path,
    cv_nbr: int,
):
    f = open(yaml_file_path)
    params = yaml.safe_load(f)

    params["feature_set"] = generate_feature_set_from_key(params["feature_set_key"])
    params["model_skip_connection"] = parse_skip_connection_from_key(
        params["model_skip_connection_key"]
    )
    params["model_stride"] = [1, 2, 2, 2, 2, 2]

    args = Dict2Class(params)
    f.close()

    model = select_foresttrav_model(args, len(args.feature_set))

    # Load the torch model with the correct cv number
    torch_model_file = Path(yaml_file_path).parent / (
        Path(args.torch_model_path).stem[:-1] + f"{cv_nbr}.pl"
    )
    model.load_state_dict(torch.load(torch_model_file))
    model.eval()

    # Load the scaler with the correct cv number
    scaler_path = Path(yaml_file_path).parent / (
        Path(args.scaler_path).stem[:-1] + f"{cv_nbr}.joblib"
    )
    scaler = load(scaler_path)

    return model, scaler, args.feature_set


def load_unet_model_from_yaml(yaml_file_path: Path):
    f = open(yaml_file_path)
    params = yaml.safe_load(f)

    params["feature_set"] = generate_feature_set_from_key(params["feature_set_key"])
    params["nfeatures"] = len(params["feature_set"])
    params["model_skip_connection"] = parse_skip_connection_from_key(
        params["model_skip_connection_key"]
    )
    params["model_stride"] = [1, 2, 2, 2, 2, 2]

    args = Dict2Class(params)

    f.close()
    model = select_foresttrav_model(args)
    torch_model_file = Path(yaml_file_path).parent / args.torch_model_path
    model.load_state_dict(torch.load(torch_model_file))
    model.eval()

    # scaler = load_scaler(args.scaler_path)
    scaler_path = Path(yaml_file_path).parent / args.scaler_path
    scaler = load(scaler_path)

    return model, scaler, args.feature_set


def generate_feature_set_from_key(string_key):
    """Generates feature sets from keys and returns a list of features"""

    features_set_keys = {}

    if string_key.find("occ") >= 0:
        features_set_keys["mean_count"] = (1,)
        features_set_keys["occupancy_log_probability"] = 1

    if string_key.find("int") >= 0:
        features_set_keys["intensity_mean"] = (1,)
        features_set_keys["intensity_covariance"] = 1

    if string_key.find("perm") >= 0:
        features_set_keys["permeability"] = (1,)

    if string_key.find("mr") >= 0:
        features_set_keys["secondary_sample_count"] = (1,)

    if string_key.find("sr") >= 0:
        features_set_keys["secondary_sample_count"] = (1,)
        features_set_keys["secondary_sample_range_mean"] = (1,)
        features_set_keys["secondary_sample_range_std_dev"] = 1
        
    if string_key.find("hm") >= 0:
        features_set_keys["hit_count"] = 1
        features_set_keys["miss_count"] = 1
        

    if string_key.find("ev") >= 0:
        features_set_keys["ev_lin"] = (1,)
        features_set_keys["ev_plan"] = 1
        features_set_keys["ev_sph"] = (1,)

    if string_key.find("rgb") >= 0:
        features_set_keys["red"] = 1
        features_set_keys["green"] = 1
        features_set_keys["blue"] = 1

    if string_key.find("theta") >= 0:
        features_set_keys["theta"] = (1,)

    if string_key.find("lin") >= 0:
        features_set_keys["ev_lin"] = (1,)

    if string_key.find("ndt") >= 0:
        features_set_keys["intensity_mean"] = (1,)
        features_set_keys["intensity_covariance"] = 1
        features_set_keys["permeability"] = 1
        features_set_keys["ndt_rho"] = 1
        features_set_keys["theta"] = 1

    if string_key.find("ftm") >= 0:
        features_set_keys["mean_count"] = 1
        features_set_keys["occupancy_prob"] = 1
        features_set_keys["intensity_mean"] = 1
        features_set_keys["intensity_covariance"] = 1
        features_set_keys["permeability"] = 1
        features_set_keys["secondary_sample_count"] = 1
        features_set_keys["secondary_sample_range_mean"] = 1
        features_set_keys["secondary_sample_range_std_dev"] = 1
        features_set_keys["red"] = 1
        features_set_keys["green"] = 1
        features_set_keys["blue"] = 1
        features_set_keys["ndt_rho"] = 1
        features_set_keys["theta"] = 1
        features_set_keys["ev_lin"] = 1
        features_set_keys["ev_plan"] = 1
        features_set_keys["ev_sph"] = 1

    if string_key.find("dist_base") >= 0:
        features_set_keys["mean_count"] = 1
        features_set_keys["occupancy_log_probability"] = 1
        features_set_keys["intensity_mean"] = 1
        features_set_keys["intensity_covariance"] = 1
        features_set_keys["hit_count"] = 1
        features_set_keys["miss_count"] = 1
        features_set_keys["secondary_sample_count"] = (1,)
         
    
    if string_key.find("ohm") >= 0:
        features_set_keys["mean_count"] = 1
        features_set_keys["occupancy_log_probability"] = 1
        features_set_keys["intensity_mean"] = 1
        features_set_keys["intensity_covariance"] = 1
        # features_set_keys["traversal"] = 1
        features_set_keys["hit_count"] = 1
        features_set_keys["miss_count"] = 1
        # features_set_keys["permeability"] = 1
        features_set_keys["covariance_xx_sqrt"] = 1
        features_set_keys["covariance_xy_sqrt"] = 1
        features_set_keys["covariance_xz_sqrt"] = 1
        features_set_keys["covariance_yy_sqrt"] = 1
        features_set_keys["covariance_yz_sqrt"] = 1
        features_set_keys["covariance_zz_sqrt"] = 1

    if string_key.find("rhm") >= 0:
        # Note: rhm the features covariance_xz_sqrt, and covarianc_yy_sqrt are SWAPED!
        features_set_keys["mean_count"] = 1
        features_set_keys["occupancy_log_probability"] = 1
        features_set_keys["intensity_mean"] = 1
        features_set_keys["intensity_covariance"] = 1
        # features_set_keys["traversal"] = 1
        features_set_keys["hit_count"] = 1
        features_set_keys["miss_count"] = 1
        # features_set_keys["permeability"] = 1
        features_set_keys["covariance_xx_sqrt"] = 1
        features_set_keys["covariance_xy_sqrt"] = 1
        features_set_keys["covariance_yy_sqrt"] = 1
        features_set_keys["covariance_xz_sqrt"] = 1
        features_set_keys["covariance_yz_sqrt"] = 1
        features_set_keys["covariance_zz_sqrt"] = 1

    if len(features_set_keys) < 1:
        msg = f"Feature set key [ {string_key} ] not known. Check using the correct key"
        raise ValueError(msg)

    return [key for key in features_set_keys.keys()]


def parse_skip_connection_from_key(model_skip_connection_key):
    if model_skip_connection_key == "s0":
        return [0, 0, 0, 0, 0, 0]
    elif model_skip_connection_key == "s1":
        return [1, 0, 0, 0, 0, 0]
    elif model_skip_connection_key == "s2":
        return [1, 1, 0, 0, 0, 0]
    elif model_skip_connection_key == "s3":
        return [1, 1, 1, 0, 0, 0]
    elif model_skip_connection_key == "s4":
        return [1, 1, 1, 1, 0, 0]
    elif model_skip_connection_key == "s5":
        return [1, 1, 1, 1, 1, 1]

    raise ValueError(f"No valid skip_key found: {model_skip_connection_key}")


def select_foresttrav_model(params):
    """ """
    model_name = params.model_name
    nfeatures = params.nfeatures

    if model_name == "UNet6LMCD":
        model_skip = params.model_skip_connection[:5]
        model_stride = params.model_stride[:6]
        return UnetModels.UNet6LMCD(
            in_nchannel=nfeatures,
            out_nchannel=2,
            D=3,
            encoder_nchannels=params.model_nfeature_enc_ch_out,
            skip_conection=model_skip,
            stride=model_stride,
        )
    elif model_name == "UNet5LMCD":
        model_skip = params.model_skip_connection[:4]
        model_stride = params.model_stride[:5]
        return UnetModels.UNet5LMCD(
            nfeatures=nfeatures,
            out_nchannel=2,
            D=3,
            encoder_nchannels=params.model_nfeature_enc_ch_out,
            skip_conection=model_skip,
            stride=model_stride,
        )
    elif model_name == "UNet4LMCD":
        model_skip = params.model_skip_connection[:3]
        model_stride = params.model_stride[:4]
        return UnetModels.UNet4LMCD(
            nfeatures=nfeatures,
            out_nchannel=2,
            D=3,
            encoder_nchannels=params.model_nfeature_enc_ch_out,
            skip_conection=model_skip,
            stride=model_stride,
        )
    elif model_name == "UNet3LMCD":
        model_skip = params.model_skip_connection[:2]
        model_stride = params.model_stride[:3]
        return UnetModels.UNet3LMCD(
            nfeatures=nfeatures,
            out_nchannel=2,
            D=3,
            encoder_nchannels=params.model_nfeature_enc_ch_out,
            skip_conection=model_skip,
            stride=model_stride,
        )
    else:
        msg = f"Could not find {model_name}"
        raise ValueError(msg)
