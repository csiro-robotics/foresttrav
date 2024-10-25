from pathlib import Path
import torch

try:
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install pytorch_lightning`."
    )
from joblib import dump


import scnn_tm.utils as utils
from scnn_tm import models
from scnn_tm.models.OnlineFotrestTravDataLoader import OnlineDataSetLoader
import yaml


def setup_models(params):
    """Helper function to setup the model based on
    - feature_set_key: Keys words which identify the feature set used
    - Loads pervious model and scaler if necessary

    """
    model = None
    scaler = None
    feature_set = None
    feature_set_from_key = utils.generate_feature_set_from_key(params.feature_set_key)
    params.model_skip_connection = utils.parse_skip_connection_from_key(
        params.model_skip_connection_key
    )

    if hasattr(params, "pre_trained_model_file"):
        print(f"Using model to finetune")
        model, scaler, feature_set = utils.load_unet_model_from_yaml(
            params.pre_trained_model_file
        )

        assert feature_set_from_key == feature_set
    else:
        print("Training a new model")
        model = utils.select_foresttrav_model(params)
        feature_set = feature_set_from_key

    params.set_new_attribute("feature_set", feature_set)

    # Setup Data Loader
    return model, scaler


def setup_data_loaders(params, scaler):
    """
    Note: The data loader will initialize the default scaler if requried
    """
    data_augmenter = None
    if params.use_data_augmentation:
        print("Using data augmentation")
        data_augmenter = models.DataAugmenter.DataPatchAugmenter(params=params)

    data_set_factory = None
    if params.training_strategy_key == "debug":
        raise Exception("No strategy define")

    elif params.training_strategy_key == "simple_online_training":
        data_set_factory = OnlineDataSetLoader(
            target_feature_set=params.feature_set,
            min_pose_dist=1.6,
            voxel_size=voxel_size,
        )

        new_data_batch = data_set_factory.load_online_data_set(
            params.data_set_file, 0.0
        )
        updated_node_keys = data_set_factory.add_new_data(new_data_batch=new_data_batch)
        data_set_factory.use_all_data()

    else:
        raise ValueError(f"Cloud not find strateg [{params.startegy}]")

    return data_set_factory


def train_model(params):
    #
    num_devices = min(params.ngpus, torch.cuda.device_count())
    print(f"Testing {num_devices} GPUs.")

    # Setup model
    model, scaler = setup_models(params)
    data_set = setup_data_loaders(params, scaler)
    model.train()
    # Combine into PlModule
    pl_module = models.OnlineForestTrav.ForestTravePLOnline(
        model,
        data_loader=data_set,
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
        optimizer_name=params.optimizer_name,
        voxel_size=params.voxel_size,
        batch_size=params.batch_size,
        val_batch_size=params.batch_size,
        params=params,
    )

    trainer = Trainer(
        max_epochs=params.max_epochs,
        accelerator=params.accelerator,
        devices=1,
        check_val_every_n_epoch=params.check_val_every_n_epoch,
        callbacks=[
            EarlyStopping(monitor="train_loss", patience=params.es_patience, mode="min")
        ],
    )
    trainer.fit(pl_module)

    if hasattr(params, "model_out_dir"):
        print(f"Saving model to {params.model_out_dir}")
        simple_model_save(
            model_out_dir=params.model_out_dir,
            model=model,
            scaler=pl_module.get_scaler(),
            feature_set_key=params.feature_set_key,
            cv_nbr=params.cv_nbr,
        )

    if params.evaluate_on_test:
        a = 1

    return model, pl_module.get_scaler()


def simple_model_save(model_out_dir, model, scaler, feature_set_key=None, cv_nbr=0):
    """Simple function to write out a model compatible with ME learning"""

    model_out_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_pl_file = model_out_dir / f"{model.__class__.__name__}_cv_{cv_nbr}.pl"
    torch.save(model.state_dict(), model_pl_file)

    # Save the data scaler
    scaler_file = model_out_dir / f"scaler_cv_{cv_nbr}.joblib"
    dump(scaler, scaler_file)

    # Save the config file
    config_file = model_out_dir / "model_config.yaml"
    # if config_file.exists():
    #     return

    config_file.touch()
    with open(config_file, "w") as file_descriptor:
        yaml_params = {}
        yaml_params["model_name"] = model.__class__.__name__
        yaml_params["model_nfeature_enc_ch_out"] = model.ENCODER_CH_OUT[0]
        yaml_params["model_skip_connection_key"] = f"s{model.SKIP_CONNECTION.count(1)}"
        yaml_params["model_dropout"] = 0.0
        yaml_params["feature_set_key"] = feature_set_key

        # We make the assumption that the model is in the same dir
        yaml_params["torch_model_path"] = str(model_pl_file)
        yaml_params["scaler_path"] = str(scaler_file)
        yaml.safe_dump(yaml_params, file_descriptor)


########################## DEBUGUNG STUFF ##########################
HDF5_DEBUG_FILE = Path(
    "/data/debug/test_online_trainer/auto_train_data/QCAT_FOREST_2.hdf5"
    # "/data/debug/test_online_trainer/auto_train_data/heritage_forest_no_ohm_clear.hdf5"
    # "/data/debug/test_data_fuser/heritage_forest_large_footprint.hdf5"
)
DEBUG_PARAMS_PATH = Path(
    "/nve_ml_ws/src/scnn_tm/config/online_learning/23_06_07_ts_online_learning.yaml"
)
USE_PRE_TRAIN = False
PRE_TRAINED_MODEL_YAML = Path(
    "/data/debug/test_online_trainer/models_ts/qf1_23_07_03_Unet5MCD_s3_nf8_ohm_rgb_sr_cv_patch_ADAM/model_config.yaml"
)

OUT_DIR = Path("/data/debug/test_online_trainer/models_ts/hf_base_ohm_rgb_sr_01")


def test_training():
    params = utils.load_yamls_as_struct(DEBUG_PARAMS_PATH)
    # This is the only weird one...
    feature_set = utils.generate_feature_set_from_key(params.feature_set_key)
    params.set_new_attribute("feature_set", feature_set)
    params.set_new_attribute("nfeatures", len(feature_set))
    params.set_new_attribute("data_set_file", HDF5_DEBUG_FILE)
    if USE_PRE_TRAIN:
        params.set_new_attribute("pre_trained_model_file", PRE_TRAINED_MODEL_YAML)
        assert Path(PRE_TRAINED_MODEL_YAML).exists()

    params.set_new_attribute("model_out_dir", OUT_DIR)
    params.set_new_attribute("max_cv", 10)

    # This is the
    for cv_nbr in range(params.max_cv):
        params.set_new_attribute("cv_nbr", cv_nbr)
        model, scaler = train_model(params)


if __name__ == "__main__":
    test_training()
