from pathlib import Path
from yaml import safe_load
import shutil

# This script parses models and sorts them into a model base dir based on
# data_set key training_strategy, model_name, and voxel_size
# i.e cv_scene_lfe_cl_01_UNet5LMCD


SOURCE_PL_DIR = "/data/forest_trav_paper/raw_models"
# TARGET_PL_WAND_DIR = "/data/forest_trav_paper/ts_models"


PRAMS = {
    "out_dir": "/data/forest_trav_paper/ts_models",
    "process_all": False,
    "data_set_keys": ["lfe_hl"],
    "training_strategy_keys": [ "cv_test_train"],
    "voxel_sizes": [0.1, 0.2],
    "model_names": ["UNet5LMCD"],
    "model_nfeature_enc_ch_out": [16],
}


def parse_data_dir(experiment_dir: Path, params):

    # Load the config file of the dir
    yaml_conig_file = experiment_dir / Path("files") / "config.yaml"

    if not yaml_conig_file.is_file():
        return

        # Open yaml file with all the configurations3
    with open(yaml_conig_file, "r") as f:
        config = safe_load(f)

    # Check if we meet the processing requiremnts if any
    if not isValidExperiment(config, params):
        return

    # Tell that we are doing something
    print(f"Processing {experiment_dir.name}")

    # Generate file dir sorted by
    training_key = config["param/training_strategy_key"]["value"]
    data_set_key = config["param/data_set_key"]["value"]
    model_name = config["param/model_name"]["value"]
    voxel_size = config["voxel_size"]["value"]
    feature_set_key = config["param/feature_set_key"]["value"]
    dst_dir = (
        Path(params["out_dir"])
        / f"{training_key}_{data_set_key}_{ int(voxel_size * 100.0)}_{model_name}_{feature_set_key}"
    )
    print(str(dst_dir))

    # Make the directory
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    src_dir = Path(experiment_dir) / "files"
    shutil.copytree(src=src_dir, dst=dst_dir, dirs_exist_ok=True)
    print(f"Finished {experiment_dir.name}")


def isValidExperiment(experiment_congig: dict, params: dict) -> bool:

    # All files are valid
    if params["process_all"]:
        return True

    # Check for the correct values
    if not (
        experiment_congig["param/model_name"]["value"] in params["model_names"]
    ) and (params["model_names"]):
        return False

    if not (
        experiment_congig["param/data_set_key"]["value"] in params["data_set_keys"]
    ) and (params["data_set_keys"]):
        return False

    if not (
        experiment_congig["param/training_strategy_key"]["value"]
        in params["training_strategy_keys"]
    ) and (params["training_strategy_keys"]):
        return False

    if not (
        experiment_congig["param/model_nfeature_enc_ch_out"]["value"]
        in params["model_nfeature_enc_ch_out"]
    ) and (params["model_nfeature_enc_ch_out"]):
        return False

    if not (experiment_congig["voxel_size"]["value"] in params["voxel_sizes"]) and (
        params["voxel_sizes"]
    ):
        return False

    return True


def main(args):

    for dir in Path(SOURCE_PL_DIR).iterdir():

        if not dir.is_dir():
            continue

        parse_data_dir(Path(dir), params=PRAMS)


if __name__ == "__main__":
    main([])
