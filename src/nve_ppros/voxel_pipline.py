# System
import os
import glob
import subprocess
from pathlib import Path
from shutil import copyfile

# Third party
import numpy as np
import pandas as pd


from scipy.sparse import data

from core.nve_io import create_out_dir

# TODO: Add a file with all the parameters to


def find_first_file(data_path: Path, pattern: str):
    files = []
    first_file = ""

    for file in data_path.glob(pattern):
        # print(file)
        files.append(file)

    if not files:
        print(" No files found for ", data_path, " \n with pattern ", pattern)
    elif len(files) > 1:
        print("Found more than one file, check for conflicts \n", files)
        first_file = files[0]
    else:
        first_file = files[0]

    return first_file


def get_bounds(traj_file, padding):

    bound_str = "["
    if traj_file:
        df = pd.read_csv(traj_file, skiprows=0, delimiter="\s+")

        bound_str += str(df["x"].min() + padding[0])
        bound_str += ", " + str(df["y"].min() + padding[1])
        bound_str += ", " + str(df["z"].min() + padding[2])
        bound_str += ", " + str(df["x"].max() + padding[3])
        bound_str += ", " + str(df["y"].max() + padding[4])
        bound_str += ", " + str(df["z"].max() + padding[5]) + "]"

    return bound_str


def get_gt_labels(data_dir: Path, pattern=""):
    """Get all gt labels from a dir in data_dir with name label_dir_name"""

    search_pattern = "*.ply"
    if pattern:
        search_pattern = f"*_{pattern}_" + search_pattern

    gt_file_names = []
    if data_dir.is_dir() and data_dir.exists():
        gt_file_names = [str(path_i) for path_i in data_dir.glob(search_pattern)]

    return gt_file_names


def run_self_labler(
    map_file: Path,
    traj_file: Path,
    out_dir: Path,
    hl_files: list,
    map_roi: list,
    config_file: Path,
):
    """Runs the robot-self labelling node for offline processing

    Args:
        map_file (Path):  Path to ohm map file
        traj_file (Path): Path to semantic trajectory file
        out_dir (Path):   Path to output directory
        hl_files (list):  List of hand labelled data files
        map_roi (list):   List of map bounds
        config_file (Path): Path to config file
    """
    arg_load_rosparam = "rosparam load {} && ".format(config_file)
    arg_load_rosparam += (
        'rosparam set /offline_te_labeler_node/map_roi "{}" && '.format(map_roi)
    )
    hl_files_str = ", ".join(hl_files)
    arg_load_rosparam += (
        'rosparam set /offline_te_labeler_node/hl_files "[{}]" && '.format(hl_files_str)
    )

    args_pipeline = "rosrun nve_robot_labelling offline_te_labeler_node __name:=offline_te_labeler_node"

    args_pipeline += " _map_file:={}".format(map_file)
    args_pipeline += " _semantic_traj_file:={}".format(traj_file)
    args_pipeline += " _out_dir:={}".format(out_dir)

    print(arg_load_rosparam + args_pipeline)

    p = subprocess.run(
        [arg_load_rosparam + args_pipeline],
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
    )
    print(p)


def run_scene_feature_extractor(
    map_file, bounds, out_dir, file_name, config_file, adj_mode
):
    """Scene feature extractor in a set of bounds
    in: map_file    ohm map file
    in: out_dir     output directory
    in: file_name   output file name
    in: config_file     Location of config file
    """

    # Load bounds

    args_pipeline = "ros2 run nve_ia voxel_map_extractor_node --ros-args "
    args_pipeline += ' -p "ohm_file:={}"'.format(map_file)
    args_pipeline += ' -p "out_dir:={}"'.format(out_dir)
    args_pipeline += ' -p "out_file:={}"'.format(file_name)
    args_pipeline += ' -p "map_bounds:={}"'.format(bounds)
    args_pipeline += ' -p "adj_mode:={}"'.format(adj_mode)
    args_pipeline += ' --params-file "{}"'.format(config_file)
    print(args_pipeline)
    p = subprocess.run([args_pipeline], shell=True, check=True, stdout=subprocess.PIPE)


def save_parameters_to_file(file_path, params):
    out_file_path = file_path.joinpath("voxel_labler_config.txt")

    with open(out_file_path, "w") as f:
        
        print(" Parameters used to extract voxel features \n ", file=f)
        for key, value in params.items():
            print(
                "Param: {key_str}  [{value_tr}]".format(key_str=key, value_tr=value),
                file=f,
            )


OHM_MAPS = [
    "/data/processed/2021_10_28_04_03_11Z/ohm_maps/global_2021_10_28_04_03_11Z_tm_map_v0.4_ep_rgb_w1.0.ohm",
    "/data/processed/2021_10_28_04_13_12Z/ohm_maps/global_2021_10_28_04_13_12Z_tm_map_v0.4_ep_rgb_w1.0.ohm",
    "/data/processed/2021_11_18_00_23_24Z/ohm_maps/global_2021_11_18_00_23_24Z_tm_map_v0.4_ep_rgb_w1.0.ohm",
    "/data/processed/2021_12_13_23_42_55Z/ohm_maps/global_2021_12_13_23_42_55Z_tm_map_v0.4_ep_rgb_w1.0.ohm",
    "/data/processed/2021_12_13_23_51_33Z/ohm_maps/global_2021_12_13_23_51_33Z_tm_map_v0.4_ep_rgb_w1.0.ohm",
    "/data/processed/2021_12_14_00_02_12Z/ohm_maps/global_2021_12_14_00_02_12Z_tm_map_v0.4_ep_rgb_w1.0.ohm",
    "/data/processed/2021_12_14_00_14_53Z/ohm_maps/global_2021_12_14_00_14_53Z_tm_map_v0.4_ep_rgb_w1.0.ohm",
    "/data/processed/2021_10_28_03_54_49Z/ohm_maps/global_2021_10_28_03_54_49Z_tm_map_v0.4_ep_rgb_w1.0.ohm",
    "/data/processed/2022_02_14_23_47_33Z/ohm_maps/global_2022_02_14_23_47_33Z_tm_map_v0.4_ep_rgb_w1.0.ohm",
    "/data/processed/2022_02_14_23_52_38Z/ohm_maps/global_2022_02_14_23_52_38Z_tm_map_v0.4_ep_rgb_w1.0.ohm",
    "/data/processed/2022_02_15_00_00_35Z/ohm_maps/global_2022_02_15_00_00_35Z_tm_map_v0.4_ep_rgb_w1.0.ohm",
]

CONFIG_FILE = "/data/processed/config/config_pipline.yaml"
ROOT_OUT_DIR = "/data/processed/feature_sets"

def setParams():
    """Sets the parameters for the pipeline"""
    # Parameters which define our methods
    traj_type = "global"
    map_mode = "tm"
    adj_mode = 0
    res = 0.4
    use_gt = True
    use_lfd = True
    # gt_pattern = "*"
    gt_pattern = "cl"
    # map_roi_padding = [-3.0, -3.0, -2.0, 3.0, 3.0, 2.0]
    map_roi_padding = [-5.0, -5.0, -2.0, 5.0, 5.0, 2.0]
    colour_mode = "*ep_rgb"  # ndt_rgb_b-1.05"

    params = {}
    params["traj_type"] = traj_type
    params["map_mode"] = map_mode
    params["resolution"] = res
    params["adj_mode"] = adj_mode
    params["use_gt"] = use_gt
    params["gt_pattern"] = gt_pattern
    params["use_lfd"] = use_lfd
    params["map_roi_padding"] = map_roi_padding
    params["colour_mode"] = colour_mode

    return params


def timestamp_today():
    return pd.to_datetime("today").strftime("%Y_%m_%d_%H_%M")


def generate_output_dir(root_output_dir: Path):

    if root_output_dir.exists():
        return
    root_output_dir.mkdir(parents=True)


def main():

    params = setParams()
    config_file = CONFIG_FILE
    # Generate the output dir with all variable string
    out_dir_name = timestamp_today() + "_ohm-res{}_adj_{}".format(
        str(params["resolution"]), str(params["adj_mode"])
    )

    if params["use_gt"]:
        out_dir_name = out_dir_name + "_{}".format(params["gt_pattern"])

    if params["colour_mode"]:
        out_dir_name = out_dir_name + "_{}".format(params["colour_mode"])

    # Labeling pipeline

    map_files = OHM_MAPS
    out_dir_base = ROOT_OUT_DIR / Path(timestamp_today())
    failed_data_sets = []
    for map_file in map_files:

        map_file = Path(map_file)

        # THIS IS BRITTLE and bad code!
        data_dir = map_file.parent.parent

        # Create the out directories for the feature extraction
        out_dir = out_dir_base / data_dir.name 
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        if not map_file.is_file():
            print("Could not find map file for:", map_file)
            continue

        # Find trajectory
        def_traj_pattern = "*{traj_type}*wildcat_traj_labeled.txt".format(
            traj_type=params["traj_type"]
        )
        traj_file = find_first_file(data_dir / "raw_data", def_traj_pattern)

        if not map_file or not traj_file:
            print("Could not find traj or map file for:", data_dir)
            continue

        # Get ground truth labels
        hl_files = []
        if params["use_gt"]:
            hl_files = get_gt_labels(
                data_dir / Path("raw_data") / Path("te_segmentation"),
                params["gt_pattern"],
            )

        # bounds
        map_roi = get_bounds(traj_file=traj_file, padding=params["map_roi_padding"])

        try:
            run_self_labler(
            map_file=map_file,
            traj_file=traj_file,
            out_dir=out_dir,
            config_file=config_file,
            hl_files=hl_files,
            map_roi=map_roi,
            )
        except:
            print(" Issue with the following data set: {}".format(data_dir.name))
            failed_data_sets.append(data_dir.name)
            continue
    
    # Print the failed data sets at the end so we have it in one place
    for failed_data_set in failed_data_sets:
        print("Failed data set: {}".format(failed_data_set))

# Save model as
if __name__ == "__main__":
    main()
