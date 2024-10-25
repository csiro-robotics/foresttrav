# System
# import os
# import glob
import subprocess
from pathlib import Path

from dataclasses import dataclass

# Third party
# import pandas as pd
# import numpy as np
# from scipy.sparse import data

# from nve_eval_core.nve_io import create_out_dir

# All the root data directories of the files we want to process
DATA_DIRS = [
    # "/data/processed/2021_10_28_04_03_11Z",
    # "/data/processed/2021_10_28_04_13_12Z",
    # "/data/processed/2021_11_18_00_23_24Z",
    # "/data/processed/2021_12_13_23_42_55Z",
    # "/data/processed/2021_12_13_23_51_33Z",
    # "/data/processed/2021_12_14_00_02_12Z",
    # "/data/processed/2021_12_14_00_14_53Z",
    # "/data/processed/2022_02_14_23_52_38Z",
    # "/data/processed/2022_02_15_00_00_35Z",
    # "/data/processed/2021_10_28_03_54_49Z",
    # "/data/processed/2022_02_14_23_47_33Z",
    "/data/processed/2023_04_06_04_15_05Z",
]
CONFIG_FILE = Path("/data/processed/config/config_pipline.yaml")


def main():

    data_dirs = DATA_DIRS
    config_file = CONFIG_FILE

    # Parameters we can change
    traj_type = "global"  # Can be either odom or global
    map_mode = "tm"
    resolution = [0.1]
    colour_modes = [0]
    colour_param = 1.0
    colour_mode = "ep"

    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.is_dir():
            continue
        for res in resolution:
            for colour_mode in colour_modes:

                # Check if we have done it already
                try:
                    ohm_colirization_pipeline(
                        data_path=data_path,
                        traj_type=traj_type, 
                        map_mode=map_mode,
                        res=res,
                        colour_mode=colour_mode,
                        colour_param=colour_param,
                        config_file=config_file,
                    )
                except BaseException as e:
                    print(e)
                    pass


def check_files(path_list):
    is_valid = True
    for path_i in path_list:
        is_valid = is_valid and Path(path_i).is_file()

    return is_valid


def find_first_file(data_path:Path, pattern)->str:
    files = []
    first_file = ""
    
    files = [str(file_path) for file_path in data_path.glob(pattern)]

    if not files:
        print(" NO files found for ", data_path, " \n with pattern ", pattern)
    elif len(files) > 1:
        print("Found more than one file, check for conflicts \n", files)
        first_file = files[0]
    else:
        first_file = files[0]

    return first_file


def run_ohm_colourization(
    map_file: Path,
    traj_file: Path,
    pcl_file: Path,
    video_file: Path,
    video_ts: Path,
    path_out_file: Path,
    colour_mode: int,
    colour_param: float,
    config_file: Path,
):
    """ """
    arg_load_rosparam = "rosparam load {} && ".format(config_file)

    args_pipeline = 'rosrun ohm_offline_colour_fuser ohm_offline_colour_fuser_node __name:=ohm_offline_colour_fuser_node'
    args_pipeline += ' "_ohm_file:={}"'.format(map_file)
    args_pipeline += ' "_traj_file:={}"'.format(traj_file)
    args_pipeline += ' "_pcl_file:={}"'.format(pcl_file)
    args_pipeline += ' "_video_file:={}"'.format(video_file)
    args_pipeline += ' "_video_timestamp_file:={}"'.format(video_ts)
    args_pipeline += ' "_colour_mode:={}"'.format(colour_mode)
    args_pipeline += ' "_colour_param:={}"'.format(colour_param)
    args_pipeline += ' "_path_out_file:={}"'.format(path_out_file)

    print(args_pipeline)
    p = subprocess.run([arg_load_rosparam + args_pipeline], shell=True, check=True, stdout=subprocess.PIPE)
    print(p)


def ohm_colirization_pipeline(
    data_path: Path,
    traj_type: str,
    map_mode: str,
    res: float,
    colour_mode: str,
    colour_param: float,
    config_file: Path,
):
    """
    This function runs the colourization pipeline for a given dataset

    Parameters
    ----------
    data_path : Path
        Path to the dataset
    traj_type : str
        Type of trajectory to use, can be either odom or global
    map_mode : str
        Type of map to use, can be either occ, ndt, tm, tmt
    res : float
        Resolution of the map
    colour_mode : str
        Type of colourization to use, can be either ep, es, ndt
    colour_param : float
        Parameter for the colourization
    config_file : Path
        Path to the config file

    """
    raw_data_dir = data_path / "raw_data"
    ohm_maps_dir = data_path / "ohm_maps"

    def_map_pattern = f"*{traj_type}*{map_mode}*map_v{res}.ohm"
    map_file = find_first_file(ohm_maps_dir, def_map_pattern)

    # Get full trajectory with no semantic label
    def_traj_pattern = f"*{traj_type}*wildcat_traj.txt"
    tra_file = find_first_file(raw_data_dir, def_traj_pattern)

    def_ply_pattern = f"*{traj_type}*wildcat_velodyne.ply"
    ply_file = find_first_file(raw_data_dir, def_ply_pattern)

    video_ts_pattern = "*camera0.timestamps"
    video_ts_file = find_first_file(raw_data_dir, video_ts_pattern)

    video_stream_pattern = "*camera0-*.ts"
    video_stream_file = find_first_file(raw_data_dir, video_stream_pattern)

    out_file = colour_mode_name(
        map_file=map_file, colour_mode=colour_mode, param=colour_param
    )
    # print(out_file)

    if not check_files(
        [map_file, tra_file, ply_file, video_stream_file, video_ts_file, config_file]
    ):
        print("ERROR: Could not find all files")
        return 0

    # if Path(out_file).is_file():
    #     print("Found rbg file with same name, skipping processing")
    #     return 0

    print(f"Start processing {data_path.stem}")
    run_ohm_colourization(
        map_file=map_file,
        traj_file=tra_file,
        pcl_file=ply_file,
        video_file=video_stream_file,
        video_ts=video_ts_file,
        path_out_file=out_file,
        config_file=config_file,
        colour_mode=colour_mode,
        colour_param=colour_param,
    )

def colour_mode_name(map_file:Path , colour_mode:str , param:float):
    """ Generate the colour mode name for the output file.
    Pattern: _{colou_mode}_rgb_p-{param}.ohm

    Parameters:
    ----------
    map_file : Path
        Path to the map file
    colour_mode : str
        Type of colourization to use, can be either ep, es, ndt
    param : float
        Parameter for the colourization

    Returns:
    ----------
    out_file_path : Path
        Path to the output file with colour mode name
    """

    colour_string = ""
    if colour_mode == 0:
        colour_string = "_ep_rgb_w" + str(param)
    elif colour_mode == 1:
        colour_string = "_ndt_rgb_b-" + str(param)

    return Path(str(map_file).replace(".ohm", colour_string + ".ohm"))





# Save model as
if __name__ == "__main__":
    main()
