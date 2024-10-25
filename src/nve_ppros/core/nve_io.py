# System
import os
import glob
from pathlib import Path


# Third party
import pandas as pd
import numpy as np

# Specific for our data types and dir
def load_and_append_dir(data_dir, pattern=""):
    """Loads and concatenats either
        - all data in a dir
        - All data with a fixed size pattern fix size patterns
    in: data_dir    Absolute path to the directory
    in: pattern     Pattern we aim to match, note it cannot contain a wildcard!

    out: data       Concatenated data pd.DataFrame
    """
    data = []

    files = get_files_in_dir(data_dir=data_dir, pattern=pattern)

    for file_name in files:

        # print(file_name)
        if os.path.isfile(file_name) and os.path.getsize(file_name) > 0:
            df = pd.read_csv(file_name, sep=",")

            if df.empty:  # skip if the frame is empty
                continue
        else:
            continue

        if 0 == len(data):
            data = df
        else:
            data = pd.concat([data, df])
    return data


def get_files_in_dir(data_dir, pattern=""):
    """Loads all"""
    files = []
    scenes = []
    if not bool(pattern):
        files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
    else:
        files = glob.glob(
            os.path.join(data_dir, "*" + pattern + "*")
        )  # Ugly padding but avoids mishaps
    return files


def find_map_and_traj(data_dir, mode, res):
    """ """
    map_find = os.path.join(data_dir, "*{mode}_v{res}*.ohm".format(mode=mode, res=res))
    map_path = glob.glob(map_find)
    traj_path = glob.glob(os.path.join(data_dir, "*_traj_labeled.txt"))

    return [map_path, traj_path]


def create_out_dir(out_base_dir):
    """ """
    os.mkdir(os.path.join(out_base_dir, "features_data"))
    os.mkdir(os.path.join(out_base_dir, "scenes"))
    os.mkdir(os.path.join(out_base_dir, "pred"))
    os.mkdir(os.path.join(out_base_dir, "models"))



def find_processed_ohm_file(data_dir:Path, traj_type: str, mode:str, res:float):
    """ Finds the processed ohm file in the data_dir. We expect to see the following pattertn
    
    Parameters
    ----------
    data_dir : Path

    traj_type : str
        Either odom or global
    mode : str  
        Either occ, ndt, tm, tmt
    res : float
        Resolution of the map
    
    """
    def_map_pattern = f"*{traj_type}*{mode}*map_v{res}.ohm"
    map_find =  list(data_dir.glob(def_map_pattern))

    if len(map_find) > 1:
        print(f"WARNING: Multiple maps found. Returning the first: {map_find[0]}")
        return map_find[0]

    return map_find[0]