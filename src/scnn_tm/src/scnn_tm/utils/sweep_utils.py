# Utils for sweeps uswing wand. Mostly parsing keys and other elements
from pathlib import Path
import pandas as pd

# Defaults to load data sets for training, test and debug
# TODO: Make this a more general framework and not relying and folder organizations?

ROOT_DATA_DIR = Path("/data/forest_trav")

TRAIN_DATA_SETS_NAME = [
    '2021_10_28_03_54_49Z',
    '2021_10_28_04_03_11Z',
    '2021_10_28_04_13_12Z',
    '2021_11_18_00_23_24Z',
    '2021_12_13_23_42_55Z',
    '2021_12_13_23_51_33Z',
    '2021_12_14_00_02_12Z',
    '2021_12_14_00_14_53Z',]

TEST_DATA_SETS_NAME= [ "2022_02_14_23_47_33Z"]

DEBUG_DATA_SET_NAME = ['2021_12_14_00_02_12Z', '2021_12_14_00_14_53Z',]

def get_data_set_files(data_set_key:str, voxel_size: float, valid_data_set_names:list):
    """ data_set_key: lfe, lfe_cl, lfe_hl, scenes
        voxel_size: float   
    """
    data_set_dir = ROOT_DATA_DIR / (data_set_key+f"_v{voxel_size}")
    
    data_sets =  [ file for file in data_set_dir.iterdir() if  (file.stem[2::] in valid_data_set_names ) ]
    
    if not data_sets:
        message  = f"No valid data set key found: {data_set_key}"
        raise ValueError(message)
    
    return data_sets


# Specific instances we use throughout the codebase
def train_data_set_by_key(data_set_key:str, voxel_size: float):
    return  get_data_set_files(data_set_key, voxel_size, TRAIN_DATA_SETS_NAME )

def test_data_set_by_key(data_set_key:str, voxel_size: float):
    return get_data_set_files(data_set_key, voxel_size, TEST_DATA_SETS_NAME )

def debug_data_set_key():
    return get_data_set_files("lfe_cl", 0.1, TRAIN_DATA_SETS_NAME )

def scenes_by_key(voxel_size:float):
    """ Loads all the scene files of a certain resolution
    
    Returns a list of scene files
    
    """
    data_set_dir = ROOT_DATA_DIR / ( f"scene_v{voxel_size}")
    scene_files =  [ file for file in data_set_dir.iterdir() if file.is_file()  ]
    
    if not scene_files:
        message  = f"No valid scene files found in : {str(ROOT_DATA_DIR)}"
        raise ValueError(message)
    
    return get_data_set_files("scenes", 0.1, TRAIN_DATA_SETS_NAME )

def data_set_name_without_nte():
    " The data set names that contain no non-traversable labells"
    return ["2021_10_28_03_54_49Z", "2021_12_13_23_42_55Z"]
