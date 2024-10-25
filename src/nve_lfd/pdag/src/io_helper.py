from pathlib import Path

from plyfile import PlyData, PlyElement

import numpy as np
import pandas as pd


def read_ply_file(file_name: str) -> np.ndarray:
    """Reads a ply file and returns the data as a numpy array"""
    plydata = PlyData.read(file_name)

    return np.array(plydata["vertex"].data[["x", "y", "z"]].tolist())


def read_all_ply_files_as_df(data_dir: Path) -> pd.DataFrame:
    """Reads all ply files in a directory and returns the data as DataFrame
    Args:
      data_dir: Path to the directory
    Returns:
      DataFrame: DataFramewith all the data
    """
    data = pd.DataFrame()

    if type(data_dir) != Path:
        data_dir = Path(data_dir)

    for file in data_dir.iterdir():
        if not file.is_file():
            continue

        data = pd.concat([data, read_ply_as_df(file)])

    return data


def read_ply_as_df(file) -> pd.DataFrame:
    """Reads all ply files in a directory and returns the data as DataFrame
    Args:
      file: Absolute path to the file
    Returns:
      DataFrame: DataFrame with all the data. If not file was found will return an empty  dataframe
    """
    if type(file) != Path:
        file = Path(file)

    if not file.is_file() or ".ply" not in str(file):
        print(f"File cannot be loaded: {str(file)}")
        return pd.DataFrame()

    plydata = PlyData.read(file)

    return pd.DataFrame(plydata["vertex"].data)


def load_all_csv(data_dir: Path) -> pd.DataFrame:
    """Loads all files in a directory and concatenates them into a single dataframe
    Args:
      data_dir: Path to the directory
    Returns:
      pd.DataFrame: Dataframe with all the data
    """
    data = pd.DataFrame()

    if type(data_dir) != Path:
        data_dir = Path(data_dir)

    for file in data_dir.iterdir():
        if not file.is_file():
            continue

        if ".csv" not in str(file):
            continue

        df = pd.read_csv(file, sep=",")
        data = pd.concat([data, df])

    return data
