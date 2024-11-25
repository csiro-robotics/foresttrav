# Copyright (c) 2024
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

from pathlib import Path

import numpy as np
import pandas as pd


def load_semantic_csv_files(file_path: Path):
    df_class1 = pd.read_csv(Path(file_path) / "semantic_cloud_class_1.csv")
    df_class1["label"] = 0
    df_class2 = pd.read_csv(Path(file_path) / "semantic_cloud_class_2.csv")
    df_class2["label"] = 1

    return pd.concat([df_class1, df_class2])



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


def load_data_set(file_name: Path) -> pd.DataFrame:
    """Parse to load the appropriate data set for ForestTrav data set

    Args:
        file_name (_type_):

    Raises:
        FileNotFoundError: _description_
        NotImplemented: _description_
        FileNotFoundError: _description_
        FileNotFoundError: _description_

    Returns:
        pd.DataFrame: Combined data set containing w
    """
    data_set = None
    file_path = Path(file_name)
    if file_path.is_dir():
        if (
            Path(file_path / "semantic_cloud_class_1.csv").exists()
            and Path(file_path / "semantic_cloud_class_2.csv").exists()
        ):
            data_set = load_semantic_csv_files(file_path)
        else:
            msg = f"No file found in the directory: {str(file_path)}"
            raise FileNotFoundError(msg)

    elif file_path.is_file():
        # Parse the files. Assume it is a whole data set and not a single instance
        if "csv" in file_path.suffix:
            data_set = pd.read_csv(file_path)
        elif "ply" in file_path.suffix:
            raise NotImplemented("PLY reader not implemented")
        else:
            raise FileNotFoundError()

    else:
        msg = f"File/dir path is invalid: {str(file_path)}"
        raise FileNotFoundError(msg)

    return data_set
