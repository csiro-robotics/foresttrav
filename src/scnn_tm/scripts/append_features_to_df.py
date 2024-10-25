import warnings
import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data_sets", default=[], type=list)
parser.add_argument("--out_dir", default=[], type=str)


def add_permeability_to_dataframe(df: pd.DataFrame):
    """Computes and adds the "permeability" feature to the ndt representation if possible"""

    if not "hit" or not "miss" in df.columns:
        return df

    df["permeability"] = []

    for index, row in df.iterrows():
        row["permeability"] = float(row["miss"]) / (float(row["hit"] + row["miss"]))

    return df


def add_eigenvalue_features(df: pd.DataFrame) -> pd.DataFrame:
    "Adds the eigenvalue ndt features (linearity, planarity and spheriocity) as well as the surface normal and and angle to gravity"
    
    


if __name__ == "__main__":
    args = parser.parse_args()

    for data_file in args.data_sets:
        data_file_path = Path(data_file)

        if (not data_file_path.exists()) or (data_file_path.isFile()):
            warnings.warn("Passed item is not a file", FileExistsWarning)
            continue

        # Load the data file

        # Append all the features

        df = add_permeability_to_dataframe(df)

        df = add_eigenvalue_features(df)

        # Save the data files
