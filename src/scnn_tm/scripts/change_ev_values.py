import warnings
import argparse
import pandas as pd
import numpy as np
from pathlib import Path




DATA_SET_FILES = [
"/data/processed/feature_sets/lfe_hl_v0.1/2022_02_14_23_47_33Z/semantic_cloud_class_1.csv",
"/data/processed/feature_sets/lfe_hl_v0.1/2022_02_14_23_47_33Z/semantic_cloud_class_2.csv",
]    

# ROOT_FILE_DIR = Path("/data/processed/ohm_scans/2022_02_14_23_47_33Z/ohm_scans_v01")
# DATA_SET_FILES = [file for file in ROOT_FILE_DIR.iterdir()]


FEATURES_TO_RENAME = {
    "cov_xx": "covariance_xx_sqrt", 
    "cov_xy": "covariance_xy_sqrt", 
    "cov_xz": "covariance_xz_sqrt", 
    "cov_yy": "covariance_yy_sqrt", 
    "cov_yz": "covariance_yz_sqrt", 
    "cov_zz": "covariance_zz_sqrt",
}

parser = argparse.ArgumentParser()
parser.add_argument("--data_sets", default= DATA_SET_FILES, type=list)
parser.add_argument("--out_dir", default=[], type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    for data_file in args.data_sets:
        data_file_path = Path(data_file)

        if (not data_file_path.exists()) or (not data_file_path.is_file()):
            warnings.warn("Passed item is not a file")
            continue

        # Load the data file
        df = pd.read_csv(data_file_path)
        
        # Rename the wrong covariance featues
        df = df.rename(columns={"covariance_xz_sqrt": "temp_cov"})
        df = df.rename(columns={"covariance_yy_sqrt": "covariance_xz_sqrt"})
        df = df.rename(columns={"temp_cov": "covariance_yy_sqrt"})
        # Add the occupancy log probability        
                
        
        df[wanted_features].to_csv(path_or_buf = data_file_path, index=False)
