import sys, os

sys.path.append(os.path.abspath("/nve_ml_ws/src/nve_eval"))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

from ohm_scan_map_comparision import compare_scan_map
from nve_eval_core.DataSetAugmenter import DataSetAugmenter, BAD_FEATURES

LATS_SCENE_DATA_SET = [
    ("/data/processed/2021_12_14_00_14_53Z/ohm_scans_org/1639441260.486418_pose_cloud.csv","/data/processed/feature_sets/2023_02_14_00_30_lfe_hl/2021_12_14_00_14_53Z"),
    ("/data/processed/2021_12_14_00_02_12Z/ohm_scans/1639440420.739426_pose_cloud.csv","/data/processed/feature_sets/2023_02_14_00_30_lfe_hl/2021_12_14_00_02_12Z"),
    ("/data/processed/2021_12_13_23_51_33Z/ohm_scans/1639439705.981243_pose_cloud.csv","/data/processed/feature_sets/2023_02_14_00_30_lfe_hl/2021_12_13_23_51_33Z"),
    ("/data/processed/2021_12_13_23_42_55Z/ohm_scans/1639439190.554594_pose_cloud.csv","/data/processed/feature_sets/2023_02_14_00_30_lfe_hl/2021_12_13_23_42_55Z"),
    ("/data/processed/2021_11_18_00_23_24Z/ohm_scans/1637195214.784616_pose_cloud.csv","/data/processed/feature_sets/2023_02_14_00_30_lfe_hl/2021_11_18_00_23_24Z"),
    ("/data/processed/2021_10_28_04_13_12Z/ohm_scans/1635394721.578308_pose_cloud.csv","/data/processed/feature_sets/2023_02_14_00_30_lfe_hl/2021_10_28_04_13_12Z"),
    ("/data/processed/2021_10_28_04_03_11Z/ohm_scans/1635394332.280371_pose_cloud.csv","/data/processed/feature_sets/2023_02_14_00_30_lfe_hl/2021_10_28_04_03_11Z"),
]
OUT_DIR_MAPS = "/data/processed/scene_feature_sets/lfe_hl"
OUT_DIR_SCENE = "/data/processed/scene_feature_sets/scenes"

# The experimental dir will be
# scan_feature_maps
# |_ feature_set_<style> {hl, hlcf, lfe}
#   |_<data_set_timestamp>
#       |_semantic_cloud_class_1.csv
#       |_semantic_cloud_class_2.csv
# |_ scenes
def scene_to_map(file_source, file_target):

    # Load the data
    df_source = pd.read_csv(file_source)
    
    # Load gt data, shuffle and reset index
    df_target = pd.DataFrame()
    df_0  = pd.read_csv(Path(file_target) / "semantic_cloud_class_1.csv")
    df_1  = pd.read_csv(Path(file_target) / "semantic_cloud_class_2.csv")
    df_target = pd.concat([df_0, df_1])
    df_target = df_target.sample(frac=1).reset_index(drop=True)
    
    # Associate scan with GT map
    coord_key = ["x", "y", "z"]
    id_pairs = compare_scan_map(
        df_source[coord_key].values,
        df_target[coord_key].values,
        voxel_sixe=0.1,
        local_map_bounds=[-5.0, -5.0, -2.0, 5.0, 5.0, 2.0],
    )

    # Getlabels and label_prob
    df_source["label"] = -1
    df_source["label_prob"] = -1.0
    id_pairs = np.array(id_pairs)

    label_arr = df_source["label"]
    label_arr[id_pairs[:, 0]] = df_target["label"].values[id_pairs[:, 1]]
    df_source["label"] = label_arr

    label_prob_arr = df_source["label_prob"]
    label_prob_arr[id_pairs[:, 0]] = df_target["label_prob"].values[id_pairs[:, 1]]
    df_source["label_prob"] = label_prob_arr

    df_source_filtered = df_source[df_source["label_prob"] >= 0.0]

    # Augmentation feature set with perm + adjacency
    data_augmenter = DataSetAugmenter(voxel_size = 0.1)
    
    # Add permeability
        # AUgment the data
    df = data_augmenter.add_occ_prob(df_source_filtered)
    df = data_augmenter.add_occ(df)
    df = data_augmenter.add_permeability(df)
    df = data_augmenter.add_eigenvalue_features(df)
    df, wanted_features = data_augmenter.add_adjacnecy_header(df, BAD_FEATURES)
    df = data_augmenter.add_adjacency_feature( df, wanted_features)
    
    # Save the data frame
    ds_timestamp = Path(file_source).parent.parent.name
    scene_file_name = Path(OUT_DIR_SCENE) / f"{ds_timestamp}_full_map.csv"
    scene_file_name.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(scene_file_name, index=False)
    
    out_dir_smeantic_maps = Path(OUT_DIR_MAPS) / ds_timestamp
    out_dir_smeantic_maps.mkdir(parents=True, exist_ok=True)
    data_augmenter.split_and_save_data_set(df,out_dir_smeantic_maps )


if __name__ == "__main__":
    for file_pair in LATS_SCENE_DATA_SET:
        scene_to_map(file_pair[0], file_pair[1])
