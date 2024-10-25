from pathlib import Path
import numpy as np
import pandas as pd

from ohm_scan_map_comparision import (
    load_data_set,
    compare_scan_map,
    associate_points_to_voxel_cloud,
)
from scnn_tm.utils import point_to_voxel_centre, overlay_two_clouds, point_in_voxel

# What could be the issue? 
# - Not aligned clouds?
# - Issue with the comparison? 

SCENE_PAIRS_DIR = [
    (
        "/data/processed/ohm_scans/2021_12_14_00_14_53Z/ohm_scans_v01",
        "/data/processed/feature_sets/lfe_hl_v0.1/2021_12_14_00_14_35Z",
    ),
    (
        "/data/processed/ohm_scans/2022_02_14_23_47_33Z/ohm_scans_v01",
        "/data/processed/feature_sets/lfe_hl_v0.1/2022_02_14_23_47_33Z",
    ),
    (
        "/data/temp/test_scene",
        "/data/processed/feature_sets/lfe_hl_v0.1/2022_02_14_23_47_33Z",
    ),
    (
        "/nve_ml_ws/src/scnn_tm/tests/config/scan_comparison_ohm_scan.csv",
        "/nve_ml_ws/src/scnn_tm/tests/config/scan_comparsion_gt_ohm_map.csv",
    ),
]


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--voxel_size", default=0.1, type=float)
parser.add_argument(
    "--local_map_bounds", default=[-7.0, -7.0, -5.0, 10.0, 10.0, 3.0], type=list
)
parser.add_argument("--data_dirs", default=SCENE_PAIRS_DIR, type=list)
parser.add_argument("--map_dir_index", default=2, type=int)


def test_main(args):
    data_set_pairs = args.data_dirs[args.map_dir_index]
    source_scans_df, target_map_df, _ = load_data_set(
        ohm_scans_dir=Path(data_set_pairs[0]),
        data_ground_truth_dir=Path(data_set_pairs[1]),
    )

    # Load the data set without the data set laoder
    # source_scans_df = [pd.read_csv(args.data_dirs[-1][0], sep=",")]
    # target_map_df = pd.read_csv(args.data_dirs[-1][1], sep=",")

    pos_keys = ["x", "y", "z"]
    X_coords_target = target_map_df[pos_keys].to_numpy()
    for i in range(len(source_scans_df)):
        X_coords_source = source_scans_df[i][pos_keys].to_numpy()

        # Compute the source labels from model
        id_pairs, X_coords_source = compare_scan_map(
            source_scan=X_coords_source,
            target_map=X_coords_target,
            voxel_size=args.voxel_size,
            local_map_bounds=args.local_map_bounds,
        )

        # id_pairs = associate_points_to_voxel_cloud(
        #     source_cloud=X_coords_source,
        #     target_cloud=X_coords_target,
        #     voxel_size=args.voxel_size,
        # )

        if not id_pairs:
            print(f"Skipped ohm_scan {i} since no pairs could be found")
            return

        id_pairs_arr = np.vstack(id_pairs)
        # Get the voxels that are missing

        # Evaluate how many of the scans we have associated
        print(
            f"Total number of map points: { X_coords_target.shape[0]} and scan points: {X_coords_source.shape[0]} , scan/map fraction { X_coords_source.shape[0]/ X_coords_target.shape[0]}"
        )
        print(
            "Fraction of scan id-paris found: ",
            len(id_pairs) / X_coords_source.shape[0],
        )
        print(
            "Fraction of total ud-pairs found: ",
            len(id_pairs) / X_coords_target.shape[0],
        )

        # Visualise the voxel centre of the clouds that match
        source_cloud_voxel_centre_cloud = np.array(
            [
                point_to_voxel_centre(X_coords_source[id_pair[0], 0:3], args.voxel_size)
                for id_pair in id_pairs
            ]
        )

        target_cloud_voxel_centre_cloud = np.array(
            [
                point_to_voxel_centre(X_coords_target[id_pair[1], 0:3], args.voxel_size)
                for id_pair in id_pairs
            ]
        )
        for point_source, point_target in zip(
            source_cloud_voxel_centre_cloud, target_cloud_voxel_centre_cloud
        ):
            assert point_in_voxel(point_source, point_target, args.voxel_size)

        overlay_two_clouds(
            source_cloud_voxel_centre_cloud, target_cloud_voxel_centre_cloud
        )

        # Show the points we did not find
        source_found_ids_list = {key: 0 for key in id_pairs_arr[:, 0]}
        failed_source_cloud_voxel_centre_cloud = np.array(
            [
                point_to_voxel_centre(X_coords_source[i, 0:3], args.voxel_size)
                for i in range(X_coords_source.shape[0])
                if i not in source_found_ids_list
            ]
        )

        failed_source_cloud_voxel_org_cloud = np.array(
            [
                X_coords_source[i, 0:3]
                for i in range(X_coords_source.shape[0])
                if i not in source_found_ids_list
            ]
        )

        target_found_ids_list = {key: 0 for key in id_pairs_arr[:, 1]}
        failed_target_cloud_voxel_centre_cloud = np.array(
            [
                point_to_voxel_centre(X_coords_target[i, 0:3], args.voxel_size)
                for i in range(X_coords_target.shape[0] - 1)
                if i not in target_found_ids_list
            ]
        )

        failed_target_cloud_voxel_org_cloud = np.array(
            [
                X_coords_target[i, 0:3]
                for i in range(X_coords_target.shape[0] - 1)
                if i not in target_found_ids_list
            ]
        )
        overlay_two_clouds(
            source_cloud_voxel_centre_cloud, failed_target_cloud_voxel_centre_cloud
        )

        id_failed_centre  =  associate_points_to_voxel_cloud(
            source_cloud=failed_source_cloud_voxel_centre_cloud,
            target_cloud=failed_target_cloud_voxel_centre_cloud,
            voxel_size=0.1,
        )
        assert not id_failed_centre
        
        
        id_failed_cloud  =  associate_points_to_voxel_cloud(
            source_cloud=failed_source_cloud_voxel_org_cloud,
            target_cloud=failed_target_cloud_voxel_org_cloud,
            voxel_size=0.1,
        )
        assert not id_failed_centre
        
        # Save the files
        pd.DataFrame(source_cloud_voxel_centre_cloud).to_csv(
            "/data/temp/found_source_cloud_test.csv", index=False
        )
        pd.DataFrame(target_cloud_voxel_centre_cloud).to_csv(
            "/data/temp/found_target_cloud_test.csv", index=False
        )
        pd.DataFrame(failed_source_cloud_voxel_centre_cloud).to_csv(
            "/data/temp/failed_source_cloud_test.csv", index=False
        )
        pd.DataFrame(failed_target_cloud_voxel_centre_cloud).to_csv(
            "/data/temp/failed_target_cloud_test.csv", index=False
        )
        pd.DataFrame(failed_source_cloud_voxel_org_cloud).to_csv(
            "/data/temp/failed_source_cloud_test_org.csv", index=False
        )
        pd.DataFrame(failed_target_cloud_voxel_org_cloud).to_csv(
            "/data/temp/failed_target_cloud_test_org.csv", index=False
        )


def load_data_set(ohm_scans_dir: Path, data_ground_truth_dir: Path):
    """Copy of the initial one"""
    files = [file for file in ohm_scans_dir.iterdir()]
    files = sorted(files)
    file_times = [float(file.stem[0:-6]) for file in files[-2:-1]]
    ohm_scans_dfs = [pd.read_csv(file, sep=",") for file in files[-2:-1]]

    # Load the refrence map
    ohm_map_df1 = pd.read_csv(data_ground_truth_dir / "semantic_cloud_class_1.csv")
    ohm_map_df1["label"] = 0
    ohm_map_df2 = pd.read_csv(data_ground_truth_dir / "semantic_cloud_class_2.csv")
    ohm_map_df2["label"] = 1
    target_map_df = pd.concat([ohm_map_df2, ohm_map_df1])

    return ohm_scans_dfs, target_map_df, file_times


if __name__ == "__main__":
    args = parser.parse_args()
    test_main(args)
