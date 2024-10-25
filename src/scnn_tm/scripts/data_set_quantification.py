import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate

from scnn_tm.utils import (
    get_cloud_bounds,
    mask_point_in_bounds,
    overlay_two_clouds,
    associate_points_to_voxel_cloud,
    point_to_voxel_centre,
)
import copy

DATA_SET = {
    # 1: "/data/processed/feature_sets/lfe_hl_v0.1/2021_12_14_00_02_12Z",
    # 2: "/data/processed/feature_sets/lfe_hl_v0.1/2021_12_14_00_14_53Z",
    # 3: "/data/processed/feature_sets/lfe_hl_v0.1/2021_12_13_23_51_33Z",
    # 4: "/data/processed/feature_sets/lfe_hl_v0.1/2021_12_13_23_42_55Z",
    # 5: "/data/processed/feature_sets/lfe_hl_v0.1/2021_10_28_03_54_49Z",
    # 6: "/data/processed/feature_sets/lfe_hl_v0.1/2021_11_18_00_23_24Z",
    # 7: "/data/processed/feature_sets/lfe_hl_v0.1/2021_10_28_04_03_11Z",
    # 8: "/data/processed/feature_sets/lfe_hl_v0.1/2021_10_28_04_13_12Z",
    9: "/data/processed/feature_sets/lfe_hl_v0.1/2022_02_14_23_47_33Z",
}


def get_map_statistics(data_set_dir: Path, voxel_size=0.1) -> list:
    class_0 = pd.read_csv(data_set_dir / "semantic_cloud_class_1.csv")
    nte_examples_total = class_0.shape[0]
    nte_examples_lfe = class_0[class_0["label"] < 3.0].shape[0]

    class_1 = pd.read_csv(data_set_dir / "semantic_cloud_class_2.csv")
    te_examples_total = class_1.shape[0]
    te_examples_lfe = class_1[class_1["label"] < 3.0].shape[0]

    data_set_df = pd.concat([class_0, class_1])

    # Get dimensions
    coords = data_set_df[["x", "y", "z"]].to_numpy()
    map_dim = get_cloud_bounds(coords)
    dx = np.abs(map_dim[0] - map_dim[3])
    dy = np.abs(map_dim[1] - map_dim[4])
    dz = np.abs(map_dim[2] - map_dim[5])

    #
    total_samples = nte_examples_total + te_examples_total
    p_te_hl = np.round( (te_examples_total - te_examples_lfe) / total_samples, 3)
    p_te_lfe = np.round(  te_examples_lfe / total_samples, 3)
    p_nte_hl = np.round( (nte_examples_total - nte_examples_lfe) / total_samples, 3)
    p_nte_lfe = np.round(  nte_examples_lfe / total_samples, 3)
    
    density = above_ground_density(cloud=coords, bounds=map_dim, voxel_size=0.1, n_voxels=10)
    
    mean_density = np.mean(density)

    return [
        np.round(total_samples,2),
        np.round(p_te_hl,2),
        np.round(p_te_lfe,2),
        np.round(p_nte_hl,2),
        np.round(p_nte_lfe,2),
        np.round(mean_density,2),
        np.round(dx,2),
        np.round(dy,2),
        np.round(dz,2),
    ]


    

def above_ground_density(cloud, bounds, voxel_size, n_voxels):
    p_c_min = point_to_voxel_centre(np.array(bounds[0:3]), voxel_size=voxel_size)
    
    
    dx_i = int(np.abs(bounds[3] - bounds[0]) / 0.1 )
    dy_i = int(np.abs(bounds[4] - bounds[1]) / 0.1 )

    collum_density = []
    for x_i in range(0, dx_i):
        for y_i in range(0, dy_i):
            x_i_p = p_c_min[0] + 0.1 * float(x_i)
            y_i_p = p_c_min[1] + 0.1 * float(y_i)

            density = calculate_collume_density(
                cloud,
                x_i_c=x_i_p,
                y_i_c=y_i_p,
                z_min=bounds[2],
                z_max=bounds[5],
                voxel_size=voxel_size,
                n_voxels=n_voxels,
            )

            if density < 0:
                continue

            collum_density.append(density)
    
    return np.array(collum_density)


def calculate_collume_density(cloud, x_i_c, y_i_c, z_min, z_max, voxel_size, n_voxels):
    # Find ground point
    candidate_points = [
        np.array([x_i_c, y_i_c, z_min + 0.1 * float(z_i)])
        for z_i in range(int(np.abs(z_min - z_max) / voxel_size))
    ]

    associated_ids = associate_points_to_voxel_cloud(
        source_cloud=candidate_points,
        target_cloud=cloud,
        voxel_size=0.1,
        non_found_padding=True,
    )

    ground_point_found = False
    collume_mean_density = -1.0
    n_voxels_checked = 0
    for ids in associated_ids:
        # We have not found the ground point yet
        if not ground_point_found:
            if ids[0] < 0:
                continue
            else:
                ground_point_found = True
                collume_mean_density = 1.0

        # Have a ground point and need to break if we have check n_voxels
        if not (n_voxels_checked < n_voxels):
            break

        if ids[0] >= 0:
            collume_mean_density += 1.0

        # Updated that we checked the voxel
        n_voxels_checked += 1.0

    # Return the collum mean
    return collume_mean_density / n_voxels


map_statistic = []
for key, value in DATA_SET.items():
    map_statistic.append(get_map_statistics(Path(value), voxel_size=0.1))
print(tabulate(map_statistic))
