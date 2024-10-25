# Copyright (c) 2024
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

import pandas as pd
import numpy as np
from pathlib import Path
from scnn_tm.utils import (
    get_cloud_bounds,
    points_to_voxel_ids,
    id_to_voxel_centre,
)



def costmap_conversion(
    point_cloud: np.array,
    voxel_size: float,
    voxels_per_collum: int,
):
    """
    ARGS:
        point_cloud (np.array): Pointcloud with fields [x,y,z,te_prob]
        voxel_size (float): Voxel size [m]
        point_per_collum (int): Points per collum for 3D to 2.5D conversion

    RETURN:
        dense_costmap
    """

    # Get the voxel ids
    voxel_ids = points_to_voxel_ids(point_cloud, voxel_size)

    # Initialise the cloud
    bounds = get_cloud_bounds(point_cloud)
    min_voxel_id = points_to_voxel_ids(
        np.array([bounds[0], bounds[1], bounds[2]]), voxel_size
    )
    max_voxel_id = points_to_voxel_ids(
        np.array([bounds[3], bounds[4], bounds[5]]), voxel_size
    )

    collum_cloud = {
        (voxel_idx, voxel_idy): []
        for voxel_idx in range(min_voxel_id[0], max_voxel_id[0]+1)
        for voxel_idy in range(min_voxel_id[1], max_voxel_id[1]+1)
    }
    b = 1
    
    # point := [x,y,z, te_prob]
    for voxel_id, row in zip(voxel_ids, point_cloud):
        collum_cloud[(voxel_id[0],voxel_id[1])].append(row[2:4])

    # Cost each
    costmap, empty_spaces = collum_costing_simple(
        collum_cloud=collum_cloud, voxels_per_collum=voxels_per_collum
    )

    # Fill out the missing gaps

    # DEBUG OUTPUT not the actual cloud coordinates but the voxel centres
    costmap_as_pcl = [
        list(id_to_voxel_centre(key[0], key[1], 0, voxel_size=voxel_size)[:2]) + values
        for key, values in costmap.items()
    ]
    return  np.array(costmap_as_pcl)



def collum_costing_simple(collum_cloud: np.array, voxels_per_collum: int):
    """ """
    missing_cloud_keys = {}
    for key, value in collum_cloud.items():

        # No data, contiue
        if not value:
            collum_cloud[key] = [-99.0, -1.0]#[-np.inf, -np.inf]
            missing_cloud_keys[key] = False
            continue

        # values [z, te_prob] sort by ascending z
        collum_arr = np.array(value)
        sorted_indices = np.argsort(collum_arr[:, 0])
        collum_arr = collum_arr[sorted_indices]
        # Set the values to z=-inf if there are no elements and te_prob
        if collum_arr.shape[0] > voxels_per_collum:
            collum_cloud[key] = [
                collum_arr[0, 0],
                np.mean(collum_arr[:voxels_per_collum, 1]),
            ]
        else:
            collum_cloud[key] = [collum_arr[0, 0], np.mean(collum_arr[:, 1])]

    return collum_cloud, missing_cloud_keys


import timeit
DEBUG_OUT = "/data/debug/test_costmap_conversion"
def debug_main():
    out_dir = Path(DEBUG_OUT)
    pcl_file = Path(
        "/data/forest_trav/lfe_hl_v0.1/1_2021_12_14_00_02_12Z.csv"
    )


    df = pd.read_csv(pcl_file)
    data = df[["x", "y", "z", "label_prob"]].to_numpy()

    start_time = timeit.default_timer()

    # Generate the 2d costmap
    cosmap = costmap_conversion(point_cloud=data, voxel_size=0.1, voxels_per_collum=10)
    end_time = timeit.default_timer()

    execution_time = end_time - start_time

    execution_time_milliseconds = execution_time * 1000

    print("Execution time:", execution_time_milliseconds, "milliseconds")
    # Save the 2d costmap

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / pcl_file.name
    pd.DataFrame(cosmap).to_csv(
        out_file, header=["x", "y", "z", "mean_cost",], index=False
    )
    


if __name__ == "__main__":
    debug_main()
