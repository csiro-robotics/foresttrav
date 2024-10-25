import numpy as np
import pandas as pd
from scnn_tm.utils import (
    point_in_voxel,
    associate_points_to_voxel_cloud,
    point_to_voxel_centre,
)


# Test if a point lies within a voxel and some variation
def test_point_in_voxels():
    # Simple test in voxel
    source_query_point = np.array([[0.05, 0.01, 0.01]])
    target_querry_point = np.array([[0.05, 0.05, 0.05]])
    assert point_in_voxel(source_query_point[0], target_querry_point[0], 0.1)

    source_query_point = np.array([[0.05, 0.01, 0.01]])
    target_querry_point = np.array([[0.00, 0.00, 0.00]])
    assert point_in_voxel(source_query_point[0], target_querry_point[0], 0.1)

    source_query_point = np.array([[0.00, 0.00, 0.00]])
    target_querry_point = np.array([[0.00, 0.00, 0.00]])
    assert point_in_voxel(source_query_point[0], target_querry_point[0], 0.1)

    # Moved one voxel and it should failr
    source_query_point = np.array([[0.1, 0.01, 0.01]])
    target_querry_point = np.array([[0.00, 0.00, 0.00]])
    assert not point_in_voxel(source_query_point[0], target_querry_point[0], 0.1)


def test_voxel_ids():
    points = np.array(
        [
            np.array([x, y, 0.01])
            for x in np.arange(-1.0, 1.0, 0.1)
            for y in np.arange(-1.0, 1.0, 0.1)
        ]
    )

    for i in range(points.shape[0]):
        new_points = np.array([points[i]])
        id_pair = associate_points_to_voxel_cloud(new_points, points, 0.1)

        assert i == id_pair[0][1]
        assert (new_points[0] == points[id_pair[0][1]]).all()


def test_voxel_on_test_data():
    VOXEL_SIZE = 0.1
    SOURCE_DATA = "/nve_ml_ws/src/scnn_tm/tests/config/scan_comparison_ohm_scan.csv"
    TARGET_DATA = "/nve_ml_ws/src/scnn_tm/tests/config/scan_comparsion_gt_ohm_map.csv"
    source_data = pd.read_csv(SOURCE_DATA)[["x", "y", "z"]].to_numpy()
    target_data = pd.read_csv(TARGET_DATA)[["x", "y", "z"]].to_numpy()

    # Find ids for corresponding ids between target and source cloud
    id_pairs = associate_points_to_voxel_cloud(
        source_cloud=source_data, target_cloud=target_data, voxel_size=VOXEL_SIZE
    )

    # Go trough all the points that found id and check that the voxel centres match!

    for id_pair in id_pairs:
        voxcel_centre_source = point_to_voxel_centre(source_data[id_pair[0], 0:3], VOXEL_SIZE)
        voxcel_centre_target = point_to_voxel_centre(target_data[id_pair[1], 0:3], VOXEL_SIZE)
        assert (voxcel_centre_source == voxcel_centre_target).all()



# def test_align_and_associate_voxel_cloud():
#     VOXEL_SIZE = 0.1
#     SOURCE_DATA = "/nve_ml_ws/src/scnn_tm/tests/config/scan_comparison_ohm_scan.csv"
#     TARGET_DATA = "/nve_ml_ws/src/scnn_tm/tests/config/scan_comparsion_gt_ohm_map.csv"
#     source_data = pd.read_csv(SOURCE_DATA)[["x", "y", "z"]].to_numpy()
#     target_data = pd.read_csv(TARGET_DATA)[["x", "y", "z"]].to_numpy()

#     # Find ids for corresponding ids between target and source cloud
#     id_pairs = (
#         source_cloud=source_data, target_cloud=target_data, voxel_size=VOXEL_SIZE
#     )

#     # Go trough all the points that found id and check that the voxel centres match!

#     for id_pair in id_pairs:
#         voxcel_centre_source = point_to_voxel_centre(source_data[id_pair[0], 0:3], VOXEL_SIZE)
#         voxcel_centre_target = point_to_voxel_centre(target_data[id_pair[1], 0:3], VOXEL_SIZE)
#         assert (voxcel_centre_source == voxcel_centre_target).all()
    
