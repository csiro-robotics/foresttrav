# MIT License
#
# Copyright (c) 2023 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

import numpy as np
from sklearn.neighbors import KDTree

""" Methods to associate to index and find point in or between voxalised point cloud

Assumption: 
    (1) Voxel position starts at (0,0,0), front bottom left corner with no shift. 
    (2) Empty voxel are expected to retrun the centre pointi i.e (0.05, 0.05, 0.05) for the point (0,0,0) and voxel size 0.1
    (3) Eacn voxel can contain only a single point
    (4) A point is contained in a voxel if it falls on the lower bounds, but not the upper, (0,0,0) 
"""


def point_to_voxel_id(point: np.array, voxel_size: float) -> np.array:
    return (point // voxel_size).astype(int)

def points_to_voxel_ids(point_cloud: np.array, voxel_size: float) -> np.array:
    return (point_cloud // voxel_size).astype(int)


def point_in_voxel(source_querry_point, target_voxel_point, voxel_size):
    """Checks if the querry point lies within the voxel bounds given by the voxel point.
    Assumptions:
        -   Voxels start at (0,0,0) and have same leave size for all sides
        -
    Args:
        - querry_point: Point to check
        - voxel_point: Point of the voxel and from which the bounds of the voxels are infered
        - voxel_size: Size of the voxel leafs
    Retrun:
        True if the queery_point lies within a voxel, False otherwise
    """
    return (
        point_to_voxel_id(point=source_querry_point, voxel_size=voxel_size)
        == point_to_voxel_id(point=target_voxel_point, voxel_size=voxel_size)
    ).all()


def id_to_voxel_centre(id_x, id_y, id_z, voxel_size):
    return np.array(
        [
            float(id_x) * voxel_size + voxel_size / 2.0,
            float(id_y) * voxel_size + voxel_size / 2.0,
            float(id_z) * voxel_size + voxel_size / 2.0,
        ]
    )


def points_to_voxel_centre(point_cloud, voxel_size):
    return voxel_size * (point_cloud // voxel_size) + voxel_size / 2.0


def find_point_pairs(
    source_points: np.array,
    target_points: np.array,
    voxel_size: float,
):
    """Finds the id pairs which match source and target cloud given a voxelised point cloud.
        Uses the knowledge that each voxel should be a) unique and b) x,y,z can be used as
        a hash key

    Args:
        source_cloud:   np.ndarray    Contains the position of the
        target_cloud:   np.ndarray    target cloud we whish the points to
        voxel_size:     float         Voxel leaf size, unifrom for all sides

    Return:
        id pairs:       [id_source, id_target]
    """

    # t0 = time.perf_counter()
    target_voxel_centre = points_to_voxel_ids(target_points, voxel_size)
    target_point_keys = {
        tuple(target_point): index
        for index, target_point in enumerate(target_voxel_centre)
    }

    # t1 = time.perf_counter()
    source_voxel_centre = points_to_voxel_ids(source_points, voxel_size)
    source_point_keys = {
        tuple(source_point): index
        for index, source_point in enumerate(source_voxel_centre)
    }
    # t2= time.perf_counter()
    shared_keys = set(target_point_keys.keys()) & set(source_point_keys.keys())

    # t3 = time.perf_counter()
    id_pairs = [
        (source_point_keys[key_i], target_point_keys[key_i]) for key_i in shared_keys
    ]
    # t4 = time.perf_counter()

    # print(f"Time duration of target_point voxelisation:{t1-t0}")
    # print(f"Time duration of source_point_voxelisationt:{t2-t1}")
    # print(f"Time duration of shared_keyst:{t3-t2}")
    # print(f"Time duration of comparsion t:{t4-t3}")

    return id_pairs


def get_voxel_cloud_bounds(
    cloud: np.ndarray,
    voxel_size: float,
    padding: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
) -> list:
    """Generates box bounds for a voxel cloud (voxelised point cloud). Assumes origin to be at [0,0,0]

    Args:
        cloud (np.ndarray): PointCloud in global frame NxD
        padding (float, optional): Padding added to the points. Defaults to 0.0.

    Returns:
        list: _description_
    """
    x_min = np.min(cloud[:, 0] - padding[0])
    y_min = np.min(cloud[:, 1] - padding[1])
    z_min = np.min(cloud[:, 2] - padding[2])

    p_min = (
        points_to_voxel_centre(np.array([x_min, y_min, z_min]), voxel_size)
        - voxel_size / 2.0
    )

    x_max = np.max(cloud[:, 0] + padding[3])
    y_max = np.max(cloud[:, 1] + padding[4])
    z_max = np.max(cloud[:, 2] + padding[5])
    p_max = (
        points_to_voxel_centre(np.array([x_max, y_max, z_max]), voxel_size)
        + voxel_size / 2.0
    )

    return [p_min[0], p_min[1], p_min[2], p_max[0], p_max[1], p_max[2]]
