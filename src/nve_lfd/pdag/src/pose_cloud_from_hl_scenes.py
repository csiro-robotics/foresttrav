from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.neighbors import KDTree

import io_helper
from dataclasses import dataclass

SEMANTIC_CLOUD_FILE = (
    "/data/debug/debug_hl_pose_clouds/2021_12_14_00_14_53Z/hl_feature_cloud"
)
GROUND_CLOUD_FILE = "/data/debug/debug_hl_pose_clouds/2021_12_14_00_14_53Z/ground_cloud"
POSE_CLOUD_DIR = Path(
    "/data/debug/debug_hl_pose_clouds/2021_12_14_00_14_53Z/hl_pose_clouds"
)
SANITY_CLOUD_DIR = POSE_CLOUD_DIR.parent / "hl_pose_sanity_cloud"
NUMBER_OF_POSES = 1000


def main_test():
    data = np.random.rand(100, 3)
    data[:, 0] = data[:, 0] * 10.0
    data[:, 1] = data[:, 1] * 10.0
    data[:, 2] = data[:, 2] * 1.2
    # print(data)

    sceen_search_tree = KDTree(data)

    point = np.random.rand(1, 3)
    point[:, 0] = point[:, 0] * 10.0
    point[:, 1] = point[:, 1] * 10.0
    point[:, 2] = point[:, 2] * 1.2
    # print(point)

    T_mr_hat = align_pose_to_ground(point[0], sceen_search_tree, data)

    T_rha_m = np.linalg.inv(T_mr_hat)
    p_i = np.array([0.0, 0.0, 0.0, 0.0])
    p_i[0:3] = data[3]
    print(T_rha_m * p_i.transpose())


@dataclass
class SamplingBounds:
    """Sampling bounds for a given map, in asbolute values."""

    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    phi_min: float = 0.0
    phi_max: float = 2 * np.pi


@dataclass
class RobotBounds:
    """Robot bounds for a given map, in asbolute values."""

    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    radius: float = 0.0


class PoseCloud:
    def __init__(self, pose: np.ndarray):
        self.pose = pose
        self.cloud = None
        self.cloud_features = None

    def get_pose(self):
        return self.get_pose


# TODO: Sort such that the collums are ot label, x, y, z
# - [ ]: Sort such that the collums are ot  x, y, z, label


def main():

    # Load the necessary data and files
    semantic_data_df = io_helper.load_all_csv(SEMANTIC_CLOUD_FILE)

    # Reorder the collums to be x, y, z, label,....
    semantic_data_df.rename(
        columns={"pos_x": "x", "pos_y": "y", "pos_z": "z"}, inplace=True
    )
    pos_header = ["x", "y", "z", "label"]
    new_header = [e for e in semantic_data_df.columns if e not in pos_header]
    semantic_data_df = semantic_data_df.reindex(pos_header + new_header, axis=1)

    ground_cloud_df = io_helper.read_all_ply_files_as_df(GROUND_CLOUD_FILE)

    # Get robot bounds
    semantic_cloud = semantic_data_df.to_numpy()

    # Only positions for the kdtree
    semantic_cloud_xyz = semantic_data_df[["x", "y", "z"]].to_numpy()
    ground_cloud_xyz = ground_cloud_df[["x", "y", "z"]].to_numpy()

    # Generate sampling bounds
    # TODO: Genetate the sampling bounds from the semantic cloud
    bounds_arr = get_map_bounds(semantic_cloud_xyz)
    bounds = SamplingBounds(
        x_min=bounds_arr[0],
        x_max=bounds_arr[1],
        y_min=bounds_arr[2],
        y_max=bounds_arr[3],
    )

    robot_bounds = RobotBounds(
        x_min=-0.8, x_max=0.2, y_min=-0.5, y_max=0.5, z_min=-1.0, z_max=0.5, radius=1.0
    )
    poses = [random_pose(bounds) for i in range(NUMBER_OF_POSES)]

    # Generate ground aligned poses
    ground_knn_tree = KDTree(ground_cloud_xyz)
    ground_aligned_poses = [
        align_pose_to_ground(pose, ground_knn_tree, ground_cloud_xyz) for pose in poses
    ]

    # Generate traversability assesment for ground_aligned_poses
    semantic_knn_tree = KDTree(semantic_cloud_xyz)
    traversability = [
        check_pose_for_collision(
            pose, semantic_knn_tree, semantic_cloud, robot_bounds=robot_bounds
        )
        for pose in ground_aligned_poses
    ]

    # Save the poses and traversability as ply files
    local_clouds = []
    local_clouds_m = []
    for pose in ground_aligned_poses:
        local_cloud, local_cloud_M = generate_feature_cloud_in_robto_frame(
            pose, semantic_knn_tree, semantic_cloud, robot_bounds=robot_bounds
        )
        local_clouds_m.append(local_cloud_M)
        local_clouds.append(local_cloud)
        assert len(local_cloud) == len(local_cloud_M)

    # Save the poses and traversability as ply files
    final_cloud = np.ndarray(shape=(0, semantic_cloud.shape[1]))
    for i, pose_cloud in enumerate(local_clouds_m):

        header_semantic_df = semantic_data_df.columns.to_list()
        if pose_cloud.shape[0] < 1:
            continue

        pose_cloud[:, header_semantic_df.index("label")] = traversability[i]
        final_cloud = np.concatenate((final_cloud, pose_cloud), axis=0)

    filename = SANITY_CLOUD_DIR / "sanity_cloud.csv"
    df = pd.DataFrame(
        data=final_cloud[:, 0:4], index=None, columns=header_semantic_df[0:4]
    )
    df.to_csv(filename, sep=" ", index=False, header=True, float_format="%.6f")

    # Store the indivudual poses
    for i, pose_cloud in enumerate(local_clouds):

        header_semantic_df = semantic_data_df.columns.to_list()
        filename = POSE_CLOUD_DIR / f"pose_cloud{i}.csv"

        if pose_cloud.shape[0] < 1:
            continue

        pose_cloud[:, header_semantic_df.index("label")] = traversability[i]
        header_semantic_df[header_semantic_df.index("label")] = "traversability"

        df = pd.DataFrame(
            data=pose_cloud[:, 0:7], index=None, columns=header_semantic_df[0:7]
        )
        df.to_csv(filename, sep=" ", index=False, header=True, float_format="%.6f")


# PCL utils
def get_map_bounds(cloud: np.array, padding: float = 0.0):
    """Get the map bounds from the semantic cloud"""
    x_min = np.min(cloud[:, 0] + padding)
    x_max = np.max(cloud[:, 0] - padding)
    y_min = np.min(cloud[:, 1] + padding)
    y_max = np.max(cloud[:, 1] - padding)
    z_min = np.min(cloud[:, 2] + padding)
    z_max = np.max(cloud[:, 2] - padding)
    return x_min, x_max, y_min, y_max, z_min, z_max


def generate_feature_cloud_in_robto_frame(
    T_mr: np.array, kdtree: KDTree, cloud_arr: np.array, robot_bounds
):
    """ " """
    # Find the subset of points close to the pose
    inds = kdtree.query_radius(
        T_mr[0:3, 3].reshape(1, -1), r=robot_bounds.radius, return_distance=False
    )
    if len(inds[0]) < 1:
        return np.ndarray((0, 0)), np.ndarray((0, 0))

    # Transform the points to the robot frame
    T_r_m = np.linalg.inv(T_mr)

    # Check if the points are within the robot bounds and store the robot centric point cloud
    cloud_transformed = cloud_arr[inds[0]]
    transformed_xyz = [do_transform(p, T_r_m) for p in cloud_transformed]
    valid_ids = [
        check_point_within_bounds(p_i_r[0:3], robot_bounds) for p_i_r in transformed_xyz
    ]
    cloud_transformed[:, 0:3] = transformed_xyz

    # Note this could be solved by dealing with the ids of the points rather than the points themselves
    return cloud_transformed[valid_ids], cloud_arr[inds[0]][valid_ids]


def do_transform(point: np.array, tansform: np.ndarray):
    """Transform a point with a given transform
    Assumes the first 3 elements of the point are the position

    """
    return tansform[0:3, 0:3].dot(point[0:3].transpose()) + tansform[0:3, 3]


def check_pose_for_collision(
    T_mr: np.array, kdtree: KDTree, data_arr: np.array, robot_bounds
) -> bool:
    """Checks if the pose is in collision using the semantic cloud. The labels are 1 for collision and 2 for free-space
    Args:
      T_mr: The pose to check for collision
      kdtree: The kdtree of the semantic cloud
      data_arr: The semantic cloud with [x,y,z,label] format
      robot_bounds: The robot bounds in robot centric frame

    Returns:
      True if the pose is in collision, False otherwise.
    """

    collision_label = 0

    # Check if any points are found within the radius and abort if not so
    inds = kdtree.query_radius(
        T_mr[0:3, 3].reshape(1, -1), r=robot_bounds.radius, return_distance=False
    )
    if len(inds[0]) < 1:
        return

    # Get inverse to get the transfrom from map to robot frame
    # TODO: Check if there is faster invers function for numpy
    T_r_m = np.linalg.inv(T_mr)

    # Check if any of the points are in collision, by tranforming the points to the pose frame and checking the bounding box
    for idx in inds[0]:
        p_i_m = np.ones(shape=(4))
        p_i_m[0:3] = data_arr[idx, 0:3]
        l_i = data_arr[idx, 3]

        # Only need to check points if there are non-traversable
        if l_i != collision_label:
            continue

        # T_mr
        p_i_r = T_r_m.dot(p_i_m)

        if check_point_within_bounds(p_i_r, robot_bounds):
            return True

    # No collision points were found
    return False


def check_point_within_bounds(p_i: np.array, robot_bounds: RobotBounds) -> bool:
    """Checks if the point is within a bounding box representing the robots dimensions
    Args:
      p_i: The point to check
      robot_bounds: The robot bounds in robot centric frame [x_min, x_max, y_min, y_max, z_min, z_max, r_max]

    Returns:
      True if the point is within the bounds, False otherwise
    """
    if (
        p_i[0] > robot_bounds.x_min
        and p_i[0] < robot_bounds.x_max
        and p_i[1] > robot_bounds.y_min
        and p_i[1] < robot_bounds.y_max
        and p_i[2] > robot_bounds.z_min
        and p_i[2] < robot_bounds.z_max
    ):
        return True
    return False


def random_pose(bounds: SamplingBounds) -> np.ndarray:
    """Generates a random pose using uniform sampling with random position and yaw
    Args:
      bounds: The bounds to sample from @class SamplingBounds

    Returns:
      The pose as a 4x4 transformation matrix
    """
    x = np.random.uniform(bounds.x_min, bounds.x_max)
    y = np.random.uniform(bounds.y_min, bounds.y_max)
    z = 0.0
    phi = np.random.uniform(bounds.phi_min, bounds.phi_max)

    pose = np.zeros((4, 4))
    pose[0:3, 0:3] = [
        [np.cos(phi), -np.sin(phi), 0.0],
        [np.sin(phi), np.cos(phi), 0.0],
        [0.0, 0.0, 1.0],
    ]
    pose[0:3, 3] = [x, y, z]
    pose[3, 3] = 1.0

    return pose


# TODO: There needs to be a check to see if this makes sense!
def align_pose_to_ground(
    T_mr: np.ndarray, kdtree: KDTree, data_arr: np.array
) -> np.ndarray:
    """Aligns the pose to the ground plane based on the normal of the ground plane.
    Equations from "Driving on Point Clouds, eq 22 - 28

    Args:
      t_mr: The pose to align to the ground plane
      kdtree: The kdtree of the point cloud
      data_arr: The point cloud with [x,y,z] format

    Returns:
      The aligned pose as 4x4 transformation matrix T_map_robotground
    """

    # Get the closest ground point t_mr_g
    t_mr_g = data_arr[
        kdtree.query(T_mr[0:3, 3].reshape(1, -1), k=1, return_distance=False)[0][0]
    ]
    inds = kdtree.query_radius(t_mr_g.reshape(1, -1), r=0.5, return_distance=False)

    # if len(inds[0]) < 10:
    if True:
        print("Used unit transform for ground pose")
        T_mr[0:3, 3] = t_mr_g
        return T_mr

    mean_point = np.mean(data_arr[inds[0]], axis=0)
    cov_point = np.cov(data_arr[inds[0]].transpose())

    w, v = np.linalg.eigh(cov_point)

    # Axis vectors in world frame
    x_mr = T_mr[0:3, 0]
    y_mr = T_mr[0:3, 1]
    z_mr = T_mr[0:3, 2]

    # Normal vector of plane using x_axis as prior
    n = np.sign(np.dot(x_mr, v[0])) * v[0]

    x_m_rhat = np.cross(y_mr, n)
    x_m_rhat /= np.linalg.norm(x_m_rhat)

    R = np.zeros((3, 3))
    R[0:3, 0] = x_m_rhat
    R[0:3, 1] = np.cross(x_m_rhat, n)
    R[0:3, 2] = n

    # t_m_rhat = t_mr_g + np.dot((mean_point - t_mr),n) / np.dot(n,z_mr) *z_mr
    t_m_rhat = t_mr_g

    T_m_rhat = np.zeros((4, 4))
    T_m_rhat[0:3, 0:3] = R
    T_m_rhat[0:3, 3] = t_m_rhat
    T_m_rhat[3, 3] = 1.0

    return T_m_rhat


def main_load_test():
    """Main function for testing the loading of the point cloud"""
    # Load the point cloud
    semantic_data_df = io_helper.read_all_ply_files_as_df(SEMANTIC_CLOUD_FILE)
    ground_cloud_df = io_helper.read_all_ply_files_as_df(GROUND_CLOUD_FILE)

    # Get robot bounds
    semantic_cloud = semantic_data_df.to_numpy()

    # Only positions for the kdtree
    semantic_cloud_xyz = semantic_data_df[["x", "y", "z"]].to_numpy()
    ground_cloud_xyz = ground_cloud_df[["x", "y", "z"]].to_numpy()


if __name__ == "__main__":
    main()
    # main_test()
    # main_load_test()
