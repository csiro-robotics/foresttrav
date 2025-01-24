# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz


import copy
import time

import numpy as np

from scnn_tm.utils import (
    find_point_pairs,
    get_voxel_cloud_bounds,
    point_to_voxel_id,
    points_to_voxel_ids,
    mask_for_points_within_bounds,
)


def pose_to_key(pose, voxel_size) -> tuple:
    """Makes an imutable tuple out of a pose for a voxel centre with given voxel size

    Args:
        pose (np.array):    Pose given as nxk vector, where the first three element are the x,y,z coords [m]
        voxel_size (float): Voxel size for which the centre is defined

    Returns:
        tuple: Hashable voxel index (idx, idy, idz)
    """
    point = point_to_voxel_id(pose[:3], voxel_size)
    return (point[0], point[1], point[2])


def mask_for_pose(pose, voxel_size, patch_width, cloud):
    """Generates a mask for a given pose p with a box around patch width (patch_width) for a point cloud

    Args:
        pose (np.array): Given pose
        voxel_size (float): voxel size of the voxel map [m]
        patch_width (float): width of the patch around the pose [m]
        cloud (np.ndarray): Coordinates of the point cloud (n,3) [m]

    Returns:
        np.array((n,1) bool: Logical mask of all points within the bounding box around the pose
    """
    # Hacky way to extend to get bounds by adding padding to a point to get a box
    padding = [
        patch_width / 2.0,
        patch_width / 2.0,
        patch_width / 2.0,
        patch_width / 2.0,
        patch_width / 2.0,
        patch_width / 2.0,
    ]
    return mask_for_points_within_bounds(
        cloud=cloud,
        bounds=get_voxel_cloud_bounds(pose, voxel_size=voxel_size, padding=padding),
    )


class ForestTravGraph:
    """Forest data graph uses "nodes" as way to store and maintain data

    Each node is defined by a key, where the pose x,y,z is converted to a voxel-centre tuple (hashable)
    Each node contains the data fields "coords", "features", "labels", "label_prob" and "label_obs" and labels. Each of the fields index corresponds to others
        coords: Cooridnates in static global frame (do not transform, that will violate with the voxel grid structure assumption)
        feature: Features of the feature clouds. Feature cloud keys (name of the fearues)
        labels_prob: Labels associated to the features, in the binary case [0.0,1.0] where -1 is used for unknonw/unobserverd
        label_obs: Binari label to encoude if the voxel has been directly observed {1} or the estimate is generate differently{0},

    """

    def __init__(
        self,
        voxel_size,
        patch_width,
        min_dist_r,
        debug_prints=False,
        use_unlabeled_data=True,
    ) -> None:
        """Initialisation of the graph.

        Args:
            voxel_size (float):     Voxel size [m]
            path_width (int):       Number of voxels when generating the training "cubes" of the data set
            min_dist_r  (float):    Minimum distrance between two poses in [m]
            debug_prints (bool):    Flag to enable debug printing, and timing
        """
        self.data_fields = {}
        self.keys = iter(self.data_fields)

        self.voxel_size = voxel_size
        self.min_dist_r = min_dist_r
        self.patch_width = patch_width
        self.key_distance = 0.1  # Random number much smaller than radius

        self.debug_prints = debug_prints
        self.use_unlabeled_data = use_unlabeled_data

    def add_new_data(self, new_data_batch):
        """Adding a new data batch to the graph


        Args:
            new_data_batch (dict): Data batch containing all the relevant fields for the graph
        """

        # Initialize graph if empty
        times = [time.perf_counter()]  # 0
        self.initialize_new_graph(data_batch=new_data_batch)

        # Add all poses to the graph that are valid
        newly_added_poses = self.add_valid_poses_to_the_graph(data_batch=new_data_batch)
        times.append(time.perf_counter())  # 1

        # Process the collision and feature map
        fused_map = self.fuse_data(data_batch=new_data_batch)
        if fused_map is None:
            return False
        times.append(time.perf_counter())  # 2

        # Generate possible candidate poses
        # Case 1: Poses inside the current feature map
        # Case 2: Poses partially inside the current feature map (ignored for now)
        # TODO(fab): Implement partial fusion
        # TODO(fab): Case where there is only one valid pose, the initial one
        all_graph_poses = {
            key: data_field["pose"]
            for key, data_field in self.data_fields.items()
            if key not in newly_added_poses
        }

        # If no poses are found, return false:
        if not all_graph_poses:
            return False

        poses_inside_fm, poses_part_inside_fm = generate_valid_poses(
            candidate_poses=all_graph_poses,
            cloud_coord=fused_map["coords"],
            voxel_size=self.voxel_size,
            patch_width=self.patch_width,
        )
        # Assumption that newly added poses must be contained in the fused map
        newly_added_poses.update(poses_inside_fm)
        times.append(time.perf_counter())  # 3

        # Generate the new nodes and add them to the graph
        self.update_poses_with_new_data(
            poses_to_update=newly_added_poses, data=fused_map
        )
        times.append(time.perf_counter())  # 4

        if self.debug_prints:
            print("New Iteration")
            print(f"Found {len(newly_added_poses)} new poeses")
            print(f"Updated { self.num_of_updated_poses_last_cycle} new new_poses")
            print(f"Total number of nodes { len(self.data_fields)} ")
            print(f"Elapsed time  add_valid_poses:  {  times[1] - times[0]}")
            print(f"Elapsed time  fuse_data:  {  times[2] - times[1]}")
            print(f"Elapsed time  generate_valid_poses:  {   times[3] - times[2]}")
            print(
                f"Elapsed time  update_poses_with_new_data:  {   times[4] - times[3]}"
            )

    def initialize_new_graph(self, data_batch):
        """Zero initialization where we pop the first item"""

        if len(self.data_fields) < 1:
            pose_i = data_batch["poses"].pop(0)
            self[pose_i] = {"pose": pose_i}

    def add_valid_poses_to_the_graph(self, data_batch):
        """Adds new poses to the graph with empty data fields

        Args:
            data_batch (dict): Data as dictionary, se class description
        """
        new_poses_added = {}  # Keeping track of new poses
        
        current_graph_poses = np.array(
            [value["pose"][:3] for key, value in self.data_fields.items()]
        )  # All curent poses in graph
        
        for pose_i in data_batch["poses"]:
            # TODO: Change condition to something else than a distance?
            # TODO: Make this a criterion/loss function than can be swamped?
            if (
                np.linalg.norm(current_graph_poses - pose_i[:3], axis=1)
                < self.min_dist_r
            ).any():
                continue

            self[pose_i] = {"pose": pose_i}

            new_poses_added[pose_to_key(pose_i, self.voxel_size)] = pose_i
            current_graph_poses = np.vstack([current_graph_poses, pose_i[:3]])
            # Need to add poses to ensure we

        return new_poses_added

    def update_poses_with_new_data(self, poses_to_update, data):
        """_summary_

        Args:
            poses_to_update (_type_): _description_
            data (_type_): _description_
        """

        # This is very slow. Lets see if we can paralize this
        self.num_of_updated_poses_last_cycle = 0
        for key, pose in poses_to_update.items():
            mask = mask_for_pose(
                pose.reshape(1, 7),
                self.voxel_size,
                patch_width=self.patch_width,
                cloud=data["coords"],
            )

            # Continue if not valid
            if not mask.any():
                continue

            self[key] = {
                "pose": pose,
                "features": data["features"][mask],
                "coords": data["coords"][mask],
                "label_prob": data["label_prob"][mask],
                "label_obs": data["label_obs"][mask],
            }
            self.num_of_updated_poses_last_cycle += 1

    def fuse_data(self, data_batch: dict):
        """Function to fuse all the data. First the feature map is generated to stich together all the various observations. Assumption is that the the feature cloud are orded in
        descedning importance. For the voxels in the feature map, the collision data is associated, label_prob in [0,1] for observed cells.

        Args:
            data_batch (dict): New data batch. Each field, "coords", "feature_clouds" are ordered in the descending order wiht descending time stamps (newer to older). Ther is only a single "collision_map" field, that contains the lfe update.
        """
        # Fuse all the feature clouds. Assume ordered (descending)
        fuse_feature_map = voxel_wise_submap_fusion(
            data_batch["feature_clouds"], self.voxel_size
        )

        # Fuse the collision and feature cloud, collision map is stored as labe_prob!
        fused_map = fuse_collision_map_and_cloud(
            collision_map=data_batch["collision_map"],
            feature_map=fuse_feature_map,
            voxel_size=self.voxel_size,
            use_unlabeled_data=self.use_unlabeled_data,
        )

        return fused_map

    def return_valid_key(self, key):
        if type(key) is np.array or type(key) is np.ndarray:
            return pose_to_key(key[:3], self.key_distance)
        elif type(key) is tuple and len(key) == 3:
            return key
        else:
            msg = f"Invalid key received: {key}"
            raise TypeError(msg)

    # TODO(fab): Is this where one should filter out the data?f
    def get_patch_data_set_copy(self, label_prob_threshold=0.5):
        """Generate a valid data cube from all the nodes."""
        # data_set = [ copy.deepcopy(node) for node in self.data_fields]
        data_set = []
        for key, node in self.data_fields.items():
            # non-filled node
            if not "features" in node:
                continue

            labels = np.full((node["features"].shape[0], 1), 0, dtype=int)
            mask_te = node["label_prob"] > label_prob_threshold
            labels[mask_te] = 1.0

            # The unobserved type. Need to mark them properly to avoid issues down the line
            mask_inv = node["label_prob"] < 0.0
            labels[mask_inv] = -1
            node["label"] = labels

            data_set.append(copy.deepcopy(node))

        return data_set

    def __contains__(self, key: np.array):
        return self.return_valid_key(key) in self.nodes

    def __getitem__(self, key):
        return self.data_fields.get(self.return_valid_key(key), None)

    def __setitem__(self, key, data):
        valid_key = self.return_valid_key(key)
        self.data_fields[valid_key] = data

    def __len__(self):
        return len(self.data_fields)

    def keys(self):
        return self.data_fields.keys()

    def values(self):
        return self.data_fields.values()

    def items(self):
        return self.data_fields.items()

    def __iter__(self):
        return iter(self.data_fields)


def fuse_collision_map_and_cloud(
    collision_map, feature_map, voxel_size, use_unlabeled_data
):
    """Generates an label_prob and label_observation from a collision map. It intersects the collision map, generated by robot traversing the terrain with
        a smaller feature map (ohm-map).
        Label_prob: Nx1 contains the label probability  [0,1] for each voxel. It is -1 if not defined
        Label_obs: Nx1 contains if the voxel has been visited/intersected with the robot bounding box (LfE sense); 0 for not, 1 for visited

    Args:
        collision_map (np.array):   Robot collision map (Kx4) containing the label [x,y,z, prob]
        feature_map (np.array):     Voxelised feature map (NxD+3) where a sample contains [x,y,z, f_0,... fn]
        voxel_size (float):         Voxel sized used for the map

    Returns:
        tuple: Returns the label_prob, and label_obs which are Nx1
    """
    feature_map_bounds = get_voxel_cloud_bounds(feature_map, voxel_size=voxel_size)
    filtered_collision_map = collision_map[
        mask_for_points_within_bounds(cloud=collision_map, bounds=feature_map_bounds)
    ]

    # Find the same ids for the feature map and the filtered collision map
    id_pairs_all = np.array(
        find_point_pairs(
            source_points=filtered_collision_map[:, :3],
            target_points=feature_map[:, :3],
            voxel_size=voxel_size,
        )
    )

    if id_pairs_all.shape[0] < 1:
        # msg = f"No valid id pairs found"
        # raise ValueError(msg)
        return None

    ids_collision_map = id_pairs_all[:, 0]
    ids_featrue_map = id_pairs_all[:, 1]

    # Collision map is fused into the feature map
    label_prob = np.full((feature_map.shape[0], 1), -1, dtype=float)

    label_prob[ids_featrue_map] = (
        filtered_collision_map[ids_collision_map][:, 3]
    ).reshape(-1, 1)

    label_obs = np.full((feature_map.shape[0], 1), 0)
    label_obs[ids_featrue_map] = 1

    # Check if dimensions align
    assert feature_map.shape[0] == label_obs.shape[0]
    assert feature_map.shape[0] == label_prob.shape[0]

    if use_unlabeled_data:
        return {
            "coords": feature_map[:, :3],
            "features": feature_map[:, 3:],
            "label_prob": label_prob,
            "label_obs": label_obs,
        }
    # Only use lsbaled data:
    return {
        "coords": feature_map[:, :3][ids_featrue_map],
        "features": feature_map[:, 3:][ids_featrue_map],
        "label_prob": label_prob[ids_featrue_map],
        "label_obs": label_obs[ids_featrue_map],
    }


def voxel_wise_submap_fusion(
    feature_clouds,
    voxel_size,
) -> np.ndarray:
    """Fuses a list of voxels scans into a single fused cloud. Assumes the scans to be ordered in decreasing importance.
        Each voxel is identified by global voxel_ids (@func points_to_batch_ids), and is the latest observation.

    Args:
        feature_clouds (list): List of voxel clouds with decreasing importance (temporarly decresing)
        voxel_size (float): Voxel leaf size [m]

    Returns:
        np.ndarray: Fused voxel cloud
    """
    voxel_ids = set()
    fuse_map = np.empty(shape=(0, feature_clouds[0].shape[1]))
    for cloud in feature_clouds:

        # Generate the voxel centres and the mask of the unique voxels
        cloud_voxel_ids = points_to_voxel_ids(cloud[:, :3], voxel_size)
        mask_unique_voxels = np.array(
            [
                False if tuple(voxel_id) in voxel_ids else True
                for voxel_id in cloud_voxel_ids
            ]
        )

        # Generate the fused map and update the voxel centres
        fuse_map = np.concatenate(
            [
                fuse_map,
                cloud[mask_unique_voxels],
            ]
        )
        new_voxel_ids = {
            tuple(voxel_id) for voxel_id in cloud_voxel_ids[mask_unique_voxels]
        }
        voxel_ids.update(new_voxel_ids)

    return fuse_map


def generate_valid_poses(
    candidate_poses: dict, cloud_coord: np.array, voxel_size: float, patch_width: float
):
    """_summary_

    Args:
        candidate_poses (dict): _description_
        cloud_coord (np.array): Coordinates of the cloud
        voxel_size (float): Size of voxel leaf [m]
        patch_width (float): Total width of the patch/ cell [m]

    Returns:
        valid_poses: Tuple of Key, pose that fit the given constraints
    """

    if cloud_coord.shape[1] != 3:
        msg = "The cloud_coord has an invalid shape"
        raise RuntimeError(msg)

    # Separate poses into keys, poses itself
    poses_keys = np.array([key for key in candidate_poses.keys()])
    poses = np.array([pose_i for key, pose_i in candidate_poses.items()])

    # Note: We do not compress the z axis as we tend to crop tight in z anyway
    padding = [
        -patch_width / 2.0,
        -patch_width / 2.0,
        0.0,
        -patch_width / 2.0,
        -patch_width / 2.0,
        0.0,
    ]
    bounds = get_voxel_cloud_bounds(cloud_coord, voxel_size=voxel_size, padding=padding)

    mask_poses_completely_inside = mask_for_points_within_bounds(
        cloud=poses[:, :3], bounds=bounds
    )
    pose_in_bounds = {
        tuple(key): pose
        for key, pose in zip(
            poses_keys[mask_poses_completely_inside],
            poses[mask_poses_completely_inside],
        )
    }

    return pose_in_bounds, {}
