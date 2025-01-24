# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz


import numpy as np
from dataclasses import dataclass

from scnn_tm.utils.point_cloud_utils import (
    mask_for_points_within_bounds,
    get_cloud_bounds,
)


@dataclass
class MapPatch:
    """Map patch represents a local 3D patch for a voxalised map.

    Members:
        map_id (int): The map-id the patch belongs to
        bounds (np.array):  Bounds of the local map [m] [x_min, y_min, z_min, x_max, y_max ,z_max]
        global_patch_id (int): For a set of MapPatchBounds the unique id for all patches generated from multiple data sets

    """
    map_id: int
    bounds: np.array
    global_patch_id: int


def generate_map_patches(
    data_set_coords: list,
    voxel_size: float,
    nvoxel_leaf: int,
    min_number_of_samples,
):
    """Generates map patches for all the maps of and stores as a single list

    Args:
        data_set_coords (list): List of coordinates of a voxel data set, each list entry is treated as unique voxel-map
        voxel_size (float): Size of the voxel's in the map
        nvoxel_leaf (int): Number of voxel's in each dimension (x,y,z)
        min_number_of_samples (int, optional): Minimum number of samples required per-patch to be considered valid.

    Returns:
        _type_: _description_
    """
    map_patches = []
    for i, sceen_coords in enumerate(data_set_coords):
        map_patches.extend(
            generate_evenly_spaced_patches(
                sceen_coords,
                map_id=i,
                voxel_size=voxel_size,
                nvoxel_leaf=nvoxel_leaf,
                min_number_of_samples=min_number_of_samples,
            )
        )

    for i, patch_i in enumerate(map_patches):
        patch_i.global_patch_id = i

    return map_patches


def generate_evenly_spaced_patches(
    data_coords: np.ndarray,
    map_id: int,
    voxel_size: float,
    nvoxel_leaf: float,
    min_number_of_samples: int = 20,
) -> list:
    """_summary_

    Args:
        data_coords (np.ndarray):   Coordinates of the voxels points in x,y,z in [m]
        map_id (int):               Map id of the map (purely used to assign)
        voxel_size (float):         Voxels size of the map
        nvoxel_leaf (float):        Patch size in voxels, for each axis
        min_number_of_samples (int, optional): Minimum number of samples/voxels required within patch to be valid

    Returns:
        list[MapPatches]: _description_
    """

    data = []

    cloud_bounds = get_cloud_bounds(data_coords)
    voxel_step_per_map = voxel_size * nvoxel_leaf

    for x_i in np.arange(cloud_bounds[0], cloud_bounds[3], voxel_step_per_map):
        for y_i in np.arange(cloud_bounds[1], cloud_bounds[4], voxel_step_per_map):
            for z_i in np.arange(cloud_bounds[2], cloud_bounds[5], voxel_step_per_map):
                # Check if we want to add the patch to the bound
                patch_i = MapPatch(
                    map_id=map_id,
                    bounds=np.array(
                        [
                            x_i,
                            y_i,
                            z_i,
                            x_i + float(voxel_step_per_map),
                            y_i + float(voxel_step_per_map),
                            z_i + float(voxel_step_per_map),
                        ]
                    ),
                    global_patch_id=0,
                )

                if not is_valid_map_patch(data_coords, patch_i, min_number_of_samples):
                    continue

                data.append(patch_i)

    return data


def mask_for_points_within_patch(data_coords: np.array, map_patch: MapPatch):
    """Helper function to return the mask for patch"""
    return mask_for_points_within_bounds(cloud=data_coords, bounds=map_patch.bounds)


def is_valid_map_patch(data_coords, map_patch, nsample_threshold):
    "For a list of coordinates check if it is a valid patch by requiring n samples within the bounds"
    return (
        np.count_nonzero(mask_for_points_within_patch(data_coords, map_patch))
        > nsample_threshold
    )


def has_identical_patches(patches_i, patches_k):
    """Checks if patch i and k contain idential patches"""
    for patch_i in patches_i:
        for patch_k in patches_k:
            if (
                patch_i.map_id == patch_i.map_id
                and (patch_i.bounds == patch_k.bounds).all()
            ):
                return True

    return False


def mask_for_sampled_patch(
    data_coords: np.array,
    patch: MapPatch,
    map_bounds: list,
    voxel_size: float,
    nvoxel_leaf: int,
) -> np.array:
    """Samples a random patch within the field of data coords and returns it if valid.

    Args:
        data_coords (np.array): _description_
        patch (MapPatch): _description_
        map_bounds (list): _description_
        voxel_size (float): _description_
        nvoxel_leaf (int): _description_

    Returns:
        np.array: _description_
    """

    total_voxel_step = voxel_size * float(nvoxel_leaf)
    padding = total_voxel_step / 2.0

    valid_patch = False
    while not valid_patch:
        # Generate the X,Y,Z coordinates
        x_i = np.random.uniform(
            low=map_bounds[0] + padding, high=map_bounds[3] - padding
        )
        y_i = np.random.uniform(
            low=map_bounds[1] + padding, high=map_bounds[4] - padding
        )
        z_i = np.random.uniform(
            low=map_bounds[2] + padding, high=map_bounds[5] - padding
        )
        # we want to keep the ground a much as possible

        patch.bounds = np.array(
            [
                x_i,
                y_i,
                z_i,
                x_i + float(total_voxel_step),
                y_i + float(total_voxel_step),
                z_i + float(total_voxel_step),
            ]
        )

        valid_patch = is_valid_map_patch(data_coords=data_coords, map_patch=patch)

    return mask_for_points_within_bounds(data_coords=data_coords, patch=patch)


def convert_patches_to_data_set(raw_data_sets, patches):
    """Generate a data set instance from the patches with the fields
        dict{
            "coords": np.array()
            "labels: ...
        }

    Args:
        raw_data_sets list[dict]: list of data sets represented as dicts
        patches (_type_): list of map patches
        fields (_type_): fields, with key value pars defining features, etc

    Returns:
        list(dict): Returns
    """

    data_set = []

    for patch in patches:
        raw_data_set = raw_data_sets[patch.map_id]

        mask_for_cloud = mask_for_points_within_patch(
            data_coords=raw_data_set["coords"], map_patch=patch
        )

        # We dont need the map id and name anymore, remove it
        data_set_i = {
            key: raw_data_set[key][mask_for_cloud]
            for key in raw_data_set.keys()
            if key not in ["id", "name"]
        }

        data_set.append(data_set_i)

    # Check that we have all the patches
    assert len(data_set) == len(patches)

    return data_set
