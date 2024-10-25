# Author: Fabio Ruetz

# Point clouds are represented as nd-arrays
import numpy as np


def box_filter(cloud_arr, bounds, s_idx=0):
    """Filters a cloud based on bounds
    Args:
        cloud   nd-array: [...., x, y, z, ....]
        bounds  nd-array; [x_min, x_max, y_min, ..., z_max]
        s_idx   int id where the x, elemnts starts
        new_cloud_arr   filtered cloud nd-array
    Note: Modifies the cloud_arr
    """
    # can only do one comparions per slice... numpy

    # cloud_arr = cloud_arr[
    mask_0 = bounds[0] <= cloud_arr[:, s_idx + 0]
    mask_1 = bounds[1] >= cloud_arr[:, s_idx + 0]
    mask_2 = bounds[2] <= cloud_arr[:, s_idx + 1]
    mask_3 = bounds[3] >= cloud_arr[:, s_idx + 1]
    mask_4 = bounds[4] <= cloud_arr[:, s_idx + 2]
    mask_5 = bounds[5] >= cloud_arr[:, s_idx + 2]

    bool_mask = mask_0 & mask_1 & mask_2 & mask_3 & mask_4 & mask_5
    cloud_arr = cloud_arr[bool_mask]

    return cloud_arr
