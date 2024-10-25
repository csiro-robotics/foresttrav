import numpy as np
import pandas as pd


def get_cloud_bounds(cloud: np.ndarray, padding: float = 0.0)->list:
    """Get the absolute bounds of a point cloud in its current frame as an axis aligned bounding box"""
    x_min = np.min(cloud[:, 0] - padding)
    y_min = np.min(cloud[:, 1] - padding)
    z_min = np.min(cloud[:, 2] - padding)
    x_max = np.max(cloud[:, 0] + padding)
    y_max = np.max(cloud[:, 1] + padding)
    z_max = np.max(cloud[:, 2] + padding)

    return [x_min, y_min, z_min, x_max, y_max, z_max]


def mask_for_points_within_bounds(cloud: np.ndarray, bounds: list):
    """Returns the a logical masks for all  points of a cloud that are within the @p bounds map_bounds"""
    mask_x = np.logical_and(cloud[:, 0] >= bounds[0], cloud[:, 0] < bounds[3])
    mask_y = np.logical_and(cloud[:, 1] >= bounds[1], cloud[:, 1] < bounds[4])
    mask_z = np.logical_and(cloud[:, 2] >= bounds[2], cloud[:, 2] < bounds[5])

    return np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

def mask_for_points_within_column(cloud: np.ndarray, voxel_centre_point: np.array, voxel_size:float):
    ''' Returns a logical array that masks the points of a cloud have the same x,y voxel centre'''
    x_min = voxel_centre_point[0] - voxel_size/2.0
    x_max = voxel_centre_point[0] + voxel_size/2.0
    y_min = voxel_centre_point[1] - voxel_size/2.0
    y_max = voxel_centre_point[1] + voxel_size/2.0
    
    mask_x = np.logical_and(cloud[:, 0] >= x_min, cloud[:, 0] < x_max)
    mask_y = np.logical_and(cloud[:, 1] >=y_min, cloud[:, 1] < y_max)
    return  np.logical_and(mask_y, mask_x)

def convert_ohm_cloud_to_feature_cloud(ohm_cloud: np.array, org_feature_set, desired_feature_set) -> np.array:
    """ Receives the feature set of an ohm cloud and generates the full ftm feature set without adjacency

    """

    # Generate an empty data frame where to store the data
    data_df = pd.DataFrame(ohm_cloud,  columns=org_feature_set)
   
    # Assign all the original features
    add_permeability_to_dataframe(data_df)

    add_occ_prob_to_dataframe(data_df)

    add_eigenvalue_features_to_dataframe(data_df)

    return data_df[["x", "y", "z"]].to_numpy(), data_df[desired_feature_set].to_numpy()


def add_permeability_to_dataframe(data_frame: pd.DataFrame) -> None:
    " Adds the permeability to the data frame"
    data_frame.loc[:, "permeability"] = data_frame["miss_count"].values / \
        (data_frame["miss_count"].values + data_frame["hit_count"].values)


def add_occ_prob_to_dataframe(data_frame: pd.DataFrame):
    data_frame.loc[:, "occupancy_prob"] = np.exp(
        data_frame["occupancy_log_probability"].values) / (1.0 + np.exp(data_frame["occupancy_log_probability"].values))

# Not used comonly


def add_occ_to_dataframe(self, data_frame):

    occ_mask = data_frame.loc[:, "occupancy_prob"] > 0.5
    data_frame["occupancy"] = 0.0
    occ_arr = data_frame["occupancy"].to_numpy()
    occ_arr[occ_mask] = 1.0
    data_frame.loc[:, "occupancy"] = occ_arr


def add_eigenvalue_features_to_dataframe(data_frame: pd.DataFrame) -> None:
    cov_index = ["covariance_xx_sqrt",	"covariance_xy_sqrt",	"covariance_xz_sqrt",
                 "covariance_yy_sqrt",	"covariance_yz_sqrt",	"covariance_zz_sqrt"]

    # cov_index = ["cov_xx","cov_xy", "cov_xz", "cov_yy", "cov_yz" , "cov_zz" ]
    feature_set_view = data_frame[cov_index].values

    ev_data = [caluclate_ev_features(feature_set_view[i]) for i in range(feature_set_view.shape[0])]

    ev_features = ["ndt_rho", "theta", "ev_lin", "ev_plan", "ev_sph"]
    data_frame[ev_features] = 0.0
    data_frame.loc[:, ev_features] = np.array(ev_data)


def caluclate_ev_features(triag_lower):

    H = np.array([[triag_lower[0], triag_lower[1], triag_lower[2]],
                  [triag_lower[1], triag_lower[3], triag_lower[4]],
                  [triag_lower[2], triag_lower[4], triag_lower[5]]])
    v, w = np.linalg.eigh(H)

    ndt_rho = v[0]
    theta = gravity_aligned_angle(w[:, 0])

    return np.array([v[0], theta, v[2] - v[1], v[1] - v[0], v[2]])


def correct_colour_rgbf(data_frame):

    # Check if we are in 0 - 255 RGB format
    if np.count_nonzero(data_frame["red"] > 10) > 1:

        data_frame.loc[:, ["red"]] = data_frame.loc[:, ["red"]] / 255.0
        data_frame.loc[:, ["green"]] = data_frame.loc[:, ["green"]] / 255.0
        data_frame.loc[:, ["blue"]] = data_frame.loc[:, ["blue"]] / 255.0

    return data_frame


def gravity_aligned_angle(ev_0):
    eg = np.array([0.0, 0.0, 1.0])
    return np.abs(np.arccos(np.dot(np.abs(ev_0), np.array([0.0, 0.0, 1.0]))))
