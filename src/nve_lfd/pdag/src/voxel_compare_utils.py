import numpy as np
from sklearn.neighbors import KDTree


def associate_voxel_ids_by_refrence_points(
    cloud_querry: np.ndarray,  
    cloud_refrence, 
    point_refrence, 
    voxel_size, padding=True
    
):
    """ Associates voxel ids between two refrence 
    Args:
        cloud_queery: ndarray where [[x,y,z], ...[xm,ym,zm]] m number of points
        point_querries: points in list
        cloud_refrence: refrence cloud we wish the cloud to associate to
    Return:
        id: [id_q, id_ref]
    """

    id_pairs = []
    
    # Get the ids from the orginal clouds
    # Note: Is this the best way? 
    kd_tree_querry = KDTree(cloud_querry)
    dd_s, ids_q = kd_tree_querry.query(point_refrence, k=1)

    kd_tree_ref = KDTree(cloud_refrence)
    dd_ref, ids_ref = kd_tree_ref.query(point_refrence, k=1)

    for i, id in enumerate(ids_q):

        # If there is no pair
        if not ids_q[i][0] or not ids_ref[i][0]:
            if padding:
                id_pairs.append((-1, -1))
            continue
        
        assert dd_s[i][0] == 0, "We are getting the ids for the refrence point and refrence cloud. This should be 0!"

        # Check if the pair matches
        if not point_in_voxel(cloud_querry[ids_q[i][0]], cloud_refrence[ids_q[i][0]], voxel_size
        ):
            if padding:
                id_pairs.append((-1, -1))
            continue

        # The pairs seem to match
        id_pairs.append((ids_q[i][0], ids_ref[i][0]))
    return id_pairs


def associate_points_to_voxel_cloud(voxel_cloud,  voxel_size, querry_points):
    """ For a given point set (querry_points) we aim to find the associate point from the voxel_cloud
        Assumption: (1) Starts at (0,0,0) with no shift. (2) Empty voxel are expected to retrun the centre point 

    Args:
        voxel_cloud:    np.array    Contains the position of the voxel cloud
        voxel_size:     float       Voxel leaf size, unifrom for all sides
        querry_points:              List of points we wish to find the associated voxel_point for

    Return:
        ids : id of point in cloud, -1 if no point found
    """

    voxel_ids = []
    
    # Get the ids from the orginal clouds
    voxel_kd_tree = KDTree(voxel_cloud)
    dd, ids_vc = voxel_kd_tree.query(querry_points, k=1)

    for i, id_i in enumerate(ids_vc):

        # If there is no pair
        if not id_i[0]:
            voxel_ids.append(-1)
            continue

        if dd[i] == 0:
            voxel_ids.append(ids_vc[i][0])
            continue
        
        # Check if the pair matches
        if not point_in_voxel(querry_points[i], voxel_cloud[ids_vc[i][0]], voxel_size):
            voxel_ids.append(-1)
            continue

        # The pairs seem to match
        voxel_ids.append(ids_vc[i][0])
    return voxel_ids



def point_in_voxel(querry_point, voxel_point, voxel_size):
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

    # Get the bounds we need to compare
    x_low = voxel_size * float(voxel_point[0] // voxel_size)
    x_high = x_low + voxel_size

    y_low = voxel_size * float(voxel_point[1] // voxel_size)
    y_high = y_low + voxel_size

    z_low = voxel_size * float(voxel_point[2] // voxel_size)
    z_high = z_low + voxel_size

    return (
        x_low <= querry_point[0] < x_high
        and y_low <= querry_point[1] < y_high
        and z_low <= querry_point[2] < z_high
    )
