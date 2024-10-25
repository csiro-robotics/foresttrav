from plyfile import PlyData, PlyElement
import numpy as np


def writeToPlyFile(path, cloud_array, types_tup):
    """Write a point cloud in nd array to a given path
    input: path             Absolute file path
    input: cloud_array      np.ndArray of point cloud features
    input: types_tup        list of tuples ('key', type) defined for plyfile
    Example for wildcat cloud:
    dtype=[('time','f8'),('x','f8'), ('y','f8'), ('z','f8'), ('intensity','u4'), ('returnNum','u4')]
    Note: PlyData needs a list of tuples of point attributes (why the hell that is nobody knows)
    """
    # cloud_tupple_list =
    cloud_tupple_list = np.asarray(list(map(tuple, cloud_array)), dtype=types_tup)
    el = PlyElement.describe(
        cloud_tupple_list,
        "vertex",
    )
    PlyData([el], text=False).write(path)
