import numpy as np
import open3d as o3d

import matplotlib
from matplotlib import pyplot as plt


LABEL_TO_COLOUR_MAP = {
    0: (1.0, 0.0, 0.0),
    1: (0.0, 1.0, 0.0),
}

CLASIIFICATION_TO_COLOUR_MAP = {
    "TP": (0.0, 1.0, 0.0),
    "FP": (1.0, 0.0, 0.0),
    "TN": (0.0, 0.0, 1.0),
    "FN": (1.0, 0.5, 0.0),
    "NONE": (0.0, 0.0, 0.0),
}

CLASIIFICATION_TO_CR_LABEL = {
    "TP": 1,
    "TN": 2,
    "FP": 3,
    "FN": 4,
    "NONE": 0,
}

def classification_comparsion(pred, target_label):

    if (pred == target_label) and target_label == 1:
        return "TP"
    elif (pred == target_label) and target_label == 0:
        return "TN"

    if (pred != target_label) and target_label == 1:
        return "FN"
    elif (pred != target_label) and target_label == 0:
        return "FP"

    return "NONE"


def visualize_te_classification(cloud_coords: np.array, y_pred: np.array):
    pred_pcd = o3d.geometry.PointCloud() 
    pred_pcd.points = o3d.utility.Vector3dVector(np.vstack(cloud_coords))
    
    colours_mask = y_pred <1
    colours = np.zeros((colours_mask.shape[0], 3))
    colours[colours_mask] = np.array([1.0, 0, 0])
    colours[colours_mask] = np.array([0.0, 1.0, 0])

    pred_pcd.colors = o3d.utility.Vector3dVector(colours)
    print("Visualizing pcd")
    o3d.visualization.draw_geometries([pred_pcd])


def visualize_classification_difference(cloud_coords:np.ndarray , source_labels:list, target_labels:list)->None:
    # Convert to pointcloud
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(np.vstack(cloud_coords))
    colours = []

    for i in range(len(source_labels)):
        colours.append(
            CLASIIFICATION_TO_COLOUR_MAP[
                classification_comparsion(source_labels[i], target_labels[i])
            ]
        )

    pred_pcd.colors = o3d.utility.Vector3dVector(colours)
    o3d.visualization.draw_geometries([pred_pcd])

def visualize_probability(cloud_coords: np.ndarray, cloud_prob: np.ndarray):
    cmap = plt.cm.get_cmap('viridis')
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.vstack(cloud_coords))
    
    colours = []
    for prob in cloud_prob:
        colours.append(cmap(prob)[0:3])
    cloud.colors = o3d.utility.Vector3dVector(colours)
    o3d.visualization.draw_geometries([cloud])
    


def overlay_two_clouds(source_cloud, target_cloud):
    source_cloud_pcd = o3d.geometry.PointCloud()
    source_cloud_pcd.points = o3d.utility.Vector3dVector(source_cloud)
    source_cloud_pcd.paint_uniform_color([1, 0.0, 0])
    
    tatrget_cloud_pcd = o3d.geometry.PointCloud()
    tatrget_cloud_pcd.points = o3d.utility.Vector3dVector(target_cloud)
    tatrget_cloud_pcd.paint_uniform_color([0, 0.651, 0.929])
    
    o3d.visualization.draw_geometries([source_cloud_pcd, tatrget_cloud_pcd])
    
    
def visualize_cloud(cloud):
    source_cloud_pcd = o3d.geometry.PointCloud()
    source_cloud_pcd.points = o3d.utility.Vector3dVector(cloud)
    source_cloud_pcd.paint_uniform_color([1, 0.0, 0])
    o3d.visualization.draw_geometries([cloud])
    