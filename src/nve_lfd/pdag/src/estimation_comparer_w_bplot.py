## This script compares the estimation results of a TE classifier with a ground truth
##
## Inputs:
## 1. Ground truth file
## 2. Estimator files [Esimtaor and features?]
import pandas as pd
import numpy as np
import joblib as jb
from io_helper import read_ply_as_df
from evaluate_differential_enttorpy import (
    compute_bernoulli_stats,
    compute_beta_statistics,
)
from voxel_compare_utils import (
    associate_voxel_ids_by_refrence_points,
    associate_points_to_voxel_cloud,
)
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import open3d as o3d

# Define the debug values
# TODO: Extend to deal with time series of a scene:

SCENE_FILES = [
    "/data/debug_comparison_class/ground_truth_ply/2021_12_14_00_14_53Z_hl.ply",
]

GT_SCENE_FILE = (
    "/data/debug_comparison_class/ground_truth_ply/2021_12_14_00_14_53Z_hl.ply"
)
ESTIMATOR_FILE = "/data/debug_comparison_class/models/svc_ff_adj_best_fit.joblib"
FEATURE_SET_FILE = "/data/debug_comparison_class/ff_adj_features.joblib"

VOXEL_SIZE = 0.4
MAX_DIST = np.sqrt(3.0 * 0.1 * 0.1)


def load_all():
    scenes = [read_ply_as_df(scene_file) for scene_file in SCENE_FILES]
    gt_scene = read_ply_as_df(GT_SCENE_FILE)
    estimator = jb.load(ESTIMATOR_FILE)
    feature_sets = jb.load(FEATURE_SET_FILE)

    # HACK to combat the issue with the cloud compare saving convetion adding scalar
    scalar_set = ["scalar_" + f for f in feature_sets]

    return scenes, gt_scene, estimator, scalar_set


def visualize_cloud(pos, labels):
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pos)
    colours = colourise_for_labels(labels=labels)
    pcl.colors = o3d.utility.Vector3dVector(colours)
    o3d.visualization.draw_geometries([pcl])
    b = 1


COLOUR_LABELS = {0: [1.0, 0, 0], 1: [0, 1.0, 0]}


def colourise_for_labels(labels):
    colours = [COLOUR_LABELS[label] for label in labels]
    return colours


def main():
    (
        scenes,
        gt_scene,
        estimator,
        feature_set,
    ) = load_all()

    # Setup the gt comparision
    cloud_gt = gt_scene[["x", "y", "z"]].to_numpy()
    kd_tree_gt = KDTree(gt_scene[["x", "y", "z"]].to_numpy())

    # Random draw n samples
    # roi_points = draw_random_roi_points(cloud_gt, 20000)
    # roi_points = cloud_gt[np.random.randint(len(cloud_gt), size=50000)]
    roi_points = cloud_gt

    dd_gt, ids_gt = kd_tree_gt.query(roi_points, k=1)
    max_dist = MAX_DIST
    gt_labels = gt_scene[["scalar_label"]].to_numpy()

    # DEBUG
    l_gt_labels = gt_labels.tolist()
    gt_label_list = [l[0] for l in l_gt_labels]
    # visualize_cloud(cloud_gt, gt_label_list)
    # Main loop over scenes
    # L = [ L_s0, L_s2, ..., Lsn] labels for each scene for roi where L_si(l_hat, l_gt)
    #  = [ L_s0, L_s2, ..., Lsn] labels for each scene for roi where L_si(l_hat, l_gt)
    L = []

    # F = [F_so, F_s1, ..., F_sn] where n is the number of scenes
    # F_si = [[mean_count, H_occ, H_perm]_x0, ...., [ ]_xk] where k is the number of points in roi
    # and the values are taken from the scenes i
    F = []
    roi_id_pairs = []

    for scene in scenes:

        X_scene = scene[feature_set].to_numpy()
        y_label = estimator.predict(X_scene)

        # Find the ids of the scenes and labels between the gr and estimated scene
        cloud_xyz_s = scene[["x", "y", "z"]].to_numpy()

        # DEBUG VISUALIZATION
        # visualize_cloud(cloud_xyz_s, y_label)

        roi_id_pairs.append(
            associate_voxel_ids_by_refrence_points(
                cloud_querry=cloud_xyz_s,
                cloud_refrence=cloud_gt,
                point_refrence=roi_points,
                voxel_size=VOXEL_SIZE,
            )
        )

        # Note: Invalid ids are not considered (-1, -1)
        L.append(l_from_ids(y_label, gt_label_list, roi_id_pairs[-1]))
        F.append(f_from_ids(scene, roi_id_pairs[-1]))
    L_tot = []
    F_tot = []
    L_tot = [L_tot + L_i for L_i in L][0]
    F_tot = [F_tot + F_i for F_i in F][0]
    
    f_2 = [f[1] for f in F_tot]
    generate_classification_report_per_feature(f_2, feature_name="Perm occupancy", label_pairs=L_tot, number_of_bins= 100)


    # Split the data in bins and calculate the classification metrics
    # Generate the data from the plot
    # d1, d2, d3 = split_data_into_classification_resutls(L, F)

    # # Plot all
    # fig, axs = plt.subplots(4, 1)
    # fig.suptitle(f"Classification resuts")

    # # 2: x: Time, y: Permeability Entropy
    # axs[0].set_title("Class Results for ray_count")
    # axs[0].boxplot(d1[0:4])
    # axs[0].set_xlabel("Class (TP:0, TN:1 FP:2, FN:3, U:4")
    # axs[0].set_ylabel(" Ray Count")
    # axs[1].set_title("Class Results for Occ Entropy")
    # axs[1].boxplot(d2[0:4])
    # axs[1].set_xlabel("Class (TP:0, TN:1 FP:2, FN:3, U:4")
    # axs[1].set_ylabel(" Occ Entropy")
    # axs[2].set_title("Class Results for Perm Entropy")
    # axs[2].set_title("Class Results for Occ Entropy")
    # axs[2].boxplot(d3[0:4])
    # axs[2].set_xlabel("Class (TP:0, TN:1 FP:2, FN:3, U:4")
    # axs[2].set_ylabel("Perm Entropy")
    # bins = np.linspace(0, 100, 100)
    # axs[3].hist(d1[0], bins, alpha=0.5, label="TP")
    # axs[3].hist(d1[1], bins, alpha=0.5, label="TN")
    # axs[3].hist(d1[2], bins, alpha=0.5, label="FP")
    # axs[3].hist(d1[3], bins, alpha=0.5, label="FN")
    # axs[3].legend(loc="upper right")
    # axs[3].set_xlabel("Class (TP:0, TN:1 FP:2, FN:3, U:4")
    # axs[3].set_ylabel("Perm Entropy")

    # fig.set_size_inches(12, 12)
    # plt.tight_layout()
    # plt.show()

    # print("Done")


def generate_classification_report_per_feature(
    feature_vector: np.array, feature_name:str, label_pairs, number_of_bins: int
):
    """ Generates a classification report per feature or values and the associated labels

    Args: 
        feature_vecor:  Single feature vector of size (k,1)
        label_pairs:      
    
    """

    assert len(label_pairs) > 1
    assert len(label_pairs) == len(feature_vector)

    min_value = np.min(feature_vector)
    max_value = np.max(feature_vector)

    if max_value > 500:
        max_value = 500

    step_size = (max_value - min_value) / float(number_of_bins)

    thresholds = np.arange(0, max_value, step=step_size)

    # thresholds  = [x*x for x in range(100)]

    labels_pred = np.array([label_pair[0] for label_pair in label_pairs])
    labels_gt = np.array([label_pair[1] for label_pair in label_pairs])

    # Scores for 
    f1_score_data = []
    precission_score_data = []
    recall_score_data = []

    f1_score_data_bin = []
    precission_score_data_bin = []
    recall_score_data_bin = []

    for treshold in thresholds:
        
        logical_idx =  feature_vector < treshold
        logical_idx_bin = np.logical_and(feature_vector > treshold-step_size, feature_vector < treshold)
        y_pred_th = labels_pred[logical_idx]
        y_gt_th = labels_gt[logical_idx]
        if len(y_pred_th) < 1:
            f1_score_data.append(0)
            precission_score_data.append(0)
            recall_score_data.append(0)
        else:
            f1_score_data.append(f1_score(y_true=y_gt_th, y_pred=y_pred_th))
            precission_score_data.append(precision_score(y_true=y_gt_th, y_pred=y_pred_th))
            recall_score_data.append(recall_score(y_true=y_gt_th, y_pred=y_pred_th))

        # The bined version
        y_pred_th_bin = labels_pred[logical_idx_bin]
        y_gt_th_bin = labels_gt[logical_idx_bin]
        if len(labels_pred) < 1:
            f1_score_data_bin.append(0)
            precission_score_data_bin.append(0)
            recall_score_data.append(0)
        else:
            f1_score_data_bin.append(f1_score(y_true=y_gt_th_bin, y_pred=y_pred_th_bin))
            precission_score_data_bin.append(precision_score(y_true=y_gt_th_bin, y_pred=y_pred_th_bin))
            recall_score_data_bin.append(recall_score(y_true=y_gt_th_bin, y_pred=y_pred_th_bin))

    # Plot: X: Entropy
    # Plot: Y: F1-scroe
    fig, axs = plt.subplots(3, 2)
    fig.suptitle(f"Classification resuts")

    # 2: x: Time, y: Permeability Entropy
    axs[0,0].set_title(f"F1-score  {feature_name}")
    axs[0,0].plot(thresholds,f1_score_data )
    axs[0,0].set_xlabel(f"{feature_name}")
    axs[0,0].set_ylabel("F1-score")

    axs[0,1].set_title(f"F1-score binned {feature_name}")
    axs[0,1].plot(thresholds,f1_score_data_bin )
    axs[0,1].set_xlabel(f"{feature_name}")
    axs[0,1].set_ylabel("F1-score")


    axs[1,0].set_title(f"Precission vs {feature_name}")
    axs[1,0].plot(thresholds,precission_score_data )
    axs[1,0].set_xlabel(f"{feature_name}")
    axs[1,0].set_ylabel("Precission")

    axs[1,1].set_title(f"Precission bin vs {feature_name}")
    axs[1,1].plot(thresholds,precission_score_data_bin )
    axs[1,1].set_xlabel(f"{feature_name}")
    axs[1,1].set_ylabel("Precission")
    
    axs[2,0].set_title(f"Recall vs {feature_name}")
    axs[2,0].plot(thresholds,recall_score_data )
    axs[2,0].set_xlabel(f"{feature_name}")
    axs[2,0].set_ylabel("Recall")

    axs[2,1].set_title(f"Recall bin vs {feature_name}")
    axs[2,1].plot(thresholds,recall_score_data_bin )
    axs[2,1].set_xlabel(f"{feature_name}")
    axs[2,1].set_ylabel("Recall")

    plt.show()
    plt.waitforbuttonpress()


# 0: TP, 1: TN, 2: FP, 3:FN, 4: Uknonw/Invalid
TE_COMPARE_DICT = {(-1, -1): 4, (0, 0): 1, (1, 0): 2, (0, 1): 3, (1, 1): 0}


def split_data_into_classification_resutls(L, F):
    data_ray_count = [[], [], [], [], []]
    data_occ = [[], [], [], [], []]
    data_perm = [[], [], [], [], []]
    for i in range(len(L)):
        L_i = L[i]
        F_i = F[i]
        assert len(L_i) == len(F_i)
        for j in range(len(L_i)):
            class_result = TE_COMPARE_DICT[L_i[j]]
            data_ray_count[class_result].append(F_i[j][0])
            data_occ[class_result].append(F_i[j][1])
            data_perm[class_result].append(F_i[j][2])

    return data_ray_count, data_occ, data_perm


def draw_random_roi_points(cloud: np.array, sample_number):
    random_ids = np.random.randint(2, cloud.shape[0], size=sample_number)

    random_points = cloud[random_ids]
    return random_points


def f_from_ids(scene: pd.DataFrame, id_pairs: list):
    """ """
    F_tn = []
    for i, id_pair in enumerate(id_pairs):
        if id_pair[0] < 0:
            continue
        # total_count = scene["scalar_mean_count"][ids[0]]
        total_count = scene["scalar_count"][id_pair[0]]
        # odds = scene["scalar_occupancy_log_probability"][ids[0]]
        odds = scene["scalar_occ_f"][id_pair[0]]
        p_occ = np.exp(float(odds)) / float(1.0 + np.exp(float(odds)))
        occ_mean, occ_variance, occ_entropy = compute_bernoulli_stats(p_occ)

        perm = scene["scalar_perm"][id_pair[0]]
        mean, variance, entropy = compute_bernoulli_stats(perm)
        F_tn.append([total_count, occ_entropy, entropy])

    return F_tn


def l_from_ids(
    labels_dataset_1: np.array, labels_dataset_2: np.array, id_pairs: list
) -> list:
    """Goes trough the layers and associates the labels"""
    Ln = []
    for id in id_pairs:

        if id[0] < 0 or id[1] < 0:
            # Ln.append((-1, -1))
            continue

        Ln.append((labels_dataset_1[id[0]], labels_dataset_2[id[1]]))
    return Ln


def compare_labels(label_pred, label_gt):
    return TE_COMPARE_DICT[(label_pred, label_gt)]


if __name__ == "__main__":
    main()
