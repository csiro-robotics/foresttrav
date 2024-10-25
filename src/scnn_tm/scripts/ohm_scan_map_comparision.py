import sys, os

sys.path.append(os.path.abspath("/nve_ml_ws/src/nve_eval"))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

import open3d as o3d
import torch
import torch.nn as nn

from joblib import dump

from scnn_tm.utils import (
    visualize_classification_difference,
    associate_points_to_voxel_cloud,
    generate_feature_set_from_key,
    overlay_two_clouds,
    )
from scnn_tm.models.ScnnFtmEstimators import (
    ScnnFtmEnsemble,
    ScnnFtmEstimator,
    parse_cv_models_from_dir,
)

# from nve_classical_ml_train import *
from joblib import load

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    matthews_corrcoef,
)
import matplotlib.pyplot as plt

''' This script generates ohm-scan to map comparsions data sets.
    - Note: The evaluation is done in the "plot_scan_map_comparison.py"

    

'''

# This will be set with map_dir_index
SCENE_PAIRS_DIR = [
    (
        "/data/processed/ohm_scans/2021_12_14_00_14_53Z/ohm_scans_v01",
        "/data/processed/feature_sets/lfe_hl_v0.1/2021_12_14_00_14_53Z",
    ),
    (
        "/data/processed/ohm_scans/2021_12_14_00_14_53Z/ohm_scans_v02",
        "/data/processed/feature_sets/lfe_hl_v0.2/2021_12_14_00_14_53Z",
    ),
    (
        "/data/processed/ohm_scans/2022_02_14_23_47_33Z/ohm_scans_v01",
        "/data/processed/feature_sets/lfe_hl_v0.1/2022_02_14_23_47_33Z",
    ),
    (
        "/data/processed/ohm_scans/2022_02_14_23_47_33Z/ohm_scans_v02",
        "/data/processed/feature_sets/lfe_hl_v0.2/2022_02_14_23_47_33Z",
    ),
]

USE_SCNN = True
UNET_MODEL_FILE = "/data/forest_trav_paper/ts_models/cv_test_train_lfe_hl_10_UNet5LMCD_ohm_mr/model_config.yaml"
SCNN_ENS_DIR = "/data/forest_trav_paper/ts_models/cv_test_train_lfe_hl_10_UNet5LMCD_ohm_mr"

# NDT_ADJ
# SVM_MODEL_FILE = "/data/processed/experiments/ablation_lfe_hl/lfe_hl/exp_2023_02_20_08_37/model/svm_ndt_adjestimator.joblib"
# SVM_FEATURE_FILE = "/data/processed/experiments/ablation_lfe_hl/lfe_hl/exp_2023_02_20_08_37/model/svm_ndt_adjfeatures.joblib"
# SVM_SCALER_FILE = "/data/processed/experiments/ablation_lfe_hl/lfe_hl/exp_2023_02_20_08_37/model/svm_ndt_adjscaler.joblib"

# FTM
SVM_MODEL_FILE = "/data/processed/experiments/paper_ablation_2023_04_01_new_colour/lfe_hl_v0.1/exp_2023_04_02_22_52/model/svm_ftmestimator_cv_0.joblib"
SVM_FEATURE_FILE = "/data/processed/experiments/paper_ablation_2023_04_01_new_colour/lfe_hl_v0.1/exp_2023_04_02_22_52/model/svm_ftmfeatures_cv_0.joblib"
SVM_SCALER_FILE = "/data/processed/experiments/paper_ablation_2023_04_01_new_colour/lfe_hl_v0.1/exp_2023_04_02_22_52/model/svm_ftmscaler_cv_0.joblib"


parser = argparse.ArgumentParser()
parser.add_argument("--voxel_size", default=0.1, type=float)
parser.add_argument("--local_map_bounds", default=[-15, -15, -5, 15, 15, 5], type=list)
parser.add_argument("--data_dirs", default=SCENE_PAIRS_DIR, type=list)
parser.add_argument("--visualize", default=False, type=bool)
parser.add_argument("--mode_type", default="scnn_ens", type=str)
parser.add_argument("--map_dir_index", default=2, type=int)

parser.add_argument("--save_data", default=True, type=bool)
parser.add_argument("--out_dir", default="/data/forest_trav_paper/temporal_analysis", type=str)

# TODO: Parameter for every nth visualisation
# TODO: (optional) Show a non-blocking visualisation
# TODO: (optional) Optimize the speed of the code (currently very slow)


def align_with_icp(
    source_cloud: o3d.geometry.PointCloud(),
    target_cloud: o3d.geometry.PointCloud(),
    threshold,
    T_target_source=np.eye((4)),
) -> np.ndarray:
    """Returns the transformation matrix between a source and target cloud"""
    
    source_cloud.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
    source_cloud.orient_normals_to_align_with_direction()
    target_cloud.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
    target_cloud.orient_normals_to_align_with_direction()
    
    mu, sigma = 0, 0.07  # mean and standard deviation
    p2pl = o3d.pipelines.registration.TransformationEstimationPointToPlane(o3d.pipelines.registration.TukeyLoss(k=sigma))
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        threshold,
        T_target_source,
        p2pl,
    )

    return reg_p2p.transformation


# TODO: This needs to move into a class "FtmDataSetProcessor"
def load_data_set(ohm_scans_dir: Path, data_ground_truth_dir: Path):
    # Load the ohm scans
    files = [file for file in ohm_scans_dir.iterdir()]
    files = sorted(files)
    file_times = [float(file.stem[0:-6]) for file in files[-2:-1]]
    ohm_scans_dfs = [pd.read_csv(file, sep=",") for file in files[-2:-1]]

    # Load the refrence map
    ohm_map_df1 = pd.read_csv(data_ground_truth_dir / "semantic_cloud_class_1.csv")
    ohm_map_df1["label"] = 0
    ohm_map_df2 = pd.read_csv(data_ground_truth_dir / "semantic_cloud_class_2.csv")
    ohm_map_df2["label"] = 1
    target_map_df = pd.concat([ohm_map_df2, ohm_map_df1])

    return ohm_scans_dfs, target_map_df, file_times


def compare_scan_map(source_scan, target_map, voxel_size, local_map_bounds):
    """Comparse and aligns a scan and a map returns id pairs

    Assumptions:    In the current form this method assumes that source and target
                    are arelady well aligned and we are fixing small errors.

    Args:

        source_can: (ndarray)   Scan which should be aligned

    Return:
        id_pairs:   map_id_pair    [id_source, id_target]
    """

    # Crop to bounding box before ICP, helps with convergence
    # axis_aligned_bb = o3d.geometry.AxisAlignedBoundingBox()
    # axis_aligned_bb.min_bound = np.array(local_map_bounds[0:3])
    # axis_aligned_bb.max_bound = np.array(local_map_bounds[3:6])

    source_scan_pcd = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(source_scan)
    )
    target_map_pcd = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(target_map)
    )


    # Align ohm_scans to
    T_target_source = align_with_icp(
        source_scan_pcd, target_map_pcd, threshold=0.34
    )
    print(T_target_source)
    ohm_scan_transformed = source_scan_pcd.transform(T_target_source)

    # overlay_two_clouds(ohm_scan_transformed.points, target_map_pcd.points)
    # Associate points of ohm_scan_tranformed to the source_map
    id_pairs = associate_points_to_voxel_cloud(
        source_cloud=np.asarray(ohm_scan_transformed.points),
        target_cloud=np.asarray(target_map_pcd.points),
        voxel_size=voxel_size,
    )

    return id_pairs, np.array(ohm_scan_transformed.points)


def main(args):

    # Load estimator
    model = 0
    scaler = 0
    feature_set = 0

    # Load the models based on the model_config
    if args.mode_type == "scnn":
        model_name = Path(UNET_MODEL_FILE).parent.stem
        estimator = ScnnFtmEstimator(model_file=UNET_MODEL_FILE, cv_nbr=0)
        # feature_set = generate_feature_set_from_key("ftm")
        feature_set = estimator.feature_set
    elif args.mode_type == "scnn_ens":
        model_name = Path(SCNN_ENS_DIR).stem
        ensemble_model_files = parse_cv_models_from_dir(SCNN_ENS_DIR, 10)
        feature_set = generate_feature_set_from_key("ohm_mr")
        estimator = ScnnFtmEnsemble(
            model_config_files=ensemble_model_files, device="cuda", input_feature_set = feature_set
        )
    else:
        model_name = Path(SVM_MODEL_FILE).stem
        model = load(SVM_MODEL_FILE)
        scaler = load(SVM_SCALER_FILE)
        feature_set = load(SVM_FEATURE_FILE)

    # Setup a data set pair

    data_set_pairs = args.data_dirs[args.map_dir_index]
    source_scans_df, target_map_df, file_times = load_data_set(
        Path(data_set_pairs[0]), Path(data_set_pairs[1])
    )

    # TODO: Abstract the feature list
    feature_coords = ["x", "y", "z"]
    label_key = ["label"]

    X_coords_target = target_map_df[feature_coords].to_numpy()
    y_target = target_map_df[label_key].to_numpy()
    L = []
    F = []

    for i in range(len(source_scans_df)):
        X_coords = source_scans_df[i][feature_coords].to_numpy()
        X_features = source_scans_df[i][feature_set].to_numpy()
        X_feature_mean_count = source_scans_df[i][["mean_count"]].to_numpy()

        y_pred = []
        if args.mode_type == "scnn":

            y_pred, y_std = estimator.predict(
                X_coords=X_coords, X_features=X_features, voxel_size=args.voxel_size
            )

            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0

        elif args.mode_type == "scnn_ens":
            y_pred, y_std = estimator.predict(
                X_coords=X_coords, X_features=X_features, voxel_size=args.voxel_size
            )

            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
        else:
            y_pred = model.predict(scaler.transform(X_features))

            # Compute the source labels from model
        id_pairs = compare_scan_map(
            X_coords, X_coords_target, args.voxel_size, args.local_map_bounds
        )

        if not id_pairs:
            print(f"Skipped ohm_scan {i} since no pairs could be found")
            return

        id_pairs = np.array(id_pairs)
        L.append(
            np.column_stack(
                (
                    y_pred[id_pairs[:, 0]].astype(int).squeeze(),
                    y_target[id_pairs[:, 1]].astype(int).squeeze(),
                )
            )
        )
        F.append(X_feature_mean_count[id_pairs[:, 0]])

        if args.visualize:
            visualize_classification_difference(
                X_coords[id_pairs[:, 0]],
                y_pred[id_pairs[:, 0]].astype(int).squeeze(),
                y_target[id_pairs[:, 1]].astype(int).squeeze(),
            )

        # Pickle the suckers"
    if args.save_data:
        data = {"L": L, "F": F, "times": np.array(file_times)}
        filename = Path(args.out_dir) / f"{model_name}.joblib"
        dump(value=data, filename=filename)

    # Generate figures and save data

    if args.visualize:
        fig = generate_classification_plots_ohm_scans_map(np.array(file_times), L, F)

    print("Finished ohm_scan_comparsion")

def temporal_classification_metrics(times, L):

    # for each metric we use [total, class_0, class_1]_metric
    f1_score_times = []
    pre_score_times = []
    rec_score_times = []
    mcc_score_times = []
    for i, label_pair in enumerate(L):

        # F1-score
        score = classification_report(
            y_true=label_pair[:, 1], y_pred=label_pair[:, 0], output_dict=True
        )

        f1_score_times.append(
            np.array(
                [
                    score["macro avg"]["f1-score"],
                    score["0"]["f1-score"],
                    score["1"]["f1-score"],
                ]
            )
        )
        pre_score_times.append(
            np.array(
                [
                    score["macro avg"]["precision"],
                    score["0"]["precision"],
                    score["1"]["precision"],
                ]
            )
        )
        rec_score_times.append(
            np.array(
                [
                    score["macro avg"]["recall"],
                    score["0"]["recall"],
                    score["1"]["recall"],
                ]
            )
        )

    assert len(times) == len(f1_score_times)

    return (
        np.vstack(f1_score_times),
        np.vstack(pre_score_times),
        np.vstack(rec_score_times),
    )  # , precision_score_time, recall_score_time


def feature_classification_metric(L: list, F: list):
    """Classification metrics that are based on features"""

    L_arr_stacked = np.vstack(L)
    F_arr_stacked = np.vstack(F)

    return generate_binned_classification_scores_by_feature(
        labels_source=L_arr_stacked[:, 0],
        labels_target=L_arr_stacked[:, 1],
        feature_vector=F_arr_stacked.squeeze(),
        n_bin=100,
        bin_min=0,
        bin_max=500,
    )


def generate_binned_classification_scores_by_feature(
    labels_source, labels_target, feature_vector, n_bin, bin_min, bin_max
):
    """Generates classification metrics based on bins"""

    step_size = (bin_max - bin_min) / float(n_bin)

    thresholds = np.arange(bin_min, bin_max, step=step_size)

    # Binned data
    f1_score_data_bin = []
    precission_score_data_bin = []
    recall_score_data_bin = []

    for treshold in thresholds:

        # logical_idx =  feature_vector < treshold
        logical_idx_bin = np.logical_and(
            feature_vector > treshold - step_size, feature_vector < treshold
        )
        y_pred_th_bin = labels_source[logical_idx_bin]
        y_gt_th_bin = labels_target[logical_idx_bin]
        score = classification_report(
            y_true=y_gt_th_bin, y_pred=y_pred_th_bin, output_dict=True
        )
        if not "0" in score or not "1" in score:
            f1_score_data_bin.append(np.array([0, 0, 0]))
            precission_score_data_bin.append(np.array([0, 0, 0]))
            recall_score_data_bin.append(np.array([0, 0, 0]))
        else:
            f1_score_data_bin.append(
                np.array(
                    [
                        score["macro avg"]["f1-score"],
                        score["0"]["f1-score"],
                        score["1"]["f1-score"],
                    ]
                )
            )
            precission_score_data_bin.append(
                np.array(
                    [
                        score["macro avg"]["precision"],
                        score["0"]["precision"],
                        score["1"]["precision"],
                    ]
                )
            )
            recall_score_data_bin.append(
                np.array(
                    [
                        score["macro avg"]["recall"],
                        score["0"]["recall"],
                        score["1"]["recall"],
                    ]
                )
            )

    return (
        np.vstack(f1_score_data_bin),
        np.vstack(precission_score_data_bin),
        np.vstack(recall_score_data_bin),
        thresholds,
    )


def generate_classification_plots_ohm_scans_map(times, L, F, model_name=None):
    fig, axs = plt.subplots(3, 2)
    fig.suptitle("Classification report ")

    f1_score_times, prec_score_time, recall_score = temporal_classification_metrics(
        times, L
    )
    times = (times - times[0]) / 1.0e9

    # Time element of evaluations:
    # Plot f1, precision and recall over time
    axs[0, 0].set_title("F1-score over time")
    axs[0, 0].plot(times, f1_score_times[:, 0], label="f1_tot")
    axs[0, 0].plot(times, f1_score_times[:, 1], label="f1_c0")
    axs[0, 0].plot(times, f1_score_times[:, 2], label="f1_c1")
    axs[0, 0].set_xlabel("time[s]")
    axs[0, 0].set_ylabel("F1-score")
    axs[0, 0].legend(loc="upper right")

    prec_tot, prec_c0, prec_c1 = zip(*prec_score_time)
    axs[1, 0].set_title("Precision over time")
    axs[1, 0].plot(times, prec_score_time[:, 0], label="prec_tot")
    axs[1, 0].plot(times, prec_score_time[:, 1], label="prec_c0")
    axs[1, 0].plot(times, prec_score_time[:, 2], label="prec_c1")
    axs[1, 0].set_xlabel("time[s]")
    axs[1, 0].set_ylabel("Precision")
    axs[1, 0].legend(loc="upper right")

    axs[2, 0].set_title("Recall over time")
    axs[2, 0].plot(times, recall_score[:, 0], label="rec_tot")
    axs[2, 0].plot(times, recall_score[:, 1], label="rec_c0")
    axs[2, 0].plot(times, recall_score[:, 2], label="rec_c1")
    axs[2, 0].set_xlabel("time[s]")
    axs[2, 0].set_ylabel("Recall")
    axs[2, 0].legend(loc="upper right")

    (
        f1_score_data_bin,
        precission_score_data_bin,
        recall_score_data_bin,
        thresholds,
    ) = feature_classification_metric(L, F)

    axs[0, 1].set_title("F1-score over Observations")
    axs[0, 1].plot(thresholds, f1_score_data_bin[:, 0], label="f1_tot_bin")
    axs[0, 1].plot(thresholds, f1_score_data_bin[:, 1], label="f1_c0_bin")
    axs[0, 1].plot(thresholds, f1_score_data_bin[:, 2], label="f1_c1_bin")
    axs[0, 1].set_xlabel(" Observations[ ]")
    axs[0, 1].set_ylabel("F1-score")
    axs[0, 1].legend(loc="upper right")

    axs[1, 1].set_title("Precision over Observations")
    axs[1, 1].plot(thresholds, precission_score_data_bin[:, 0], label="prec_tot_bin")
    axs[1, 1].plot(thresholds, precission_score_data_bin[:, 1], label="prec_c0_bin")
    axs[1, 1].plot(thresholds, precission_score_data_bin[:, 2], label="prec_c1_bin")
    axs[1, 1].set_xlabel("Observations [ ]")
    axs[1, 1].set_ylabel("Precision")
    axs[1, 1].legend(loc="upper right")

    axs[2, 1].set_title("Recall over Observations")
    axs[2, 1].plot(thresholds, recall_score_data_bin[:, 0], label="rec_tot_bin")
    axs[2, 1].plot(thresholds, recall_score_data_bin[:, 1], label="rec_c0_bin")
    axs[2, 1].plot(thresholds, recall_score_data_bin[:, 2], label="rec_c1")
    axs[2, 1].set_xlabel("Observations")
    axs[2, 1].set_ylabel("Recall")
    axs[2, 1].legend(loc="upper right")

    plt.show()
    plt.waitforbuttonpress()
    b = 1
    return fig


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
