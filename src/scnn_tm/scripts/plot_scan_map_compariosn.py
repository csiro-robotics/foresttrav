from joblib import load
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import matthews_corrcoef
from scnn_tm.utils.point_cloud_utils import mask_point_in_bounds, get_cloud_bounds

# We are aiming to evaluate a temporal component of the method by ploting
# MCC / time  for a train set (How much does temporal phenomenons influenc the model)
# MCC score over obserations

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({"font.weight": "bold"})
plt.rcParams.update({"axes.labelweight":"bold"})


FILE_DICT = {
    "NDT-TM": "svm_ndt_adjestimator.joblib",
    "FTM": "svm_ftmestimator_cv_0.joblib",
    "ForestTrav": "test_train_lfe_hl_10_UNet5LMCD_ftm.joblib",
    # "SCNN NDT": "test_train_lfe_hl_10_UNet5LMCD_ndt.joblib",
    # "scnn_occ": "test_train_lfe_hl_10_UNet5LMCD_occ.joblib",
    # "SCNN LIDAR BASIC": "test_train_lfe_hl_10_UNet5LMCD_occ_perm_int.joblib",
    # "scnn_occ_per_int_ev_sr": "test_train_lfe_hl_10_UNet5LMCD_occ_perm_int_ev_sr.joblib",
    # "SCNN OCC RGB": "test_train_lfe_hl_10_UNet5LMCD_occ_rgb.joblib",
}

TEST_SET_DIR = "/data/scnn_models_23_04_01/temporal_analysis/test_set_scene"

# TRAIN_SET_DIR = ("/data/scnn_models_23_04_01/temporal_analysis/2021_12_14_00_14_53Z_scene")
# TEST_ORG_SCAN_DIR = "/data/processed/ohm_scans/2022_02_14_23_47_33Z/ohm_scans_v01"

def load_scan_arr(ohm_scans_dir: Path):
    # Load the ohm scans
    files = [file for file in ohm_scans_dir.iterdir()]
    files = sorted(files)
    file_times = [float(file.stem[0:-6]) for file in files]
    ohm_scans_arr = [pd.read_csv(file, sep=",")[["x","y","z"]].to_numpy() for file in files]

    return ohm_scans_arr, file_times


def mcc_score_vs_time(times, L):

    return np.array(
        [
            matthews_corrcoef(y_true=label_pair[:, 1], y_pred=label_pair[:, 0])
            for label_pair in L
        ]
    )


def generate_binned_classification_scores_by_feature(
    labels_source,
    labels_target,
    feature_vector,
    n_bin,
    bin_min,
    bin_max,
):
    """Generates classification metrics based on bins"""

    step_size = (bin_max - bin_min) / float(n_bin)

    thresholds = np.arange(bin_min, bin_max, step=step_size)

    # Binned data
    mcc_score_data_bin = []

    for treshold in thresholds:

        logical_idx = feature_vector < treshold
        # logical_idx_bin = np.logical_and(
        #     feature_vector > treshold - step_size, feature_vector < treshold
        # )
        y_pred_th_bin = labels_source[logical_idx]
        y_true_th_bin = labels_target[logical_idx]
        mcc_score_data_bin.append(
            matthews_corrcoef(y_pred=y_pred_th_bin, y_true=y_true_th_bin)
        )

    return np.vstack(mcc_score_data_bin), thresholds


def main():

    # Load the test set files
    fig = plt.figure()
    axes1 = fig.add_subplot(111)
    axes2 = axes1.twinx() 

    # Load the train set files
    for key, value in FILE_DICT.items():
        temp_dict = load(Path(TEST_SET_DIR) / value)

        temp_dict["mcc_score"] = mcc_score_vs_time(temp_dict["times"], temp_dict["L"])
        times = (temp_dict["times"] - temp_dict["times"][0]) / 1.0e9
        # axes1plot(times, temp_dict["mcc_score"], label=key)

        linestl= '-'
        if  "SVM" in key:
            linestl = '-.'
        axes1.plot(times, temp_dict["mcc_score"],  linestyle=linestl, linewidth=2,label=key)

    # for key, value in FILE_DICT.items():
    #     temp_dict = load(Path(TRAIN_SET_DIR) / value)
    #     times = (temp_dict["times"] - temp_dict["times"][0]) / 1.0e9
    #     temp_dict["mcc_score"] = mcc_score_vs_time(temp_dict["times"], temp_dict["L"])
        
    #     linestl= ':'
    #     if  "SVM" in key:
    #         linestl = '-.'
    #     axes1plot(times, temp_dict["mcc_score"],  linestyle=linestl, label=key)

    # #### Caluclation for obersavtion vs performance!
    # for key, value in FILE_DICT.items():
    #     temp_dict = load(Path(TEST_SET_DIR) / value)
        
    #     #
    #     L_arr_stacked = np.vstack(temp_dict["L"])
    #     F_arr_stacked = np.vstack(temp_dict["F"])
        
    #     (
    #         mcc_scores_binned,
    #         observations,
    #     ) = generate_binned_classification_scores_by_feature(
    #         labels_source = L_arr_stacked[:,0],
    #         labels_target=L_arr_stacked[:, 1],
    #         feature_vector=F_arr_stacked.squeeze(),
    #         n_bin=50,
    #         bin_min=0.0,
    #         bin_max=1000,
    #     )

    #     axs[2].plot(observations[1::], mcc_scores_binned[1::], label=key)

    #### Calculation of temporal observation vs performance
    class_0 = pd.read_csv("/data/processed/feature_sets/lfe_hl_v0.1/2022_02_14_23_47_33Z/semantic_cloud_class_1.csv")
    total_number_of_negative_labels =  class_0.shape[0]
    
    class_1 = pd.read_csv("/data/processed/feature_sets/lfe_hl_v0.1/2022_02_14_23_47_33Z/semantic_cloud_class_2.csv")
    total_number_of_positive_labels = class_1.shape[0]
    
    total_number_of_labels = total_number_of_negative_labels + total_number_of_positive_labels
    
    # Load any file, does not matter
    temp_dict = load(Path(TEST_SET_DIR) / FILE_DICT["SCNN FTM"])
    times = (temp_dict["times"] - temp_dict["times"][0]) / 1.0e9
        
    L =  temp_dict["L"]
    total_coverage_percentage = [ L_i[:,1].shape[0]/total_number_of_labels for L_i in  L]
    axes2.fill_between(times, total_coverage_percentage,  color= "b", alpha= 0.2)

    # Calculate the scan percentage
    target_df = pd.concat([class_0, class_0])
    target_cloud_arr = target_df[["x", "y", "z"]].to_numpy()
    target_map_bounds = get_cloud_bounds(target_cloud_arr, 0.0)
    
    # if True:
    #     scann_arr, _ = load_scan_arr(Path(TEST_ORG_SCAN_DIR))
    #     n_scans = [scan[is_cloud_in_bounds(cloud = scan, bounds = target_map_bounds)].shape[0] for scan in scann_arr]
        
    #     scan_percentage = []
    #     for L_i, scan_i in zip(L, n_scans):
    #         scan_percentage.append(L_i.shape[0] / float(scan_i))
        # for scan in scans:
        #     mask_of_scan = is_cloud_in_bounds(cloud = scan, bounds = target_map_bounds)
        #     scan_in_bounds = 
        #     n_scans.append(scan[scan].shape[0])

        # axes2.hist(scan_percentage, label="Ohm-scan percentage")
        
    # non_traversable_coverage_percentage = []
    # traversable_coverage_percentage = []
    # for L_i in L:
    #     positive_count_l_i = L_i[ L_i[:,1]  == 1.0, 1].shape[0]
    #     negative_count_l_i = L_i[:,1].shape[0] - positive_count_l_i
        
    #     traversable_coverage_percentage.append(positive_count_l_i /total_number_of_positive_labels )
    #     non_traversable_coverage_percentage.append(negative_count_l_i / total_number_of_negative_labels)

        
    # axs[3].plot(times, traversable_coverage_percentage, label="positive percentage")
    # axs[3].plot(times, non_traversable_coverage_percentage, label="negative percentage")
    
    
    axes1.set_title("Estimators Sensitivity to Information Quality")
    axes1.set_xlabel("time [s]")
    axes1.set_ylabel("MCC")
    axes1.set_ylim([0, 0.8])
    axes1.legend(loc="upper left")

    # axes2.set_title("Temporal performance in an unknown environment")
    # axes2.set_xlabel("time [s]")
    # axes2.set_ylabel("MCC")
    # axes2.legend(loc="upper right")
    
    # axs[2].set_title("Performance vs observations in a unknown environment")
    # axs[2].set_xlabel(" Number of observations")
    # axs[2].set_ylabel("MCC")
    # axs[2].legend(loc="upper right")
    
    # axes2.set_title("Temporal point association")
    # axes2.set_xlabel("time [s]")
    axes2.set_ylabel("Association percentage")
    axes2.set_ylim([0, 0.8])
    # axes2.legend(loc="upper right")

    plt.show()
    plt.waitforbuttonpress()



if __name__ == "__main__":
    main()
