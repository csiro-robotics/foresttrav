from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
from io_helper import read_ply_file
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import beta, bernoulli
from sklearn.neighbors import KDTree

# TODO:
# - Test if the ply loading works

# Goal of this script is to compare the change of entropy, variance or other metric for a few voxels
# over a time frame
#
# Features to comprare:
# - Occupancy and permeability are assumed to be a beta distribution
#
# arg:
#   voxels_of_interest: list of pts
#

DATA_ROOT_DIR = "/data/debug/online_ss_tests/exp_5_no_reset"
ROI_CLOUD_FILE =  "roi_cloud.ply"
VOXEL_SIZE = 0.4

ROI_CLOUDS = [[4.72218943,	-0.75659823,	0.70498532],
[5.00957966,	-0.65063536,	0.62717497],
[5.02639294,	-0.26666668,	0.72375369],
[4.25259066,	0.2635386,	    0.6044966],
[5.07448673,	0.15327469,	    0.49540567],
[4.56070375,	0.74799609,	    0.59198433],
[4.35698938,	0.89149559,	    0.50713587],
[5.09481907,	-1.44946241,	1.44516134],
[5.14252186,	-1.12844574,	1.35366571],
[4.23734093,	0.37927663,	    1.35092866],
[4.26549387,	0.66940373, 	1.31495607],
]

class ExperimentDataFiles:

    def __init__(self, data_root_file, roi_cloud_name):
        self.exp_root_dir = Path(data_root_file)
        self.time_cloud_dir = self.exp_root_dir / ("time_clouds")
        if self.time_cloud_dir.exists():
            self.time_cloud_dir.mkdir(parents=True, exist_ok=True)

        self.roi_cloud_file = Path(data_root_file) / "roi_clouds" / roi_cloud_name

        # TODO: Make this more explicit or based on time stamp?
        timestamp = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        
        results_dir = Path(data_root_file) / "results" 
        if not results_dir.exists():
            results_dir.mkdir(parents=True, exist_ok=True)
        self.results_pdf_file = Path(data_root_file) / "results" / f"{timestamp}_results.pdf"


class FeaturesStatistics:
    """Stores the statistical feature of a voxel as list
    Note:
      - Assumptions are made that all the entries are ordered temporally
      - ! dataclass's are singletons...
    """
    def __init__(self):
        self.time_stamp =  []
        self.ray_end_count = []

        # Permeability Features
        self.perm_entropy = []
        self.perm_mean = []
        self.perm_variance = []

        # Occupancy Features
        self.occ_prob = []
        self.occ_entropy = []
        self.occ_mean = []
        self.occ_variance = []


def compute_beta_statistics(_alpha, _beta) -> list:
    mean, variance = beta.stats(_alpha, _beta, moments="mv")
    entropy = beta.entropy(_alpha, _beta)

    return [mean, variance, entropy]

def compute_bernoulli_stats(p:float)->list:
    mean, variance = bernoulli.stats(p, moments="mv")
    entropy = bernoulli.entropy(p)
    return [mean, variance, entropy]

def main():

    # Get time clouds 
    experiment_paths = ExperimentDataFiles(data_root_file=DATA_ROOT_DIR, roi_cloud_name = ROI_CLOUD_FILE)


    files = [file for file in experiment_paths.time_cloud_dir.iterdir() ]
    files = sorted(files)

    file_times = [file.stem[0:-6] for file in files]
    times = [(float(file_t) - float(file_times[0]))*1e-9 for file_t in file_times]


    data_sets = []
    for file in files:
        if not Path(file).is_file():
            continue
        data_sets.append(pd.read_csv(file))

    voxel_size = VOXEL_SIZE
   
    # Read the ply file and only extract the points positions
    # voxels_of_interrest = read_ply_file(file_name=experiment_paths.roi_cloud_file)
    voxels_of_interrest = ROI_CLOUDS

    max_radius = np.sqrt(3 * voxel_size * voxel_size)

    # Generate the KdTree to find the clostest neighboor
    feature_stats = [ FeaturesStatistics() for x in voxels_of_interrest]
    ts_idx = 0
    for scene in data_sets:

        scene_pos_array = scene[["x", "y", "z"]].to_numpy()
        kdtree = KDTree(scene_pos_array)

        # Associate the voxel of interrest to the ids in the scene
        dist_nn, all_nn_indices = kdtree.query(voxels_of_interrest, k=1)
        for i, nn_ids in enumerate(all_nn_indices):
            
            # Skip since we have no point found or to far away
            if len(nn_ids) < 1 or dist_nn[i][0] > max_radius:
                
                continue
            
            id_in_scene = nn_ids[0]    
            # Check if the p_roi[i] lies within the bounds of the point given by i
            if not point_in_voxel(voxels_of_interrest[i], scene_pos_array[id_in_scene], voxel_size):
                continue

            # Number of observations at the end of a voxel
            feature_stats[i].ray_end_count.append(scene["mean_count"][id_in_scene])
            feature_stats[i].time_stamp.append(times[ts_idx])

            # Occpupancy
            odds = scene["occupancy_log_probability"][id_in_scene]
            p_occ = np.exp(float(odds)) / float(1.0 + np.exp(float(odds)))
            feature_stats[i].occ_prob.append(p_occ)

            # Beta Prior
            alpha_occ = p_occ * feature_stats[i].ray_end_count[-1]
            beta_occ = int(feature_stats[i].ray_end_count[-1]) - alpha_occ


            # Permeability [miss_count, opportunities]
            miss_count = scene["miss_count"][id_in_scene]
            total_count = scene["opportunities"][id_in_scene]

                # Beta Prior
            alpha_perm = scene["miss_count"][id_in_scene]
            beta_perm = scene["opportunities"][id_in_scene] - alpha_perm
            
            # occ_mean, occ_variance, occ_entropy = compute_bernoulli_stats(feature_stats[i].occ_prob[-1]) 

            # mean, variance, entropy = compute_bernoulli_stats(float(miss_count)/(float(total_count)))
            
            occ_mean, occ_variance, occ_entropy = compute_beta_statistics(alpha_occ, beta_occ) 
            mean, variance, entropy = compute_beta_statistics(alpha_perm, beta_perm) 

            # Add the values to the feature states
            feature_stats[i].occ_variance.append(occ_variance)
            feature_stats[i].occ_mean.append(occ_mean)
            feature_stats[i].occ_entropy.append(occ_entropy)
            feature_stats[i].perm_entropy.append(entropy)
            feature_stats[i].perm_mean.append(mean)
            feature_stats[i].perm_variance.append(variance)

            
        #Incerment the timestamp
        ts_idx +=1

    print("Start writting data to file")

    with PdfPages(experiment_paths.results_pdf_file) as export_pdf:
        for i, pos in enumerate(voxels_of_interrest):
            
            if not feature_stats[i].ray_end_count:
                continue
            
            # Assure it is not a singleton
            assert len(feature_stats[i].perm_entropy) <= len(data_sets)

            fig_perm = plot_voxel_of_interrest(feature_stats[i],i)

            fig_occ = plot_occ_voxel_statistics(feature_stats[i],i)
            
            if True:
                export_pdf.savefig(fig_perm)
                export_pdf.savefig(fig_occ)
            plt.close(fig_perm)
            plt.close(fig_occ)



def plot_occ_voxel_statistics(feature_stats: FeaturesStatistics, voxel_id, final_feature_stats = FeaturesStatistics()):
    a = 1
    fig, axs = plt.subplots(4, 2)

    fig.suptitle(f'Voxel [{voxel_id}]: Occ Features')
    time_axis = feature_stats.time_stamp

     # 2: x: Time, y: Permeability Entropy
    axs[0,0].set_title("Occ Entropy vs time")
    axs[0,0].plot( time_axis,  feature_stats.occ_entropy)
    axs[0,0].set_xlabel("Time")
    axs[0,0].set_ylabel("OCC H")
    # if final_feature_stats.occ_entropy:
    #     axs[1,0].plot( time_axis, [final_feature_stats.occ_entropy[0] for x in time_axis])

    # 3: x: Total number of rays, y: Permeability Entropy
    axs[0,1].set_title("OCC Entropy vs #-rays")
    axs[0,1].scatter( feature_stats.ray_end_count,  feature_stats.occ_entropy)
    axs[0,1].set_xlabel("#-rays")
    axs[0,1].set_ylabel("OCC H")
    # if final_feature_stats.occ_entropy:
    #     axs[1,1].plot( time_axis, [final_feature_stats.occ_entropy[0] for x in time_axis])

    # 4: x: Time, y: Permeability mean
    axs[1,0].set_title("OCC Mean vs #-rays")
    axs[1,0].plot( time_axis,  feature_stats.occ_mean)
    axs[1,0].set_xlabel("time")
    axs[1,0].set_ylabel("OCC mean")
    # if final_feature_stats.occ_mean:
    #     axs[2,0].plot( time_axis, [final_feature_stats.occ_mean[0] for x in time_axis])

    # 4: x: Ray count, y: Permeability mean
    axs[1,1].set_title("OCC Mean vs #-rays")
    axs[1,1].scatter( feature_stats.ray_end_count, feature_stats.occ_mean)
    axs[1,1].set_xlabel("#-rays")
    axs[1,1].set_ylabel("OCC mean")
    # if final_feature_stats.occ_mean:
    #     axs[2,1].plot( time_axis, [final_feature_stats.occ_mean[0] for x in time_axis])

    # 5 OCC vs time
    axs[2,0].set_title("OCC Var vs time")
    axs[2,0].plot( time_axis, feature_stats.occ_variance)
    axs[2,0].set_xlabel("time")
    axs[2,0].set_ylabel("OCC var")
    # if final_feature_stats.occ_variance:
    #     axs[2,0].plot( time_axis, [final_feature_stats.occ_variance[0] for x in time_axis])
    # 6 Perme Var vs #-rays
    axs[2,1].set_title("OCC Var vs #-rays")
    axs[2,1].scatter( feature_stats.ray_end_count, feature_stats.occ_variance)
    axs[2,1].set_xlabel("#-rays")
    axs[2,1].set_ylabel("OCC var")
    # if final_feature_stats.occ_variance:
    #     axs[2,1].plot( time_axis, [final_feature_stats.occ_variance[0] for x in time_axis])

    axs[3,0].set_title("OCC Prob vs time")
    axs[3,0].plot( time_axis, feature_stats.occ_prob)
    axs[3,0].set_xlabel("time")
    axs[3,0].set_ylabel("OCC var")
    # if final_feature_stats.occ_prob:
    #     axs[2,0].plot( time_axis, [final_feature_stats.occ_prob[0] for x in time_axis])

    axs[3,1].set_title("OCC Prob vs #-rays")
    axs[3,1].scatter( feature_stats.ray_end_count, feature_stats.occ_prob)
    axs[3,1].set_xlabel("#-rays")
    axs[3,1].set_ylabel("OCC var")
    # if final_feature_stats.occ_prob:
    #     axs[3,1].plot( time_axis, [final_feature_stats.occ_prob[0] for x in time_axis])
        
    # 5: x: Time, y: Permeability variance 
    fig.set_size_inches(24, 24)
    plt.tight_layout()
    return fig




def plot_voxel_of_interrest(feature_stats: FeaturesStatistics, voxel_id, final_feature_stats = FeaturesStatistics()):
    """"
    
    """
    #TODO: Reduce the font sizes of the titles and captation
    #TODO: Reduce the scatterplot size of the points
    #TODO: Increase the plot image size
    time_axis = feature_stats.time_stamp
    fig, axs = plt.subplots(4, 2)
    fig.suptitle(f'Voxel [{voxel_id}]: Permeability Features')
    # Plot the voxel features of interrst
    # 1: x: Time, y: Number of rays ending in a voxel
    axs[0,0].set_title("Ray count /t")
    axs[0,0].plot( time_axis, feature_stats.ray_end_count)
    axs[0,0].set_xlabel("t[s]")
    axs[0,0].set_ylabel("# ray's")
    # if final_feature_stats.ray_end_count:
    #     axs[0,0].plot( time_axis, [final_feature_stats.ray_end_count[0] for x in time_axis])


    # 2: x: Time, y: Permeability Entropy
    axs[1,0].set_title("Perm Entropy vs time")
    axs[1,0].plot( time_axis,  feature_stats.perm_entropy)
    axs[1,0].set_xlabel("Time")
    axs[1,0].set_ylabel("Perm H")
    # if final_feature_stats.perm_entropy:
    #     axs[1,0].plot( time_axis, [final_feature_stats.perm_entropy[0] for x in time_axis])

    # 3: x: Total number of rays, y: Permeability Entropy
    axs[1,1].set_title("Perm Entropy vs #-rays")
    axs[1,1].scatter( feature_stats.ray_end_count,  feature_stats.perm_entropy)
    axs[1,1].set_xlabel("#-rays")
    axs[1,1].set_ylabel("Perm H")
    # if final_feature_stats.perm_entropy:
    #     axs[1,1].plot( time_axis, [final_feature_stats.perm_entropy[0] for x in time_axis])

    # 4: x: Time, y: Permeability mean
    axs[2,0].set_title("Perm Mean vs #-rays")
    axs[2,0].plot( time_axis,  feature_stats.perm_mean)
    axs[2,0].set_xlabel("time")
    axs[2,0].set_ylabel("Perm mean")
    # if final_feature_stats.perm_mean:
    #     axs[2,0].plot( time_axis, [final_feature_stats.perm_mean[0] for x in time_axis])

    # 4: x: Ray count, y: Permeability mean
    axs[2,1].set_title("Perm Mean vs #-rays")
    axs[2,1].scatter( feature_stats.ray_end_count, feature_stats.perm_mean)
    axs[2,1].set_xlabel("#-rays")
    axs[2,1].set_ylabel("Perm mean")
    # if final_feature_stats.perm_mean:
    #     axs[2,1].plot( time_axis, [final_feature_stats.perm_mean[0] for x in time_axis])

    # 5 Perm vs time
    axs[3,0].set_title("Perm Var vs time")
    axs[3,0].plot( time_axis, feature_stats.perm_variance)
    axs[3,0].set_xlabel("time")
    axs[3,0].set_ylabel("Perm var")
    # if final_feature_stats.perm_variance:
    #     axs[3,0].plot( time_axis, [final_feature_stats.perm_variance[0] for x in time_axis])
    # 6 Perme Var vs #-rays
    axs[3,1].set_title("Perm Var vs #-rays")
    axs[3,1].scatter( feature_stats.ray_end_count, feature_stats.perm_variance)
    axs[3,1].set_xlabel("#-rays")
    axs[3,1].set_ylabel("Perm var")
    # if final_feature_stats.perm_variance:
    #     axs[3,1].plot( time_axis, [final_feature_stats.perm_variance[0] for x in time_axis])
        
    # 5: x: Time, y: Permeability variance 
    fig.set_size_inches(24, 24)
    plt.tight_layout()
    return fig


def point_in_voxel(querry_point, voxel_point, voxel_size):
    """ Checks if the querry point lies within the voxel bounds given by the voxel point.
    Assumptions: 
    -   Voxels start at (0,0,0) and have same leave size for all sides
    
    Args:
        - querry_point: Point to check
        - voxel_point: Point of the voxel and from which the bounds of the voxels are infered
        - voxel_size: Size of the voxel leafs
    Retrun:
        True if the queery_point lies within a voxel, False otherwise
    """

    # Get the bounds we need to compare
    x_low =  voxel_size* float(voxel_point[0] // voxel_size)
    x_high = x_low + voxel_size

    y_low = voxel_size* float(voxel_point[1] // voxel_size)
    y_high = y_low + voxel_size

    z_low = voxel_size* float(voxel_point[2] // voxel_size)
    z_high = z_low + voxel_size

    return  x_low <= querry_point[0] < x_high and \
            y_low <= querry_point[1] < y_high and \
            z_low <= querry_point[2] < z_high

if __name__ == "__main__":
    b =1
    main()
