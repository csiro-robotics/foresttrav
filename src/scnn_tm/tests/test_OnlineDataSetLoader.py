import pytest
from pathlib import Path

import numpy as np
import pandas as pd

import scnn_tm.utils as utils
from scnn_tm.models.ForestTravGraph import ForestTravGraph
from scnn_tm.models.OnlineForestTrav import (
    OnlineDataSetLoader,
)

# /data/forest_trav/online_lfe_data/heritage_forest_no_ohm_clear.hdf5
# /data/forest_trav/online_lfe_data/QCAT_FOREST_1.hdf5
# /data/forest_trav/online_lfe_data/QCAT_FOREST_2.hdf5
############################################## THIS SHOULD BE IN TEST FILE
HDF5_DEBUG_FILE = Path("/data/forest_trav/online_lfe_data/QCAT_FOREST_2.hdf5")

FEATURE_SET = [
    "mean_count",
    "traversal",
    "hit_count",
    "red",
]

TIMES = [
    # 0,
    # 1687322725.424786000,
    # 1687322725.424786000,
    1687323155.733388000,
]  # 1687323275.739095000]


def test_bad_features_set():

    with pytest.raises(ValueError) as exc_info:
        data_loader = OnlineDataSetLoader(
            target_feature_set=[],
            voxel_size=0.1,
        )

    with pytest.raises(ValueError) as exec_info:
        data_loader = OnlineDataSetLoader(
            target_feature_set=["dummy"],
            voxel_size=0.1,
        )
        
        data_loader.load_online_data_set_raw(HDF5_DEBUG_FILE)
        
    with pytest.raises(ValueError) as exec_info:
        data_loader = OnlineDataSetLoader(
            target_feature_set=["dummy"],
            voxel_size=0.1,
        )
        
        data_loader.load_online_data_set_filtered(HDF5_DEBUG_FILE)

def test_data_loading_feature_cloud():
    
    # The feature set contains all elements required
    data_loader = OnlineDataSetLoader(
        target_feature_set=FEATURE_SET,
        voxel_size=0.1,
    )
    
    # Test the new data batch
    new_data_batch = data_loader.load_online_data_set_filtered(HDF5_DEBUG_FILE)
    
    assert new_data_batch["feature_clouds"][-1].shape[1] == (3+ len(FEATURE_SET))
    assert new_data_batch["feature_clouds"][-1].shape[1] != len(data_loader.source_feature_set)
    
    # Raw data loading, when we want all the features...
    new_raw_data  = data_loader.load_online_data_set_raw(HDF5_DEBUG_FILE) 
    assert new_raw_data["feature_clouds"][-1].shape[1] == len(data_loader.source_feature_set)
    

# Testing the the values of the data seems somewhat nonsensical?


# def test_ForesTravGraph():
#     data_graph = ForestTravGraph(
#         voxel_size=0.1, patch_width=3.2, min_dist_r=1.0, use_unlabeled_data=True
#     )
#     data_loader = OnlineDataSetLoader(
#         target_feature_set=FEATURE_SET,
#         voxel_size=0.1,
#     )

#     for t_k in TIMES:
#         new_data_batch = data_loader.load_online_data_set_filtered(HDF5_DEBUG_FILE, t_k)
#         data_graph.add_new_data(new_data_batch=new_data_batch)

#     # Save cloud to file
#     coords = np.vstack(
#         [data["coords"] for key, data in data_graph.items() if "coords" in data]
#     )
#     labels = np.vstack(
#         [data["label_prob"] for key, data in data_graph.items() if "label_prob" in data]
#     )

#     poses = np.vstack([data["pose"][:3] for key, data in data_graph.items()])
#     Path(DEBUG_CSV_OUT).mkdir(parents=True, exist_ok=True)

#     pd.DataFrame(np.concatenate([coords, labels.reshape(-1, 1)], axis=1)).to_csv(
#         DEBUG_CSV_OUT / "test_pcl_cloud.csv",
#         index=False,
#         header=["x", "y", "z", "prob"],
#     )
#     pd.DataFrame(poses).to_csv(
#         DEBUG_CSV_OUT / "test_poses.csv", index=False, header=False
#     )


# FULL_GRAPH = [
#     "/data/forest_trav/online_lfe_data/qcat_forest_1/forest_1_1687325030.5586674.hdf5",
#     "/data/forest_trav/online_lfe_data/qcat_forest_1/forest_1_1687324970.5594018.hdf5",
#     "/data/forest_trav/online_lfe_data/qcat_forest_1/forest_1_1687324910.5655336.hdf5",
#     "/data/forest_trav/online_lfe_data/qcat_forest_1/forest_1_1687324850.054619.hdf5",
#     "/data/forest_trav/online_lfe_data/qcat_forest_1/forest_1_1687324790.0548322.hdf5",
#     "/data/forest_trav/online_lfe_data/qcat_forest_1/forest_1_1687324730.0532465.hdf5",
#     "/data/forest_trav/online_lfe_data/qcat_forest_1/forest_1_1687324669.5610383.hdf5",
#     "/data/forest_trav/online_lfe_data/qcat_forest_1/forest_1_1687324609.5534067.hdf5",
#     "/data/forest_trav/online_lfe_data/qcat_forest_1/forest_1_1687324549.5558023.hdf5",
#     "/data/forest_trav/online_lfe_data/qcat_forest_1/forest_1_1687324489.5621443.hdf5",
#     "/data/forest_trav/online_lfe_data/qcat_forest_1/forest_1_1687324429.5553713.hdf5",
# ]


# def test_ForesTravGraphdIncremental():
#     data_graph = ForestTravGraph(
#         voxel_size=0.1, patch_width=3.2, min_dist_r=1.0, use_unlabeled_data=True
#     )
#     data_loader = OnlineDataSetLoader(
#         target_feature_set=FEATURE_SET,
#         voxel_size=0.1,
#     )

#     FULL_GRAPH.sort()
#     i = 0
#     Path(DEBUG_CSV_OUT).mkdir(parents=True, exist_ok=True)
#     for data_set_file in FULL_GRAPH:
#         new_data_batch = data_loader.load_online_data_set_raw(data_set_file, 0.0)
#         data_graph.add_new_data(new_data_batch=new_data_batch)

#         data_set = data_graph.get_patch_data_set_copy(0.5)

#         file_name = DEBUG_CSV_OUT / f"test_data_incremental_{i}.csv"
#         safe_data_to_csv(data_set, file_name)
#         i += 1

#     # Save cloud to file


# def test_add_new_data():
#     """Test to check whether the addition of new data works as expected"""


# def safe_data_to_csv(data_set, file_path):

#     # Get the coords
#     coords = np.vstack([data["coords"] for data in data_set])

#     labels_prob = np.vstack([data["label_prob"] for data in data_set])

#     labels = np.vstack([data["label"] for data in data_set])

#     labels_obs = np.vstack([data["label_obs"] for data in data_set])

#     out_file = Path(file_path)

#     pd.DataFrame(
#         np.concatenate(
#             [
#                 coords,
#                 labels.reshape(-1, 1),
#                 labels_prob.reshape(-1, 1),
#                 labels_obs.reshape(-1, 1),
#             ],
#             axis=1,
#         )
#     ).to_csv(
#         file_path,
#         index=False,
#         header=["x", "y", "z", "label", "prob", "obs"],
#     )


# if __name__ == "__main__":
#     test_filtered_feature_cloud()
