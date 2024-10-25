
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import scnn_tm.utils as utils
from scnn_tm.models.ForestTravGraph import ForestTravGraph
from scnn_tm.models.OnlineForestTrav import (
    OnlineDataSetLoader,
)

FEATURE_SET = [
    "mean_count",
    "traversal",
    "hit_count",
    "red",
]


DATA_DIR = "/data/temp/qcat_forest_1_lmix"
OUTDIR = "/data/online_learning/temp/qcat_forest_1_lmix"
USE_UL= False

def test_ForesTravGraphdIncremental():
    data_graph = ForestTravGraph(
        voxel_size=0.1, patch_width=4.8, min_dist_r=0.5, use_unlabeled_data=USE_UL
    )
    data_loader = OnlineDataSetLoader(
        target_feature_set=FEATURE_SET,
        voxel_size=0.1,
    )

    files = [ file for file in Path(DATA_DIR).iterdir() if file.is_file()]
    files.sort()
    
    
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    i =0
    for data_set_file in files:
        new_data_batch = data_loader.load_online_data_set_filtered(data_set_file, 0.0)
        data_graph.add_new_data(new_data_batch=new_data_batch)
        
        data_set = data_graph.get_patch_data_set_copy(0.5)
        
        file_name = Path(OUTDIR) / f"test_data_incremental_{i}.csv"
        safe_data_to_csv(data_set, file_name)
        file_name = Path(OUTDIR) / f"poses_data_invremental_{i}.csv"
        save_node_posed(data_set, file_name)
        i+= 1

    # Save cloud to file

def safe_data_to_csv(data_set, file_path):
    
    # Get the coords
    coords = np.vstack(
        [data["coords"] for data in data_set]
    )
    
    labels_prob = np.vstack(
        [data["label_prob"] for  data in data_set]
    )

    labels = np.vstack(
        [data["label"] for  data in data_set]
    )
    
    labels_obs = np.vstack(
        [data["label_obs"] for data in data_set]
    )
    
    out_file = Path(file_path)

    pd.DataFrame(np.concatenate([coords, labels.reshape(-1, 1),labels_prob.reshape(-1, 1), labels_obs.reshape(-1,1) ], axis=1)).to_csv(
       file_path,
        index=False,
        header=["x", "y", "z", "label", "prob", "obs"],
        
    )

def save_node_posed(data_set: list, file_path: Path):    
    # Get the trajectory...
    node_pos = np.vstack([data["pose"][:3] for data in data_set])
    pd.DataFrame(node_pos).to_csv(
       file_path,
        index=False,
        header=["pos_x", "pos_y", "pos-z"],
        
    )    
    
if __name__ == "__main__":
    test_ForesTravGraphdIncremental()
