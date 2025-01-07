# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz
from pathlib import Path
import numpy as np


from scnn_tm.models.ForestTravDataSet import ForestTravDataReader

DATA_SET_FILES = ["/data/forest_trav/lfe_hl_v0.2/2_2021_12_14_00_14_53Z.csv"]


DATA_FIELDS = {
    "coords": ["x", "y", "z"],
    "feature_set": [
        "occupancy_prob",
        "intensity_mean",
        "intensity_covariance",
        "miss_count",
        "hit_count",
        "permeability",
    ],
}

def test_load_simple_csv():
    data_loader = ForestTravDataReader(
        data_sets_files=DATA_SET_FILES,
        feature_set=DATA_FIELDS["feature_set"]
    )
