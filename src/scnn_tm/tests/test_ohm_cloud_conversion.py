# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

from pathlib import Path
import numpy as np
import pandas as pd
import copy

from scnn_tm.utils.point_cloud_utils import convert_ohm_cloud_to_feature_cloud
from scnn_tm.utils.scnn_io import generate_feature_set_from_key

# This is a hardcoded feature set defined by "ohm-mapping" assumed to be static
OHM_SCAN_FILE = Path(__file__).parent / "config" / "test_ohm_scan.csv"
OHM_MAP_FILE = Path(__file__).parent / "config" / "test_ohm_feature_set.csv"


def test_ohm_to_feature_cloud_conversion():
  df_ohm_scan = pd.read_csv(OHM_SCAN_FILE)
  
  ohm_cloud_arr  = copy.deepcopy(df_ohm_scan.to_numpy())
  org_feature_set = list(df_ohm_scan.columns)
  
  # Generate a random cloud with   
  X_coords, feature_data = convert_ohm_cloud_to_feature_cloud(ohm_cloud_arr, org_feature_set, generate_feature_set_from_key("ftm"))
  
  # Test set
  df_ohm_fs = pd.read_csv(OHM_MAP_FILE)
  ohm_fs_coords = df_ohm_fs[["x","y","z"]].to_numpy()
  ohm_fs_feature_data = df_ohm_fs[generate_feature_set_from_key("ftm")].to_numpy()
  
  # Check for no mutation in coordinates
  assert (np.isclose(X_coords , ohm_fs_coords)).all()
  
  # Check that the new features are present and not -1 (just initialized)
  for i in range(feature_data.shape[1], 2):
    assert (np.isclose(feature_data[:, i] ,ohm_fs_feature_data[:, i])).all()
    
test_ohm_to_feature_cloud_conversion()
