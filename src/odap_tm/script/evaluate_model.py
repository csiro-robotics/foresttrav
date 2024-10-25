# Copyright (c) 2023
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from torchsparse.utils.collate import sparse_collate_fn

from odap_tm.models.BinaryTEEnsemble import BinaryFtmEnsemble
from odap_tm.models.io_model_parsing import load_model_from_config
from odap_tm.setup_training import evaluate_cv_model
from scnn_tm.models.ForestTravDataSet import ForestTravDataReader
from scnn_tm.utils import generate_feature_set_from_key


def main(params):
    
    model_configs = [
        (file_i, 1)
        for file_i in Path(params['model_dir']).iterdir()
        if (file_i.is_file() and "model_config" in file_i.name)
    ]
    
    
    # Load all params
    feature_set = generate_feature_set_from_key(params["feature_set_key"])
    
    ensemle = BinaryFtmEnsemble(
        model_config_files=model_configs, input_feature_set=feature_set, device="cuda",
    )
    # Load the data sets
    data_set = ForestTravDataReader(
        data_sets_files=params["test_set"], feature_set=feature_set
    )

    # Load test
    te_prob, te_var = ensemle.predict(
        data_set.raw_data_set[0]["coords"],  data_set.raw_data_set[0]["features"], params['voxel_size']
    )
    
    y_target = data_set.raw_data_set[0]["label"]

    # te_prob > 0.5
    te_prob[te_prob < 0.5] = 0
    te_prob[te_prob >= 0.5] = 1
    print(matthews_corrcoef(y_pred=te_prob, y_true=y_target))
    
    evaluate_cv_model

# Exemplar model file to load
PARAMS = {
    "model_name": "test_model",
    # "model_dir": '/data/nav_experiments/online_model/incremental_models/1722840376_662922859',
    "model_dir": '/data/nav_experiments/online_model/incremental_models/1722840376_662922859',
    "test_set": ["/data/forest_trav/lfe_hl_v0.1/9_2022_02_14_23_47_33Z.csv"],
    "feature_set_key": "ohm_mr",
    'voxel_size': 0.1,
}

# 0.17796365280861728
# 0.18433061236243523
# 0.2352825610886823
# 0.3296472227582429
# 0.43647396800272614
# 0.5388787391148656
# 0.6303082035777882



if __name__ == "__main__":
    main(PARAMS)
