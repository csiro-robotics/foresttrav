# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz
from pathlib import Path
from odap_tm.models.io_model_parsing import model_config_from_file


MOCK_BAD_MODEL_PARAMS = {
    "model_name": "RandomModel",
    "cv_num": 2,
    "feature_set": ["mean_count", "occ_log_probability"],
    "nfeatures": 2,
    "model_skip_connection_key": "s2",
    "model_stride": [1, 2, 2, 2, 2, 2, 2],
    "model_skip_connection": [1, 1, 1, 1, 1, 1],
    "model_nfeature_enc_ch_out": 10,
}


def test_model_config_from_filed():
    model_config_file = Path(
        "/data/base_models_24_5_16_hl/dense_base/24_05_16_10_49_UNet4THM_occ_int_hm_mr_test_train_epoch_150_fn_12/model_config_cv_5.yaml"
    )
    models_param = model_config_from_file(config_file=model_config_file, convert_dic_to_obj=False)
    

def test_loading_models():
    model_file = Path(
        "/data/base_models/dense_forest_base/24_04_17_10_17_UNet3LMCD_occ_test_train/model_config_cv_0.yaml"
    )
    assert model_file.exists()
    params = load_model(model_file, False)
    
    model_files = load_model(model_file)


test_model_config_from_filed()
