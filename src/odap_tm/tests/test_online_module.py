# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz
from pathlib import Path
import pytest
from copy import deepcopy

from odap_tm.models.OnlineLearingModule import OnlineActiveLearingModule
from scnn_tm.utils import load_yamls_as_struct


# TODO: Learn model from "scratch" like a goldfish (load and forget)
DEFAULT_CONFIG_FILE = Path(__file__).parent / "config" / "default_online_learning_params.yaml"
DEBUG_ONLINE_DATA = Path("/data/forest_trav/online_lfe_data/small_debug_data.hdf5")
DEBUG_EMPTY_ONLINE_DATA = [ '/data/debug/incremental_online_learning/qcat_forest_1_lmix_1722557430.0349348.hdf5',
                           '/data/debug/incremental_online_learning/qcat_forest_1_lmix_1722557450.0350096.hdf5']

def test_valid_initialisation():
    
    learning_params = load_yamls_as_struct(Path(DEFAULT_CONFIG_FILE))
    online_learner = OnlineActiveLearingModule(learning_params)
    
    # Update the online learner
    online_learner.update_online_dataset([DEBUG_ONLINE_DATA])
    # online_learner.train_new_model(model_nbr=0)

    
def test_combination_using_unlabeled_data():
    """ Test invalid configurations when using unlabelled data"""
    
    learning_params = load_yamls_as_struct(Path(DEFAULT_CONFIG_FILE))
    with pytest.raises(ValueError) as exc_info:
            # Invalid case:
        params = deepcopy(learning_params) 
        params.loss_function_tag = "TwoHeadLoss"
        params.use_unlabeled_data = True
        params.model_name = "UNet4THM"
        online_learner = OnlineActiveLearingModule(params)
        a =1
    
    with pytest.raises(ValueError) as exc_info:
            # Invalid case:
        params = deepcopy(learning_params) 
        params.loss_function_tag = "TwoHeadProbLoss"
        params.use_unlabeled_data = True
        params.model_name = "UNet4LMCD"
        online_learner = OnlineActiveLearingModule(params)
        a =1
    

    # Valid case
    params = deepcopy(learning_params) 
    params.loss_function_tag = "TwoHeadProbLoss"
    params.use_unlabeled_data = True
    params.model_name = "UNet4THM"
    online_learner = OnlineActiveLearingModule(params)
    
    # Valid case
    params = deepcopy(learning_params) 
    params.loss_function_tag = "TwoHeadLoss"
    params.use_unlabeled_data = False
    params.model_name = "UNet4LMCD"
    online_learner = OnlineActiveLearingModule(params)


def test_use_default_scaler():
    learning_params = load_yamls_as_struct(Path(DEFAULT_CONFIG_FILE))
    
    with pytest.raises(FileNotFoundError) as exc_info:
            # Invalid case:
        params = deepcopy(learning_params) 
        params.use_default_scaler = True
        params.default_scaler_path = ""

        online_learner = OnlineActiveLearingModule(params)    

    params = deepcopy(learning_params) 
    params.use_default_scaler = True
    params.default_scaler_path = "/data/base_models_24_5_16_hl/full_base/24_05_16_12_42_UNet4THM_occ_int_hm_mr_test_train_epoch_150_fn_12/scaler_cv_8.joblib"
    online_learner = OnlineActiveLearingModule(params)
    
    for model_files in online_learner.base_models:
        online_learner.default_scaler == model_files["scaler"]

    # TODO(fab): The feature set should match for the scaler?


# TODO(fab): The test should remove the running of the model?
def test_empty_data():
    ''' Test to determine if the training data is handled correctly in the case of emtpy data.
        Case 1: The hdf5 file is empty and does not add to the graph -> raise Value error
        
    TODO: The data should be tested with a @fn data_is_valid() and ensure a minimum number of training examples are available 
    '''
    learning_params = load_yamls_as_struct(Path(DEFAULT_CONFIG_FILE))
    
    # Valid case
    params = deepcopy(learning_params) 
    params.loss_function_tag = "TwoHeadLoss"
    params.use_unlabeled_data = False
    params.model_name = "UNet4LMCD"
    params.max_epochs = 1
    
    
    online_learner = OnlineActiveLearingModule(params)
    # This needs to be asserted
    online_learner.update_online_dataset([DEBUG_EMPTY_ONLINE_DATA[0]])
    
    # This should fail with the empty data
    assert not online_learner.train_new_model(model_nbr=0)


def test_empty_and_valid_data():
    ''' Test to determine if the training data is handled correctly in the case of emtpy data.
        Case 1: The hdf5 file is empty and does not add to the graph -> raise Value error
        Case 2: The tow hdf5 files, one is empty and the second one has valid data.
            - Integrating the first one should raise an exception (ValueException)
            - Integrating the second one should pass
            - Training should start (no exception)
    
    '''
    learning_params = load_yamls_as_struct(Path(DEFAULT_CONFIG_FILE))
    
    # Valid case
    params = deepcopy(learning_params) 
    params.loss_function_tag = "TwoHeadLoss"
    params.use_unlabeled_data = False
    params.model_name = "UNet4LMCD"
    params.max_epochs = 1
    online_learner = OnlineActiveLearingModule(params)
    online_learner.update_online_dataset(DEBUG_EMPTY_ONLINE_DATA)
    
    # This will run but is annoying
    online_learner.train_new_model(model_nbr=0)
