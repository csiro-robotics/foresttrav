# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz
import pytest as pt

from pathlib import Path
import numpy as np

from odap_tm.models.io_model_parsing import model_selection, dict_to_params, MODEL_MAP, MODEL_MAP_PARAMS

from odap_tm.models.io_plm_parsing  import setup_pl_module, PLM_MAP

#######################   PLM Loading      ############################

def test_PLM_load():
    
    # Default parameters
    model_components = { 
                        "train_data_set": 0.2,
                        "val_data_set": 0.1,
                        "test_data_set": 0.4, 
                        "model": None, 
                        "scaler": None, 
                        }
    learing_params = {
        'lr': 0.001,
        'weight_decay': 0.01,
        'voxel_size': 0.1,
        'batch_size': 16,
        'val_batch_size': 16,
        'loss_function_tag': 'TwoHeadLoss',
        'ce_weight': 1.5,
        'rec_weight': 0.001,
        'ce_label_weight': 1.5,
        'ce_threshold': 0.5,
        'label_weights': [1.0, 1.0]
    }
    
    # Normal loading with loss
    plm_module = setup_pl_module(params=learing_params, model_files=model_components, plm_tag="UNet4THM")
    
    # Loading with prob loss
    learing_params['loss_function_tag'] = 'TwoHeadProbLoss'
    plm_module = setup_pl_module(params=learing_params, model_files=model_components, plm_tag="UNet4THM")
    
    for tag in PLM_MAP.keys():
        plm_module = setup_pl_module(params=learing_params, model_files=model_components, plm_tag=tag)

    # Load the model based on the 
test_PLM_load()
#######################   Model Loading      ############################

def test_missing_params():
    params_dic  ={
        'model_name':"UNet4THM",
    }
    
    with pt.raises(TypeError):
        params = dict_to_params(params_dic, MODEL_MAP_PARAMS, params_dic['model_name'] )
    

def test_load_wrong_model():
    # Case where somehow the model name gets mutated and results in a error
    params_dic  ={
        'model_name':"Uhnt2",
        'model_stride': [1,2,2,2,2,2,2],
        'nfeatures': 10,
        'model_skip_connection': [1,1,1,1,1,1], 
        'model_nfeature_enc_ch_out': 10,
    }
    
    
    with pt.raises(KeyError):
        params = dict_to_params(params_dic, MODEL_MAP_PARAMS ,list(MODEL_MAP.keys())[0])
        params.model_name = "RandomModel"
        model = model_selection(params=params)
        
def test_load_all_valid_models():
    # Check all valid models and ensures the interfaces work
    
    params_dic  ={
        'model_name':"Uhnt2",
        'model_stride': [1,2,2,2,2,2,2],
        'nfeatures': 10,
        'model_skip_connection': [1,1,1,1,1,1], 
        'model_nfeature_enc_ch_out': 10,
    }    
    for model_name in MODEL_MAP.keys():
        params = dict_to_params(params_dic,MODEL_MAP_PARAMS, model_name)
        params.model_name = model_name
        model = model_selection(params=params)
    
def test_load_model_params_from_dict():
    params_dic  ={
        'model_name':"Uhnt2",
        'model_stride': [1,2,2,2,2,2,2],
        'nfeatures': 10,
        'model_skip_connection': [1,1,1,1,1,1], 
        'model_nfeature_enc_ch_out': 10,
        'dummy_one': 1,
        'dummy_two': "Not usefull variable"
    }
    with pt.raises(KeyError):
        params = dict_to_params(params_dic, MODEL_MAP_PARAMS, params_dic['model_name'] )
    

    for model_name in MODEL_MAP.keys():
        params = dict_to_params(params_dic, MODEL_MAP_PARAMS,model_name)
        params.model_name = model_name
        model = model_selection(params=params)

def test_laod_from_dict():
    params_dic  ={
        'model_name':"RandomModel",
        'model_stride': [1,2,2,2,2,2,2],
        'nfeatures': 10,
        'model_skip_connection': [1,1,1,1,1,1], 
        'model_nfeature_enc_ch_out': 10,
        'dummy_one': 1,
        'dummy_two': "Not usefull variable"
    }
    with pt.raises(KeyError):
        params = dict_to_params(params_dic, MODEL_MAP_PARAMS, params_dic['model_name'] )
    

    for model_name in MODEL_MAP.keys():
        params_dic['model_name'] = model_name
        model = model_selection(params=params_dic)
