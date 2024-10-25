# Copyright (c) 2023
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

from dataclasses import dataclass

from scnn_tm.models.FtmMePlModule import ForestTravPostProcessdDataPLModule, ForestTravPLMParams
from odap_tm.models.UnetMultiHeadPLModule import UnetMultiHeadPLModule, TwoHeadPLMParams
# from odap_tm.models.PUVoxTrav import  PUVoxTravPLM

from odap_tm.models.io_model_parsing import model_selection, dict_to_params, obj_to_params

PLM_MAP = {
    # Definitons
    "UnetMultiHeadPLModule":UnetMultiHeadPLModule,
    "UNet4THM": UnetMultiHeadPLModule,
    "UNet3THM": UnetMultiHeadPLModule,
    
    # Old models from scnn
    'ForestTravPostProcessdDataPLModule':ForestTravPostProcessdDataPLModule,
    "UNet3LMCD":ForestTravPostProcessdDataPLModule,
    "UNet4LMCD":ForestTravPostProcessdDataPLModule,
    "UNet5LMCD":ForestTravPostProcessdDataPLModule,
    
    # 'PUVoxTravPLM': PUVoxTravPLM,
}

PLM_PARAMS_MAP = {
    'UnetMultiHeadPLModule': TwoHeadPLMParams,   
    "UNet4THM": TwoHeadPLMParams,
    "UNet3THM": TwoHeadPLMParams,
    # 'ForestTravPostProcessdDataPLModule':ForestTravPostProcessdDataPLModule,
    
    'ForestTravPostProcessdDataPLModule':ForestTravPLMParams,
    "UNet3LMCD":ForestTravPLMParams,
    "UNet4LMCD":ForestTravPLMParams,
    "UNet5LMCD":ForestTravPLMParams,
}

def setup_pl_module(params:object, model_files, plm_tag:str):
    
    # Parse the model if it is one of the PLM module
    if plm_tag in PLM_MAP.keys():
        params_plm = plm_params_from_tag(params, plm_tag)
        return PLM_MAP[plm_tag](model_files, params_plm)
    
    # Not a correct name provided so we try to resove it with the modele name
    if hasattr(params,"model_name"):
        print('Setup PLM modules from model_name, ignoring plm_tag')
        if params.model_name in PLM_MAP.keys():
            params_plm = plm_params_from_tag(params, params.model_name)
            return PLM_MAP[params.model_name] (model_files, params_plm)
    
    # Raise exception as we could not find the correct model
    msg = "No PL module could be found with the given name"
    raise ValueError(msg)


def plm_params_from_tag(dparams, tag)->dataclass:
    
    # If is a dictonary, conver
    if type(dparams) is dict:
        return  dict_to_params(dparams, PLM_PARAMS_MAP, tag)

    # If it is an obeject this will convert it into the  correc data class
    return obj_to_params(dparams, PLM_PARAMS_MAP[tag])
