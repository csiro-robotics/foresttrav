# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz



def evaluate_test_data(model: object, 
                   scaler: object, 
                   test_data_set: object,
                   voxel_size:float,
                   device: str = 'cuda',):
    """ Evaluate the test data set for a model

    Args:
        model (obj): Model to evaluate the test set on
        scaler (obj): Scaler associated with the model, used to scale the test_set
        voxel_size (obj): Voxel leaf size in meters [m]
        device (str):   Key to define which accelerator is used {'cpu', 'cuda', 'gpu'}
    """
    # Set model to evaluation
    model.to(device)
    model.eval()

    # Extract feature set
    # Check what the test data sets are like and in what dat format
    
    X_scaled = scaler.scale(test_data_set.X_features)
    
    # Easy(dirty) way to deal with the two types of models. 
  
    return  model.predict_te_classification(
        X_coords=test_data_set.X_features,
        X_features=X_scaled,
        voxel_size=voxel_size,
        device=device)
