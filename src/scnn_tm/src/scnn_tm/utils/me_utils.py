# MIT License
#
# Copyright (c) 2023 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

import numpy as np
from torchsparse.utils.collate import sparse_collate_fn

# Turns a dictionary into a class
class Dict2Class(object):
    def __init__(self, my_dict):

        for key in my_dict:
            setattr(self, key, my_dict[key])



def predics_labels_for_cv(model, data_loader, device = "cuda"):
    ''' The labels for the test set for this cross val set from the data loader'''
    # Loop over all data sets and get labels_pred and labels
    y_pred = []
    model.eval().to(device)
    
    # To interresting part here is that the data seems to need the batch coordinate at the end
    dl_batched = [data_loader[i] for i in range(len(data_loader)-1)]
    batched_data = sparse_collate_fn(dl_batched)
    
    # A marvel of torch conversion
    logits = model(batched_data["input"].to(device))
    _, y_pred = logits.F.max(1)
    y_pred = y_pred.cpu().numpy()
    
    y_target = batched_data["label"].F.squeeze().cpu().numpy()
    return y_pred, np.hstack(y_target)



