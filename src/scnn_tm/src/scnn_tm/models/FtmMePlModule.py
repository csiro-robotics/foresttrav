# Copyright (c) 2024
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

try:
    from pytorch_lightning.core import LightningModule
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install pytorch_lightning`."
    )

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchmetrics.classification import MatthewsCorrCoef

from scnn_tm.models.ForestTravDataSetFactory import ForestTravDataSetFactory
from scnn_tm.models.OnlineForestTrav import OnlineForestTravDataFactory
from torchsparse.utils.collate import sparse_collate_fn 

from dataclasses import dataclass


def setup_data_set_factory(train_strategy: str, params, scaler):
    """Setup for the training strategy and how to form/train test sets

    Parameters:
        startegy (str):     Strategy key which defines what data set factory should be called and how to handle the data_sets

    Return:
        data_set_factory        Returns the data_set_factory with the splits and weighting for the data.
    """
    data_set_factory = None

    if not (
        hasattr(params, "test_data_sets")
        and hasattr(params, "train_data_sets")
        and hasattr(params, "feature_set")
    ):
        msg = "Parameters are missing definition of data sets (train or test) or feature ses."
        raise ValueError(msg)
    
    # Online data set loader
    # TODO: Add the online data set and load the correct one
    if "online" in params.training_strategy_key:
        #
        params.patch_width = params.voxel_size * params.nvoxel_leaf   
        data_set_factory = OnlineForestTravDataFactory(
            params=params,
            scaler=scaler,
        )
    elif "test_train" in train_strategy:
        data_set_factory = ForestTravDataSetFactory(
                params=params,
                scaler=scaler,
            )
    else:
        raise ValueError(
            f"Cloud not find strategy [{train_strategy}] for {params.data_set_key}"
        )

    return data_set_factory


@dataclass
class ForestTravPLMParams:
    lr: float
    weight_decay: float
    voxel_size: float
    batch_size: int
    label_weights: list
    val_batch_size: int
    use_unlabelled_data: bool

class ForestTravPostProcessdDataPLModule(LightningModule):
    """
        Segmentation Module for PostProcessed data
    """

    def __init__(
        self,
        model_files: dict,
        params: ForestTravPLMParams,
    ):
        super().__init__()
        self.model_files = model_files
        self.model = model_files["model"]
        self.params = params

        self.learning_params = model_files["params"]
        self.save_hyperparameters(ignore=["model", "data_loader", "model_files"])
        
        self.initialise_data_set()

        self.use_ul = params.use_unlabelled_data

        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(self.learning_params.ce_label_weights),
        )


        self.mcc_scorer = MatthewsCorrCoef(
            task="binary",
            num_classes=2,
            # compute_on_step=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data_loader_,
            batch_size=self.params.batch_size,
            collate_fn=sparse_collate_fn,
            shuffle=True,
            num_workers=16,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data_loader_,
            batch_size=self.params.val_batch_size,
            collate_fn=sparse_collate_fn,
            shuffle=False,
            num_workers=16,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data_loader_,
            batch_size=self.params.batch_size,
            collate_fn=sparse_collate_fn,
            shuffle=False,
            num_workers=16,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Must clear cache at regular interval
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()

        stensor = batch["input"]
        
        loss = 0.0
        if self.use_ul:
            mask_valid_labels =  (batch["label_obs"].F > 0.0).squeeze()
            
            if not mask_valid_labels.any():
                return 0
            
            loss = self.criterion(self(stensor).F[mask_valid_labels, :].squeeze(), batch["label"].F[mask_valid_labels].long().squeeze())
        else:
            loss = self.criterion(
            self(stensor).F.squeeze(), batch["label"].F.long().squeeze()
         )
            
        # 
        self.log("train_loss", loss, batch_size=self.params.batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        stensor = stensor = batch["input"]
        
        y_label = []
        y_pred = []
        loss = 0.0
        if self.use_ul:
            mask_valid_labels =  (batch["label_obs"].F > 0.0).squeeze()
            loss = self.criterion(self(stensor).F[mask_valid_labels, :].squeeze(), batch["label"].F[mask_valid_labels].long().squeeze())
            y_label = batch["label"].F[mask_valid_labels].long().squeeze()
            y_pred = torch.argmax(self(stensor).F[mask_valid_labels].squeeze(), 1)
        else:
            loss = self.criterion(
            self(stensor).F.squeeze(), batch["label"].F.long().squeeze()
         )
            y_label = batch["label"].F.long().squeeze()
            y_pred = torch.argmax(self(stensor).F.squeeze(), 1)
            

        self.mcc_scorer.update(y_pred, y_label)
        self.log("val_loss", loss, batch_size=self.params.batch_size, prog_bar=True)
        self.log(
            "val_mcc",
            self.mcc_scorer,
            on_epoch=True,
            on_step=False,
            batch_size=self.params.batch_size,
        )

        return loss

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

    def initialise_data_set(self):
        # Label weights should add to

        self.train_data_loader_ = self.model_files['train_data_set']
        self.val_data_loader_ = self.model_files['val_data_set']
        self.test_data_loader_ = self.model_files['test_data_set']

        # Make sure again that the data loader do not share the same data!
        assert not self.train_data_loader_ == self.val_data_loader_
        assert not self.train_data_loader_ == self.test_data_loader_
        assert not self.val_data_loader_ == self.test_data_loader_
