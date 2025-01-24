# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz


from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.classification import MatthewsCorrCoef

try:
    from pytorch_lightning.core import LightningModule
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install pytorch_lightning`."
    )

from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn


@dataclass
class TwoHeadPLMParams:
    lr: float
    weight_decay: float
    batch_size: int
    val_batch_size: int
    ce_loss_weight: float
    rec_loss_weight: float
    loss_function_tag: str
    ce_label_weights: list
    ce_threshold: float
    shuffle: bool  # Bool to shuffel the data

    def __post_init__(self):
        for attr in self.__annotations__:
            if getattr(self, attr, None) is None:
                raise ValueError(f"Missing required attribute: {attr}")


REQ_TMH_PLM_PARAMS = [
    "lr",
    "weight_decay",
    "batch_size",
    "val_batch_size",
    "ce_loss_weight",
    "rec_loss_weight",
    "loss_function_key",
    "ce_label_weights",
    "ce_threshold",
    "shuffle",
]


class TwoHeadLoss(nn.Module):
    def __init__(self, ce_loss_weight, recon_weight, ce_label_weights):
        super(TwoHeadLoss, self).__init__()
        self.ce_loss_weight = ce_loss_weight
        self.recon_weight = recon_weight
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor(ce_label_weights))
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, recon_targets, labels, label_obs=None):

        # TODO(fab): Should this check  be in the forward function?
        if not (type(label_obs) is None):
            if not (label_obs > 0).all():
                msg = "Error, this loss can't use unlabelled data."
                raise ValueError(msg)
        # Assuming outputs is a tuple of (output1, output2)
        cl_outpt, recon_output = outputs

        # Cross entropy loss for output1 and labels
        ce_loss = self.cross_entropy(cl_outpt.F.squeeze(), labels)

        # Mean squared error loss for output2 and targets
        recon_loss = self.mse_loss(recon_output.F, recon_targets.F)

        # Total lossce_label_weights
        loss = self.ce_loss_weight * ce_loss + self.recon_weight * recon_loss

        return loss, ce_loss, recon_loss


# TODO: There is mixing tow concepts here. Using the loss function with observed vs label_prob, which can be pseudo labelled data..
class TwoHeadProbLoss(nn.Module):
    def __init__(self, ce_loss_weight, recon_weight, ce_label_weights, ce_threshold):
        super(TwoHeadProbLoss, self).__init__()
        self.ce_loss_weight = ce_loss_weight
        self.recon_weight = recon_weight
        self.label_threshold = ce_threshold
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor(ce_label_weights))
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, recon_targets, labels, label_obs):
        # Assuming outputs is a tuple of (output1, output2)
        cl_outpt, recon_output = outputs

        # Mean squared error loss for output2 and targets
        recon_loss = self.mse_loss(recon_output.F, recon_targets.F)

        # Cross entropy loss for labels
        mask_valid_labels = label_obs > 0.0

        # use label observations?
        ce_loss = self.cross_entropy(
            cl_outpt.F[mask_valid_labels].squeeze(), labels[mask_valid_labels]
        )

        # Total loss
        loss = self.ce_loss_weight * ce_loss + self.recon_weight * recon_loss

        return loss, ce_loss, recon_loss


class UnetMultiHeadPLModule(LightningModule):
    def __init__(
        self,
        model_files: dict,
        params: TwoHeadPLMParams,
    ):
        super().__init__()

        # Generate parameter
        self.model = model_files["model"]
        self.params = params
        self.learning_params = model_files["params"]
        self.save_hyperparameters(ignore=["model", "data_loader", "model_files"])

        self.setup_data_set(model_files)

        self.set_loss_function(params)

        self.mcc_scorer = MatthewsCorrCoef(
            task="binary",
            num_classes=2,
            # compute_on_step=False, # This should not be an issue?
        )

    def set_loss_function(self, params):
        if params.loss_function_tag == "TwoHeadLoss":
            self.criterion = TwoHeadLoss(
                ce_loss_weight=params.ce_loss_weight,
                recon_weight=params.rec_loss_weight,
                ce_label_weights=params.ce_label_weights,
            )
        elif params.loss_function_tag == "TwoHeadProbLoss":
            self.criterion = TwoHeadProbLoss(
                ce_loss_weight=params.ce_loss_weight,
                recon_weight=params.rec_loss_weight,
                ce_label_weights=params.ce_label_weights,
                ce_threshold=params.ce_threshold,
            )
        else:
            msg = "Loss function not defined"
            raise ValueError(msg)

    def train_dataloader(self):
        return DataLoader(
            self.train_data_loader_,
            batch_size=self.params.batch_size,
            collate_fn=sparse_collate_fn,
            shuffle=self.params.shuffle,
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        # Must clear cache at regular interval
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()

        stensor = batch["input"]

        loss = self.criterion(
            self(stensor),
            stensor,
            batch["label"].F.long().squeeze(),
            batch["label_obs"].F.squeeze(),
        )

        self.log(
            f"train_loss",
            loss[0],
            batch_size=self.params.batch_size,
            prog_bar=True,
        )
        self.log(f"ce_loss", loss[1], batch_size=self.params.batch_size)
        self.log(f"recon_los", loss[2], batch_size=self.params.batch_size)
        return loss[0]

    def validation_step(self, batch, batch_idx):
        stensor = batch["input"]
        loss = self.criterion(
            self(stensor),
            stensor,
            batch["label"].F.long().squeeze(),
            batch["label_obs"].F.squeeze(),
        )

        # The need when using unlabeled data
        y_label = batch["label"].F.long().squeeze()
        mask = y_label > 0.0

        y_pred = torch.argmax(self(stensor)[0].F.squeeze(), 1)

        self.mcc_scorer.update(y_pred[mask], y_label[mask])
        self.log(f"val_loss", loss[0], batch_size=self.params.batch_size)
        self.log(f"val_cl", loss[1], batch_size=self.params.batch_size)
        self.log(f"val_recon", loss[2], batch_size=self.params.batch_size)
        self.log(
            f"val_mcc",
            self.mcc_scorer,
            on_epoch=True,
            on_step=False,
            batch_size=self.params.batch_size,
        )

        return loss[0]

    def configure_optimizers(self):
        return Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )

    def setup_data_set(self, model_files):
        # Label weights should add to
        self.model = model_files["model"]
        self.scaler = model_files["scaler"]
        self.train_data_loader_ = model_files["train_data_set"]
        self.val_data_loader_ = model_files["val_data_set"]

    def get_scaler(self):
        return self.scaler
