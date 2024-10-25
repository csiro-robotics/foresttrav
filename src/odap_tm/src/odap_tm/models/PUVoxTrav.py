# Copyright (c) 2023
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

from dataclasses import dataclass, field
from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.classification import MatthewsCorrCoef
from torch import linalg as LA

try:
    from pytorch_lightning.core import LightningModule
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install pytorch_lightning`."
    )
import torchsparse
from torchsparse import nn as spnn
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn


from matplotlib import pyplot as plt

""" This is a implementation of a 3D version of the Wild Visual Navigation paper for 3D voxel based representations. Similarly, the PU learning is based on the
    idea that the we can label the possitive and unknown cases.
    The difference is the inclusion of another way of handling the non-traversable cases...
    ToDo:
        - [ ] Check that the initial PU method works, without the voxel labelling
        - [ ] Check the inclusion of the inclusion of the collisions into the costing
        - [ ] The confidence scoring needs to be examined and clearly defined for what is sensible for a voxel representation, LogGuassian? 
    Returns:
"""

def loss_rec_per_channel(predict, target):
    # return 1.0 / predict.shape[1]* torch.square(LA.norm(predict - target, dim=1))
    # return torch.mean(torch.square(predict - target), dim=1)
    return torch.sum(torch.square(predict - target), dim=1)
    # return torch.nn.functional.pairwise_distance(predict, target).square()


def confidence_exp_function(x, std, mean, k_sigma):
    return torch.exp(-1.0 * (torch.square(x - mean)) / (2.0 * (std * k_sigma) ** 2))


def confidence_score(
    label: torch.Tensor,
    label_obs: torch.Tensor,
    frec_pred: torch.Tensor,
    frec_target: torch.Tensor,
    k_sigma: float,
) -> torch.Tensor:
    """Calculates the confidence score based on WVN places. Unclear why this would work or how well this works

    Args:
        label       (torch.Tensor):    Labels which are used to mask te-scores, 0:NTE, 1:TE,  -1:(unknown)
        label_obs   (torch.Tensor):    Labels which are used to mask te-scores, 1:Observed by the robot, 0:Not observed by the robot, (unknown)
        frec_pred   (torch.Tensor):    Tensor containing the features of the prediction
        frec_target (torch.Tensor):    Tensor containing the features target
        k_sigma     (float):           Scaling factor to set scale the exponential component

    Returns:
        torch.Tensor: Tensor containing a
    """

    # Reduction less MSE loss used for

    # There are two considerations
    #   a) Use only traversable examples (as per paper) spred,F== 1
    #   b) Use only observed examples (allow to observe nte-examples) : spred.F != -1
    # This holds for equations (4) -(6) and is calculated below
    mask_te = torch.logical_and((label_obs == 1).squeeze(), (label == 1).squeeze())
    frec_pred_te = frec_pred[mask_te]
    frec_target_te = frec_target[mask_te]

    # Loss from equation (3) and
    loss_rec_te = loss_rec_per_channel(frec_pred_te, frec_target_te)
    loss_rec_te = torch.clip(loss_rec_te, 0.0, 100)
    std_te, mean_te = torch.std_mean(loss_rec_te)

    # Confidence estimation c(L(f_n)), assumption that it is on the whole feature set f_n
    loss_rec_total = loss_rec_per_channel(frec_pred, frec_target)
    c_fn = confidence_exp_function(
        loss_rec_total, std=std_te, mean=mean_te, k_sigma=k_sigma
    )

    # This means that if we are lower than the mean, it should be TE right?
    mask_smaller_mean = loss_rec_total < mean_te
    # c_fn[mask_smaller_mean] = 1.0

    return c_fn


class PUVoxTravLoss(nn.Module):
    def __init__(self, w_trav, w_recon, k_sigma):
        super(PUVoxTravLoss, self).__init__()
        self.w_trav = w_trav
        self.w_recon = w_recon
        self.k_sigma = k_sigma
        self.sigm = nn.Sigmoid()
        self.ce = torch.nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduce="none")

    def forward(
        self,
        outputs: SparseTensor,
        srec_target: SparseTensor,
        sproba_target: SparseTensor,
        slabels: SparseTensor,
        slabels_obs: SparseTensor,
    ):
        """_summary_

        Args:
            outputs (SparseTensor): _description_
            srec_target (SparseTensor): _description_
            sproba_target (SparseTensor): _description_
            slabels (SparseTensor): _description_

        Returns:
            _type_: _description_
        """
        # Assuming outputs is a tuple of (output1, output2)
        prob_out, recon_output = outputs

        # Only for known (LFE) features
        # TODO: Check if this converges without this hack?
        mask_lfe = torch.logical_and(
            (slabels_obs.F == 1).squeeze(), (slabels.F == 1).squeeze()
        )
        # loss_recon = self.mse_loss(recon_output.F[mask_lfe], srec_target.F[mask_lfe])
        loss_recon = (
            torch.nn.functional.pairwise_distance(
                recon_output.F[mask_lfe], srec_target.F[mask_lfe]
            )
            .square()
            .mean()
        )

        # Calculate the traversability loss
        loss_trav = self.loss_traversaility(
            labels=slabels.F.long().squeeze(),
            labels_obs=slabels_obs.F.long().squeeze(),
            frecon_pred=recon_output.F,
            frecon_target=srec_target.F,
            prob_pred=self.sigm(prob_out.F),
            prob_target=sproba_target.F,
        )

        # Total loss
        # loss = self.w_trav * loss_trav + self.w_recon * loss_recon
        loss = self.w_recon * loss_recon
        # loss  = loss_recon = self.mse_loss(recon_output.F[mask_lfe], srec_target.F[mask_lfe])

        return loss.float(), loss_trav.float(), loss_recon.float()

    def loss_traversaility(
        self,
        labels: torch.Tensor,
        labels_obs: torch.Tensor,
        frecon_pred: torch.Tensor,
        frecon_target: torch.Tensor,
        prob_pred: torch.Tensor,
        prob_target: torch.Tensor,
    ) -> float:
        """
        Calculates the traversability loss based on WVN paper, equations (3) - (8)

        Args:
            labels          (torch.Tensor):     Containing the target(true) labels features 0: for NTE, 1: for TE and -1: for Unobseverd
            labels_obs      (torch.Tensor):     Containing the observation label for each label. 1: Observed by robot experience(LFE), 0: Not observed by LFE
            frecon_pred     (torch.Tensor):     Sparse tensor containing the predicted reconstruction features
            frecon_target   (torch.Tensor):     Sparse tensor containing the target(true) reconstruction features
            prob_pred       (torch.Tensor):     Containing the predicted traversability probability
            prob_target     (torch.Tensor):     Containing the target(true) labels features

        Returns:
            float64: Returns the los as single real number [0,inf)
        """
        c_fn = confidence_score(
            label=labels,
            label_obs=labels_obs,
            frec_target=frecon_target,
            frec_pred=frecon_pred,
            k_sigma=self.k_sigma,
        )

        # Non-traversable loss
        # All elements that are either NTE or not observed
        mask_nte = torch.logical_or((labels_obs != 1).squeeze(), labels != 1)
        # f_target_nte = prob_target[mask_nte]
        f_pred_nte = prob_pred[mask_nte]
        c_f_nte = c_fn[mask_nte]

        # TODO: Check the loss function
        loss_nte = ((1.0 - c_f_nte) * LA.norm(f_pred_nte - 0.0, dim=1)).mean()

        # Traversable loss
        # MSE loss for all the traversable labeled and that have been observed elements!
        # mask_te = torch.logical_and((labels == 1).squeeze()  , (labels_obs == 1).squeeze())
        mask_te = (labels == 1).squeeze()

        prob_pred_te = prob_pred[mask_te]
        prob_target_te = prob_target[mask_te]
        loss_te = (
            torch.nn.functional.pairwise_distance(prob_pred_te, prob_target_te)
            .square()
            .mean()
        )

        return loss_nte + loss_te


class PUVoxTravPLM(LightningModule):
    def __init__(
        self,
        model,
        data_loader,
        lr: float,
        weight_decay: float,
        voxel_size: float,
        batch_size: int,
        params: list,
        val_batch_size=16,
    ):
        super().__init__()

        # Generate parameters
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)

        # TODO: 1) Move the data setup outside of the class and implement the timestamp component
        self.setup_data_set(data_loader)

        self.save_hyperparameters(ignore=["model", "data_loader"])

        # Setup the criterion
        self.criterion = PUVoxTravLoss(w_recon=1.0, w_trav=0.0, k_sigma=2)

        # MCC scores
        self.mcc_scorer = MatthewsCorrCoef(
            task="binary",
            num_classes=2,
            compute_on_step=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data_loader_,
            batch_size=self.batch_size,
            collate_fn=sparse_collate_fn,
            shuffle=self.params.shuffle,
            num_workers=16,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data_loader_,
            batch_size=self.val_batch_size,
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

        # Unpack data
        stensor = batch["input"]

        loss, loss_trav, loss_recon = self.criterion(
            outputs=self(stensor),
            srec_target=stensor,
            sproba_target=batch["label_prob"],
            slabels=batch["label"],
            slabels_obs=batch["label_obs"],
        )

        self.log(f"train_loss", loss, batch_size=self.batch_size, prog_bar=True)
        self.log(f"loss_trav", loss_trav, batch_size=self.batch_size)
        self.log(f"loss_recon", loss_recon, batch_size=self.batch_size)
        return loss.float()

    def validation_step(self, batch, batch_idx):
        stensor = batch["input"]

        loss, loss_trav, loss_recon = self.criterion(
            outputs=self(stensor),
            srec_target=stensor,
            sproba_target=batch["label_prob"],
            slabels=batch["label"],
            slabels_obs=batch["label_obs"],
        )

        # This is for the mcc score
        # TODO: Evaluate only the observed labels?
        mask_known = (batch["label_obs"].F > 0).squeeze()

        self.threshold = 0.5

        # Check if this makes sense!
        y_pred = self.model.predict_te(stensor, self.threshold)
        y_label = batch["label"].F.long().squeeze()[mask_known]
        y_pred = y_pred[mask_known].squeeze()

        self.mcc_scorer.update(y_pred, y_label)

        # Log all the scores:
        # TODO: Remove unecessary logs
        self.log(f"val_loss", loss, batch_size=self.batch_size)
        self.log(f"val_trav", loss_trav, batch_size=self.batch_size)
        self.log(f"val_recon", loss_recon, batch_size=self.batch_size)
        self.log(
            f"val_mcc",
            self.mcc_scorer,
            on_epoch=True,
            on_step=False,
            batch_size=self.batch_size,
        )

        return loss.float()

    def configure_optimizers(self):
        torch.autograd.set_detect_anomaly(True)
        # return Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer = Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sceduler = scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.9
        )
        return [optimizer], [sceduler]

    def setup_data_set(self, data_set_loader):
        # Label weights should add to
        self.data_loader = data_set_loader
        self.train_data_loader_ = self.data_loader.train_data_set
        self.val_data_loader_ = self.data_loader.val_data_set

    def get_scaler(self):
        return self.data_loader.scaler


class PUVoxTravUnet4L(nn.Module):
    STRIDE = [1, 2, 2, 2]
    SKIP_CONNECTION = [1, 1, 1]
    KERNEL_SIZE = 3

    # THIS it what is hould like for nfeature_enc_ch_out=8
    ENCODER_CH_IN = [0, 0, 0, 0]
    ENCODER_CH_OUT = [0, 0, 0, 0]

    DECODER_CH_IN = [0, 0, 0, 0]
    DECODER_CH_OUT = [0, 0, 0, 0]

    def __init__(
        self,
        nfeatures: int,
        encoder_nchannels: int = 8,
        stride: list = [1, 2, 2, 2],
        kernel_size=3,
    ):
        nn.Module.__init__(self)
        self.ENCODER_CH_IN[0] = nfeatures
        self.ENCODER_CH_OUT[0] = encoder_nchannels
        self.SKIP_CONNECTION = [0, 0, 0, 0]
        self.STRIDE = stride
        self.KERNEL_SIZE = kernel_size

        self.set_network_properties()

        self.block0 = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.ENCODER_CH_IN[0],
                out_channels=self.ENCODER_CH_OUT[0],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[0],
            ),
            spnn.BatchNorm(self.ENCODER_CH_OUT[0]),
            spnn.ReLU(True),
        )

        self.block1 = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.ENCODER_CH_IN[1],
                out_channels=self.ENCODER_CH_OUT[1],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[1],
            ),
            spnn.BatchNorm(self.ENCODER_CH_OUT[1]),
            spnn.ReLU(True),
        )

        self.block2 = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.ENCODER_CH_IN[2],
                out_channels=self.ENCODER_CH_OUT[2],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[2],
            ),
            spnn.BatchNorm(self.ENCODER_CH_OUT[2]),
            spnn.ReLU(True),
        )

        self.block3 = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.ENCODER_CH_IN[3],
                out_channels=self.ENCODER_CH_OUT[3],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[3],
            ),
            spnn.BatchNorm(self.ENCODER_CH_OUT[3]),
            spnn.ReLU(True),
        )

        self.block3_tr = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.DECODER_CH_IN[0],
                out_channels=self.DECODER_CH_OUT[0],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-1],
                transposed=True,
            ),
            spnn.BatchNorm(self.DECODER_CH_OUT[0]),
            spnn.ReLU(True),
        )

        self.block2_tr = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.DECODER_CH_IN[1],
                out_channels=self.DECODER_CH_OUT[1],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-2],
                transposed=True,
            ),
            spnn.BatchNorm(self.DECODER_CH_OUT[1]),
            spnn.ReLU(True),
        )

        self.block1_tr = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.DECODER_CH_IN[2],
                out_channels=self.DECODER_CH_OUT[2],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-3],
                transposed=True,
            ),
            spnn.BatchNorm(self.DECODER_CH_OUT[2]),
            spnn.ReLU(True),
        )

        self.conv0_tr = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.DECODER_CH_IN[3],
                out_channels=nfeatures,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-4],
                transposed=True,
            ),
        )

        self.te_head = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=nfeatures,
                out_channels=1,
                kernel_size=self.KERNEL_SIZE,
                stride=1,
                transposed=True,
            ),
        )

    def set_network_properties(self):
        # Encoder network
        for i in range(1, len(self.STRIDE)):
            self.ENCODER_CH_IN[i] = self.ENCODER_CH_OUT[i - 1]
            self.ENCODER_CH_OUT[i] = self.ENCODER_CH_IN[i] * self.STRIDE[i]

        # Decoder network
        self.DECODER_CH_IN = copy.deepcopy(self.ENCODER_CH_OUT)
        self.DECODER_CH_IN.sort(reverse=True)
        self.DECODER_CH_OUT = copy.deepcopy(self.ENCODER_CH_IN)
        self.DECODER_CH_OUT.sort(reverse=True)
        
        # self.DECODER_CH_IN[0] = int(self.ENCODER_CH_OUT[-1])
        # self.DECODER_CH_OUT[0] = int(self.DECODER_CH_IN[0] / self.STRIDE[-1])
        # for i in range(1, len(self.STRIDE)):
        #     self.DECODER_CH_IN[i] = int(self.DECODER_CH_OUT[i - 1]+ self.SKIP_CONNECTION[i - 1] * self.ENCODER_CH_OUT[-1 - i])
        #     self.DECODER_CH_OUT[i] = int(self.DECODER_CH_IN[0] / self.STRIDE[-1 - i])

    def forward(self, x):
        out_s1 = self.block0(x)

        out_s2 = self.block1(out_s1)

        out_s4 = self.block2(out_s2)

        out_s8 = self.block3(out_s4)

        out = self.block3_tr(out_s8)
        # if self.SKIP_CONNECTION[0] > 0:
        # # out = torchsparse.cat([out, out_s4])
        #
        out = self.block2_tr(out)
        # if self.SKIP_CONNECTION[1] > 0:
        # out = torchsparse.cat([out, out_s2])

        out = self.block1_tr(out)
        # if self.SKIP_CONNECTION[2] > 0:
        # out = torchsparse.cat([out, out_s1])

        out_recon = self.conv0_tr(out)

        # Note: We return a sparse tensor without an activation function here. We expect to apply it in the Loss!
        return self.te_head(out_recon), out_recon

    def predict_te(self, x, threshold=0.5):
        return (nn.Sigmoid()(self(x)[0].F) > threshold).int()
