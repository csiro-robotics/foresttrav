# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.utils.collate import sparse_collate
from torchsparse.utils.quantize import sparse_quantize


@dataclass
class UNetTHMParams:
    model_name: str
    model_stride: list
    nfeatures: int
    model_nfeature_enc_ch_out: int
    model_skip_connection: list

    def __post_init__(self):
        for attr in self.__annotations__:
            if getattr(self, attr, None) is None:
                raise ValueError(f"Missing required attribute: {attr}")


def predict_te_classification(
    model: nn.Module,
    X_coords: np.array,
    X_features: np.array,
    voxel_size: float,
    device: str,
):
    """Predicts the classification from a TwoHeadModel


    Args:
        model (nn.Module): Deep learning model
        X_coords (np.array): Coordinates of the voxel cloud [x, y, z] in global frame [m] as e 3xn matrix
        X_features (np.array): Feature matrix of the voxel loud [[f_0, ...f_i]_0,...[f_0,....f_i]_n] as a ixn matrix
        voxel_size (float): Leaf size of the voxel in meters
        device (str): Accelerator

    Returns:
        _type_: _description_
    """
    s_coords = X_coords // voxel_size

    # Conversion to SparseTensor
    coords = torch.tensor(s_coords, dtype=torch.int)
    feats = torch.tensor(X_features, dtype=torch.float)
    stensor = SparseTensor(coords=coords, feats=feats)
    stensor = sparse_collate([stensor]).to(device)

    # Model prediction for classificatoin (y_cl) and reconstruction (y_recon)
    y_cl, _ = model(stensor)

    # Binary classification
    _, y_pred_binary = y_cl.F.max(1)

    # Probability logits for class 1 (Traversable)
    y_pred_logits = y_cl.F.softmax(dim=1)[:, 1]

    return y_pred_binary.cpu().numpy(), y_pred_logits.cpu().detach().numpy()


class UNet3THM(nn.Module):
    STRIDE = [1, 2, 2]
    SKIP_CONNECTION = [1, 1]
    KERNEL_SIZE = 3

    # THIS it what is hould like for nfeature_enc_ch_out=8
    ENCODER_CH_IN = [0, 0, 0]
    ENCODER_CH_OUT = [0, 0, 0]

    DECODER_CH_IN = [0, 0, 0]
    DECODER_CH_OUT = [0, 0, 0]

    def __init__(
        self,
        nfeatures: int,
        encoder_nchannels: int = 8,
        skip_conection: list = [1, 1],
        stride: list = [1, 2, 2],
        kernel_size=3,
    ):
        nn.Module.__init__(self)
        self.ENCODER_CH_IN[0] = nfeatures
        self.ENCODER_CH_OUT[0] = encoder_nchannels
        self.SKIP_CONNECTION = skip_conection[:2]
        self.STRIDE = stride[:3]
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

        self.block2_tr_rec = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.ENCODER_CH_OUT[2],
                out_channels=self.ENCODER_CH_IN[2],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-1],
                transposed=True,
            ),
            spnn.BatchNorm(self.ENCODER_CH_IN[2]),
            spnn.ReLU(True),
        )

        self.block1_tr_rec = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.ENCODER_CH_OUT[1],
                out_channels=self.ENCODER_CH_IN[1],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-2],
                transposed=True,
            ),
            spnn.BatchNorm(self.ENCODER_CH_IN[1]),
            spnn.ReLU(True),
        )

        # Last Recon layer must have original input size
        self.block0_tr_rec = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.ENCODER_CH_OUT[0],
                out_channels=self.ENCODER_CH_IN[0],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-3],
                transposed=True,
            ),
        )

        self.block2_tr_cl = torch.nn.Sequential(
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

        self.block1_tr_cl = torch.nn.Sequential(
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

        self.block0_tr_cl = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.DECODER_CH_IN[2],
                out_channels=2,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-3],
                transposed=True,
            ),
        )

    def set_network_properties(self):
        # Encoder network
        for i in range(1, len(self.STRIDE)):
            self.ENCODER_CH_IN[i] = self.ENCODER_CH_OUT[i - 1]
            self.ENCODER_CH_OUT[i] = self.ENCODER_CH_IN[i] * self.STRIDE[i]

        # Decoder network
        self.DECODER_CH_IN[0] = int(self.ENCODER_CH_OUT[-1])
        self.DECODER_CH_OUT[0] = int(self.DECODER_CH_IN[0] / self.STRIDE[-1])
        for i in range(1, len(self.STRIDE)):
            self.DECODER_CH_IN[i] = int(
                self.DECODER_CH_OUT[i - 1]
                + self.SKIP_CONNECTION[i - 1] * self.ENCODER_CH_OUT[-1 - i]
            )
            self.DECODER_CH_OUT[i] = int(self.DECODER_CH_IN[i] / self.STRIDE[-1 - i])

    def forward(self, x):
        out_s1 = self.block0(x)

        out_s2 = self.block1(out_s1)

        out_s4 = self.block2(out_s2)

        # Decoder Start
        out_rec = self.block2_tr_rec(out_s4)
        out_cl = self.block2_tr_cl(out_s4)
        
        if self.SKIP_CONNECTION[1] > 0:
            out_cl = torchsparse.cat([out_cl, out_s2])
            
        out_rec = self.block1_tr_rec(out_rec)
        out_cl = self.block1_tr_cl(out_cl)
        if self.SKIP_CONNECTION[0] > 0:
            out_cl = torchsparse.cat([out_cl, out_s1])

        out_rec = self.block0_tr_rec(out_rec)
        out_cl = self.block0_tr_cl(out_cl)

        return out_cl, out_rec

    def predict_te_classification(
        self,
        X_coords,
        X_features,
        voxel_size,
        te_threshold: float = 0.5,
        device="cuda",
    ):
        return predict_te_classification(
            model=self,
            X_coords=X_coords,
            X_features=X_features,
            voxel_size=voxel_size,
            device=device,
        )


class UNet4THM(nn.Module):
    # THIS it what is hould like for nfeature_enc_ch_out=8
    ENCODER_CH_IN = [0, 0, 0, 0]
    ENCODER_CH_OUT = [0, 0, 0, 0]

    DECODER_CH_IN = [0, 0, 0, 0]
    DECODER_CH_OUT = [0, 0, 0, 0]

    def __init__(
        self,
        nfeatures: int,
        encoder_nchannels: int = 8,
        skip_conection: list = [0, 0, 0],
        stride: list = [1, 2, 2, 2],
        kernel_size=3,
    ):
        nn.Module.__init__(self)
        self.ENCODER_CH_IN[0] = nfeatures
        self.ENCODER_CH_OUT[0] = encoder_nchannels
        self.SKIP_CONNECTION = skip_conection[:3]
        self.STRIDE = stride[:4]
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

        # Reconstruction head f_rec
        self.block3_tr_rec = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.ENCODER_CH_OUT[3],
                out_channels=self.ENCODER_CH_IN[3],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-1],
                transposed=True,
            ),
            spnn.BatchNorm(self.ENCODER_CH_IN[3]),
            spnn.ReLU(True),
        )
        
        self.block2_tr_rec = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.ENCODER_CH_OUT[2],
                out_channels=self.ENCODER_CH_IN[2],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-2],
                transposed=True,
            ),
            spnn.BatchNorm(self.ENCODER_CH_IN[2]),
            spnn.ReLU(True),
        )
        
        self.block1_tr_rec = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.ENCODER_CH_OUT[1],
                out_channels=self.ENCODER_CH_IN[1],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-3],
                transposed=True,
            ),
            spnn.BatchNorm(self.ENCODER_CH_IN[1]),
            spnn.ReLU(True),
        )
        
        self.conv0_tr_rec = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.ENCODER_CH_OUT[0],
                out_channels=self.ENCODER_CH_IN[0],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-4],
                transposed=True,
            ),
        )

        # TE head f_te
        self.block3_tr_cl = torch.nn.Sequential(
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
       
        self.block2_tr_cl = torch.nn.Sequential(
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
        
        self.block1_tr_cl = torch.nn.Sequential(
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
        
        self.conv0_tr_cl = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.DECODER_CH_IN[3],
                out_channels=2,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-4],
                transposed=True,
            ),
        )

    def set_network_properties(self):
        # Encoder network
        for i in range(1, len(self.STRIDE)):
            self.ENCODER_CH_IN[i] = self.ENCODER_CH_OUT[i - 1]
            self.ENCODER_CH_OUT[i] = self.ENCODER_CH_IN[i] * self.STRIDE[i]

        # Decoder network
        self.DECODER_CH_IN[0] = int(self.ENCODER_CH_OUT[-1])
        self.DECODER_CH_OUT[0] = int(self.DECODER_CH_IN[0] / self.STRIDE[-1])
        for i in range(1, len(self.STRIDE)):
            self.DECODER_CH_IN[i] = int(
                self.DECODER_CH_OUT[i - 1]
                + self.SKIP_CONNECTION[i - 1] * self.ENCODER_CH_OUT[-1 - i]
            )
            self.DECODER_CH_OUT[i] = int(self.DECODER_CH_IN[i] / self.STRIDE[-1 - i])

        b = 1

    def forward(self, x):

        # Decoder
        out_s1 = self.block0(x)
        out_s2 = self.block1(out_s1)
        out_s4 = self.block2(out_s2)
        out_s8 = self.block3(out_s4)

        out_rec = self.block3_tr_rec(out_s8)
        out_cl = self.block3_tr_cl(out_s8)
        if self.SKIP_CONNECTION[0] > 0:
            out_cl = torchsparse.cat([out_cl, out_s4])

        out_rec = self.block2_tr_rec(out_rec)
        out_cl = self.block2_tr_cl(out_cl)
        if self.SKIP_CONNECTION[1] > 0:
            out_cl = torchsparse.cat([out_cl, out_s2])

        out_rec = self.block1_tr_rec(out_rec)
        out_cl = self.block1_tr_cl(out_cl)
        
        if self.SKIP_CONNECTION[2] > 0:
            out_cl = torchsparse.cat([out_cl, out_s1])

        out_rec = self.conv0_tr_rec(out_rec)
        out_cl = self.conv0_tr_cl(out_cl)

        return out_cl, out_rec

    def predict_te_classification(
        self,
        X_coords,
        X_features,
        voxel_size,
        te_threshold: float = 0.5,
        device="cuda",
    ):
        return predict_te_classification(
            model=self,
            X_coords=X_coords,
            X_features=X_features,
            voxel_size=voxel_size,
            device=device,
        )
