# Copyright (c) 2024
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.utils.collate import sparse_collate


def predict_te_classification(    
    model: nn.Module,
    X_coords: np.array,
    X_features: np.array,
    voxel_size: float,
    device: str,):
    ''' Predicts a Unet model for infrence'''

    s_coords = X_coords // voxel_size
    
    coords = torch.tensor(s_coords, dtype=torch.int)
    feats =  torch.tensor(X_features, dtype=torch.float)
    stensor = SparseTensor(coords=coords, feats=feats)
    stensor = sparse_collate([stensor]).to(device)
    
    y_cl = model(stensor) 
    _, y_pred_binary = y_cl.F.max(1)
    y_pred_logits = y_cl.F.softmax(dim=1)[:, 1]
    
    return   y_pred_binary.cpu().numpy(), y_pred_logits.cpu().detach().numpy()
    
@dataclass
class UNetMCDParams:
    model_name: str
    model_stride: list 
    nfeatures: int
    model_nfeature_enc_ch_out: int 
    model_skip_connection: list
    
    def __post_init__(self):
        for attr in self.__annotations__:
            if getattr(self, attr, None) is None:
                raise ValueError(f"Missing required attribute: {attr}")


class UNet5LMCD(nn.Module):
    STRIDE = [1, 2, 2, 2, 2]
    SKIP_CONNECTION = [1, 1, 1, 1]
    KERNEL_SIZE = 3

    # THIS it what is hould like for encoder_nchannels=8
    ENCODER_CH_IN = [0, 0, 0, 0, 0]
    ENCODER_CH_OUT = [0, 0, 0, 0, 0]

    DECODER_CH_IN = [0, 0, 0, 0, 0]
    DECODER_CH_OUT = [0, 0, 0, 0, 0]

    def __init__(
        self,
        nfeatures: int,
        encoder_nchannels: int = 8,
        skip_conection: list = [1, 1, 1, 1],
        stride: list = [1, 2, 2, 2, 2],
        D: int = 3,
        out_nchannel: int = 2,
    ):
        nn.Module.__init__(self)
        self.ENCODER_CH_IN[0] = nfeatures
        self.ENCODER_CH_OUT[0] = encoder_nchannels
        self.SKIP_CONNECTION = skip_conection[:4]
        self.STRIDE = stride[:5]

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

        self.block4 = torch.nn.Sequential(
 
            spnn.Conv3d(
                in_channels=self.ENCODER_CH_IN[4],
                out_channels=self.ENCODER_CH_OUT[4],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[4],
            ),
            spnn.BatchNorm(self.ENCODER_CH_OUT[4]),
            spnn.ReLU(True),
        )

        self.block4_tr = torch.nn.Sequential(
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

        # 64 + 64 = 128
        self.block3_tr = torch.nn.Sequential(
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

        # 64 + 32 = 92 / 2 = 46
        self.block2_tr = torch.nn.Sequential(
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

        self.block1_tr = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.DECODER_CH_IN[3],
                out_channels=self.DECODER_CH_OUT[3],
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-4],
                transposed=True,
            ),
            spnn.BatchNorm(self.DECODER_CH_OUT[3]),
            spnn.ReLU(True),
        )

        self.conv0_tr = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.DECODER_CH_IN[4],
                out_channels=out_nchannel,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE[-5],
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

        out_s8 = self.block3(out_s4)

        out_s16 = self.block4(out_s8)

        # Decoder
        out = self.block4_tr(out_s16)
        if self.SKIP_CONNECTION[0] > 0:
            out = torchsparse.cat([out, out_s8])

        out = self.block3_tr(out)
        if self.SKIP_CONNECTION[1] > 0:
            out =torchsparse.cat([out, out_s4])

        out = self.block2_tr(out)
        if self.SKIP_CONNECTION[2] > 0:
            out = torchsparse.cat([out, out_s2])

        out = self.block1_tr(out)
        if self.SKIP_CONNECTION[3] > 0:
            out = torchsparse.cat([out, out_s1])

        return self.conv0_tr(out)

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


class UNet4LMCD(nn.Module):
    STRIDE = [1, 2, 2, 2]
    SKIP_CONNECTION = [1, 1, 1]
    KERNEL_SIZE = 3

    # THIS it what is hould like for encoder_nchannels=8
    ENCODER_CH_IN = [0, 0, 0, 0]
    ENCODER_CH_OUT = [0, 0, 0, 0]

    DECODER_CH_IN = [0, 0, 0, 0]
    DECODER_CH_OUT = [0, 0, 0, 0]

    def __init__(
        self,
        nfeatures: int,
        out_nchannel: int = 2 ,
        encoder_nchannels: int = 8,
        skip_conection: list = [1, 1, 1],
        stride: list = [1, 2, 2, 2],
        D: int = 3,
    ):
        nn.Module.__init__(self)
        self.ENCODER_CH_IN[0] = nfeatures
        self.ENCODER_CH_OUT[0] = encoder_nchannels
        self.SKIP_CONNECTION = skip_conection[:3]
        self.STRIDE = stride[:4]

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
                out_channels=out_nchannel,
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

    def forward(self, x):

        out_s1 = self.block0(x)

        out_s2 = self.block1(out_s1)

        out_s4 = self.block2(out_s2)

        out_s8 = self.block3(out_s4)

        out = self.block3_tr(out_s8)
        if self.SKIP_CONNECTION[0] > 0:
            out = torchsparse.cat([out, out_s4])

        out = self.block2_tr(out)
        if self.SKIP_CONNECTION[1] > 0:
            out = torchsparse.cat([out, out_s2])

        out = self.block1_tr(out)
        if self.SKIP_CONNECTION[2] > 0:
            out = torchsparse.cat([out, out_s1])

        return self.conv0_tr(out)

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


class UNet3LMCD(nn.Module):
    STRIDE = [1, 2, 2]
    SKIP_CONNECTION = [1, 1]
    KERNEL_SIZE = 3

    # THIS it what is hould like for encoder_nchannels=8
    ENCODER_CH_IN = [0, 0, 0]
    ENCODER_CH_OUT = [0, 0, 0]

    DECODER_CH_IN = [0, 0, 0]
    DECODER_CH_OUT = [0, 0, 0]

    def __init__(
        self,
        nfeatures: int,
        D: int = 3,
        encoder_nchannels: int = 8,
        skip_conection: list = [1, 1],
        stride: list = [1, 2, 2],
        out_nchannel: int = 2,
    ):
        nn.Module.__init__(self)
        self.ENCODER_CH_IN[0] = nfeatures
        self.ENCODER_CH_OUT[0] = encoder_nchannels
        self.SKIP_CONNECTION = skip_conection[:2]
        self.STRIDE = stride[:3]

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


        self.block2_tr = torch.nn.Sequential(
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

        self.block1_tr = torch.nn.Sequential(
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

        self.block0_tr = torch.nn.Sequential(
            spnn.Conv3d(
                in_channels=self.DECODER_CH_IN[2],
                out_channels=out_nchannel,
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

        out = self.block2_tr(out_s4)
        if self.SKIP_CONNECTION[0] > 0:
            out = torchsparse.cat([out, out_s2])

        out = self.block1_tr(out)
        if self.SKIP_CONNECTION[1] > 0:
            out = torchsparse.cat([out, out_s1])

        return self.block0_tr(out)
    
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


SCNN_MODEL_MAP = {
    'UNet3LMCD': UNet3LMCD,
    'UNet4LMCD': UNet4LMCD,
    'UNet5LMCD': UNet5LMCD,
}
