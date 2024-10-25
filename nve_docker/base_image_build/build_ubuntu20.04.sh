#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Base image with most CUDA dependencies installed
UBUNTU_BASE=nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Base image with pytorch
IMG_TORCH=nve_ml_torch
docker build --build-arg BASE_IMAGE=$UBUNTU_BASE -t $IMG_TORCH -f Dockerfile.ml_torch .

IMG_TORCHSPARSE=nve_ml_torchsparse
docker build --build-arg BASE_IMAGE=$IMG_TORCH -t $IMG_TORCHSPARSE -f Dockerfile.ml_torchsparse .

IMG_ML_ROS=nve_ml_ros:frozen
docker build --build-arg BASE_IMAGE=$IMG_TORCHSPARSE -t $IMG_ML_ROS -f Dockerfile.ros .
