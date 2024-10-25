#!/bin/bash

Clone the repo 
git clone https://github.com/dusty-nv/jetson-containers.git /opt/jetson-containers
cd  /opt/jetson-containers

# Check if build.sh exists and is executable
if [ -x "build.sh" ]; then
    # Run the build script
    IMG_JTS= nve_ml_jts
    ./build.sh  --name=$IMG_JTS torch:1.13  ros:noetic-ros-base
else
    echo "build.sh not found or not executable"
    exit 1
fi

# Note: This may not work. Check the las docker image and build with the command below manually
PARENT_IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | head -n 1)

cd "$(dirname "$0")"
if [ -f "Dockerfile.ml_orin" ]; then
    IMG_TORCHSPARSE=jts_nve_ml_ros:base
    docker build --build-arg BASE_IMAGE=$PARENT_IMG -t $IMG_TORCHSPARSE -f Dockerfile.ml_orin .
else 
    echo "could not find local Dockerfile"
fi