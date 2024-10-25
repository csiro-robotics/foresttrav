# DOCKER NVE WS

## Overview
Docker usage for the ML application for Navigation in Vegetated Environments (NVE).
The package contains build files to generate a base image for amd64 and nvidia arm
platroms ( jestson or orin). 

## Ubuntu 20.04 amd64 image:
Build the image in `base_image_build` directory. 

`sudo sh ./nve_docker/base_image_build/build_ubuntu20.04.sh'

The base image created should be called 'nve_ml_base:lastest`

## Building the arm base image
The arm base image is build on the dusty repo and adde

`sudo sh ./nve_docker/base_image_build/build_orin.sh'

The resulting base image sould be called 'jts_ml_base:latest

## Building deployment

Before building check that your parent image is correctly done

### Building it

We use multi-stage Docker builds to optimise the build process.
Use `export DOCKER_IMAGE=<tag>` to manually specify a custom image name.
Run the the following steps from the root of the repository. The context
of the docker image is build in the root of the workspace.

1. Build the frozen-stage: `nve_docker/build_image.bash --target frozen_stage .`
2. Run the deps script: `nve_docker/run_deps.bash`
3. Build the base-stage: `nve_docker/build_image.bash --target base_stage .`

The base-stage image is sufficient for development as it contains all project
dependencies. For deployment, we add the following steps:

4. Run the build script: `nve_docker/run_build.bash`
5. Run the tests script (optional): `docker/run_tests.bash`
6. Build the prod-stage: `nve_docker/build_image.bash --target prod_stage .`

This prod-stage image contains all project binaries under /refarm/install.

## FAQ:

### The jetsin script is failing:
Check that the 'PARENT_IMG' in 'build_orin.sh' is the same as what you generated from the duyst repo.
Versions are known to change depending on the jetson version. 

### My GPU does not seem supported:
Check the 'TORCH_CUDA_ARCH_LIST' in the 'Dockerfile.orin' ot the ' Dockerfile.ml_torch'. The [link](https://developer.nvidia.com/cuda-gpus) will tell you
if you need to add your GPU to the list (check the numbers).

### I want to use a different ubuntu version?
Try changing the parent image in the base build file and hope for the best. Good luck! 
