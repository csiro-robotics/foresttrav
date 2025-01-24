# Navigation in Vegetated environments (NVE)

NVE is a mono repo for the for traverasbilit estimation (TE) using 3D probalistic voxels and online adaptive TE estimation.

# Overview

This workspace depends on ros noetic and combines functionally the packages required to enable navigation in vegetated environments for the ForestTrav method.

The primary targets platform are ROS noetic on Unbuntu 20.04, amd64, CUDA 11.7 and TorchSparse. This packages requires an nvidia GPU.

# Installation

This repo relies on docker and `VSCode` with `Dev Containter` addon to run. Within the docker container, the main workspace will be under `/foresttrav_ws`. The data or training data is assumed to be under `/data`. 
- [VSCode](https://code.visualstudio.com/)
  - [DevContainer](https://code.visualstudio.com/docs/devcontainers/containers)
- [Nvidia Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker](https://docs.docker.com/engine/install/)


## Pulling the repository
This repository uses submodules. Use the following command to pull the repo with all the correct modules and commits.
```bash
git clone --recursive https://github.com/csiro-robotics/foresttrav.git
```

## Building the docker image and the source code
Firtst, navigate to the sub-directory `docker` inside the repo and build the base image using `docker build . ` command. 

Sedondly, build the docker image using vscode using `Dev Container`
1. Open vscode with , `CTRL-P` and choose `Dev Conainters: Build and Open`
2. In the root directory of the repo build it using `colcon` with the following command or use the `build task`, `CTRL-SHIFT-B` and `Build`:
  
   ```bash
   colcon build --symlink-install --merge-install --cmake-args
   -DCMAKE_BUILD_TYPE=Release' -Wall -Wextra -Wpedantic`
   ```

The source code will be build inside the container and persist within the mounted repo.

## Usage of ForestTrav 
ForestTrav models rely on the [`ForestTrav Data Set`](https://data.csiro.au/collection/csiro:58941) and the [ForestTrav Rosbags](https://data.csiro.au/collection/csiro:58941)

Recommendation is to download the `lfe_hl_0.1` data set. This is the data fusing self-supervised labelling of the robot with hand-labelling at 0.1m voxel resolution. 

### How to train a new model
To train a new model, use the `train_model.py` in `odap_tm` package. All of the configurations are stored in `odap_tm/config/default_model_train.yaml`


### How to run a ForestTrav online with a rosbag
To run a new model, use the `te_estimator.launch` from the `nve_startup` pkg. The ForestTrav Rosbags can be found [here](https://data.csiro.au/collection/csiro:58941)
In therminal 1 run:
```bash
roslaunch nve_startup te_estimator.launch
```
In terminal 2 navigate into the directory of the data sets and play the rosbag:
```bash
rosbag play *
```
To view the robot model and the point cloud run 
```bash
roslaunch nve_startup show_squash.launch
```

# Assumptions about data and representation:
For all the packages, the map or world representation is assumed to be static and grid aligned world frame. Further, each voxel is assumed to contain one, and only one point or distribution.

# Refrence

Please cite ForestTrav paper if you are using the traversbiliy estimation or the the ForestTrav data set. 

```latex
@article{ruetz2024foresttrav,
  title={ForestTrav: 3D LiDAR-only forest traversability estimation for autonomous ground vehicles},
  author={Ruetz, Fabio and Lawrance, Nicholas and Hern{\'a}ndez, Emili and Borges, Paulo and Peynot, Thierry},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```
Please cite XXXX if you are using the online ROS-bags or the online adaptive traversability estmimation extension for ForestTrav.
```latex
@article{ruetz2024adaptive,
  title={Adaptive Online Traversability Estimation For Unstrucutred, Densely Vegetated Environemtns},
  author={Ruetz, Fabio and Lawrance, Nicholas and Hern{\'a}ndez, Emili and Borges, Paulo and Peynot, Thierry},
  journal={preprint},
  year={2024},
  publisher={-}

}
```

# Main Dependencies to run without Docker
The following are the main dependencies required to be installed if one would like to run without the docker container, and wont be supported.
- Ubuntu 20.04
- CUDA 11.7.1
- Pytorch 11.8
- Pytorch Lightning 
- TorchSparse 2.0.0b
- OHM: Occupancy Homogenous Mapping

We use docker and the nvidia image `11.7.nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04` to build ontop of it. 