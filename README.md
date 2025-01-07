# Navigation in Vegetated environments (NVE)

NVE is a mono repo for the for traverasbilit estimation (TE) using 3D probalistic voxels and online adaptive TE estimation.

# Overview

This workspace depends on ros noetic and combines functionally the packages required to enable navigation in vegetated environments

The primary target is ROS noetic on Unbuntu 20.04, amd64, CUDA 11.7 and TorchSparse

The pakcages are managed using vcs-tools. We avoid the use of submodules on purpose for this repo. 

## Main Dependencies
- Ubuntu 20.04
- CUDA 11.7.1
- Pytorch 11.8
- Pytorch Lightning 
- TorchSparse 2.0.0b
- OHM: Occupancy Homogenous Mapping

We use docker and the nvidia image `11.7.nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04` to build ontop of it. 

# Table of Contents
- [Paper and Video](#paper-and-video)
- [Instalation](#installation)  
- [Training a new model](#running-model)


## Installation and Building the Code
This repo relies on docker and `vscode` with `Dev Containter` addon to run. Within the docker container, the main workspace will be under `/foresttrav_ws`. The data or training data is assumed to be under `/foresttrav_ws/data`. 



### Pulling the repository
This repository uses submodules. Use the following command to pull the repo with all the correct modules and commits.
```bash
git clone --recursive https://github.com/csiro-robotics/foresttrav.git
```

### Building the docker image and source rep
1. Open vscode, `CTRL-P` and choose `Dev Conainters: Build and Open`
2. In the root directory of the repo build it using `colcon` with the following command or use the `build task`, `CTRL-SHIFT-B` and `Build`:
  
   ```bash
   colcon build --symlink-install --merge-install --cmake-args
   -DCMAKE_BUILD_TYPE=Release' -Wall -Wextra -Wpedantic`
   ```


The source code will be build inside the container and persist within the mounted repo.
## Train and Run ForestTrav Models 
ForestTrav models rely on the ForestTrav data set, available here. Recomendation is to download the `lfe_hl_0.1` data set. This is the data fusing self-supervised labelling of the robot with hand-labelling at 0.1m voxel resolution. 

### How to train a new model
To train a new model, use the `train_model.py` in `scnn_tm` package. All of the configurations are stored in `scnn_tm/config/default_model_train.yaml`

##$ How to run a ForestTrav model
To run a new model, use the `te_estimation.launch` from the `nve_ws` pkg. 

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

