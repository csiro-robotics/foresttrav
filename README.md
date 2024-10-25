# Navigation in Vegetated environments (NVE)

This is a meta repo for navigation in vegetated environment workspace

# Overview

This workspace depends on ros noetic and combines functionally the packages required to enable navigation in vegetated environments

The primary target is ROS noetic on Unbuntu 20.04, am64 and used CUDA > 11.7.nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04  

The pakcages are managed using vcs-tools. We avoid the use of submodules on purpose for this repo. 

## Docker Image

<!-- ## QUICKSTART
1. Install host dependencies
> # Docker (https://docs.docker.com/get-docker/)
curl https://get.docker.com | sh
sudo systemctl --now enable docker
sudo usermod -aG docker $USER
newgrp docker

# Nvidia runtime for Docker (if you have an Nvidia GPU w/ installed drivers) (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) 
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# VSCode (https://code.visualstudio.com/docs/setup/setup-overview)
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg
sudo apt install apt-transport-https
sudo apt update
sudo apt install code
We develop in a docker container via vscode. Read about it [here](https://code.visualstudio.com/docs/remote/containers). 

This means that you shouldn't have to worry about versioning, OS, or software setup (theoretically). You will only need docker and vscode. 

2. Clone the project dependencies using vcs-tools

3. Start development

## Repo Setup
Clone the repo and 
```bash
git submodule update --init --recursive
```

1. Install [Docker](https://docs.docker.com/get-docker/) are installed and the user has been added to the docker group (usually requires a restart).

1. Install [VSCode](https://code.visualstudio.com/).

1. Then open VSCode and follow the prompts to install the Remote-Containers extension. 
   
   ![recommend_extensions](doc/recomend_extensions.png) 
   
   and follow the prompts to build the container.

   ![open_container](doc/open_container.png)

1. Pro tip. Use the terminals in VS code for running stuff. Use a local terminal for git stuff.


 -->
