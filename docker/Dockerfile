ARG BASE_IMAGE=nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
# ARG BASE_IMAGE
FROM ${BASE_IMAGE} as nve_ml_base

ENV DEBIAN_FRONTEND=noninteractive

# Hack since certificate of docker image seems to be out of date?
# This (hopefully) will be deprecated once they solved the issues on nvidia side
# RUN  rm /etc/apt/sources.list.d/cuda.list && apt-key del 7fa2af80 

# Install language 
RUN apt-get update &&  apt-get install -y \
   locales \
   && locale-gen en_US.UTF-8 \
   && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 

ENV LANG en_US.UTF-8

# Install timezone
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
   && export DEBIAN_FRONTEND=noninteractive \
   && apt-get update && apt-get install -y \
   tzdata \
   && dpkg-reconfigure --frontend noninteractive tzdata 

##############################################
# You should modify this to match your GPU compute capability
# https://developer.nvidia.com/cuda-gpus
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
##############################################

# Install development dependencies
RUN apt-get update && apt-get --assume-yes install \
   software-properties-common \
   bash-completion \
   build-essential \
   git \
   ninja-build \ 
   cmake\ 
   libopenblas-dev \
   libsparsehash-dev \
   xterm \
   xauth \
   openssh-server \ 
   tmux \
   wget \
   curl \
   python3-pip \
   mate-desktop-environment-core

# Upgrade pip version
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN apt install python3-pip -y \
   && pip3 install --upgrade pip 
# && pip3 install ingnore-installed pyyaml click

FROM nve_ml_base as nve_ml_pytorch
# Correct pytorch version for torchsparse v2.0.0b
RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 pytorch-lightning
RUN pip3 install --ignore-installed blinker pyyaml click
RUN pip3 install matplotlib \
   pillow \
   numpy \
   scipy \
   cython \
   scikit-image \
   scikit-learn \
   scikit-learn-intelex \
   opencv-python \
   open3d \
   h5py \
   easydict \
   tabulate \
   wandb


FROM nve_ml_pytorch as nve_ml_torchsparse
###########################################
#  TORCHSPARSE
###########################################
# For faster build, use more jobs.
ENV MAX_JOBS=8

# TorchSpare version v2.0.0b commit, note if you dont want to use CUDA, set FORCE_CUDA=0
RUN apt-get update && apt-get install libsparsehash-dev \
   && git clone --branch v2.0.0 --recurse-submodules https://github.com/mit-han-lab/torchsparse.git  /opt/torchsparse \
   && cd /opt/torchsparse \
   && FORCE_CUDA=1 pip install . 


# ###########################################
# #  ROS INSTALL
# ###########################################
FROM dev_nve_noetic_headless AS dev_nve_noetic

# Install ROS
RUN apt-get update && apt-get install -y \
   curl \
   gnupg2 \
   lsb-release \
   sudo \
   && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -\
   && echo "deb http://packages.ros.org/ros/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/ros-focal.list \
   && apt-get update && apt-get install -y \
   ros-noetic-desktop-full \
   python3-argcomplete \
   && apt-get autoremove -y  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
   bash-completion \
   build-essential \
   cmake \
   gdb \
   git \
   pylint3 \
   python3-argcomplete \
   python3-colcon-common-extensions \
   python3-rosdep \
   python3-vcstool \
   python3-rosinstall \
   python3-wstool \
   vim \
   wget \
   htop \
   tmux \
   ninja-build \
   libopenblas-dev \
   libsparsehash-dev \
   ros-noetic-rviz \
   ros-noetic-gazebo-ros-pkgs \
   ros-noetic-perception-pcl \
   ros-noetic-navigation \ 
   libgl1-mesa-dev \
   ros-noetic-grid-map \
   fish \
   zlib1g-dev\ 
   libglm-dev\
   libglew-dev\ 
   libglfw3-dev\ 
   libpng-dev\ 
   libeigen3-dev\ 
   googletest\
   opencl-headers\ 
   ocl-icd-dev\ 
   ocl-icd-libopencl1\ 
   ocl-icd-opencl-dev\
   intel-opencl-icd\
   libtbb-dev \ 
   libpdal-dev\
   libopenblas-dev \ 
   # Install ros distro testing packages
   python3-autopep8 \
   && rosdep init || echo "rosdep already initialized" \
   && apt-get autoremove -y  && apt-get clean -y && rm -rf /var/lib/apt/lists/*


#
# OpenCL dependencies for OHM
#
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
   && add-apt-repository ppa:intel-opencl/intel-opencl \
   && apt-get update \
   && apt-get install --assume-yes --no-install-recommends intel-opencl-icd \
   && rm -rf /var/lib/apt/lists/* \
   && mkdir -p /etc/OpenCL/vendors \
   && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Install the missing python packages
RUN pip3 install colcon-mixin rosbags 



###########################################
#  Add user and bash file source
#     This is done in the .devcontainer/Docker file and should be toggled on if the user wishes to use without vscode
###########################################
# ARG USERNAME=ros
# ARG USER_UID=1000
# ARG USER_GID=$USER_UID

# # Create a non-root user
# RUN groupadd --gid $USER_GID $USERNAME \
#    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID --groups video -m $USERNAME \
#    # [Optional] Add sudo support for the non-root user
#    && apt-get update && apt-get install -y sudo \
#    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
#    && chmod 0440 /etc/sudoers.d/$USERNAME \
#    && echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc 


# USER $USERNAME

# ENV WROKSPACE_HOME=/nve_ws

# # Adding aliases for easier life and running it
# RUN echo " Adding aliases to bashrc" \
#    && echo "alias snoe='source /opt/ros/noetic/setup.bash'"     >> /home/$USERNAME/.bashrc \
#    && echo "alias sros='snoe && source  /nve_ws/install/setup.bash'" >> /home/$USERNAME/.bashrc \
#    && echo "if [ -f /opt/ros/noetic/setup.bash ]; then source /opt/ros/noetic/setup.bash; fi" >> /home/$USERNAME/.bashrc \
#    && echo "if [ -f /nve_ws/install/setup.bash ]; then source /nve_ws/install/setup.bash; fi" >> /home/ros/.bashrc 