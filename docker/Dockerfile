##############################################
# Created from template ros2.dockerfile.jinja
##############################################

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 AS base_nve_noetic
ENV DEBIAN_FRONTEND=noninteractive

# Install language
RUN apt-get update && apt-get install -y \
  locales \
  && locale-gen en_US.UTF-8 \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
  && apt-get autoremove -y  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV LANG en_US.UTF-8

# Install timezone
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
  && export DEBIAN_FRONTEND=noninteractive \
  && apt-get update && apt-get install -y \
  tzdata \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && apt-get autoremove -y  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Define ros distros
ENV ROS1_DISTRO=noetic

# Install ROS
RUN apt-get update && apt-get install -y \
   curl \
   gnupg2 \
   lsb-release \
   sudo \
   && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -\
   && echo "deb http://packages.ros.org/ros/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/ros-focal.list \
   && apt-get update && apt-get install -y \
   ros-${ROS1_DISTRO}-desktop-full \
   python3-argcomplete \
   && apt-get autoremove -y  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=

# ###########################################
# #  Headless development 
# ###########################################
FROM base_nve_noetic AS dev_nve_noetic_headless

RUN apt-get update && apt-get install -y \
   bash-completion \
   build-essential \
   cmake \
   gdb \
   git \
   pylint3 \
   python3-argcomplete \
   python3-colcon-common-extensions \
   python3-pip \
   python3-rosdep \
   python3-vcstool \
   python3-rosinstall \
   python3-wstool \
   vim \
   wget \
   htop \
   tmux \
   # Install ros distro testing packages
   python3-autopep8 \
   && rosdep init || echo "rosdep already initialized" \
   && apt-get autoremove -y  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create a non-root user
RUN groupadd --gid $USER_GID $USERNAME \
   && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID --groups video -m $USERNAME \
   # [Optional] Add sudo support for the non-root user
   && apt-get update && apt-get install -y sudo \
   && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
   && chmod 0440 /etc/sudoers.d/$USERNAME \
   && rm -rf /var/lib/apt/lists/* \
   && echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc 

# ###########################################
# #  ADD ME
# ###########################################


ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN apt-get  update && apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN pip3 install torch torchvision torchaudio  nvidia-pyindex pytorch-lightning

RUN export CUDA_HOME=/usr/local/cuda && pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                          --install-option="--force_cuda" \
                          --install-option="--blas=openblas" 
# RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
# RUN cd MinkowskiEngine; python setup.py install --force_cuda --blas=openblas

# ###########################################
# #  GUI development 
# ###########################################
FROM dev_nve_noetic_headless AS dev_nve_noetic

RUN apt-get update && apt-get -y install --no-install-recommends \
   ros-${ROS1_DISTRO}-rviz \
   ros-${ROS1_DISTRO}-gazebo-ros-pkgs \
   ros-${ROS1_DISTRO}-perception-pcl \
   ros-${ROS1_DISTRO}-navigation \ 
   libgl1-mesa-dev \
   fish \ 
   && apt-get autoremove -y  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# See requirements https://bitbucket.csiro.au/projects/NAV/repos/ohm/browse/OpenCL.md
RUN apt-get update && apt-get -y install --no-install-recommends \
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
   && apt-get autoremove -y  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# TODO: Remove (this is for testing)
RUN apt-get update && apt-get -y install --no-install-recommends \
   mesa-utils \
   clinfo \
   && apt-get autoremove -y  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
   && apt-get upgrade --assume-yes --no-install-recommends \
   && apt-get install --assume-yes --no-install-recommends \
   software-properties-common \
   && add-apt-repository ppa:intel-opencl/intel-opencl \
   && apt-get update \
   && apt-get install --assume-yes --no-install-recommends intel-opencl-icd \
   && rm -rf /var/lib/apt/lists/* \
   && mkdir -p /etc/OpenCL/vendors \
   && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Install Machine Learning python packafges
RUN pip3 install colcon-mixin rosbags matplotlib pandas scikit-learn opencv-python scikit-learn-intelex wandb open3d tabulate rosnumpy
RUN pip3 install --upgrade numpy click


# Adding aliases for easier life and running it
ARG USERNAME=ros
RUN echo " Adding aliases to bashrc" \
   && echo "alias snoe='source /opt/ros/noetic/setup.bash'"     >> /home/$USERNAME/.bashrc \
   && echo "alias sros='snoe && source  /nve_ws/install/setup.bash'" >> /home/$USERNAME/.bashrc \
   && echo "if [ -f /opt/ros/${ROS1_DISTRO}/setup.bash ]; then source /opt/ros/${ROS1_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc \
   && echo "if [ -f /nve_ws/install/setup.bash ]; then source /nve_ws/install/setup.bash; fi" >> /home/ros/.bashrc 

ENV COLCON_HOME=/nve_ws/.colcon
