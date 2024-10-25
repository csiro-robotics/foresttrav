#!/usr/bin/env bash
set -e

# This script is used to run this project's Docker image optionally with X
# forwarding and nvidia-docker2 support.
# All arguments to this script will be appended to a docker run command.
# Environment vars that will be read:
#   $DOCKER_ARGS: array containing additional Docker arguments (will be split on " ")
#   $DOCKER_IMAGE: Docker image to run
#   $DOCKER_NAME: value of `docker run`'s `--name` argument
#   $DOCKER_NETWORK: value of `docker run`'s `--network` argument (default is "host")
#   $NVIDIA: set to "false" to not try to use nvidia-docker2
#   $X_FORWARDING: set to "false" to not try to connect to the X server
#
# For an alternative approach, see rocker (https://github.com/osrf/rocker)
#
# Example command line:
# ./run.bash /bin/bash

# Get the path to this script
SCRIPTPATH="$(dirname "$0")"

# Read environment vars
IFS=" " read -a DOCKER_ARGS <<< "$DOCKER_ARGS"

# Image tag
if [ -z "$DOCKER_IMAGE" ]; then
  # Use branch's Docker image if it exists or just try to use latest
  DOCKER_IMAGE="jts_nve_ml_ros"
  # Docker image tags are the branch with any illegal characters replaced with
  # underscores, use sed to generate this
  DOCKER_TAG="$(cd "$SCRIPTPATH"; git rev-parse --abbrev-ref HEAD | sed -E s/[^a-zA-Z0-9_\.\-]/_/g || echo latest)"
  if docker inspect "$DOCKER_IMAGE:$DOCKER_TAG" > /dev/null 2>&1; then
    DOCKER_IMAGE="$DOCKER_IMAGE:$DOCKER_TAG"
  else
    DOCKER_IMAGE="$DOCKER_IMAGE:latest"
  fi
fi

# Container name
if [ -n "$DOCKER_NAME" ]; then
  DOCKER_ARGS+=("--name")
  DOCKER_ARGS+=("$DOCKER_NAME")
fi

# Docker network argument
if [ -z "$DOCKER_NETWORK" ]; then
  DOCKER_NETWORK="host"
fi
DOCKER_ARGS+=("--network")
DOCKER_ARGS+=("$DOCKER_NETWORK")

# Hostname
if [ -n "$DOCKER_HOSTNAME" ] ; then
  DOCKER_ARGS+=("--hostname")
  DOCKER_ARGS+=("$DOCKER_HOSTNAME")
fi

# IP
if [ -n "$DOCKER_IP" ]; then
  DOCKER_ARGS+=("--ip")
  DOCKER_ARGS+=("$DOCKER_IP")
fi

# Start interactive if stdin is a terminal
if [ -t 0 ]; then
  DOCKER_ARGS+=("--interactive")
  DOCKER_ARGS+=("--tty")
fi

# NVIDIA
if [ "$NVIDIA" != "false" ]; then
  NVIDIA="true"
fi
if [ "$NVIDIA" == "true" ]; then
  # Use lspci to check for the presence of a NVIDIA graphics card
  HAS_NVIDIA="$(lspci | grep -i nvidia | wc -l)"

  # Check if nvidia-docker2 is installed
  HAS_NVIDIA_DOCKER2="$(dpkg -l | grep nvidia-docker2 > /dev/null 2>&1; echo $?)"

  # Features of nvidia container toolkit have replaced nvidia-docker2
  HAS_NVIDIA_CONTAINER_TOOLKIT="$(dpkg -l | grep nvidia-container-toolkit | wc -l)"

  if test ${HAS_NVIDIA_DOCKER2} -eq 0 || test ${HAS_NVIDIA_CONTAINER_TOOLKIT} -gt 0; then
    HAS_NVIDIA_CONTAINER_SUPPORT="true"
  else
    HAS_NVIDIA_CONTAINER_SUPPORT="false"
  fi

  # If both check out, use nvidia's container support
  if [ ${HAS_NVIDIA} -gt 0 ] && [ "${HAS_NVIDIA_CONTAINER_SUPPORT}" = "true" ]; then
    DOCKER_ARGS+=("--env"); DOCKER_ARGS+=("NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics")
    DOCKER_ARGS+=("--env"); DOCKER_ARGS+=("NVIDIA_VISIBLE_DEVICES=all")
    DOCKER_ARGS+=("--gpus"); DOCKER_ARGS+=("all")
    DOCKER_ARGS+=("--runtime"); DOCKER_ARGS+=("nvidia")

    # Force usage of discrete gpu when using nvidia-prime in 'on-demand' mode. Setting these variables in other modes
    # or gpu configurations may cause crashes.
    if [ -f /etc/prime-discrete ] && grep "on-demand" /etc/prime-discrete > /dev/null 2>&1 ; then
      DOCKER_ARGS+=("--env"); DOCKER_ARGS+=("__NV_PRIME_RENDER_OFFLOAD=1")
      DOCKER_ARGS+=("--env"); DOCKER_ARGS+=("__GLX_VENDOR_LIBRARY_NAME=nvidia")
    fi

    # Mount CUDA toolkit if it exists
    if [ -z "$CUDA_TOOLKIT_ROOT_DIR" ]; then
      CUDA_TOOLKIT_ROOT_DIR="$(realpath /etc/alternatives/cuda)"
    fi
    if [ -d "$CUDA_TOOLKIT_ROOT_DIR" ]; then
      DOCKER_ARGS+=("--mount"); DOCKER_ARGS+=("type=bind,source=$CUDA_TOOLKIT_ROOT_DIR,destination=/usr/local/cuda,readonly")
    fi

    # Mount NVVM if it exists until nvidia-container-toolkit 1.12
    # https://github.com/NVIDIA/nvidia-container-toolkit/releases/tag/v1.12.0-rc.1
    LIBNVIDIA_NVVM="/usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so"
    if [ -r "$LIBNVIDIA_NVVM" ]; then
      LIBNVIDIA_NVVM="$(realpath "$LIBNVIDIA_NVVM")"
      DOCKER_ARGS+=("--mount"); DOCKER_ARGS+=("type=bind,source=$LIBNVIDIA_NVVM,destination=$LIBNVIDIA_NVVM,readonly")
    fi
  fi
fi

# X forwarding
if [ "$X_FORWARDING" != "false" ]; then
  X_FORWARDING="true"
fi
if [ "$X_FORWARDING" == "true" ]; then
  # If an X server is not already accessible
  if ! xset q 1> /dev/null 2> /dev/null; then
    # If the XAUTHORITY file has not already been set
    if [ -z "$XAUTHORITY" ] || [ ! -r "$XAUTHORITY" ]; then
      # Check if there is a running Xorg instance with an accessible Xauthority file
      while read -r XAUTHORITY_FILE; do
        if [ -n "$XAUTHORITY_FILE" ] && [ -r "$XAUTHORITY_FILE" ]; then
          XAUTHORITY="$XAUTHORITY_FILE"
          break
        fi
      done < <(ps -fC Xorg | grep -oP -- "-auth ([^ ]+)" | cut -d " " -f 2 | sort -r)
    fi
    if [ -z "$XAUTHORITY" ] || [ ! -r "$XAUTHORITY" ]; then
      echo "Could not find an accessible XAUTHORITY file in ps."
      export XAUTHORITY=""
    else
      echo "XAUTHORITY is $XAUTHORITY"
      export XAUTHORITY
    fi

    # Check if there is any display that can be connected to
    while read -r SOCKET; do
      export DISPLAY=:$(echo $SOCKET | grep -oP "\\d+\$")
      if xset q 1> /dev/null 2> /dev/null; then
        break
      else
        export DISPLAY=""
      fi
    done < <(find /tmp/.X11-unix -type s | sort -r)
    if [ -z "$DISPLAY" ]; then
      echo "Could not find running Xorg server."
      export DISPLAY=""
    fi
    echo "DISPLAY is $DISPLAY"
  fi

  # Set the XAUTH authentication family to FamilyWild to allow connections to
  # any hostname
  DOCKER_XAUTHORITY="$XAUTHORITY"
  if [ -e "$DOCKER_XAUTHORITY" ]; then
    DOCKER_XAUTHORITY=~/docker.xauth
    touch "$DOCKER_XAUTHORITY"
    # The first 2 bytes of each Xauthority line is the authentication family,
    # use sed to replace it with ffff which is FamilyWild and merge the output
    # into the Xauthority file that will be used in the Docker image
    XAUTH_LIST="$(xauth nlist "$DISPLAY" | sed -e "s/^..../ffff/")"
    if [ -n "$XAUTH_LIST" ]; then
        echo "$XAUTH_LIST" | xauth -f "$DOCKER_XAUTHORITY" nmerge -
    fi
    chmod a+r "$DOCKER_XAUTHORITY"
    export XAUTHORITY="$DOCKER_XAUTHORITY"
  fi
fi
if [ -f "$XAUTHORITY" ]; then
  DOCKER_ARGS+=("--mount")
  DOCKER_ARGS+=("type=bind,source=$XAUTHORITY,destination=$XAUTHORITY,readonly")
  DOCKER_ARGS+=("--env")
  DOCKER_ARGS+=("XAUTHORITY")
fi

# Mounts

if [ -z "$DATA_DIRECTORY" ]; then
  DATA_DIRECTORY="/data"
fi
if [ -d "$DATA_DIRECTORY" ]; then
  DOCKER_ARGS+=("--mount")
  DOCKER_ARGS+=("type=bind,source=$DATA_DIRECTORY,destination=/data")
fi

if [ -z "$SERIAL_DIRECTORY" ]; then
  SERIAL_DIRECTORY="/dev/serial"
fi
if [ -d "$SERIAL_DIRECTORY" ]; then
  DOCKER_ARGS+=("--mount")
  DOCKER_ARGS+=("type=bind,source=$SERIAL_DIRECTORY,destination=$SERIAL_DIRECTORY,readonly")
fi

# Environment variables

if [ -n "$CORE_PATTERN" ]; then
  DOCKER_ARGS+=("--env")
  DOCKER_ARGS+=("CORE_PATTERN")
fi

if [ -n "$DISPLAY" ]; then
  DOCKER_ARGS+=("--env")
  DOCKER_ARGS+=("DISPLAY")
fi

if [ -n "$ROS_MASTER_URI" ]; then
  DOCKER_ARGS+=("--env")
  DOCKER_ARGS+=("ROS_MASTER_URI")
fi

if [ -n "$TERM" ]; then
  DOCKER_ARGS+=("--env")
  DOCKER_ARGS+=("TERM")
fi

# Start container
echo "$DOCKER_IMAGE"
echo "${DOCKER_ARGS[@]}"
echo "$@"
docker run \
  --device "/dev/dri:/dev/dri" \
  --ipc=host \
  --mount "type=bind,source=/etc/localtime,destination=/etc/localtime,readonly" \
  --pid=host \
  --privileged \
  --rm \
  "${DOCKER_ARGS[@]}" \
  "$DOCKER_IMAGE" \
  "$@"
