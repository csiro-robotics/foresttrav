#!/usr/bin/env bash
set -e

# This script is used to run the refarm environment on an autonomy computer with
# the local workspace mounted in the Docker container.
SCRIPTPATH="$(dirname "$0")"

# ROOT DIR for the 
NVE_DOCKER_DIR="$(git -C "$SCRIPTPATH" rev-parse --show-toplevel)"

# Get the path to the repo root
REPOPATH=$(dirname "$(git -C "$SCRIPTPATH" rev-parse --show-toplevel)")

# Source ~/platform.env if it exists
if [ -f "$HOME/platform.env" ]; then
  set -a
  source "$HOME/platform.env"
  set +a
fi
echo "Robot Name: $ROBOT_NAME"
echo "Platform: $PLATFORM"
echo "Model Directory: $MODEL_DIR"
# Set the Docker container's name to a non-unique value to avoid multiple images running
export DOCKER_NAME=nve_docker.service

# Docker needs the source file since the python is not cleanly installed? SymLink?
export DOCKER_ARGS="$DOCKER_ARGS --mount type=bind,source=$REPOPATH/,destination=/nve_ws"
# export DOCKER_ARGS="$DOCKER_ARGS --mount type=bind,source=$HOME/data,destination=/data/"

"$NVE_DOCKER_DIR/run.bash" "$@"
