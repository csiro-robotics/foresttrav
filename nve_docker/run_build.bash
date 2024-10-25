#!/usr/bin/env bash
set -e

# This script is used to build the nve_ws environment with your local
# workspace mounted in the Docker container.

# Get the path to this script
SCRIPTPATH="$(dirname "$0")"

# See run.bash for options

# Get the path to the repo root
REPOPATH=$(dirname "$(git -C "$SCRIPTPATH" rev-parse --show-toplevel)")

# Tell Docker to mount it
export DOCKER_ARGS="$DOCKER_ARGS --mount type=bind,source=$REPOPATH/,destination=/nve_ws/"

# Also mount a ccache directory
export DOCKER_ARGS="$DOCKER_ARGS --mount type=volume,source=nve_ws-ccache,destination=/ccache"

# Don't build with NVIDIA libraries by default
if [ "$NVIDIA" != "true" ]; then
  export NVIDIA="false"
fi

# Generate build artefacts
"$SCRIPTPATH/run.bash" bash -c "CCACHE_DIR=/ccache colcon build --event-handlers console_cohesion+ status-"
