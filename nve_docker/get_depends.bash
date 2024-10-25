#!/usr/bin/env bash
set -e

# This script is used to collect the dependencies of the nve_ws environment
# with your local workspace mounted in the Docker container.

# Get the path to this script
SCRIPTPATH="$(dirname "$0")"

# See run.bash for options

# Get the path to the repo root
REPOPATH=$(dirname "$(git -C "$SCRIPTPATH" rev-parse --show-toplevel)")
echo $REPOPATH
# Tell Docker to mount it
export DOCKER_ARGS="$DOCKER_ARGS --mount type=bind,source=$REPOPATH/,destination=/nve_ws/"

# Generate install_deps.bash
# Take rosdep output, remove comments and batch apt commands for faster installation
"$SCRIPTPATH/run.bash" "set -o pipefail \
  && rosdep update \
  && rosdep install --default-yes --from-paths src --ignore-src --reinstall --simulate -r | sort > rosdep.out \
  && printf \"#!/usr/bin/env bash\n\" > install_deps.bash \
  && cat rosdep.out | sed \"/apt-get install -y/d\" | sed \"/^#/d\" >> install_deps.bash \
  && printf \"apt-get install -y \" >> install_deps.bash \
  && cat rosdep.out | sed -n \"/apt-get install -y/p\" | awk '{print \$4}' ORS=' ' >> install_deps.bash \
  && chmod +x install_deps.bash \
  && cat install_deps.bash"
