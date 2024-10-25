#!/usr/bin/env bash
set -e

# Get the path to this script
SCRIPTPATH="$(dirname "$0")"

# Get the path to the repo root
REPOPATH=$(dirname "$(git -C "$SCRIPTPATH" rev-parse --show-toplevel)")

# Get the latest commit metadata
COMMIT_HASH="$(git -C "$REPOPATH" log --max-count=1 --pretty=format:"%h")"
COMMIT_AUTHOR="$(git -C "$REPOPATH" log --max-count=1 --pretty=format:"%aN <%aE>")"
COMMIT_TIMESTAMP="$(git -C "$REPOPATH" log --max-count=1 --pretty=format:"%aI")"

# Build the container with context set to project root
docker build \
    --force-rm \
    --build-arg PARENT_IMAGE="jts_nve_ml_ros:base" \
    --tag "${DOCKER_IMAGE:-"jts_nve_ml_ros:latest"}" \
    --label "commit_hash=$COMMIT_HASH" \
    --label "commit_author=$COMMIT_AUTHOR" \
    --label "commit_timestamp=$COMMIT_TIMESTAMP" \
    --file "$REPOPATH/nve_docker/Dockerfile" \
    "$@"
