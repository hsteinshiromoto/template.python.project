#!/usr/bin/env bash

# ---
# Export environ variables defined in .env file:
# ---

set -a # automatically export all variables
source .env
set +a

# Check if variable is defined in .env file
if [[ -z ${REGISTRY_USER} ]]; then
	echo "Error! Variable REGISTRY_USER is not defined" 1>&2
	exit 64

fi

# ---
# Global Variables
# ---

PROJECT_DIR=$(pwd)
PROJECT_NAME=$(basename ${PROJECT_DIR})

REGISTRY=registry.gitlab.com/${REGISTRY_USER}
DOCKER_TAG=latest
DOCKER_IMAGE=${REGISTRY}/${PROJECT_NAME}:${DOCKER_TAG}
DOCKER_PROJECT_DIR=/home/${PROJECT_NAME}

RED='\033[1;31m'
BLUE='\033[1;34m'
GREEN='\033[1;32m'
NC='\033[0m'

CONTAINER_ID=$(docker ps -qf "ancestor=${DOCKER_IMAGE}")

# ---
# Run Commands
# ---

if [[ -z "$CONTAINER_ID" ]]; then
	echo "Creating container from image ${DOCKER_IMAGE}"
	docker run -it --env-file .env -e uid=$UID -e docker_user=$(whoami) -P -v ${PROJECT_DIR}:${DOCKER_PROJECT_DIR} ${DOCKER_IMAGE} /bin/bash
	CONTAINER_ID=$(docker ps -qf "ancestor=${DOCKER_IMAGE}")

else
	echo "Container already running"

fi

port1=$(docker ps -f "ancestor=${DOCKER_IMAGE}" | grep -o "0.0.0.0:[0-9]*->[0-9]*" | cut -d ":" -f 2 | head -n 1)
port2=$(docker ps -f "ancestor=${DOCKER_IMAGE}" | grep -o "0.0.0.0:[0-9]*->[0-9]*" | cut -d ":" -f 2 | sed -n 2p)

echo -e "Container ID: ${RED}${CONTAINER_ID}${NC}"
#echo "Port mappings: ${BLUE}${port1}, ${port2}${NC}"

docker exec -it ${CONTAINER_ID} /bin/bash