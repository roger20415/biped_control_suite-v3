#!/bin/bash
NETWORK_NAME=host  #"biped_bridge_network"
IMAGE_PATH="registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image"
IMAGE_TAG="latest"
ENV_FILE="./.env"

create_network() {
    echo "Checking if network $NETWORK_NAME exists..."
    if ! docker network ls | grep -q "$NETWORK_NAME"; then
        echo "Creating network $NETWORK_NAME..."
        docker network create $NETWORK_NAME || { echo "Failed to create network $NETWORK_NAME"; exit 1; }
    else
        echo "Network $NETWORK_NAME already exists."
    fi
}

run_container() {
    echo "Running the Docker container with the image $IMAGE_PATH..."
    docker run -it --rm --gpus all \
        -v "$(pwd)/src:/workspaces/src" \
        -v "$(pwd)/collect_data.sh:/workspaces/collect_data.sh" \
        --network $NETWORK_NAME \
        --ipc host \
        --env-file $ENV_FILE \
        $IMAGE_PATH:$IMAGE_TAG \
        /bin/bash -c "export FASTRTPS_DEFAULT_PROFILES_FILE=/humble_ws/fastdds.xml && exec bash" \
    || { echo "Failed to run Docker container"; exit 1; }

}

main() {
    create_network
    run_container
}

main