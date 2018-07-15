#!/bin/bash

# Bring over needed files 
cp ../../tensor-python/sentiment-analysis/*.py ./

if [[ $? == 0  ]]; then
    DOCKER_IMAGE=$(basename $(pwd))
    [ -z ${docker_image_prefix} ] && { docker_image_prefix="josephkphan/tensorflow"; }

    # Build the docker image
    echo "Building docker image ${docker_image_prefix}-${DOCKER_IMAGE}"
    docker build --no-cache=true -t ${docker_image_prefix}-${DOCKER_IMAGE} .

    echo "Now available docker images for ${docker_image_prefix}-${DOCKER_IMAGE}"
    docker images|grep "${docker_image_prefix}-${DOCKER_IMAGE}"
fi
# cleanup
rm *.py
