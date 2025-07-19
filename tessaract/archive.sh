#!/bin/bash
set -e
CONTAINER=pytorch/torchserve:latest-cpu
MODEL=dummy_model.json
NAME=tessaract
VERSION=$3
EXTRA=$4
if [ $EXTRA ]; then
    EXTRA="--extra-files ${EXTRA}"
    else EXTRA="--extra-files models/"
fi
if [ -z $VERSION ];then
  VERSION='1.0'
fi
echo "VERSION: ${VERSION}"
# create mar
docker run --rm --shm-size 25gb \
-v /mnt/data/github/model_serve/tessaract/:/home/model-server \
-v /mnt/data/github/model_serve/tessaract/model_store:/model_store \
-v /mnt/data/github/model_serve/tessaract/models/:/models \
--entrypoint /bin/bash \
--workdir /home/model-server \
$CONTAINER \
-c \
"torch-model-archiver \
--model-name ${NAME} \
--version ${VERSION} \
--serialized-file /models/${MODEL} \
--handler handler.py \
--requirements-file requirements.txt \
${EXTRA} \
--force 
"
