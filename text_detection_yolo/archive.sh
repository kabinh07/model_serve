#!/bin/bash
set -e
CONTAINER=pytorch/torchserve:latest-cpu
MODEL=text_detection.pt
NAME=text_detection_yolo
VERSION=$1
EXTRA=$2
if [ $EXTRA ]; then
    EXTRA="--extra-files ${EXTRA}"
    else EXTRA="--extra-files ./models"
fi
if [ -z $VERSION ];then
  VERSION='1.0'
fi
echo "VERSION: ${VERSION}"
# create mar
docker run --rm --shm-size 25gb \
-v /mnt/data/projects/model_serve/text_detection_yolo/:/home/model-server \
-v /mnt/data/projects/model_serve/text_detection_yolo/model_store:/model_store \
-v /mnt/data/projects/model_serve/text_detection_yolo/models:/models \
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
