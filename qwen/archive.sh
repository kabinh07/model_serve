#!/bin/bash
set -e
CONTAINER=pytorch/torchserve:latest-cpu
MODEL=qwen2_5_vl_7b_instruct_q4
NAME=qwen
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
-v /mnt/data/github/model_serve/qwen/:/home/model-server \
-v /mnt/data/github/model_serve/qwen/model_store:/model_store \
-v /mnt/data/github/model_serve/qwen/models/${MODEL}:/models \
--entrypoint /bin/bash \
--workdir /home/model-server \
$CONTAINER \
-c \
"torch-model-archiver \
--model-name ${NAME} \
--version ${VERSION} \
--serialized-file /models/model-00001-of-00002.safetensors \
--handler handler.py \
--requirements-file requirements.txt \
${EXTRA} \
--force 
"
