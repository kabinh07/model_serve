#!/bin/bash
set -e

# --- Configuration ---
CONTAINER=pytorch/torchserve:latest-cpu
MODEL=5.imgW_200_best_accuracy.pth
NAME=easy_ocr
VERSION=${3:-"1.0"}  # Defaults to 1.0 if not provided

# Construct the --extra-files argument
if [ ! -z "$4" ]; then
    EXTRA_FILES="$4"
else 
    EXTRA_FILES="./dependencies,models/opt.txt"
fi

echo "📦 Packaging Version: ${VERSION}"
echo "📋 Extra files: ${EXTRA_FILES}"

# --- Setup Directories ---
BASE_DIR="/mnt/data/projects/model_serve/easy_ocr"
TARGET_DIR="/mnt/data/torchserve/model_store"

mkdir -p "${BASE_DIR}/model_store"

# --- Run Archiver in Docker ---
docker run --rm --shm-size 25gb \
-v "${BASE_DIR}:/home/model-server" \
-v "${BASE_DIR}/model_store:/model_store" \
-v "${BASE_DIR}/models:/models" \
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
--extra-files ${EXTRA_FILES} \
--export-path /model_store \
--force"

# --- Deployment ---
# Delete existing model from deployment folder
if [ -f "${TARGET_DIR}/${NAME}.mar" ]; then
    echo "♻️  Removing old model from deployment directory..."
    rm -f "${TARGET_DIR}/${NAME}.mar"
fi

# Copy new model to deployment folder
if [ -f "${BASE_DIR}/model_store/${NAME}.mar" ]; then
    echo "🚀 Copying new model to TorchServe..."
    cp "${BASE_DIR}/model_store/${NAME}.mar" "${TARGET_DIR}/"
    echo "✅ Model archived and deployed successfully: ${TARGET_DIR}/${NAME}.mar"
else
    echo "❌ Error: .mar file was not created."
    exit 1
fi

# --- TorchServe Management API ---
echo "🔄 Unregistering old model from TorchServe (if exists)..."
curl -v -X DELETE "http://localhost:8081/models/${NAME}/${VERSION}" || true

sleep 2

echo "🔌 Registering new model to TorchServe..."
curl -v -X POST "http://localhost:8081/models?url=${NAME}.mar&initial_workers=1"

echo "✨ Done!"
