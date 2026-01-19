#!/bin/bash
set -e

# --- Configuration ---
CONTAINER=pytorch/torchserve:latest-cpu
NAME=easy_ocr_vanilla_en_finetune_bn_craft
VERSION=${3:-"1.0"} # Defaults to 1.0 if not provided

# Define the specific EasyOCR model files required
# You must ensure these 3 files exist in your local 'models/' folder!
DETECTION_MODEL="craft_mlt_25k.pth"
LANG_EN="english_g2.pth"
LANG_BN="bengali-fintune.pth" 

# Construct the --extra-files argument strictly for EasyOCR requirements
# We map these paths to the /models mount inside the container
EXTRA_FILES_ARG="/models/${DETECTION_MODEL},/models/${LANG_EN},/models/bengali-fintune/model/${LANG_BN},/models/bengali-fintune/user_network/bengali-fintune.py,/models/bengali-fintune/user_network/bengali-fintune.yaml"

# If user passed custom extra files as $4, append them (e.g. custom configs)
if [ ! -z "$4" ]; then
    EXTRA_FILES_ARG="${EXTRA_FILES_ARG},$4"
fi

echo "📦 Packaging Version: ${VERSION}"
echo "file list: ${EXTRA_FILES_ARG}"

# --- Setup Directories ---
# Base project path (Parent of the script execution)
BASE_DIR="/mnt/data/projects/model_serve/easy_ocr_finetune_bn_craft"
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
--serialized-file /models/${DETECTION_MODEL} \
--handler handler.py \
--requirements-file requirements.txt \
--extra-files ${EXTRA_FILES_ARG} \
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
# We use port 8081 (default management port)

echo "🔄 Unregistering old model from TorchServe (if exists)..."
# We add '|| true' so the script doesn't stop if the model isn't currently loaded (e.g., first run)
curl -v -X DELETE "http://localhost:8081/models/${NAME}/${VERSION}" || true

# Wait a brief moment to ensure cleanup (optional but recommended)
sleep 2

echo "🔌 Registering new model to TorchServe..."
curl -v -X POST "http://localhost:8081/models?url=${NAME}.mar&initial_workers=1"

echo "✨ Done!"