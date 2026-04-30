#!/usr/bin/env bash
# build.sh - runs during Render's BUILD phase (no request timeout)
set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Downloading model weights ==="
MODEL_PATH="pneumonia_unknown_model.pth"
MODEL_URL="https://github.com/Likith-2004/Chest-X-Ray-Pneumonia-Detection/releases/download/v1.0.0/pneumonia_unknown_model.pth"

if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists, skipping download."
else
    echo "Downloading from: $MODEL_URL"
    curl -L --retry 3 --retry-delay 5 -o "$MODEL_PATH" "$MODEL_URL"
    echo "Download complete. Size: $(du -sh $MODEL_PATH | cut -f1)"
fi
