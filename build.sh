#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Running custom build.sh script..."

# 1. Install dependencies
pip install -r requirements.txt

# 2. Prune unnecessary torch files (most common cause of large size)
# These are typically files not needed for inference on a serverless function.
echo "Pruning torch unnecessary files..."
if [ -d "./.vercel/output/src/node_modules/torch" ]; then
    rm -rf ./.vercel/output/src/node_modules/torch/share/
    find ./.vercel/output/src/node_modules/torch/ -name "*.h" -delete
    find ./.vercel/output/src/node_modules/torch/ -name "*.cmake" -delete
    find ./.vercel/output/src/node_modules/torch/ -name "*.lib" -delete
fi

# You might need to adjust the path depending on where Vercel installs Python packages.
# A more robust way: find the site-packages directory
SITE_PACKAGES_DIR=$(find . -type d -name "site-packages")

if [ -d "$SITE_PACKAGES_DIR/torch" ]; then
    echo "Pruning torch files within site-packages..."
    rm -rf "$SITE_PACKAGES_DIR/torch/share"
    find "$SITE_PACKAGES_DIR/torch/" -name "*.h" -delete
    find "$SITE_PACKAGES_DIR/torch/" -name "*.cmake" -delete
    find "$SITE_PACKAGES_DIR/torch/" -name "*.lib" -delete
fi

# 3. Clean pip cache
echo "Cleaning pip cache..."
pip cache purge

echo "Custom build.sh script finished."