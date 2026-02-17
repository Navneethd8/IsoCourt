#!/bin/bash

# Configuration
SOURCE_ROOT="/Volumes/NavDisk/BadCoach/backend"
DEST_ROOT="/Volumes/NavDisk/BadCoachHf/BadCoach"

echo "🚀 Starting Deployment to Hugging Face..."
echo "Source: $SOURCE_ROOT"
echo "Dest: $DEST_ROOT"

# Ensure destination exists
if [ ! -d "$DEST_ROOT" ]; then
    echo "❌ Error: Destination directory not found!"
    exit 1
fi

# 1. Sync Code Files
echo "📦 Syncing Code..."
cp "$SOURCE_ROOT/model.py" "$DEST_ROOT/" || echo "Warning: model.py copy failed"
cp "$SOURCE_ROOT/api/server.py" "$DEST_ROOT/api/" || echo "Warning: server.py copy failed"
cp "$SOURCE_ROOT/badminton_detector.py" "$DEST_ROOT/" || echo "Warning: badminton_detector.py copy failed"
cp "$SOURCE_ROOT/train_fast.py" "$DEST_ROOT/" || echo "Warning: train_fast.py copy failed"
cp "$SOURCE_ROOT/pose_utils.py" "$DEST_ROOT/" || echo "Warning: pose_utils.py copy failed"

# 2. Sync Models
echo "cb Syncing Models..."
# Create models directory if it doesn't exist
mkdir -p "$DEST_ROOT/models"

cp "$SOURCE_ROOT/models/badminton_model.pth" "$DEST_ROOT/models/" || echo "Warning: badminton_model.pth copy failed"
cp "$SOURCE_ROOT/models/pose_landmarker_lite.task" "$DEST_ROOT/models/" || echo "Warning: pose_landmarker_lite.task copy failed"
cp "$SOURCE_ROOT/models/model_registry.json" "$DEST_ROOT/models/" || echo "Warning: model_registry.json copy failed"

# 3. Git Operations
echo "⬆️ Pushing to Hugging Face..."
cd "$DEST_ROOT" || exit

# Configure LFS
echo "🔧 Configuring Git LFS..."
git lfs install
git lfs track "*.pth" "*.pt" "*.task" "*.onnx"
git add .gitattributes

# Stage Changes
git add .

# Check if there are changes to commit
if git diff-index --quiet HEAD --; then
    echo "⚠️ No changes to commit."
else
    # Commit
    git commit -m "Deploy update: $(date)"
    
    # Push
    echo "☁️ Pushing to remote..."
    git push origin main
    
    echo "✅ Successfully pushed to Hugging Face!"
fi

echo "🎉 Deployment Script Completed!"
