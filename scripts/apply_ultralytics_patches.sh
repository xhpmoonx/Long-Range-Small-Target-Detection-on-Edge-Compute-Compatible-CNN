#!/bin/bash
set -e

ULTRA_PATH=$(python3 - <<'PY'
import ultralytics
from pathlib import Path
print(Path(ultralytics.__file__).resolve().parent)
PY
)

echo "Ultralytics path: $ULTRA_PATH"

cp patches/ultralytics/cfg/models/v8/yolov8.yaml "$ULTRA_PATH/cfg/models/v8/yolov8.yaml"
cp patches/ultralytics/nn/modules/block.py "$ULTRA_PATH/nn/modules/block.py"
cp patches/ultralytics/nn/modules/conv.py "$ULTRA_PATH/nn/modules/conv.py"
cp patches/ultralytics/nn/modules/__init__.py "$ULTRA_PATH/nn/modules/__init__.py"
cp patches/ultralytics/nn/tasks.py "$ULTRA_PATH/nn/tasks.py"

echo "Patched ultralytics successfully."