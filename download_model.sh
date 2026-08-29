#!/usr/bin/env bash
# Fetch the fixed base model once, before inference.
# Inference itself runs fully offline (local_files_only=True).
set -euo pipefail
TARGET="${1:-./models/Qwen2.5-3B-Instruct}"
python - "$TARGET" <<'PY'
import sys
from huggingface_hub import snapshot_download
target = sys.argv[1]
snapshot_download(
    "Qwen/Qwen2.5-3B-Instruct",
    local_dir=target,
    allow_patterns=["*.json", "*.safetensors", "*.txt"],
)
print("saved to", target)
PY
