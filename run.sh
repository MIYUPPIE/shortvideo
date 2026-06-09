#!/usr/bin/env bash
# Convenience wrapper: runs the pipeline inside the tf_gpu conda env.
# Usage: ./run.sh INPUT.mp4 -o out/ --vertical --captions
set -euo pipefail
cd "$(dirname "$0")"
exec conda run --no-capture-output -n tf_gpu python pipeline.py "$@"
