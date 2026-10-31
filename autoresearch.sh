#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# The pipeline preflights the immutable source/data input, acquires the shared
# gpu-lock, and runs CUDA in an ordinary foreground Concourse task container.
# See concourse/fewnerd-probe.yml. No host CUDA execution or ComfyUI calls.
exec fly -t local trigger-job --job ncm-fewnerd-cache-probe/full-probe --watch
