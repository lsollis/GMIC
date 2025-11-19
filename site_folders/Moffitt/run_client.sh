#!/usr/bin/env bash
# Usage: ./run_client.sh /path/to/data [gpus]
# Examples: ./run_client.sh ./data 0,1 | ./run_client.sh ./data all | ./run_client.sh ./data none

set -Eeuo pipefail
trap 'echo "❌ Error on line $LINENO"; exit 1' ERR

# Always run from the script directory (so compose finds docker-compose.yml)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Parse & validate ---
DATA_DIR="${1:-}"
GPUS="${2:-all}"

if [[ -z "$DATA_DIR" ]]; then
  echo "Usage: $0 /path/to/data [gpus]"
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Error: Data directory '$DATA_DIR' does not exist."
  exit 1
fi

# --- Resolve absolute paths ---
if command -v realpath >/dev/null 2>&1; then
  DATA_DIR_ABS="$(realpath "$DATA_DIR")"
elif command -v readlink >/dev/null 2>&1; then
  DATA_DIR_ABS="$(readlink -f "$DATA_DIR" || python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$DATA_DIR")"
else
  DATA_DIR_ABS="$(python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$DATA_DIR")"
fi

# --- Env for compose interpolation ---
export CURRENT_DIR="$SCRIPT_DIR"   # used by volumes: - "${CURRENT_DIR}:/workspace"
export DATA_DIR="$DATA_DIR_ABS"    # used by volumes: - "${DATA_DIR}:/workspace/data"
export GPU_DEVICES="$GPUS"         # used in compose: NVIDIA_VISIBLE_DEVICES: ${GPU_DEVICES}

# --- Select Podman Compose implementation ---
if podman compose version >/dev/null 2>&1; then
  COMPOSE=(podman compose)
elif podman-compose version >/dev/null 2>&1; then
  COMPOSE=(podman-compose)
else
  echo "❌ Podman Compose not found."
  exit 1
fi

# --- Display summary ---
echo "📂 Mounting data directory: $DATA_DIR"
echo "🖥️  Using GPUs: $GPU_DEVICES"
echo "🐳 (Actually using Podman, no Docker)"

# --- Build image ---
echo "🔨 Building image with Podman Compose..."
"${COMPOSE[@]}" build gmic_fl

# --- Start container *detached* ---
echo "🚀 Starting FL client container with Podman Compose (detached)..."
"${COMPOSE[@]}" up -d gmic_fl

echo "Waiting for tmux session fl_client inside container..."

# --- Wait for tmux and attach ---
podman exec -it gmic_fl_container bash -lc '
for i in $(seq 1 120); do
  if tmux has-session -t fl_client 2>/dev/null; then
    echo "[wait] tmux session found, attaching..."
    exec tmux attach -t fl_client
  fi
  echo "[wait] tmux not ready yet, retrying..."
  sleep 1
done
echo "[wait] Timed out waiting for tmux session fl_client."
exit 1
'
