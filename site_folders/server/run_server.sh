#!/usr/bin/env bash
# docker run script for FL server (NVFLARE)
# Usage:
#   ./run_server.sh           # open interactive bash in the server image
#   ./run_server.sh -d        # start the FL server daemonized
#
# ENV (optional):
#   SVR_NAME   - container name (default: flserver)
#   NETARG     - network/ports (default: --net=host). Example to pin ports:
#                NETARG="-p 8443:8443 -p 8002:8002"
#   DOCKER_IMAGE - image name (default: gmic-fl-nvflare)
export UID="${UID:-$(id -u)}"
export GID="${GID:-$(id -g)}"

set -Eeuo pipefail
trap 'echo "❌ Error on line $LINENO"; exit 1' ERR

# Always run from the script directory (so compose finds docker-compose.yml)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Env for compose interpolation ---
export CURRENT_DIR="$SCRIPT_DIR"   # used by volumes: - "${CURRENT_DIR}:/workspace"

# --- Pick compose command ---
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif docker-compose version >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "❌ Docker Compose not found (need 'docker compose' or 'docker-compose')"
  exit 1
fi

# --- Build & run (returns container exit code) ---
echo "🔨 Building Docker image..."
"${COMPOSE[@]}" build

echo "🚀 Starting FL client container..."
set -o pipefail
"${COMPOSE[@]}" up -d
docker exec -it gmic_fl_container tmux attach -t fl_server

exit_code=$?
echo "ℹ️  Container exit code: $exit_code"
exit "$exit_code"