#!/usr/bin/env bash
set -Eeuo pipefail

SESSION_NAME="fl_client"

echo "[start_fl_client] starting tmux session ${SESSION_NAME}..."

tmux new-session -d -s "${SESSION_NAME}" \
  "bash -lc '
    echo \"[fl_client] starting NVFLARE client...\";
    python -u -m nvflare.private.fed.app.client.client_train \
      -m /workspace \
      -s fed_client.json \
      --set uid=Moffitt secure_train=true config_folder=config org=Moffitt;
    ret=\$?;
    echo;
    echo \"[fl_client] client_train exited with code \$ret\";
    echo \"[fl_client] Press Enter to keep this shell open, or Ctrl+C to close.\";
    read _;
    exec bash
  '"

status=$?
if [ "$status" -ne 0 ]; then
  echo "[start_fl_client] ERROR: failed to start tmux session (exit $status)"
  exit "$status"
fi

tmux set-option -g remain-on-exit on || true

echo "[start_fl_client] tmux sessions:"
tmux ls || echo "[start_fl_client] no tmux sessions found"

# Keep container alive so you can attach later
tail -f /dev/null
