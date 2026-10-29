#!/usr/bin/env bash
# Wait for another unit to finish, then train a config (queued experiments).
set -euo pipefail
unit=$1; cfg=$2
while [ "$(systemctl --user is-active "$unit")" = "active" ]; do sleep 60; done
uv run propella-bekko train --config "configs/$cfg.json" > "logs/train_$cfg.log" 2>&1
