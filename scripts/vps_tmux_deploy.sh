#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
RUN_AT_UTC="${RUN_AT_UTC:-01:10}"
MONITOR_PORT="${MONITOR_PORT:-8765}"
MAX_ORDER_USDT="${MAX_ORDER_USDT:-250}"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Missing virtualenv Python: $PYTHON" >&2; exit 1; }
[[ -f "$ROOT/.env" ]] || { echo "Missing $ROOT/.env" >&2; exit 1; }

if [[ "${1:-}" != "--after-pull" ]]; then
    echo "[1/7] stopping old tmux sessions"
    tmux kill-session -t futures_daemon 2>/dev/null || true
    tmux kill-session -t futures_monitor 2>/dev/null || true

    echo "[2/7] backing up runtime state"
    if [[ -d runtime ]]; then
        backup="runtime.backup.$(date -u +%Y%m%d_%H%M%S)"
        cp -a runtime "$backup"
        echo "runtime_backup=$ROOT/$backup"
    fi

    echo "[3/7] updating repository"
    git pull --ff-only
    exec bash "$ROOT/scripts/vps_tmux_deploy.sh" --after-pull
fi

echo "[4/7] installing dependencies"
"$PYTHON" -m pip install -e '.[dev]'

echo "[5/7] running tests"
"$PYTHON" -m pytest -q

echo "[6/7] building monitor data"
"$PYTHON" scripts/build_monitor_dashboard_data.py
mkdir -p logs runtime

printf -v daemon_command '%q ' \
    "$PYTHON" scripts/run_daemon.py \
    --run-at-utc "$RUN_AT_UTC" \
    --run-on-start \
    --execute \
    --max-deploy-usdt 0 \
    --max-order-usdt "$MAX_ORDER_USDT"

printf -v monitor_command '%q ' \
    "$PYTHON" scripts/serve_monitor.py \
    --host 0.0.0.0 \
    --port "$MONITOR_PORT"

echo "[7/7] starting daemon and monitor"
tmux new-session -d -s futures_daemon \
    "cd $(printf '%q' "$ROOT") && exec $daemon_command >> logs/futures_daemon_stdout.log 2>&1"
tmux new-session -d -s futures_monitor \
    "cd $(printf '%q' "$ROOT") && exec $monitor_command >> logs/futures_monitor.log 2>&1"

sleep 2
if ! tmux has-session -t futures_daemon 2>/dev/null; then
    echo "futures_daemon failed to stay running; inspect logs/futures_daemon_stdout.log" >&2
    exit 1
fi
if ! tmux has-session -t futures_monitor 2>/dev/null; then
    echo "futures_monitor failed to stay running; inspect logs/futures_monitor.log" >&2
    exit 1
fi
echo
tmux list-sessions
echo
echo "Deployment started."
echo "Monitor: http://<VPS_PUBLIC_IP>:$MONITOR_PORT/"
echo "Daemon log:  tail -f $ROOT/logs/futures_daemon.log"
echo "Monitor log: tail -f $ROOT/logs/futures_monitor.log"
