#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$ROOT_DIR/rviz_web"
PORT=5003
PIDFILE="$APP_DIR/.rviz_web.pid"

kill_by_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    mapfile -t pids < <(lsof -tiTCP:"$port" -sTCP:LISTEN || true)
  else
    mapfile -t pids < <(ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p && $1 == "LISTEN" {print $NF}' | grep -Po 'pid=\K[0-9]+' || true)
  fi
  for pid in "${pids[@]:-}"; do
    [ -n "$pid" ] || continue
    if ps -p "$pid" >/dev/null 2>&1; then
      echo "$(date): Stopping process on port $port (PID: $pid)"
      kill "$pid" 2>/dev/null || true
      sleep 1
      if ps -p "$pid" >/dev/null 2>&1; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
  done
}

stop_pidfile() {
  if [ -f "$PIDFILE" ]; then
    local pid
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
      echo "$(date): Stopping RVIZ service (PID: $pid)"
      kill "$pid" 2>/dev/null || true
      for i in {1..5}; do
        if ps -p "$pid" >/dev/null 2>&1; then sleep 1; else break; fi
      done
      if ps -p "$pid" >/dev/null 2>&1; then
        echo "$(date): Forcing stop (SIGKILL) PID: $pid"
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
    rm -f "$PIDFILE" 2>/dev/null || true
  fi
}

stop_pidfile
kill_by_port "$PORT"

echo "$(date): rviz_web stopped (port $PORT)"
