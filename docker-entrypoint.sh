#!/bin/bash
set -euo pipefail

SIM_PORT="${SIM_PORT:-5051}"
PORT="${PORT:-8080}"
EMITTERS="${EMITTERS:-8}"

SIM_PID=""
SRV_PID=""

shutdown() {
  trap - EXIT INT TERM
  [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null || true
  [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap shutdown EXIT INT TERM

python main.py simulate --port "$SIM_PORT" --emitters "$EMITTERS" &
SIM_PID=$!

# The pipeline reads its receiver layout from the range once at startup, so the
# range has to be answering before the server boots. Without receivers loaded it
# can classify but never geolocate, and the map stays empty.
# A bare TCP connect via bash's /dev/tcp rather than shelling out to Python each
# time. Spawning an interpreter per poll costs seconds on a shared vCPU and was
# delaying the server by nearly a minute after the range was already listening.
ready=0
for _ in $(seq 1 120); do
  if (exec 3<>"/dev/tcp/127.0.0.1/${SIM_PORT}") 2>/dev/null; then
    exec 3<&- 2>/dev/null || true
    exec 3>&- 2>/dev/null || true
    ready=1
    break
  fi
  if ! kill -0 "$SIM_PID" 2>/dev/null; then
    echo "simulated range exited during startup" >&2
    exit 1
  fi
  sleep 0.5
done

if [ "$ready" -ne 1 ]; then
  echo "simulated range never became ready" >&2
  exit 1
fi

python main.py server --port "$PORT" &
SRV_PID=$!

# Polled rather than 'wait -n', which needs bash 4.3+ and is missing on the bash
# 3.2 that ships with macOS, where this script also gets run during development.
while kill -0 "$SIM_PID" 2>/dev/null && kill -0 "$SRV_PID" 2>/dev/null; do
  sleep 2
done

echo "a component exited; stopping container" >&2
exit 1
