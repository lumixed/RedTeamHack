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

# The server goes first and is never made to wait for the range. Waiting kept port
# $PORT closed long enough that the platform proxy gave up and served an error page;
# starting them together instead just moved the problem, because generating a
# scenario saturates the single shared core the server needs to finish booting.
# Its pipeline polls for the receiver config until the range answers, and the
# dashboard reports the feed as NO DATA until then rather than pretending otherwise.
python main.py server --port "$PORT" &
SRV_PID=$!

# Give the server the core long enough to bind before the range starts competing.
for _ in $(seq 1 40); do
  if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
    exec 3<&- 2>/dev/null || true
    exec 3>&- 2>/dev/null || true
    break
  fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "dashboard server exited during startup" >&2
    exit 1
  fi
  sleep 0.25
done

python main.py simulate --port "$SIM_PORT" --emitters "$EMITTERS" &
SIM_PID=$!

# Polled rather than 'wait -n', which needs bash 4.3+ and is missing on the bash
# 3.2 that ships with macOS, where this script also gets run during development.
while kill -0 "$SIM_PID" 2>/dev/null && kill -0 "$SRV_PID" 2>/dev/null; do
  sleep 2
done

echo "a component exited; stopping container" >&2
exit 1
