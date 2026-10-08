#!/bin/sh
# Dev helper: start/stop a background kiln server, tracked by pidfile.  usage: tools/srv.sh start|stop|restart [log]
cd "$(dirname "$0")/.."
LOG=${2:-build/server.log}
stop() { [ -f build/server.pid ] && kill "$(cat build/server.pid)" 2>/dev/null; rm -f build/server.pid; sleep 2; }
start() {
    nohup ./data/venv/bin/python -m kiln serve > "$LOG" 2>&1 &
    echo $! > build/server.pid
    for i in $(seq 1 60); do curl -s localhost:8080/health > /dev/null && return 0; sleep 1; done
    echo "server did not come up"; tail "$LOG"; return 1
}
case "$1" in start) start ;; stop) stop ;; restart) stop; start ;; esac
