#!/usr/bin/env bash
# kiln: local Qwen3.8-27B server for this machine. `kiln start` runs it as systemd user services
# (the model server and, if bun is installed, the web chat) and prints their URLs.
set -e
ROOT="$(dirname "$(readlink -f "$0")")"
PY="$ROOT/data/venv/bin/python"
API=http://127.0.0.1:8080
UNITS=~/.config/systemd/user
if [ -f "$UNITS/kiln-web.service" ]; then WEB=kiln-web; else WEB=; fi    # the web chat is optional

usage() {
    cat <<EOF
usage: kiln <command>

  start     load the model, start the web chat, print their URLs
  stop      stop both and free the GPU
  restart   stop, then start (after rebuilding or editing the code)
  status    is it running, and where
  logs      follow the request log (prompt and cached tokens, TTFT, tok/s)

one-time setup:
  install   install the systemd services and put 'kiln' on your PATH
  opencode  make kiln OpenCode's default model (edits ~/.config/opencode/opencode.json)
EOF
}

urls() {
    echo "  API       $API/v1"
    if [ -n "$WEB" ] && systemctl --user is-active -q kiln-web; then echo "  web chat  http://127.0.0.1:8081"; fi
}

# the server only listens once the model has loaded: wait for that, then print the URLs
ready() {
    if ! curl -sf "$API/health" >/dev/null; then
        printf "loading the model"
        until curl -sf "$API/health" >/dev/null; do
            if ! systemctl --user is-active -q kiln; then
                echo " failed:"; journalctl --user -u kiln -n 15 -o cat --no-pager; exit 1
            fi
            printf .; sleep 1
        done
        echo " done"
    fi
    echo "kiln is running"; urls
}

case "${1:-help}" in
  start)    [ -f "$UNITS/kiln.service" ] || { echo "kiln is not installed: run '$ROOT/kiln.sh install' first"; exit 1; }
            systemctl --user start kiln $WEB; ready ;;
  restart)  systemctl --user restart kiln $WEB; ready ;;
  stop)     systemctl --user stop kiln $WEB; echo "kiln stopped" ;;
  status)   if curl -sf "$API/health" >/dev/null; then echo "kiln is running"; urls
            elif systemctl --user is-active -q kiln; then echo "kiln is loading the model ('kiln logs' to follow)"
            else echo "kiln is stopped ('kiln start' starts it)"; fi ;;
  logs)     exec journalctl --user -u kiln -f -o cat ;;
  install)  mkdir -p "$UNITS" ~/.local/bin
            sed "s|@ROOT@|$ROOT|g" "$ROOT/kiln.service" > "$UNITS/kiln.service"
            if BUN="$(command -v bun)"; then
                sed "s|@ROOT@|$ROOT|g; s|@BUN@|$BUN|g" "$ROOT/kiln-web.service" > "$UNITS/kiln-web.service"
                (cd "$ROOT/web" && "$BUN" install --frozen-lockfile >/dev/null)
            else
                echo "bun not found: installing without the web chat"
            fi
            ln -sf "$ROOT/kiln.sh" ~/.local/bin/kiln
            systemctl --user daemon-reload
            echo "installed: 'kiln start' loads the model and prints its URLs, 'kiln opencode' points OpenCode at it" ;;
  opencode) cd "$ROOT" && exec "$PY" -m kiln opencode "${@:2}" ;;
  # not in the usage: the foreground server the service runs (takes --port, --draft, ...) and the raw stats
  serve)    cd "$ROOT" && exec "$PY" -m kiln serve "${@:2}" ;;
  stats)    exec curl -s "$API/kiln/stats" ;;
  help|-h|--help) usage ;;
  *)        usage; exit 1 ;;
esac
