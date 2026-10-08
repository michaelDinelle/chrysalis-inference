#!/usr/bin/env bash
# kiln: local Qwen3.8-27B server for this machine (OpenAI-compatible API on 127.0.0.1:8080).
set -e
ROOT="$(dirname "$(readlink -f "$0")")"
PY="$ROOT/data/venv/bin/python"
case "${1:-help}" in
  serve)    cd "$ROOT" && exec "$PY" -m kiln serve "${@:2}" ;;
  start|stop|restart|status) exec systemctl --user "$1" kiln ;;
  logs)     exec journalctl --user -u kiln -f -o cat ;;
  install)  mkdir -p ~/.config/systemd/user ~/.local/bin
            sed "s|@ROOT@|$ROOT|g" "$ROOT/kiln.service" > ~/.config/systemd/user/kiln.service
            if BUN="$(command -v bun)"; then         # the web UI is optional
                sed "s|@ROOT@|$ROOT|g; s|@BUN@|$BUN|g" "$ROOT/kiln-web.service" > ~/.config/systemd/user/kiln-web.service
                (cd "$ROOT/web" && "$BUN" install --frozen-lockfile >/dev/null)
            fi
            ln -sf "$ROOT/kiln.sh" ~/.local/bin/kiln
            systemctl --user daemon-reload
            echo "installed: 'kiln start' runs the server, 'kiln web' the web UI, 'systemctl --user enable kiln' starts it at login" ;;
  web)      systemctl --user "${2:-start}" kiln-web
            [ "${2:-start}" = stop ] || echo "kiln web UI: http://127.0.0.1:8081" ;;
  opencode) cd "$ROOT" && exec "$PY" -m kiln opencode "${@:2}" ;;
  stats)    exec curl -s localhost:8080/kiln/stats ;;
  *)        echo "usage: kiln serve|start|stop|restart|status|logs|install|opencode|stats|web [stop|restart]" ;;
esac
