#!/usr/bin/env bash
# Chrysalis Inference: local Qwen3.8-27B and Qwen3.8-Flash-Next server for this machine (one model on the GPU at a time;
# a request that names the other model switches to it). `chrysalis start` runs it as systemd user services
# (the model server and, if bun is installed, the web chat) and prints their URLs.
set -e
ROOT="$(dirname "$(readlink -f "$0")")"
PY="$ROOT/data/venv/bin/python"
API=http://127.0.0.1:8080
UNITS=~/.config/systemd/user
if [ -f "$UNITS/chrysalis-web.service" ]; then WEB=chrysalis-web; else WEB=; fi    # the web chat is optional

usage() {
    cat <<EOF
usage: chrysalis <command>

  start     load the model, start the web chat, print their URLs
            (requests pick the model: qwen3.8-27b or qwen3.8-flash-next; Chrysalis switches between them)
            --lan: also on the local network, with a QR code that connects a phone or another device
  stop      stop both and free the GPU
  restart   stop, then start (after rebuilding or editing the code; takes --lan too)
  status    is it running, and where
  logs      follow the request log (prompt and cached tokens, TTFT, tok/s)

one-time setup:
  install   install the systemd services and put 'chrysalis' on your PATH
EOF
}

# Where both services listen: 127.0.0.1, or 0.0.0.0 after `chrysalis start --lan`. It lives in the systemd user
# manager's environment (CHRYSALIS_HOST), so the web chat's Start button starts the model the same way. CHRYSALIS_MODEL
# there (set by the web chat's model picker) is the model Chrysalis loads at start.
host() { systemctl --user show-environment | sed -n 's/^CHRYSALIS_HOST=//p'; }
lan_ip() { ip -4 route show default | awk '{ for (i = 1; i < NF; i++) if ($i == "src") { print $(i + 1); exit } }'; }
# the subnet on the default route's interface: what a firewall lets in
lan_subnet() {
    local dev
    dev=$(ip -4 route show default | awk '{ for (i = 1; i < NF; i++) if ($i == "dev") { print $(i + 1); exit } }')
    ip -4 route show dev "$dev" scope link proto kernel | awk 'NR == 1 { print $1 }'
}

# A firewall that drops what comes in from the network (ufw, as Omarchy and others ship it) keeps other devices out of
# Chrysalis's ports. On the way onto the network, let this network in (and nothing from outside it), asking for the
# password once. ufw's rules may be root's to read: what was let in is noted in ~/.config/chrysalis/ufw too.
UFW_NOTE=~/.config/chrysalis/ufw
blocked() {
    systemctl is-active -q ufw 2>/dev/null || return 1
    local net; net=$(lan_subnet)
    ! grep -qs "allow tcp 8080:8082 0.0.0.0/0 any $net in" /etc/ufw/user.rules && ! grep -qsx "$net" "$UFW_NOTE"
}
let_in() {
    blocked || return 0
    local net; net=$(lan_subnet)
    echo "ufw keeps other devices out: letting $net reach Chrysalis (TCP 8080-8082) takes your password, once"
    if [ -t 0 ] && sudo ufw allow from "$net" to any port 8080:8082 proto tcp comment "Chrysalis Inference" >/dev/null; then
        mkdir -p "${UFW_NOTE%/*}" && echo "$net" >> "$UFW_NOTE"
    else
        echo "  not changed: sudo ufw allow from $net to any port 8080:8082 proto tcp"
    fi
}

urls() {
    local at=127.0.0.1
    if [ "$(host)" = 0.0.0.0 ]; then at=$(lan_ip); fi
    echo "  API       http://$at:8080/v1"
    if [ -n "$WEB" ] && systemctl --user is-active -q chrysalis-web; then
        if [ "$at" = 127.0.0.1 ]; then
            echo "  web chat  http://$at:8081"
        else
            # the network gets HTTPS (a browser gives the microphone, voice mode, only to secure pages); this machine
            # keeps plain HTTP on 127.0.0.1, which is one. A device starts at the setup page (plain HTTP, port 8082):
            # it trusts this machine's authority there, then the chat opens over HTTPS (web/setup.ts)
            echo "  web chat  https://$at:8081   (here: http://127.0.0.1:8081)"
            # a code that lets one device in (web/devices.ts), from the web server, which may still be starting
            local pair="" i
            for i in $(seq 30); do
                pair=$(curl -sf -X POST -H "Content-Type: application/json" -d "{}" http://127.0.0.1:8081/api/pair) && break
                sleep 0.1
            done
            read -r link code < <(printf %s "$pair" | "$PY" -Ic 'import json, sys; p = json.load(sys.stdin); print(p["url"], p["code"][:4] + "-" + p["code"][4:])' 2>/dev/null) || true
            if [ -n "${link:-}" ]; then
                echo "  devices   scan this with a phone's camera to connect it; elsewhere open ${link%%#*} and enter $code"
                if command -v qrencode >/dev/null; then qrencode -t ANSIUTF8 -m 2 "$link"; fi
                echo "            the code works once, for 10 minutes ('chrysalis status' prints a new one)"
            fi
            local ca="${CHRYSALIS_TLS_DIR:-$HOME/.config/chrysalis/tls}/ca.pem"
            if [ -f "$ca" ]; then
                echo "  authority SHA-256 $(openssl x509 -in "$ca" -noout -fingerprint -sha256 | cut -d= -f2)"
                echo "            a device that trusts this computer shows the same in its certificate details"
            fi
            if blocked; then echo "  ufw keeps other devices out: 'chrysalis start --lan' lets this network in"; fi
        fi
    fi
}

# /health answers once the model has loaded (~1.5 s): poll it 10 times a second, then print the URLs
ready() {
    if ! curl -sf "$API/health" >/dev/null; then
        printf "loading the model"
        local i=0
        until curl -sf "$API/health" >/dev/null; do
            if (( ++i % 10 == 0 )); then
                if ! systemctl --user is-active -q chrysalis; then
                    echo " failed:"; journalctl --user -u chrysalis -n 15 -o cat --no-pager; exit 1
                fi
                printf .
            fi
            sleep 0.1
        done
        echo " done"
    fi
    echo "Chrysalis is running ($(loaded))"; urls
}

# which model is on the GPU right now, and the PCIe link when it came up slower than this machine's best
# (chrysalis/link.py)
loaded() { curl -sf "$API/chrysalis/stats" | "$PY" -Ic '
import json, sys
s = json.load(sys.stdin)
l = s.get("link") or {}
slow = "; PCIe link at %g GT/s x%d, slower than its best (%g GT/s): chrysalis restart" % (l["gts"], l["width"], l["best_gts"]) if l.get("slow") else ""
print(s["model"], "loaded" + slow)' 2>/dev/null || echo loading; }

case "${1:-help}" in
  start|restart)
            [ -f "$UNITS/chrysalis.service" ] || { echo "Chrysalis is not installed: run '$ROOT/chrysalis.sh install' first"; exit 1; }
            case "${2:-}" in --lan) to=0.0.0.0 ;; "") to=127.0.0.1 ;; *) usage; exit 1 ;; esac
            was=$(host)
            if [ $to = 0.0.0.0 ]; then let_in; fi
            systemctl --user set-environment CHRYSALIS_HOST=$to
            if [ "$1" = restart ]; then
                systemctl --user restart chrysalis $WEB
            else
                if [ "${was:-127.0.0.1}" != $to ]; then systemctl --user try-restart chrysalis $WEB; fi   # move running ones
                systemctl --user start chrysalis $WEB
            fi
            ready ;;
  stop)     systemctl --user stop chrysalis $WEB; echo "Chrysalis stopped" ;;
  status)   if curl -sf "$API/health" >/dev/null; then echo "Chrysalis is running ($(loaded))"; urls
            elif systemctl --user is-active -q chrysalis; then echo "Chrysalis is loading the model ('chrysalis logs' to follow)"
            else echo "Chrysalis is stopped ('chrysalis start' starts it)"; fi ;;
  logs)     exec journalctl --user -u chrysalis -f -o cat ;;
  install)  mkdir -p "$UNITS" ~/.local/bin
            sed "s|@ROOT@|$ROOT|g" "$ROOT/chrysalis.service" > "$UNITS/chrysalis.service"
            if BUN="$(command -v bun)"; then
                sed "s|@ROOT@|$ROOT|g; s|@BUN@|$BUN|g" "$ROOT/chrysalis-web.service" > "$UNITS/chrysalis-web.service"
                (cd "$ROOT/web" && "$BUN" install --frozen-lockfile >/dev/null)
            else
                echo "bun not found: installing without the web chat"
            fi
            ln -sf "$ROOT/chrysalis.sh" ~/.local/bin/chrysalis
            systemctl --user daemon-reload
            echo "installed: 'chrysalis start' loads the model and prints its URLs" ;;
  # not in the usage: the foreground server the service runs (takes --port, --draft, ...) and the raw stats
  serve)    cd "$ROOT" && exec "$PY" -m chrysalis serve --host "${CHRYSALIS_HOST:-127.0.0.1}" ${CHRYSALIS_MODEL:+--use "$CHRYSALIS_MODEL"} "${@:2}" ;;
  stats)    exec curl -s "$API/chrysalis/stats" ;;
  help|-h|--help) usage ;;
  *)        usage; exit 1 ;;
esac
