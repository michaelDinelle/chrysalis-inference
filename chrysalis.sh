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
            --lan: the web chat and the API also on the local network, for the devices let in: a QR code
            connects a phone or another computer
  stop      stop both and free the GPU
  restart   stop, then start (after rebuilding or editing the code; takes --lan too)
  status    is it running, and where
  logs      follow the request log (prompt and cached tokens, TTFT, tok/s)
  allow     let in the device asking to join with this code (--lan; Connect a device does the same)
  key       a key for an app on another computer to use the API with (--lan)

one-time setup:
  install   install the systemd services and put 'chrysalis' on your PATH
EOF
}

# Where the web chat listens: 127.0.0.1, or the network too after `chrysalis start --lan`, for the devices it lets in, and
# with Chrysalis's API behind the same keys (web/devices.ts): Chrysalis itself listens on 127.0.0.1 only. It lives in the
# systemd user manager's environment (CHRYSALIS_HOST). CHRYSALIS_MODEL
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
# the web chat's ports (8081 the chat and the API, 8082 the setup page, 8083 the trust check). On the way onto the network, let this network in (and nothing from outside it), asking for the
# password once. ufw's rules may be root's to read: what was let in is noted in ~/.config/chrysalis/ufw too.
UFW_NOTE=~/.config/chrysalis/ufw
blocked() {
    systemctl is-active -q ufw 2>/dev/null || return 1
    local net; net=$(lan_subnet)
    ! grep -qsE "allow tcp 808[01]:8083 0.0.0.0/0 any $net in" /etc/ufw/user.rules && ! grep -qsx "$net 8081:8083" "$UFW_NOTE"
}
let_in() {
    blocked || return 0
    local net; net=$(lan_subnet)
    echo "ufw keeps other devices out: letting $net reach Chrysalis (TCP 8081-8083) takes your password, once"
    if [ -t 0 ] && sudo ufw allow from "$net" to any port 8081:8083 proto tcp comment "Chrysalis Inference" >/dev/null; then
        mkdir -p "${UFW_NOTE%/*}" && echo "$net 8081:8083" >> "$UFW_NOTE"
    else
        echo "  not changed: sudo ufw allow from $net to any port 8081:8083 proto tcp"
    fi
}

urls() {
    local at=127.0.0.1
    if [ "$(host)" = 0.0.0.0 ]; then at=$(lan_ip); fi
    echo "  API       http://127.0.0.1:8080/v1"
    if [ -n "$WEB" ] && systemctl --user is-active -q chrysalis-web; then
        if [ "$at" = 127.0.0.1 ]; then
            echo "  web chat  http://$at:8081"
        else
            echo "            https://$at:8081/v1 from the network, with a key ('chrysalis key')"
            # the network gets HTTPS (a browser gives the microphone, voice mode, only to secure pages); this machine
            # keeps plain HTTP on 127.0.0.1, which is one. A device starts at the setup page (plain HTTP, port 8082):
            # it trusts this machine's authority there, then asks to join over HTTPS and shows its own code to allow
            # here (web/setup.ts, web/devices.ts)
            echo "  web chat  https://$at:8081   (here: http://127.0.0.1:8081)"
            echo "  devices   scan this with a phone's camera, or open http://$at:8082 on the device: it sets the device up,"
            echo "            then the device shows a code to allow in Connect a device or with 'chrysalis allow <code>'"
            if command -v qrencode >/dev/null; then qrencode -t ANSIUTF8 -m 2 "http://$at:8082"; fi
            asking
            local ca="${CHRYSALIS_TLS_DIR:-$HOME/.config/chrysalis/tls}/ca.pem"
            if [ -f "$ca" ]; then
                echo "  authority SHA-256 $(openssl x509 -in "$ca" -noout -fingerprint -sha256 | cut -d= -f2)"
                echo "            a device that trusts this computer shows the same in its certificate details"
            fi
            if blocked; then echo "  ufw keeps other devices out: 'chrysalis start --lan' lets this network in"; fi
        fi
    fi
}

# the devices asking to join, from the web server (which may still be starting), each with the code it shows
asking() {
    local p="" i
    for i in $(seq 30); do p=$(curl -sf http://127.0.0.1:8081/api/pair) && break; sleep 0.1; done
    printf %s "$p" | "$PY" -Ic '
import json, sys
for a in json.load(sys.stdin).get("waiting", []):
    print("  asking    %-24s %s %s%s" % (a["name"], a["code"][:3], a["code"][3:], "  (allowed)" if a["allowed"] else ""))' 2>/dev/null || true
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
            if [ $to = 0.0.0.0 ]; then
                [ -n "$WEB" ] || { echo "--lan is the web chat's (it lets devices in), and bun is not installed: see 'chrysalis install'"; exit 1; }
                let_in
            fi
            systemctl --user set-environment CHRYSALIS_HOST=$to
            if [ "$1" = restart ]; then
                systemctl --user restart chrysalis $WEB
            else
                if [ "${was:-127.0.0.1}" != $to ] && [ -n "$WEB" ]; then systemctl --user try-restart $WEB; fi   # move a running one
                systemctl --user start chrysalis $WEB
            fi
            ready ;;
  stop)     systemctl --user stop chrysalis $WEB; echo "Chrysalis stopped" ;;
  status)   if curl -sf "$API/health" >/dev/null; then echo "Chrysalis is running ($(loaded))"; urls
            elif systemctl --user is-active -q chrysalis; then echo "Chrysalis is loading the model ('chrysalis logs' to follow)"
            else echo "Chrysalis is stopped ('chrysalis start' starts it)"; fi ;;
  logs)     exec journalctl --user -u chrysalis -f -o cat ;;
  allow)    code=$(printf %s "${*:2}" | tr -cd 0-9)          # its digits: "482 913" as the device shows it, too
            [ -n "$code" ] || { echo "usage: chrysalis allow <code>   (the code the device shows; 'chrysalis status' lists them)"; exit 1; }
            r=$(curl -s -w "\n%{http_code}" -X POST -H "Content-Type: application/json" -d "{\"code\": \"$code\"}" http://127.0.0.1:8081/api/pair/allow) \
                || { echo "the web chat is not running: 'chrysalis start --lan'"; exit 1; }
            if [ "${r##*$'\n'}" = 200 ]; then
                echo "allowed: $(printf %s "${r%$'\n'*}" | "$PY" -Ic 'import json, sys; print(json.load(sys.stdin)["name"])')"
            else
                echo "no device is asking with that code (an ask lasts 10 minutes; 'chrysalis status' lists them)"; exit 1
            fi ;;
  key)      [ "$(host)" = 0.0.0.0 ] || { echo "Chrysalis is on this computer only: 'chrysalis start --lan' opens it to the network"; exit 1; }
            k=$(curl -sf -X POST -H "Content-Type: application/json" -d "{\"name\": \"${2:-An app}\"}" http://127.0.0.1:8081/api/pair/key) \
                || { echo "the web chat is not running: 'chrysalis start --lan'"; exit 1; }
            echo "  key       $(printf %s "$k" | "$PY" -Ic 'import json, sys; print(json.load(sys.stdin)["key"])')"
            echo "  API       https://$(lan_ip):8081/v1 (OpenAI's SDKs: Authorization: Bearer <key>; Anthropic's: x-api-key)"
            echo "            shown once (Chrysalis keeps its hash); Connect a device in the web chat lists and removes it" ;;
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
  # (on this machine only: the network reaches the API through the web chat, with a key)
  serve)    cd "$ROOT" && exec "$PY" -m chrysalis serve --host 127.0.0.1 ${CHRYSALIS_MODEL:+--use "$CHRYSALIS_MODEL"} "${@:2}" ;;
  stats)    exec curl -s "$API/chrysalis/stats" ;;
  help|-h|--help) usage ;;
  *)        usage; exit 1 ;;
esac
