#!/usr/bin/env bash
# Chrysalis Inference: local Qwen3.8-27B and Qwen3.8-Flash-Next server for this machine (one model on the GPU at a time;
# a request that names the other model switches to it). Nothing to install: `chrysalis.sh start`, in this folder wherever
# it is, starts the model server and the web chat and prints their URLs; `chrysalis.sh stop` stops them.
set -e
ROOT="$(dirname "$(readlink -f "$0")")"
PY="$ROOT/data/venv/bin/python"
API=http://127.0.0.1:8080
# the web chat here: plain HTTP on this machine, HTTPS on the network (web/server.ts)
WEB_URL=http://127.0.0.1:8081
TLS_DIR="${CHRYSALIS_TLS_DIR:-$HOME/.config/chrysalis/tls}"
# this machine's own servers, quietly. A port no one listens on is refused at once, except in WSL's mirrored networking,
# where the connection hangs until TCP gives up (about 2 minutes).
here() { curl -s --connect-timeout 1 "$@"; }
# A release carries its own bun; a checkout uses the system's. A Mac's is Chrysalis.app's program (release, below),
# which is bun where BUN_BE_BUN is set.
MAC=; [ "$(uname)" = Darwin ] && MAC=1
APP="$ROOT/Chrysalis.app" WEB_LOG=~/Library/Logs/chrysalis-web.log
BUN="$ROOT/bin/bun"; [ -n "$MAC" ] && BUN="$APP/Contents/MacOS/Chrysalis"; [ -x "$BUN" ] || BUN=bun
export BUN_BE_BUN=1
get() { (cd "$ROOT" && "$PY" -m chrysalis get "$@"); }      # a model's files (chrysalis/get.py)
# Flash-Next wants 64 GB of RAM (its cold experts live there)
roomy() { awk '/^MemTotal/ { exit !($2 >= 58 * 1024 * 1024) }' /proc/meminfo; }
# this machine's GPU is one the engines are built for (gfx1100: the RX 7900 XTX, XT and GRE); without one, the web chat
# alone, which uses a Chrysalis it finds on the network (web/link.ts) or another backend (Ollama, LM Studio, llama.cpp
# ...). In WSL2 there is no /sys/class/kfd (the GPU comes
# through Windows's driver, /dev/dxg): HIP itself is asked for the card's architecture there.
has_gpu() {
    [ -e /dev/kfd ] && grep -qs "gfx_target_version 110000" /sys/class/kfd/kfd/topology/nodes/*/properties && return 0
    [ -e /dev/dxg ] && [ -x "$PY" ] || return 1
    LD_LIBRARY_PATH="$ROOT/data/rocm/lib:$ROOT/data/rocm/lib/rocm_sysdeps/lib" HSA_ENABLE_DXG_DETECTION=1 "$PY" -Ic '
import ctypes, sys
hip = ctypes.CDLL("libamdhip64.so.7")
n, arch = ctypes.c_int(), ctypes.create_string_buffer(256)
hip.hipGetDeviceCount(ctypes.byref(n))
for d in range(n.value):                       # (gfx1100: the RX 7900s, the PRO W7900 and W7800)
    if hip.hipDeviceGetName(arch, 256, d) == 0 and (b"7900" in arch.value or b"W7800" in arch.value): sys.exit(0)
sys.exit(1)' 2>/dev/null
}

# The model server and the web chat run as transient systemd user units, which nothing writes to disk: gone once they
# stop (--collect: a failed one too, so the next start can take its name). Flash-Next's cold experts (~43 GB) live in the
# server's memory, and with the high swappiness of a zram setup (Omarchy: 150) the kernel moved them to swap under
# pressure as soon as file cache: every turn then waited on swap-ins (12 s to a first token, 9 tok/s). Its own memory
# never swaps.
# A Mac has no RX 7900: its web chat alone runs as a launchd job (submitted: kept alive in the login session, gone once
# removed), as Chrysalis.app. macOS lets a program onto the local network (to find the computer with the GPU and reach
# it) once the user allows its app, which it looks up by the program's signature in Launch Services; for bun alone it
# finds none, and turns its packets away. So the app is signed (ad hoc: a release is put together on Linux) and
# registered.
up() {
    if [ -n "$MAC" ]; then
        launchctl list "$1" >/dev/null 2>&1 && return
        codesign -v "$APP" 2>/dev/null || codesign --force -s - "$APP" 2>/dev/null
        /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP"
        : > "$WEB_LOG"
        launchctl submit -l "$1" -o "$WEB_LOG" -e "$WEB_LOG" -- /usr/bin/env NODE_ENV=production BUN_BE_BUN=1 "$BUN" "$ROOT/web/server.ts"
        return
    fi
    [ "$(systemctl --user show -p LoadState --value "$1")" = loaded ] && return     # running, or restarting
    case $1 in
      chrysalis)
        systemd-run --user -q --collect --unit chrysalis -p MemorySwapMax=0 -p Restart=on-failure -p RestartSec=3 \
            "$ROOT/chrysalis.sh" serve ;;
      chrysalis-web)
        [ -d "$ROOT/web/node_modules" ] || (cd "$ROOT/web" && "$BUN" install --frozen-lockfile >/dev/null)
        systemd-run --user -q --collect --unit chrysalis-web -p WorkingDirectory="$ROOT/web" -E NODE_ENV=production \
            -p Restart=on-failure "$BUN" server.ts ;;
    esac
}
running() { if [ -n "$MAC" ]; then launchctl list "$1" >/dev/null 2>&1; else systemctl --user is-active -q "$1"; fi; }
# stopped, and gone (launchd's job until its process has exited: the next one takes its port)
halt() {
    local u i
    if [ -z "$MAC" ]; then systemctl --user stop "$@" 2>/dev/null || true; return 0; fi
    for u in "$@"; do
        launchctl remove "$u" 2>/dev/null || continue
        for i in $(seq 50); do launchctl list "$u" >/dev/null 2>&1 || break; sleep 0.1; done
    done
}

usage() {
    cat <<EOF
usage: chrysalis.sh <command>

  start     load the model, start the web chat, print their URLs
            (requests pick the model: qwen3.8-27b or qwen3.8-flash-next; Chrysalis switches between them)
            --host 0.0.0.0: on the local network too, for the phones and computers let in: a QR code connects
            a phone, and Chrysalis on another computer finds this one and uses its GPU
  stop      stop both and free the GPU
  restart   stop, then start (after rebuilding or editing the code; takes --host too)
  status    is it running, and where
  logs      follow the request log (prompt and cached tokens, TTFT, tok/s)
  allow     let in the phone or computer asking with this code (Connect a device does the same)
  get       download a model's files now (qwen3.8-27b or qwen3.8-flash-next; start gets the one it loads)
EOF
}

# Where the web chat listens: 127.0.0.1, or the network too after `chrysalis.sh start --host 0.0.0.0`, for the phones it
# lets in, and the link for the computers that use this one's Chrysalis (web/link.ts): Chrysalis itself listens on
# 127.0.0.1 only. It lives in the systemd user manager's environment (CHRYSALIS_HOST). CHRYSALIS_MODEL there (set by the
# web chat's model picker) is the model Chrysalis loads at start. A Mac's is this machine only: it has no engine to share.
host() { if [ -n "$MAC" ]; then echo 127.0.0.1; else systemctl --user show-environment | sed -n 's/^CHRYSALIS_HOST=//p'; fi; }
lan_ip() { ip -4 route show default | awk '{ for (i = 1; i < NF; i++) if ($i == "src") { print $(i + 1); exit } }'; }
# the subnet on the default route's interface: what a firewall lets in
lan_subnet() {
    local dev
    dev=$(ip -4 route show default | awk '{ for (i = 1; i < NF; i++) if ($i == "dev") { print $(i + 1); exit } }')
    ip -4 route show dev "$dev" scope link proto kernel | awk 'NR == 1 { print $1 }'
}

# A firewall that drops what comes in from the network (ufw, as Omarchy and others ship it) keeps other devices out of
# the web chat's ports (TCP 8081 the chat, 8082 the link, 8083 the trust check) and the queries that find the link (mDNS,
# UDP 5353). On the way onto the network, let this network in (and nothing from outside it), asking for the password
# once. ufw's rules may be root's to read: what was let in is noted in ~/.config/chrysalis/ufw too.
UFW_NOTE=~/.config/chrysalis/ufw
blocked() {
    systemctl is-active -q ufw 2>/dev/null || return 1
    local net; net=$(lan_subnet)
    { ! grep -qsE "allow tcp 808[01]:8083 0.0.0.0/0 any $net in" /etc/ufw/user.rules || ! grep -qsF "allow udp 5353 0.0.0.0/0 any $net in" /etc/ufw/user.rules; } &&
        ! grep -qsx "$net 8081:8083 5353" "$UFW_NOTE"
}
let_in() {
    blocked || return 0
    local net; net=$(lan_subnet)
    echo "ufw keeps other devices out: letting $net reach Chrysalis (TCP 8081-8083, mDNS) takes your password, once"
    if [ -t 0 ] && sudo sh -c "ufw allow from $net to any port 8081:8083 proto tcp comment 'Chrysalis Inference' &&
            ufw allow from $net to any port 5353 proto udp comment 'Chrysalis Inference: found on the network'" >/dev/null; then
        mkdir -p "${UFW_NOTE%/*}" && echo "$net 8081:8083 5353" >> "$UFW_NOTE"
    else
        echo "  not changed: run 'chrysalis.sh start --host 0.0.0.0' in a terminal, which asks for the password"
    fi
}

urls() {
    local at=127.0.0.1
    if [ "$(host)" = 0.0.0.0 ]; then at=$(lan_ip); fi
    echo "  API       http://127.0.0.1:8080/v1"
    if running chrysalis-web; then
        echo "  web chat  http://localhost:8081"
        if [ "$at" != 127.0.0.1 ]; then
            # A new phone gets the connect page there, past the warning its browser shows once: it trusts this machine's
            # authority, then shows its own code to allow here (web/setup.ts, web/devices.ts). Chrysalis on another
            # computer finds this one by itself (web/link.ts), and asks with a code too.
            echo "  phones    scan this, or open https://$at:8081 on the phone, past its warning: it sets the phone up,"
            echo "            then shows a code to allow in Connect a device or with 'chrysalis.sh allow <code>'"
            if command -v qrencode >/dev/null; then qrencode -t ANSIUTF8 -m 2 "https://$at:8081"; fi
            echo "  computers start Chrysalis there: it finds this one, and asks with a code too"
            asking
            local ca="$TLS_DIR/ca.pem"
            if [ -f "$ca" ]; then
                echo "  authority SHA-256 $(openssl x509 -in "$ca" -noout -fingerprint -sha256 | cut -d= -f2)"
                echo "            a device that trusts this computer shows the same in its certificate details"
            fi
            if blocked; then echo "  ufw keeps other devices out: 'chrysalis.sh start --host 0.0.0.0' lets this network in"; fi
        fi
    fi
}

# the phones and computers asking to join, from the web server (which may still be starting), each with the code it shows
asking() {
    local p="" i
    for i in $(seq 30); do p=$(here -f "$WEB_URL/api/pair") && break; sleep 0.1; done
    printf %s "$p" | "$BUN" -e '
for (const a of (await Bun.stdin.json()).waiting ?? [])
  console.log(`  asking    ${a.name.padEnd(24)} ${a.code.slice(0, 3)} ${a.code.slice(3)}${a.allowed ? "  (allowed)" : ""}`);' 2>/dev/null || true
}

# /health answers once the model has loaded (~1.5 s): poll it 10 times a second, then print the URLs
ready() {
    if ! here -f "$API/health" >/dev/null; then
        printf "loading the model"
        local i=0
        until here -f "$API/health" >/dev/null; do
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
# (chrysalis/link.py). The web chat's answers are read with bun, which a Mac has too, and not Python.
loaded() { here -f "$API/chrysalis/stats" | "$BUN" -e '
const s = await Bun.stdin.json(), l = s.link ?? {};
console.log(s.model, "loaded" + (l.slow ? `; PCIe link at ${l.gts} GT/s x${l.width}, slower than its best (${l.best_gts} GT/s): chrysalis.sh restart` : ""));' 2>/dev/null || echo loading; }

# A release, into dist/: the archive that runs Chrysalis on another Linux machine with nothing to build, set up or install
# there (unpacked anywhere, chrysalis/chrysalis.sh start), install.ps1 beside it for Windows, the Start menu's icon, the
# Mac's archive, and the API's Docker image. In the archive: both engines and ggml, built for any Linux from about 2019 on (native, below), only the ROCm libraries they load
# (~0.5 GB of the 11 GB SDK, gfx1100's hipBLASLt kernels), the server and its Python (python-build-standalone, which runs
# from wherever it is unpacked, with the server's packages), the web chat with its production packages and the bun that
# runs it. The models are not in it: the server downloads the one it loads.
# It needs docker, and magick for the icon.
release() {
    set -uo pipefail
    local VER="${1:-$(git -C "$ROOT" describe --tags --always)}" OUT=dist/chrysalis R=data/rocm/lib
    cd "$ROOT"
    rm -rf "$OUT" dist/native && mkdir -p "$OUT"/{bin,data/rocm/lib/rocm_sysdeps/lib,data/rocm/lib/llvm/lib,data/rocm/lib/hipblaslt/library,data/models/vision,web} dist/native
    docker run --rm -e OWNER="$(id -u):$(id -g)" -v "$PWD:/src:ro" -v "$PWD/dist/native:/out" quay.io/pypa/manylinux_2_28_x86_64 bash /src/chrysalis.sh native
    cp -r dist/native/build dist/native/ggml "$OUT/"
    # the libraries the engines load, under the names they ask for (the SDK's are symlinks to versioned files)
    LD_LIBRARY_PATH="$PWD/$R:$PWD/$R/rocm_sysdeps/lib:$PWD/dist/native/ggml/avx2" ldd "$OUT"/build/*.so |
        awk -v r="$PWD/$R/" 'index($3, r) == 1 { print $3 }' | sort -u | while read -r f; do
        cp -L "$f" "$OUT/data/rocm/lib/${f#"$PWD/$R/"}"
    done
    cp -r "$R/hipblaslt/library/gfx1100" "$OUT/data/rocm/lib/hipblaslt/library/"
    cp chrysalis.sh LICENSE "$OUT/"
    cp -r chrysalis "$OUT/" && rm -rf "$OUT/chrysalis/__pycache__"
    cp data/models/{tokenizer.json,tokenizer_config.json,chat_template.jinja,config.json,generation_config.json} "$OUT/data/models/"
    cp data/models/vision/preprocessor_config.json "$OUT/data/models/vision/"
    # the server's Python at data/venv, where this file runs it from, with its packages (the standard library does the rest)
    UV=$(mktemp -d) && trap 'rm -rf "$UV"' EXIT
    curl -fsSL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar -xz -C "$UV" --strip-components=1
    "$UV/uv" python install -q --no-bin 3.12 --install-dir "$UV/python"
    mv "$(find "$UV/python" -mindepth 1 -maxdepth 1 -type d -name 'cpython-3.12.*')" "$OUT/data/venv"
    "$UV/uv" pip install -q --python "$OUT/data/venv/bin/python3" --break-system-packages numpy tokenizers pillow llguidance "huggingface_hub[hf_xet]"
    ln -sf python3 "$OUT/data/venv/bin/python"
    # the web chat without its tests, and only the packages it runs with
    find web -maxdepth 1 -type f \( -name '*.ts' -o -name '*.html' -o -name '*.css' -o -name '*.js' -o -name '*.svg' -o -name '*.json' -o -name 'bun.lock' \) \
        ! -name '*.test.ts' ! -name '*.check.ts' ! -name 'fake-*' ! -name 'bench-*' -exec cp {} "$OUT/web/" \;
    (cd "$OUT/web" && bun install --production --frozen-lockfile >/dev/null)
    cp "$(readlink -f "$(command -v bun)")" "$OUT/bin/bun"
    echo "$VER" > "$OUT/VERSION"
    tar -C dist -czf dist/chrysalis-linux-x86_64.tar.gz chrysalis
    cp install.ps1 dist/
    # the Start menu's icon on Windows: the butterfly
    magick -background none -density 384 web/logo.svg -define icon:auto-resize=256,64,48,32,16 dist/chrysalis.ico
    # The Mac's (Apple silicon), unpacked and started the same way: no RX 7900 there, so the web chat alone, and
    # Chrysalis.app, which runs it (up). The app's program is bun for macOS with the app's opening compiled in: opened
    # (Finder, Spotlight, the Dock) it starts Chrysalis and opens the chat; as bun (BUN_BE_BUN) it is the web chat, which
    # macOS then knows as the app. macOS's local network permission goes by the program's Mach-O UUID too, which every
    # copy of a bun release shares (an Allow for Chrysalis would let any bun in): the app's bun gets one of its own.
    # Its icon is the butterfly: an icns of one 1024-pixel PNG.
    local M=dist/macos/chrysalis A=dist/macos/chrysalis/Chrysalis.app/Contents
    rm -rf dist/macos && mkdir -p "$A/MacOS" "$A/Resources" && cp -r chrysalis.sh LICENSE "$OUT/VERSION" "$OUT/web" "$M/"
    curl -fsSL -o "$UV/bun.zip" "https://github.com/oven-sh/bun/releases/download/bun-v$(bun --version)/bun-darwin-aarch64.zip"
    unzip -q -j "$UV/bun.zip" -d "$UV/darwin"
    bun -e 'const file = process.argv.at(-1), b = Buffer.from(await Bun.file(file).arrayBuffer());   // LC_UUID: 0x1b
for (let i = 0, at = 32; i < b.readUInt32LE(16); i++, at += b.readUInt32LE(at + 4))
  if (b.readUInt32LE(at) === 0x1b) crypto.getRandomValues(b.subarray(at + 8, at + 24));
await Bun.write(file, b);' "$UV/darwin/bun"
    printf '%s\n' 'import { resolve } from "node:path";' 'const root = resolve(process.execPath, "../../../..");' \
        'Bun.spawnSync([`${root}/chrysalis.sh`, "start"]);' 'Bun.spawnSync(["open", "http://localhost:8081"]);' > "$UV/open.ts"
    bun build --compile --target=bun-darwin-arm64 --compile-executable-path="$UV/darwin/bun" "$UV/open.ts" --outfile "$A/MacOS/Chrysalis" >/dev/null
    printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?>' '<plist version="1.0"><dict>' \
        '<key>CFBundleIdentifier</key><string>io.github.michaeldinelle.chrysalis</string>' '<key>CFBundleName</key><string>Chrysalis</string>' \
        '<key>CFBundleExecutable</key><string>Chrysalis</string>' '<key>CFBundlePackageType</key><string>APPL</string>' \
        '<key>CFBundleIconFile</key><string>chrysalis</string>' '<key>LSBackgroundOnly</key><true/>' \
        '<key>NSLocalNetworkUsageDescription</key><string>Chrysalis finds the computer on your network that runs it, and answers from its GPU.</string>' \
        '</dict></plist>' > "$A/Info.plist"
    magick -background none -density 384 web/logo.svg -resize 880x880 -gravity center -extent 1024x1024 png:"$UV/icon.png"
    bun -e 'const [png, out] = process.argv.slice(-2), p = await Bun.file(png).bytes(), h = Buffer.alloc(16);
h.write("icns"); h.writeUInt32BE(16 + p.length, 4); h.write("ic10", 8); h.writeUInt32BE(8 + p.length, 12);
await Bun.write(out, Buffer.concat([h, p]));' "$UV/icon.png" "$A/Resources/chrysalis.icns"
    tar -C dist/macos -czf dist/chrysalis-macos-arm64.tar.gz chrysalis
    # The API alone in a container: the web chat runs as systemd user services, which a container has not. The models
    # are not in the image: the server downloads the one it loads into the /models volume as it first starts, and their
    # configs go in beside them at each start.
    printf '%s\n' "FROM ubuntu:24.04" \
        "RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*" \
        "COPY chrysalis /opt/chrysalis" \
        "RUN mv /opt/chrysalis/data/models /opt/chrysalis/configs && ln -s /models /opt/chrysalis/data/models" \
        "VOLUME /models" "EXPOSE 8080" \
        'CMD ["sh", "-c", "cp -r /opt/chrysalis/configs/. /models/ && exec /opt/chrysalis/chrysalis.sh serve --host 0.0.0.0"]' |
        docker build -q -t ghcr.io/michaeldinelle/chrysalis:latest -f - dist >/dev/null
    du -sh "$OUT" dist/chrysalis-linux-x86_64.tar.gz dist/chrysalis-macos-arm64.tar.gz
}

# The release's native code, in the manylinux_2_28 image (glibc 2.28; gcc-toolset-14's libstdc++ puts what is newer than
# the system's into the binary), as the ROCm libraries themselves are built: built on this machine, the engines asked for
# glibc 2.38 and GLIBCXX_3.4.35, which Ubuntu 24.04 does not have. ggml (Flash-Next's cold experts on the CPU) twice: for
# any AVX2 CPU, and with AVX-512 and VNNI (Zen 4, recent Intel), which serve picks between by the CPU it runs on.
native() {
    set -uo pipefail
    local ROCM=/src/data/rocm GCC=/opt/rh/gcc-toolset-14/root/usr W=/work a b l
    mkdir -p "$W" && cp -r /src/engine /src/engine_fn "$W/" && cp -r /src/data/llama.cpp "$W/llama.cpp" && rm -rf "$W/llama.cpp/build"
    ggml() {                      # $1: its folder in /out, then the CPU features beyond x86-64
        local b=$W/ggml-$1
        cmake -S $W/llama.cpp -B "$b" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF "${@:2}" >/dev/null
        cmake --build "$b" -j"$(nproc)" --target ggml-cpu ggml-base >/dev/null
        mkdir -p "/out/ggml/$1" && cp -L "$b"/bin/libggml-cpu.so "$b"/bin/libggml-base.so "/out/ggml/$1/"
    }
    ggml avx2 -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON
    ggml avx512 -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON -DGGML_AVX512_VBMI=ON
    local FLAGS=(-O3 -std=c++20 -fPIC --offload-arch=gfx1100 -Wno-unused-result --gcc-toolchain=$GCC)
    local LIBS=(-L$ROCM/lib -lhipblaslt -pthread -ldl -Wl,--no-undefined)   # (glibc 2.28 keeps threads and dlopen in libraries of their own)
    cd "$W"
    local SRC=(engine/relayout.hip engine/quant.hip engine/ops.hip engine/gdn.hip engine/attn.hip engine/vision.hip engine/model.cpp engine/load.cpp engine/api.cpp)
    local FN_SRC=(engine_fn/relayout.hip engine_fn/quant.hip engine_fn/ops.hip engine_fn/gdn.hip engine_fn/attn.hip engine_fn/moe.hip engine_fn/ple.hip
                  engine/vision.hip engine_fn/model.cpp engine_fn/load.cpp engine_fn/experts.cpp engine_fn/api.cpp engine_fn/cpu_moe.cpp engine_fn/cpu_vnni.cpp)
    mkdir -p /out/build
    $ROCM/bin/hipcc "${FLAGS[@]}" -x hip -shared -o /out/build/libchrysalis.so "${SRC[@]}" "${LIBS[@]}" & a=$!
    $ROCM/bin/hipcc "${FLAGS[@]}" -I$W/llama.cpp/ggml/include -x hip -shared -o /out/build/libchrysalis_fn.so "${FN_SRC[@]}" "${LIBS[@]}" \
        -L/out/ggml/avx2 -lggml-cpu -lggml-base & b=$!
    wait $a && wait $b
    chown -R "$OWNER" /out
}

case "${1:-help}" in
  start|restart)
            case "${2:-}:${3:-}" in --host:0.0.0.0) to=0.0.0.0 ;; --host:127.0.0.1|:) to=127.0.0.1 ;; *) usage; exit 1 ;; esac
            if [ -n "$MAC" ] && [ $to = 0.0.0.0 ]; then
                echo "a Mac has no RX 7900 to share: its Chrysalis is the one on the computer with the GPU, which"
                echo "'chrysalis.sh start --host 0.0.0.0' there puts on the network"; exit 1
            fi
            gpu=; if has_gpu; then gpu=1; fi
            if [ $to = 0.0.0.0 ]; then let_in; fi
            # a restart starts both again; a start moves a running web chat to or from the network
            if [ "$1" = restart ]; then halt chrysalis chrysalis-web
            elif [ "$(host)" != $to ]; then halt chrysalis-web; fi
            [ -n "$MAC" ] || systemctl --user set-environment CHRYSALIS_HOST=$to
            if [ -z "$gpu" ]; then
                up chrysalis-web
                # (Chrysalis.app opens it next)
                for i in $(seq 100); do here -f -o /dev/null "$WEB_URL/" && break; sleep 0.1; done
                echo "the web chat: http://localhost:8081 (no RX 7900 here: it uses Chrysalis on another computer on your network, or set"
                echo "its address, at the top, to your Ollama, LM Studio or llama.cpp)"
                exit 0
            fi
            # The model it loads, the one picked last or the 27B, is downloaded first; on a machine with the RAM for
            # Flash-Next, it follows in the background, for the model picker to offer once it is here
            model=$(systemctl --user show-environment | sed -n 's/^CHRYSALIS_MODEL=//p')
            get ${model:+"$model"}
            if roomy && ! get --check qwen3.8-flash-next; then
                systemd-run --user -q --collect --unit chrysalis-get "$ROOT/chrysalis.sh" get qwen3.8-flash-next 2>/dev/null || true
                echo "  Qwen3.8-Flash-Next is downloading in the background: the model picker has it once it is here"
            fi
            up chrysalis
            up chrysalis-web
            ready ;;
  # the web chat's Start: the model server alone
  up)       if has_gpu; then up chrysalis; fi ;;
  stop)     halt chrysalis chrysalis-web; echo "Chrysalis stopped" ;;
  status)   if here -f "$API/health" >/dev/null; then echo "Chrysalis is running ($(loaded))"; urls
            elif running chrysalis; then echo "Chrysalis is loading the model ('chrysalis.sh logs' to follow)"
            elif running chrysalis-web; then echo "the web chat is running at http://localhost:8081, without Chrysalis (stopped here, or not linked yet to the one on your network)"
            else echo "Chrysalis is stopped ('chrysalis.sh start' starts it)"; fi ;;
  # (a Mac's is the web chat's: its Chrysalis, and the requests' log, are on the computer it is linked to)
  logs)     if [ -n "$MAC" ]; then exec tail -f "$WEB_LOG"; fi
            exec journalctl --user -u chrysalis -f -o cat ;;
  allow)    code=$(printf %s "${*:2}" | tr -cd 0-9)          # its digits: "482 913" as the device shows it, too
            [ -n "$code" ] || { echo "usage: chrysalis.sh allow <code>   (the code the phone or computer shows; 'chrysalis.sh status' lists them)"; exit 1; }
            r=$(here -w "\n%{http_code}" -X POST -H "Content-Type: application/json" -d "{\"code\": \"$code\"}" "$WEB_URL/api/pair/allow") \
                || { echo "the web chat is not running: 'chrysalis.sh start --host 0.0.0.0'"; exit 1; }
            if [ "${r##*$'\n'}" = 200 ]; then
                echo "allowed: $(printf %s "${r%$'\n'*}" | "$BUN" -e 'console.log((await Bun.stdin.json()).name)')"
            else
                echo "no device is asking with that code (an ask lasts 10 minutes; 'chrysalis.sh status' lists them)"; exit 1
            fi ;;
  get)      get "${@:2}" ;;
  # not in the usage: the foreground server the unit runs (takes --port, --draft, ...), the raw stats (on this machine
  # only: the network reaches the API through the link), and building a release (release [version]; native is its build
  # container's part)
  # the engines load the ROCm libraries beside them (a release carries only those), ahead of the path they were built with,
  # and a release's ggml built for this CPU: with AVX-512 and VNNI where it has them, else for any AVX2 CPU
  serve)    ggml="$ROOT/ggml/avx2"; grep -qw avx512_vnni /proc/cpuinfo && ggml="$ROOT/ggml/avx512"
            export LD_LIBRARY_PATH="$ROOT/data/rocm/lib:$ROOT/data/rocm/lib/rocm_sysdeps/lib:$ggml${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            # in WSL2 the GPU is Windows's: ROCm reaches it through /dev/dxg (AMD's librocdxg) once told to look
            [ -e /dev/dxg ] && export HSA_ENABLE_DXG_DETECTION=1
            cd "$ROOT" && exec "$PY" -m chrysalis serve --host 127.0.0.1 ${CHRYSALIS_MODEL:+--use "$CHRYSALIS_MODEL"} "${@:2}" ;;
  stats)    here "$API/chrysalis/stats" ;;
  release)  release "${@:2}" ;;
  native)   native ;;
  help|-h|--help) usage ;;
  *)        usage; exit 1 ;;
esac
