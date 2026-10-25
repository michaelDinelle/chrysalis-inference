#!/usr/bin/env bash
set -e
ROOT="$(dirname "$(readlink -f "$0")")"
PY="$ROOT/data/venv/bin/python"
API=http://127.0.0.1:8080
WEB_URL=http://127.0.0.1:8081
TLS_DIR="${CHRYSALIS_TLS_DIR:-$HOME/.config/chrysalis/tls}"
here() { curl -s --connect-timeout 1 "$@"; }
MAC=; [ "$(uname)" = Darwin ] && MAC=1
APP="$ROOT/Chrysalis.app"
BUN="$ROOT/bin/bun"; [ -n "$MAC" ] && BUN="$APP/Contents/MacOS/Chrysalis"; [ -x "$BUN" ] || BUN=bun
export BUN_BE_BUN=1
get() { (cd "$ROOT" && "$PY" -m chrysalis get "$@"); }
roomy() { awk '/^MemTotal/ { exit !($2 >= 58 * 1024 * 1024) }' /proc/meminfo; }
has_gpu() {
    [ -e /dev/kfd ] && grep -qs "gfx_target_version 110000" /sys/class/kfd/kfd/topology/nodes/*/properties && return 0
    [ -e /dev/dxg ] && [ -x "$PY" ] || return 1
    LD_LIBRARY_PATH="$ROOT/data/rocm/lib:$ROOT/data/rocm/lib/rocm_sysdeps/lib" HSA_ENABLE_DXG_DETECTION=1 "$PY" -Ic '
import ctypes, sys
hip = ctypes.CDLL("libamdhip64.so.7")
n, arch = ctypes.c_int(), ctypes.create_string_buffer(256)
hip.hipGetDeviceCount(ctypes.byref(n))
for d in range(n.value):
    if hip.hipDeviceGetName(arch, 256, d) == 0 and (b"7900" in arch.value or b"W7800" in arch.value): sys.exit(0)
sys.exit(1)' 2>/dev/null
}

engine() {
    local run=("$ROOT/chrysalis.sh" serve)
    systemd-run --user --scope -q true 2>/dev/null && run=(systemd-run --user --scope -q -p MemorySwapMax=0 "${run[@]}")
    until "${run[@]}" </dev/null; do sleep 3; done
}

usage() {
    cat <<EOF
usage: chrysalis.sh [--host 0.0.0.0]

  Runs Chrysalis in this terminal: the model and the web chat, their log, and the questions when a phone or computer
  wants in. Ctrl+C stops it. Requests pick the model (qwen3.8-27b or qwen3.8-flash-next); Chrysalis switches between them.
  --host 0.0.0.0: on the local network too, for the phones and computers let in: a QR code connects a phone, and
  Chrysalis on another computer finds this one and uses its GPU

  chrysalis.sh get <model>   download a model's files now (Chrysalis gets the one it loads by itself)
EOF
}

lan_ip() { ip -4 route get 1.1.1.1 2>/dev/null | awk '{ for (i = 1; i < NF; i++) if ($i == "src") { print $(i + 1); exit } }'; }
lan_subnet() {
    local dev
    dev=$(ip -4 route show default | awk '{ for (i = 1; i < NF; i++) if ($i == "dev") { print $(i + 1); exit } }')
    ip -4 route show dev "$dev" scope link proto kernel | awk 'NR == 1 { print $1 }'
}

WIN=; [ -z "${WSL_DISTRO_NAME:-}" ] || WIN=$(wslpath 'C:\Windows')
win() {
    "$WIN/System32/WindowsPowerShell/v1.0/powershell.exe" -NoProfile -NonInteractive \
        -EncodedCommand "$(printf %s "${1//@DISTRO@/$WSL_DISTRO_NAME}" | iconv -t UTF-16LE | base64 -w0)" 2>/dev/null | tr -d '\r'
}
wsl_ready() {
    [ -n "$WIN" ] || return 0
    if [ "$(wslinfo --networking-mode 2>/dev/null)" = nat ]; then
        local cfg add="" gb
        cfg="$(wslpath "$(win '$env:USERPROFILE')")/.wslconfig"
        grep -qsiE '^\s*networkingMode\s*=' "$cfg" || add="networkingMode=mirrored"
        gb=$(awk '/^MemTotal/ { printf "%d", $2 * 2 / 1048576 }' /proc/meminfo)
        if [ "$gb" -ge 60 ] && ! grep -qsiE '^\s*memory\s*=' "$cfg"; then add="${add:+$add }memory=$((gb - 6))GB"; fi
        if [ -n "$add" ]; then
            { [ -f "$cfg" ] && cat "$cfg"; true; } | awk -v add="$add" '
                { print } tolower($0) ~ /^[ \t]*\[wsl2\][ \t\r]*$/ && !done { n = split(add, a, " "); for (i = 1; i <= n; i++) print a[i]; done = 1 }
                END { if (!done) { print "[wsl2]"; n = split(add, a, " "); for (i = 1; i <= n; i++) print a[i] } }' > "$cfg.new" && mv "$cfg.new" "$cfg"
            echo "WSL starts again to take Chrysalis's settings (once): Chrysalis opens again by itself"
            win "$(cat <<'EOF'
$l = (Get-ChildItem HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss | Get-ItemProperty | Where-Object DistributionName -eq '@DISTRO@').ShortcutPath
$then = @"
Start-Sleep 2
& "$env:SystemRoot\System32\wsl.exe" --shutdown
Start-Process explorer.exe -ArgumentList '"$l"'
"@
$enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($then))
Start-Process -WindowStyle Hidden powershell -ArgumentList '-NoProfile', '-EncodedCommand', $enc
EOF
)"
            exit 1
        fi
    fi
    if [ "$(dpkg-query -W -f='${Status}' hsa-runtime-rocr4wsl-amdgpu 2>/dev/null)" != "install ok installed" ]; then
        echo "getting AMD's ROCm runtime for WSL (once)"
        "$WIN/System32/wsl.exe" -d "$WSL_DISTRO_NAME" -u root -- sh -c 'export DEBIAN_FRONTEND=noninteractive; dpkg --configure -a; apt-get update -q && curl -fsSL -o /tmp/amdgpu-install.deb https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb &&
            apt-get install -yq /tmp/amdgpu-install.deb && amdgpu-install -y --usecase=wsl --no-dkms' </dev/null >/dev/null 2>&1 || echo "  that failed: start again to try again"
    fi
}

UFW_NOTE=~/.config/chrysalis/ufw
blocked() {
    systemctl is-active -q ufw 2>/dev/null || return 1
    local net; net=$(lan_subnet)
    { ! grep -qsE "allow tcp 808[01]:8083 0.0.0.0/0 any $net in" /etc/ufw/user.rules || ! grep -qsF "allow udp 5353 0.0.0.0/0 any $net in" /etc/ufw/user.rules; } &&
        ! grep -qsx "$net 8081:8083 5353" "$UFW_NOTE"
}
let_in() {
    if [ -n "$WIN" ]; then
        [ -n "$(win 'Get-NetFirewallHyperVRule -Name Chrysalis -ErrorAction SilentlyContinue | ForEach-Object Name')" ] && return 0
        echo "Windows keeps other devices out: letting the private networks reach Chrysalis (TCP 8081-8083) asks Windows for an administrator, once"
        win "$(cat <<'EOF'
$rule = "New-NetFirewallHyperVRule -Name Chrysalis -DisplayName 'Chrysalis Inference' -Direction Inbound -Action Allow -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 8081-8083 -RemoteAddresses 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 100.64.0.0/10, 169.254.0.0/16"
$enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($rule))
Start-Process powershell -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList '-NoProfile', '-EncodedCommand', $enc
EOF
)"
        return 0
    fi
    blocked || return 0
    local net; net=$(lan_subnet)
    echo "ufw keeps other devices out: letting $net reach Chrysalis (TCP 8081-8083, mDNS) takes your password, once"
    if [ -t 0 ] && sudo sh -c "ufw allow from $net to any port 8081:8083 proto tcp comment 'Chrysalis Inference' &&
            ufw allow from $net to any port 5353 proto udp comment 'Chrysalis Inference: found on the network'" >/dev/null; then
        mkdir -p "${UFW_NOTE%/*}" && echo "$net 8081:8083 5353" >> "$UFW_NOTE"
    else
        echo "  not changed: run 'chrysalis.sh --host 0.0.0.0' in a terminal, which asks for the password"
    fi
}

urls() {
    local at=127.0.0.1
    if [ "$1" = 0.0.0.0 ]; then at=$(lan_ip); fi
    echo "  chat    http://localhost:8081"
    echo "  API     http://127.0.0.1:8080/v1"
    if [ "$at" != 127.0.0.1 ]; then
        echo "  phone   scan this, or open https://$at:8081 past its warning; this terminal asks to let it in"
        if command -v qrencode >/dev/null; then qrencode -t ANSIUTF8 -m 2 "https://$at:8081"; fi
        echo "  laptop  run Chrysalis there: it finds this one, and both terminals ask with the same code"
        if blocked; then echo "  ufw keeps other devices out: 'chrysalis.sh --host 0.0.0.0' in a terminal lets this network in"; fi
    fi
}

CHATTER='^chrysalis(-fn)?: (loading |CPU |weights |loaded in |warmed up |the expert pool as the last run)'
loading() {
    local spin=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏) t0=$SECONDS i=0 name=model step="loading the weights" line buf= rc vram= ram= ctx n pct=0 bar rest
    local sizes='weights ([0-9.]+) GB.*buffers ([0-9.]+) GB.*\(([0-9.]+) GB\)' cold='cold experts ([0-9.]+) GB' tokens='context ([0-9]+)'
    local progress='"progress": ([0-9]+)\.?([0-9]*)'
    while :; do
        rc=0; IFS= read -r -t 0.1 line || rc=$?
        if (( rc > 128 )); then
            buf+=$line
            [ -t 1 ] || continue
            if (( i % 3 == 0 )) && [[ $(here -m 0.5 "$API/health") =~ $progress ]]; then
                rest=${BASH_REMATCH[2]}00 pct=$((10#${BASH_REMATCH[1]} * 100 + 10#${rest:0:2}))
            fi
            printf -v bar '%*s%*s' $((pct / 5)) "" $((20 - pct / 5)) ""
            rest=${bar:pct / 5} bar=${bar:0:pct / 5}
            printf '\r\e[K  \e[35m%s\e[0m \e[1m%s\e[0m  \e[35m%s\e[90m%s\e[0m %3d%%  \e[2m%s · %d s\e[0m' \
                "${spin[i++ % 10]}" "$name" "${bar// /━}" "${rest// /━}" $pct "$step" $((SECONDS - t0))
            continue
        fi
        (( rc == 0 )) || return 1
        line=$buf$line buf=
        if [[ $line =~ $sizes ]]; then
            vram=$(awk "BEGIN { printf \"%.1f\", ${BASH_REMATCH[1]} + ${BASH_REMATCH[2]} + ${BASH_REMATCH[3]} }") step="warming up"
            [[ $line =~ $cold ]] && ram=$(awk "BEGIN { printf \"%.1f\", ${BASH_REMATCH[1]} }")
        fi
        case $line in
            "chrysalis: loading "*) name=${line#chrysalis: loading } ;;
            "chrysalis: serving "*)
                [[ $line =~ $tokens ]] && n=${BASH_REMATCH[1]}
                while (( n >= 1000 )); do ctx=$(printf ',%03d' $((n % 1000)))$ctx n=$((n / 1000)); done
                printf '\r\e[K  \e[32m✓\e[0m \e[1m%s\e[0m ready in %d s  \e[2m%s tokens of context · %s GB VRAM%s\e[0m\n' \
                    "$name" $((SECONDS - t0)) "$n$ctx" "$vram" "${ram:+ · $ram GB RAM}"
                return 0 ;;
            *) [[ $line =~ $CHATTER ]] || printf '\r\e[K%s\n' "$line" ;;
        esac
    done
}

release() {
    set -uo pipefail
    local VER="${1:-$(git -C "$ROOT" describe --tags --always)}" OUT=dist/chrysalis R=data/rocm/lib
    cd "$ROOT"
    rm -rf dist && mkdir -p "$OUT"/{bin,data/rocm/lib/rocm_sysdeps/lib,data/rocm/lib/llvm/lib,data/rocm/lib/hipblaslt/library,data/models/vision,web} dist/native
    docker run --rm -e OWNER="$(id -u):$(id -g)" -v "$PWD:/src:ro" -v "$PWD/dist/native:/out" quay.io/pypa/manylinux_2_28_x86_64 bash /src/chrysalis.sh native
    cp -r dist/native/build dist/native/ggml "$OUT/"
    LD_LIBRARY_PATH="$PWD/$R:$PWD/$R/rocm_sysdeps/lib:$PWD/dist/native/ggml/avx2" ldd "$OUT"/build/*.so |
        awk -v r="$PWD/$R/" 'index($3, r) == 1 { print $3 }' | sort -u | while read -r f; do
        cp -L "$f" "$OUT/data/rocm/lib/${f#"$PWD/$R/"}"
    done
    cp -r "$R/hipblaslt/library/gfx1100" "$OUT/data/rocm/lib/hipblaslt/library/"
    cp chrysalis.sh LICENSE "$OUT/"
    cp -r chrysalis "$OUT/" && rm -rf "$OUT/chrysalis/__pycache__"
    cp data/models/{tokenizer.json,tokenizer_config.json,chat_template.jinja,config.json,generation_config.json} "$OUT/data/models/"
    cp data/models/vision/preprocessor_config.json "$OUT/data/models/vision/"
    UV=$(mktemp -d) && trap 'rm -rf "$UV"' EXIT
    curl -fsSL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar -xz -C "$UV" --strip-components=1
    "$UV/uv" python install -q --no-bin 3.12 --install-dir "$UV/python"
    mv "$(find "$UV/python" -mindepth 1 -maxdepth 1 -type d -name 'cpython-3.12.*')" "$OUT/data/venv"
    "$UV/uv" pip install -q --python "$OUT/data/venv/bin/python3" --break-system-packages numpy tokenizers pillow llguidance "huggingface_hub[hf_xet]" onnxruntime==1.31.0 sherpa-onnx==1.13.8 kokoro-onnx==0.4.7 websockets==17.2
    ln -sf python3 "$OUT/data/venv/bin/python"
    find web -maxdepth 1 -type f \( -name '*.ts' -o -name '*.html' -o -name '*.css' -o -name '*.js' -o -name '*.svg' -o -name '*.json' -o -name 'bun.lock' \) \
        ! -name '*.test.ts' ! -name '*.check.ts' ! -name 'fake-*' ! -name 'bench-*' -exec cp {} "$OUT/web/" \;
    (cd web && bun install --frozen-lockfile >/dev/null)
    bun build web/server.ts --target=bun --production --outdir "$OUT/web" >/dev/null
    for p in mermaid/dist/mermaid.esm.min.mjs mermaid/dist/chunks/mermaid.esm.min aframe/dist/aframe-v1.8.0.min.js three/package.json three/build three/examples/jsm; do
        mkdir -p "$OUT/web/node_modules/${p%/*}" && cp -r "web/node_modules/$p" "$OUT/web/node_modules/$p"
    done
    find "$OUT/web/node_modules" -name '*.map' -delete
    cp "$(readlink -f "$(command -v bun)")" "$OUT/bin/bun"
    echo "$VER" > "$OUT/VERSION"
    tar -C dist -czf dist/chrysalis-linux-x86_64.tar.gz chrysalis
    magick -background none -density 384 web/logo.svg -define icon:auto-resize=256,64,48,32,16 dist/chrysalis.ico
    printf '%s\n' "FROM ubuntu:24.04" \
        "RUN apt-get update && apt-get install -y --no-install-recommends systemd systemd-sysv dbus-user-session curl ca-certificates iproute2 openssl && rm -rf /var/lib/apt/lists/*" \
        "RUN userdel -r ubuntu && useradd -m -u 1000 -s /bin/bash chrysalis && mkdir -p /var/lib/systemd/linger && touch /var/lib/systemd/linger/chrysalis" \
        "COPY --chown=chrysalis:chrysalis chrysalis /home/chrysalis/chrysalis" \
        "COPY chrysalis.ico /usr/share/chrysalis/chrysalis.ico" \
        "RUN printf '[boot]\nsystemd=true\n[user]\ndefault=chrysalis\n' > /etc/wsl.conf && printf '[oobe]\ndefaultUid=1000\ndefaultName=Chrysalis\n[shortcut]\nicon=/usr/share/chrysalis/chrysalis.ico\n[windowsterminal]\nenabled=false\n' > /etc/wsl-distribution.conf" \
        "RUN printf '%s\n' 'if [ \"\$(ps -o comm= -p \$PPID)\" != login ]; then' '    (until curl -s -o /dev/null http://127.0.0.1:8081/; do sleep 0.5; done; explorer.exe http://localhost:8081) &' '    exec ~/chrysalis/chrysalis.sh' 'fi' >> /home/chrysalis/.bashrc" |
        docker build -q -t chrysalis-wsl -f - dist >/dev/null
    local c; c=$(docker create chrysalis-wsl) && docker export "$c" | bsdtar -czf dist/chrysalis.wsl --exclude .dockerenv @- && docker rm "$c" >/dev/null
    local M=dist/macos/chrysalis A=dist/macos/chrysalis/Chrysalis.app/Contents
    rm -rf dist/macos && mkdir -p "$A/MacOS" "$A/Resources" && cp -r chrysalis.sh LICENSE "$OUT/VERSION" "$OUT/web" "$M/"
    curl -fsSL -o "$UV/bun.zip" "https://github.com/oven-sh/bun/releases/download/bun-v$(bun --version)/bun-darwin-aarch64.zip"
    unzip -q -j "$UV/bun.zip" -d "$UV/darwin"
    bun -e 'const file = process.argv.at(-1), b = Buffer.from(await Bun.file(file).arrayBuffer());
for (let i = 0, at = 32; i < b.readUInt32LE(16); i++, at += b.readUInt32LE(at + 4))
  if (b.readUInt32LE(at) === 0x1b) crypto.getRandomValues(b.subarray(at + 8, at + 24));
await Bun.write(file, b);' "$UV/darwin/bun"
    printf '%s\n' 'import { resolve } from "node:path";' 'const root = resolve(process.execPath, "../../../..");' \
        'Bun.spawnSync(["open", "-a", "Terminal", `${root}/chrysalis.sh`]);' \
        'for (let i = 0; i < 300 && !(await fetch("http://127.0.0.1:8081/").then(() => true, () => false)); i++) await Bun.sleep(200);' \
        'Bun.spawnSync(["open", "http://localhost:8081"]);' > "$UV/open.ts"
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
    printf '%s\n' "FROM ubuntu:24.04" \
        "RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*" \
        "COPY chrysalis /opt/chrysalis" \
        "RUN mv /opt/chrysalis/data/models /opt/chrysalis/configs && ln -s /models /opt/chrysalis/data/models" \
        "VOLUME /models" "EXPOSE 8080" \
        'CMD ["sh", "-c", "cp -r /opt/chrysalis/configs/. /models/ && exec /opt/chrysalis/chrysalis.sh serve --host 0.0.0.0"]' |
        docker build -q -t ghcr.io/michaeldinelle/chrysalis:latest -f - dist >/dev/null
    du -sh "$OUT" dist/chrysalis-linux-x86_64.tar.gz dist/chrysalis.wsl dist/chrysalis-macos-arm64.tar.gz
}

native() {
    set -uo pipefail
    local ROCM=/src/data/rocm GCC=/opt/rh/gcc-toolset-14/root/usr W=/work a b l
    mkdir -p "$W" && cp -r /src/engine "$W/" && cp -r /src/data/llama.cpp "$W/llama.cpp" && rm -rf "$W/llama.cpp/build"
    ggml() {
        local b=$W/ggml-$1
        cmake -S $W/llama.cpp -B "$b" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF "${@:2}" >/dev/null
        cmake --build "$b" -j"$(nproc)" --target ggml-cpu ggml-base >/dev/null
        mkdir -p "/out/ggml/$1" && cp -L "$b"/bin/libggml-cpu.so "$b"/bin/libggml-base.so "/out/ggml/$1/"
    }
    ggml avx2 -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON
    ggml avx512 -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON -DGGML_AVX512_VBMI=ON
    local FLAGS=(-O3 -std=c++20 -fPIC --offload-arch=gfx1100 -Wno-unused-result --gcc-toolchain=$GCC)
    local LIBS=(-L$ROCM/lib -lhipblaslt -pthread -ldl -Wl,--no-undefined)
    cd "$W"
    local SRC=(engine/27b/relayout.hip engine/27b/quant.hip engine/27b/ops.hip engine/gdn.hip engine/27b/attn.hip engine/vision.hip engine/27b/model.cpp engine/27b/load.cpp engine/27b/api.cpp)
    local FN_SRC=(engine/flash-next/relayout.hip engine/flash-next/quant.hip engine/flash-next/ops.hip engine/gdn.hip engine/flash-next/attn.hip engine/flash-next/moe.hip engine/flash-next/ple.hip
                  engine/vision.hip engine/flash-next/model.cpp engine/flash-next/load.cpp engine/flash-next/experts.cpp engine/flash-next/api.cpp engine/flash-next/cpu_moe.cpp engine/flash-next/cpu_vnni.cpp)
    mkdir -p /out/build
    $ROCM/bin/hipcc "${FLAGS[@]}" -Iengine/27b -x hip -shared -o /out/build/libchrysalis.so "${SRC[@]}" "${LIBS[@]}" & a=$!
    $ROCM/bin/hipcc "${FLAGS[@]}" -Iengine/flash-next -I$W/llama.cpp/ggml/include -x hip -shared -o /out/build/libchrysalis_fn.so "${FN_SRC[@]}" "${LIBS[@]}" \
        -L/out/ggml/avx2 -lggml-cpu -lggml-base & b=$!
    wait $a && wait $b
    chown -R "$OWNER" /out
}

case "${1:-}" in
  ""|--host) case "${1:-}:${2:-}" in --host:0.0.0.0) to=0.0.0.0 ;; --host:127.0.0.1|:) to=127.0.0.1 ;; *) usage; exit 1 ;; esac
            if [ -n "$MAC" ] && [ $to = 0.0.0.0 ]; then
                echo "a Mac has no RX 7900 to share: its Chrysalis is the one on the computer with the GPU, which"
                echo "'chrysalis.sh --host 0.0.0.0' there puts on the network"; exit 1
            fi
            if here -f -o /dev/null "$WEB_URL/"; then echo "Chrysalis is running already, in another terminal (Ctrl+C there stops it)"; exit 1; fi
            wsl_ready
            if [ $to = 0.0.0.0 ]; then let_in; fi
            if [ -n "$MAC" ]; then
                codesign -v "$APP" 2>/dev/null || codesign --force -s - "$APP" 2>/dev/null
            fi
            trap 'trap - EXIT; kill 0' EXIT
            if has_gpu; then
                model=$(cat "$TLS_DIR/model" 2>/dev/null || true)
                get ${model:+"$model"}
                if roomy && ! get --check qwen3.8-flash-next; then
                    get qwen3.8-flash-next >/dev/null 2>&1 &
                    echo "  Qwen3.8-Flash-Next is downloading in the background: the model picker has it once it is here"
                fi
                exec 3< <(trap "" INT; engine 2>&1)
                loading <&3
                grep --line-buffered -vE "$CHATTER" <&3 &
            else
                echo "no RX 7900 here: the web chat uses Chrysalis on another computer on your network, or your Ollama, LM Studio or llama.cpp"
            fi
            urls $to
            web=server.ts; [ -f "$ROOT/web/server.js" ] && web=server.js
            [ -d "$ROOT/web/node_modules" ] || (cd "$ROOT/web" && "$BUN" install --frozen-lockfile >/dev/null)
            cd "$ROOT/web" && NODE_ENV=production CHRYSALIS_HOST=$to "$BUN" $web ;;
  gpu)      has_gpu ;;
  get)      get "${@:2}" ;;
  serve)    ggml="$ROOT/ggml/avx2"; grep -qw avx512_vnni /proc/cpuinfo && ggml="$ROOT/ggml/avx512"
            export LD_LIBRARY_PATH="$ROOT/data/rocm/lib:$ROOT/data/rocm/lib/rocm_sysdeps/lib:$ggml${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            [ -e /dev/dxg ] && export HSA_ENABLE_DXG_DETECTION=1
            m=$(cat "$TLS_DIR/model" 2>/dev/null || true)
            cd "$ROOT" && exec "$PY" -m chrysalis serve --host 127.0.0.1 ${m:+--use "$m"} "${@:2}" ;;
  stats)    here "$API/chrysalis/stats" ;;
  release)  release "${@:2}" ;;
  native)   native ;;
  help|-h|--help) usage ;;
  *)        usage; exit 1 ;;
esac
