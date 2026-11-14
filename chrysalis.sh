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
PKGS=(numpy tokenizers pillow llguidance "huggingface_hub[hf_xet]" onnxruntime==1.31.0 sherpa-onnx==1.13.8 kokoro-onnx==0.4.7 websockets==17.2)
fetch() {
    curl -fsSL -o "$3" "$1" && [ "$( (sha256sum "$3" 2>/dev/null || shasum -a 256 "$3") | cut -d' ' -f1)" = "$2" ]
}
uv_get() {
    local os sum
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64) os=aarch64-apple-darwin sum=743f85035a1996ecb350659e01cd9faea9709829be09b1b4b03c73eaf3c9cd73 ;;
        Linux-x86_64) os=x86_64-unknown-linux-gnu sum=f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f ;;
        *) return 1 ;;
    esac
    fetch "https://github.com/astral-sh/uv/releases/download/0.13.5/uv-$os.tar.gz" $sum "$1/uv.tar.gz" &&
        tar -xzf "$1/uv.tar.gz" -C "$1" --strip-components=1
}
setup() {
    local T
    if [ ! -x "$PY" ]; then
        echo "getting Python and the packages Chrysalis uses (once)"
        T=$(mktemp -d)
        if uv_get "$T" && "$T/uv" python install -q --no-bin 3.12 --install-dir "$T/python" && mkdir -p "$ROOT/data" &&
            mv "$(find "$T/python" -mindepth 1 -maxdepth 1 -type d -name 'cpython-3.12.*')" "$ROOT/data/venv" &&
            "$T/uv" pip install -q --python "$ROOT/data/venv/bin/python3" --break-system-packages "${PKGS[@]}"; then
            ln -sf python3 "$ROOT/data/venv/bin/python"
        else
            echo "  that failed: start Chrysalis again to try again"
            rm -rf "$ROOT/data/venv"
        fi
        rm -rf "$T"
    fi
    if [ "$BUN" = bun ] && ! command -v bun >/dev/null; then
        local arch=linux-x64 sum=3e730528d89775f03f87f05c06787c4df4099d0a37f3f252e7b829cb23816e18
        [ -z "$MAC" ] || { arch=darwin-aarch64; sum=80afd4c069b46a6fa8f3ec520bebc750a7843e4a8d5b9583d29ad436c00a3403; }
        echo "getting Bun for the web chat (once)"
        T=$(mktemp -d)
        mkdir -p "$ROOT/bin" && fetch "https://github.com/oven-sh/bun/releases/download/bun-v1.4.3/bun-$arch.zip" $sum "$T/bun.zip" &&
            unzip -q -j "$T/bun.zip" -d "$ROOT/bin" && BUN="$ROOT/bin/bun"
        rm -rf "$T"
    fi
}
mem_gb() { if [ -n "$MAC" ]; then echo $(( $(sysctl -n hw.memsize) >> 30 )); else awk '/^MemTotal/ { printf "%d", $2 / 1048576 }' /proc/meminfo; fi; }
roomy() { (( $(mem_gb) >= 58 )); }
apple() { [ -n "$MAC" ] && [ "$(uname -m)" = arm64 ] && (( $(mem_gb) >= 30 )); }
nvidia() { [ -z "$MAC" ] && [ -e /dev/nvidiactl ] && [ -f "$ROOT/Makefile" ]; }
LLAMA_REV=9c2e0e491
build() {
    if nvidia; then
        [ -f "$ROOT/build/libchrysalis.so" ] && [ -f "$ROOT/build/libchrysalis_fn.so" ] && return 0
        local cuda=/usr/local/cuda/bin L="$ROOT/data/llama.cpp"
        command -v nvcc >/dev/null || [ -x "$cuda/nvcc" ] ||
            { echo "the CUDA engines build with NVIDIA's CUDA toolkit (nvcc): install it, then start Chrysalis again"; return 0; }
        if [ ! -f "$L/build/bin/libggml-cpu.so" ]; then
            echo "getting llama.cpp's CPU kernels, which Flash-Next's CPU experts run on (once)"
            { [ -d "$L/.git" ] || git clone -q https://github.com/ggml-org/llama.cpp "$L"; } && git -C "$L" checkout -q "$LLAMA_REV" &&
                cmake -S "$L" -B "$L/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
                    -DLLAMA_BUILD_TESTS=OFF >/dev/null && cmake --build "$L/build" -j"$(nproc)" --target ggml-cpu ggml-base >/dev/null ||
                { echo "  that failed: start Chrysalis again to try again"; return 0; }
        fi
        echo "building Chrysalis's CUDA engines for this GPU (once, a few minutes)"
        PATH="$PATH:$cuda" make -C "$ROOT" -s engines >/dev/null 2>&1 || echo "  that failed: 'make engines' shows why"
        return 0
    fi
    apple && [ -f "$ROOT/Makefile" ] || return 0
    [ -f "$ROOT/build/libchrysalis.dylib" ] && [ -f "$ROOT/build/libchrysalis_fn.dylib" ] && return 0
    if ! xcrun -f clang >/dev/null 2>&1; then
        echo "the Metal engines build with Apple's command line tools: 'xcode-select --install', then start Chrysalis again"; return 0
    fi
    echo "building Chrysalis's Metal engines (once, about a minute)"
    make -C "$ROOT" -s engines >/dev/null 2>&1 || echo "  that failed: 'xcodebuild -downloadComponent MetalToolchain' gets Metal's compiler"
}
has_gpu() {
    if [ -n "$MAC" ]; then apple && [ -f "$ROOT/build/libchrysalis.dylib" ]; return; fi
    nvidia && [ -f "$ROOT/build/libchrysalis.so" ] && [ -f "$ROOT/build/libchrysalis_fn.so" ] && return 0
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

usage() {
    cat <<EOF
usage: chrysalis.sh [--host 0.0.0.0 | --insecure] [--model <model>]

  Runs Chrysalis in this terminal: the model and the web chat, each request as it finishes, and the questions when a
  phone or computer wants in. Type help there for what it takes: local, network and insecure move it without stopping,
  verbose shows what the servers print. Ctrl+C stops it. Nothing it shows is written to disk.
  Requests pick the model (qwen3.8-27b or qwen3.8-flash-next); Chrysalis switches between them.
  --host 0.0.0.0: on the local network too, encrypted, for the phones and computers let in: a QR code connects a
  phone, and Chrysalis on another computer finds this one and uses its GPU
  --insecure: on the local network over plain HTTP, as llama.cpp serves: no certificates and nobody to let in, the
  chat on port 8081 and the API on 8080. Anyone on the network can use it and read what is sent, and other devices
  get no voice mode and no OpenCode
  --model qwen3.8-27b or qwen3.8-flash-next: start with that model and ask nothing

  chrysalis.sh get <model>   download a model's files now (Chrysalis gets the one it loads by itself)
  chrysalis.sh link          retune the GPU's PCIe link at every boot and wake, when the board's own tuning fails
  chrysalis.sh opencode      say whether OpenCode here is kept to Chrysalis's models (it is, until you change it)
  chrysalis.sh opencode outside   allow models outside this computer, after a warning; takes effect at once
  chrysalis.sh opencode only      keep OpenCode here to Chrysalis's models again
EOF
}

opencode_models() {
    local flag="$TLS_DIR/opencode-outside" up= answer
    here -f -o /dev/null "$WEB_URL/" && up=1
    case "${1:-}" in
      "")   if { [ -n "$up" ] && here "$WEB_URL/api/threads" | grep -q '"outside":true'; } || { [ -z "$up" ] && [ -f "$flag" ]; }; then
                echo "Models outside this computer are allowed: a thread on one sends that company your prompts, your files and what its tools read."
                echo "chrysalis.sh opencode only keeps OpenCode here to Chrysalis's models again."
            else
                echo "Chrysalis only: OpenCode here runs on this computer's own models, and nothing is sent anywhere else."
                echo "chrysalis.sh opencode outside allows the others."
            fi ;;
      only) if [ -n "$up" ]; then here -f -X DELETE "$WEB_URL/api/threads/outside" >/dev/null || { echo "Chrysalis did not take that" >&2; exit 1; }
            else rm -f "$flag"; fi
            echo "Chrysalis only: OpenCode here is kept to this computer's own models." ;;
      outside)
            [ -t 0 ] || { echo "run this in a terminal: it asks before it allows models outside this computer" >&2; exit 1; }
            echo "Models outside this computer"
            echo "A thread on one of these models sends that company your prompts, your files and whatever its tools read"
            echo "on this computer. Chrysalis cannot take any of it back. Threads on Chrysalis's own models stay here as before."
            read -r -p "Allow them? [y/N] " answer
            case "$answer" in y|Y|yes) ;; *) echo "Kept to Chrysalis only."; exit 0 ;; esac
            if [ -n "$up" ]; then here -f -X POST -H 'Content-Type: application/json' -d '{}' "$WEB_URL/api/threads/outside" >/dev/null || { echo "Chrysalis did not take that" >&2; exit 1; }
            else mkdir -p "$TLS_DIR" && : > "$flag"; fi
            echo "Allowed. chrysalis.sh opencode only takes it back." ;;
      *)    usage; exit 1 ;;
    esac
}

pick() {
    local ids=(qwen3.8-27b) names=(Qwen3.8-27B) notes=("Quickest replies · 4 s to load") at=0 n i key last
    if roomy; then ids+=(qwen3.8-flash-next) names+=(Qwen3.8-Flash-Next) notes+=("Fast at any length · 20 s to load"); fi
    n=${#ids[@]}
    (( n > 1 )) && [ -t 0 ] && [ -t 1 ] || return 0
    last=$(cat "$TLS_DIR/model" 2>/dev/null || true)
    for i in "${!ids[@]}"; do
        [ "${ids[i]}" != "$last" ] || at=$i
        get --check "${ids[i]}" || notes[i]+=" · downloads first"
    done
    printf '\e[?25l  \e[1mWhich model?\e[0m\n'
    while :; do
        for i in "${!ids[@]}"; do
            if (( i == at )); then printf '\r\e[K  \e[35m●\e[0m \e[1m%-19s\e[0m %s\n' "${names[i]}" "${notes[i]}"
            else printf '\r\e[K  \e[2m○ %-19s %s\e[0m\n' "${names[i]}" "${notes[i]}"; fi
        done
        printf '\r\e[K  \e[2m↑↓ to choose · Enter to start\e[0m'
        IFS= read -rsn1 key
        if [ "$key" = $'\e' ]; then read -rsn2 -t 0.05 key || true; fi
        case $key in
            "[A"|k) at=$(( (at + n - 1) % n )) ;;
            "[B"|j) at=$(( (at + 1) % n )) ;;
            [1-9]) if (( key <= n )); then at=$((key - 1)); break; fi ;;
            "") break ;;
        esac
        printf '\r\e[%dA' "$n"
    done
    printf '\r\e[%dA\e[J\e[?25h' $((n + 1))
    mkdir -p "$TLS_DIR" && printf %s "${ids[at]}" > "$TLS_DIR/model"
}

lan_ip() {
    if [ -n "$MAC" ]; then ipconfig getifaddr "$(route -n get default 2>/dev/null | awk '/interface:/ { print $2 }')"; return; fi
    ip -4 route get 1.1.1.1 2>/dev/null | awk '{ for (i = 1; i < NF; i++) if ($i == "src") { print $(i + 1); exit } }'
}
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
    local net ports='808[01]'; net=$(lan_subnet)
    [ "${1:-}" != insecure ] || ports=8080
    { ! grep -qsE "allow tcp $ports:8083 0.0.0.0/0 any $net in" /etc/ufw/user.rules || ! grep -qsF "allow udp 5353 0.0.0.0/0 any $net in" /etc/ufw/user.rules; } &&
        ! grep -qsE "^$net $ports:8083 5353\$" "$UFW_NOTE"
}
let_in() {
    local from=8081 rule=Chrysalis script
    [ "${1:-}" != insecure ] || { from=8080; rule=ChrysalisOpen; }
    if [ -n "$WIN" ]; then
        [ -n "$(win "Get-NetFirewallHyperVRule -Name $rule -ErrorAction SilentlyContinue | ForEach-Object Name")" ] && return 0
        echo "Windows keeps other devices out: letting the private networks reach Chrysalis (TCP $from-8083) asks Windows for an administrator, once"
        script="$(cat <<'EOF'
$rule = "New-NetFirewallHyperVRule -Name @RULE@ -DisplayName 'Chrysalis Inference' -Direction Inbound -Action Allow -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts @FROM@-8083 -RemoteAddresses 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 100.64.0.0/10, 169.254.0.0/16"
$enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($rule))
Start-Process powershell -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList '-NoProfile', '-EncodedCommand', $enc
EOF
)"
        script=${script//@RULE@/$rule}
        win "${script//@FROM@/$from}"
        return 0
    fi
    blocked "${1:-}" || return 0
    local net; net=$(lan_subnet)
    echo "ufw keeps other devices out: letting $net reach Chrysalis (TCP $from-8083, mDNS) takes your password, once"
    if [ -t 0 ] && sudo sh -c "ufw allow from $net to any port $from:8083 proto tcp comment 'Chrysalis Inference' &&
            ufw allow from $net to any port 5353 proto udp comment 'Chrysalis Inference: found on the network'" >/dev/null; then
        mkdir -p "${UFW_NOTE%/*}" && echo "$net $from:8083 5353" >> "$UFW_NOTE"
    else
        echo "  not changed: Chrysalis started in a terminal asks for the password"
    fi
}

LINK_FIX=/usr/local/libexec/chrysalis-pcie-link
LINK_NOTE=~/.config/chrysalis/pcie-link
tune_link() {
    [ -z "$MAC$WIN" ] && command -v systemctl >/dev/null || return 0
    if cmp -s "$ROOT/system/chrysalis-pcie-link" "$LINK_FIX"; then return 0; fi
    if [ "${1:-}" != ask ] && grep -qsx no "$LINK_NOTE"; then return 0; fi
    local v big=
    for v in /sys/class/drm/card*/device/mem_info_vram_total; do
        if [ -r "$v" ] && (( $(cat "$v") >= 4 << 30 )); then big=1; fi
    done
    [ -n "$big" ] || return 0
    echo "A GPU's PCIe link comes up below its speed when the board's power-on tuning fails: Chrysalis retunes it at every boot and wake, which takes your password, once"
    if [ -t 0 ] && sudo sh -c "install -Dm755 '$ROOT/system/chrysalis-pcie-link' $LINK_FIX &&
            install -Dm644 '$ROOT/system/chrysalis-pcie-link.service' /etc/systemd/system/chrysalis-pcie-link.service &&
            install -Dm644 '$ROOT/system/90-chrysalis-pcie-link.rules' /etc/udev/rules.d/90-chrysalis-pcie-link.rules &&
            systemctl daemon-reload && systemctl enable --now chrysalis-pcie-link.service" >/dev/null 2>&1; then
        rm -f "$LINK_NOTE"
    else
        mkdir -p "${LINK_NOTE%/*}" && echo no > "$LINK_NOTE"
        echo "  not installed: 'chrysalis.sh link' in a terminal installs it"
    fi
}

release() {
    set -uo pipefail
    local VER="${1:-$(git -C "$ROOT" describe --tags --always)}" OUT=dist/chrysalis R=data/rocm/lib
    cd "$ROOT"
    rm -rf dist && mkdir -p "$OUT"/{bin,data/rocm/lib/rocm_sysdeps/lib,data/rocm/lib/llvm/lib,data/rocm/lib/hipblaslt/library,data/models/qwen3.8-27b,data/models/qwen3.8-flash-next,web} dist/native
    docker run --rm -e OWNER="$(id -u):$(id -g)" -v "$PWD:/src:ro" -v "$PWD/dist/native:/out" quay.io/pypa/manylinux_2_28_x86_64 bash /src/chrysalis.sh native
    cp -r dist/native/build dist/native/ggml "$OUT/"
    LD_LIBRARY_PATH="$PWD/$R:$PWD/$R/rocm_sysdeps/lib:$PWD/dist/native/ggml/avx2" ldd "$OUT"/build/*.so |
        awk -v r="$PWD/$R/" 'index($3, r) == 1 { print $3 }' | sort -u | while read -r f; do
        cp -L "$f" "$OUT/data/rocm/lib/${f#"$PWD/$R/"}"
    done
    cp -r "$R/hipblaslt/library/gfx1100" "$OUT/data/rocm/lib/hipblaslt/library/"
    cp chrysalis.sh LICENSE "$OUT/"
    cp -r chrysalis "$OUT/" && rm -rf "$OUT/chrysalis/__pycache__"
    cp data/models/{tokenizer.json,tokenizer_config.json,chat_template.jinja} "$OUT/data/models/"
    cp data/models/qwen3.8-27b/{config.json,generation_config.json,preprocessor_config.json} "$OUT/data/models/qwen3.8-27b/"
    cp data/models/qwen3.8-flash-next/preprocessor_config.json "$OUT/data/models/qwen3.8-flash-next/"
    UV=$(mktemp -d) && trap 'rm -rf "$UV"' EXIT
    curl -fsSL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar -xz -C "$UV" --strip-components=1
    "$UV/uv" python install -q --no-bin 3.12 --install-dir "$UV/python"
    mv "$(find "$UV/python" -mindepth 1 -maxdepth 1 -type d -name 'cpython-3.12.*')" "$OUT/data/venv"
    "$UV/uv" pip install -q --python "$OUT/data/venv/bin/python3" --break-system-packages "${PKGS[@]}"
    ln -sf python3 "$OUT/data/venv/bin/python"
    find web -maxdepth 1 -type f \( -name '*.ts' -o -name '*.tsx' -o -name '*.html' -o -name '*.css' -o -name '*.js' -o -name '*.svg' -o -name '*.json' -o -name 'bun.lock' \) \
        ! -name '*.test.ts' ! -name '*.check.ts' ! -name 'fake-*' ! -name 'bench-*' -exec cp {} "$OUT/web/" \;
    (cd web && bun install --frozen-lockfile >/dev/null)
    bun build web/server.ts web/cli.ts --target=bun --production --external react-devtools-core --outdir "$OUT/web" >/dev/null
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
    local H=engine/hip
    local SRC=($H/27b/relayout.hip $H/27b/quant.hip $H/27b/ops.hip $H/gdn.hip $H/27b/attn.hip $H/vision.hip $H/27b/model.cpp $H/27b/load.cpp $H/27b/api.cpp)
    local FN_SRC=($H/flash-next/relayout.hip $H/flash-next/quant.hip $H/flash-next/ops.hip $H/gdn.hip $H/flash-next/attn.hip $H/flash-next/moe.hip $H/flash-next/ple.hip
                  $H/vision.hip $H/flash-next/model.cpp $H/flash-next/load.cpp $H/flash-next/experts.cpp $H/flash-next/api.cpp $H/flash-next/cpu_moe.cpp $H/flash-next/cpu_vnni.cpp)
    mkdir -p /out/build
    $ROCM/bin/hipcc "${FLAGS[@]}" -I$H/27b -x hip -shared -o /out/build/libchrysalis.so "${SRC[@]}" "${LIBS[@]}" & a=$!
    $ROCM/bin/hipcc "${FLAGS[@]}" -I$H/flash-next -I$W/llama.cpp/ggml/include -x hip -shared -o /out/build/libchrysalis_fn.so "${FN_SRC[@]}" "${LIBS[@]}" \
        -L/out/ggml/avx2 -lggml-cpu -lggml-base & b=$!
    wait $a && wait $b
    chown -R "$OWNER" /out
}

case "${1:-}" in
  ""|--host|--insecure|--model)
            mode=local first=
            while [ $# -gt 0 ]; do
                case "$1:${2:-}" in
                    --host:0.0.0.0) mode=network; shift 2 ;;
                    --host:127.0.0.1) shift 2 ;;
                    --insecure:*) mode=insecure; shift ;;
                    --model:qwen3.8-27b|--model:qwen3.8-flash-next) first=$2; shift 2 ;;
                    *) usage; exit 1 ;;
                esac
            done
            if here -f -o /dev/null "$WEB_URL/"; then echo "Chrysalis is running already, in another terminal (Ctrl+C there stops it)"; exit 1; fi
            wsl_ready
            tune_link
            if [ $mode != local ]; then let_in $mode; fi
            if [ -n "$MAC" ]; then
                codesign -v "$APP" 2>/dev/null || codesign --force -s - "$APP" 2>/dev/null
            fi
            trap '[ ! -t 1 ] || printf "\e[?25h"' EXIT
            setup
            build
            cli=cli.ts; [ -f "$ROOT/web/cli.js" ] && cli=cli.js
            [ -f "$ROOT/web/cli.js" ] || [ -d "$ROOT/web/node_modules/ink" ] || (cd "$ROOT/web" && "$BUN" install --frozen-lockfile >/dev/null)
            run=("$BUN" "$ROOT/web/$cli" --mode $mode)
            if has_gpu; then
                if [ -n "$first" ]; then mkdir -p "$TLS_DIR" && printf %s "$first" > "$TLS_DIR/model"; else pick; fi
                model=$(cat "$TLS_DIR/model" 2>/dev/null || true)
                get ${model:+"$model"}
                if roomy && ! get --check qwen3.8-flash-next; then run+=(--get qwen3.8-flash-next); fi
            else
                run+=(--no-model)
            fi
            exec "${run[@]}" ;;
  firewall) let_in "${2:-}" ;;
  gpu)      has_gpu ;;
  link)     tune_link ask ;;
  opencode) opencode_models "${2:-}" ;;
  get)      setup; get "${@:2}" ;;
  serve)    if [ -z "$MAC" ]; then
                ggml="$ROOT/ggml/avx2"; grep -qw avx512_vnni /proc/cpuinfo && ggml="$ROOT/ggml/avx512"
                export LD_LIBRARY_PATH="$ROOT/data/rocm/lib:$ROOT/data/rocm/lib/rocm_sysdeps/lib:$ggml${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                [ -e /dev/dxg ] && export HSA_ENABLE_DXG_DETECTION=1
                if [ -z "${HIP_VISIBLE_DEVICES+x}${ROCR_VISIBLE_DEVICES+x}" ]; then g=$("$BUN" "$ROOT/web/gpu.ts" 2>/dev/null) && [ -n "$g" ] && export HIP_VISIBLE_DEVICES="$g"; fi
            fi
            m=$(cat "$TLS_DIR/model" 2>/dev/null || true)
            cd "$ROOT" && exec "$PY" -m chrysalis serve --host 127.0.0.1 ${m:+--use "$m"} "${@:2}" ;;
  stats)    here "$API/chrysalis/stats" ;;
  release)  release "${@:2}" ;;
  native)   native ;;
  help|-h|--help) usage ;;
  *)        usage; exit 1 ;;
esac
