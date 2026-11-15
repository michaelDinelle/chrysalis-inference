#!/usr/bin/env bash
set -e
ROOT="$(dirname "$(readlink -f "$0")")"
PY="$ROOT/data/venv/bin/python"
MAC=; [ "$(uname)" = Darwin ] && MAC=1
APP="$ROOT/Chrysalis.app"
BUN="$ROOT/bin/bun"; [ -n "$MAC" ] && BUN="$APP/Contents/MacOS/Chrysalis"; [ -x "$BUN" ] || BUN=bun
export BUN_BE_BUN=1
launch() { exec "$BUN" "$ROOT/web/launch.ts" "$@"; }
PKGS=(numpy tokenizers pillow llguidance "huggingface_hub[hf_xet]" onnxruntime==1.31.0 sherpa-onnx==1.13.8 kokoro-onnx==0.4.7 websockets==17.2 pypdfium2==5.15.2 av==19.0.2)
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
    if [ -x "$PY" ] && ! "$PY" -c "import pypdfium2, av" 2>/dev/null; then
        echo "getting the packages this version of Chrysalis added (once)"
        "$PY" -m pip install -q --break-system-packages "${PKGS[@]}" 2>/dev/null || echo "  that failed: start Chrysalis again to try again"
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

release() {
    set -uo pipefail
    local VER="${1:-$(git -C "$ROOT" describe --tags --always)}" OUT=dist/chrysalis R=data/rocm/lib
    cd "$ROOT"
    rm -rf dist && mkdir -p "$OUT"/{bin,data/rocm/lib/rocm_sysdeps/lib,data/rocm/lib/llvm/lib,data/rocm/lib/hipblaslt/library,data/models/qwen3.8-27b,data/models/qwen3.8-flash-next,web} dist/native
    docker run --rm -e OWNER="$(id -u):$(id -g)" -v "$PWD:/src:ro" -v "$PWD/dist/native:/out" quay.io/pypa/manylinux_2_28_x86_64 bash /src/chrysalis.sh native
    cp -r dist/native/build dist/native/ggml "$OUT/"
    LD_LIBRARY_PATH="$PWD/$R:$PWD/$R/rocm_sysdeps/lib:$PWD/dist/native/ggml/avx2" ldd "$OUT"/build/*.so "$OUT/build/gpu" |
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
    bun build --compile --target=bun-windows-x64 web/launch.ts --outfile dist/Chrysalis.exe >/dev/null
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
    $ROCM/bin/hipcc "${FLAGS[@]}" -s -o /out/build/gpu $H/gpu_pick.hip $H/gpu.hip "${LIBS[@]}" & l=$!
    wait $a && wait $b && wait $l
    chown -R "$OWNER" /out
}

case "${1:-}" in
  ""|--host|--insecure|--model)
            wsl_ready
            if [ -n "$MAC" ]; then
                codesign -v "$APP" 2>/dev/null || codesign --force -s - "$APP" 2>/dev/null
            fi
            setup
            build
            launch "$@" ;;
  get)      setup; launch "$@" ;;
  release)  release "${@:2}" ;;
  native)   native ;;
  *)        launch "$@" ;;
esac
