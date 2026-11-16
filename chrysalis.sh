#!/usr/bin/env bash
set -e
ROOT="$(dirname "$(readlink -f "$0")")"
MAC=; [ "$(uname)" = Darwin ] && MAC=1
export CHRYSALIS_HOME="$ROOT"
BUN="$ROOT/bin/bun"
case "$ROOT" in *.app/Contents/Resources)
    CHRYSALIS_HOME="$HOME/Library/Application Support/Chrysalis" BUN="$ROOT/../MacOS/Chrysalis"
    export PYTHONPYCACHEPREFIX="$CHRYSALIS_HOME/data/pycache"
    mkdir -p "$CHRYSALIS_HOME" && ln -sfn "$ROOT/chrysalis" "$CHRYSALIS_HOME/chrysalis" ;;
esac
PY="$CHRYSALIS_HOME/data/venv/bin/python"
[ -x "$BUN" ] || BUN=bun
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
        if uv_get "$T" && "$T/uv" python install -q --no-bin 3.12 --install-dir "$T/python" && mkdir -p "$CHRYSALIS_HOME/data" &&
            mv "$(find "$T/python" -mindepth 1 -maxdepth 1 -type d -name 'cpython-3.12.*')" "$CHRYSALIS_HOME/data/venv" &&
            "$T/uv" pip install -q --python "$CHRYSALIS_HOME/data/venv/bin/python3" --break-system-packages "${PKGS[@]}"; then
            ln -sf python3 "$CHRYSALIS_HOME/data/venv/bin/python"
        else
            echo "  that failed: start Chrysalis again to try again"
            rm -rf "$CHRYSALIS_HOME/data/venv"
        fi
        rm -rf "$T"
    fi
    if [ -x "$PY" ] && ! "$PY" -c 'import sys, importlib.metadata as m
def held(p):
    n, _, v = p.split("[")[0].partition("==")
    try:
        return v in ("", m.version(n))
    except m.PackageNotFoundError:
        return False
sys.exit(not all(map(held, sys.argv[1:])))' "${PKGS[@]}" 2>/dev/null; then
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
                cmake --log-level=WARNING -S "$L" -B "$L/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TOOLS=OFF \
                    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_COMMON=OFF >/dev/null && cmake --build "$L/build" -j"$(nproc)" --target ggml-cpu ggml-base >/dev/null ||
                { echo "  that failed: start Chrysalis again to try again"; return 0; }
        fi
        echo "building Chrysalis's CUDA engines for this GPU (once, a few minutes)"
        PATH="$PATH:$cuda" make -C "$ROOT" -s -j"$(nproc)" engines >/dev/null 2>&1 || echo "  that failed: 'make engines' shows why"
        return 0
    fi
    if [ -z "$MAC" ]; then
        [ -f "$ROOT/build/libchrysalis.so" ] && [ -f "$ROOT/build/libchrysalis_fn.so" ] && return 0
        echo "getting Chrysalis's engines for the RX 7900 and the GPU runtime they use (once, 127 MB)"
        curl -fsSL "${CHRYSALIS_RELEASE:-https://github.com/michaelDinelle/chrysalis-inference/releases/latest/download}/chrysalis-engines-linux-amd.tar.gz" |
            tar -xz -C "$ROOT" || echo "  that failed: start Chrysalis again to try again"
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
    [ ! -f data/cache/hipblaslt.txt ] || { mkdir -p "$OUT/data/cache" && cp data/cache/hipblaslt.txt "$OUT/data/cache/"; }
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
    mkdir -p "$OUT/web/node_modules/react-devtools-core" && printf '%s\n' '{ "name": "react-devtools-core", "type": "module", "main": "index.js" }' > "$OUT/web/node_modules/react-devtools-core/package.json" &&
        printf '%s\n' 'export default { initialize() {}, connectToDevTools() {} };' > "$OUT/web/node_modules/react-devtools-core/index.js"
    cp "$(readlink -f "$(command -v bun)")" "$OUT/bin/bun"
    echo "$VER" > "$OUT/VERSION"
    tar -C "$OUT" -czf dist/chrysalis-engines-linux-amd.tar.gz build ggml data/rocm data/cache
    local A=dist/macos/Chrysalis.app/Contents M=dist/macos/Chrysalis.app/Contents/Resources N="${VER#v}" && N="${N%%-*}"
    rm -rf dist/macos && mkdir -p "$A/MacOS" "$M/chrysalis" && cp -r chrysalis.sh LICENSE "$OUT/VERSION" "$OUT/web" "$M/" &&
        cp -r chrysalis/__init__.py chrysalis/voice "$M/chrysalis/" && rm -rf "$M/chrysalis/voice/__pycache__"
    curl -fsSL -o "$UV/bun.zip" "https://github.com/oven-sh/bun/releases/download/bun-v$(bun --version)/bun-darwin-aarch64.zip"
    unzip -q -j "$UV/bun.zip" -d "$UV/darwin"
    bun -e 'const file = process.argv.at(-1), b = Buffer.from(await Bun.file(file).arrayBuffer());
for (let i = 0, at = 32; i < b.readUInt32LE(16); i++, at += b.readUInt32LE(at + 4))
  if (b.readUInt32LE(at) === 0x1b) crypto.getRandomValues(b.subarray(at + 8, at + 24));
await Bun.write(file, b);' "$UV/darwin/bun"
    printf '%s\n' 'import { mkdirSync, writeFileSync } from "node:fs";' 'import { homedir } from "node:os";' 'import { join, resolve } from "node:path";' \
        'const dir = join(homedir(), "Library", "Application Support", "Chrysalis"), sh = resolve(process.execPath, "../../Resources/chrysalis.sh");' \
        'mkdirSync(dir, { recursive: true });' \
        'writeFileSync(join(dir, "Chrysalis.command"), "#!/bin/sh\nexec /usr/bin/env bash \"" + sh.replace(/["$`\\]/g, "\\$&") + "\"\n", { mode: 0o755 });' \
        'Bun.spawnSync(["open", "-a", "Terminal", join(dir, "Chrysalis.command")]);' > "$UV/open.ts"
    bun build --compile --target=bun-darwin-arm64 --compile-executable-path="$UV/darwin/bun" "$UV/open.ts" --outfile "$A/MacOS/Chrysalis" >/dev/null
    printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?>' '<plist version="1.0"><dict>' \
        '<key>CFBundleIdentifier</key><string>io.github.michaeldinelle.chrysalis</string>' '<key>CFBundleName</key><string>Chrysalis</string>' \
        '<key>CFBundleExecutable</key><string>Chrysalis</string>' '<key>CFBundlePackageType</key><string>APPL</string>' \
        "<key>CFBundleShortVersionString</key><string>$N</string>" "<key>CFBundleVersion</key><string>$N</string>" '<key>LSMinimumSystemVersion</key><string>13.0</string>' \
        '<key>CFBundleIconFile</key><string>chrysalis</string>' '<key>LSBackgroundOnly</key><true/>' \
        '<key>NSLocalNetworkUsageDescription</key><string>Chrysalis finds the computer on your network that runs it, and answers from its GPU.</string>' \
        '</dict></plist>' > "$A/Info.plist"
    magick -background none -density 384 web/logo.svg -resize 880x880 -gravity center -extent 1024x1024 png:"$UV/icon.png"
    bun -e 'const [png, out] = process.argv.slice(-2), p = await Bun.file(png).bytes(), h = Buffer.alloc(16);
h.write("icns"); h.writeUInt32BE(16 + p.length, 4); h.write("ic10", 8); h.writeUInt32BE(8 + p.length, 12);
await Bun.write(out, Buffer.concat([h, p]));' "$UV/icon.png" "$A/Resources/chrysalis.icns"
    printf '%s\n' "FROM ubuntu:24.04" \
        "RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*" \
        "COPY chrysalis /opt/chrysalis" \
        "RUN mv /opt/chrysalis/data/models /opt/chrysalis/configs && ln -s /models /opt/chrysalis/data/models" \
        "VOLUME /models" "EXPOSE 8080" \
        'CMD ["sh", "-c", "cp -r /opt/chrysalis/configs/. /models/ && exec /opt/chrysalis/chrysalis.sh serve --host 0.0.0.0"]' |
        docker build -q -t ghcr.io/michaeldinelle/chrysalis:latest -f - dist >/dev/null
    du -sh "$OUT" dist/chrysalis-engines-linux-amd.tar.gz dist/macos
}

dmg() {
    set -euo pipefail
    local from out X app dev team plist
    from="$(cd "${1:-$ROOT/dist/macos}" && pwd)" && out="$(dirname "$from")/Chrysalis.dmg"
    T=$(mktemp -d) && trap 'rm -rf "$T"' EXIT
    X="$T/Chrysalis.xcarchive" app="$T/Chrysalis.xcarchive/Products/Applications/Chrysalis.app" plist='<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict>'
    mkdir -p "$X/Products/Applications" "$T/Chrysalis" && ditto "$from/Chrysalis.app" "$app" && xattr -cr "$app" && ln -s /Applications "$T/Chrysalis/Applications"
    printf '%s\n' "$plist" '<key>com.apple.security.cs.allow-jit</key><true/>' '<key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>' \
        '<key>com.apple.security.cs.disable-executable-page-protection</key><true/>' '<key>com.apple.security.cs.allow-dyld-environment-variables</key><true/>' \
        '<key>com.apple.security.cs.disable-library-validation</key><true/>' '</dict></plist>' > "$T/bun.entitlements"
    dev=$(security find-identity -v -p codesigning | awk '/"Apple Development: / { print $2; exit }')
    codesign --force --options runtime --entitlements "$T/bun.entitlements" -s "${dev:--}" "$app"
    team=$(codesign -dv "$app" 2>&1 | sed -n 's/^TeamIdentifier=//p' | grep -vx 'not set' || true)
    if [ -z "$team" ]; then
        echo "no Apple Development certificate in the keychain, so no team to sign for: the app is signed ad hoc, and macOS opens it only after Open Anyway in Privacy & Security"
        mv "$app" "$T/Chrysalis/"
    else
        printf '%s\n' "$plist" '<key>ApplicationProperties</key><dict><key>ApplicationPath</key><string>Applications/Chrysalis.app</string>' \
            '<key>Architectures</key><array><string>arm64</string></array>' \
            "<key>CFBundleIdentifier</key><string>$(plutil -extract CFBundleIdentifier raw "$app/Contents/Info.plist")</string>" \
            "<key>CFBundleShortVersionString</key><string>$(plutil -extract CFBundleShortVersionString raw "$app/Contents/Info.plist")</string>" \
            "<key>CFBundleVersion</key><string>$(plutil -extract CFBundleVersion raw "$app/Contents/Info.plist")</string>" \
            "<key>SigningIdentity</key><string>Apple Development</string><key>Team</key><string>$team</string></dict>" \
            "<key>ArchiveVersion</key><integer>2</integer><key>CreationDate</key><date>$(date -u +%Y-%m-%dT%H:%M:%SZ)</date>" \
            '<key>Name</key><string>Chrysalis</string><key>SchemeName</key><string>Chrysalis</string></dict></plist>' > "$X/Info.plist"
        for to in export upload; do
            printf '%s\n' "$plist" '<key>method</key><string>developer-id</string><key>signingStyle</key><string>automatic</string>' \
                "<key>teamID</key><string>$team</string><key>destination</key><string>$to</string></dict></plist>" > "$T/$to.plist"
        done
        xcodebuild -exportArchive -archivePath "$X" -exportOptionsPlist "$T/export.plist" -exportPath "$T/signed" -allowProvisioningUpdates > "$T/signed.log" 2>&1 ||
            { grep -E "error:|FAILED" "$T/signed.log"; return 1; }
        echo "signed with Developer ID for team $team; sending it to Apple to notarize"
        if ! xcodebuild -exportArchive -archivePath "$X" -exportOptionsPlist "$T/upload.plist" -exportPath "$T/upload" -allowProvisioningUpdates > "$T/upload.log" 2>&1; then
            grep "error:" "$T/upload.log" || tail -3 "$T/upload.log"
            echo "Apple did not take it: the dmg holds the app signed with Developer ID, not notarized"
        else
            until xcodebuild -exportNotarizedApp -archivePath "$X" -exportPath "$T/notarized" > "$T/notarized.log" 2>&1 && [ -d "$T/notarized/Chrysalis.app" ]; do
                grep -q "is processing" "$T/notarized.log" ||
                    { grep "error:" "$T/notarized.log" || tail -3 "$T/notarized.log"; echo "Apple did not notarize it: the dmg holds the app signed with Developer ID, not notarized"; break; }
                sleep 30
            done
            if [ -d "$T/notarized/Chrysalis.app" ]; then
                xcrun stapler validate -q "$T/notarized/Chrysalis.app" || xcrun stapler staple -q "$T/notarized/Chrysalis.app"
                rm -rf "$T/signed/Chrysalis.app" && mv "$T/notarized/Chrysalis.app" "$T/signed/" && echo "notarized by Apple, its ticket stapled to the app"
            fi
        fi
        mv "$T/signed/Chrysalis.app" "$T/Chrysalis/"
    fi
    codesign -v --deep --strict "$T/Chrysalis/Chrysalis.app"
    hdiutil create -quiet -volname Chrysalis -srcfolder "$T/Chrysalis" -format ULMO -ov "$out"
    ls -l "$out" && shasum -a 256 "$out"
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
        mkdir -p "/out/ggml/$1" && cp -L "$b"/bin/libggml-cpu.so.0 "$b"/bin/libggml-base.so.0 "/out/ggml/$1/"
    }
    ggml avx2 -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON
    ggml avx512 -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON -DGGML_AVX512_VBMI=ON
    local FLAGS=(-O3 --offload-compress -std=c++20 -fPIC --offload-arch=gfx1100 -Wno-unused-result -ffunction-sections -fdata-sections -fvisibility=hidden
                 -fvisibility-inlines-hidden --gcc-toolchain=$GCC)
    local LIBS=(-L$ROCM/lib -lhipblaslt -pthread -ldl -Wl,--no-undefined -Wl,--gc-sections)
    cd "$W"
    local H=engine/hip G=engine/gpu
    local SRC=($H/27b/relayout.hip $H/27b/quant.hip $H/27b/ops.hip $H/gdn.hip $H/27b/attn.hip $H/vision.hip $H/gpu.hip $G/27b/model.cpp $G/27b/load.cpp $G/27b/api.cpp)
    local FN_SRC=($H/flash-next/relayout.hip $H/flash-next/quant.hip $H/flash-next/ops.hip $H/gdn.hip $H/flash-next/attn.hip $H/flash-next/moe.hip $H/flash-next/ple.hip
                  $H/vision.hip $H/gpu.hip $G/flash-next/model.cpp $G/flash-next/load.cpp $H/flash-next/experts.cpp $G/flash-next/api.cpp $G/flash-next/cpu_moe.cpp)
    mkdir -p /out/build
    $ROCM/lib/llvm/bin/clang++ -O3 -std=c++20 -fPIC -ffunction-sections -fdata-sections -fvisibility=hidden -fvisibility-inlines-hidden --gcc-toolchain=$GCC \
        -c -o "$W/cpu_exact.o" engine/cpu_exact.cpp || return 1
    $ROCM/bin/hipcc "${FLAGS[@]}" -I$H/27b -I$H -I$G/27b -x hip -shared -s -o /out/build/libchrysalis.so "${SRC[@]}" "${LIBS[@]}" & a=$!
    $ROCM/bin/hipcc "${FLAGS[@]}" -I$H/flash-next -I$H -I$G/flash-next -I$W/llama.cpp/ggml/include -shared -s -o /out/build/libchrysalis_fn.so -x hip "${FN_SRC[@]}" -x none "$W/cpu_exact.o" \
        "${LIBS[@]}" -L$W/ggml-avx2/bin -lggml-cpu -lggml-base & b=$!
    $ROCM/bin/hipcc "${FLAGS[@]}" -s -o /out/build/gpu $H/gpu_pick.hip $H/gpu.hip "${LIBS[@]}" & l=$!
    wait $a && wait $b && wait $l
    chown -R "$OWNER" /out
}

case "${1:-}" in
  ""|--host|--insecure|--model)
            setup
            build
            launch "$@" ;;
  get)      setup; launch "$@" ;;
  release)  release "${@:2}" ;;
  dmg)      dmg "${@:2}" ;;
  native)   native ;;
  *)        launch "$@" ;;
esac
