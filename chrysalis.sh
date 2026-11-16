#!/usr/bin/env bash
set -e
ROOT="$(dirname "$(readlink -f "$0")")"
MAC=; [ "$(uname)" = Darwin ] && MAC=1
export CHRYSALIS_HOME="$ROOT"
BUN="$ROOT/bin/bun"
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
  native)   native ;;
  *)        launch "$@" ;;
esac
