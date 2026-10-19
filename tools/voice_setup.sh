#!/usr/bin/env bash
# kiln's voice service (voice/): its venv, data/voice-venv, and its models in data/voice/<name>/. What is there already
# is kept, so running it again only fetches what is missing.
#   tools/voice_setup.sh           the defaults: silero, parakeet-110m, smart-turn, kokoro
#   tools/voice_setup.sh NAME...   other voices: kitten supertonic pocket
set -euo pipefail
ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"
V="$ROOT/data/voice"
VENV="$ROOT/data/voice-venv"
SHERPA=https://github.com/k2-fsa/sherpa-onnx/releases/download

if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q onnxruntime==1.31.0 sherpa-onnx==1.13.8 kokoro-onnx==0.4.7 numpy==2.5.4 websockets==17.2
fi
mkdir -p "$V"

# a sherpa-onnx release archive, unpacked as data/voice/<name>
sherpa() {
    [ -d "$V/$1" ] && return
    local tmp; tmp=$(mktemp -d "$V/.fetch.XXXXXX")
    echo "voice: fetching $1"
    curl -fsSL "$SHERPA/$2/$3.tar.bz2" | tar xj -C "$tmp"
    mv "$tmp/$3" "$V/$1" && rmdir "$tmp"
}
# one file as data/voice/<name>/<file>
file() {
    [ -f "$V/$1/$3" ] && return
    mkdir -p "$V/$1"
    echo "voice: fetching $1/$3"
    curl -fsSL -o "$V/$1/$3.part" "$2" && mv "$V/$1/$3.part" "$V/$1/$3"
}

fetch() {
    case "$1" in
        silero) file silero https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad_16k_op15.onnx silero_vad.onnx ;;
        smart-turn) file smart-turn https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx smart-turn-v3.2-cpu.onnx ;;
        kokoro)
            file kokoro https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx kokoro-v1.0.int8.onnx
            file kokoro https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin voices-v1.0.bin ;;
        parakeet-110m) sherpa parakeet-110m asr-models sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8 ;;
        kitten) sherpa kitten tts-models kitten-micro-en-v0_8 ;;
        supertonic) sherpa supertonic tts-models sherpa-onnx-supertonic-3-tts-int8-2026-05-11 ;;
        pocket) sherpa pocket tts-models sherpa-onnx-pocket-tts-int8-2026-01-26 ;;
        *) echo "voice: no model named $1" >&2; exit 1 ;;
    esac
}

if [ $# -eq 0 ]; then set -- silero parakeet-110m smart-turn kokoro; fi
for name in "$@"; do fetch "$name"; done
