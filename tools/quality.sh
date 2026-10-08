#!/bin/sh
# KL divergence of quantized GGUFs against the Q8_0 reference on a code + prose text (llama.cpp perplexity tool).
# usage: tools/quality.sh model.gguf [model2.gguf ...]
set -e
cd "$(dirname "$0")/.."
B=data/llama.cpp/build/bin
REF=data/models/Qwen3.8-27B-Q8_0.gguf
mkdir -p build
if [ ! -f build/eval.txt ]; then
    # ~40K tokens: C++ and Python sources plus English documentation
    cat data/llama.cpp/src/llama-sampler.cpp data/llama.cpp/tools/server/server-context.cpp kiln/*.py \
        data/llama.cpp/docs/build.md data/llama.cpp/tools/server/README.md | head -c 160000 > build/eval.txt
fi
[ -f build/q8.kld ] || $B/llama-perplexity -m "$REF" -f build/eval.txt -c 4096 -ngl 40 --kl-divergence-base build/q8.kld > build/q8.log 2>&1
for m in "$@"; do
    echo "== $m ($(du -h "$m" | cut -f1))"
    $B/llama-perplexity -m "$m" -f build/eval.txt -c 4096 -ngl 99 --kl-divergence-base build/q8.kld --kl-divergence 2>&1 \
        | grep -E "^Mean +KLD|^Same top p|^RMS Δp|^99.0% +KLD"
done
