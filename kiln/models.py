"""The models kiln can serve, one at a time on the GPU (a request for the other one switches). chunk: tokens per prefill
pass (the 27B prefills in chunks of it; Flash-Next reads each of its experts in system RAM once per pass). snapshots:
prompt-cache snapshots kept in pinned RAM (153 MB each for the 27B; 115 MB for Flash-Next, whose experts already take
most of the RAM)."""
MODELS = {
    "qwen3.8-27b": dict(path="data/models/kiln-mix.gguf", lib="build/libkiln.so", draft=3, chunk=2048, snapshots=40),
    "qwen3.8-flash-next": dict(path="data/models/flash-next/kiln-fn.gguf", lib="build/libkiln_fn.so", draft=1, chunk=8192,
                               snapshots=16),
}
DEFAULT = "qwen3.8-27b"
