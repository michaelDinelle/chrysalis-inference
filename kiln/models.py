"""The models kiln can serve, one at a time on the GPU (a request for the other one switches)."""
MODELS = {
    "qwen3.8-27b": dict(path="data/models/kiln-mix.gguf", lib="build/libkiln.so", draft=3),
    "qwen3.8-flash-next": dict(path="data/models/flash-next/kiln-fn.gguf", lib="build/libkiln_fn.so", draft=1),
}
DEFAULT = "qwen3.8-27b"
