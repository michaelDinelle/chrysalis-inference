MODELS = {
    "qwen3.8-27b": dict(path="data/models/chrysalis-mix.gguf", lib="build/libchrysalis.so", draft=3, lookup=7, chunk=2048, snapshots=40,
                        vision="data/models/vision/mmproj-27b-bf16.gguf", vision_stays=True, pictures=2 << 30, warm=True),
    "qwen3.8-flash-next": dict(path="data/models/flash-next/chrysalis-fn.gguf", lib="build/libchrysalis_fn.so", draft=2, lookup=3, chunk=8192,
                               snapshots=16, vision="data/models/vision/mmproj-fn-bf16.gguf", vision_stays=False,
                               pictures=128 << 20),
}
DEFAULT = "qwen3.8-27b"
