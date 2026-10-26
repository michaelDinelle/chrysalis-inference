MODELS = {
    "qwen3.8-27b": dict(path="data/models/chrysalis-mix.gguf", lib="build/libchrysalis.so", draft=3, lookup=7, chunk=2048, snapshots=40,
                        vision="data/models/vision/mmproj-27b-bf16.gguf", vision_stays=True, pictures=2 << 30, warm=True),
    "qwen3.8-flash-next": dict(path="data/models/flash-next/chrysalis-fn.gguf", lib="build/libchrysalis_fn.so", draft=2, lookup=3, chunk=8192,
                               snapshots=16, vision="data/models/vision/mmproj-fn-bf16.gguf", vision_stays=False,
                               pictures=128 << 20, upstream=("unsloth/Qwen3.8-Flash-Next-GGUF", "766911a6b7369840a91dbcd95f9f997acaab6cd6",
                                                             "data/models/flash-next/src", ["Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf"])),
}
DEFAULT = "qwen3.8-27b"
