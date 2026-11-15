import os
import sys

WINDOWS = sys.platform == "win32"
NVIDIA = os.path.join(os.environ.get("SystemRoot", ""), "System32", "nvcuda.dll") if WINDOWS else "/dev/nvidiactl"
BACKEND = os.environ.get("CHRYSALIS_BACKEND") or ("metal" if sys.platform == "darwin" else "cuda" if os.path.exists(NVIDIA) else "hip")
LIB = ".dylib" if BACKEND == "metal" else ".dll" if WINDOWS else ".so"
MODELS = {
    "qwen3.8-27b": dict(path="data/models/qwen3.8-27b/qwen3.8-27b.gguf", lib="build/libchrysalis" + LIB, draft=3, lookup=7, chunk=2048, snapshots=40,
                        vision="data/models/qwen3.8-27b/mmproj.gguf", vision_stays=True, pictures=2 << 30, warm=True),
    "qwen3.8-flash-next": dict(path="data/models/qwen3.8-flash-next/qwen3.8-flash-next.gguf", lib="build/libchrysalis_fn" + LIB, draft=2, lookup=3, chunk=8192,
                               snapshots=16, vision="data/models/qwen3.8-flash-next/mmproj.gguf", vision_stays=False,
                               pictures=128 << 20, upstream=("unsloth/Qwen3.8-Flash-Next-GGUF", "766911a6b7369840a91dbcd95f9f997acaab6cd6",
                                                             "data/models/qwen3.8-flash-next/src", ["Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf"])),
}
ON = {"metal": {"qwen3.8-27b": dict(snapshots=12, warm=False), "qwen3.8-flash-next": dict(vision=None)}}
for name, tuned in ON.get(BACKEND, {}).items():
    MODELS[name].update(tuned)
DEFAULT = "qwen3.8-27b"
