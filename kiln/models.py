"""The models kiln can serve, one at a time on the GPU (a request for the other one switches). chunk: tokens per prefill
pass (the 27B prefills in chunks of it; Flash-Next reads each of its experts in system RAM once per pass). snapshots:
prompt-cache snapshots kept in pinned RAM (153 MB each for the 27B; 115 MB for Flash-Next, whose experts already take
most of the RAM). vision: the vision tower's GGUF (pictures are read only when it is there); vision_stays: its weights
(0.9 GB) stay in pinned RAM between pictures, else they are read again for each request with new pictures (~0.35 s), as
Flash-Next does: its n-gram table pages need the room; pictures: bytes of picture embeddings kept in pinned RAM for
prefills (a 1080p screenshot's: 21 MB for the 27B, 10 MB for Flash-Next), the least recently used dropped beyond it.
lookup: drafts copied from earlier in the conversation per step, at most, in place of the MTP head's where the last tokens
appeared before (0: MTP only; Flash-Next's verify pass takes 3 drafts at most). warm: while this model runs, the others'
files are read into the page cache for a switch (kiln/warm.py)."""
MODELS = {
    "qwen3.8-27b": dict(path="data/models/kiln-mix.gguf", lib="build/libkiln.so", draft=3, lookup=7, chunk=2048, snapshots=40,
                        vision="data/models/vision/mmproj-27b-bf16.gguf", vision_stays=True, pictures=2 << 30, warm=True),
    "qwen3.8-flash-next": dict(path="data/models/flash-next/kiln-fn.gguf", lib="build/libkiln_fn.so", draft=2, lookup=3, chunk=8192,
                               snapshots=16, vision="data/models/vision/mmproj-fn-bf16.gguf", vision_stays=False,
                               pictures=128 << 20),
}
DEFAULT = "qwen3.8-27b"
