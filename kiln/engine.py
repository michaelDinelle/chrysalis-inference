"""ctypes binding to the engines' C API: build/libkiln.so (27B) or build/libkiln_fn.so (Flash-Next), per kiln/models.py."""
import ctypes as C
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LT_CACHE = os.path.join(ROOT, "data", "cache", "hipblaslt.txt")   # the GEMM algorithm per shape, timed once (engine/lt_tune.h)


class KVFull(RuntimeError):
    """The KV cache has no free pages left for this request."""


class MissingImage(ValueError):
    """A prompt's image tokens name a picture the engine has not encoded (kiln_image_encode)."""


IMG_BASE, IMG_ROWS = 1 << 30, 1 << 14   # image token ids: IMG_BASE + picture slot * IMG_ROWS + row (engine/model.h)


STAGES = ("top_k", "temperature", "top_p", "min_p")   # the sampler's stages, by their ids in engine/sampler.h


class Sampling(C.Structure):
    """Mirrors struct Sampling in engine/sampler.h. ban: a token id never drawn (-1: none). order: the stages' ids
    in the order they act. dry_multiplier 0 turns DRY off. lookup: drafts copied from earlier in the conversation
    (Engine.set_lookup), at most; 0 leaves drafting to the MTP head. halt: a token that ends a step once emitted
    (-1: none)."""
    _fields_ = [("temperature", C.c_float), ("top_p", C.c_float), ("min_p", C.c_float),
                ("presence_penalty", C.c_float), ("top_k", C.c_int), ("n_draft", C.c_int), ("ban", C.c_int),
                ("dry_multiplier", C.c_float), ("dry_base", C.c_float), ("dry_allowed_length", C.c_int),
                ("order", C.c_int * 4), ("lookup", C.c_int), ("halt", C.c_int)]

    def __init__(self, order=STAGES, **kw):
        kw.setdefault("ban", -1)
        kw.setdefault("halt", -1)
        kw.setdefault("dry_base", 1.75)
        kw.setdefault("dry_allowed_length", 2)
        super().__init__(**kw)
        self.order[:] = [STAGES.index(s) for s in order]


def wake_gpus():
    """Opening a GPU's render node resumes it from runtime suspend, ~0.65 s on the 7900 XTX that hipInit would
    otherwise wait for: start that before the imports and the library load. The node is closed right away (the GPU
    stays up for its autosuspend delay, 5 s)."""
    import glob
    import threading

    def wake(node):
        try:
            os.close(os.open(node, os.O_RDWR))
        except OSError:
            pass
    for node in glob.glob("/dev/dri/renderD*"):
        threading.Thread(target=wake, args=(node,), daemon=True).start()


_loading = None                  # the library whose kiln_init / kiln_warmup is running


def load_progress():
    """0..1 while an engine loads (bytes of weights read, then warmup), None when nothing is loading."""
    lib = _loading
    if lib is None or not hasattr(lib, "kiln_load_progress"):
        return None
    lib.kiln_load_progress.restype = C.c_float
    return round(float(lib.kiln_load_progress()), 3)


def load_async(*args, **kw):
    """Engine(*args, **kw) in a thread (the C++ load runs without the GIL): returns a future."""
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)
    future = pool.submit(Engine, *args, **kw)
    pool.shutdown(wait=False)
    return future


class Engine:
    """warmup=False leaves the last step of loading (token embeddings, snapshot buffers, first kernel launches) to a
    later warmup() call, which must come before any other use: the server runs it after it starts listening."""

    def __init__(self, model, max_seqs=4, umax=2048, reserve_mb=256, lib=os.path.join(ROOT, "build", "libkiln.so"), warmup=True):
        self.lib = L = C.CDLL(lib)
        L.kiln_init.restype = C.c_int
        L.kiln_init.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_longlong]
        L.kiln_prefill.argtypes = [C.c_int, C.POINTER(C.c_int), C.c_int]
        L.kiln_prefill.restype = C.c_int
        self.vision = hasattr(L, "kiln_image_encode")
        if self.vision:
            L.kiln_prefill_pos.argtypes = [C.c_int, C.POINTER(C.c_int), C.c_void_p, C.c_int]
            L.kiln_prefill_pos.restype = C.c_int
            L.kiln_vision_open.argtypes = [C.c_char_p]
            L.kiln_image_encode.argtypes = [C.c_int, C.c_void_p, C.c_int, C.c_int]
            L.kiln_image_encode.restype = C.c_int
            L.kiln_image_free.argtypes = [C.c_int]
            L.kiln_video_encode.argtypes = [C.c_int, C.c_void_p, C.c_int, C.c_int]
            L.kiln_video_encode.restype = C.c_int
            L.kiln_image_read.argtypes = [C.c_int, C.c_void_p]
            L.kiln_image_read.restype = C.c_int
        self.persistent = hasattr(L, "kiln_snap_import")    # snapshots it can write out and read back (export_snapshot)
        if self.persistent:
            L.kiln_snap_bytes.argtypes, L.kiln_snap_bytes.restype = [C.c_int], C.c_int64
            L.kiln_snap_export.argtypes, L.kiln_snap_export.restype = [C.c_int, C.c_void_p], C.c_int
            L.kiln_snap_import.argtypes, L.kiln_snap_import.restype = [C.c_void_p, C.c_int64], C.c_int
        L.kiln_generate.argtypes = [C.c_int, C.POINTER(Sampling), C.POINTER(C.c_int), C.c_int, C.c_void_p, C.c_void_p]
        L.kiln_draw.argtypes, L.kiln_draw.restype = [C.c_int, C.c_void_p, C.c_void_p, C.POINTER(Sampling)], C.c_int
        L.kiln_masked_top.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
        self.masked = hasattr(L, "kiln_generate_masked")    # held answers step with drafts (draft, generate_masked)
        if self.masked:
            L.kiln_draft.argtypes = [C.c_int, C.POINTER(Sampling), C.c_void_p, C.c_int, C.c_int, C.c_void_p]
            L.kiln_generate_masked.argtypes = [C.c_int, C.POINTER(Sampling), C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int]
            L.kiln_seq_pending.argtypes = [C.c_int, C.c_int]
        self.batched = hasattr(L, "kiln_generate_batch")     # steps of several sequences in one pass (generate_batch)
        if self.batched:
            L.kiln_generate_batch.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p]
        L.kiln_seq_context.argtypes = [C.c_int, C.POINTER(C.c_int), C.c_int]
        L.kiln_dry_breakers.argtypes = [C.POINTER(C.c_int), C.c_int]
        L.kiln_seq_begin.argtypes = [C.c_int, C.c_uint64]
        L.kiln_last_logits.argtypes = [C.POINTER(C.c_float)]
        if "KILN_LT_CACHE" not in os.environ:
            os.makedirs(os.path.dirname(LT_CACHE), exist_ok=True)
            os.environ["KILN_LT_CACHE"] = LT_CACHE
        global _loading
        _loading = L
        try:
            self.capacity = L.kiln_init(model.encode(), max_seqs, umax, reserve_mb)
            if warmup:
                self.warmup()
        finally:
            _loading = None

    def warmup(self):
        global _loading
        _loading = self.lib
        try:
            self.lib.kiln_warmup()
        finally:
            _loading = None

    def reserve_snapshots(self, n):
        """Pinned buffers for n snapshots, made by warmup() (or now, if it already ran), so no request waits for one."""
        if hasattr(self.lib, "kiln_reserve_snapshots"):
            self.lib.kiln_reserve_snapshots(n)

    def shutdown(self):
        """Release the GPU and the engine's host memory so another engine can load (model switch)."""
        if hasattr(self.lib, "kiln_shutdown"):
            self.lib.kiln_shutdown.argtypes = [C.c_int]
            self.lib.kiln_shutdown(0)

    def new_seq(self):
        sid = self.lib.kiln_seq_new()
        if sid < 0:
            raise RuntimeError("no free sequence slot")
        return sid

    def free_seq(self, sid):
        self.lib.kiln_seq_free(sid)

    def seq_len(self, sid):
        return self.lib.kiln_seq_len(sid)

    def truncate(self, sid, n):
        self.lib.kiln_seq_truncate(sid, n)

    def prefill(self, sid, tokens, pos=None):
        """pos: None for text, else the tokens' M-RoPE positions, int32 [n][3] (kiln/vision.py)."""
        arr = (C.c_int * len(tokens))(*tokens)
        if pos is None:
            r = self.lib.kiln_prefill(sid, arr, len(tokens))
        else:
            import numpy as np
            pos = np.ascontiguousarray(pos, dtype=np.int32)
            assert pos.shape == (len(tokens), 3)
            r = self.lib.kiln_prefill_pos(sid, arr, pos.ctypes.data, len(tokens))
        if r == -2:
            raise MissingImage("the prompt names a picture that is not encoded")
        if r < 0:
            raise KVFull()

    def vision_open(self, path):
        """The vision tower's GGUF (its weights load with the first picture)."""
        self.lib.kiln_vision_open(path.encode())

    def image_encode(self, slot, patches, grid):
        """A picture's embeddings under `slot` for its image tokens; patches: kiln.vision.patches(). False if it is too
        large for the engine's scratch memory."""
        import numpy as np
        patches = np.ascontiguousarray(patches, dtype=np.float32)
        return self.lib.kiln_image_encode(slot, patches.ctypes.data, grid[0], grid[1]) == 0

    def video_encode(self, slot, patches, grid):
        """A video's frame pair under `slot`, as image_encode; patches: kiln.vision.pair_patches()'s for the pair."""
        import numpy as np
        patches = np.ascontiguousarray(patches, dtype=np.float32)
        return self.lib.kiln_video_encode(slot, patches.ctypes.data, grid[0], grid[1]) == 0

    def image_free(self, slot):
        self.lib.kiln_image_free(slot)

    def vision_release(self):
        """The vision tower's weights out of pinned RAM (the next picture reads them again)."""
        self.lib.kiln_vision_release()

    def image_read(self, slot, rows, width):
        """A picture's embeddings, float32 [rows][width] (for tests)."""
        import numpy as np
        out = np.empty((rows, width), np.float32)
        return out[:self.lib.kiln_image_read(slot, out.ctypes.data)]

    def begin(self, sid, seed):
        """Start a reply after the prompt is prefilled: seeds the sampler and resets the presence penalty."""
        self.lib.kiln_seq_begin(sid, seed)

    def generate(self, sid, sampling, max_out=16, rows=False):
        """The next tokens; with rows, also each one's candidates (logits [n][64], ids [n][64]) for log probs."""
        import numpy as np
        out = (C.c_int * max_out)()
        rv = np.empty((max_out, 64), np.float32) if rows else None
        ri = np.empty((max_out, 64), np.int32) if rows else None
        n = self.lib.kiln_generate(sid, C.byref(sampling), out, max_out, rv.ctypes.data if rows else None, ri.ctypes.data if rows else None)
        if n < 0:
            raise KVFull()
        if n > max_out:
            raise RuntimeError(f"kiln_generate wrote {n} tokens into room for {max_out}")
        return (list(out[:n]), rv[:n], ri[:n]) if rows else list(out[:n])

    def masked_top(self, bitmask):
        """The 64 largest of the last prefill's logits among the tokens a bitmask allows (uint32 words, bit t % 32 of word
        t / 32): (float32 logits, int32 ids), -inf past the allowed ones."""
        import numpy as np
        bitmask = np.ascontiguousarray(bitmask, np.uint32)
        v, ids = np.empty(64, np.float32), np.empty(64, np.int32)
        self.lib.kiln_masked_top(bitmask.ctypes.data, v.ctypes.data, ids.ctypes.data)
        return v, ids

    def draw(self, sid, values, ids, sampling):
        """A token drawn from 64 candidates (float32 logits, int32 ids; -inf for none) as generate() draws, with the
        sequence's random state; it counts as the reply's (presence penalty, DRY). It still has to be prefilled."""
        import numpy as np
        values = np.ascontiguousarray(values, np.float32)
        ids = np.ascontiguousarray(ids, np.int32)
        assert len(values) == len(ids) == 64
        return self.lib.kiln_draw(sid, values.ctypes.data, ids.ctypes.data, C.byref(sampling))

    def set_pending(self, sid, tok):
        """tok (from draw()) becomes the sequence's pending token: its next step reads it, where a prefill would."""
        self.lib.kiln_seq_pending(sid, tok)

    def draft(self, sid, sampling, fixed, max_k):
        """Drafts for a held answer's next step from its pending token: `fixed` (the tokens its grammar forces next) first,
        then the MTP head's guesses after them, at most max_k (up to 7) in all."""
        import numpy as np
        fixed = np.ascontiguousarray(fixed, np.int32)
        out = np.empty(8, np.int32)
        n = self.lib.kiln_draft(sid, C.byref(sampling), fixed.ctypes.data, len(fixed), max_k, out.ctypes.data)
        if n < 0:
            raise KVFull()
        return out[:n].tolist()

    def generate_masked(self, sid, sampling, drafts, masks, max_out=8):
        """One step of a held answer: the pending token and drafts verified with masks [len(drafts) + 1][VOCAB / 32]
        (uint32 bitmasks) keeping each position to what its grammar allows; the tokens emitted, the last one pending."""
        import numpy as np
        drafts = np.ascontiguousarray(drafts, np.int32)
        masks = np.ascontiguousarray(masks, np.uint32)
        assert masks.shape[0] == len(drafts) + 1
        out = np.empty(8, np.int32)
        n = self.lib.kiln_generate_masked(sid, C.byref(sampling), drafts.ctypes.data, len(drafts), masks.ctypes.data,
                                          out.ctypes.data, max_out)
        if n < 0:
            raise KVFull()
        return out[:n].tolist()

    def generate_batch(self, sids, samplings, max_out=16):
        """One speculative step for each sequence, all in one pass (up to 8 sequences): their token lists."""
        import numpy as np
        n = len(sids)
        ids = np.ascontiguousarray(sids, np.int32)
        sps = (Sampling * n)(*samplings)
        out, counts = np.empty((n, max_out), np.int32), np.empty(n, np.int32)
        if self.lib.kiln_generate_batch(n, ids.ctypes.data, C.cast(sps, C.c_void_p), out.ctypes.data, max_out, counts.ctypes.data) < 0:
            raise KVFull()
        return [out[i, :counts[i]].tolist() for i in range(n)]

    def set_lookup(self, sid, tokens):
        """The conversation so far (int32 array), where the next reply's drafts may be copied from (prompt lookup)."""
        import numpy as np
        if hasattr(self.lib, "kiln_seq_lookup"):
            tokens = np.ascontiguousarray(tokens, dtype=np.int32)
            self.lib.kiln_seq_lookup(sid, C.c_void_p(tokens.ctypes.data), len(tokens))

    def set_context(self, sid, tokens):
        """The prompt's tail that DRY looks back over in the next reply."""
        arr = (C.c_int * len(tokens))(*tokens)
        self.lib.kiln_seq_context(sid, arr, len(tokens))

    def set_dry_breakers(self, ids):
        arr = (C.c_int * len(ids))(*ids)
        self.lib.kiln_dry_breakers(arr, len(ids))

    def snapshot(self, sid):
        """Capture the sequence's state at its current length; returns a snapshot id."""
        return self.lib.kiln_snapshot(sid)

    def restore(self, sid, snap):
        """Make sequence `sid` identical to the snapshot (shares its KV pages copy-on-write)."""
        if self.lib.kiln_restore(sid, snap) < 0:
            raise RuntimeError(f"snapshot {snap} is not in use")

    def free_snapshot(self, snap):
        if self.lib.kiln_snap_free(snap) < 0:
            raise RuntimeError(f"snapshot {snap} is not in use")

    def export_snapshot(self, snap):
        """A snapshot as bytes (uint8 array: its state and KV pages) that import_snapshot takes back in a later run of
        the same model and engine build."""
        import numpy as np
        n = self.lib.kiln_snap_bytes(snap)
        buf = np.empty(max(n, 0), np.uint8)
        if n < 0 or self.lib.kiln_snap_export(snap, buf.ctypes.data) < 0:
            raise RuntimeError(f"snapshot {snap} is not in use")
        return buf

    def import_snapshot(self, data):
        """A snapshot from export_snapshot's bytes; None if they are not this engine's layout. Raises KVFull when the KV
        pool has no room for it."""
        import numpy as np
        data = np.ascontiguousarray(np.frombuffer(data, np.uint8))
        snap = self.lib.kiln_snap_import(data.ctypes.data, len(data))
        if snap == -2:
            raise KVFull()
        return None if snap < 0 else snap

    def free_tokens(self):
        return self.lib.kiln_free_tokens()

    def last_logits(self):
        import numpy as np
        buf = np.empty(248320, dtype=np.float32)
        self.lib.kiln_last_logits(buf.ctypes.data_as(C.POINTER(C.c_float)))
        return buf
