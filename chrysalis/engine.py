import ctypes as C
import os

from .models import BACKEND, LIB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LT_CACHE = os.path.join(ROOT, "data", "cache", "cublaslt.txt" if BACKEND == "cuda" else "hipblaslt.txt")
CALIBRATION = os.path.join(ROOT, "data", "cache", "flash-next-calibration.txt")


class KVFull(RuntimeError):
    pass


class MissingImage(ValueError):
    pass


IMG_BASE, IMG_ROWS = 1 << 30, 1 << 14


STAGES = ("penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature")
MAX_ORDER, NSTAT = 32, 6


class Sampling(C.Structure):
    _fields_ = [("temperature", C.c_float), ("dynatemp_range", C.c_float), ("dynatemp_exponent", C.c_float), ("top_k", C.c_int),
                ("top_p", C.c_float), ("min_p", C.c_float), ("typical_p", C.c_float), ("xtc_probability", C.c_float),
                ("xtc_threshold", C.c_float), ("top_n_sigma", C.c_float), ("repeat_last_n", C.c_int), ("repeat_penalty", C.c_float),
                ("frequency_penalty", C.c_float), ("presence_penalty", C.c_float), ("dry_multiplier", C.c_float), ("dry_base", C.c_float),
                ("dry_allowed_length", C.c_int), ("dry_penalty_last_n", C.c_int), ("mirostat", C.c_int), ("mirostat_tau", C.c_float),
                ("mirostat_eta", C.c_float), ("min_keep", C.c_int), ("n_order", C.c_int), ("order", C.c_int * MAX_ORDER),
                ("n_draft", C.c_int), ("lookup", C.c_int), ("ban", C.c_int), ("halt", C.c_int)]
    DEFAULTS = dict(temperature=1.0, dynatemp_exponent=1.0, top_k=20, top_p=0.95, typical_p=1.0, xtc_threshold=0.1, top_n_sigma=-1.0,
                    repeat_last_n=64, repeat_penalty=1.0, dry_base=1.75, dry_allowed_length=2, dry_penalty_last_n=-1, mirostat_tau=5.0,
                    mirostat_eta=0.1, n_draft=3, ban=-1, halt=-1)

    def __init__(self, order=STAGES, **kw):
        if len(order) > MAX_ORDER:
            raise ValueError(f"samplers lists at most {MAX_ORDER} samplers")
        super().__init__(**{**self.DEFAULTS, **kw})
        self.n_order = len(order)
        self.order[:len(order)] = [STAGES.index(s) for s in order]


def wake_gpus():
    import glob
    import threading

    def wake(node):
        try:
            os.close(os.open(node, os.O_RDWR))
        except OSError:
            pass
    for node in glob.glob("/dev/dri/renderD*"):
        threading.Thread(target=wake, args=(node,), daemon=True).start()


_loading = None


def load_progress():
    lib = _loading
    if lib is None or not hasattr(lib, "chrysalis_load_progress"):
        return None
    lib.chrysalis_load_progress.restype = C.c_float
    return round(float(lib.chrysalis_load_progress()), 3)


def load_async(*args, **kw):
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)
    future = pool.submit(Engine, *args, **kw)
    pool.shutdown(wait=False)
    return future


class Engine:

    def __init__(self, model, max_seqs=4, umax=2048, reserve_mb=256, lib=os.path.join(ROOT, "build", "libchrysalis" + LIB), warmup=True):
        if hasattr(os, "add_dll_directory"):
            for folder in (os.path.dirname(os.path.abspath(lib)), *(os.path.join(os.environ[k], "bin") for k in ("HIP_PATH", "CUDA_PATH") if os.environ.get(k))):
                if os.path.isdir(folder):
                    os.add_dll_directory(folder)
        self.lib = L = C.CDLL(lib)
        L.chrysalis_init.restype = C.c_int
        L.chrysalis_init.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_longlong]
        L.chrysalis_prefill.argtypes = [C.c_int, C.POINTER(C.c_int), C.c_int]
        L.chrysalis_prefill.restype = C.c_int
        self.vision = hasattr(L, "chrysalis_image_encode")
        if self.vision:
            L.chrysalis_prefill_pos.argtypes = [C.c_int, C.POINTER(C.c_int), C.c_void_p, C.c_int]
            L.chrysalis_prefill_pos.restype = C.c_int
            L.chrysalis_vision_open.argtypes = [C.c_char_p]
            L.chrysalis_image_encode.argtypes = [C.c_int, C.c_void_p, C.c_int, C.c_int]
            L.chrysalis_image_encode.restype = C.c_int
            L.chrysalis_image_free.argtypes = [C.c_int]
            L.chrysalis_video_encode.argtypes = [C.c_int, C.c_void_p, C.c_int, C.c_int]
            L.chrysalis_video_encode.restype = C.c_int
            L.chrysalis_image_read.argtypes = [C.c_int, C.c_void_p]
            L.chrysalis_image_read.restype = C.c_int
        self.persistent = hasattr(L, "chrysalis_snap_import")
        if self.persistent:
            L.chrysalis_snap_bytes.argtypes, L.chrysalis_snap_bytes.restype = [C.c_int], C.c_int64
            L.chrysalis_snap_export.argtypes, L.chrysalis_snap_export.restype = [C.c_int, C.c_void_p], C.c_int
            L.chrysalis_snap_import.argtypes, L.chrysalis_snap_import.restype = [C.c_int, C.c_int64, C.c_int64], C.c_int
        L.chrysalis_generate.argtypes = [C.c_int, C.POINTER(Sampling), C.POINTER(C.c_int), C.c_int, C.c_void_p, C.c_void_p]
        L.chrysalis_draw.argtypes, L.chrysalis_draw.restype = [C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(Sampling)], C.c_int
        L.chrysalis_masked_top.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
        self.masked = hasattr(L, "chrysalis_generate_masked")
        if self.masked:
            L.chrysalis_draft.argtypes = [C.c_int, C.POINTER(Sampling), C.c_void_p, C.c_int, C.c_int, C.c_void_p]
            L.chrysalis_generate_masked.argtypes = [C.c_int, C.POINTER(Sampling), C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int]
            L.chrysalis_seq_pending.argtypes = [C.c_int, C.c_int]
        self.batched = hasattr(L, "chrysalis_generate_batch")
        if self.batched:
            L.chrysalis_generate_batch.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p]
        L.chrysalis_seq_context.argtypes = [C.c_int, C.c_void_p, C.c_int]
        L.chrysalis_seq_breakers.argtypes = [C.c_int, C.c_void_p, C.c_int]
        L.chrysalis_seq_bias.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_int]
        L.chrysalis_seq_push.argtypes = [C.c_int, C.c_void_p, C.c_int]
        L.chrysalis_seq_begin.argtypes = [C.c_int, C.c_uint64, C.POINTER(Sampling)]
        L.chrysalis_last_logits.argtypes = [C.POINTER(C.c_float)]
        if "CHRYSALIS_LT_CACHE" not in os.environ:
            os.makedirs(os.path.dirname(LT_CACHE), exist_ok=True)
            os.environ["CHRYSALIS_LT_CACHE"] = LT_CACHE
        if hasattr(L, "chrysalis_calibration"):
            os.makedirs(os.path.dirname(CALIBRATION), exist_ok=True)
            L.chrysalis_calibration.argtypes = [C.c_char_p]
            L.chrysalis_calibration(CALIBRATION.encode())
        global _loading
        _loading = L
        try:
            self.capacity = L.chrysalis_init(model.encode(), max_seqs, umax, reserve_mb)
            if warmup:
                self.warmup()
        finally:
            _loading = None

    def warmup(self):
        global _loading
        _loading = self.lib
        try:
            self.lib.chrysalis_warmup()
        finally:
            _loading = None

    def reserve_snapshots(self, n):
        if hasattr(self.lib, "chrysalis_reserve_snapshots"):
            self.lib.chrysalis_reserve_snapshots(n)

    def shutdown(self):
        if hasattr(self.lib, "chrysalis_shutdown"):
            self.lib.chrysalis_shutdown.argtypes = [C.c_int]
            self.lib.chrysalis_shutdown(0)

    def new_seq(self):
        sid = self.lib.chrysalis_seq_new()
        if sid < 0:
            raise RuntimeError("no free sequence slot")
        return sid

    def free_seq(self, sid):
        self.lib.chrysalis_seq_free(sid)

    def seq_len(self, sid):
        return self.lib.chrysalis_seq_len(sid)

    def truncate(self, sid, n):
        self.lib.chrysalis_seq_truncate(sid, n)

    def prefill(self, sid, tokens, pos=None):
        arr = (C.c_int * len(tokens))(*tokens)
        if pos is None:
            r = self.lib.chrysalis_prefill(sid, arr, len(tokens))
        else:
            import numpy as np
            pos = np.ascontiguousarray(pos, dtype=np.int32)
            assert pos.shape == (len(tokens), 3)
            r = self.lib.chrysalis_prefill_pos(sid, arr, pos.ctypes.data, len(tokens))
        if r == -2:
            raise MissingImage("the prompt names a picture that is not encoded")
        if r < 0:
            raise KVFull()

    def vision_open(self, path):
        self.lib.chrysalis_vision_open(path.encode())

    def image_encode(self, slot, patches, grid):
        import numpy as np
        patches = np.ascontiguousarray(patches, dtype=np.float32)
        return self.lib.chrysalis_image_encode(slot, patches.ctypes.data, grid[0], grid[1]) == 0

    def video_encode(self, slot, patches, grid):
        import numpy as np
        patches = np.ascontiguousarray(patches, dtype=np.float32)
        return self.lib.chrysalis_video_encode(slot, patches.ctypes.data, grid[0], grid[1]) == 0

    def image_free(self, slot):
        self.lib.chrysalis_image_free(slot)

    def vision_release(self):
        self.lib.chrysalis_vision_release()

    def image_read(self, slot, rows, width):
        import numpy as np
        out = np.empty((rows, width), np.float32)
        return out[:self.lib.chrysalis_image_read(slot, out.ctypes.data)]

    def begin(self, sid, seed, sampling=None):
        self.lib.chrysalis_seq_begin(sid, seed, None if sampling is None else C.byref(sampling))

    def generate(self, sid, sampling, max_out=16, rows=False):
        import numpy as np
        out = (C.c_int * max_out)()
        rv = np.empty((max_out, 64), np.float32) if rows else None
        ri = np.empty((max_out, 64), np.int32) if rows else None
        n = self.lib.chrysalis_generate(sid, C.byref(sampling), out, max_out, rv.ctypes.data if rows else None, ri.ctypes.data if rows else None)
        if n < 0:
            raise KVFull()
        if n > max_out:
            raise RuntimeError(f"chrysalis_generate wrote {n} tokens into room for {max_out}")
        return (list(out[:n]), rv[:n], ri[:n]) if rows else list(out[:n])

    def masked_top(self, bitmask):
        import numpy as np
        bitmask = np.ascontiguousarray(bitmask, np.uint32)
        v, ids, stats = np.empty(64, np.float32), np.empty(64, np.int32), np.empty(NSTAT, np.float32)
        self.lib.chrysalis_masked_top(bitmask.ctypes.data, v.ctypes.data, ids.ctypes.data, stats.ctypes.data)
        return v, ids, stats

    def draw(self, sid, values, ids, stats, sampling):
        import numpy as np
        values = np.ascontiguousarray(values, np.float32)
        ids = np.ascontiguousarray(ids, np.int32)
        assert len(values) == len(ids) == 64
        stats = None if stats is None else np.ascontiguousarray(stats, np.float32)
        return self.lib.chrysalis_draw(sid, values.ctypes.data, ids.ctypes.data, None if stats is None else stats.ctypes.data,
                                       C.byref(sampling))

    def set_pending(self, sid, tok):
        self.lib.chrysalis_seq_pending(sid, tok)

    def draft(self, sid, sampling, fixed, max_k):
        import numpy as np
        fixed = np.ascontiguousarray(fixed, np.int32)
        out = np.empty(8, np.int32)
        n = self.lib.chrysalis_draft(sid, C.byref(sampling), fixed.ctypes.data, len(fixed), max_k, out.ctypes.data)
        if n < 0:
            raise KVFull()
        return out[:n].tolist()

    def generate_masked(self, sid, sampling, drafts, masks, max_out=8):
        import numpy as np
        drafts = np.ascontiguousarray(drafts, np.int32)
        masks = np.ascontiguousarray(masks, np.uint32)
        assert masks.shape[0] == len(drafts) + 1
        out = np.empty(8, np.int32)
        n = self.lib.chrysalis_generate_masked(sid, C.byref(sampling), drafts.ctypes.data, len(drafts), masks.ctypes.data,
                                          out.ctypes.data, max_out)
        if n < 0:
            raise KVFull()
        return out[:n].tolist()

    def generate_batch(self, sids, samplings, max_out=16, rows=False):
        import numpy as np
        n = len(sids)
        ids = np.ascontiguousarray(sids, np.int32)
        sps = (Sampling * n)(*samplings)
        out, counts = np.empty((n, max_out), np.int32), np.empty(n, np.int32)
        rv = np.empty((n, max_out, 64), np.float32) if rows else None
        ri = np.empty((n, max_out, 64), np.int32) if rows else None
        if self.lib.chrysalis_generate_batch(n, ids.ctypes.data, C.cast(sps, C.c_void_p), out.ctypes.data, max_out, counts.ctypes.data,
                                             rv.ctypes.data if rows else None, ri.ctypes.data if rows else None) < 0:
            raise KVFull()
        if rows:
            return [(out[i, :counts[i]].tolist(), rv[i, :counts[i]], ri[i, :counts[i]]) for i in range(n)]
        return [out[i, :counts[i]].tolist() for i in range(n)]

    def set_lookup(self, sid, tokens):
        import numpy as np
        if hasattr(self.lib, "chrysalis_seq_lookup"):
            tokens = np.ascontiguousarray(tokens, dtype=np.int32)
            self.lib.chrysalis_seq_lookup(sid, C.c_void_p(tokens.ctypes.data), len(tokens))

    def set_context(self, sid, tokens):
        import numpy as np
        tokens = np.ascontiguousarray(tokens, np.int32)
        self.lib.chrysalis_seq_context(sid, tokens.ctypes.data, len(tokens))

    def set_bias(self, sid, bias):
        import numpy as np
        ids = np.ascontiguousarray([t for t, _ in bias], np.int32)
        vals = np.ascontiguousarray([b for _, b in bias], np.float32)
        self.lib.chrysalis_seq_bias(sid, ids.ctypes.data, vals.ctypes.data, len(ids))

    def set_breakers(self, sid, flat):
        import numpy as np
        flat = np.ascontiguousarray(flat, np.int32)
        self.lib.chrysalis_seq_breakers(sid, flat.ctypes.data, len(flat))

    def push(self, sid, tokens):
        import numpy as np
        tokens = np.ascontiguousarray(tokens, np.int32)
        self.lib.chrysalis_seq_push(sid, tokens.ctypes.data, len(tokens))

    def snapshot(self, sid):
        return self.lib.chrysalis_snapshot(sid)

    def restore(self, sid, snap):
        if self.lib.chrysalis_restore(sid, snap) < 0:
            raise RuntimeError(f"snapshot {snap} is not in use")

    def free_snapshot(self, snap):
        if self.lib.chrysalis_snap_free(snap) < 0:
            raise RuntimeError(f"snapshot {snap} is not in use")

    def export_snapshot(self, snap):
        import numpy as np
        n = self.lib.chrysalis_snap_bytes(snap)
        buf = np.empty(max(n, 0), np.uint8)
        if n < 0 or self.lib.chrysalis_snap_export(snap, buf.ctypes.data) < 0:
            raise RuntimeError(f"snapshot {snap} is not in use")
        return buf

    def import_snapshot(self, f):
        at = f.tell()
        snap = self.lib.chrysalis_snap_import(f.fileno(), at, os.fstat(f.fileno()).st_size - at)
        if snap == -2:
            raise KVFull()
        return None if snap < 0 else snap

    def free_tokens(self):
        return self.lib.chrysalis_free_tokens()

    def pool_save(self, path):
        if not hasattr(self.lib, "chrysalis_pool_save"):
            return False
        self.lib.chrysalis_pool_save.argtypes = [C.c_char_p]
        return bool(self.lib.chrysalis_pool_save(path.encode()))

    def pool_warm(self, path):
        if not hasattr(self.lib, "chrysalis_pool_warm"):
            return -1
        self.lib.chrysalis_pool_warm.argtypes = [C.c_char_p]
        global _loading
        _loading = self.lib
        try:
            return self.lib.chrysalis_pool_warm(path.encode())
        finally:
            _loading = None

    def expert_stats(self):
        if not hasattr(self.lib, "chrysalis_expert_stats"):
            return None
        out = (C.c_int64 * 12)()
        self.lib.chrysalis_expert_stats(out)
        return {"placed": out[0], "placed_file_tier": out[1], "decode_picks": out[2], "decode_picks_in_pool": out[3],
                "dma_bytes_per_s": out[4] * 1e6, "gpu_share": out[5] / 1e6, "mirrored": out[6], "mirror_cap": out[7],
                "memory_limited": out[8], "waited_s": out[9] / 1e6, "waited_steps": out[10], "decode_steps": out[11]}

    def last_logits(self):
        import numpy as np
        buf = np.empty(248320, dtype=np.float32)
        self.lib.chrysalis_last_logits(buf.ctypes.data_as(C.POINTER(C.c_float)))
        return buf
