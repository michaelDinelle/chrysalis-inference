"""ctypes binding to build/libkiln.so."""
import ctypes as C
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class KVFull(RuntimeError):
    """The KV cache has no free pages left for this request."""


class Sampling(C.Structure):
    _fields_ = [("temperature", C.c_float), ("top_p", C.c_float), ("min_p", C.c_float),
                ("presence_penalty", C.c_float), ("top_k", C.c_int), ("n_draft", C.c_int)]


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
        L.kiln_generate.argtypes = [C.c_int, C.POINTER(Sampling), C.POINTER(C.c_int), C.c_int]
        L.kiln_seq_begin.argtypes = [C.c_int, C.c_uint64]
        L.kiln_last_logits.argtypes = [C.POINTER(C.c_float)]
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

    def prefill(self, sid, tokens):
        arr = (C.c_int * len(tokens))(*tokens)
        if self.lib.kiln_prefill(sid, arr, len(tokens)) < 0:
            raise KVFull()

    def begin(self, sid, seed):
        """Start a reply after the prompt is prefilled: seeds the sampler and resets the presence penalty."""
        self.lib.kiln_seq_begin(sid, seed)

    def generate(self, sid, sampling, max_out=16):
        out = (C.c_int * max_out)()
        n = self.lib.kiln_generate(sid, C.byref(sampling), out, max_out)
        if n < 0:
            raise KVFull()
        return list(out[:n])

    def snapshot(self, sid):
        """Capture the sequence's state at its current length; returns a snapshot id."""
        return self.lib.kiln_snapshot(sid)

    def restore(self, sid, snap):
        """Make sequence `sid` identical to the snapshot (shares its KV pages copy-on-write)."""
        self.lib.kiln_restore(sid, snap)

    def free_snapshot(self, snap):
        self.lib.kiln_snap_free(snap)

    def free_tokens(self):
        return self.lib.kiln_free_tokens()

    def last_logits(self):
        import numpy as np
        buf = np.empty(248320, dtype=np.float32)
        self.lib.kiln_last_logits(buf.ctypes.data_as(C.POINTER(C.c_float)))
        return buf
