"""ctypes binding to build/libkiln.so."""
import ctypes as C
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class KVFull(RuntimeError):
    """The KV cache has no free pages left for this request."""


class Sampling(C.Structure):
    _fields_ = [("temperature", C.c_float), ("top_p", C.c_float), ("min_p", C.c_float),
                ("presence_penalty", C.c_float), ("top_k", C.c_int), ("n_draft", C.c_int), ("seed", C.c_uint64)]


class Engine:
    def __init__(self, model, max_seqs=4, umax=2048, reserve_mb=256, lib=os.path.join(ROOT, "build", "libkiln.so")):
        self.lib = L = C.CDLL(lib)
        L.kiln_init.restype = C.c_int
        L.kiln_init.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_longlong]
        L.kiln_prefill.argtypes = [C.c_int, C.POINTER(C.c_int), C.c_int]
        L.kiln_prefill.restype = C.c_int
        L.kiln_generate.argtypes = [C.c_int, C.POINTER(Sampling), C.POINTER(C.c_int), C.c_int]
        L.kiln_seq_seed.argtypes = [C.c_int, C.c_uint64]
        L.kiln_last_logits.argtypes = [C.POINTER(C.c_float)]
        self.capacity = L.kiln_init(model.encode(), max_seqs, umax, reserve_mb)

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
