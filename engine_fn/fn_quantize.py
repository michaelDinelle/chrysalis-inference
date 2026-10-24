import argparse
import ctypes as C
import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "llama.cpp", "gguf-py"))
from gguf import GGUFReader

F32, F16, Q5_0, Q8_0, Q2_K, Q3_K, Q6_K, IQ4_NL, IQ4_XS, I32 = 0, 1, 6, 8, 10, 11, 14, 20, 23, 26
BLOCK = {Q5_0: (32, 22), Q8_0: (32, 34), Q2_K: (256, 84), Q3_K: (256, 110), Q6_K: (256, 210), IQ4_NL: (32, 18), IQ4_XS: (256, 136),
         F16: (1, 2), F32: (1, 4), I32: (1, 4)}
NL, NE, D, FE = 48, 512, 2560, 640

ggml = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "llama.cpp", "build", "bin", "libggml-base.so"))
ggml.ggml_quantize_chunk.restype = C.c_size_t
ggml.ggml_quantize_chunk.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_int64, C.c_int64, C.c_int64, C.c_void_p]
for t in (Q5_0, Q2_K, Q3_K, Q6_K, IQ4_NL, IQ4_XS, Q8_0):
    ggml.ggml_quantize_init(t)


def row_bytes(t, n):
    b, s = BLOCK[t]
    assert n % b == 0, (t, n)
    return n // b * s


def nbytes(t, ne):
    rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
    return row_bytes(t, ne[0]) * rows


def quantize(t, x, imatrix=None):
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows, n = x.shape
    if t == F32:
        return x.tobytes()
    if t == F16:
        return x.astype(np.float16).tobytes()
    out = np.empty(row_bytes(t, n) * rows, np.uint8)
    im = None if imatrix is None else np.ascontiguousarray(imatrix, dtype=np.float32)
    got = ggml.ggml_quantize_chunk(t, x.ctypes.data, out.ctypes.data, 0, rows, n, None if im is None else im.ctypes.data)
    assert got == out.size, (got, out.size)
    return out.tobytes()


def dequant(rt, raw, n_per_row):
    raw = np.asarray(raw).reshape(-1).view(np.uint8)
    if rt == "Q8_0":
        blk = raw.reshape(-1, 34)
        d = blk[:, :2].copy().view(np.float16).astype(np.float32)
        q = blk[:, 2:].view(np.int8).astype(np.float32)
        return (q * d).reshape(-1, n_per_row)
    if rt == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(-1, n_per_row)
    if rt == "F16":
        return raw.view(np.float16).astype(np.float32).reshape(-1, n_per_row)
    if rt == "F32":
        return raw.view(np.float32).reshape(-1, n_per_row)
    raise ValueError(rt)


class Source:
    def __init__(self, src):
        self.readers, self.tensors = [], {}
        files = sorted(os.path.join(src, "Q8_0", f) for f in os.listdir(os.path.join(src, "Q8_0")) if f.endswith(".gguf"))
        files.append(os.path.join(src, "MTP", "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"))
        for f in files:
            r = GGUFReader(f)
            self.readers.append((f, r))
            for t in r.tensors:
                self.tensors[t.name] = (f, t)
        self.meta = {}
        f0, r0 = self.readers[0]
        for k, fld in r0.fields.items():
            if k.startswith("qwen4exp."):
                try:
                    self.meta[k] = fld.contents()
                except Exception:
                    pass

    def get(self, name):
        f, t = self.tensors[name]
        return t

    def f32(self, name):
        t = self.get(name)
        ne = [int(x) for x in t.shape]
        return dequant(t.tensor_type.name, t.data, ne[0])


class Imatrix:
    def __init__(self, path):
        r = GGUFReader(path)
        self.t = {t.name: np.array(t.data) for t in r.tensors}

    def vec(self, name, expert=None):
        s, c = self.t.get(name + ".in_sum2"), self.t.get(name + ".counts")
        if s is None:
            return None
        c = c.reshape(-1).astype(np.float64)
        if expert is None:
            return (s.reshape(-1) / max(c[0], 1.0)).astype(np.float32)
        s = s.reshape(NE, -1)
        if c[expert] <= 0:
            return None
        return (s[expert] / c[expert]).astype(np.float32)


class Writer:

    def __init__(self, path):
        self.path, self.kv, self.tensors = path, [], []

    def add_kv(self, key, value):
        self.kv.append((key, value))

    def add_tensor(self, name, ne, ttype, produce):
        self.tensors.append((name, list(ne), ttype, produce))

    @staticmethod
    def _s(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    def _value(self, v):
        if isinstance(v, bool):
            return struct.pack("<I", 7) + struct.pack("<?", v)
        if isinstance(v, int):
            return struct.pack("<I", 11) + struct.pack("<q", v)
        if isinstance(v, float):
            return struct.pack("<I", 6) + struct.pack("<f", v)
        if isinstance(v, str):
            return struct.pack("<I", 8) + self._s(v)
        if isinstance(v, (list, tuple)):
            if all(isinstance(x, int) for x in v):
                return struct.pack("<II", 9, 11) + struct.pack("<Q", len(v)) + b"".join(struct.pack("<q", x) for x in v)
            if all(isinstance(x, float) for x in v):
                return struct.pack("<II", 9, 6) + struct.pack("<Q", len(v)) + b"".join(struct.pack("<f", x) for x in v)
        raise TypeError(v)

    def write(self):
        head = b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.kv) + 1)
        head += self._s("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", 32)
        for k, v in self.kv:
            head += self._s(k) + self._value(v)
        off, infos = 0, b""
        for name, ne, t, _ in self.tensors:
            infos += self._s(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", x) for x in ne) + struct.pack("<IQ", t, off)
            off += (nbytes(t, ne) + 31) // 32 * 32
        start = (len(head) + len(infos) + 31) // 32 * 32
        total = start + off
        t0 = time.time()
        with open(self.path, "wb") as f:
            f.write(head + infos + b"\0" * (start - len(head) - len(infos)))
            done = 0
            for i, (name, ne, t, produce) in enumerate(self.tensors):
                n = nbytes(t, ne)
                written = 0
                for chunk in produce():
                    f.write(chunk)
                    written += len(chunk)
                assert written == n, (name, written, n)
                f.write(b"\0" * ((n + 31) // 32 * 32 - n))
                done += n
                if i % 25 == 0 or n > 1 << 30:
                    el = time.time() - t0
                    print(f"  {done / 1e9:7.2f} / {total / 1e9:.2f} GB  {el:6.0f} s  {name}", flush=True)
        print(f"wrote {self.path}: {total / 1e9:.2f} GB in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--hot", type=int, default=5600, help="experts in VRAM")
    ap.add_argument("--q2", type=float, default=0.2, help="share of all experts in Q2_K (the least used)")
    ap.add_argument("--route", action="append", default=[], help="routing counts [48][512] (.npy) and their weight, FILE:W")
    a = ap.parse_args()
    src, out, n_hot, q2_frac = a.src, a.out, a.hot, a.q2
    S = Source(src)
    I = Imatrix(os.path.join(src, "imatrix_unsloth.gguf"))
    pool = ThreadPoolExecutor(int(os.environ.get("THREADS", os.cpu_count())))

    calib = np.stack([I.t[f"blk.{l}.ffn_gate_exps.weight.counts"].reshape(-1) for l in range(NL)]).astype(np.float64).reshape(-1)
    routes = []
    for spec in a.route:
        f, wt = spec.rsplit(":", 1)
        routes.append((os.path.basename(f), np.load(f).astype(np.float64).reshape(-1), float(wt)))
    w_calib = 1 - sum(wt for _, _, wt in routes)
    if w_calib < 0: sys.exit("fn_quantize: route weights sum to more than 1")
    use = w_calib * calib / calib.sum()
    for _, cnt, wt in routes: use += wt * cnt / cnt.sum()
    order = np.argsort(-use, kind="stable")
    n_q2 = int(q2_frac * order.size)
    tier = np.ones(order.size, np.int32)
    tier[order[:n_hot]] = 0
    tier[order[order.size - n_q2:]] = 2
    hot_ids, q2_ids = order[:n_hot], order[order.size - n_q2:]
    tier = tier.reshape(NL, NE)
    hot = [[e for e in range(NE) if tier[l, e] == 0] for l in range(NL)]
    for name, cnt in [("calibration", calib)] + [(f, c) for f, c, _ in routes]:
        print(f"{name}: {n_hot} hot experts serve {100 * cnt[hot_ids].sum() / cnt.sum():.1f}% of expert work, "
              f"{n_q2} Q2_K experts {100 * cnt[q2_ids].sum() / cnt.sum():.2f}%")
    print(f"hot experts per layer {min(map(len, hot))}..{max(map(len, hot))}")

    W = Writer(out)
    W.add_kv("general.architecture", "chrysalisfn")
    W.add_kv("general.name", "Qwen3.8-Flash-Next (Chrysalis: Q6_K/Q8_0 dense, IQ4_XS VRAM / Q3_K + Q2_K RAM experts)")
    for k in ("ple.layer_multipliers", "ple.head_offsets", "ple.head_vocab_sizes"):
        W.add_kv("chrysalisfn." + k, [int(x) for x in S.meta["qwen4exp." + k]])
    ng = S.get("per_layer_token_embd.weight")
    ng_file = S.tensors["per_layer_token_embd.weight"][0]
    W.add_kv("chrysalisfn.ngram_file", os.path.relpath(ng_file, os.path.dirname(os.path.abspath(out))))
    W.add_kv("chrysalisfn.ngram_offset", int(ng.data_offset))
    W.add_kv("chrysalisfn.ngram_rows", int(ng.shape[1]))
    W.add_kv("chrysalisfn.n_hot", n_hot)
    if routes:
        W.add_kv("chrysalisfn.routes", ", ".join(f"{f}:{wt:g}" for f, _, wt in routes))

    def keep(name, out_name=None, dims=None):
        t = S.get(name)
        ne = dims or [int(x) for x in t.shape]
        tt = {"Q8_0": Q8_0, "F32": F32, "F16": F16}[t.tensor_type.name]
        raw = t.data

        def produce():
            yield np.asarray(raw).reshape(-1).view(np.uint8).tobytes()
        W.add_tensor(out_name or name, ne, tt, produce)

    def requant(name, tt, imat_name=None, out_name=None):
        t = S.get(name)
        ne = [int(x) for x in t.shape]
        im = I.vec(imat_name) if imat_name else None

        def produce():
            x = S.f32(name)
            step = max(1, (1 << 24) // ne[0])
            jobs = [pool.submit(quantize, tt, x[i:i + step], im) for i in range(0, x.shape[0], step)]
            for j in jobs:
                yield j.result()
        W.add_tensor(out_name or name, ne, tt, produce)

    def experts(l, prefix, mat, tt, ids, out_name):
        t = S.get(f"{prefix}ffn_{mat}_exps.weight")
        ne = [int(x) for x in t.shape]
        raw = np.asarray(t.data).reshape(NE, -1)

        def produce(raw=raw, ne=ne, rt=t.tensor_type.name):
            def one(e):
                x = dequant(rt, raw[e], ne[0])
                im = I.vec(f"blk.{l}.ffn_{mat}_exps.weight", e) if l < NL else None
                return quantize(tt, x, im)
            jobs = [pool.submit(one, e) for e in ids]
            for j in jobs:
                yield j.result()
        if ids:
            W.add_tensor(f"blk.{l}.{out_name}.weight", [ne[0], ne[1], len(ids)], tt, produce)

    keep("token_embd.weight")
    requant("output.weight", Q6_K, "output.weight" if I.vec("output.weight") is not None else None)
    keep("output_hc_norm.weight"); keep("output_hc_down.weight"); keep("output_hc_up.weight")

    for l in range(NL + 1):
        p = f"blk.{l}."
        for k in ("hc_attn_norm", "hc_attn_down", "hc_attn_up", "hc_attn_inject", "hc_ffn_norm", "hc_ffn_down", "hc_ffn_up", "hc_ffn_inject"):
            keep(p + k + ".weight")
        if l == NL or l % 4 == 3:
            requant(p + "attn_q.weight", Q6_K, p + "attn_q.weight")
            keep(p + "attn_k.weight"); keep(p + "attn_v.weight"); keep(p + "attn_output.weight")
            keep(p + "attn_q_norm.weight"); keep(p + "attn_k_norm.weight")
            requant(p + "indexer.q_proj.weight", Q8_0); requant(p + "indexer.k_proj.weight", Q8_0)
            keep(p + "indexer.q_norm.weight"); keep(p + "indexer.k_norm.weight")
        else:
            requant(p + "attn_qkv.weight", Q6_K, p + "attn_qkv.weight")
            requant(p + "attn_gate.weight", Q6_K, p + "attn_gate.weight")
            keep(p + "ssm_out.weight")
            for k in ("ssm_conv1d.weight", "ssm_a", "ssm_dt.bias", "ssm_norm.weight"):
                keep(p + k)

            def ab(p=p):
                x = np.concatenate([S.f32(p + "ssm_alpha.weight"), S.f32(p + "ssm_beta.weight")])
                yield x.astype(np.float16).tobytes()
            W.add_tensor(p + "ssm_ab.weight", [D, 96], F16, ab)
            if l == 1:
                for k in ("ple_key.weight", "ple_value.weight", "ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv1d.weight"):
                    keep(p + k)
        if l == NL:
            for k in ("nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.hc_head_norm.weight",
                      "nextn.hc_head_down.weight", "nextn.hc_head_up.weight"):
                keep(p + k)

        def router(p=p):
            yield S.f32(p + "ffn_gate_inp.weight").astype(np.float16).tobytes()
        W.add_tensor(p + "ffn_router.weight", [D, NE], F16, router)
        keep(p + "ffn_gate_inp_shexp.weight")
        requant(p + "ffn_gate_shexp.weight", Q6_K, p + "ffn_gate_shexp.weight")
        requant(p + "ffn_up_shexp.weight", Q6_K, p + "ffn_up_shexp.weight")
        keep(p + "ffn_down_shexp.weight")
        if l < NL:
            h = hot[l]
            c0 = [e for e in range(NE) if tier[l, e] == 1]
            c1 = [e for e in range(NE) if tier[l, e] == 2]
            slot = np.zeros(NE, np.int32)
            for i, e in enumerate(h):
                slot[e] = i
            for i, e in enumerate(c0 + c1):
                slot[e] = -(i + 1)
            W.add_tensor(p + "ffn_exp_slot", [NE], I32, lambda slot=slot: iter([slot.tobytes()]))
            experts(l, p, "gate", IQ4_XS, h, "ffn_gate_hot"); experts(l, p, "up", IQ4_XS, h, "ffn_up_hot")
            experts(l, p, "down", IQ4_NL, h, "ffn_down_hot")
            experts(l, p, "gate", Q3_K, c0, "ffn_gate_cold0"); experts(l, p, "up", Q3_K, c0, "ffn_up_cold0")
            experts(l, p, "gate", Q2_K, c1, "ffn_gate_cold1"); experts(l, p, "up", Q2_K, c1, "ffn_up_cold1")
            experts(l, p, "down", IQ4_NL, c0 + c1, "ffn_down_cold")
        else:
            experts(l, p, "gate", Q3_K, list(range(NE)), "ffn_gate_mtp"); experts(l, p, "up", Q3_K, list(range(NE)), "ffn_up_mtp")
            experts(l, p, "down", IQ4_NL, list(range(NE)), "ffn_down_mtp")
    W.write()
    json.dump({"tier": tier.tolist()}, open(out + ".tiers.json", "w"))


if __name__ == "__main__":
    main()
