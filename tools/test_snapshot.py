"""Snapshots: restore gives bit-identical continuations, a snapshot at a prompt's end samples without a prefill, and
copy-on-write protects shared pages.
  python tools/test_snapshot.py [--use qwen3.8-flash-next]   (default: the 27B)"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kiln.engine import Engine, Sampling
from kiln.models import MODELS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = MODELS[sys.argv[sys.argv.index("--use") + 1] if "--use" in sys.argv else "qwen3.8-27b"]
eng = Engine(os.path.join(ROOT, spec["path"]), umax=spec["chunk"], lib=os.path.join(ROOT, spec["lib"]))
if hasattr(eng.lib, "kiln_tune"):        # Flash-Next: its prefill split stops adapting once settled; here it is fixed now
    import ctypes as C
    eng.lib.kiln_tune.restype, eng.lib.kiln_tune.argtypes = C.c_double, [C.c_char_p, C.c_double]
    eng.lib.kiln_tune(b"adapt_left", 0)
random.seed(3)
p1 = [random.randrange(1000, 50000) for _ in range(300)]
p2 = [random.randrange(1000, 50000) for _ in range(130)]
greedy = Sampling(temperature=0.0, top_p=1.0, min_p=0.0, presence_penalty=0.0, top_k=1, n_draft=spec["draft"])
sampled = Sampling(temperature=1.0, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=spec["draft"])


def gen(sid, sp, n=60):
    out = []
    while len(out) < n:
        out += eng.generate(sid, sp, 16)
    return out[:n]


a = eng.new_seq(); eng.prefill(a, p1); snap = eng.snapshot(a)
eng.prefill(a, p2); la = eng.last_logits()
out_a = gen(a, greedy)
b = eng.new_seq(); eng.restore(b, snap); eng.prefill(b, p2); lb = eng.last_logits()
out_b = gen(b, greedy)
c = eng.new_seq(); eng.prefill(c, p1 + p2); lc = eng.last_logits()
fails = 0
ok = np.array_equal(la, lb); fails += not ok
print(f"restored logits identical            {'ok' if ok else 'FAIL'}  max diff {np.abs(la-lb).max():.3g}")
ok = out_a == out_b; fails += not ok
print(f"continuations identical (with MTP)   {'ok' if ok else 'FAIL'}")
# the same tokens in other passes round differently (more on Flash-Next, whose CPU and GPU share the cold experts by pass)
tol = 1.0 if hasattr(eng.lib, "kiln_tune") else 0.5
ok = int(la.argmax()) == int(lc.argmax()) and np.abs(la - lc).max() < tol; fails += not ok
print(f"one-shot prefill agrees              {'ok' if ok else 'FAIL'}  max diff {np.abs(la-lc).max():.3g}")
eng.truncate(c, 0); eng.prefill(c, p1 + p2); lt = eng.last_logits()
ok = np.array_equal(lc, lt); fails += not ok
print(f"truncate to 0 = fresh sequence       {'ok' if ok else 'FAIL'}  max diff {np.abs(lc-lt).max():.3g}")
# a snapshot at the end of a prompt keeps the next token's distribution: a restore there draws the same reply, greedy
# or sampled with the same seed, with nothing prefilled
eng.truncate(c, 0); eng.prefill(c, p1 + p2); end = eng.snapshot(c)
eng.begin(c, 5); out_c = gen(c, sampled)
d = eng.new_seq(); eng.restore(d, end); eng.begin(d, 5); out_d = gen(d, sampled)
eng.truncate(d, 0); eng.restore(d, end); out_g = gen(d, greedy)
eng.truncate(c, 0); eng.prefill(c, p1 + p2); out_h = gen(c, greedy)
ok = out_c == out_d and out_g == out_h; fails += not ok
print(f"restore at a prompt's end = prefill  {'ok' if ok else 'FAIL'}  sampled {out_c == out_d}, greedy {out_g == out_h}")
eng.free_snapshot(snap); eng.free_snapshot(end)
free0 = eng.free_tokens()
for s_ in (a, b, c, d): eng.free_seq(s_)
ok = eng.free_tokens() > free0; fails += not ok
print(f"pages returned after free            {'ok' if ok else 'FAIL'}")
print("all ok" if not fails else f"{fails} FAILURES"); sys.exit(fails != 0)
