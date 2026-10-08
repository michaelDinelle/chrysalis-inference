"""Snapshots: restore gives bit-identical continuations, and copy-on-write protects shared pages."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kiln.engine import Engine, Sampling
eng = Engine("data/models/kiln-mix.gguf")
random.seed(3)
p1 = [random.randrange(1000, 50000) for _ in range(300)]
p2 = [random.randrange(1000, 50000) for _ in range(130)]
greedy = Sampling(temperature=0.0, top_p=1.0, min_p=0.0, presence_penalty=0.0, top_k=1, n_draft=3, seed=0)
a = eng.new_seq(); eng.prefill(a, p1); snap = eng.snapshot(a)
eng.prefill(a, p2); la = eng.last_logits()
out_a = []
while len(out_a) < 60: out_a += eng.generate(a, greedy, 16)
b = eng.new_seq(); eng.restore(b, snap); eng.prefill(b, p2); lb = eng.last_logits()
out_b = []
while len(out_b) < 60: out_b += eng.generate(b, greedy, 16)
c = eng.new_seq(); eng.prefill(c, p1 + p2); lc = eng.last_logits()
fails = 0
ok = np.array_equal(la, lb); fails += not ok
print(f"restored logits identical            {'ok' if ok else 'FAIL'}  max diff {np.abs(la-lb).max():.3g}")
ok = out_a[:60] == out_b[:60]; fails += not ok
print(f"continuations identical (with MTP)   {'ok' if ok else 'FAIL'}")
ok = int(la.argmax()) == int(lc.argmax()) and np.abs(la - lc).max() < 0.5; fails += not ok
print(f"one-shot prefill agrees              {'ok' if ok else 'FAIL'}  max diff {np.abs(la-lc).max():.3g}")
eng.truncate(c, 0); eng.prefill(c, p1 + p2); lt = eng.last_logits()
ok = np.array_equal(lc, lt); fails += not ok
print(f"truncate to 0 = fresh sequence       {'ok' if ok else 'FAIL'}  max diff {np.abs(lc-lt).max():.3g}")
eng.free_snapshot(snap)
free0 = eng.free_tokens()
for s_ in (a, b, c): eng.free_seq(s_)
ok = eng.free_tokens() > free0; fails += not ok
print(f"pages returned after free            {'ok' if ok else 'FAIL'}")
print("all ok" if not fails else f"{fails} FAILURES"); sys.exit(fails != 0)
