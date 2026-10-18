"""Compare kiln's next-token distributions with llama.cpp's on the same GGUF (KL divergence, top-1), on the decode,
prefill and mixed paths. The weights are the same, so what differs is kiln's arithmetic (int8 activations, the 5-bit
V cache): a path fails past its MAX_KL or under MIN_TOP1. The reference, build/cmp/ref.bin, is written by build/ref_logits
(make build/ref_logits) when it is missing or with --fresh.
  data/venv/bin/python tools/compare_ref.py [--fresh]"""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from kiln.engine import Engine

MODEL = "data/models/kiln-mix.gguf"
# mean KL measured 2026-10-18 (deterministic): decode 0.0022, prefill 0.0035, mixed 0.0022; a decode step that forgets
# twice as fast lands at 0.024
MAX_KL, MIN_TOP1 = {"decode": 0.004, "prefill": 0.008, "mixed": 0.004}, 0.95
TEXT = '''<|im_start|>user
Write a Python function that merges overlapping intervals.<|im_end|>
<|im_start|>assistant
<think>
The standard approach: sort by start, then sweep, extending the last interval when they overlap.
</think>

def merge(intervals):
    intervals.sort(key=lambda iv: iv[0])
    out = []
    for start, end in intervals:
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out

The function runs in O(n log n) time because of the sort; the sweep itself is linear. Edge cases such as
empty input, a single interval, or intervals that only touch at an endpoint are handled naturally.'''

tok = Tokenizer.from_file("data/models/tokenizer.json")
ids = tok.encode(TEXT, add_special_tokens=False).ids
n = len(ids)
os.makedirs("build/cmp", exist_ok=True)
np.array(ids, dtype=np.int32).tofile("build/cmp/tokens.bin")
if not os.path.exists("build/cmp/ref.bin") or "--fresh" in sys.argv:
    subprocess.run(["build/ref_logits", MODEL, "build/cmp/tokens.bin", "build/cmp/ref.bin"], check=True,
                   stderr=subprocess.DEVNULL)
ref = np.fromfile("build/cmp/ref.bin", dtype=np.float32).reshape(n, -1)

def logsoftmax(x):
    x = x.astype(np.float64); x -= x.max(); return x - np.log(np.exp(x).sum())

def stats(name, pairs, max_kl):
    kls, top1, dmax = [], 0, 0
    for i, ours in pairs:
        lp, lq = logsoftmax(ref[i]), logsoftmax(ours)
        kls.append(float((np.exp(lp) * (lp - lq)).sum()))
        top1 += int(ref[i].argmax() == ours.argmax())
        dmax = max(dmax, float(np.abs((ref[i] - ref[i].max()) - (ours - ours.max())).max()))
    ok = np.mean(kls) <= max_kl and top1 / len(pairs) >= MIN_TOP1
    print(f"{name:28s} positions {len(pairs):4d}  mean KL {np.mean(kls):.5f}  max KL {np.max(kls):.4f}  "
          f"top-1 agree {100*top1/len(pairs):.1f}%  {'ok' if ok else 'FAIL'}")
    return ok

eng = Engine(MODEL)
s = eng.new_seq()
pairs = []
for i, t in enumerate(ids):                 # decode path: one token at a time
    eng.prefill(s, [t]); pairs.append((i, eng.last_logits()))
fails = not stats("decode (T=1 per token)", pairs, MAX_KL["decode"])
eng.free_seq(s)

pairs = []
for cut in (16, 64, n // 2, n):             # prefill path: chunked GEMMs
    s = eng.new_seq(); eng.prefill(s, ids[:cut]); pairs.append((cut - 1, eng.last_logits())); eng.free_seq(s)
fails += not stats("prefill (one chunk)", pairs, MAX_KL["prefill"])

s = eng.new_seq(); eng.prefill(s, ids[:40]); pairs = []
for i in range(40, n):                      # mixed: prefill then decode on top
    eng.prefill(s, [ids[i]]); pairs.append((i, eng.last_logits()))
fails += not stats("prefill 40 + decode", pairs, MAX_KL["mixed"])
print("all ok" if not fails else f"{fails} FAILURES")
sys.exit(fails != 0)
