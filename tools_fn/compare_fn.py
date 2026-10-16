"""Compare kiln's Flash-Next next-token distributions with llama.cpp's on the Q8_0 source (KL divergence, top-1).

The two run different weights (kiln: Q6_K/Q8_0 dense, Q5_0 hot / IQ4_XS+IQ4_NL cold experts; llama.cpp: Q8_0), so the
numbers are kiln's quantization error plus any implementation error: decode, prefill and mixed paths must agree
with each other, and the KL should match the predicted ~0.01-0.02.
  python tools_fn/compare_fn.py [--fresh] [--ref-only]
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenizers import Tokenizer  # noqa: E402
from kiln.engine import Engine  # noqa: E402

MODEL = os.path.join(ROOT, "data/models/flash-next/kiln-fn.gguf")
REF = os.path.join(ROOT, "data/models/flash-next/src/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf")
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

tok = Tokenizer.from_file(os.path.join(ROOT, "data/models/tokenizer.json"))
ids = tok.encode(TEXT, add_special_tokens=False).ids
n = len(ids)
out_dir = os.path.join(ROOT, "build_fn/cmp")
os.makedirs(out_dir, exist_ok=True)
np.array(ids, dtype=np.int32).tofile(os.path.join(out_dir, "tokens.bin"))
ref_path = os.path.join(out_dir, "ref.bin")
if not os.path.exists(ref_path) or "--fresh" in sys.argv:
    print(f"reference: llama.cpp on the Q8_0 source, {n} tokens (CPU, weights streamed from disk)...", flush=True)
    subprocess.run([os.path.join(ROOT, "build_fn/ref_logits_fn"), REF, os.path.join(out_dir, "tokens.bin"), ref_path], check=True)
ref = np.fromfile(ref_path, dtype=np.float32).reshape(n, -1)
if "--ref-only" in sys.argv:
    sys.exit(0)


def logsoftmax(x):
    x = x.astype(np.float64)
    x -= x.max()
    return x - np.log(np.exp(x).sum())


def stats(name, pairs):
    kls, top1 = [], 0
    for i, ours in pairs:
        lp, lq = logsoftmax(ref[i]), logsoftmax(ours)
        kls.append(float((np.exp(lp) * (lp - lq)).sum()))
        top1 += int(ref[i].argmax() == ours.argmax())
    print(f"{name:28s} positions {len(pairs):4d}  mean KL {np.mean(kls):.5f}  max KL {np.max(kls):.4f}  "
          f"top-1 agree {100 * top1 / len(pairs):.1f}%", flush=True)
    return np.mean(kls)


eng = Engine(MODEL, max_seqs=2, umax=512, lib=os.path.join(ROOT, "build/libkiln_fn.so"))
s = eng.new_seq()
pairs = []
for i, t in enumerate(ids):                 # decode path: one token at a time
    eng.prefill(s, [t])
    pairs.append((i, eng.last_logits()))
stats("decode (T=1 per token)", pairs)
eng.free_seq(s)

pairs = []
for cut in (16, 64, n // 2, n):             # prefill path
    s = eng.new_seq()
    eng.prefill(s, ids[:cut])
    pairs.append((cut - 1, eng.last_logits()))
    eng.free_seq(s)
stats("prefill (one chunk)", pairs)

s = eng.new_seq()
eng.prefill(s, ids[:40])
pairs = []
for i in range(40, n):                      # mixed: prefill then decode on top
    eng.prefill(s, [ids[i]])
    pairs.append((i, eng.last_logits()))
stats("prefill 40 + decode", pairs)
