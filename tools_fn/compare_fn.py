"""Compare Chrysalis's Flash-Next next-token distributions with llama.cpp's on the Q8_0 source (KL divergence, top-1).

The two run different weights (chrysalis-fn.gguf: Q6_K/Q8_0 dense, IQ4_XS experts in VRAM, Q3_K / Q2_K in RAM;
llama.cpp: Q8_0), so the numbers are Chrysalis's quantization error plus any implementation error: decode, prefill and
mixed paths must agree with each other, and the KL should match the prediction (~0.017): a path fails past its MAX_KL or
under MIN_TOP1, a prefill path also past SELF_KL from the decode path at the same positions. The cold experts' CPU / GPU
split is fixed here (all of a prefill's on the CPU, gfrac 0.3 at decode, the expert pool as loaded): the load-time
calibration and the pool would otherwise move it, and with it the rounding, so every run gives the same numbers. Then
the pool moves (engine_fn/experts.cpp) and the paths run again under the same limits: once after a prose reply has
pushed the text's experts out of VRAM (the file's VRAM-tier ones then computed by the CPU from the model file), once
after the text itself has pulled them back in (RAM-tier experts in their own formats in pool slots, VRAM-tier ones read
back from the file into other slots).
  python tools_fn/compare_fn.py [--fresh] [--ref-only] [--text FILE --tag NAME] [--model FILE]
"""
import os
import subprocess
import sys

import ctypes as C

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenizers import Tokenizer  # noqa: E402
from chrysalis.engine import Engine  # noqa: E402
from chrysalis.models import MODELS  # noqa: E402

MODEL = os.path.join(ROOT, "data/models/flash-next/chrysalis-fn.gguf")
REF = os.path.join(ROOT, "data/models/flash-next/src/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf")
# mean KL with this split (2026-10-18, the same every run with the pool as loaded): decode 0.0155, prefill 0.0119,
# mixed 0.0108. Quantization sets most of it, so a broken kernel moves it less than on the 27B: a decode step that
# forgets twice as fast gives 0.0284 and 0.0218. Once the pool has moved, each run's pool differs (it follows a sampled
# reply) and computes the experts on other devices, so the numbers do too, each pool's the same every run (bit for bit):
# decode 0.0142-0.0165, prefill 0.0108-0.0171. Prefill is judged over 52 prompt lengths (over 4 it went from 0.002 to
# 0.015 on one position's rounding) and against Chrysalis's own decode path at the same positions, which no pool moves
# far: 0.0046-0.0077 (SELF_KL).
MAX_KL, MIN_TOP1 = {"decode": 0.021, "prefill": 0.021, "mixed": 0.016}, 0.92
SELF_KL = 0.012
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

def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


MODEL = arg("--model", MODEL)
if arg("--text"):                            # another evaluation text, with its own reference directory
    TEXT = open(arg("--text")).read()
tok = Tokenizer.from_file(os.path.join(ROOT, "data/models/tokenizer.json"))
ids = tok.encode(TEXT, add_special_tokens=False).ids
n = len(ids)
CUTS = range(12, n + 1, 3)  # prompt lengths of the prefill path (each a pass of over 8 tokens)
out_dir = os.path.join(ROOT, "build_fn/cmp" + ("_" + arg("--tag") if arg("--tag") else ""))
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


def stats(name, pairs, max_kl):
    kls, top1 = [], 0
    for i, ours in pairs:
        lp, lq = logsoftmax(ref[i]), logsoftmax(ours)
        kls.append(float((np.exp(lp) * (lp - lq)).sum()))
        top1 += int(ref[i].argmax() == ours.argmax())
    print(f"{name:28s} positions {len(pairs):4d}  mean KL {np.mean(kls):.5f}  max KL {np.max(kls):.4f}  "
          f"top-1 agree {100 * top1 / len(pairs):.1f}%  {'ok' if np.mean(kls) <= max_kl and top1 / len(pairs) >= MIN_TOP1 else 'FAIL'}", flush=True)
    np.save(os.path.join(out_dir, "kl_" + name.split()[0] + ".npy"), np.array([[i, k] for (i, _), k in zip(pairs, kls)]))
    if len(pairs) > 1024:                    # by position: long-context paths (sparse selection past 2048 cells) show here
        pos = np.array([i for i, _ in pairs]); kk = np.array(kls)
        print("   by position: " + "  ".join(f"{a}-{b}: {kk[(pos >= a) & (pos < b)].mean():.4f}"
                                            for a, b in ((0, 512), (512, 1024), (1024, 2048), (2048, 2560), (2560, 3072), (3072, 1 << 20))
                                            if ((pos >= a) & (pos < b)).any()), flush=True)
    return np.mean(kls) <= max_kl and top1 / len(pairs) >= MIN_TOP1


eng = Engine(MODEL, max_seqs=2, umax=MODELS["qwen3.8-flash-next"]["chunk"], lib=os.path.join(ROOT, "build/libchrysalis_fn.so"))
eng.lib.chrysalis_tune.restype, eng.lib.chrysalis_tune.argtypes = C.c_double, [C.c_char_p, C.c_double]
eng.lib.chrysalis_tune(b"split", 1); eng.lib.chrysalis_tune(b"gfrac", 0.3); eng.lib.chrysalis_tune(b"experts", 0)


def against(name, pairs, decoded):
    """Mean KL of a prefill path from Chrysalis's decode path at the same positions."""
    kls = []
    for i, ours in pairs:
        lp, lq = logsoftmax(decoded[i]), logsoftmax(ours)
        kls.append(float((np.exp(lp) * (lp - lq)).sum()))
    ok = np.mean(kls) <= SELF_KL
    print(f"{name:28s} positions {len(pairs):4d}  mean KL {np.mean(kls):.5f} from the decode path  {'ok' if ok else 'FAIL'}", flush=True)
    return ok


def paths(tag):
    s = eng.new_seq()
    pairs = []
    for i, t in enumerate(ids):             # decode path: one token at a time
        eng.prefill(s, [t])
        pairs.append((i, eng.last_logits()))
    bad = not stats(f"decode (T=1 per token){tag}", pairs, MAX_KL["decode"])
    decoded = dict(pairs)
    eng.free_seq(s)

    pairs = []
    for cut in CUTS:                        # prefill path
        s = eng.new_seq()
        eng.prefill(s, ids[:cut])
        pairs.append((cut - 1, eng.last_logits()))
        eng.free_seq(s)
    bad += not stats(f"prefill (one chunk){tag}", pairs, MAX_KL["prefill"])
    bad += not against(f"prefill = decode{tag}", pairs, decoded)
    if tag:                                 # the cold experts staged on the GPU instead (the file's through pinned memory)
        eng.lib.chrysalis_tune(b"split", 0)
        pairs = []
        for cut in CUTS:
            s = eng.new_seq()
            eng.prefill(s, ids[:cut])
            pairs.append((cut - 1, eng.last_logits()))
            eng.free_seq(s)
        eng.lib.chrysalis_tune(b"split", 1)
        bad += not stats(f"prefill, cold on the GPU{tag}", pairs, MAX_KL["prefill"])
        bad += not against(f"prefill, cold on the GPU = decode{tag}", pairs, decoded)

    s = eng.new_seq()
    eng.prefill(s, ids[:40])
    pairs = []
    for i in range(40, n):                  # mixed: prefill then decode on top
        eng.prefill(s, [ids[i]])
        pairs.append((i, eng.last_logits()))
    bad += not stats(f"prefill 40 + decode{tag}", pairs, MAX_KL["mixed"])
    eng.free_seq(s)
    return bad


def placed():
    x = eng.expert_stats()
    return x["placed"], x["placed_from_file"]


def move_pool(prompt, steps):
    """The pool follows a reply to prompt (token ids) for steps decode steps, then holds still."""
    from chrysalis.engine import Sampling
    eng.lib.chrysalis_tune(b"experts", 1)
    s = eng.new_seq()
    eng.prefill(s, prompt)
    eng.begin(s, 7)
    sp = Sampling(temperature=1.0, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=2)
    for _ in range(steps):
        eng.generate(s, sp, 16)
    eng.free_seq(s)
    eng.lib.chrysalis_tune(b"experts", 0)


fails = paths("")
prose = tok.encode("<|im_start|>user\nWrite a long story about a lighthouse keeper and a storm.<|im_end|>\n<|im_start|>assistant\n",
                   add_special_tokens=False).ids
a = placed()
move_pool(prose, 400)
b = placed()
ok = b[0] - a[0] >= 1000 and b[1] - a[1] > 0
print(f"pool after a prose reply: {b[0] - a[0]} experts placed, {b[1] - a[1]} read from the model file  {'ok' if ok else 'FAIL'}", flush=True)
fails += not ok
fails += paths(", pool moved by prose")
move_pool(ids, 400)
c = placed()
ok = c[0] - b[0] >= 300 and c[1] - b[1] > 0
print(f"pool after the text: {c[0] - b[0]} experts placed, {c[1] - b[1]} read from the model file  {'ok' if ok else 'FAIL'}", flush=True)
fails += not ok
fails += paths(", pool moved by the text")
print("all ok" if not fails else f"{fails} FAILURES")
sys.exit(fails != 0)
