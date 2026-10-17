"""Speed against context depth, for either model (same content and method for both): real source code fills the
context in segments up to ~260K tokens; prefill speed is timed per segment, and at each depth (from a snapshot, so the
context is not disturbed) decode speed without drafts and with the model's MTP drafts (Qwen sampling) and the time to
first token of a 1,024-token append (an agent turn: a tool's output).
  python tools/bench_depth.py qwen3.8-27b|qwen3.8-flash-next [max_depth=260000]"""
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kiln.chat import Tokens  # noqa: E402
from kiln.engine import Engine, Sampling  # noqa: E402
from kiln.models import MODELS  # noqa: E402

name = sys.argv[1]
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 260000
spec = MODELS[name]
tok = Tokens(os.path.join(ROOT, "data/models/tokenizer.json"))
enc = lambda s: tok.full.encode(s, add_special_tokens=False).ids
files = sorted(glob.glob(os.path.join(ROOT, "data/llama.cpp/src/**/*.cpp"), recursive=True) +
               glob.glob(os.path.join(ROOT, "data/llama.cpp/ggml/src/**/*.c*"), recursive=True) +
               glob.glob(os.path.join(ROOT, "data/llama.cpp/tools/**/*.cpp"), recursive=True) +
               glob.glob(os.path.join(ROOT, "data/llama.cpp/common/*.cpp")))
ctx = enc("<|im_start|>user\nHere is a large part of a C/C++ code base:\n\n")
for f in files:                                          # distinct files, so the n-gram rows are not all cached
    if len(ctx) >= max_depth + 2048:
        break
    ctx += enc(f"// ===== {os.path.relpath(f, ROOT)} =====\n" + open(f, errors="replace").read() + "\n")
assert len(ctx) >= max_depth + 2048, f"not enough code: {len(ctx)} tokens"
filler, ctx = ctx[max_depth:max_depth + 1024], ctx[:max_depth]   # the append test's tokens come after the context
probe = enc("\n\nIn detail, explain how the code above is organized and what its main data structures are.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
eng = Engine(os.path.join(ROOT, spec["path"]), max_seqs=2, umax=spec["chunk"], lib=os.path.join(ROOT, spec["lib"]))
print(f"{name}: KV capacity {eng.capacity} tokens", flush=True)
depths = [d for d in (8192, 32768, 65536, 98304, 131072, 163840, 196608, 229376) if d < max_depth] + [max_depth]
s = eng.new_seq()
eng.begin(s, 7)
done, total_pf = 0, 0.0
print(f"{'depth':>7} {'prefill tok/s':>13} {'decode d=0':>10} {'decode d=' + str(spec['draft']):>10} {'append 1K TTFT':>14}", flush=True)
for d in depths:
    t0 = time.time()
    eng.prefill(s, ctx[done:d])
    dt = time.time() - t0
    total_pf += dt
    pf = (d - done) / dt
    done = d
    res = []
    for k in (0, spec["draft"]):                      # decode after a question, from a snapshot of the context
        snap = eng.snapshot(s)
        eng.prefill(s, probe)
        eng.begin(s, 11)
        sp = Sampling(temperature=0.6, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=k)
        out, t0 = [], time.time()
        while len(out) < 192:
            out += eng.generate(s, sp, 16)
        res.append(len(out) / (time.time() - t0))
        eng.restore(s, snap); eng.free_snapshot(snap)
    snap = eng.snapshot(s)                              # an agent turn: 1,024 new tokens, then the first token
    sp = Sampling(temperature=0.0, top_p=1.0, min_p=0.0, presence_penalty=0.0, top_k=1, n_draft=0)
    t0 = time.time()
    eng.prefill(s, filler)
    eng.generate(s, sp, 1)
    app = time.time() - t0
    eng.restore(s, snap); eng.free_snapshot(snap)
    print(f"{d:>7} {pf:>13.0f} {res[0]:>10.1f} {res[1]:>10.1f} {app:>13.2f}s", flush=True)
print(f"filling {max_depth} tokens from empty: {total_pf:.1f} s of prefill ({max_depth / total_pf:.0f} tok/s)", flush=True)
