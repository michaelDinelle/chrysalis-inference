import glob, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import ctypes as C
from tokenizers import Tokenizer
from chrysalis.engine import Engine, Sampling

DATA = os.path.join(ROOT, "data")
MODEL = os.path.join(DATA, "models/qwen3.8-27b/qwen3.8-27b.gguf")
tok = Tokenizer.from_file(os.path.join(DATA, "models/tokenizer.json"))
what = [a for a in sys.argv[1:] if not a.startswith("--")] or ["decode", "prefill", "ttft", "depth"]
depths = [32768, 131072, 262144]
for a in sys.argv:
    if a.startswith("--depths="):
        depths = [int(x) for x in a.split("=")[1].split(",")]

src = sorted(glob.glob(os.path.join(ROOT, "engine", "*.cpp")) + glob.glob(os.path.join(ROOT, "chrysalis", "*.py")))
CODE = tok.encode("".join(open(p).read() for p in src), add_special_tokens=False).ids
while len(CODE) < 40000:
    CODE += CODE


def chat(user):
    return tok.encode(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n", add_special_tokens=False).ids


e = Engine(MODEL, lib=os.environ.get("CHRYSALIS_LIB", os.path.join(ROOT, "build", "libchrysalis.dylib")))
L = e.lib


def decode(s, sp, n=256):
    e.begin(s, 1)
    got, t0 = [], None
    while len(got) < n:
        out = e.generate(s, sp, 16)
        if t0 is None:
            t0, base = time.time(), len(got) + len(out)
        got += out
        if out and out[-1] in (248046, 248044):
            break
    return (len(got) - base) / (time.time() - t0), len(got)


PROMPT = chat("Write a Python module that parses a CSV file of transactions, groups them by month and category, and prints a "
              "summary table with totals, averages and the largest transaction of each group. Include type hints and tests.")
if "decode" in what:
    for name, sp in [("greedy, no drafts", Sampling(temperature=0, n_draft=0)),
                     ("greedy, MTP 3 drafts", Sampling(temperature=0, n_draft=3)),
                     ("greedy, MTP 2 drafts", Sampling(temperature=0, n_draft=2)),
                     ("Qwen sampling, no drafts", Sampling(temperature=0.6, top_p=0.95, top_k=20, n_draft=0)),
                     ("Qwen sampling, MTP 3 drafts", Sampling(temperature=0.6, top_p=0.95, top_k=20, n_draft=3))]:
        s = e.new_seq()
        e.prefill(s, PROMPT)
        rate, n = decode(s, sp)
        print(f"decode {name:30s} {rate:6.1f} tok/s ({n} tokens)", flush=True)
        e.free_seq(s)
    body = "".join(open(p).read() for p in sorted(glob.glob(os.path.join(ROOT, "engine", "lookup.h"))))
    p = chat("Repeat this file exactly, with no changes and no commentary:\n\n" + body) + tok.encode("</think>\n\n", add_special_tokens=False).ids
    for name, sp in [("rewrite, MTP 3 drafts", Sampling(temperature=0, n_draft=3)),
                     ("rewrite, prompt lookup + MTP", Sampling(temperature=0, n_draft=3, lookup=7))]:
        s = e.new_seq(); e.prefill(s, p); e.set_lookup(s, p)
        rate, n = decode(s, sp, 400)
        print(f"decode {name:30s} {rate:6.1f} tok/s ({n} tokens)", flush=True)
        e.free_seq(s)

if "prefill" in what:
    for n in (512, 4096, 32768):
        s = e.new_seq()
        t = time.time(); e.prefill(s, CODE[:n]); dt = time.time() - t
        print(f"prefill {n:6d} tokens {dt:7.2f} s  {n / dt:7.1f} tok/s", flush=True)
        e.free_seq(s)

if "ttft" in what:
    sp = Sampling(temperature=0, n_draft=3)
    s = e.new_seq()
    t = time.time(); e.prefill(s, PROMPT); e.begin(s, 1); e.generate(s, sp, 1)
    print(f"TTFT short prompt ({len(PROMPT)} tokens, cold) {time.time() - t:.3f} s", flush=True)
    e.free_seq(s)
    conv = chat("Read these files.")[:-3] + CODE[:4000]
    s = e.new_seq(); e.prefill(s, conv); snap = e.snapshot(s); e.free_seq(s)
    for n in (16, 64, 200, 1000):
        turn = tok.encode("<|im_end|>\n<|im_start|>user\n<tool_response>\n", add_special_tokens=False).ids + CODE[5000:5000 + n] + \
            tok.encode("\n</tool_response><|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).ids
        s = e.new_seq()
        t = time.time(); e.restore(s, snap); e.prefill(s, turn); e.begin(s, 1); e.generate(s, sp, 1)
        print(f"TTFT agent turn: {len(turn):5d} new tokens on a 4K cached conversation  {time.time() - t:.3f} s", flush=True)
        e.free_seq(s)
    s = e.new_seq()
    t = time.time(); e.restore(s, snap); e.begin(s, 1); e.generate(s, sp, 1)
    print(f"TTFT resent prompt (restore only)  {time.time() - t:.3f} s", flush=True)
    e.free_seq(s); e.free_snapshot(snap)

if "depth" in what and hasattr(L, "chrysalis_bench_fill"):
    for n in depths:
        s = e.new_seq()
        e.prefill(s, PROMPT)
        if L.chrysalis_bench_fill(s, n) < 0:
            print(f"depth {n}: does not fit"); e.free_seq(s); continue
        for name, sp in [("no drafts", Sampling(temperature=0, n_draft=0)), ("MTP 3 drafts", Sampling(temperature=0, n_draft=3))]:
            e.prefill(s, PROMPT[-4:])
            rate, k = decode(s, sp, 64)
            print(f"decode at {e.seq_len(s):6d} tokens, {name:14s} {rate:6.1f} tok/s", flush=True)
        e.free_seq(s)
print(f"free KV tokens {e.free_tokens()} of capacity {e.capacity}")
