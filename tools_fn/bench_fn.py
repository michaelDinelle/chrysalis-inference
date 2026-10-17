"""Flash-Next speed on this machine: load time, decode tok/s (drafts 0/1/2, Qwen sampling and greedy), and the time to
first token of appends of various sizes on top of an existing conversation (what an agent turn costs).
  python tools_fn/bench_fn.py [n_tokens] [--decode-only] [--model FILE] [--use qwen3.8-27b]   (--use: the same for another model)"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenizers import Tokenizer  # noqa: E402
from kiln.engine import Engine, Sampling  # noqa: E402
from kiln.models import MODELS  # noqa: E402

PROMPT = ("<|im_start|>user\nImplement an LRU cache in Python with O(1) get and put, using a dict and a "
          "doubly linked list. Include type hints, docstrings and a few unit tests with pytest.<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n")
tok = Tokenizer.from_file(os.path.join(ROOT, "data/models/tokenizer.json"))
ids = tok.encode(PROMPT, add_special_tokens=False).ids
n_tok = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 512
# filler for the appends: real code (kiln's own sources), tokenized once
code = "".join(open(os.path.join(ROOT, f)).read() for f in ("engine_fn/model.cpp", "engine_fn/load.cpp", "kiln/server.py"))
filler = tok.encode(code, add_special_tokens=False).ids
while len(filler) < 16384 + 4096 + 64:
    filler += filler

t0 = time.time()
spec = MODELS[sys.argv[sys.argv.index("--use") + 1] if "--use" in sys.argv else "qwen3.8-flash-next"]
model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else os.path.join(ROOT, spec["path"])
eng = Engine(model, max_seqs=3, umax=spec["chunk"], lib=os.path.join(ROOT, spec["lib"]))
print(f"load: {time.time() - t0:.1f} s, KV capacity {eng.capacity} tokens", flush=True)

for k in sorted({0, 1, 2, spec["draft"]}):
    for temp in (1.0, 0.0):
        s = eng.new_seq(); eng.begin(s, 1234)
        eng.prefill(s, ids)
        sp = Sampling(temperature=temp, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=k)
        out, steps = [], 0
        t0 = time.time()
        while len(out) < n_tok:
            out += eng.generate(s, sp, 16); steps += 1
        dt = time.time() - t0
        print(f"decode drafts={k} temp={temp}: {len(out) / dt:6.1f} tok/s  {len(out) / steps:.2f} tokens/step", flush=True)
        eng.free_seq(s)
print("sample:", repr(tok.decode(out[:160])), flush=True)
# decode inside a coding conversation: ~4K tokens of real code in context, then a request to change it
ctx = tok.encode("<|im_start|>user\nHere is a file from my project:\n```\n", add_special_tokens=False).ids + filler[:4000] + \
      tok.encode("\n```\nRefactor the longest function in it into smaller helpers and show the new code.<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
for k in sorted({0, spec["draft"]}):
    s = eng.new_seq(); eng.begin(s, 99)
    eng.prefill(s, ctx)
    sp = Sampling(temperature=0.6, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=k)
    out, steps = [], 0
    t0 = time.time()
    while len(out) < n_tok:
        out += eng.generate(s, sp, 16); steps += 1
    dt = time.time() - t0
    print(f"decode in a 4K code context, drafts={k}: {len(out) / dt:6.1f} tok/s  {len(out) / steps:.2f} tokens/step", flush=True)
    eng.free_seq(s)

if "--decode-only" in sys.argv:
    sys.exit(0)
# appends on top of a 16K-token conversation: prefill time of the append + the first token
s = eng.new_seq(); eng.begin(s, 1)
t0 = time.time()
eng.prefill(s, filler[:16384])
print(f"prefill 16K: {time.time() - t0:.2f} s ({16384 / (time.time() - t0):.0f} tok/s)", flush=True)
pos = 16384
sp = Sampling(temperature=0.0, top_p=1.0, min_p=0.0, presence_penalty=0.0, top_k=1, n_draft=0)
for n in (16, 64, 256, 1024, 4096):
    snap = eng.snapshot(s)
    chunk = filler[pos:pos + n]
    t0 = time.time()
    eng.prefill(s, chunk)
    eng.generate(s, sp, 1)
    print(f"append {n:5d} tokens: {1000 * (time.time() - t0):7.1f} ms to first token", flush=True)
    eng.restore(s, snap); eng.free_snapshot(snap)
