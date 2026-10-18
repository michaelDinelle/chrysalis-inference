"""Decode speed on chat-formatted coding tasks with Qwen's sampling settings: writing new code (thinking first), and
rewriting a file the prompt holds, where prompt-lookup drafts can copy from it. --together N: N replies to the coding
task at once, their steps taken in one pass (generate_batch) against taken in turn.
usage: bench_decode.py [n_tokens] [drafts...] [--lookup N ...] [--together N]   (also reports tokens per step)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from kiln.engine import Engine, Sampling

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW = ("<|im_start|>user\nImplement an LRU cache in Python with O(1) get and put, using a dict and a "
       "doubly linked list. Include type hints, docstrings and a few unit tests with pytest.<|im_end|>\n"
       "<|im_start|>assistant\n<think>\n")
EDIT = ("<|im_start|>user\nRename the class `Cache` to `PromptCache` in this file and give me the whole file back, "
        "nothing else.\n\n```python\n" + open(os.path.join(ROOT, "kiln/cache.py")).read() + "```<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
args = sys.argv[1:]
together = 0
if "--together" in args:
    i = args.index("--together")
    together = int(args[i + 1])
    args = args[:i] + args[i + 2:]
lookups = [0]
if "--lookup" in args:
    i = args.index("--lookup")
    lookups = [int(a) for a in args[i + 1:]]
    args = args[:i]
n_tok = int(args[0]) if args else 512
drafts = [int(a) for a in args[1:]] or [0, 2, 3, 4]
tok = Tokenizer.from_file(os.path.join(ROOT, "data/models/tokenizer.json"))
eng = Engine(os.path.join(ROOT, "data/models/kiln-mix.gguf"), max_seqs=max(4, together + 1))
if together:
    ids = tok.encode(NEW, add_special_tokens=False).ids
    for k in drafts:
        sp = Sampling(temperature=1.0, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=k)
        for mode in ("in turn", "together"):
            ss = [eng.new_seq() for _ in range(together)]
            for i, s in enumerate(ss):
                eng.prefill(s, ids); eng.begin(s, 1234 + i)
            outs, steps = [[] for _ in ss], 0
            t0 = time.time()
            while min(map(len, outs)) < n_tok:
                got = eng.generate_batch(ss, [sp] * len(ss)) if mode == "together" else [eng.generate(s, sp, 16) for s in ss]
                for o, g in zip(outs, got):
                    o += g
                steps += 1
            dt = time.time() - t0
            total = sum(map(len, outs))
            print(f"{together} replies {mode:8s} drafts={k}: {total / dt:6.1f} tok/s in all, {total / dt / together:5.1f} each, "
                  f"{total / steps / together:.2f} tokens per reply per step", flush=True)
            for s in ss:
                eng.free_seq(s)
    sys.exit(0)
for task, prompt in (("new code", NEW), ("file edit", EDIT)):
    ids = tok.encode(prompt, add_special_tokens=False).ids
    for k in drafts:
        for lk in lookups:
            for temp in (1.0, 0.0):
                s = eng.new_seq(); eng.begin(s, 1234)
                eng.prefill(s, ids)
                if lk:
                    eng.set_lookup(s, ids)
                sp = Sampling(temperature=temp, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=k, lookup=lk)
                out, steps = [], 0
                t0 = time.time()
                while len(out) < n_tok:
                    out += eng.generate(s, sp, 16); steps += 1
                dt = time.time() - t0
                print(f"{task:9s} drafts={k} lookup={lk} temp={temp}: {len(out)/dt:6.1f} tok/s  {len(out)/steps:.2f} tokens/step", flush=True)
                eng.free_seq(s)
print("sample:", repr(tok.decode(out[:200])))
