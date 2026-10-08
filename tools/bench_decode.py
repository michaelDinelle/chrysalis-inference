"""Decode speed on a realistic chat-formatted coding task with Qwen's sampling settings.
usage: bench_decode.py [n_tokens] [drafts...]   (also reports acceptance and prefill speed)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from kiln.engine import Engine, Sampling

PROMPT = ("<|im_start|>user\nImplement an LRU cache in Python with O(1) get and put, using a dict and a "
          "doubly linked list. Include type hints, docstrings and a few unit tests with pytest.<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n")
tok = Tokenizer.from_file("data/models/tokenizer.json")
ids = tok.encode(PROMPT, add_special_tokens=False).ids
n_tok = int(sys.argv[1]) if len(sys.argv) > 1 else 512
drafts = [int(a) for a in sys.argv[2:]] or [0, 2, 3, 4]
eng = Engine("data/models/kiln-mix.gguf")
for k in drafts:
    for temp in (1.0, 0.0):
        s = eng.new_seq(); eng.lib.kiln_seq_seed(s, 1234)
        eng.prefill(s, ids)
        sp = Sampling(temperature=temp, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=k, seed=0)
        out, steps = [], 0
        t0 = time.time()
        while len(out) < n_tok:
            out += eng.generate(s, sp, 16); steps += 1
        dt = time.time() - t0
        print(f"drafts={k} temp={temp}: {len(out)/dt:6.1f} tok/s  {len(out)/steps:.2f} tokens/step")
        eng.free_seq(s)
print("sample:", repr(tok.decode(out[:200])))
