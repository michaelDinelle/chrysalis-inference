"""Decode speed deep into a real context: ~N tokens of real C++ source as the user message, then a question,
answered with thinking at Qwen's sampling settings.  usage: bench_longctx.py [tokens=120000] [gen=512]"""
import glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kiln.chat import Renderer, Tokens
from kiln.engine import Engine, Sampling

target = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 512
tok = Tokens("data/models/tokenizer.json")
files = sorted(glob.glob("data/llama.cpp/src/**/*.cpp", recursive=True) + glob.glob("data/llama.cpp/ggml/src/ggml-cuda/*.cu"))
parts, total = [], 0
for f in files:
    text = open(f, errors="replace").read()
    n = len(tok.full.encode(text, add_special_tokens=False).ids)
    if total + n > target:
        break
    parts.append(f"// ===== {f} =====\n{text}")
    total += n
question = "\n\nIn the code above, how does the server decide which slot handles a new request? Answer briefly."
prompt, _ = Renderer(tok, lambda k: None).render([{"role": "user", "content": "".join(parts) + question}])
eng = Engine("data/models/kiln-mix.gguf")
s = eng.new_seq()
t0 = time.time()
eng.prefill(s, prompt)
pf = time.time() - t0
print(f"context {len(prompt)} tokens, prefill {pf:.1f}s ({len(prompt) / pf:.0f} tok/s)", flush=True)
for k in (3, 0):
    eng.free_seq(s); s = eng.new_seq(); eng.prefill(s, prompt)
    sp = Sampling(temperature=1.0, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=k)
    out, steps, t0 = [], 0, time.time()
    while len(out) < n_gen:
        out += eng.generate(s, sp, 16); steps += 1
    dt = time.time() - t0
    print(f"drafts={k}: {len(out) / dt:.1f} tok/s, {len(out) / steps:.2f} tokens/step at depth {len(prompt)}", flush=True)
print("sample:", repr(tok.decode(out[:160])))
