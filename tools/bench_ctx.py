"""Prefill throughput and decode speed as the context grows. usage: bench_ctx.py depth1 depth2 ..."""
import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chrysalis.engine import Engine, Sampling
depths = [int(a) for a in sys.argv[1:]] or [0, 32768, 131072]
eng = Engine("data/models/chrysalis-mix.gguf")
s = eng.new_seq()
random.seed(0)
filled = 0
greedy = lambda k: Sampling(temperature=0.0, top_p=1.0, min_p=0.0, presence_penalty=0.0, top_k=1, n_draft=k)
for depth in depths:
    t0 = time.time(); n = depth - filled
    while filled < depth:
        chunk = min(8192, depth - filled)
        eng.prefill(s, [random.randrange(1000, 100000) for _ in range(chunk)]); filled += chunk
    if n > 0: print(f"prefill {filled - n:>6} -> {filled:>6}: {n / (time.time() - t0):7.1f} tok/s", flush=True)
    for k in (0, 3):
        eng.prefill(s, [random.randrange(1000, 100000)])     # fresh logits at this depth
        for _ in range(2): eng.generate(s, greedy(k), 16)
        t0 = time.time(); out = 0; steps = 12
        for _ in range(steps): out += len(eng.generate(s, greedy(k), 16))
        dt = time.time() - t0
        print(f"  depth {eng.seq_len(s):>6}  drafts={k}: {1000*dt/steps:6.1f} ms/step  {out/dt:6.1f} tok/s", flush=True)
        eng.truncate(s, eng.seq_len(s)) ; filled = eng.seq_len(s)
