"""Time a decode-path forward (T tokens appended) for T = 1..8."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kiln.engine import Engine
eng = Engine("data/models/kiln-mix.gguf")
s = eng.new_seq(); eng.prefill(s, list(range(1000, 1300)))
for T in range(1, 9):
    for _ in range(3): eng.prefill(s, list(range(2000, 2000 + T)))
    t0 = time.time(); n = 20
    for _ in range(n): eng.prefill(s, list(range(2000, 2000 + T)))
    dt = (time.time() - t0) / n
    print(f"T={T}: {dt*1000:6.2f} ms per forward (incl. MTP catch-up, lm_head, top-k)")
