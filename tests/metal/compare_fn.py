import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from chrysalis.engine import Engine
from chrysalis.models import MODELS

MAX_KL, MIN_TOP1, SELF_KL = {"decode": 0.021, "prefill": 0.021, "mixed": 0.016}, 0.92, 0.012
REF = os.path.join(ROOT, "build/cmp_fn")
ids = np.fromfile(os.path.join(REF, "tokens.bin"), np.int32).tolist()
n = len(ids)
ref = np.fromfile(os.path.join(REF, "ref.bin"), np.float32).reshape(n, -1)
CUTS = range(12, n + 1, 3)


def logsoftmax(x):
    x = x.astype(np.float64)
    x -= x.max()
    return x - np.log(np.exp(x).sum())


def kl(p, q):
    lp, lq = logsoftmax(p), logsoftmax(q)
    return float((np.exp(lp) * (lp - lq)).sum())


def stats(name, pairs, max_kl):
    kls = [kl(ref[i], ours) for i, ours in pairs]
    top1 = sum(int(ref[i].argmax() == ours.argmax()) for i, ours in pairs) / len(pairs)
    ok = np.mean(kls) <= max_kl and top1 >= MIN_TOP1
    print(f"{name:28s} positions {len(pairs):4d}  mean KL {np.mean(kls):.5f}  max KL {np.max(kls):.4f}  "
          f"top-1 agree {100 * top1:.1f}%  {'ok' if ok else 'FAIL'}", flush=True)
    return ok


def against(name, pairs, decoded):
    k = np.mean([kl(decoded[i], ours) for i, ours in pairs])
    print(f"{name:28s} positions {len(pairs):4d}  mean KL {k:.5f} from the decode path  {'ok' if k <= SELF_KL else 'FAIL'}", flush=True)
    return k <= SELF_KL


spec = MODELS["qwen3.8-flash-next"]
eng = Engine(os.path.join(ROOT, spec["path"]), max_seqs=2, umax=spec["chunk"], lib=os.path.join(ROOT, spec["lib"]))
s = eng.new_seq()
pairs = []
for i, t in enumerate(ids):
    eng.prefill(s, [t])
    pairs.append((i, eng.last_logits().copy()))
eng.free_seq(s)
fails = not stats("decode (T=1 per token)", pairs, MAX_KL["decode"])
decoded = dict(pairs)
pairs = []
for cut in CUTS:
    s = eng.new_seq()
    eng.prefill(s, ids[:cut])
    pairs.append((cut - 1, eng.last_logits().copy()))
    eng.free_seq(s)
fails += not stats("prefill (one chunk)", pairs, MAX_KL["prefill"])
fails += not against("prefill = decode", pairs, decoded)
s = eng.new_seq()
eng.prefill(s, ids[:40])
pairs = []
for i in range(40, n):
    eng.prefill(s, [ids[i]])
    pairs.append((i, eng.last_logits().copy()))
eng.free_seq(s)
fails += not stats("prefill 40 + decode", pairs, MAX_KL["mixed"])
eng.shutdown()
print("all ok" if not fails else f"{fails} FAILURES")
sys.exit(fails != 0)
