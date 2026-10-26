import ctypes as C
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chrysalis.engine import Engine
from chrysalis.models import MODELS


def logsoftmax(x):
    x = x.astype(np.float64)
    x -= x.max()
    return x - np.log(np.exp(x).sum())


if __name__ == "__main__":
    use, tokens_path, ref_path = sys.argv[1:4]
    spec = MODELS[use]
    ids = np.fromfile(tokens_path, dtype=np.int32)
    ref = np.fromfile(ref_path, dtype=np.float32).reshape(len(ids), -1)
    eng = Engine(os.path.join(ROOT, spec["path"]), max_seqs=2, umax=spec["chunk"], lib=os.path.join(ROOT, spec["lib"]))
    if hasattr(eng.lib, "chrysalis_tune"):
        eng.lib.chrysalis_tune.restype, eng.lib.chrysalis_tune.argtypes = C.c_double, [C.c_char_p, C.c_double]
        eng.lib.chrysalis_tune(b"split", 1); eng.lib.chrysalis_tune(b"gfrac", 0.3); eng.lib.chrysalis_tune(b"experts", 0)
    s = eng.new_seq()
    kls, top1 = [], 0
    for i, t in enumerate(ids):
        eng.prefill(s, [int(t)])
        ours = eng.last_logits()
        lp, lq = logsoftmax(ref[i]), logsoftmax(ours)
        kls.append(float((np.exp(lp) * (lp - lq)).sum()))
        top1 += int(ref[i].argmax() == ours.argmax())
    print(json.dumps({"kl_mean": float(np.mean(kls)), "kl_p999": float(np.percentile(kls, 99.9)), "kl_max": float(np.max(kls)),
                      "top1": top1 / len(ids)}))
