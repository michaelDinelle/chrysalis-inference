"""Routing statistics of Flash-Next on a corpus: picks per (layer, expert) from kiln's own prefill, saved as [48][512]
counts. fn_quantize.py can blend them with the imatrix counts when it picks the experts that live in VRAM, so the
VRAM tier fits the work the model actually does here (coding) rather than only Unsloth's calibration text.
  python tools_fn/route_stats.py MODEL OUT.npy ROOT:TOKENS [ROOT:TOKENS ...]
Files under each ROOT (source code and Markdown; build trees, dependencies and data skipped) are sampled at random
(fixed seed), at most 4,096 tokens each, until the root's token budget is reached; each file is its own sequence."""
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenizers import Tokenizer  # noqa: E402
from kiln.engine import Engine  # noqa: E402
from kiln.models import MODELS  # noqa: E402

EXT = {".c", ".cc", ".cpp", ".h", ".hpp", ".hip", ".cu", ".py", ".ts", ".tsx", ".js", ".rs", ".go", ".gml", ".sh", ".md", ".json"}
SKIP = {"node_modules", ".git", "build", "build_fn", "dist", "target", "__pycache__", "venv", ".venv", "rocm", "models"}


def files_under(root):
    out = []
    for d, dirs, names in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP and not x.startswith(".")]
        for n in names:
            p = os.path.join(d, n)
            if os.path.splitext(n)[1] in EXT and 200 < os.path.getsize(p) < 200_000:
                out.append(p)
    return sorted(out)


def main():
    model, out = sys.argv[1], sys.argv[2]
    tok = Tokenizer.from_file(os.path.join(ROOT, "data/models/tokenizer.json"))
    eng = Engine(model, max_seqs=2, umax=MODELS["qwen3.8-flash-next"]["chunk"], lib=os.path.join(ROOT, "build/libkiln_fn.so"))
    import ctypes as C
    eng.lib.kiln_route_counts.argtypes = [C.c_void_p]
    eng.lib.kiln_route_counts(None)
    rng = random.Random(7)
    total = 0
    for spec in sys.argv[3:]:
        root, budget = spec.rsplit(":", 1)
        budget = int(budget)
        fs = files_under(os.path.expanduser(root))
        rng.shuffle(fs)
        used = nfiles = 0
        for f in fs:
            if used >= budget:
                break
            try:
                text = open(f, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            ids = tok.encode(text, add_special_tokens=False).ids[:4096]
            if len(ids) < 64:
                continue
            s = eng.new_seq()
            eng.prefill(s, ids)
            eng.free_seq(s)
            used += len(ids); nfiles += 1
        total += used
        print(f"{root}: {nfiles} files, {used} tokens", flush=True)
    counts = np.zeros((48, 512), dtype=np.int64)
    eng.lib.kiln_route_counts(counts.ctypes.data)
    np.save(out, counts)
    print(f"{total} tokens, {counts.sum()} picks -> {out}")


if __name__ == "__main__":
    main()
