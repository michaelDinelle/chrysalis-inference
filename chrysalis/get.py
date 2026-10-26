import os
import sys
import time

from .models import MODELS

REPO = "michaelDinelle/chrysalis"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def files(name):
    spec = MODELS[name]
    repo, rev, into, names = spec.get("upstream") or (None, None, None, [])
    return [(REPO, None, "data/models", p.removeprefix("data/models/")) for p in (spec["path"], spec["vision"])] + [(repo, rev, into, f) for f in names]


def missing(name):
    return [f for f in files(name) if not os.path.exists(os.path.join(ROOT, f[2], f[3]))]


def fetch(name):
    todo = missing(name)
    if not todo:
        return
    from huggingface_hub import HfApi, hf_hub_download
    from tqdm.auto import tqdm
    try:
        total = sum(i.size for repo, rev, _, f in todo for i in HfApi().get_paths_info(repo, [f], revision=rev)) or 1
    except Exception as e:
        sys.exit(f"  the models' repo cannot be reached ({type(e).__name__}): it tries again as Chrysalis next starts")
    done, t0, last, tty = 0, time.time(), 0.0, sys.stderr.isatty()

    def draw(end=False):
        rate = done / max(time.time() - t0, 0.1)
        left = (total - done) / rate if rate else 0
        bar = int(28 * done / total)
        tail = "done" if end else "starting" if done < total / 200 else f"{rate / 1e6:.0f} MB/s, {int(left // 60)}:{int(left % 60):02d} left"
        line = f"  {name:<20} \033[35m{'━' * bar}\033[2m{'━' * (28 - bar)}\033[0m  {done / 1e9:4.1f} / {total / 1e9:.1f} GB  {tail}"
        sys.stderr.write(("\r\033[K" + line + ("\n" if end else "")) if tty else (line + "\n" if end else ""))

    class Bar(tqdm):
        def __init__(self, *a, **k):
            super().__init__(*a, **{**k, "file": open(os.devnull, "w")})

        def update(self, n=1):
            nonlocal done, last
            done += n or 0
            if time.time() - last > 0.25:
                last = time.time()
                draw()
            return super().update(n)

    try:
        for repo, rev, into, f in todo:
            hf_hub_download(repo, f, revision=rev, local_dir=os.path.join(ROOT, into), tqdm_class=Bar)
    except Exception as e:
        sys.exit(f"\n  the download stopped ({type(e).__name__}): it carries on from where it was as Chrysalis next starts")
    done = total
    draw(end=True)
