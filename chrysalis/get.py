"""A model's files from Hugging Face the first time it is wanted: its weights and vision tower, at their MODELS paths
under data/models, laid out the same in REPO. Hugging Face's client resumes them, and its Xet transfers ran at the
line's speed here (104 MB/s, where one HTTP stream got 87 and eight ranges at once 96). One bar for the model: the
client's own are one per file."""
import os
import sys
import time

from .models import MODELS

REPO = "michaelDinelle/chrysalis"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def missing(name):
    """The model's files not here yet, as paths in REPO."""
    spec = MODELS[name]
    return [p.removeprefix("data/models/") for p in (spec["path"], spec["vision"]) if not os.path.exists(os.path.join(ROOT, p))]


def fetch(name):
    todo = missing(name)
    if not todo:
        return
    from huggingface_hub import HfApi, hf_hub_download
    from tqdm.auto import tqdm
    try:
        total = sum(i.size for i in HfApi().get_paths_info(REPO, todo)) or 1
    except Exception as e:                  # a line, not a traceback: no network, the repo not reachable
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
        for f in todo:
            hf_hub_download(REPO, f, local_dir=os.path.join(ROOT, "data/models"), tqdm_class=Bar)
    except Exception as e:                  # no network, a full disk
        sys.exit(f"\n  the download stopped ({type(e).__name__}): it carries on from where it was as Chrysalis next starts")
    done = total
    draw(end=True)
