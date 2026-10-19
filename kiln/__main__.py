"""kiln command line: `python -m kiln serve` starts the OpenAI-compatible server."""
import argparse
import os

from .models import DEFAULT, MODELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(prog="kiln", description="Qwen3.8 inference server for one RX 7900 XTX (27B and Flash-Next)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the OpenAI-compatible server")
    s.add_argument("--use", default=DEFAULT, choices=list(MODELS), help="model to load first (requests can switch)")
    s.add_argument("--model", default=None, help="model file (default: the --use model's)")
    s.add_argument("--tokenizer", default=os.path.join(ROOT, "data/models/tokenizer.json"))
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--slots", type=int, default=3, help="sequences kept resident on the GPU")
    s.add_argument("--draft", type=int, default=None, help="MTP draft tokens per step (0 disables speculation; default: the model's)")
    s.add_argument("--chunk", type=int, default=None, help="prefill chunk size (bigger is faster, costs VRAM; default: the model's)")
    s.add_argument("--lookup", type=int, default=None,
                   help="drafts copied from earlier in the conversation per step, at most (0 disables; default: the model's)")
    s.add_argument("--max-pixels", type=int, default=None,
                   help="the most pixels a picture is read at, fewer tokens for large ones (default: the official 16777216)")
    s.add_argument("--link-tries", type=int, default=4,
                   help="restarts to have a PCIe link that came up slower than this machine's best trained again (kiln/link.py)")
    a = ap.parse_args()
    if a.cmd == "serve":
        # Start what takes longest first: the GPU wakes and the weights load while the rest of Python starts up.
        import threading
        from . import link
        from .engine import load_async, wake_gpus
        dev = link.device()
        # a restart for the link: the GPU has to suspend to train it again; one kept awake (a display on it, another
        # process) cannot be, so the check only says how it came up
        if dev and os.environ.get(link.TRY) and not link.wait_suspended(dev):
            os.environ[link.TRY] = str(a.link_tries)
        wake_gpus()
        node = dev and link.render_node(dev)
        if node:
            def check():
                try:
                    os.close(os.open(node, os.O_RDWR))     # returns once the GPU is awake, its link trained
                except OSError:
                    return
                link.check(dev, a.link_tries)
            threading.Thread(target=check, daemon=True).start()
        spec = MODELS[a.use]
        path = a.model or os.path.join(ROOT, spec["path"])
        engine = load_async(path, max_seqs=a.slots, umax=a.chunk or spec["chunk"], warmup=False, lib=os.path.join(ROOT, spec["lib"]))
        from .server import Server, serve
        serve(lambda: Server(path, a.tokenizer, slots=a.slots, n_draft=a.draft, umax=a.chunk, engine=engine, warmup=False,
                             model=a.use, lookup=a.lookup, max_pixels=a.max_pixels), a.host, a.port, model=a.use)


if __name__ == "__main__":
    main()
