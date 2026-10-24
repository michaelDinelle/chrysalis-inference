import argparse
import os

from .models import DEFAULT, MODELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(prog="chrysalis", description="Qwen3.8 inference server for one RX 7900 XTX (27B and Flash-Next)")
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
                   help="restarts to have a PCIe link that came up slower than this machine's best trained again (chrysalis/link.py)")
    g = sub.add_parser("get", help="download a model's files (serve does, for the one it loads)")
    g.add_argument("model", nargs="?", default=os.environ.get("CHRYSALIS_MODEL") or DEFAULT, choices=list(MODELS))
    g.add_argument("--check", action="store_true", help="download nothing: exit 1 if any are missing")
    a = ap.parse_args()
    if a.cmd == "get":
        from . import get
        if a.check:
            raise SystemExit(1 if get.missing(a.model) else 0)
        get.fetch(a.model)
    if a.cmd == "serve":
        if not a.model:
            from . import get
            get.fetch(a.use)
        import threading
        from . import link
        from .engine import load_async, wake_gpus
        dev = link.device()
        if dev and os.environ.get(link.TRY) and not link.wait_suspended(dev):
            os.environ[link.TRY] = str(a.link_tries)
        wake_gpus()
        node = dev and link.render_node(dev)
        if node:
            def check():
                try:
                    os.close(os.open(node, os.O_RDWR))
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
