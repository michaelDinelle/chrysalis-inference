"""kiln command line: `python -m kiln serve` starts the OpenAI-compatible server."""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(prog="kiln", description="Qwen3.8-27B inference server for one RX 7900 XTX")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the OpenAI-compatible server")
    s.add_argument("--model", default=os.path.join(ROOT, "data/models/kiln-mix.gguf"))
    s.add_argument("--tokenizer", default=os.path.join(ROOT, "data/models/tokenizer.json"))
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--slots", type=int, default=3, help="sequences kept resident on the GPU")
    s.add_argument("--draft", type=int, default=3, help="MTP draft tokens per step (0 disables speculation)")
    s.add_argument("--chunk", type=int, default=2048, help="prefill chunk size (bigger is faster, costs VRAM)")
    a = ap.parse_args()
    if a.cmd == "serve":
        # Start what takes longest first: the GPU wakes and the weights load while the rest of Python starts up.
        from .engine import load_async, wake_gpus
        wake_gpus()
        engine = load_async(a.model, max_seqs=a.slots, umax=a.chunk, warmup=False)
        from .server import Server, serve
        serve(lambda: Server(a.model, a.tokenizer, slots=a.slots, n_draft=a.draft, umax=a.chunk, engine=engine, warmup=False),
              a.host, a.port)


if __name__ == "__main__":
    main()
