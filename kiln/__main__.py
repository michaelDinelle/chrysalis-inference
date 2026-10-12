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
    o = sub.add_parser("opencode", help="add kiln as the default provider in ~/.config/opencode/opencode.json")
    o.add_argument("--config", default=os.path.expanduser("~/.config/opencode/opencode.json"))
    o.add_argument("--url", default="http://127.0.0.1:8080/v1")
    o.add_argument("--context", type=int, default=0, help="context limit to advertise (default: ask the server)")
    a = ap.parse_args()
    if a.cmd == "opencode":
        return opencode(a)
    if a.cmd == "serve":
        # Start what takes longest first: the GPU wakes and the weights load while the rest of Python starts up.
        from .engine import load_async, wake_gpus
        wake_gpus()
        engine = load_async(a.model, max_seqs=a.slots, umax=a.chunk, warmup=False)
        from .server import Server, serve
        serve(lambda: Server(a.model, a.tokenizer, slots=a.slots, n_draft=a.draft, umax=a.chunk, engine=engine, warmup=False),
              a.host, a.port)



def opencode(a):
    """Merge a kiln provider into OpenCode's config (keeps every other setting; backs the file up first)."""
    import json
    import shutil
    import urllib.request
    ctx = a.context
    if not ctx:
        try:
            ctx = json.load(urllib.request.urlopen(a.url + "/models", timeout=5))["data"][0]["context_length"]
        except Exception:
            ctx = 262144
    cfg = {}
    if os.path.exists(a.config):
        shutil.copy(a.config, a.config + ".bak")
        cfg = json.load(open(a.config))
    cfg.setdefault("$schema", "https://opencode.ai/config.json")
    cfg["model"] = "kiln/qwen3.8-27b"
    cfg["small_model"] = "kiln/qwen3.8-27b-fast"
    cfg.setdefault("provider", {})["kiln"] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "kiln (local Qwen3.8-27B)",
        "options": {"baseURL": a.url, "apiKey": "kiln", "headerTimeout": False, "chunkTimeout": 900000},
        "models": {
            "qwen3.8-27b": {"name": "Qwen3.8 27B (kiln)", "tool_call": True, "reasoning": True,
                            "limit": {"context": ctx, "output": 32768},
                            "variants": {e: {"reasoning_effort": e} for e in ("xhigh", "medium", "low")}},
            "qwen3.8-27b-fast": {"name": "Qwen3.8 27B, no thinking (kiln)", "tool_call": True,
                                 "limit": {"context": ctx, "output": 16384}},
        },
    }
    cfg["compaction"] = {**cfg.get("compaction", {}), "auto": True, "prune": False}
    os.makedirs(os.path.dirname(a.config), exist_ok=True)
    json.dump(cfg, open(a.config, "w"), indent=2)
    print(f"kiln is now OpenCode's default model ({a.config}, context {ctx} tokens)")


if __name__ == "__main__":
    main()
