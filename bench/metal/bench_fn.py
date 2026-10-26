import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from chrysalis.engine import Engine, Sampling

MODEL = os.path.join(ROOT, "data/models/flash-next/chrysalis-fn.gguf")
TOKENIZER = os.path.join(ROOT, "data/models/tokenizer.json")
SHORT = "Why is the sky blue?"
STORY = "Write a long story about a lighthouse keeper who finds a message in a bottle. Make it detailed."
WIKI = (Path.home() / "models/data/wikitext-2-raw/wiki.train.raw").read_text(encoding="utf-8")


def chat(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="ttft,short,story,doc,append,4k,32k")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(ROOT, "build/bench/fn_metal.jsonl"))
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(TOKENIZER)
    enc = lambda s: tk.encode(s).ids
    t = time.time()
    e = Engine(MODEL, max_seqs=2, umax=8192, lib=os.path.join(ROOT, "build/libchrysalis_fn.dylib"))
    load_s = time.time() - t
    greedy = Sampling(temperature=0.0, top_p=1.0, min_p=0.0, top_k=1, n_draft=0)
    res = {"load_s": round(load_s, 1)}

    def prefill(sid, toks):
        for i in range(0, len(toks), 8192):
            e.prefill(sid, toks[i:i + 8192])

    def decode(toks, n):
        sid = e.new_seq()
        t0 = time.time()
        prefill(sid, toks)
        t1 = time.time()
        e.begin(sid, 1)
        out = []
        first = None
        while len(out) < n:
            got = e.generate(sid, greedy, max_out=4)
            if first is None:
                first = time.time()
            out += got
        t2 = time.time()
        e.free_seq(sid)
        return {"prompt": len(toks), "prefill_s": round(t1 - t0, 3), "tok_s": round((len(out) - 2) / (t2 - first), 2), "text": tk.decode(out[:40])}

    def wiki(n_tok, rep):
        start = (1000 + rep * 600000 + n_tok * 7) % (len(WIKI) - int(n_tok * 4.6) - 1)
        toks = enc(f"Document {rep}-{n_tok}:\n" + WIKI[start:start + int(n_tok * 4.6)])
        return toks[:n_tok]

    cells = a.cells.split(",")
    st0 = e.expert_stats()
    for c in cells:
        rows = []
        for rep in range(a.reps):
            if c == "short":
                rows.append(decode(enc(chat(SHORT)), a.n))
            elif c == "story":
                rows.append(decode(enc(chat(STORY)), a.n))
            elif c == "doc":
                doc = tk.decode(wiki(4000, rep + 10))
                rows.append(decode(enc(chat(doc + "\n\nSummarize the document above.")), a.n))
            elif c == "ttft":
                sid = e.new_seq()
                toks = enc(chat(f"({rep}) " + SHORT))
                t0 = time.time()
                prefill(sid, toks)
                e.begin(sid, 1)
                e.generate(sid, greedy, max_out=1)
                rows.append({"prompt": len(toks), "ttft_s": round(time.time() - t0, 3)})
                e.free_seq(sid)
            elif c == "append":
                sid = e.new_seq()
                prefill(sid, enc(chat(tk.decode(wiki(4000, rep + 20)) + "\n\nWhat is this about?")))
                e.begin(sid, 1)
                for _ in range(16):
                    e.generate(sid, greedy, max_out=4)
                turn = enc("<|im_end|>\n<|im_start|>user\nNow list the three most important names in it, with one line on each, and say "
                           "which of them appears first.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
                t0 = time.time()
                prefill(sid, turn)
                e.begin(sid, 2)
                e.generate(sid, greedy, max_out=1)
                rows.append({"context": e.seq_len(sid), "turn": len(turn), "ttft_s": round(time.time() - t0, 3)})
                e.free_seq(sid)
            elif c in ("4k", "32k"):
                n = 4096 if c == "4k" else 32768
                toks = wiki(n, rep)
                sid = e.new_seq()
                t0 = time.time()
                prefill(sid, toks)
                dt = time.time() - t0
                rows.append({"prompt": len(toks), "prefill_s": round(dt, 2), "tok_s": round(len(toks) / dt, 1)})
                e.free_seq(sid)
            print(c, rows[-1], flush=True)
        res[c] = rows
    st = e.expert_stats()
    res["expert_picks"] = st["decode_picks"] - st0["decode_picks"]
    res["expert_hits"] = st["decode_picks_in_pool"] - st0["decode_picks_in_pool"]
    res["expert_read_mb_s"] = st["dma_bytes_per_s"] / 1e6
    res["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    res["capacity"] = e.capacity
    res["time"] = time.strftime("%Y-%m-%d %H:%M")
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    e.shutdown()


if __name__ == "__main__":
    main()
