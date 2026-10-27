import argparse
import glob
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
MODEL = os.path.join(ROOT, "data/models/qwen3.8-flash-next/qwen3.8-flash-next.gguf")
TOKENIZER = os.path.join(ROOT, "data/models/tokenizer.json")
REQUESTS = [
    "Explain how a hash table works and when to use one.",
    "Write a Python function that merges two sorted lists, with a short explanation.",
    "What were the main causes of the French Revolution?",
    "Give me a simple recipe for vegetable soup.",
    "Write a short poem about autumn leaves.",
    "Translate into French and German: 'The meeting has been moved to Thursday afternoon.'",
    "A train travels 180 km in 2.5 hours. What is its average speed? Show your steps.",
    "Draft a polite email asking my landlord to fix a leaking faucet.",
    "Compare TCP and UDP in a few paragraphs.",
    "What are some tips for sleeping better?",
    "Summarize the plot of Romeo and Juliet.",
    "How do vaccines train the immune system?",
    "Write a bash one-liner that counts the lines of all .py files in a directory tree, and explain it.",
    "What is the difference between a list and a tuple in Python?",
    "Plan a three-day trip to Kyoto.",
    "用中文简单介绍一下机器学习。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "engine/metal/flash-next/expert-profile.bin"))
    ap.add_argument("--wiki", type=int, default=8, help="wikitext slices of --seg tokens")
    ap.add_argument("--code", type=int, default=6, help="source-file slices of --seg tokens")
    ap.add_argument("--seg", type=int, default=1000)
    ap.add_argument("--reply", type=int, default=160, help="tokens of each chat reply")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    os.environ["CHRYSALIS_FN_STATS"] = os.path.join(tmp, "counts.bin")
    os.environ["CHRYSALIS_FN_PROFILE"] = os.path.join(tmp, "none")
    from tokenizers import Tokenizer
    from chrysalis.engine import Engine, Sampling
    tk = Tokenizer.from_file(TOKENIZER)
    e = Engine(MODEL, max_seqs=1, umax=4096, lib=os.path.join(ROOT, "build/libchrysalis_fn.dylib"))
    greedy = Sampling(temperature=0.0, top_p=1.0, min_p=0.0, top_k=1, n_draft=0)

    def feed(toks):
        sid = e.new_seq()
        for i in range(0, len(toks), 8):
            e.prefill(sid, toks[i:i + 8])
        return sid

    t0 = time.time()
    wiki = open(os.path.expanduser("~/models/data/wikitext-2-raw/wiki.train.raw"), encoding="utf-8").read()
    step = (len(wiki) - 3_000_000) // max(a.wiki, 1)
    for k in range(a.wiki):
        s = 3_000_000 + k * step
        e.free_seq(feed(tk.encode(wiki[s:s + a.seg * 5]).ids[:a.seg]))
        print(f"wiki {k} {time.time() - t0:.0f} s", flush=True)
    files = sorted(glob.glob(os.path.join(ROOT, "engine/metal/flash-next/*.mm")) + glob.glob(os.path.join(ROOT, "chrysalis/*.py")))
    for k in range(a.code):
        f = files[k * len(files) // max(a.code, 1)]
        e.free_seq(feed(tk.encode(open(f, encoding="utf-8").read()).ids[:a.seg]))
        print(f"code {os.path.basename(f)} {time.time() - t0:.0f} s", flush=True)
    for q in REQUESTS:
        sid = feed(tk.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").ids)
        e.begin(sid, 1)
        n = 0
        while n < a.reply:
            n += len(e.generate(sid, greedy, max_out=4))
        e.free_seq(sid)
        print(f"reply {q[:30]!r} {time.time() - t0:.0f} s", flush=True)
    e.shutdown()
    shutil.copy(os.environ["CHRYSALIS_FN_STATS"], a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
