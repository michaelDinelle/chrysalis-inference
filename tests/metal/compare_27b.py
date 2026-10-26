import glob, os, subprocess, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tokenizers import Tokenizer
from chrysalis.engine import Engine
from chrysalis.models import MODELS

DATA = os.path.join(ROOT, "data")
MODEL = os.path.join(ROOT, MODELS["qwen3.8-27b"]["path"])
LIB = os.path.join(ROOT, MODELS["qwen3.8-27b"]["lib"])
MAX_KL, MIN_TOP1 = {"decode": 0.004, "prefill": 0.008, "mixed": 0.004, "long": 0.01}, 0.95
TEXT = open(os.path.join(ROOT, "tests", "compare_ref.py")).read().split("TEXT = '''")[1].split("'''")[0]
TAIL = 8

tok = Tokenizer.from_file(os.path.join(DATA, "models/tokenizer.json"))
os.makedirs(os.path.join(ROOT, "build/cmp"), exist_ok=True)


def long_ids(n):
    src = sorted(glob.glob(os.path.join(ROOT, "engine", "*.h")) + glob.glob(os.path.join(ROOT, "chrysalis", "*.py")) +
                 glob.glob(os.path.join(ROOT, "web", "*.ts")))
    text = "".join(open(p, errors="replace").read() for p in src)
    ids = tok.encode(text, add_special_tokens=False).ids
    while len(ids) < n:
        ids += ids
    return ids[:n]


def reference(name, ids, positions=None):
    path = os.path.join(ROOT, f"build/cmp/{name}.bin")
    tb = os.path.join(ROOT, f"build/cmp/{name}.tokens")
    same = os.path.exists(tb) and np.array_equal(np.fromfile(tb, dtype=np.int32), np.array(ids, dtype=np.int32))
    if not os.path.exists(path) or not same or "--fresh" in sys.argv or name.startswith("gen"):
        np.array(ids, dtype=np.int32).tofile(tb)
        cmd = [os.path.join(ROOT, "build/ref_logits"), MODEL, tb, path]
        if positions is not None:
            cmd += [",".join(map(str, positions)), "1024"]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    return np.fromfile(path, dtype=np.float32).reshape(-1, 248320)


def logsoftmax(x):
    x = x.astype(np.float64); x -= x.max(); return x - np.log(np.exp(x).sum())


def stats(name, pairs, max_kl):
    kls, top1 = [], 0
    for ref, ours in pairs:
        lp, lq = logsoftmax(ref), logsoftmax(ours)
        kls.append(float((np.exp(lp) * (lp - lq)).sum()))
        top1 += int(ref.argmax() == ours.argmax())
    ok = np.mean(kls) <= max_kl and top1 / len(pairs) >= MIN_TOP1 and len(pairs) > 0
    print(f"{name:34s} positions {len(pairs):4d}  mean KL {np.mean(kls):.5f}  max KL {np.max(kls):.4f}  "
          f"top-1 agree {100*top1/len(pairs):.1f}%  {'ok' if ok else 'FAIL'}", flush=True)
    return ok


args = [a for a in sys.argv[1:] if not a.startswith("--")] or ["short", "4096", "generate"]
gen = "generate" in args
args = [a for a in args if a != "generate"]
refs = {}
ids_short = tok.encode(TEXT, add_special_tokens=False).ids
for a in args:
    if a == "short":
        refs[a] = reference("short", ids_short)
    else:
        n = int(a)
        refs[a] = reference(f"long{n}", long_ids(n), list(range(n - TAIL - 1, n)))

eng = Engine(MODEL, lib=LIB)
fails = 0
for a in args:
    if a == "short":
        ids, ref, n = ids_short, refs[a], len(ids_short)
        s = eng.new_seq(); pairs = []
        for i, t in enumerate(ids):
            eng.prefill(s, [t]); pairs.append((ref[i], eng.last_logits()))
        fails += not stats("decode (T=1 per token)", pairs, MAX_KL["decode"])
        eng.free_seq(s)
        pairs = []
        for cut in (16, 64, n // 2, n):
            s = eng.new_seq(); eng.prefill(s, ids[:cut]); pairs.append((ref[cut - 1], eng.last_logits())); eng.free_seq(s)
        fails += not stats("prefill (one chunk)", pairs, MAX_KL["prefill"])
        s = eng.new_seq(); eng.prefill(s, ids[:40]); pairs = []
        for i in range(40, n):
            eng.prefill(s, [ids[i]]); pairs.append((ref[i], eng.last_logits()))
        fails += not stats("prefill 40 + decode", pairs, MAX_KL["mixed"])
        eng.free_seq(s)
    else:
        n = int(a)
        ids, ref = long_ids(n), refs[a]
        s = eng.new_seq()
        eng.prefill(s, ids[:n - TAIL]); pairs = [(ref[0], eng.last_logits())]
        for i in range(n - TAIL, n):
            eng.prefill(s, [ids[i]]); pairs.append((ref[i - (n - TAIL - 1)], eng.last_logits()))
        fails += not stats(f"{n} tokens: prefill + {TAIL} decoded", pairs, MAX_KL["long"])
        eng.free_seq(s)
if gen:
    from chrysalis.engine import Sampling
    prompt = ids_short[:60]
    replies = {}
    for nd in (0, 3):
        s = eng.new_seq(); eng.prefill(s, prompt); eng.begin(s, 1)
        out = []
        while len(out) < 64:
            out += eng.generate(s, Sampling(temperature=0, n_draft=nd), 64 - len(out))
        replies[nd] = out[:64]
        eng.free_seq(s)
    eng.shutdown()
    for nd, out in replies.items():
        toks = prompt + out
        ref = reference(f"gen{nd}", toks, list(range(len(prompt) - 1, len(toks) - 1)))
        agree = sum(int(ref[i].argmax() == out[i]) for i in range(len(out)))
        lp = [logsoftmax(ref[i])[out[i]] for i in range(len(out))]
        ok = agree >= 0.95 * len(out)
        print(f"{'generate, ' + str(nd) + ' drafts':34s} tokens {len(out):4d}  llama.cpp's top-1 {100*agree/len(out):.1f}%  "
              f"mean log p {np.mean(lp):.4f}  {'ok' if ok else 'FAIL'}", flush=True)
        fails += not ok
print("all ok" if not fails else f"{fails} FAILURES")
sys.exit(fails != 0)
