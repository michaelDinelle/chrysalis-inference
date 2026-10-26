import os
import subprocess
import sys
import time

import ctypes as C

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenizers import Tokenizer
from chrysalis.engine import Engine
from chrysalis.models import MODELS
sys.path.insert(0, os.path.join(ROOT, "data/llama.cpp/gguf-py"))
from gguf import GGUFReader

MODEL = os.path.join(ROOT, "data/models/flash-next/chrysalis-fn.gguf")
REF = os.path.join(ROOT, "data/models/flash-next/src/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf")
MAX_KL, MIN_TOP1 = {"decode": 0.021, "prefill": 0.021, "mixed": 0.016}, 0.92
SELF_KL = 0.012
TEXT = '''<|im_start|>user
Write a Python function that merges overlapping intervals.<|im_end|>
<|im_start|>assistant
<think>
The standard approach: sort by start, then sweep, extending the last interval when they overlap.
</think>

def merge(intervals):
    intervals.sort(key=lambda iv: iv[0])
    out = []
    for start, end in intervals:
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out

The function runs in O(n log n) time because of the sort; the sweep itself is linear. Edge cases such as
empty input, a single interval, or intervals that only touch at an endpoint are handled naturally.'''

def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


MODEL = arg("--model", MODEL)
if arg("--text"):
    TEXT = open(arg("--text")).read()
tok = Tokenizer.from_file(os.path.join(ROOT, "data/models/tokenizer.json"))
ids = tok.encode(TEXT, add_special_tokens=False).ids
n = len(ids)
CUTS = range(12, n + 1, 3)
out_dir = os.path.join(ROOT, "build/cmp_fn" + ("_" + arg("--tag") if arg("--tag") else ""))
os.makedirs(out_dir, exist_ok=True)
np.array(ids, dtype=np.int32).tofile(os.path.join(out_dir, "tokens.bin"))
ref_path = os.path.join(out_dir, "ref.bin")
if not os.path.exists(ref_path) or "--fresh" in sys.argv:
    print(f"reference: llama.cpp on the Q8_0 source, {n} tokens (CPU, weights streamed from disk)...", flush=True)
    subprocess.run([os.path.join(ROOT, "build/ref_logits_fn"), REF, os.path.join(out_dir, "tokens.bin"), ref_path], check=True)
ref = np.fromfile(ref_path, dtype=np.float32).reshape(n, -1)
if "--ref-only" in sys.argv:
    sys.exit(0)


def logsoftmax(x):
    x = x.astype(np.float64)
    x -= x.max()
    return x - np.log(np.exp(x).sum())


def stats(name, pairs, max_kl):
    kls, top1 = [], 0
    for i, ours in pairs:
        lp, lq = logsoftmax(ref[i]), logsoftmax(ours)
        kls.append(float((np.exp(lp) * (lp - lq)).sum()))
        top1 += int(ref[i].argmax() == ours.argmax())
    print(f"{name:28s} positions {len(pairs):4d}  mean KL {np.mean(kls):.5f}  max KL {np.max(kls):.4f}  "
          f"top-1 agree {100 * top1 / len(pairs):.1f}%  {'ok' if np.mean(kls) <= max_kl and top1 / len(pairs) >= MIN_TOP1 else 'FAIL'}", flush=True)
    np.save(os.path.join(out_dir, "kl_" + name.split()[0] + ".npy"), np.array([[i, k] for (i, _), k in zip(pairs, kls)]))
    if len(pairs) > 1024:
        pos = np.array([i for i, _ in pairs]); kk = np.array(kls)
        print("   by position: " + "  ".join(f"{a}-{b}: {kk[(pos >= a) & (pos < b)].mean():.4f}"
                                            for a, b in ((0, 512), (512, 1024), (1024, 2048), (2048, 2560), (2560, 3072), (3072, 1 << 20))
                                            if ((pos >= a) & (pos < b)).any()), flush=True)
    return np.mean(kls) <= max_kl and top1 / len(pairs) >= MIN_TOP1


eng = Engine(MODEL, max_seqs=2, umax=MODELS["qwen3.8-flash-next"]["chunk"], lib=os.path.join(ROOT, "build/libchrysalis_fn.so"))
eng.lib.chrysalis_tune.restype, eng.lib.chrysalis_tune.argtypes = C.c_double, [C.c_char_p, C.c_double]
eng.lib.chrysalis_tune(b"split", 1); eng.lib.chrysalis_tune(b"gfrac", 0.3); eng.lib.chrysalis_tune(b"experts", 0)


def against(name, pairs, decoded):
    kls = []
    for i, ours in pairs:
        lp, lq = logsoftmax(decoded[i]), logsoftmax(ours)
        kls.append(float((np.exp(lp) * (lp - lq)).sum()))
    ok = np.mean(kls) <= SELF_KL
    print(f"{name:28s} positions {len(pairs):4d}  mean KL {np.mean(kls):.5f} from the decode path  {'ok' if ok else 'FAIL'}", flush=True)
    return ok


def paths(tag):
    s = eng.new_seq()
    pairs = []
    for i, t in enumerate(ids):
        eng.prefill(s, [t])
        pairs.append((i, eng.last_logits()))
    bad = not stats(f"decode (T=1 per token){tag}", pairs, MAX_KL["decode"])
    decoded = dict(pairs)
    eng.free_seq(s)

    pairs = []
    for cut in CUTS:
        s = eng.new_seq()
        eng.prefill(s, ids[:cut])
        pairs.append((cut - 1, eng.last_logits()))
        eng.free_seq(s)
    bad += not stats(f"prefill (one chunk){tag}", pairs, MAX_KL["prefill"])
    bad += not against(f"prefill = decode{tag}", pairs, decoded)
    if tag:
        eng.lib.chrysalis_tune(b"split", 0)
        pairs = []
        for cut in CUTS:
            s = eng.new_seq()
            eng.prefill(s, ids[:cut])
            pairs.append((cut - 1, eng.last_logits()))
            eng.free_seq(s)
        eng.lib.chrysalis_tune(b"split", 1)
        bad += not stats(f"prefill, cold on the GPU{tag}", pairs, MAX_KL["prefill"])
        bad += not against(f"prefill, cold on the GPU = decode{tag}", pairs, decoded)

    s = eng.new_seq()
    eng.prefill(s, ids[:40])
    pairs = []
    for i in range(40, n):
        eng.prefill(s, [ids[i]])
        pairs.append((i, eng.last_logits()))
    bad += not stats(f"prefill 40 + decode{tag}", pairs, MAX_KL["mixed"])
    eng.free_seq(s)
    return bad


def placed():
    x = eng.expert_stats()
    return x["placed"], x["placed_file_tier"]


HOT = [(t.data_offset, int(t.n_bytes), int(t.shape[-1])) for t in GGUFReader(MODEL).tensors if t.name.endswith("_hot.weight")]
HOT_EXPERT = sum(nb for _, nb, _ in HOT) / sum(ne for _, _, ne in HOT) * 3


def mirror_rss():
    size_kb = sum((nb + 4095) // 4096 * 4 for _, nb, _ in HOT)
    found, rss = False, None
    for line in open("/proc/self/smaps"):
        if "-" in line.split()[0]:
            found = False
        elif line.startswith("Size:"):
            found = int(line.split()[1]) == size_kb
        elif found and line.startswith("Rss:"):
            rss = int(line.split()[1]) * 1024
    if rss is None:
        raise RuntimeError("no mapping the spill mirror's size")
    return rss
libc = C.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype, libc.mmap.argtypes = C.c_void_p, [C.c_void_p, C.c_size_t, C.c_int, C.c_int, C.c_int, C.c_long]
model_fd = os.open(MODEL, os.O_RDONLY)
model_size = os.fstat(model_fd).st_size
model_map = libc.mmap(None, model_size, 1, 1, model_fd, 0)


def hot_cached():
    vec = (C.c_ubyte * ((model_size + 4095) // 4096))()
    if libc.mincore(C.c_void_p(model_map), C.c_size_t(model_size), vec):
        raise OSError(C.get_errno(), "mincore")
    v = np.frombuffer(vec, np.uint8) & 1
    return int(sum(v[a // 4096:(a + nb + 4095) // 4096].sum() for a, nb, _ in HOT)) * 4096


def move_pool(prompt, steps):
    from chrysalis.engine import Sampling
    eng.lib.chrysalis_tune(b"experts", 1)
    s = eng.new_seq()
    eng.prefill(s, prompt)
    eng.begin(s, 7)
    sp = Sampling(temperature=1.0, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=2)
    for _ in range(steps):
        eng.generate(s, sp, 16)
    eng.free_seq(s)
    eng.lib.chrysalis_tune(b"experts", 0)
    time.sleep(1)


os.posix_fadvise(model_fd, 0, 0, os.POSIX_FADV_DONTNEED)
cached0 = hot_cached()
fails = paths("")
prose = tok.encode("<|im_start|>user\nWrite a long story about a lighthouse keeper and a storm.<|im_end|>\n<|im_start|>assistant\n",
                   add_special_tokens=False).ids
CAP = 500
eng.lib.chrysalis_tune(b"spill", CAP)
a = placed()
move_pool(prose, 400)
b, r1, out1 = placed(), mirror_rss(), eng.expert_stats()["mirrored"]
ok = b[0] - a[0] >= 1000 and out1 <= CAP and 0.95 * CAP * HOT_EXPERT <= r1 <= CAP * HOT_EXPERT + 64e6
print(f"pool after a prose reply: {b[0] - a[0]} experts placed, {out1} of the file's out of it (at most {CAP}), the mirror "
      f"holding {r1 / 1e9:.2f} GB ({CAP * HOT_EXPERT / 1e9:.2f} GB for {CAP})  {'ok' if ok else 'FAIL'}", flush=True)
fails += not ok
os.posix_fadvise(model_fd, 0, 0, os.POSIX_FADV_DONTNEED)
fails += paths(", pool moved by prose")
read = hot_cached() - cached0
print(f"paths with the file's experts out of the pool: {read / 1e6:.0f} MB of them read from the model file  {'ok' if read <= 0 else 'FAIL'}", flush=True)
fails += read > 0
code_ids = tok.encode("<|im_start|>user\nWrite a C++ hash map with open addressing: insert, find, erase, resize and an iterator, "
                      "then a test for each.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n```cpp\n", add_special_tokens=False).ids
move_pool(code_ids, 400)
c, r2, out2 = placed(), mirror_rss(), eng.expert_stats()["mirrored"]
ok = c[0] - b[0] >= 300 and c[1] - b[1] >= 50 and abs(r2 - out2 * HOT_EXPERT) <= 0.02 * out2 * HOT_EXPERT + 32e6
print(f"pool after code: {c[0] - b[0]} experts placed, {c[1] - b[1]} of them the file's back from the mirror, {out2} out of it, "
      f"the mirror holding {r2 / 1e9:.2f} GB ({out2 * HOT_EXPERT / 1e9:.2f} GB for {out2})  {'ok' if ok else 'FAIL'}", flush=True)
fails += not ok
fails += paths(", pool moved by code")
eng.lib.chrysalis_tune(b"spill", 100000)
eng.lib.chrysalis_tune(b"spill_spare", 1e6)
out3, r3 = eng.expert_stats()["mirrored"], mirror_rss()
move_pool(prose, 200)
d, out4, r4 = placed()[0] - c[0], eng.expert_stats()["mirrored"], mirror_rss()
ok = out3 >= 100 and out4 <= out3 // 4 and r4 <= out4 * HOT_EXPERT + 64e6 and d >= max(300, out3 - out4)
print(f"pool after prose with no RAM to spare: {d} experts placed, the file's back from the mirror, {out3} -> {out4} out of the pool, "
      f"the mirror {r3 / 1e9:.2f} -> {r4 / 1e9:.2f} GB  {'ok' if ok else 'FAIL'}", flush=True)
fails += not ok
print("all ok" if not fails else f"{fails} FAILURES")
sys.exit(fails != 0)
