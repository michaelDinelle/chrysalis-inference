"""Compare Chrysalis's Flash-Next next-token distributions with llama.cpp's on the Q8_0 source (KL divergence, top-1).

The two run different weights (chrysalis-fn.gguf: Q6_K/Q8_0 dense, IQ4_XS experts in VRAM, Q3_K / Q2_K in RAM;
llama.cpp: Q8_0), so the numbers are Chrysalis's quantization error plus any implementation error: decode, prefill and
mixed paths must agree with each other, and the KL should match the prediction (~0.017): a path fails past its MAX_KL or
under MIN_TOP1, a prefill path also past SELF_KL from the decode path at the same positions. The cold experts' CPU / GPU
split is fixed here (all of a prefill's on the CPU, gfrac 0.3 at decode, the expert pool as loaded): the load-time
calibration and the pool would otherwise move it, and with it the rounding, so every run gives the same numbers. Then
the pool moves (engine_fn/experts.cpp) and the paths run again under the same limits: once after a prose reply has
pushed the text's experts out of VRAM (the file's VRAM-tier ones then computed by the CPU from their copies in the spill
mirror), once after code with the text at its end has pulled many back in (RAM-tier experts in their own formats in pool
slots, VRAM-tier ones copied back from the mirror into other slots). Around them: the mirror holds no more than it may,
as the process's memory shows, gives a copy back as its expert returns and takes no more when the machine has no RAM to
spare; and the paths read none of those experts from the model file (computed from its mapping, they come from the page
cache or the disk, whichever has them).
  python tests/compare_fn.py [--fresh] [--ref-only] [--text FILE --tag NAME] [--model FILE]
"""
import os
import subprocess
import sys
import time

import ctypes as C

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenizers import Tokenizer  # noqa: E402
from chrysalis.engine import Engine  # noqa: E402
from chrysalis.models import MODELS  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "data/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402

MODEL = os.path.join(ROOT, "data/models/flash-next/chrysalis-fn.gguf")
REF = os.path.join(ROOT, "data/models/flash-next/src/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf")
# mean KL with this split (2026-10-18, the same every run with the pool as loaded): decode 0.0155, prefill 0.0119,
# mixed 0.0108. Quantization sets most of it, so a broken kernel moves it less than on the 27B: a decode step that
# forgets twice as fast gives 0.0284 and 0.0218. Once the pool has moved, each run's pool differs (it follows a sampled
# reply) and computes the experts on other devices, so the numbers do too, each pool's the same every run (bit for bit):
# decode 0.0142-0.0165, prefill 0.0108-0.0171. Prefill is judged over 52 prompt lengths (over 4 it went from 0.002 to
# 0.015 on one position's rounding) and against Chrysalis's own decode path at the same positions, which no pool moves
# far: 0.0046-0.0077 (SELF_KL).
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
if arg("--text"):                            # another evaluation text, with its own reference directory
    TEXT = open(arg("--text")).read()
tok = Tokenizer.from_file(os.path.join(ROOT, "data/models/tokenizer.json"))
ids = tok.encode(TEXT, add_special_tokens=False).ids
n = len(ids)
CUTS = range(12, n + 1, 3)  # prompt lengths of the prefill path (each a pass of over 8 tokens)
out_dir = os.path.join(ROOT, "build_fn/cmp" + ("_" + arg("--tag") if arg("--tag") else ""))
os.makedirs(out_dir, exist_ok=True)
np.array(ids, dtype=np.int32).tofile(os.path.join(out_dir, "tokens.bin"))
ref_path = os.path.join(out_dir, "ref.bin")
if not os.path.exists(ref_path) or "--fresh" in sys.argv:
    print(f"reference: llama.cpp on the Q8_0 source, {n} tokens (CPU, weights streamed from disk)...", flush=True)
    subprocess.run([os.path.join(ROOT, "build_fn/ref_logits_fn"), REF, os.path.join(out_dir, "tokens.bin"), ref_path], check=True)
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
    if len(pairs) > 1024:                    # by position: long-context paths (sparse selection past 2048 cells) show here
        pos = np.array([i for i, _ in pairs]); kk = np.array(kls)
        print("   by position: " + "  ".join(f"{a}-{b}: {kk[(pos >= a) & (pos < b)].mean():.4f}"
                                            for a, b in ((0, 512), (512, 1024), (1024, 2048), (2048, 2560), (2560, 3072), (3072, 1 << 20))
                                            if ((pos >= a) & (pos < b)).any()), flush=True)
    return np.mean(kls) <= max_kl and top1 / len(pairs) >= MIN_TOP1


eng = Engine(MODEL, max_seqs=2, umax=MODELS["qwen3.8-flash-next"]["chunk"], lib=os.path.join(ROOT, "build/libchrysalis_fn.so"))
eng.lib.chrysalis_tune.restype, eng.lib.chrysalis_tune.argtypes = C.c_double, [C.c_char_p, C.c_double]
eng.lib.chrysalis_tune(b"split", 1); eng.lib.chrysalis_tune(b"gfrac", 0.3); eng.lib.chrysalis_tune(b"experts", 0)


def against(name, pairs, decoded):
    """Mean KL of a prefill path from Chrysalis's decode path at the same positions."""
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
    for i, t in enumerate(ids):             # decode path: one token at a time
        eng.prefill(s, [t])
        pairs.append((i, eng.last_logits()))
    bad = not stats(f"decode (T=1 per token){tag}", pairs, MAX_KL["decode"])
    decoded = dict(pairs)
    eng.free_seq(s)

    pairs = []
    for cut in CUTS:                        # prefill path
        s = eng.new_seq()
        eng.prefill(s, ids[:cut])
        pairs.append((cut - 1, eng.last_logits()))
        eng.free_seq(s)
    bad += not stats(f"prefill (one chunk){tag}", pairs, MAX_KL["prefill"])
    bad += not against(f"prefill = decode{tag}", pairs, decoded)
    if tag:                                 # the cold experts staged on the GPU instead (the file's through pinned memory)
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
    for i in range(40, n):                  # mixed: prefill then decode on top
        eng.prefill(s, [ids[i]])
        pairs.append((i, eng.last_logits()))
    bad += not stats(f"prefill 40 + decode{tag}", pairs, MAX_KL["mixed"])
    eng.free_seq(s)
    return bad


def placed():
    x = eng.expert_stats()
    return x["placed"], x["placed_file_tier"]


# The file's VRAM-tier tensors, and how much of them is in the page cache (mincore over our own mapping of the file)
HOT = [(t.data_offset, int(t.n_bytes), int(t.shape[-1])) for t in GGUFReader(MODEL).tensors if t.name.endswith("_hot.weight")]
HOT_EXPERT = sum(nb for _, nb, _ in HOT) / sum(ne for _, _, ne in HOT) * 3   # bytes of one expert's gate, up and down


def mirror_rss():
    """The memory the spill mirror holds, as the kernel counts it: the one mapping as large as the tensors it mirrors."""
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
model_map = libc.mmap(None, model_size, 1, 1, model_fd, 0)    # PROT_READ, MAP_SHARED


def hot_cached():
    vec = (C.c_ubyte * ((model_size + 4095) // 4096))()
    if libc.mincore(C.c_void_p(model_map), C.c_size_t(model_size), vec):
        raise OSError(C.get_errno(), "mincore")
    v = np.frombuffer(vec, np.uint8) & 1
    return int(sum(v[a // 4096:(a + nb + 4095) // 4096].sum() for a, nb, _ in HOT)) * 4096


def move_pool(prompt, steps):
    """The pool follows a reply to prompt (token ids) for steps decode steps, then holds still."""
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
    time.sleep(1)                           # the loader finishes the copies it was given (into the mirror too)


os.posix_fadvise(model_fd, 0, 0, os.POSIX_FADV_DONTNEED)
cached0 = hot_cached()                      # pages beside tensors the load read through its mapping, which stay
fails = paths("")
prose = tok.encode("<|im_start|>user\nWrite a long story about a lighthouse keeper and a storm.<|im_end|>\n<|im_start|>assistant\n",
                   add_special_tokens=False).ids
# fewer of the file's experts may leave than the prose reply would push out, so the mirror is full
CAP = 500
eng.lib.chrysalis_tune(b"spill", CAP)
a = placed()
move_pool(prose, 400)
b, r1, out1 = placed(), mirror_rss(), eng.expert_stats()["mirrored"]
ok = b[0] - a[0] >= 1000 and out1 <= CAP and 0.95 * CAP * HOT_EXPERT <= r1 <= CAP * HOT_EXPERT + 64e6
print(f"pool after a prose reply: {b[0] - a[0]} experts placed, {out1} of the file's out of it (at most {CAP}), the mirror "
      f"holding {r1 / 1e9:.2f} GB ({CAP * HOT_EXPERT / 1e9:.2f} GB for {CAP})  {'ok' if ok else 'FAIL'}", flush=True)
fails += not ok
os.posix_fadvise(model_fd, 0, 0, os.POSIX_FADV_DONTNEED)      # what another program's memory would take from the cache
fails += paths(", pool moved by prose")
read = hot_cached() - cached0
print(f"paths with the file's experts out of the pool: {read / 1e6:.0f} MB of them read from the model file  {'ok' if read <= 0 else 'FAIL'}", flush=True)
fails += read > 0
code = "".join(open(os.path.join(ROOT, f)).read() for f in ("engine_fn/model.cpp", "chrysalis/server.py"))
code_ids = tok.encode("<|im_start|>user\nHere is a file from my project:\n```\n", add_special_tokens=False).ids + \
    tok.encode(code, add_special_tokens=False).ids[:6000] + tok.encode("\n```\n", add_special_tokens=False).ids + ids
move_pool(code_ids, 400)
c, r2, out2 = placed(), mirror_rss(), eng.expert_stats()["mirrored"]
ok = c[0] - b[0] >= 300 and c[1] - b[1] >= 50 and abs(r2 - out2 * HOT_EXPERT) <= 0.02 * out2 * HOT_EXPERT + 32e6
print(f"pool after code: {c[0] - b[0]} experts placed, {c[1] - b[1]} of them the file's back from the mirror, {out2} out of it, "
      f"the mirror holding {r2 / 1e9:.2f} GB ({out2 * HOT_EXPERT / 1e9:.2f} GB for {out2})  {'ok' if ok else 'FAIL'}", flush=True)
fails += not ok
fails += paths(", pool moved by code")
# with no RAM to spare (a floor above any machine's) the mirror takes no more, and the pool still follows a reply
eng.lib.chrysalis_tune(b"spill", 100000)
eng.lib.chrysalis_tune(b"spill_spare", 1e6)
out3 = eng.expert_stats()["mirrored"]
move_pool(prose, 200)
d, out4 = placed()[0] - c[0], eng.expert_stats()["mirrored"]
ok = d >= 300 and out4 <= out3
print(f"pool after prose with no RAM to spare: {d} experts placed, the file's out of it {out3} -> {out4}  {'ok' if ok else 'FAIL'}", flush=True)
fails += not ok
print("all ok" if not fails else f"{fails} FAILURES")
sys.exit(fails != 0)
