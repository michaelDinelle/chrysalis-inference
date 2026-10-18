"""Snapshots: restore gives bit-identical continuations, a snapshot at a prompt's end samples without a prefill, and
copy-on-write protects shared pages. Generation: a token's candidates are the largest logits, and speculative steps
(drafts, verify, commit) give what the plain decode path gives.
  python tools/test_snapshot.py [--use qwen3.8-flash-next]   (default: the 27B)"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kiln.engine import Engine, Sampling
from kiln.models import MODELS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = MODELS[sys.argv[sys.argv.index("--use") + 1] if "--use" in sys.argv else "qwen3.8-27b"]
eng = Engine(os.path.join(ROOT, spec["path"]), max_seqs=8, umax=spec["chunk"], lib=os.path.join(ROOT, spec["lib"]))   # 7 sequences
if hasattr(eng.lib, "kiln_tune"):        # Flash-Next: its prefill split stops adapting once settled; here it is fixed now
    import ctypes as C
    eng.lib.kiln_tune.restype, eng.lib.kiln_tune.argtypes = C.c_double, [C.c_char_p, C.c_double]
    eng.lib.kiln_tune(b"adapt_left", 0)
    eng.lib.kiln_tune(b"split", 1); eng.lib.kiln_tune(b"gfrac", 0.3)   # its CPU / GPU split fixed: the same numbers every run
random.seed(3)
p1 = [random.randrange(1000, 50000) for _ in range(300)]
p2 = [random.randrange(1000, 50000) for _ in range(130)]
greedy = Sampling(temperature=0.0, top_p=1.0, min_p=0.0, presence_penalty=0.0, top_k=1, n_draft=spec["draft"])
sampled = Sampling(temperature=1.0, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=spec["draft"])


def gen(sid, sp, n=60):
    out = []
    while len(out) < n:
        out += eng.generate(sid, sp, 16)
    return out[:n]


a = eng.new_seq(); eng.prefill(a, p1); snap = eng.snapshot(a)
eng.prefill(a, p2); la = eng.last_logits()
out_a = gen(a, greedy)
b = eng.new_seq(); eng.restore(b, snap); eng.prefill(b, p2); lb = eng.last_logits()
out_b = gen(b, greedy)
c = eng.new_seq(); eng.prefill(c, p1 + p2); lc = eng.last_logits()
fails = 0
ok = np.array_equal(la, lb); fails += not ok
print(f"restored logits identical            {'ok' if ok else 'FAIL'}  max diff {np.abs(la-lb).max():.3g}")
ok = out_a == out_b; fails += not ok
print(f"continuations identical (with MTP)   {'ok' if ok else 'FAIL'}")
# the same tokens in other passes round differently (more on Flash-Next, whose CPU and GPU share the cold experts by pass)
tol = 1.0 if hasattr(eng.lib, "kiln_tune") else 0.5
ok = int(la.argmax()) == int(lc.argmax()) and np.abs(la - lc).max() < tol; fails += not ok
print(f"one-shot prefill agrees              {'ok' if ok else 'FAIL'}  max diff {np.abs(la-lc).max():.3g}")
eng.truncate(c, 0); eng.prefill(c, p1 + p2); lt = eng.last_logits()
ok = np.array_equal(lc, lt); fails += not ok
print(f"truncate to 0 = fresh sequence       {'ok' if ok else 'FAIL'}  max diff {np.abs(lc-lt).max():.3g}")
# a snapshot at the end of a prompt keeps the next token's distribution: a restore there draws the same reply, greedy
# or sampled with the same seed, with nothing prefilled
eng.truncate(c, 0); eng.prefill(c, p1 + p2); end = eng.snapshot(c)
eng.begin(c, 5); out_c = gen(c, sampled)
d = eng.new_seq(); eng.restore(d, end); eng.begin(d, 5); out_d = gen(d, sampled)
eng.truncate(d, 0); eng.restore(d, end); out_g = gen(d, greedy)
eng.truncate(c, 0); eng.prefill(c, p1 + p2); out_h = gen(c, greedy)
ok = out_c == out_d and out_g == out_h; fails += not ok
print(f"restore at a prompt's end = prefill  {'ok' if ok else 'FAIL'}  sampled {out_c == out_d}, greedy {out_g == out_h}")
# the candidates a token is drawn from (and its log probs) are the 64 largest logits, unchanged
e = eng.new_seq(); eng.prefill(e, p2 + p1); full = eng.last_logits()
_, rv, ri = eng.generate(e, greedy, 16, rows=True)
ok = np.array_equal(np.sort(rv[0])[::-1], np.sort(full)[::-1][:64]) and np.array_equal(full[ri[0]], rv[0]); fails += not ok
print(f"candidates = the 64 largest logits   {'ok' if ok else 'FAIL'}")
# generate never writes past the room it is given, also when it first draws the pending token (the binding raises
# if the engine wrote more)
eng.truncate(e, 0); eng.prefill(e, p2)
try:
    ok = len(eng.generate(e, greedy, 1)) == 1 and len(eng.generate(e, greedy, 1)) == 1
except RuntimeError:
    ok = False
fails += not ok
print(f"generate stays within max_out       {'ok' if ok else 'FAIL'}")
# speculative steps (MTP drafts, the verify pass, the state commit) against the plain decode path from the same state,
# step by step (one token at a time on another sequence from a snapshot of the step's start; over many steps the two
# would drift apart by rounding): a step's candidates match what those single steps give, greedy takes their best, and
# the state the step commits gives the next token what the single steps' state gives it
eng.free_snapshot(snap); eng.free_snapshot(end)
f, g, h = eng.new_seq(), eng.new_seq(), eng.new_seq()
eng.prefill(f, p1)
pending = eng.generate(f, greedy, 16)[-1]    # the last token a call emits is the next call's pending one
rows_d, commit_d, best, n_tok, multi = [], [], True, 0, 0
# Per step, a pass of k + 1 tokens rounds differently from k + 1 single steps (2026-10-17, the same numbers every run):
# the 27B's committed state differs by a median 0.21 (max 0.81), Flash-Next's by 2.2 (max 5.5: a routing near-tie
# moves an expert somewhere in 48 layers at nearly every step). A commit one token short gives medians 1.47 and 5.6.
SPEC_MAX, SPEC_MED = (10.0, 3.5) if hasattr(eng.lib, "kiln_tune") else (3.0, 0.6)
for _ in range(12):
    start = eng.snapshot(f)
    toks, rv, ri = eng.generate(f, greedy, 16, rows=True)
    eng.truncate(g, 0); eng.restore(g, start); eng.free_snapshot(start)
    for j, t in enumerate(toks):            # row j: the logits after the pending token and the step's tokens before j
        eng.prefill(g, [pending] if j == 0 else [toks[j - 1]])
        lg = eng.last_logits()
        rows_d.append(float(np.abs(lg[ri[j]] - rv[j]).max()))
        best &= bool(lg[t] >= lg.max() - SPEC_MAX)
    end_ = eng.snapshot(f)                  # the committed state: f processed the pending token and the accepted drafts
    eng.truncate(h, 0); eng.restore(h, end_); eng.free_snapshot(end_)
    eng.prefill(g, [toks[-1]]); lg_g = eng.last_logits()
    eng.prefill(h, [toks[-1]]); lg_h = eng.last_logits()
    commit_d.append(float(np.abs(lg_g - lg_h).max()))
    pending, n_tok, multi = toks[-1], n_tok + len(toks), multi + (len(toks) > 1)
ok = (multi >= 3 and best and max(rows_d) < SPEC_MAX and max(commit_d) < SPEC_MAX and np.median(rows_d) < SPEC_MED
      and np.median(commit_d) < SPEC_MED); fails += not ok
print(f"speculative steps = decode path      {'ok' if ok else 'FAIL'}  {n_tok} tokens in 12 steps ({multi} with drafts); rows max "
      f"{max(rows_d):.3g} median {np.median(rows_d):.3g}; committed state max {max(commit_d):.3g} median {np.median(commit_d):.3g}")
print("   committed state per step: " + " ".join(f"{x:.2f}" for x in commit_d))
free0 = eng.free_tokens()
for s_ in (a, b, c, d, e, f, g, h): eng.free_seq(s_)
ok = eng.free_tokens() > free0; fails += not ok
print(f"pages returned after free            {'ok' if ok else 'FAIL'}")
print("all ok" if not fails else f"{fails} FAILURES"); sys.exit(fails != 0)
