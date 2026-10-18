"""Snapshots: restore gives bit-identical continuations, a snapshot at a prompt's end samples without a prefill,
copy-on-write protects shared pages, and one written out and read back continues as the original does. Generation: a
token's candidates are the largest logits, and speculative steps (drafts, verify, commit; masked for held answers) give
what the plain decode path gives.
  python tools/test_snapshot.py [--use qwen3.8-flash-next]   (default: the 27B)"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kiln.engine import Engine, Sampling
from kiln.structured import VOCAB
from kiln.models import MODELS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = MODELS[sys.argv[sys.argv.index("--use") + 1] if "--use" in sys.argv else "qwen3.8-27b"]
eng = Engine(os.path.join(ROOT, spec["path"]), max_seqs=8, umax=spec["chunk"], lib=os.path.join(ROOT, spec["lib"]))   # 7 sequences
if hasattr(eng.lib, "kiln_tune"):        # Flash-Next: its prefill split stops adapting once settled; here it is fixed now
    import ctypes as C
    eng.lib.kiln_tune.restype, eng.lib.kiln_tune.argtypes = C.c_double, [C.c_char_p, C.c_double]
    eng.lib.kiln_tune(b"adapt_left", 0)
    eng.lib.kiln_tune(b"split", 1); eng.lib.kiln_tune(b"gfrac", 0.3)   # its CPU / GPU split fixed: the same numbers every run
    eng.lib.kiln_tune(b"experts", 0)                                     # (the expert pool too: compare_fn checks it moving)
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
# Per step, a pass of k + 1 tokens rounds differently from k + 1 single steps (2026-10-18, the same numbers every run):
# the 27B's committed state differs by a median 0.15-0.19 (max 0.24), Flash-Next's by 0.9 (max 6.5: a routing near-tie
# moves an expert somewhere in 48 layers at nearly every step). A commit one token short gave medians 1.47 and 5.6.
SPEC_MAX, SPEC_MED = (10.0, 3.5) if hasattr(eng.lib, "kiln_tune") else (3.0, 0.6)


def spec_steps(name, prompt, sp, min_len):
    """Steps of f from the prompt against single decode steps on g and h; ok when at least three steps emit min_len
    tokens or more (drafts of that kind were accepted)."""
    global fails
    eng.truncate(f, 0); eng.prefill(f, prompt)
    if sp.lookup:
        eng.set_lookup(f, prompt)
    pending = eng.generate(f, sp, 16)[-1]    # the last token a call emits is the next call's pending one
    rows_d, commit_d, best, n_tok, multi = [], [], True, 0, 0
    for _ in range(12):
        start = eng.snapshot(f)
        toks, rv, ri = eng.generate(f, sp, 16, rows=True)
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
        pending, n_tok, multi = toks[-1], n_tok + len(toks), multi + (len(toks) >= min_len)
    ok = (multi >= 3 and best and max(rows_d) < SPEC_MAX and max(commit_d) < SPEC_MAX and np.median(rows_d) < SPEC_MED
          and np.median(commit_d) < SPEC_MED); fails += not ok
    print(f"{name:36s} {'ok' if ok else 'FAIL'}  {n_tok} tokens in 12 steps ({multi} of {min_len}+); rows max "
          f"{max(rows_d):.3g} median {np.median(rows_d):.3g}; committed state max {max(commit_d):.3g} median {np.median(commit_d):.3g}")
    print("   committed state per step: " + " ".join(f"{x:.2f}" for x in commit_d))


# speculative steps (drafts, the verify pass, the state commit) against the plain decode path from the same state, step
# by step (one token at a time on another sequence from a snapshot of the step's start; over many steps the two would
# drift apart by rounding): a step's candidates match what those single steps give, greedy takes their best, and the
# state the step commits gives the next token what the single steps' state gives it
spec_steps("speculative steps = decode path", p1 + p1[:60], greedy, 2)   # it goes on copying: drafts that land
# prompt-lookup drafts (copied from earlier in the conversation: up to 7, Flash-Next 3) through the same step: on a
# prompt that repeats itself the model goes on copying, in steps longer than the MTP head's drafts can make
rep = [random.randrange(1000, 50000) for _ in range(150)]
spec_steps("prompt-lookup steps = decode path", rep + rep + rep[:40],
           Sampling(temperature=0.0, top_p=1.0, min_p=0.0, presence_penalty=0.0, top_k=1, n_draft=spec["draft"], lookup=7),
           spec["draft"] + 2)


# A held answer's steps (kiln/server.py: response_format, strict tools) verify drafts with each position's logits masked to
# what its grammar allows there, the tokens it forces first among them. Against single decode steps from the step's start
# with the same masks applied here: every token is allowed and greedy's best of those, a forced draft is taken, and steps
# carry several tokens (on a prompt that repeats itself, so drafts land where the mask lets them).
def held_steps(name, prompt, sp):
    global fails
    vocab = np.arange(VOCAB)
    allowed = vocab[vocab % 3 != 0]                   # a third of the vocabulary refused
    allow = np.zeros(VOCAB // 32, np.uint32)
    np.bitwise_or.at(allow, allowed >> 5, (1 << (allowed & 31)).astype(np.uint32))
    eng.truncate(f, 0); eng.prefill(f, prompt)
    if sp.lookup:
        eng.set_lookup(f, prompt)
    pending = eng.draw(f, *eng.masked_top(allow), sp)
    eng.set_pending(f, pending)
    ok_all, best, taken, n_tok, multi = bool(allow[pending >> 5] >> (pending & 31) & 1), True, True, 1, 0
    for step in range(12):
        start = eng.snapshot(f)
        fixed = [int(allowed[(step * 7919) % len(allowed)])] if step % 3 == 2 else []
        drafts = eng.draft(f, sp, fixed, 7)
        rows = [allow] * (len(drafts) + 1)
        if fixed:
            rows[0] = np.zeros_like(allow)
            rows[0][fixed[0] >> 5] = np.uint32(1 << (fixed[0] & 31))
        k = next((i for i, t in enumerate(drafts) if not rows[i][t >> 5] >> (t & 31) & 1), len(drafts))
        toks = eng.generate_masked(f, sp, drafts[:k], np.stack(rows[:k + 1]))
        eng.truncate(g, 0); eng.restore(g, start); eng.free_snapshot(start)
        for j, t in enumerate(toks):
            eng.prefill(g, [pending] if j == 0 else [toks[j - 1]])
            lg = np.where(np.unpackbits(rows[j].view(np.uint8), bitorder="little")[:VOCAB].astype(bool), eng.last_logits(), -np.inf)
            ok_all &= bool(np.isfinite(lg[t]))
            best &= bool(lg[t] >= lg.max() - SPEC_MAX)
        taken &= not fixed or toks[0] == fixed[0]
        pending, n_tok, multi = toks[-1], n_tok + len(toks), multi + (len(toks) >= 2)
    ok = ok_all and best and taken and multi >= 3; fails += not ok
    print(f"{name:36s} {'ok' if ok else 'FAIL'}  {n_tok} tokens in 12 steps ({multi} of 2+); all allowed {ok_all}, "
          f"best allowed {best}, forced drafts taken {taken}")


if eng.masked:
    held_steps("held steps = masked decode path", rep + rep + rep[:40], greedy)
else:
    fails += 1
    print(f"{'held steps = masked decode path':36s} FAIL  this engine has no masked steps (kiln_generate_masked)")
free0 = eng.free_tokens()
for s_ in (a, b, c, d, e, f, g, h): eng.free_seq(s_)
ok = eng.free_tokens() > free0; fails += not ok
print(f"pages returned after free            {'ok' if ok else 'FAIL'}")
# A new sequence drafts from its own conversation only: on the slot just freed, the same prompt and seed redraw the same
# sampled reply with prompt lookup on (were the last occupant's tokens still its history, the reply would copy drafts from
# them once it repeated 16, and other drafts take other random draws: measured on Flash-Next, apart at token 18)
look = Sampling(temperature=1.0, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=spec["draft"], lookup=7)


def redraw():
    s_ = eng.new_seq(); eng.prefill(s_, p2); eng.begin(s_, 9)
    out = gen(s_, look, 300)
    eng.free_seq(s_)
    return out


r1, r2 = redraw(), redraw()
ok = r1 == r2; fails += not ok
print(f"a seed redraws its reply in a new seq {'ok' if ok else 'FAIL'}  first difference at "
      f"{next((i for i in range(len(r1)) if r1[i] != r2[i]), None)}")
# Steps of several sequences in one pass (parallel requests) draw what each draws alone with the same seed and as many
# drafts: two sequences (3 drafts each, all 8 rows) and three (1 draft each), sampled.
if getattr(eng, "batched", False):
    def alone(prompt, seed, sp, steps):
        s_ = eng.new_seq(); eng.prefill(s_, prompt); eng.begin(s_, seed)
        got = [t for _ in range(steps) for t in eng.generate(s_, sp, 16)]
        eng.free_seq(s_)
        return got

    def together(prompts, seeds, sp, steps):
        ss = [eng.new_seq() for _ in prompts]
        for s_, pr, sd in zip(ss, prompts, seeds):
            eng.prefill(s_, pr); eng.begin(s_, sd)
        got = [[] for _ in ss]
        for _ in range(steps):
            for g_, t in zip(got, eng.generate_batch(ss, [sp] * len(ss))):
                g_ += t
        for s_ in ss: eng.free_seq(s_)
        return got
    p3 = [random.randrange(1000, 50000) for _ in range(77)]
    three = Sampling(temperature=1.0, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=spec["draft"])
    one = Sampling(temperature=1.0, top_p=0.95, min_p=0.0, presence_penalty=0.0, top_k=20, n_draft=1)
    two = together([p1, p2], [21, 22], three, 15)
    many = together([p1, p2, p3], [23, 24, 25], one, 15)
    ok = (two == [alone(p1, 21, three, 15), alone(p2, 22, three, 15)] and
          many == [alone(p1, 23, one, 15), alone(p2, 24, one, 15), alone(p3, 25, one, 15)] and min(map(len, two)) > 15)
    fails += not ok
    print(f"sequences stepped together = alone   {'ok' if ok else 'FAIL'}  two: {[len(t) for t in two]} tokens, three: {[len(t) for t in many]}")
# A snapshot written out and read back (kiln/cache.py keeps the prefixes sessions share on disk): what follows the
# imported one is what follows the original, bit for bit, in pages of its own; bytes cut short or of another layout are
# refused.
if eng.persistent:
    x, y = eng.new_seq(), eng.new_seq()
    eng.prefill(x, p2 + p1)
    orig = eng.snapshot(x)
    data = eng.export_snapshot(orig)
    before = eng.free_tokens()
    back = eng.import_snapshot(data)
    taken = before - eng.free_tokens()
    eng.truncate(x, 0); eng.restore(x, orig); eng.prefill(x, p2); lo = eng.last_logits()
    eng.restore(y, back); eng.prefill(y, p2); lb = eng.last_logits()
    bad = data.copy(); bad[3] ^= 1
    refused = eng.import_snapshot(bad) is None and eng.import_snapshot(data[:-1]) is None
    ok = back is not None and np.array_equal(lo, lb) and taken >= len(p2 + p1) and refused; fails += not ok
    print(f"snapshot written out and read back   {'ok' if ok else 'FAIL'}  {len(data) / 1e6:.0f} MB, {taken} tokens of pages, "
          f"max diff {np.abs(lo - lb).max():.3g}, other bytes refused {refused}")
    for s_ in (x, y): eng.free_seq(s_)
    eng.free_snapshot(orig); eng.free_snapshot(back)
print("all ok" if not fails else f"{fails} FAILURES"); sys.exit(fails != 0)
