import importlib.util, itertools, os, sys, types
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("cache", sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "chrysalis/cache.py"))
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
cache.time = types.SimpleNamespace(time=itertools.count(1).__next__)


class FakeEngine:
    def __init__(self):
        self.seqs, self.snaps, self.limit = [], [], 0

    def new_seq(self):
        self.seqs.append([])
        return len(self.seqs) - 1

    def reserve_snapshots(self, n):
        self.limit = n

    def snapshot(self, sid):
        free = [i for i, s in enumerate(self.snaps) if s is None]
        i = free[0] if free else len(self.snaps)
        assert i < self.limit, f"more than the {self.limit} reserved snapshot buffers in use"
        if i == len(self.snaps):
            self.snaps.append(None)
        self.snaps[i] = list(self.seqs[sid])
        return i

    def restore(self, sid, snap):
        assert self.snaps[snap] is not None, f"restore of freed snapshot {snap}"
        self.seqs[sid] = list(self.snaps[snap])

    def free_snapshot(self, snap):
        assert self.snaps[snap] is not None, f"double free of snapshot {snap}"
        self.snaps[snap] = None

    def truncate(self, sid, n):
        del self.seqs[sid][n:]

    def prefill(self, sid, tokens):
        self.seqs[sid] += list(tokens)

    def free_tokens(self):
        return 1 << 20


def serve(c, e, prompt, reply=0):
    prompt = np.array(prompt, np.int32)
    slot, cached = c.acquire(prompt)
    assert e.seqs[slot.sid] == list(prompt[:cached]), "the slot does not hold the prefix it claims"
    e.prefill(slot.sid, prompt[cached:])
    slot.tokens = prompt.copy()
    c.checkpoint(slot, logits=True)
    out = list(range(10_000, 10_000 + reply))
    e.prefill(slot.sid, out)
    slot.tokens = np.concatenate([prompt, np.array(out, np.int32)])
    slot.busy = False
    return cached


e = FakeEngine()
c = cache.Cache(e, n_slots=3, max_snapshots=3, min_snapshot_gap=8)
old = [1, 2, 3, 4, 5]
serve(c, e, old, reply=3)
for t in range(2):
    serve(c, e, [100 + t] * 6, reply=20)
assert len(c.entries) == 3 and c.entries[0].tokens.tolist() == old, "setup: the old thread's end is the oldest snapshot"
cached = serve(c, e, old, reply=3)
assert cached == len(old), f"expected the whole prompt from the snapshot, got {cached}"
assert len(c.entries) <= c.max_snapshots
print("regenerate on an old thread: ok")

for i in range(200):
    serve(c, e, [7, 7, 7] + [i % 13] * (i % 11 + 1), reply=i % 17)
    assert all(e.snaps[en.snap] is not None for en in c.entries)
print("churn: ok")

e = FakeEngine()
c = cache.Cache(e, n_slots=3, max_snapshots=6, min_snapshot_gap=8)
agent = [9] * 40
serve(c, e, agent, reply=10)
before = agent + list(range(10_000, 10_010))
subs = [[200 + k] * 20 for k in range(3)]
for step in range(6):
    for k in range(3):
        serve(c, e, subs[k], reply=5)
        subs[k] = subs[k] + list(range(10_000, 10_005)) + [300 + k] * 12
assert not any(len(s.tokens) and s.tokens.tolist()[:len(before)] == before for s in c.slots), "setup: the agent lost its slot"
cached = serve(c, e, before + [400] * 30, reply=5)
assert cached == len(before), f"the agent's step read {len(before) + 30 - cached} tokens again (cached {cached} of {len(before)})"
print("an agent waiting on its sub-agents keeps its cache: ok")
