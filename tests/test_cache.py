import importlib.util, itertools, json, os, sys, tempfile, threading, types, urllib.error, urllib.request
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


def serve_turn(c, e, prompt, turn, reply=0):
    prompt = np.array(prompt, np.int32)
    slot, cached = c.acquire(prompt)
    e.prefill(slot.sid, prompt[cached:turn])
    slot.tokens = prompt[:turn].copy()
    c.checkpoint(slot)
    e.prefill(slot.sid, prompt[turn:])
    slot.tokens = prompt.copy()
    c.checkpoint(slot, logits=True)
    out = list(range(10_000, 10_000 + reply))
    e.prefill(slot.sid, out)
    slot.tokens = np.concatenate([prompt, np.array(out, np.int32)])
    slot.busy = False
    return cached


e = FakeEngine()
c = cache.Cache(e, n_slots=3, max_snapshots=4, min_snapshot_gap=8)
reply_prompt = [1, 2, 3, 4]
for k in range(3):
    serve_turn(c, e, [600 + k] * 30 + reply_prompt, 30, reply=10)
serve_turn(c, e, [5] * 50 + reply_prompt, 50, reply=30)
cached = serve(c, e, [5] * 50 + [1, 2] + [77] * 20, reply=5)
assert cached >= 50, f"the follow-up read the conversation again (cached {cached} of 50)"
print("a full cache keeps the newest conversation up to its reply for the follow-up: ok")


class SavingEngine(FakeEngine):
    def export_snapshot(self, snap):
        return np.array(self.snaps[snap], np.int32).tobytes()


def written():
    for t in [t for t in threading.enumerate() if t is not threading.current_thread() and t.daemon]:
        t.join()


with tempfile.TemporaryDirectory() as folder:
    e = SavingEngine()
    disk = cache.Disk(os.path.join(folder, "model"), "identity")
    c = cache.Cache(e, n_slots=2, max_snapshots=4, min_snapshot_gap=8, disk=disk)
    tools, secret = [3] * 20, [3] * 20 + [77] * 12
    slot, _ = c.acquire(np.array(secret, np.int32))
    e.prefill(slot.sid, secret)
    slot.tokens = np.array(secret, np.int32)
    c.checkpoint(slot, pinned=True, kept=False)
    slot.busy = False
    written()
    assert os.listdir(disk.dir) == [], f"a request kept off the disk wrote {os.listdir(disk.dir)}"
    slot, _ = c.acquire(np.array(tools + [5], np.int32))
    e.truncate(slot.sid, 0)
    e.prefill(slot.sid, tools)
    slot.tokens = np.array(tools, np.int32)
    c.checkpoint(slot, pinned=True)
    slot.busy = False
    written()
    saved = [cache.Disk.meta(os.path.join(disk.dir, f)) for f in os.listdir(disk.dir)]
    assert [m["tokens"] for m in saved] == [tools], f"the next ordinary request saved {saved}"
print("a request kept off the disk leaves no snapshot there, and the next ordinary one is saved: ok")

sys.path.insert(0, ROOT)
from chrysalis import server


class Asked:
    def __init__(self):
        self.kept, self.metrics, self.stored = [], types.SimpleNamespace(add=lambda *a, **k: None), {}
        self.active = "qwen3.8-27b"

    def awake(self):
        pass

    def chat(self, body, kept=True):
        self.kept.append(kept)
        raise server.RequestError("seen")
        yield


asked, ready = Asked(), threading.Event()
ready.set()
httpd = server.HTTPServer(("127.0.0.1", 0), server.make_handler({"srv": asked}, ready))
threading.Thread(target=httpd.serve_forever, daemon=True).start()
bodies = {"/v1/chat/completions": {"messages": [{"role": "user", "content": "hi"}]},
          "/v1/messages": {"model": "qwen3.8-27b", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
          "/v1/responses": {"input": "hi"}}
for path, body in bodies.items():
    for unkept in (False, True):
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", **({"X-Chrysalis-Unkept": "1"} if unkept else {})})
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            assert e.code == 400, f"{path} answered {e.code}"
httpd.shutdown()
assert asked.kept == [True, False] * 3, f"the chat, Messages and Responses APIs passed on {asked.kept}"
print("each API keeps a request marked X-Chrysalis-Unkept off the disk: ok")
