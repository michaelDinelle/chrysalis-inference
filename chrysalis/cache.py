import hashlib
import json
import os
import threading
import time
import numpy as np


def common_prefix(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if len(diff) else n


def extends(longer, shorter):
    n = len(shorter)
    return len(longer) > n > 0 and longer[n - 1] == shorter[n - 1] and common_prefix(longer[:n], shorter) == n


class Slot:
    def __init__(self, sid):
        self.sid, self.tokens, self.used = sid, np.zeros(0, np.int32), 0.0
        self.busy = False


class Entry:
    def __init__(self, snap, tokens, pinned=False, logits=False):
        self.snap, self.tokens, self.used, self.pinned, self.logits = snap, tokens, time.time(), pinned, logits


class Disk:
    MAGIC = b"CHRYSALIS-SNAPSHOT 1\n"

    def __init__(self, directory, identity, budget=6 << 30):
        self.dir, self.identity, self.budget = directory, identity, budget
        self.lock = threading.Lock()
        self.index = {}
        os.makedirs(directory, exist_ok=True)
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            meta = self.meta(path) if name.endswith(".snap") else None
            if meta and meta.get("identity") == identity:
                self.index[path] = np.array(meta["tokens"], np.int32)
            elif name.endswith((".snap", ".tmp")):
                try:
                    os.remove(path)
                except OSError:
                    pass

    @classmethod
    def meta(cls, path):
        try:
            with open(path, "rb") as f:
                return json.loads(f.readline() == cls.MAGIC and f.readline() or b"null")
        except (OSError, ValueError):
            return None

    def path_of(self, tokens):
        return os.path.join(self.dir, hashlib.sha1(self.identity.encode() + tokens.tobytes()).hexdigest()[:24] + ".snap")

    def save(self, engine, snap, tokens):
        path = self.path_of(tokens)
        with self.lock:
            if path in self.index:
                return
            self.index[path] = None
        data = engine.export_snapshot(snap)
        meta = json.dumps({"identity": self.identity, "tokens": tokens.tolist()}).encode() + b"\n"

        def write():
            try:
                with open(path + ".tmp", "wb") as f:
                    f.write(self.MAGIC)
                    f.write(meta)
                    f.write(memoryview(data))
                os.replace(path + ".tmp", path)
                with self.lock:
                    self.index[path] = tokens.copy()
                self.trim()
            except OSError:
                with self.lock:
                    self.index.pop(path, None)
        threading.Thread(target=write, daemon=True).start()

    def best(self, prompt, longer_than):
        with self.lock:
            items = [(p, t) for p, t in self.index.items() if t is not None]
        best = None
        for path, toks in items:
            n = len(toks)
            if longer_than < n < len(prompt) and (best is None or n > len(best[1])) and common_prefix(toks, prompt) == n:
                best = (path, toks)
        return best

    def load(self, engine, path):
        try:
            with open(path, "rb") as f:
                f.readline()
                f.readline()
                os.utime(path)
                snap = engine.import_snapshot(f)
        except OSError:
            snap = None
        if snap is None:
            with self.lock:
                self.index.pop(path, None)
        return snap

    def trim(self):
        with self.lock:
            files = [(os.stat(p).st_mtime, os.stat(p).st_size, p) for p, t in self.index.items() if t is not None and os.path.exists(p)]
        files.sort()
        total = sum(f[1] for f in files)
        for _, size, path in files[:-1]:
            if total <= self.budget:
                break
            try:
                os.remove(path)
            except OSError:
                pass
            total -= size
            with self.lock:
                self.index.pop(path, None)


class Cache:
    def __init__(self, engine, n_slots, max_snapshots=40, min_snapshot_gap=512, disk=None):
        self.e = engine
        self.slots = [Slot(engine.new_seq()) for _ in range(n_slots)]
        self.entries = []
        self.max_snapshots, self.min_gap = max_snapshots, min_snapshot_gap
        self.disk = disk
        engine.reserve_snapshots(max_snapshots + 1)

    def acquire(self, prompt):
        best, kind = None, None
        reuse = 0
        free = [s for s in self.slots if not s.busy]
        for s in free:
            n = len(s.tokens)
            if 0 < n < len(prompt) and n > reuse and common_prefix(s.tokens, prompt) == n:
                best, kind, reuse = s, "live", n
        for en in self.entries:
            n = len(en.tokens)
            if (n < len(prompt) or n == len(prompt) and en.logits) and n > reuse and common_prefix(en.tokens, prompt) == n:
                best, kind, reuse = en, "snap", n
        stored = self.disk.best(prompt, reuse) if self.disk else None
        if stored:
            try:
                snap = self.disk.load(self.e, stored[0])
            except RuntimeError:
                snap = None
            if snap is not None:
                best, kind, reuse = Entry(snap, stored[1], pinned=True), "snap", len(stored[1])
                self.entries.append(best)
        if kind == "live":
            best.used, best.busy = time.time(), True
            return best, reuse
        slot = min(free, key=lambda s: s.used)
        self.evict(slot)
        if kind == "snap":
            self.e.restore(slot.sid, best.snap)
            best.used = time.time()
            slot.tokens = best.tokens.copy()
        else:
            self.e.truncate(slot.sid, 0)
            slot.tokens = np.zeros(0, np.int32)
        slot.used, slot.busy = time.time(), True
        self.trim()
        return slot, reuse

    def evict(self, slot):
        if len(slot.tokens) >= self.min_gap and not self.has(slot.tokens):
            self.entries.append(Entry(self.e.snapshot(slot.sid), slot.tokens.copy()))

    def has(self, tokens):
        return any(len(en.tokens) == len(tokens) and common_prefix(en.tokens, tokens) == len(tokens) for en in self.entries)

    def add(self, snap, tokens, pinned=False, logits=False):
        self.entries.append(Entry(snap, tokens, pinned, logits))
        self.trim()

    def checkpoint(self, slot, pinned=False, logits=False):
        if not len(slot.tokens):
            return
        for en in self.entries:
            if len(en.tokens) == len(slot.tokens) and common_prefix(en.tokens, slot.tokens) == len(slot.tokens):
                if logits and not en.logits:
                    pinned = pinned or en.pinned
                    self.entries.remove(en)
                    self.e.free_snapshot(en.snap)
                    break
                en.pinned, en.used = en.pinned or pinned, time.time()
                if pinned and self.disk:
                    self.disk.save(self.e, en.snap, en.tokens)
                return
        self.add(self.e.snapshot(slot.sid), slot.tokens.copy(), pinned, logits)
        if pinned and self.disk:
            self.disk.save(self.e, self.entries[-1].snap, self.entries[-1].tokens)

    def trim(self, need_tokens=0):
        while self.entries:
            over = len(self.entries) > self.max_snapshots
            if not over and self.e.free_tokens() >= need_tokens:
                return
            victims = [en for en in self.entries if not en.pinned] or self.entries
            states = [en.tokens for en in self.entries] + [s.tokens for s in self.slots]
            v = max(victims, key=lambda en: (sum(len(t) >= len(en.tokens) + self.min_gap and extends(t, en.tokens) for t in states) >= 2,
                                             -en.used))
            self.entries.remove(v)
            self.e.free_snapshot(v.snap)

    def stats(self):
        return {"slots": [len(s.tokens) for s in self.slots], "snapshots": len(self.entries),
                "free_kv_tokens": self.e.free_tokens()}
