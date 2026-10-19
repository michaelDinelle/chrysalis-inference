"""Prompt cache across requests: live engine sequences plus snapshots at message boundaries.

A request is served from whatever cached state shares the longest prefix with its prompt:
  * a live sequence whose tokens are a prefix of the prompt (the usual agent-loop case: pure append),
  * a snapshot whose tokens are a prefix of the prompt (new session sharing the tools/system prompt,
    an edited or undone turn, a session that was evicted from the GPU),
  * otherwise a fresh sequence.
Only the remaining suffix is prefilled. Snapshots hold the recurrent state in host RAM and share KV
pages copy-on-write; least recently used ones are dropped when RAM or KV pages run short. The pinned
ones (prefixes every session shares) are also kept on disk (Disk), so a restart or a wake from sleep
reads them back instead of prefilling them.
"""
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


class Slot:
    def __init__(self, sid):
        self.sid, self.tokens, self.used = sid, np.zeros(0, np.int32), 0.0
        self.busy = False                                # a request is writing its reply here (parallel requests)


class Entry:
    def __init__(self, snap, tokens, pinned=False, logits=False):
        # logits: taken at the end of a prompt, it holds the next token's distribution, so it can serve that whole prompt
        self.snap, self.tokens, self.used, self.pinned, self.logits = snap, tokens, time.time(), pinned, logits


class Disk:
    """Snapshots of the prefixes every session shares (the cache's pinned ones: the tool block, the static system prompt)
    on disk. A file is a header line, a line of JSON (the model and engine build it belongs to, its tokens) and the
    engine's bytes (Engine.export_snapshot): OpenCode's 8K-token prefix reads back in 0.2 s, against 6.4 s to prefill.
    Files are written by a thread and the least recently used dropped beyond `budget` bytes. The directory is the
    model's: files there of another build or model file (identity: the model file's and the engine library's names,
    sizes and modification times) can never be read again, and go when it opens, as do writes cut short."""
    MAGIC = b"CHRYSALIS-SNAPSHOT 1\n"

    def __init__(self, directory, identity, budget=6 << 30):
        self.dir, self.identity, self.budget = directory, identity, budget
        self.lock = threading.Lock()
        self.index = {}                                  # path -> tokens, this identity's files
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
        """Keep a snapshot: its bytes are taken now (the caller holds the GPU), the file is written by a thread."""
        path = self.path_of(tokens)
        with self.lock:
            if path in self.index:
                return
            self.index[path] = None                      # being written
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
        """(path, tokens) of the stored prefix of `prompt` that is longest, if longer than `longer_than` tokens."""
        with self.lock:
            items = [(p, t) for p, t in self.index.items() if t is not None]
        best = None
        for path, toks in items:
            n = len(toks)
            if longer_than < n < len(prompt) and (best is None or n > len(best[1])) and common_prefix(toks, prompt) == n:
                best = (path, toks)
        return best

    def load(self, engine, path):
        """The file's snapshot in the engine (None if it cannot be read: it is then forgotten). Raises the engine's
        KVFull when its pool has no room."""
        try:
            with open(path, "rb") as f:
                f.readline()
                f.readline()
                data = np.fromfile(f, np.uint8)
            os.utime(path)                               # recently used
        except OSError:
            data = None
        snap = None if data is None else engine.import_snapshot(data)
        if snap is None:
            with self.lock:
                self.index.pop(path, None)
        return snap

    def trim(self):
        """Drop the least recently used files of this identity beyond the budget (the newest always stays)."""
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
        """disk: a Disk to keep the pinned snapshots in (None: memory only)."""
        self.e = engine
        self.slots = [Slot(engine.new_seq()) for _ in range(n_slots)]
        self.entries = []
        self.max_snapshots, self.min_gap = max_snapshots, min_snapshot_gap
        self.disk = disk
        engine.reserve_snapshots(max_snapshots + 1)    # trim() frees one after adding it

    def acquire(self, prompt):
        """Pick a slot holding the longest reusable prefix of `prompt` (np.int32), among those no request holds; it is
        then busy until the caller clears it. Returns (slot, cached)."""
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
            except RuntimeError:                         # KVFull: no room in the pool for it now
                snap = None
            if snap is not None:
                best, kind, reuse = Entry(snap, stored[1], pinned=True), "snap", len(stored[1])
                self.entries.append(best)
        if kind == "live":
            best.used, best.busy = time.time(), True
            return best, reuse
        slot = min(free, key=lambda s: s.used)                # least recently used slot gets reassigned
        self.evict(slot)
        if kind == "snap":
            self.e.restore(slot.sid, best.snap)
            best.used = time.time()
            slot.tokens = best.tokens.copy()
        else:
            self.e.truncate(slot.sid, 0)
            slot.tokens = np.zeros(0, np.int32)
        slot.used, slot.busy = time.time(), True
        # Only now, with the restored state in the slot: trimming first could drop the very snapshot being restored (the
        # least recently used one, e.g. an old thread's end on Regenerate), and the reply would read freed KV pages.
        self.trim()
        return slot, reuse

    def evict(self, slot):
        """Keep a session that is losing its slot resumable: snapshot its end state first (trimmed by the caller)."""
        if len(slot.tokens) >= self.min_gap and not self.has(slot.tokens):
            self.entries.append(Entry(self.e.snapshot(slot.sid), slot.tokens.copy()))

    def has(self, tokens):
        return any(len(en.tokens) == len(tokens) and common_prefix(en.tokens, tokens) == len(tokens) for en in self.entries)

    def add(self, snap, tokens, pinned=False, logits=False):
        self.entries.append(Entry(snap, tokens, pinned, logits))
        self.trim()

    def checkpoint(self, slot, pinned=False, logits=False):
        """Snapshot the slot's current state (a cache point inside a prompt, or its end: logits). Pinned snapshots
        are prefixes shared by every session, evicted only when nothing else is left."""
        if not len(slot.tokens):
            return
        for en in self.entries:
            if len(en.tokens) == len(slot.tokens) and common_prefix(en.tokens, slot.tokens) == len(slot.tokens):
                if logits and not en.logits:             # taken elsewhere (an evicted reply): no distribution in it
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
        """Drop least recently used snapshots beyond the count limit, or until `need_tokens` of KV is free."""
        while self.entries:
            over = len(self.entries) > self.max_snapshots
            if not over and self.e.free_tokens() >= need_tokens:
                return
            victims = [en for en in self.entries if not en.pinned] or self.entries
            v = min(victims, key=lambda en: en.used)
            self.entries.remove(v)
            self.e.free_snapshot(v.snap)

    def stats(self):
        return {"slots": [len(s.tokens) for s in self.slots], "snapshots": len(self.entries),
                "free_kv_tokens": self.e.free_tokens()}
