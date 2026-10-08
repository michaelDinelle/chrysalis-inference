"""Prompt cache across requests: live engine sequences plus snapshots at message boundaries.

A request is served from whatever cached state shares the longest prefix with its prompt:
  * a live sequence whose tokens are a prefix of the prompt (the usual agent-loop case: pure append),
  * a snapshot whose tokens are a prefix of the prompt (new session sharing the tools/system prompt,
    an edited or undone turn, a session that was evicted from the GPU),
  * otherwise a fresh sequence.
Only the remaining suffix is prefilled. Snapshots hold the recurrent state in host RAM and share KV
pages copy-on-write; least recently used ones are dropped when RAM or KV pages run short.
"""
import hashlib
import os
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


class Entry:
    def __init__(self, snap, tokens, pinned=False):
        self.snap, self.tokens, self.used, self.pinned = snap, tokens, time.time(), pinned


class Cache:
    def __init__(self, engine, n_slots, max_snapshots=40, min_snapshot_gap=512, disk=None, max_disk=6):
        self.e = engine
        self.slots = [Slot(engine.new_seq()) for _ in range(n_slots)]
        self.entries = []
        self.max_snapshots, self.min_gap = max_snapshots, min_snapshot_gap
        self.disk, self.max_disk = disk, max_disk
        if disk:
            os.makedirs(disk, exist_ok=True)
            for f in sorted(os.listdir(disk)):
                if not f.endswith(".kv"):
                    continue
                base = os.path.join(disk, f[:-3])
                snap = engine.load_snapshot(base + ".kv") if os.path.exists(base + ".npy") else -1
                if snap >= 0:
                    self.entries.append(Entry(snap, np.load(base + ".npy"), pinned=True))
                else:                                   # written by an older engine build: stale
                    for ext in (".kv", ".npy"):
                        if os.path.exists(base + ext):
                            os.remove(base + ext)

    def acquire(self, prompt):
        """Pick a slot holding the longest reusable prefix of `prompt` (np.int32). Returns (slot, cached)."""
        best, kind = None, None
        reuse = 0
        for s in self.slots:
            n = len(s.tokens)
            if 0 < n < len(prompt) and n > reuse and common_prefix(s.tokens, prompt) == n:
                best, kind, reuse = s, "live", n
        for en in self.entries:
            n = len(en.tokens)
            if n < len(prompt) and n > reuse and common_prefix(en.tokens, prompt) == n:
                best, kind, reuse = en, "snap", n
        if kind == "live":
            best.used = time.time()
            return best, reuse
        slot = min(self.slots, key=lambda s: s.used)          # least recently used slot gets reassigned
        self.evict(slot)
        if kind == "snap":
            self.e.restore(slot.sid, best.snap)
            best.used = time.time()
            slot.tokens = best.tokens.copy()
        else:
            self.e.truncate(slot.sid, 0)
            slot.tokens = np.zeros(0, np.int32)
        slot.used = time.time()
        return slot, reuse

    def evict(self, slot):
        """Keep a session that is losing its slot resumable: snapshot its end state first."""
        if len(slot.tokens) >= self.min_gap and not self.has(slot.tokens):
            self.add(self.e.snapshot(slot.sid), slot.tokens.copy())

    def has(self, tokens):
        return any(len(en.tokens) == len(tokens) and common_prefix(en.tokens, tokens) == len(tokens) for en in self.entries)

    def add(self, snap, tokens, pinned=False):
        self.entries.append(Entry(snap, tokens, pinned))
        self.trim()

    def checkpoint(self, slot, pinned=False):
        """Snapshot the slot's current state (a message boundary inside a prompt). Pinned snapshots are
        prefixes shared by every session; they are also written to disk so a restart begins warm."""
        if not len(slot.tokens) or self.has(slot.tokens):
            return
        snap = self.e.snapshot(slot.sid)
        self.add(snap, slot.tokens.copy(), pinned)
        if pinned and self.disk:
            name = os.path.join(self.disk, hashlib.sha1(slot.tokens.tobytes()).hexdigest()[:16])
            if self.e.save_snapshot(snap, name + ".kv"):
                np.save(name + ".npy", slot.tokens)
            files = sorted((f for f in os.listdir(self.disk) if f.endswith(".kv")), key=lambda f: os.path.getmtime(os.path.join(self.disk, f)))
            for f in files[:-self.max_disk]:
                for ext in (".kv", ".npy"):
                    try:
                        os.remove(os.path.join(self.disk, f[:-3] + ext))
                    except OSError:
                        pass

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
