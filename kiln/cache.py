"""Prompt cache across requests: live engine sequences plus snapshots at message boundaries.

A request is served from whatever cached state shares the longest prefix with its prompt:
  * a live sequence whose tokens are a prefix of the prompt (the usual agent-loop case: pure append),
  * a snapshot whose tokens are a prefix of the prompt (new session sharing the tools/system prompt,
    an edited or undone turn, a session that was evicted from the GPU),
  * otherwise a fresh sequence.
Only the remaining suffix is prefilled. Snapshots hold the recurrent state in host RAM and share KV
pages copy-on-write; least recently used ones are dropped when RAM or KV pages run short. Nothing is
written to disk: a restart starts cold.
"""
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
    def __init__(self, snap, tokens, pinned=False, logits=False):
        # logits: taken at the end of a prompt, it holds the next token's distribution, so it can serve that whole prompt
        self.snap, self.tokens, self.used, self.pinned, self.logits = snap, tokens, time.time(), pinned, logits


class Cache:
    def __init__(self, engine, n_slots, max_snapshots=40, min_snapshot_gap=512):
        self.e = engine
        self.slots = [Slot(engine.new_seq()) for _ in range(n_slots)]
        self.entries = []
        self.max_snapshots, self.min_gap = max_snapshots, min_snapshot_gap
        engine.reserve_snapshots(max_snapshots + 1)    # trim() frees one after adding it

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
            if (n < len(prompt) or n == len(prompt) and en.logits) and n > reuse and common_prefix(en.tokens, prompt) == n:
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
                return
        self.add(self.e.snapshot(slot.sid), slot.tokens.copy(), pinned, logits)

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
