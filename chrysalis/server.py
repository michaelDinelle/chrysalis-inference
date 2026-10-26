import functools
import hashlib
import ipaddress
import itertools
import json
import math
import os
import random
import re
import socket
import socketserver
import sys
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import apis, link, vision
from .cache import Cache, Disk
from .chat import Clip, Content, OutputParser, Renderer, StopText, Tokens, message_key, text_of
from .documents import DocumentError, Documents
from .structured import Grammars
from .warm import Warmer
from .engine import IMG_BASE, IMG_ROWS, MAX_ORDER, STAGES, Engine, KVFull, Sampling, load_async, load_progress
from .metrics import Metrics
from .models import MODELS
from .sleep import SleepGuard, vram_used, wait_vram_freed

MODEL = "qwen3.8-27b"
_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
present = lambda name: __import__("os").path.exists(__import__("os").path.join(_ROOT, MODELS[name]["path"]))
MAX_CONTEXT = 262144
MAX_OUTPUT = 131072
DEFAULTS = {True: dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),
            False: dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5)}
SIDE_PROMPTS = ("You are a title generator", "Summarize what was done in this conversation")
COMPACT_PROMPT = "You are a context summarization agent"
TRACE = bool(os.environ.get("CHRYSALIS_TRACE"))
SNAPSHOT_DIR = os.environ.get("CHRYSALIS_SNAPSHOTS") or os.path.join(_ROOT, "data/cache")
EFFORTS = {"xhigh": "xhigh", "high": "xhigh", "medium": "medium", "low": "low", "minimal": "low"}


class RequestError(ValueError):
    code, status = "invalid_request", 400


class ContextError(RequestError):
    code = "context_length_exceeded"


class Asleep(RequestError):
    code, status = "unavailable", 503


INT32 = 2**31 - 1
SAMPLER_LETTERS = {"e": "penalties", "d": "dry", "s": "top_n_sigma", "k": "top_k", "y": "typ_p", "p": "top_p", "m": "min_p", "x": "xtc",
                   "t": "temperature", "i": "infill", "a": "adaptive_p"}
SAMPLER_NAMES = {**{n: n for n in SAMPLER_LETTERS.values()}, "nucleus": "top_p", "temp": "temperature", "typ": "typ_p",
                 **{alias: n for n in SAMPLER_LETTERS.values() if "_" in n for alias in (n.replace("_", "-"), n.replace("_", ""))}}
DRY_BREAKERS = ("\n", ":", "\"", "*")
EOG = ("<|im_end|>", "<|endoftext|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>")
TOKEN_KEY = re.compile(r"\s*[+-]?\d+")


def number(body, key, default, low=-math.inf, high=math.inf, clamp=False, whole=False):
    v = body.get(key)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or whole and v != int(v):
        raise RequestError(f"{key} must be a {'whole ' if whole else ''}number")
    v = int(v) if whole else float(v)
    if clamp:
        return max(low, min(high, v))
    if v < low or v > high:
        raise RequestError(f"{key} must be at least {low}" if high in (INT32, math.inf) else f"{key} must be between {low} and {high}")
    return v


def samplers_of(value):
    if value is None:
        return list(STAGES)
    if isinstance(value, str):
        bad = next((c for c in value if c not in SAMPLER_LETTERS), None)
        if bad is not None:
            raise RequestError(f"samplers: {bad!r} is not a sampler letter; they are {''.join(SAMPLER_LETTERS)}")
        names = [SAMPLER_LETTERS[c] for c in value]
    elif isinstance(value, list) and all(isinstance(x, str) for x in value):
        bad = next((x for x in value if x.lower() not in SAMPLER_NAMES), None)
        if bad is not None:
            raise RequestError(f"samplers: {bad!r} is not a sampler; they are {', '.join(STAGES)}")
        names = [SAMPLER_NAMES[x.lower()] for x in value]
    else:
        raise RequestError("samplers must be a list of sampler names or a string of sampler letters")
    missing = next((n for n in names if n not in STAGES), None)
    if missing:
        raise RequestError(f"samplers: Chrysalis has no {missing} sampler")
    if len(names) > MAX_ORDER:
        raise RequestError(f"samplers lists at most {MAX_ORDER} samplers")
    return names


def logit_bias_of(value, tok):
    if value is None:
        return []
    vocab = tok.full.get_vocab_size()
    if isinstance(value, dict):
        pairs = [(int(k) if TOKEN_KEY.fullmatch(k) else k, v) for k, v in value.items()]
    elif isinstance(value, list) and all(isinstance(x, list) and len(x) == 2 for x in value):
        pairs = [tuple(x) for x in value]
    else:
        raise RequestError("logit_bias must be an object of token: bias or a list of [token, bias] pairs")
    out = []
    for key, bias in pairs:
        if bias is not False and (isinstance(bias, bool) or not isinstance(bias, (int, float)) or not math.isfinite(bias)):
            raise RequestError(f"logit_bias: the bias for {key!r} must be a number, or false to ban the token")
        if isinstance(key, str):
            ids = tok.encode_plain(key)
        elif isinstance(key, int) and not isinstance(key, bool) and 0 <= key < vocab:
            ids = [key]
        else:
            raise RequestError(f"logit_bias: {key!r} is not a token id (0 to {vocab - 1}) or a string")
        out += [(t, -math.inf if bias is False else float(bias)) for t in ids]
    return out


def breakers_of(value):
    if value is None:
        return DRY_BREAKERS
    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
        raise RequestError("dry_sequence_breakers must be a non-empty list of strings")
    return tuple(value)


def sampling_of(body, defaults, tok, ban):
    d = defaults
    dry_base = number(body, "dry_base", 1.75)
    if body.get("ignore_eos") is not None and not isinstance(body.get("ignore_eos"), bool):
        raise RequestError("ignore_eos must be true or false")
    sp = Sampling(order=samplers_of(body.get("samplers")),
                  temperature=number(body, "temperature", d["temperature"], 0.0, math.inf, clamp=True),
                  dynatemp_range=number(body, "dynatemp_range", 0.0), dynatemp_exponent=number(body, "dynatemp_exponent", 1.0),
                  top_k=number(body, "top_k", d["top_k"], 0, INT32, clamp=True, whole=True),
                  top_p=number(body, "top_p", d["top_p"], 0.0, 1.0, clamp=True), min_p=number(body, "min_p", d["min_p"], 0.0, 1.0, clamp=True),
                  typical_p=number(body, "typical_p", 1.0),
                  xtc_probability=number(body, "xtc_probability", 0.0, 0.0, 1.0, clamp=True),
                  xtc_threshold=number(body, "xtc_threshold", 0.1, 0.0, 1.0, clamp=True), top_n_sigma=number(body, "top_n_sigma", -1.0),
                  repeat_last_n=number(body, "repeat_last_n", 64, -1, INT32, whole=True),
                  repeat_penalty=number(body, "repeat_penalty", 1.0),
                  frequency_penalty=number(body, "frequency_penalty", 0.0),
                  presence_penalty=number(body, "presence_penalty", d["presence_penalty"]),
                  dry_multiplier=number(body, "dry_multiplier", 0.0), dry_base=dry_base if dry_base >= 1 else 1.75,
                  dry_allowed_length=number(body, "dry_allowed_length", 2, 0, INT32, whole=True),
                  dry_penalty_last_n=number(body, "dry_penalty_last_n", -1, -1, INT32, whole=True),
                  mirostat=number(body, "mirostat", 0, 0, 2, clamp=True, whole=True), mirostat_tau=number(body, "mirostat_tau", 5.0),
                  mirostat_eta=number(body, "mirostat_eta", 0.1), min_keep=number(body, "min_keep", 0, 0, INT32, whole=True), ban=ban)
    if sp.repeat_penalty <= 0:
        raise RequestError("repeat_penalty must be greater than 0")
    bias = logit_bias_of(body.get("logit_bias"), tok)
    if body.get("ignore_eos"):
        bias += [(t, -math.inf) for t in map(tok.full.token_to_id, EOG) if t is not None]
    return (sp, bias, breakers_of(body.get("dry_sequence_breakers")), number(body, "n_probs", 0, whole=True),
            number(body, "n_predict", None, -1, INT32, whole=True))


def first_text(content):
    if isinstance(content, list):
        content = next((c.get("text") for c in content if isinstance(c, dict) and c.get("type") == "text"), "")
    return (content or "")[:200]


def merge_system(messages):
    sys_parts = [text_of(m.get("content")) for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    return ([{"role": "system", "content": "\n\n".join(p for p in sys_parts if p)}] if sys_parts else []) + rest


class LRU(OrderedDict):

    def __init__(self, limit):
        super().__init__()
        self.limit, self.lock = limit, threading.Lock()

    def put(self, k, v):
        with self.lock:
            self[k] = v
            self.move_to_end(k)
            while len(self) > self.limit:
                self.popitem(last=False)

    def get(self, k, default=None):
        with self.lock:
            if k not in self:
                return default
            self.move_to_end(k)
            return self[k]

    def values(self):
        with self.lock:
            return list(super().values())


DATE = re.compile(r"(Today's date: )([^\n]*)")


class Pictures:
    SLOTS = (2**31 - IMG_BASE) // IMG_ROWS

    def __init__(self, engine, width, budget, cap=vision.MAX_PIXELS):
        self.e, self.width, self.budget, self.cap = engine, width, budget, cap
        self.lock = threading.Lock()
        self.by_digest, self.grid, self.limit = {}, {}, {}
        self.pair = {}
        self.videos = OrderedDict()
        self.encoded = OrderedDict()

    def slot(self, key, grid):
        slot = self.by_digest.get(key)
        if slot is None:
            if len(self.grid) >= self.SLOTS:
                raise vision.ImageError("this server has seen too many pictures since it started; restart it")
            slot = len(self.grid)
            self.by_digest[key], self.grid[slot] = slot, grid
        return slot

    def ids(self, slot):
        base = IMG_BASE + slot * IMG_ROWS
        gh, gw = self.grid[slot]
        return list(range(base, base + gh * gw // 4))

    def tokens(self, item, seen, limit=None):
        if isinstance(item, Clip):
            return self.video_tokens(item, seen)
        data = vision.image_bytes(item.url)
        limit = vision.pixel_limit(item.max_pixels, limit, cap=self.cap)
        key = (vision.digest(data), limit)
        with self.lock:
            known = self.by_digest.get(key)
        grid = self.grid[known] if known is not None else vision.grid(data, limit)
        with self.lock:
            slot = self.slot(key, grid)
            self.limit[slot] = limit
        seen[slot] = data
        return self.ids(slot)

    def video_tokens(self, clip, seen):
        data = vision.image_bytes(clip.url, "videos")
        rate = float(clip.fps or vision.VIDEO_FPS)
        if not 0 < rate <= 30:
            raise vision.VideoError("a video's fps must be over 0 and at most 30")
        key = (vision.digest(data), rate)
        with self.lock:
            video = self.videos.get(key)
        if video is None:
            video = vision.Video(data, rate)
        with self.lock:
            self.videos[key] = video
            self.videos.move_to_end(key)
            while len(self.videos) > 4:
                self.videos.popitem(last=False)
            out = []
            for i, t in enumerate(video.times):
                slot = self.slot(key + (i,), video.grid)
                self.pair[slot] = i
                seen[slot] = video
                out.append((t, self.ids(slot)))
        return out

    def positions(self, prompt):
        img = prompt >= IMG_BASE
        if not img.any():
            return None
        edges = np.flatnonzero(np.diff(np.concatenate([[0], img.view(np.int8), [0]])))
        segs, prev = [], 0
        for a, b in zip(edges[::2], edges[1::2]):
            if a > prev:
                segs.append(("text", int(a - prev)))
            gh, gw = self.grid[int(prompt[a] - IMG_BASE) // IMG_ROWS]
            segs.append(("image", gh // 2, gw // 2))
            prev = b
        if prev < len(prompt):
            segs.append(("text", int(len(prompt) - prev)))
        return vision.positions(segs)

    def runs(self, prompt):
        img = prompt >= IMG_BASE
        edges = np.flatnonzero(np.diff(np.concatenate([[0], img.view(np.int8), [0]])))
        return [(int(a), int(b), int(prompt[a] - IMG_BASE) // IMG_ROWS) for a, b in zip(edges[::2], edges[1::2])]

    def ensure(self, slots, seen):
        frames, n = {}, 0
        for slot in slots:
            if slot in self.encoded:
                self.encoded.move_to_end(slot)
                continue
            src = seen[slot]
            if isinstance(src, vision.Video):
                if id(src) not in frames:
                    frames[id(src)] = src.frames()
                i = self.pair[slot]
                ok = self.e.video_encode(slot, vision.pair_patches(frames[id(src)][2 * i:2 * i + 2])[0], self.grid[slot])
            else:
                patches, grid = vision.patches(src, self.limit[slot])
                if grid != self.grid[slot]:
                    raise vision.ImageError("the picture's size reads differently from its header and its pixels")
                ok = self.e.image_encode(slot, patches, grid)
            if not ok:
                raise vision.ImageError("the picture is too large for the GPU memory Chrysalis can lend the vision tower")
            gh, gw = self.grid[slot]
            self.encoded[slot] = gh * gw // 4 * self.width * 2
            n += 1
        keep = set(slots)
        while sum(self.encoded.values()) > self.budget:
            old = next((s for s in self.encoded if s not in keep), None)
            if old is None:
                break
            del self.encoded[old]
            self.e.image_free(old)
        return n


class Turnstile:
    SIDE_IDLE = 0.3

    def __init__(self):
        self.cv = threading.Condition()
        self.busy, self.waiting = False, 0
        self.main_seen = 0.0
        self.ticket = self.serving = 0

    def acquire(self, main, arrived=None):
        with self.cv:
            if main:
                turn, self.ticket = self.ticket, self.ticket + 1
                self.waiting += 1
                self.main_seen = time.time()
                self.cv.wait_for(lambda: not self.busy and self.serving == turn)
                self.serving += 1
                self.waiting -= 1
            else:
                self.side_turn(arrived or time.time())
            self.busy = True

    def wait_side(self, arrived):
        with self.cv:
            self.side_turn(arrived)

    def side_turn(self, arrived):
        while True:
            idle = time.time() - max(self.main_seen, arrived)
            if not self.busy and not self.waiting and idle >= self.SIDE_IDLE:
                return
            self.cv.wait(None if self.busy or self.waiting else self.SIDE_IDLE - idle)

    def release(self, main=True):
        with self.cv:
            self.busy = False
            if main:
                self.main_seen = time.time()
            self.cv.notify_all()


class Lanes:

    def __init__(self, n):
        self.cv = threading.Condition()
        self.n, self.used, self.exclusive = n, 0, False

    def enter(self):
        with self.cv:
            self.cv.wait_for(lambda: self.used < self.n and not self.exclusive)
            self.used += 1

    def try_enter(self):
        with self.cv:
            if self.used < self.n and not self.exclusive:
                self.used += 1
                return True
            return False

    def changed(self, timeout):
        with self.cv:
            self.cv.wait(timeout)

    def leave(self):
        with self.cv:
            self.used -= 1
            self.cv.notify_all()

    def hold(self, mine=False):
        with self.cv:
            if mine:
                self.used -= 1
                self.cv.notify_all()
            self.cv.wait_for(lambda: not self.exclusive)
            self.exclusive = True
            self.cv.wait_for(lambda: self.used == 0)
            self.used += mine

    def release(self):
        with self.cv:
            self.exclusive = False
            self.cv.notify_all()


def admit(lanes, gpu, side, arrived):
    if not side:
        return lanes.enter()
    while True:
        gpu.wait_side(arrived)
        if lanes.try_enter():
            return
        lanes.changed(0.5)


class Held:

    def __init__(self, gpu, main, arrived=None):
        self.gpu, self.main, self.arrived, self.on = gpu, main, arrived, False
        self.slot = None

    def take(self):
        if not self.on:
            self.gpu.acquire(self.main, self.arrived)
            self.on = True

    def drop(self):
        if self.on:
            self.gpu.release(self.main)
            self.on = False


class Batcher:
    GATHER = 0.002

    def __init__(self, gpu):
        self.gpu, self.cv = gpu, threading.Condition()
        self.writing = 0
        self.queue, self.done, self.leading = {}, {}, False
        self.shared = 0

    def begin(self):
        with self.cv:
            self.writing += 1

    def end(self):
        with self.cv:
            self.writing -= 1
            self.cv.notify_all()

    def step(self, engine, sid, sp, rows=False, most=16):
        with self.cv:
            self.queue[sid] = (sp, rows, most)
            self.cv.notify_all()
            while sid not in self.done:
                if self.leading:
                    self.cv.wait()
                    continue
                self.leading = True
                deadline = time.time() + self.GATHER
                while len(self.queue) < self.writing and (left := deadline - time.time()) > 0:
                    self.cv.wait(left)
                batch, self.queue = self.queue, {}
                self.cv.release()
                try:
                    results = self.run(engine, batch)
                except BaseException as e:
                    results = dict.fromkeys(batch, e)
                finally:
                    self.cv.acquire()
                    self.leading = False
                self.done.update(results)
                self.cv.notify_all()
            got = self.done.pop(sid)
        if isinstance(got, BaseException):
            raise got
        return got

    def run(self, engine, batch):
        alone = [sid for sid, (sp, rows, most) in batch.items() if most < 16 or len(batch) == 1 or not engine.batched]
        together = [sid for sid in batch if sid not in alone]
        if len(together) == 1:
            alone += together
            together = []
        out = {}
        self.gpu.acquire(True)
        try:
            if together:
                self.shared += 1
                rows = any(batch[sid][1] for sid in together)
                try:
                    got = engine.generate_batch(together, [batch[sid][0] for sid in together], rows=rows)
                    out.update((sid, g if batch[sid][1] or not rows else g[0]) for sid, g in zip(together, got))
                except KVFull as e:
                    out.update(dict.fromkeys(together, e))
            for sid in alone:
                sp, rows, most = batch[sid]
                try:
                    out[sid] = engine.generate(sid, sp, most, rows=True) if rows else engine.generate(sid, sp, most)
                except KVFull as e:
                    out[sid] = e
        finally:
            self.gpu.release(True)
        return out


class Sticky:

    def __init__(self):
        self.convs = LRU(512)

    def apply(self, messages):
        users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not users:
            return messages
        sys_text = text_of(messages[0].get("content")) if messages[0].get("role") == "system" else ""
        cid = hashlib.sha1((DATE.sub("", sys_text) + "\0" + text_of(messages[users[0]].get("content"))).encode()).hexdigest()
        conv = self.convs.get(cid)
        if conv is None or time.time() - conv["seen"] > 36 * 3600:
            conv = {"date": None, "users": {}}
        conv["seen"] = time.time()
        self.convs.put(cid, conv)
        out = [dict(m) for m in messages]
        m = DATE.search(sys_text)
        if m:
            if conv["date"] is None:
                conv["date"] = m.group(2)
            elif conv["date"] != m.group(2):
                out[0]["content"] = DATE.sub(lambda mm: mm.group(1) + conv["date"], sys_text, count=1)
        for i in users:
            cur, old = text_of(out[i].get("content")), conv["users"].get(i)
            if old and old != cur and old.startswith(cur) and i != users[-1]:
                out[i]["content"] = old
            elif not (old and old.startswith(cur)):
                conv["users"][i] = cur
        return out


class Server:
    def __init__(self, model_path, tokenizer_path, slots=3, n_draft=None, umax=None, log=sys.stderr, engine=None, warmup=True,
                 model=MODEL, lookup=None, max_pixels=None):
        self.max_pixels = vision.pixel_limit(max_pixels)
        if engine is None:
            engine = load_async(model_path, max_seqs=slots, umax=umax or MODELS.get(model, MODELS[MODEL])["chunk"], warmup=False,
                                lib=__import__("os").path.join(_ROOT, MODELS.get(model, MODELS[MODEL])["lib"]))
        self.active, self.slots, self.umax = model, slots, umax
        self.loading = None
        self.state_lock = threading.Lock()
        self.wake_lock, self.asleep, self.suspended = threading.Lock(), False, False
        self.tok = Tokens(tokenizer_path)
        self.replay = LRU(8192)
        self.renderer = Renderer(self.tok, self.replay.get)
        self.im_start = self.tok.full.token_to_id("<|im_start|>")
        self.sticky = Sticky()
        self.sessions = LRU(64)
        self.stored = LRU(256)
        self.grammars = None
        self.documents = Documents()
        self.gpu = Turnstile()
        self.lanes = Lanes(slots)
        self.batcher = Batcher(self.gpu)
        self.replying = set()
        self.pausing = threading.Event()
        self.log = log
        self.link_dev = link.device()
        spec = MODELS.get(model, MODELS[MODEL])
        self.n_draft = spec["draft"] if n_draft is None else n_draft
        self.lookup = spec.get("lookup", 0) if lookup is None else lookup
        self.recent = LRU(64)
        self.metrics = Metrics()
        self.warmer, self.warming = Warmer(log, lambda: self.lanes.used > 0), False
        self.engine = engine.result()
        self.capacity = min(MAX_CONTEXT, self.engine.capacity)
        self.cache = Cache(self.engine, slots, max_snapshots=MODELS.get(model, MODELS[MODEL])["snapshots"],
                           disk=self.snapshot_disk(model, model_path))
        self.pictures = self.open_vision(model)
        self.warm_pool(model)
        if warmup:
            self.engine.warmup()

    def snapshot_disk(self, name, model_path):
        import os
        if not self.engine.persistent:
            return None
        lib = os.path.join(_ROOT, MODELS.get(name, MODELS[MODEL])["lib"])
        identity = "|".join(f"{os.path.basename(p)}:{st.st_size}:{st.st_mtime_ns}" for p in (model_path, lib) for st in [os.stat(p)])
        return Disk(os.path.join(SNAPSHOT_DIR, name), identity)

    def open_vision(self, name):
        import os
        spec = MODELS.get(name, {})
        path = spec.get("vision") and os.path.join(_ROOT, spec["vision"])
        if not (path and os.path.exists(path) and self.engine.vision):
            return None
        self.engine.vision_open(path)
        return Pictures(self.engine, 5120 if name == "qwen3.8-27b" else 2560, spec["pictures"], self.max_pixels)

    def resolve(self, name):
        base = str(name or "").removesuffix("-fast")
        return base if base in MODELS and present(base) else self.active

    def rewarm(self):
        import os
        if self.warming and MODELS[self.active].get("warm"):
            self.warmer.start([os.path.join(_ROOT, spec["path"]) for name, spec in MODELS.items() if name != self.active],
                              os.path.join(_ROOT, MODELS[self.active]["path"]))
        else:
            self.warmer.stop()

    def use(self, name):
        if name == self.active:
            return
        t0 = time.time()
        self.warmer.stop()
        with self.state_lock:
            self.loading = name
        try:
            self.keep_pool(force=True)
            self.engine.shutdown()
            self.load_engine(name)
            self.metrics.add("chrysalis_model_loads_total", model=name)
            self.metrics.add("chrysalis_model_load_seconds_total", time.time() - t0, model=name)
            print(f"chrysalis: switched {self.active} -> {name} in {time.time() - t0:.1f} s (context {self.capacity} tokens)",
                  file=self.log, flush=True)
            self.active = name
        finally:
            with self.state_lock:
                self.loading = None
            self.rewarm()

    def load_engine(self, name):
        import os
        spec = MODELS[name]
        self.engine = Engine(os.path.join(_ROOT, spec["path"]), max_seqs=self.slots, umax=self.umax or spec["chunk"],
                             lib=os.path.join(_ROOT, spec["lib"]), warmup=True)
        self.capacity = min(MAX_CONTEXT, self.engine.capacity)
        self.cache = Cache(self.engine, self.slots, max_snapshots=spec["snapshots"],
                           disk=self.snapshot_disk(name, os.path.join(_ROOT, spec["path"])))
        self.pictures = self.open_vision(name)
        self.n_draft, self.lookup = spec["draft"], spec.get("lookup", 0)
        self.warm_pool(name)

    def warm_pool(self, name):
        t0 = time.time()
        moved = self.engine.pool_warm(self.pool_path(name))
        if moved >= 0:
            print(f"chrysalis: the expert pool as the last run left it: {moved} experts moved in {time.time() - t0:.1f} s",
                  file=self.log, flush=True)
        self.pool_saved = time.time()

    def pool_path(self, name):
        return os.path.join(SNAPSHOT_DIR, name, "pool.bin")

    def keep_pool(self, force=False):
        if not force and time.time() - self.pool_saved < 30:
            return
        self.pool_saved = time.time()
        path = self.pool_path(self.active)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.engine.pool_save(path)

    def sleep(self):
        t0 = time.time()
        self.keep_pool(force=True)
        self.warmer.stop()
        self.pausing.set()
        self.lanes.hold()
        self.gpu.acquire(main=True)
        with self.state_lock:
            self.loading = self.active
        held = vram_used()
        self.engine.shutdown()
        wait_vram_freed(held)
        print(f"chrysalis: the machine is going to sleep: GPU freed in {time.time() - t0:.2f} s", file=self.log, flush=True)

    def unload(self):
        with self.wake_lock:
            if not self.asleep and not self.suspended:
                self.sleep()
            self.asleep, self.suspended = True, False

    def awake(self):
        with self.wake_lock:
            if self.asleep:
                self.asleep = False
                self.wake()

    def system_sleep(self):
        with self.wake_lock:
            if not self.asleep:
                self.sleep()
                self.suspended = True

    def system_wake(self):
        with self.wake_lock:
            if self.suspended:
                self.suspended = False
                self.wake()

    def wake(self):
        t0 = time.time()
        try:
            self.load_engine(self.active)
            self.metrics.add("chrysalis_model_loads_total", model=self.active)
            self.metrics.add("chrysalis_model_load_seconds_total", time.time() - t0, model=self.active)
            print(f"chrysalis: awake: {self.active} loaded in {time.time() - t0:.1f} s", file=self.log, flush=True)
            now = link.status(self.link_dev)
            if now and now["slow"]:
                print(f"chrysalis: PCIe link at {now['gts']:g} GT/s x{now['width']} after the wake, slower than this machine's "
                      f"best ({now['best_gts']:g} GT/s): starting Chrysalis again trains it again", file=self.log, flush=True)
        finally:
            with self.state_lock:
                self.loading = None
            self.pausing.clear()
            self.gpu.release()
            self.lanes.release()
            self.rewarm()

    def chat(self, body):
        messages = self.read_documents(merge_system(body.get("messages") or []))
        if not any(m.get("role") == "user" for m in messages):
            raise RequestError("the request needs at least one user message")
        final = messages[-1] if body.get("continue_final_message") else None
        if final is not None and final.get("role") != "assistant":
            raise RequestError("continue_final_message needs a final assistant message")
        aside = bool(body.get("chrysalis_aside"))
        seed = seed_of(body)
        tools = body.get("tools") if body.get("tool_choice") != "none" else None
        kw = body.get("chat_template_kwargs") or {}
        sys0 = text_of(messages[0].get("content")) if messages and messages[0].get("role") == "system" else ""
        side, compacting = sys0.startswith(SIDE_PROMPTS), sys0.startswith(COMPACT_PROMPT)
        effort = EFFORTS.get(kw.get("reasoning_effort") or body.get("reasoning_effort") or "xhigh", "xhigh")
        thinking = kw.get("enable_thinking", body.get("enable_thinking", not side))
        if str(body.get("model", "")).endswith("-fast") or body.get("reasoning_effort") == "none":
            thinking = False
        style = (thinking, effort, kw.get("preserve_thinking", True))
        rewrite = self.compaction(messages) if compacting else None
        if rewrite:
            messages, tools, style = rewrite
            thinking = style[0]
        messages = self.sticky.apply(messages)

        def render(pictures, seen):
            images = (lambda pic: pictures.tokens(pic, seen, body.get("max_pixels"))) if pictures else None
            try:
                return self.renderer.render(messages if final is None else messages[:-1], tools, *style,
                                            think_reply=False if aside else None, images=images)
            except ValueError as e:
                raise RequestError(str(e)) from None
        cont, seen, pictures, prefix = None, {}, self.pictures, []
        prompt, marks = render(pictures, seen)
        if final is not None:
            reply_thinking = bool(thinking) and not aside
            prefix = self.reply_prefix(final, body.get("chrysalis_original"), reply_thinking)
            if body.get("chrysalis_append_token") is not None:
                t = int(body["chrysalis_append_token"])
                if t in self.tok.eos or not 0 <= t < self.tok.full.get_vocab_size():
                    raise RequestError("chrysalis_append_token must be a token that continues the reply")
                prefix.append(t)
            parser = OutputParser(self.tok, reply_thinking)
            parser.feed(prefix)
            prompt, cont = prompt + prefix, (prefix, parser)
            reasoning, content = parser.raw()
            yield "prefix", {"reasoning_content": reasoning, "content": content}
        if aside:
            thinking = False
        sp, bias, breakers, n_probs, n_predict = sampling_of(body, DEFAULTS[bool(thinking)], self.tok, -1 if tools else self.tok.id["<tool_call>"])
        top_lp = int(body.get("top_logprobs") or 0) if body.get("logprobs") else n_probs if n_probs > 0 else -1
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else stops
        if not isinstance(stops, list) or not all(isinstance(x, str) for x in stops):
            raise RequestError("stop must be a string or a list of strings")
        schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools or [] if "function" in t}
        answer, strict = self.constraints(body, tools, schemas)
        if (answer is not None or strict is not None) and (top_lp >= 0 or final is not None):
            raise RequestError("response_format, strict tools and tool_choice cannot be combined with logprobs or "
                               "continue_final_message")
        budget = body.get("thinking_budget", kw.get("thinking_budget", (body.get("reasoning") or {}).get("max_tokens")))
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 0):
            raise RequestError("thinking_budget must be a whole number of tokens")
        max_tokens = (int(body.get("max_completion_tokens") or body.get("max_tokens") or MAX_OUTPUT) if n_predict is None
                      else MAX_OUTPUT if n_predict < 0 else n_predict)
        if side:
            max_tokens = min(max_tokens, 256)
        if len(prompt) + 1 >= self.capacity:
            raise ContextError(f"the request ({len(prompt)} tokens) exceeds the available context size "
                               f"({self.capacity} tokens), try increasing it")
        max_tokens = min(max_tokens, self.capacity - len(prompt) - 1)
        record = None if side or compacting or cont or aside else {"messages": messages, "tools": tools, "style": style}
        arrived = time.time()
        admit(self.lanes, self.gpu, side, arrived)
        held = Held(self.gpu, main=not side, arrived=arrived)
        try:
            want = self.active if side else self.resolve(body.get("model"))
            if want != self.active:
                self.lanes.hold(mine=True)
                try:
                    held.take()
                    if want != self.active:
                        self.use(want)
                finally:
                    self.lanes.release()
            held.take()
            if self.pausing.is_set():
                raise Asleep("the machine is going to sleep; Chrysalis loads the model again when it wakes")
            sp.n_draft, sp.lookup = self.n_draft, self.lookup
            if len(prompt) + 1 >= self.capacity:
                raise ContextError(f"the request ({len(prompt)} tokens) exceeds the available context size "
                                   f"({self.capacity} tokens), try increasing it")
            if seen and self.pictures is not pictures:
                seen = {}
                prompt, marks = render(self.pictures, seen)
                prompt += prefix
            yield from self.run(np.array(prompt, np.int32), marks, sp, seed, max_tokens, bool(thinking), schemas, record, side,
                                cont=cont, top_lp=min(top_lp, 64), aside=aside, stops=stops, seen=seen, answer=answer, held=held,
                                strict=strict, budget=budget if thinking else None, bias=bias, breakers=breakers)
        finally:
            held.drop()
            if held.slot:
                held.slot.busy = False
            self.lanes.leave()

    def constraints(self, body, tools, schemas):
        choice, only = body.get("tool_choice"), None
        if isinstance(choice, dict):
            only = (choice.get("function") or {}).get("name") or choice.get("name")
            if only not in schemas:
                raise RequestError(f"tool_choice names {only!r}, which is not among the tools")
        required = choice == "required" or only is not None
        if required and not schemas:
            raise RequestError('tool_choice "required" needs tools')
        parallel = body.get("parallel_tool_calls") is not False
        try:
            schema = Grammars.schema_of(body.get("response_format"))
            if schema is None and not required and not any((t.get("function") or {}).get("strict") for t in tools or []):
                return None, None
            if self.grammars is None:
                self.grammars = Grammars(self.tok)
            if required:
                return self.grammars.matcher(tools=schemas, only=only, parallel=parallel), None
            if schema is not None:
                return self.grammars.matcher(schema=schema, tools=schemas or None, parallel=parallel), None
            self.grammars.matcher(tools=schemas, parallel=parallel, inside=True)
            return None, lambda: self.grammars.matcher(tools=schemas, parallel=parallel, inside=True)
        except ValueError as e:
            raise RequestError(str(e)) from None

    def held_step(self, slot, sp, answer, unread, room):
        if self.engine.masked and len(unread) == 1:
            drafts = self.engine.draft(slot.sid, sp, answer.forced(), min(7, room - 1))
            masks, k = answer.masks(drafts)
            toks = self.engine.generate_masked(slot.sid, sp, drafts[:k], masks, room)
            answer.consume(toks)
            return toks, toks[-1:]
        if unread:
            self.engine.prefill(slot.sid, unread)
        tok = self.engine.draw(slot.sid, *self.engine.masked_top(answer.bitmask()), sp)
        if self.engine.masked:
            answer.consume([tok])
            if tok not in self.tok.eos:
                self.engine.set_pending(slot.sid, tok)
            return [tok], [tok]
        toks = [tok] if tok in self.tok.eos else [tok] + answer.take(tok)[:max(room - 1, 0)]
        return toks, toks

    @functools.cache
    def close_thinking(self, budget=True, newline=False):
        enc = lambda t: self.tok.full.encode(t, add_special_tokens=False).ids
        head = enc("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly "
                   "now.\n") if budget else enc("\n") if newline else []
        return head + [self.tok.id["</think>"]] + enc("\n\n")

    def inject(self, slot, held, pending, toks, keep=False):
        held.take()
        try:
            self.engine.prefill(slot.sid, pending + toks)
        except BaseException:
            held.drop()
            raise
        self.engine.push(slot.sid, toks)
        if not keep:
            held.drop()

    def read_documents(self, messages):
        try:
            return self.documents.expand(messages, self.pictures is not None)
        except DocumentError as e:
            raise RequestError(str(e)) from None

    def count_tokens(self, body):
        messages = self.read_documents(merge_system(body.get("messages") or []))
        tools = body.get("tools") if body.get("tool_choice") != "none" else None

        def images(item):
            if isinstance(item, Clip):
                video = vision.Video(vision.image_bytes(item.url, "videos"), float(item.fps or vision.VIDEO_FPS))
                return [(t, [IMG_BASE] * (video.grid[0] * video.grid[1] // 4)) for t in video.times]
            limit = vision.pixel_limit(item.max_pixels, body.get("max_pixels"), cap=self.max_pixels)
            rows, cols = vision.grid(vision.image_bytes(item.url), limit)
            return [IMG_BASE] * (rows * cols // 4)
        try:
            prompt, _ = self.renderer.render(messages, tools, body.get("enable_thinking", True), "xhigh", True,
                                             images=images if self.pictures else None)
        except ValueError as e:
            raise RequestError(str(e)) from None
        return len(prompt)

    def reply_prefix(self, msg, original, thinking):
        S = self.tok.id
        r, c = msg.get("reasoning_content") or "", text_of(msg.get("content"))
        src = original or msg
        hit = self.replay.get(message_key(src)) or self.replay.get(message_key(src, False))
        if hit is not None:
            ids, p, lr, lc = list(hit[1]), OutputParser(self.tok, thinking), 0, 0
            for n in range(len(ids) + 1):
                if (lr, lc) == (len(r), len(c)) and p.raw() == (r, c):
                    return ids[:n]
                if n == len(ids) or lr > len(r) or lc > len(c):
                    break
                for ev in p.feed([ids[n]]):
                    if ev[0] == "reasoning":
                        lr += len(ev[1])
                    elif ev[0] == "content":
                        lc += len(ev[1])
        pieces = ([Content(r), "\n", S["</think>"], "\n\n", Content(c)] if c else [Content(r)]) if thinking else [Content(c)]
        return list(self.tok.encode(pieces)[0])

    def logprobs(self, ids, rv, ri, n_top, ban):
        out = []
        for tok, v, i in zip(ids, rv, ri):
            m = float(v.max())
            lse = m + float(np.log(np.exp(v - m).sum()))
            j = np.flatnonzero(i == tok)
            top = [k for k in np.argsort(-v) if i[k] != ban][:n_top]
            out.append((float(v[j[0]]) - lse if len(j) else -99.0,
                        [{"token": self.piece(int(i[k])), "logprob": round(float(v[k]) - lse, 4), "id": int(i[k])} for k in top]))
        return out

    @functools.lru_cache(maxsize=65536)
    def piece(self, tid):
        return self.tok.decode([tid])

    @functools.cached_property
    def token_bytes(self):
        printable = [*range(33, 127), *range(161, 173), *range(174, 256)]
        byte_of = {chr(b): b for b in printable} | {chr(256 + i): b for i, b in enumerate(b for b in range(256) if b not in printable)}
        added = set(self.tok.added)
        words = (self.tok.full.id_to_token(i) or "" for i in range(self.tok.full.get_vocab_size()))
        return [w.encode() if w in added else bytes(byte_of[c] for c in w) for w in words]

    @functools.lru_cache(maxsize=16)
    def restart_sequences(self, breakers):
        out, known = [], set()
        for text in breakers:
            b = text.encode()[:40]
            for tid, word in enumerate(self.token_bytes if b else ()):
                if b in word:
                    out += [1, tid]
                    continue
                for k in range(1, len(b)):
                    if not word.endswith(b[:k]):
                        continue
                    try:
                        tail = tuple(self.tok.encode_plain(b[k:].decode())[:20])
                    except UnicodeDecodeError:
                        continue
                    if (tid, tail) not in known:
                        known.add((tid, tail))
                        out += [1 + len(tail), tid, *tail]
        return out

    def compaction(self, messages):
        user = text_of(messages[-1].get("content"))
        m = re.search(r"<conversation>\n(.*)\n</conversation>", user, re.S)
        if not m:
            return None

        def score(rec):
            texts = (first_text(x.get("content")) for x in rec["messages"] if x.get("role") == "user")
            return sum("[User]: " + t in m.group(1) for t in texts if t)
        rec = max(reversed(self.sessions.values()), key=score, default=None)
        if rec is None or score(rec) == 0:
            return None
        request = (text_of(messages[0].get("content")) + "\n\nDo not call any tools.\n\n" + user[:m.start()]
                   + "<conversation>\n(the conversation above)\n</conversation>" + user[m.end():])
        return rec["messages"] + [rec["reply"], {"role": "user", "content": request}], rec["tools"], rec["style"]

    def prefill(self, slot, prompt, a, b, pos=None, held=None):
        chunk = self.umax or MODELS[self.active]["chunk"]
        while a < b:
            if self.pausing.is_set():
                raise Asleep("the machine is going to sleep; Chrysalis loads the model again when it wakes")
            t = time.time()
            try:
                e = min(b, a + chunk)
                self.engine.prefill(slot.sid, prompt[a:e].tolist(), None if pos is None else pos[a:e])
                a = e
            except KVFull:
                a = self.engine.seq_len(slot.sid)
                slot.tokens = prompt[:a]
                if not self.free_kv(b - a + 64, keep=slot):
                    raise ContextError("the request exceeds the available context size (KV cache full), try increasing it")
            slot.tokens = prompt[:a]
            if held is not None and a < b and self.replying - {slot.sid}:
                spent = time.time() - t
                held.drop()
                time.sleep(spent)
                held.take()

    def free_kv(self, need, keep):
        self.cache.trim(need)
        for s in sorted(self.cache.slots, key=lambda s: s.used):
            if self.engine.free_tokens() >= need:
                break
            if s is not keep and not s.busy and len(s.tokens):
                self.engine.truncate(s.sid, 0)
                s.tokens = s.tokens[:0]
        return self.engine.free_tokens() >= need

    def run(self, prompt, marks, sp, seed, max_tokens, thinking, schemas, record, side=False, cont=None, top_lp=-1, aside=False,
            stops=(), seen=None, answer=None, held=None, strict=None, budget=None, bias=(), breakers=DRY_BREAKERS):
        t0 = time.time()
        trace = [] if TRACE else None
        slot, cached = self.cache.acquire(prompt[:-1] if answer else prompt)
        held.slot = slot
        if aside and cached and cached == len(slot.tokens):
            self.cache.checkpoint(slot)
        if trace is not None:
            trace.append(("acquire", time.time() - t0))
        pos = self.pictures.positions(prompt) if seen else None
        encoded = 0
        if seen:
            t = time.time()
            try:
                encoded = self.pictures.ensure([pic for a, b, pic in self.pictures.runs(prompt) if b > cached], seen)
            except vision.ImageError as e:
                raise RequestError(str(e)) from None
            if trace is not None and encoded:
                trace.append((f"{encoded} pictures", time.time() - t))
        if encoded and not MODELS[self.active].get("vision_stays", True):
            self.engine.vision_release()
        turns = np.flatnonzero(prompt == self.im_start)
        points = [(marks["tools"], True), (marks["static"], True), (marks["system"], side),
                  (int(turns[-1]) if len(turns) and not side else None, False)]
        done = cached
        for at, pinned in sorted(p for p in points if p[0] is not None and p[0] > cached) + [(len(prompt), False)]:
            if at <= done:
                continue
            t = time.time()
            self.prefill(slot, prompt, done, at, pos, held)
            if trace is not None:
                trace.append((f"prefill {at - done}", time.time() - t))
            done = at
            if at < len(prompt) or not side:
                t = time.time()
                self.cache.checkpoint(slot, pinned, logits=at == len(prompt))
                if trace is not None:
                    trace.append(("checkpoint", time.time() - t))
        t_prefill = time.time()
        text = prompt[prompt < IMG_BASE]
        dry = sp.dry_multiplier != 0
        need = len(text) if sp.repeat_last_n < 0 or dry and sp.dry_penalty_last_n < 0 else max(sp.repeat_last_n, sp.dry_penalty_last_n * dry)
        self.engine.set_context(slot.sid, text[len(text) - min(need, len(text)):])
        if sp.lookup > 0:
            self.engine.set_lookup(slot.sid, prompt)
        self.engine.begin(slot.sid, seed, sp)
        if bias:
            self.engine.set_bias(slot.sid, bias)
        if dry:
            self.engine.set_breakers(slot.sid, self.restart_sequences(breakers))
        prefix, parser = cont or ([], OutputParser(self.tok, thinking, schemas))
        stop_text = StopText(stops)
        out, finish, steps, completed, calls = [], "stop", 0, False, []
        think_end, call_start = self.tok.id["</think>"], self.tok.id["<tool_call>"]
        holding, unread, behind = answer is not None and not thinking, [], 0
        unclosed = False
        sp.halt = think_end if answer is not None and thinking else call_start if strict is not None else -1
        hsp = Sampling.from_buffer_copy(sp)
        hsp.halt = -1
        if not holding:
            held.drop()
        batched = not holding
        if batched:
            self.batcher.begin()
        self.replying.add(slot.sid)
        try:
            yield "start", {"prompt_tokens": len(prompt), "cached_tokens": cached}
            while True:
                if parser.state == "reasoning" and (unclosed or budget is not None and len(out) >= budget):
                    close = self.close_thinking(not unclosed, not self.tok.full.decode(out[-1:]).endswith("\n"))
                    if len(close) > max_tokens - len(out):
                        finish = "length"
                        break
                    if batched:
                        self.batcher.end()
                        batched = False
                    try:
                        self.inject(slot, held, out[-1:] if behind else [], close, keep=answer is not None)
                    except KVFull:
                        finish = "length"
                        break
                    out += close
                    behind, budget, unclosed = 0, None, False
                    for ev in stop_text.feed(parser.feed(close)):
                        yield ev
                    if answer is not None:
                        holding, unread = True, []
                    else:
                        self.batcher.begin()
                        batched = True
                most = 16 if budget is None or parser.state != "reasoning" else min(16, budget - len(out))
                try:
                    if holding:
                        held.take()
                        try:
                            ids, unread = self.held_step(slot, hsp, answer, unread, max_tokens - len(out))
                        finally:
                            held.drop()
                        behind = len(unread)
                    elif top_lp >= 0:
                        ids, rv, ri = self.batcher.step(self.engine, slot.sid, sp, rows=True, most=most)
                        behind = 1
                    else:
                        ids = self.batcher.step(self.engine, slot.sid, sp, most=most)
                        behind = 1
                except KVFull:
                    held.take()
                    try:
                        freed = self.free_kv(64, keep=slot)
                    finally:
                        held.drop()
                    if freed:
                        continue
                    finish = "length"
                    break
                if steps == 0:
                    t_first = time.time()
                    if trace is not None:
                        trace.append(("first step", t_first - t_prefill))
                steps += 1
                stop = next((i for i, t in enumerate(ids) if t in self.tok.eos), None)
                n, room = len(ids) if stop is None else stop, max_tokens - len(out)
                out += ids
                emit = ids[:min(n, room)]
                for ev in stop_text.feed(parser.feed(emit, self.logprobs(emit, rv, ri, top_lp, sp.ban) if top_lp >= 0 else None)):
                    calls += [ev[1]] if ev[0] == "tool_call" else []
                    yield ev
                if stop_text.hit:
                    break
                if n > room or n == room and stop is None:
                    finish = "length"
                    break
                if stop is not None and parser.state == "reasoning" and not holding:
                    out.pop()
                    behind, unclosed = 0, True
                    continue
                if stop is not None or holding and answer.done():
                    break
                if self.pausing.is_set():
                    finish = "length"
                    break
                if not holding and sp.halt >= 0 and ids[-1] == sp.halt:
                    if sp.halt == call_start:
                        answer = strict()
                    holding, unread = True, [sp.halt]
                    self.batcher.end()
                    batched = False
            for ev in stop_text.feed(parser.finish()) + stop_text.flush():
                calls += [ev[1]] if ev[0] == "tool_call" else []
                yield ev
            completed = True
        finally:
            self.replying.discard(slot.sid)
            if batched:
                self.batcher.end()
            read = len(out) - behind
            slot.tokens = np.concatenate([prompt, np.array(out[:read], np.int32)]) if out else prompt
            reasoning, content = parser.text()
            if stops:
                content = "".join(stop_text.content).strip()
            reply = {"role": "assistant", "content": content, "reasoning_content": reasoning,
                     "tool_calls": [{"type": "function", "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}} for n, a in calls]}
            if (content or calls) and not aside and not stop_text.hit:
                gen = prefix + (out[:-1] if out and out[-1] in self.tok.eos else out)[:max_tokens]
                self.replay.put(message_key(reply, False), (thinking, gen))
                self.replay.put(message_key(reply), (thinking, gen))
            if record:
                record["reply"] = reply
                key = next(text_of(m.get("content")) for m in record["messages"] if m.get("role") == "user")
                self.sessions.put(hashlib.sha1(key.encode()).hexdigest(), record)
            if not completed:
                t1 = time.time()
                self.metrics.reply(self.active, {"prompt_tokens": len(prompt), "cached_tokens": cached,
                                                 "completion_tokens": min(len(out), max_tokens), "ttft": (t_first if steps else t1) - t0,
                                                 "prefill_s": t_prefill - t0, "decode_s": t1 - t_prefill}, steps, "client")
                self.recent.put(time.time(), {"prompt_tokens": len(prompt), "cached_tokens": cached, "completion_tokens": len(out), "finish_reason": "client"})
                print(f"chrysalis: prompt {len(prompt)} (cached {cached}) | stopped by the client after {len(out)} tokens "
                      f"| slot {slot.sid}", file=self.log, flush=True)
        if finish == "stop" and calls and not stop_text.hit:
            finish = "tool_calls"
        t1 = time.time()
        n_gen = len(out)
        info = {"prompt_tokens": len(prompt), "cached_tokens": cached, "completion_tokens": min(n_gen, max_tokens),
                "finish_reason": finish, "ttft": (t_first if steps else t1) - t0, "prefill_s": t_prefill - t0,
                "decode_tps": n_gen / max(t1 - t_prefill, 1e-6), "tokens_per_step": n_gen / max(steps, 1), "seed": seed,
                "stop_sequence": stop_text.matched}
        self.recent.put(time.time(), info)
        self.metrics.reply(self.active, {**info, "decode_s": t1 - t_prefill}, steps, finish)
        self.keep_pool()
        print(f"chrysalis: prompt {len(prompt)} (cached {cached}){f' | {encoded} pictures encoded' if encoded else ''} ttft {info['ttft']:.2f}s | gen {n_gen} "
              f"{info['decode_tps']:.1f} tok/s {info['tokens_per_step']:.2f}/step | {finish} | slot {slot.sid}"
              + (" | " + ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in trace) if trace else ""),
              file=self.log, flush=True)
        yield "done", info

    def stats(self):
        with self.state_lock:
            if self.asleep:
                return {"model": self.active, "asleep": True, "models": list(MODELS)}
            if self.loading:
                return {"model": self.active, "loading": self.loading, "progress": load_progress(), "models": list(MODELS)}
            out = {"model": self.active, "models": list(MODELS), "context": self.capacity, "kv_tokens": self.engine.capacity,
                   "cache": self.cache.stats(), "recent": [{"at": t, **v} for t, v in list(self.recent.items())[-16:]], "link": link.status(self.link_dev)}
            experts = self.engine.expert_stats()
            if experts:
                out["experts"] = experts
            return out

    def gauges(self):
        g = []
        with self.state_lock:
            g.append(("chrysalis_model_loaded", "1 for the model on the GPU", [({"model": m}, int(m == self.active and not self.loading)) for m in MODELS]))
            g.append(("chrysalis_model_loading", "1 while a model loads (a switch, a wake)", [({}, int(bool(self.loading)))]))
            g.append(("chrysalis_replies_writing", "Replies being written now", [({}, len(self.replying))]))
            if not self.loading:
                c = self.cache.stats()
                g += [("chrysalis_context_tokens", "Context a request may use", [({}, self.capacity)]),
                      ("chrysalis_kv_tokens", "Tokens the KV pool holds", [({}, self.engine.capacity)]),
                      ("chrysalis_kv_free_tokens", "Tokens of the KV pool no sequence holds", [({}, c["free_kv_tokens"])]),
                      ("chrysalis_snapshots", "Prompt-cache snapshots in RAM", [({}, c["snapshots"])])]
                x = self.engine.expert_stats()
                if x:
                    g += [("chrysalis_experts_decode_picks_total", "Experts routed to by decode steps", [({}, x["decode_picks"])]),
                          ("chrysalis_experts_decode_picks_in_pool_total", "Of them, experts in the VRAM pool", [({}, x["decode_picks_in_pool"])]),
                          ("chrysalis_experts_placed_total", "Experts copied into the pool", [({}, x["placed"])]),
                          ("chrysalis_experts_mirrored", "The file's VRAM-tier experts out of the pool, copied into RAM", [({}, x["mirrored"])]),
                          ("chrysalis_experts_mirror_cap", "How many of them RAM may hold (CHRYSALIS_FN_SPILL_GB)", [({}, x["mirror_cap"])]),
                          ("chrysalis_dma_bytes_per_second", "Host-to-GPU DMA rate the pool's copies measure", [({}, x["dma_bytes_per_s"])]),
                          ("chrysalis_experts_gpu_share", "Share of the cold picks the GPU computes in decode", [({}, x["gpu_share"])])]
        ln = link.status(self.link_dev)
        if ln:
            g += [("chrysalis_pcie_link_gts", "PCIe link speed now, GT/s (the slowest link to the GPU)", [({}, ln["gts"])]),
                  ("chrysalis_pcie_link_width", "PCIe link width now, lanes", [({}, ln["width"])]),
                  ("chrysalis_pcie_link_best_gts", "The fastest link speed seen on this machine, GT/s", [({}, ln["best_gts"])]),
                  ("chrysalis_pcie_link_slow", "1 when the link is slower than this machine's best (starting Chrysalis again trains it again)", [({}, int(ln["slow"]))])]
        return g

    def load(self, name):
        if name not in MODELS:
            raise RequestError(f"unknown model {name!r}; Chrysalis has {', '.join(MODELS)}")
        with self.wake_lock:
            if self.asleep:
                self.active, self.asleep = name, False
                self.wake()
                return
        self.lanes.hold()
        try:
            self.gpu.acquire(main=True)
            try:
                if self.pausing.is_set():
                    raise Asleep("the machine is going to sleep; Chrysalis loads the model again when it wakes")
                self.use(name)
            finally:
                self.gpu.release()
        finally:
            self.lanes.release()


NAMES = {"", "localhost", socket.gethostname().lower(), socket.gethostname().lower() + ".local"}


def host_ok(host):
    name = (host or "").strip().lower()
    if name.startswith("["):
        return True
    name = name.rsplit(":", 1)[0]
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return name in NAMES


def make_handler(state, ready):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "chrysalis"

        def log_message(self, *args):
            pass

        def send_json(self, code, obj):
            if code >= 400 and state.get("srv"):
                state["srv"].metrics.add("chrysalis_http_errors_total", status=code)
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not host_ok(self.headers.get("Host")):
                return self.send_json(403, {"error": {"message": "unknown host", "type": "invalid_request_error"}})
            if not ready.is_set() and self.path.rstrip("/") in ("/health", "/v1/health", "/chrysalis/stats"):
                return self.send_json(503, {"status": "loading", "model": state.get("loading"), "progress": load_progress()})
            if self.path.split("?")[0].rstrip("/") == "/metrics":
                srv = state.get("srv")
                text = (srv.metrics.render(srv.gauges()) if srv else
                        "# HELP chrysalis_model_loading 1 while a model loads (a switch, a wake)\n# TYPE chrysalis_model_loading gauge\nchrysalis_model_loading 1\n")
                data = text.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            ready.wait()
            srv = state["srv"]
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                now = int(time.time())
                self.send_json(200, {"object": "list", "data": [
                    {"id": m, "object": "model", "created": now, "owned_by": "chrysalis", "context_length": MAX_CONTEXT}
                    for name in MODELS if present(name) for m in (name, name + "-fast")]})
            elif self.path.rstrip("/") in ("/health", "/v1/health"):
                self.send_json(200, {"status": "ok"})
            elif self.path.rstrip("/") == "/chrysalis/stats":
                self.send_json(200, srv.stats())
            else:
                self.send_json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not host_ok(self.headers.get("Host")):
                return self.send_json(403, {"error": {"message": "unknown host", "type": "invalid_request_error"}})
            if self.path.rstrip("/") == "/chrysalis/unload":
                ready.wait()
                state["srv"].unload()
                return self.send_json(200, {"asleep": True})
            if self.path.rstrip("/") == "/chrysalis/load":
                ready.wait()
                srv = state["srv"]
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                    srv.load(str(body.get("model", "")))
                except json.JSONDecodeError as e:
                    return self.send_json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
                except RequestError as e:
                    return self.send_json(e.status, {"error": {"message": str(e), "type": "invalid_request_error", "code": e.code}})
                return self.send_json(200, {"model": srv.active, "context": srv.capacity})
            path = self.path.split("?")[0].rstrip("/").removeprefix("/v1")
            if path not in ("/chat/completions", "/messages", "/messages/count_tokens", "/responses"):
                return self.send_json(404, {"error": {"message": "not found"}})
            ready.wait()
            srv = state["srv"]
            srv.awake()
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError:
                return self.send_json(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
            if path.startswith("/messages"):
                return self.messages_api(srv, body, path.endswith("count_tokens"))
            if path == "/responses":
                return self.responses_api(srv, body)
            rid, created, model = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()), body.get("model") or srv.active
            gen = srv.chat(body)
            try:
                first = next(gen)
            except RequestError as e:
                return self.send_json(e.status, {"error": {"message": str(e), "type": "invalid_request_error", "code": e.code}})
            except Exception as e:
                print(f"chrysalis: request failed: {e!r}", file=srv.log, flush=True)
                return self.send_json(500, {"error": {"message": f"internal error: {e}", "type": "server_error"}})
            try:
                if body.get("stream"):
                    self.stream(gen, first, rid, created, model, (body.get("stream_options") or {}).get("include_usage", True))
                else:
                    self.complete(gen, first, rid, created, model)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                gen.close()

        def messages_api(self, srv, req, count):
            def error(status, kind, message):
                self.send_json(status, {"type": "error", "error": {"type": kind, "message": message}})
            try:
                body = apis.messages_body(req)
                if count:
                    return self.send_json(200, {"input_tokens": srv.count_tokens(body)})
                gen = srv.chat(body)
                first = next(gen)
            except RequestError as e:
                kind = "overloaded_error" if isinstance(e, Asleep) else "invalid_request_error"
                return error(e.status, kind, str(e))
            except ValueError as e:
                return error(400, "invalid_request_error", str(e))
            except Exception as e:
                print(f"chrysalis: request failed: {e!r}", file=srv.log, flush=True)
                return error(500, "api_error", f"internal error: {e}")
            model = req.get("model") or srv.active
            events = itertools.chain([first], gen)
            try:
                if req.get("stream"):
                    self.events(apis.messages_stream(events, model),
                                lambda e: ("error", {"type": "error", "error": {"type": "api_error", "message": str(e)}}))
                else:
                    self.send_json(200, apis.messages_result(events, model))
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                gen.close()

        def responses_api(self, srv, req):
            def error(status, message, code=None):
                self.send_json(status, {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": code}})
            prev = req.get("previous_response_id")
            history = srv.stored.get(prev) if prev else None
            if prev and history is None:
                return error(404, f"no stored response {prev} (Chrysalis keeps the last {srv.stored.limit}, until it restarts)",
                             "previous_response_not_found")
            try:
                body = apis.responses_body(req, history)
                gen = srv.chat(body)
                first = next(gen)
            except RequestError as e:
                return error(e.status, str(e), e.code)
            except ValueError as e:
                return error(400, str(e))
            except Exception as e:
                print(f"chrysalis: request failed: {e!r}", file=srv.log, flush=True)
                return self.send_json(500, {"error": {"message": f"internal error: {e}", "type": "server_error", "param": None, "code": None}})
            build = apis.ResponseBuilder(req, req.get("model") or srv.active, lambda text: len(srv.tok.full.encode(text).ids))
            events = itertools.chain([first], gen)
            try:
                if req.get("stream"):
                    self.events(build.events(events),
                                lambda e: build.event("error", code="server_error", message=str(e), param=None))
                else:
                    self.send_json(200, build.result(events))
                if req.get("store", True):
                    srv.stored.put(build.id, build.history(body))
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                gen.close()

        def events(self, stream, on_error):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for name, data in stream:
                    self.event(name, data)
            except (BrokenPipeError, ConnectionResetError):
                raise
            except Exception as e:
                self.event(*on_error(e))
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def event(self, name, obj):
            data = b"event: " + name.encode() + b"\ndata: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

        def chunk(self, obj):
            data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode()) + b"\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

        def stream(self, gen, first, rid, created, model, usage):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            base = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model}
            self.chunk({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
            n_calls, ev = 0, first
            while True:
                kind, val = ev[0], ev[1]
                if kind == "prefix":
                    self.chunk({**base, "choices": [{"index": 0, "delta": {"chrysalis_prefix": val}, "finish_reason": None}]})
                    ev = next(gen)
                    continue
                if kind == "start":
                    if usage:
                        self.chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": None}], "chrysalis_prompt": val})
                    ev = next(gen)
                    continue
                if kind == "done":
                    self.chunk({**base, "seed": val["seed"], "choices": [{"index": 0, "delta": {}, "finish_reason": val["finish_reason"]}]})
                    if usage:
                        self.chunk({**base, "choices": [], "usage": usage_of(val)})
                    self.chunk(b"[DONE]")
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    return
                if kind == "tool_call":
                    name, args = val
                    delta = {"tool_calls": [{"index": n_calls, "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                                             "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}
                    n_calls += 1
                else:
                    delta = {"reasoning_content" if kind == "reasoning" else "content": val}
                choice = {"index": 0, "delta": delta, "finish_reason": None}
                if len(ev) > 2 and ev[2] is not None:
                    choice["logprobs"] = {"content" if kind == "content" else "reasoning_content": [ev[2]]}
                self.chunk({**base, "choices": [choice]})
                ev = next(gen)

        def complete(self, gen, first, rid, created, model):
            reasoning, content, calls, lps, ev = [], [], [], [], first
            while ev[0] != "done":
                kind, val = ev[0], ev[1]
                if kind in ("prefix", "start"):
                    pass
                elif kind == "tool_call":
                    calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                                  "function": {"name": val[0], "arguments": json.dumps(val[1], ensure_ascii=False)}})
                else:
                    (reasoning if kind == "reasoning" else content).append(val)
                    if kind == "content" and len(ev) > 2 and ev[2] is not None:
                        lps.append(ev[2])
                ev = next(gen)
            info = ev[1]
            msg = {"role": "assistant", "content": "".join(content).strip() or (None if calls else "")}
            if reasoning:
                msg["reasoning_content"] = "".join(reasoning).strip()
            if calls:
                msg["tool_calls"] = calls
            self.send_json(200, {"id": rid, "object": "chat.completion", "created": created, "model": model, "seed": info["seed"],
                                 "choices": [{"index": 0, "message": msg, "finish_reason": info["finish_reason"],
                                              **({"logprobs": {"content": lps}} if lps else {})}],
                                 "usage": usage_of(info)})

    return Handler


def seed_of(body):
    seed = body.get("seed")
    if seed is None:
        return random.getrandbits(32)
    if isinstance(seed, float) and seed.is_integer():
        seed = int(seed)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise RequestError("seed must be an integer")
    return random.getrandbits(32) if seed == -1 else seed % 2**64


def usage_of(info):
    return {"prompt_tokens": info["prompt_tokens"], "completion_tokens": info["completion_tokens"],
            "total_tokens": info["prompt_tokens"] + info["completion_tokens"],
            "prompt_tokens_details": {"cached_tokens": info["cached_tokens"]}}


class HTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


def serve(make_server, host="127.0.0.1", port=8080, model=None):
    state, ready = {"loading": model}, threading.Event()
    httpd = HTTPServer((host, port), make_handler(state, ready))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    srv = make_server()
    srv.gpu.acquire(main=True)
    state["srv"] = srv
    ready.set()
    print(f"chrysalis: serving {srv.active} on http://{host}:{port}/v1 (context {srv.capacity} tokens)", file=srv.log, flush=True)
    srv.engine.warmup()
    srv.gpu.release()
    srv.warming = True
    srv.rewarm()
    SleepGuard(srv.system_sleep, srv.system_wake, srv.log).start()
    threading.Event().wait()
