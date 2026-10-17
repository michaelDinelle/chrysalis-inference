"""OpenAI-compatible HTTP server around the kiln engine, built for coding-agent loops (OpenCode).

What keeps time-to-first-token low:
  * prompts render to token ids; assistant turns kiln generated are replayed as their exact ids, so each
    agent step extends the cached sequence (only new tool output / user text is prefilled);
  * snapshots at the end of the tool block and the static system prompt (shared by all sessions), the system turn
    and every prompt (with the next token's distribution, so a resent prompt needs no prefill), so new sessions,
    edited turns and evicted sessions resume from cache instead of from scratch;
  * side requests (session titles, summaries) wait until the agent requests pause;
  * OpenCode prompts are kept byte-stable (date frozen per conversation, appended reminders kept),
    and its compaction request is answered from the live conversation instead of a full re-read.
"""
import hashlib
import ipaddress
import json
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

from .cache import Cache
from .chat import OutputParser, Renderer, Tokens, message_key, text_of
from .engine import Engine, KVFull, Sampling, load_async, load_progress
from .models import MODELS

MODEL = "qwen3.8-27b"
_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
MAX_CONTEXT = 262144
# Reply length (reasoning included) when the request sets none. The model card asks for room on agentic tasks:
# 262,144 tokens of reasoning and 131,072 of final response where the two are limited separately. A reply never
# outgrows the context either; much past this it is a model stuck in a loop.
MAX_OUTPUT = 131072
DEFAULTS = {True: dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),   # Qwen3.8 card
            False: dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5)}
SIDE_PROMPTS = ("You are a title generator", "Summarize what was done in this conversation")
COMPACT_PROMPT = "You are a context summarization agent"
TRACE = bool(os.environ.get("KILN_TRACE"))     # log where each request's time to first token went
EFFORTS = {"xhigh": "xhigh", "high": "xhigh", "medium": "medium", "low": "low", "minimal": "low"}


class RequestError(ValueError):
    code = "invalid_request"


class ContextError(RequestError):
    code = "context_length_exceeded"


def first_text(content):
    """Start of a message's first text part: how OpenCode's compaction serializer begins a user message."""
    if isinstance(content, list):
        content = next((c.get("text") for c in content if isinstance(c, dict) and c.get("type") == "text"), "")
    return (content or "")[:200]


def merge_system(messages):
    """The Qwen3.8 template allows one system message, first. Fold any others into it (plugins add them)."""
    sys_parts = [text_of(m.get("content")) for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    return ([{"role": "system", "content": "\n\n".join(p for p in sys_parts if p)}] if sys_parts else []) + rest


class LRU(OrderedDict):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def put(self, k, v):
        self[k] = v
        self.move_to_end(k)
        while len(self) > self.limit:
            self.popitem(last=False)

    def get(self, k, default=None):
        if k not in self:
            return default
        self.move_to_end(k)
        return self[k]


DATE = re.compile(r"(Today's date: )([^\n]*)")


class Turnstile:
    """One request on the GPU at a time. Side requests (titles, summaries) go last: they wait while agent requests
    queue, and start only after SIDE_IDLE seconds without an agent request, counted from their own arrival too.
    OpenCode sends a new session's title request 8 ms before the session's first request, and an agent's next step
    ~25 ms after a tool returns, so a side request that took the GPU whenever it was free would hold up the turn
    the user is waiting for."""
    SIDE_IDLE = 0.3

    def __init__(self):
        self.cv = threading.Condition()
        self.busy, self.waiting = False, 0
        self.main_seen = 0.0                             # an agent request last arrived or finished

    def acquire(self, main):
        with self.cv:
            if main:
                self.waiting += 1
                self.main_seen = time.time()
                self.cv.wait_for(lambda: not self.busy)
                self.waiting -= 1
            else:
                arrived = time.time()
                while True:
                    idle = time.time() - max(self.main_seen, arrived)
                    if not self.busy and not self.waiting and idle >= self.SIDE_IDLE:
                        break
                    self.cv.wait(None if self.busy or self.waiting else self.SIDE_IDLE - idle)
            self.busy = True

    def release(self, main=True):
        with self.cv:
            self.busy = False
            if main:
                self.main_seen = time.time()
            self.cv.notify_all()


class Sticky:
    """Keeps a conversation's prompt stable across requests so it stays a cache hit:
    the date in OpenCode's environment block is frozen at the conversation's first request, and
    text that an earlier request appended to a user message (plan-mode reminders) is kept there."""

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
    def __init__(self, model_path, tokenizer_path, slots=3, n_draft=3, umax=None, log=sys.stderr, engine=None, warmup=True,
                 model=MODEL):
        """engine: a future of the Engine (engine.load_async) if its load already started; by default it loads here,
        alongside the tokenizer. warmup=False leaves Engine.warmup() to the caller (serve() runs it under the GPU lock).
        model: the name of the model that model_path holds (see MODELS); requests can switch to the others."""
        if engine is None:
            engine = load_async(model_path, max_seqs=slots, umax=umax or MODELS.get(model, MODELS[MODEL])["chunk"], warmup=False,
                                lib=__import__("os").path.join(_ROOT, MODELS.get(model, MODELS[MODEL])["lib"]))
        self.active, self.slots, self.umax = model, slots, umax
        self.loading = None              # the model a switch is loading; stats() then leaves the engine alone
        self.state_lock = threading.Lock()
        self.tok = Tokens(tokenizer_path)
        self.replay = LRU(8192)
        self.renderer = Renderer(self.tok, self.replay.get)
        self.sticky = Sticky()
        self.sessions = LRU(64)          # latest request + reply per conversation, for compaction rewrites
        self.gpu = Turnstile()
        self.n_draft, self.log = n_draft, log
        self.recent = LRU(64)
        self.engine = engine.result()
        self.capacity = min(MAX_CONTEXT, self.engine.capacity)
        self.cache = Cache(self.engine, slots, max_snapshots=MODELS.get(model, MODELS[MODEL])["snapshots"])
        if warmup:
            self.engine.warmup()

    # ---- model switching ---------------------------------------------------------------------

    def resolve(self, name):
        """The MODELS key a request names ("-fast" variants included); unknown names mean the active model."""
        base = str(name or "").removesuffix("-fast")
        return base if base in MODELS else self.active

    def use(self, name):
        """Switch the GPU to another model (the caller holds the GPU). Cached conversations of the old model are
        dropped: they are that model's KV and states."""
        if name == self.active:
            return
        import os
        t0 = time.time()
        spec = MODELS[name]
        with self.state_lock:                         # waits for a stats() call that is reading the engine
            self.loading = name
        try:
            self.engine.shutdown()
            self.engine = Engine(os.path.join(_ROOT, spec["path"]), max_seqs=self.slots, umax=self.umax or spec["chunk"],
                                 lib=os.path.join(_ROOT, spec["lib"]), warmup=True)
            self.capacity = min(MAX_CONTEXT, self.engine.capacity)
            self.cache = Cache(self.engine, self.slots, max_snapshots=spec["snapshots"])
            self.n_draft = spec["draft"]
            print(f"kiln: switched {self.active} -> {name} in {time.time() - t0:.1f} s (context {self.capacity} tokens)",
                  file=self.log, flush=True)
            self.active = name
        finally:
            with self.state_lock:
                self.loading = None

    # ---- request translation ----------------------------------------------------------------

    def chat(self, body):
        """Runs one chat completion. Yields ("reasoning"|"content", text), ("tool_call", (name, args)),
        finally ("done", info)."""
        messages = merge_system(body.get("messages") or [])
        if not any(m.get("role") == "user" for m in messages):
            raise RequestError("the request needs at least one user message")
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
        prompt, marks = self.renderer.render(messages, tools, *style)
        d = DEFAULTS[bool(thinking)]
        sp = Sampling(temperature=float(body.get("temperature", d["temperature"])), top_p=float(body.get("top_p", d["top_p"])),
                      min_p=float(body.get("min_p", d["min_p"])), presence_penalty=float(body.get("presence_penalty", d["presence_penalty"])),
                      top_k=int(body.get("top_k", d["top_k"])), n_draft=self.n_draft,
                      # A model can call only the tools it is offered. Without any, Flash-Next still writes tool calls for
                      # agent-like asks ("design an SVG": list_files, read_file; the Q8_0 original as well, p ~0.94), and
                      # the reply ends there: so without tools <tool_call> is never drawn and it answers in text.
                      ban=-1 if tools else self.tok.id["<tool_call>"])
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or MAX_OUTPUT)
        if side:
            max_tokens = min(max_tokens, 256)
        if len(prompt) + 1 >= self.capacity:
            raise ContextError(f"the request ({len(prompt)} tokens) exceeds the available context size "
                               f"({self.capacity} tokens), try increasing it")
        max_tokens = min(max_tokens, self.capacity - len(prompt) - 1)
        schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools or [] if "function" in t}
        record = None if side or compacting else {"messages": messages, "tools": tools, "style": style}
        self.gpu.acquire(main=not side)
        try:
            want = self.active if side else self.resolve(body.get("model"))   # titles never cause a switch
            if want != self.active:
                self.use(want)
                sp.n_draft = self.n_draft
                if len(prompt) + 1 >= self.capacity:
                    raise ContextError(f"the request ({len(prompt)} tokens) exceeds the available context size "
                                       f"({self.capacity} tokens), try increasing it")
            yield from self.run(np.array(prompt, np.int32), marks, sp, seed, max_tokens, bool(thinking), schemas, record, side)
        finally:
            self.gpu.release(main=not side)

    def compaction(self, messages):
        """OpenCode compacts by sending the older part of the conversation serialized into one user message,
        an uncached prefill of up to ~200K tokens. Instead, append the summary request to the conversation it
        came from, rendered exactly as before, so all of it except the request is already cached."""
        user = text_of(messages[-1].get("content"))
        m = re.search(r"<conversation>\n(.*)\n</conversation>", user, re.S)
        if not m:
            return None

        def score(rec):   # how many of the conversation's user messages appear in the serialized history
            texts = (first_text(x.get("content")) for x in rec["messages"] if x.get("role") == "user")
            return sum("[User]: " + t in m.group(1) for t in texts if t)
        rec = max(reversed(self.sessions.values()), key=score, default=None)   # ties: most recent
        if rec is None or score(rec) == 0:
            return None
        request = (text_of(messages[0].get("content")) + "\n\nDo not call any tools.\n\n" + user[:m.start()]
                   + "<conversation>\n(the conversation above)\n</conversation>" + user[m.end():])
        return rec["messages"] + [rec["reply"], {"role": "user", "content": request}], rec["tools"], rec["style"]

    # ---- engine work --------------------------------------------------------------------------

    def prefill(self, slot, prompt, a, b):
        while a < b:
            try:
                self.engine.prefill(slot.sid, prompt[a:b].tolist())
                a = b
            except KVFull:
                a = self.engine.seq_len(slot.sid)
                slot.tokens = prompt[:a]
                if not self.free_kv(b - a + 64, keep=slot):
                    raise ContextError("the request exceeds the available context size (KV cache full), try increasing it")
            slot.tokens = prompt[:a]

    def free_kv(self, need, keep):
        """Drop snapshots, then other sessions' slots, until `need` tokens of KV are free."""
        self.cache.trim(need)
        for s in sorted(self.cache.slots, key=lambda s: s.used):
            if self.engine.free_tokens() >= need:
                break
            if s is not keep and len(s.tokens):
                self.engine.truncate(s.sid, 0)
                s.tokens = s.tokens[:0]
        return self.engine.free_tokens() >= need

    def run(self, prompt, marks, sp, seed, max_tokens, thinking, schemas, record, side=False):
        t0 = time.time()
        trace = [] if TRACE else None
        slot, cached = self.cache.acquire(prompt)
        if trace is not None:
            trace.append(("acquire", time.time() - t0))
        # Cache points: the prompt is prefilled in pieces that end at them, whichever state it starts from, so a seed
        # draws the same reply from a cold cache and from a snapshot. The last one is the prompt's end, kept with the
        # next token's distribution: the same prompt again (a retry, a redraw) needs no prefill at all, and an agent
        # step is one piece (its new messages and the reply's opening tokens together). A side request's system turn
        # (the title / summary instructions) is shared by every session, so it is kept like the tool block; side
        # prompts are never resent, so their end is not kept.
        points = [(marks["tools"], True), (marks["static"], True), (marks["system"], side)]
        pos = cached
        for at, pinned in sorted(p for p in points if p[0] is not None and p[0] > cached) + [(len(prompt), False)]:
            if at <= pos:
                continue
            t = time.time()
            self.prefill(slot, prompt, pos, at)
            if trace is not None:
                trace.append((f"prefill {at - pos}", time.time() - t))
            pos = at
            if at < len(prompt) or not side:
                t = time.time()
                self.cache.checkpoint(slot, pinned, logits=at == len(prompt))
                if trace is not None:
                    trace.append(("checkpoint", time.time() - t))
        t_prefill = time.time()
        self.engine.begin(slot.sid, seed)
        parser = OutputParser(self.tok, thinking, schemas)
        out, finish, steps, completed = [], "stop", 0, False
        try:
            while True:
                try:
                    ids = self.engine.generate(slot.sid, sp, 16)
                except KVFull:
                    if self.free_kv(64, keep=slot):
                        continue
                    finish = "length"
                    break
                if steps == 0:
                    t_first = time.time()
                    if trace is not None:
                        trace.append(("first step", t_first - t_prefill))
                steps += 1
                stop = next((i for i, t in enumerate(ids) if t in self.tok.eos), None)
                n, room = len(ids) if stop is None else stop, max_tokens - len(out)   # reply tokens in this step; allowed
                out += ids
                for ev in parser.feed(ids[:min(n, room)]):
                    yield ev
                if n > room or n == room and stop is None:   # cut at max_tokens (a step can carry several tokens)
                    finish = "length"
                    break
                if stop is not None:
                    break
            for ev in parser.finish():
                yield ev
            completed = True
        finally:
            # Also runs when the client goes away mid-stream (OpenCode's Esc, a Stop button). The engine holds
            # prompt + out[:-1] (the last emitted token is still pending) and the cache must agree, or the next
            # request would be appended after the abandoned reply.
            slot.tokens = np.concatenate([prompt, np.array(out[:-1], np.int32)]) if out else prompt
            reasoning, content = parser.text()
            calls = parser.calls
            reply = {"role": "assistant", "content": content, "reasoning_content": reasoning,
                     "tool_calls": [{"type": "function", "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}} for n, a in calls]}
            if content or calls:                         # an empty reply's key would match every other one
                gen = (out[:-1] if out and out[-1] in self.tok.eos else out)[:max_tokens]   # what the client was sent
                self.replay.put(message_key(reply, False), (thinking, gen))   # for clients that drop the reasoning
                self.replay.put(message_key(reply), (thinking, gen))
            if record:                                   # the conversation's latest state, for compaction
                record["reply"] = reply
                key = next(text_of(m.get("content")) for m in record["messages"] if m.get("role") == "user")
                self.sessions.put(hashlib.sha1(key.encode()).hexdigest(), record)
            if not completed:
                print(f"kiln: prompt {len(prompt)} (cached {cached}) | stopped by the client after {len(out)} tokens "
                      f"| slot {slot.sid}", file=self.log, flush=True)
        if finish == "stop" and calls:
            finish = "tool_calls"
        t1 = time.time()
        n_gen = len(out)
        info = {"prompt_tokens": len(prompt), "cached_tokens": cached, "completion_tokens": min(n_gen, max_tokens),
                "finish_reason": finish, "ttft": (t_first if steps else t1) - t0, "prefill_s": t_prefill - t0,
                "decode_tps": n_gen / max(t1 - t_prefill, 1e-6), "tokens_per_step": n_gen / max(steps, 1), "seed": seed}
        self.recent.put(time.time(), info)
        print(f"kiln: prompt {len(prompt)} (cached {cached}) ttft {info['ttft']:.2f}s | gen {n_gen} "
              f"{info['decode_tps']:.1f} tok/s {info['tokens_per_step']:.2f}/step | {finish} | slot {slot.sid}"
              + (" | " + ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in trace) if trace else ""),
              file=self.log, flush=True)
        yield "done", info

    def stats(self):
        with self.state_lock:
            if self.loading:                          # the old engine is gone, the new one not ready
                return {"model": self.active, "loading": self.loading, "progress": load_progress(), "models": list(MODELS)}
            return {"model": self.active, "models": list(MODELS), "context": self.capacity, "kv_tokens": self.engine.capacity,
                    "cache": self.cache.stats(), "recent": list(self.recent.values())[-16:]}

    def load(self, name):
        """Switch to a model now (the web chat's model picker), after the request on the GPU, if any."""
        if name not in MODELS:
            raise RequestError(f"unknown model {name!r}; kiln has {', '.join(MODELS)}")
        self.gpu.acquire(main=True)
        try:
            self.use(name)
        finally:
            self.gpu.release()


# ---- HTTP ------------------------------------------------------------------------------------------

# DNS rebinding guard: a web page from another site that points its own name at this machine still sends that name
# as Host. Requests addressed to an IP address or to this machine's names are fine.
NAMES = {"", "localhost", socket.gethostname().lower(), socket.gethostname().lower() + ".local"}


def host_ok(host):
    name = (host or "").strip().lower()
    if name.startswith("["):                             # IPv6 literal
        return True
    name = name.rsplit(":", 1)[0]
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return name in NAMES


def make_handler(state, ready):
    """state["srv"] is the Server once `ready` is set; until then /health and /kiln/stats answer 503 and the other
    requests wait."""
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "kiln"

        def log_message(self, *args):
            pass

        def send_json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not host_ok(self.headers.get("Host")):
                return self.send_json(403, {"error": {"message": "unknown host", "type": "invalid_request_error"}})
            if not ready.is_set() and self.path.rstrip("/") in ("/health", "/v1/health", "/kiln/stats"):
                return self.send_json(503, {"status": "loading", "model": state.get("loading"), "progress": load_progress()})
            ready.wait()
            srv = state["srv"]
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                now = int(time.time())
                self.send_json(200, {"object": "list", "data": [
                    {"id": m, "object": "model", "created": now, "owned_by": "kiln", "context_length": MAX_CONTEXT}
                    for name in MODELS for m in (name, name + "-fast")]})
            elif self.path.rstrip("/") in ("/health", "/v1/health"):
                self.send_json(200, {"status": "ok"})
            elif self.path.rstrip("/") == "/kiln/stats":
                self.send_json(200, srv.stats())
            else:
                self.send_json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not host_ok(self.headers.get("Host")):
                return self.send_json(403, {"error": {"message": "unknown host", "type": "invalid_request_error"}})
            if self.path.rstrip("/") == "/kiln/load":
                ready.wait()
                srv = state["srv"]
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                    srv.load(str(body.get("model", "")))
                except (json.JSONDecodeError, RequestError) as e:
                    return self.send_json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
                return self.send_json(200, {"model": srv.active, "context": srv.capacity})
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self.send_json(404, {"error": {"message": "not found"}})
            ready.wait()
            srv = state["srv"]
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError:
                return self.send_json(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
            rid, created, model = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()), body.get("model") or srv.active
            gen = srv.chat(body)
            try:
                first = next(gen)                         # surfaces request errors before any bytes are sent
            except RequestError as e:
                return self.send_json(400, {"error": {"message": str(e), "type": "invalid_request_error", "code": e.code}})
            except Exception as e:
                print(f"kiln: request failed: {e!r}", file=srv.log, flush=True)
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
                kind, val = ev
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
                self.chunk({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
                ev = next(gen)

        def complete(self, gen, first, rid, created, model):
            reasoning, content, calls, ev = [], [], [], first
            while ev[0] != "done":
                kind, val = ev
                if kind == "tool_call":
                    calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                                  "function": {"name": val[0], "arguments": json.dumps(val[1], ensure_ascii=False)}})
                else:
                    (reasoning if kind == "reasoning" else content).append(val)
                ev = next(gen)
            info = ev[1]
            msg = {"role": "assistant", "content": "".join(content).strip() or (None if calls else "")}
            if reasoning:
                msg["reasoning_content"] = "".join(reasoning).strip()
            if calls:
                msg["tool_calls"] = calls
            self.send_json(200, {"id": rid, "object": "chat.completion", "created": created, "model": model, "seed": info["seed"],
                                 "choices": [{"index": 0, "message": msg, "finish_reason": info["finish_reason"]}],
                                 "usage": usage_of(info)})

    return Handler


def seed_of(body):
    """The request's seed, or a random one. Either way the response carries it (top-level `seed`; when streaming,
    in the finish_reason chunk), so any reply can be drawn again (README: Decode)."""
    seed = body.get("seed")
    if seed is None:
        return random.getrandbits(32)                # fits a JavaScript number exactly
    if isinstance(seed, float) and seed.is_integer():
        seed = int(seed)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise RequestError("seed must be an integer")
    return seed % 2**64                              # negative seeds wrap, as in a C uint64


def usage_of(info):
    return {"prompt_tokens": info["prompt_tokens"], "completion_tokens": info["completion_tokens"],
            "total_tokens": info["prompt_tokens"] + info["completion_tokens"],
            "prompt_tokens_details": {"cached_tokens": info["cached_tokens"]}}


class HTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        # The stock version also looks up the bind address's fully qualified name, which it never uses: a reverse
        # DNS query that runs into a timeout (4.5 s) for 0.0.0.0.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address):
        # A client that hangs up (curl -f on a 503 while the model loads, a closed tab) is not worth a traceback.
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


def serve(make_server, host="127.0.0.1", port=8080, model=None):
    """Listens right away, so clients connect while the model loads: until make_server() returns, /health answers
    503 (with the load's progress) and requests wait. The engine's warm-up then runs holding the GPU lock; requests
    queue behind it."""
    state, ready = {"loading": model}, threading.Event()
    httpd = HTTPServer((host, port), make_handler(state, ready))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    srv = make_server()
    srv.gpu.acquire(main=True)
    state["srv"] = srv
    ready.set()
    print(f"kiln: serving {srv.active} on http://{host}:{port}/v1 (context {srv.capacity} tokens)", file=srv.log, flush=True)
    srv.engine.warmup()
    srv.gpu.release()
    threading.Event().wait()
