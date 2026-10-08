"""OpenAI-compatible HTTP server around the kiln engine, built for coding-agent loops (OpenCode).

What keeps time-to-first-token low:
  * prompts render to token ids; assistant turns kiln generated are replayed as their exact ids, so each
    agent step extends the cached sequence (only new tool output / user text is prefilled);
  * snapshots at the end of the tool block (shared by all sessions), the system turn and every prompt,
    so new sessions, edited turns and evicted sessions resume from cache instead of from scratch;
  * OpenCode prompts are kept byte-stable (date frozen per conversation, appended reminders kept),
    and its compaction request is answered from the live conversation instead of a full re-read.
"""
import hashlib
import json
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .cache import Cache
from .chat import OutputParser, Renderer, Tokens, message_key, text_of
from .engine import Engine, KVFull, Sampling

MODEL = "qwen3.8-27b"
MAX_CONTEXT = 262144
DEFAULTS = {True: dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),   # Qwen3.8 card
            False: dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5)}
SIDE_PROMPTS = ("You are a title generator", "Summarize what was done in this conversation")
COMPACT_PROMPT = "You are a context summarization agent"
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
    """One request on the GPU at a time; side requests (titles, summaries) wait while agent requests queue."""

    def __init__(self):
        self.cv = threading.Condition()
        self.busy, self.waiting = False, 0

    def acquire(self, main):
        with self.cv:
            self.waiting += main
            self.cv.wait_for(lambda: not self.busy and (main or self.waiting == 0))
            self.waiting -= main
            self.busy = True

    def release(self):
        with self.cv:
            self.busy = False
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
    def __init__(self, model_path, tokenizer_path, slots=3, n_draft=3, umax=2048, cache_dir=None, log=sys.stderr):
        self.engine = Engine(model_path, max_seqs=slots, umax=umax)
        self.capacity = min(MAX_CONTEXT, self.engine.capacity)
        self.tok = Tokens(tokenizer_path)
        self.replay = LRU(8192)
        self.renderer = Renderer(self.tok, self.replay.get)
        self.cache = Cache(self.engine, slots, disk=cache_dir)
        self.sticky = Sticky()
        self.sessions = LRU(64)          # latest request + reply per conversation, for compaction rewrites
        self.gpu = Turnstile()
        self.n_draft, self.log = n_draft, log
        self.recent = LRU(64)

    # ---- request translation ----------------------------------------------------------------

    def chat(self, body):
        """Runs one chat completion. Yields ("reasoning"|"content", text), ("tool_call", (name, args)),
        finally ("done", info)."""
        messages = merge_system(body.get("messages") or [])
        if not any(m.get("role") == "user" for m in messages):
            raise RequestError("the request needs at least one user message")
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
                      top_k=int(body.get("top_k", d["top_k"])), n_draft=self.n_draft, seed=int(body.get("seed", 0) or 0))
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 32768)
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
            yield from self.run(np.array(prompt, np.int32), marks, sp, max_tokens, bool(thinking), schemas, record)
        finally:
            self.gpu.release()

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

    def run(self, prompt, marks, sp, max_tokens, thinking, schemas, record):
        t0 = time.time()
        slot, cached = self.cache.acquire(prompt)
        points = [(marks["tools"], True), (marks["static"], True), (marks["system"], False), (marks["prompt"], False)]
        pos = cached
        for at, pinned in sorted(p for p in points if p[0] is not None and p[0] > cached) + [(len(prompt), False)]:
            if at <= pos:
                continue
            self.prefill(slot, prompt, pos, at)
            pos = at
            if at < len(prompt):
                self.cache.checkpoint(slot, pinned)
        t_prefill = time.time()
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
                steps += 1
                stop = next((i for i, t in enumerate(ids) if t in self.tok.eos), None)
                out += ids
                emit = ids if stop is None else ids[:stop]
                if len(out) > max_tokens:
                    emit = emit[:max(0, len(emit) - (len(out) - max_tokens))]
                for ev in parser.feed(emit):
                    yield ev
                if stop is not None:
                    break
                if len(out) >= max_tokens:
                    finish = "length"
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
                gen = out[:-1] if out and out[-1] in self.tok.eos else out[:max_tokens]
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
                "decode_tps": n_gen / max(t1 - t_prefill, 1e-6), "tokens_per_step": n_gen / max(steps, 1)}
        self.recent.put(time.time(), info)
        print(f"kiln: prompt {len(prompt)} (cached {cached}) ttft {info['ttft']:.2f}s | gen {n_gen} "
              f"{info['decode_tps']:.1f} tok/s {info['tokens_per_step']:.2f}/step | {finish} | slot {slot.sid}",
              file=self.log, flush=True)
        yield "done", info

    def stats(self):
        return {"model": MODEL, "context": self.capacity, "kv_tokens": self.engine.capacity, "cache": self.cache.stats(),
                "recent": list(self.recent.values())[-16:]}


# ---- HTTP ------------------------------------------------------------------------------------------

def make_handler(srv: Server):
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
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                now = int(time.time())
                self.send_json(200, {"object": "list", "data": [
                    {"id": m, "object": "model", "created": now, "owned_by": "kiln", "context_length": srv.capacity}
                    for m in (MODEL, MODEL + "-fast")]})
            elif self.path.rstrip("/") in ("/health", "/v1/health"):
                self.send_json(200, {"status": "ok"})
            elif self.path.rstrip("/") == "/kiln/stats":
                self.send_json(200, srv.stats())
            else:
                self.send_json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self.send_json(404, {"error": {"message": "not found"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError:
                return self.send_json(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
            rid, created, model = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()), body.get("model") or MODEL
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
                    self.chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": val["finish_reason"]}]})
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
            self.send_json(200, {"id": rid, "object": "chat.completion", "created": created, "model": model,
                                 "choices": [{"index": 0, "message": msg, "finish_reason": info["finish_reason"]}],
                                 "usage": usage_of(info)})

    return Handler


def usage_of(info):
    return {"prompt_tokens": info["prompt_tokens"], "completion_tokens": info["completion_tokens"],
            "total_tokens": info["prompt_tokens"] + info["completion_tokens"],
            "prompt_tokens_details": {"cached_tokens": info["cached_tokens"]}}


def serve(srv: Server, host="127.0.0.1", port=8080):
    httpd = ThreadingHTTPServer((host, port), make_handler(srv))
    httpd.daemon_threads = True
    print(f"kiln: serving {MODEL} on http://{host}:{port}/v1 (context {srv.capacity} tokens)", file=srv.log, flush=True)
    httpd.serve_forever()
