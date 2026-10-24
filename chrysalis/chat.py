import functools
import json
import re
from typing import NamedTuple

from tokenizers import Tokenizer

from .vision import LOW_DETAIL

EFFORT = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
             "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
    "medium": "",
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
           "without unnecessary elaboration.",
}
TOOL_FORMAT = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")
STRUCT = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
          "<tool_response>", "</tool_response>", "<|vision_start|>", "<|vision_end|>"]
NO_VISION = "[image omitted: this model server has no vision tower loaded]"
NO_VIDEO = "[video omitted: this model server has no vision tower loaded]"


VOLATILE = re.compile(r"^(Today's date:|You are powered by the model|Here is some useful information about the environment|<env>)", re.M)


def split_static(text):
    m = VOLATILE.search(text)
    return (text[:m.start()], text[m.start():]) if m and m.start() > 0 else (text, "")


def tojson(x):
    return json.dumps(x, ensure_ascii=False)


class Content(str):
    pass


class Tokens:
    def __init__(self, path):
        self.full = Tokenizer.from_file(path)
        self.added = [t.content for _, t in sorted(self.full.get_added_tokens_decoder().items())]
        self.id = {s: self.full.token_to_id(s) for s in STRUCT}
        self.eos = {self.id["<|im_end|>"], self.id["<|endoftext|>"]}

    def encode_run(self, run):
        return self._encode_run(tuple((isinstance(p, Content), p) for p in run))

    @functools.lru_cache(maxsize=4096)
    def _encode_run(self, run):
        text = "".join(p for _, p in run)
        cuts, pos = [], 0
        for is_content, piece in run:
            if is_content:
                for tok in self.added:
                    i = piece.find(tok)
                    while i >= 0:
                        cuts.append(pos + i + 1)
                        i = piece.find(tok, i + 1)
            pos += len(piece)
        out, prev = [], 0
        for c in sorted(set(cuts)) + [len(text)]:
            if c > prev:
                out += self.full.encode(text[prev:c], add_special_tokens=False).ids
            prev = c
        return tuple(out)

    def encode(self, pieces, marks=()):
        ids, offsets, run, marks = [], [], [], sorted(marks)
        mi = 0
        for idx, p in enumerate(list(pieces) + [None]):
            if mi < len(marks) and marks[mi] == idx or p is None or isinstance(p, int):
                if run:
                    ids += self.encode_run(run)
                    run = []
            while mi < len(marks) and marks[mi] == idx:
                offsets.append(len(ids))
                mi += 1
            if p is None:
                break
            if isinstance(p, int):
                ids.append(p)
            else:
                run.append(p)
        return ids, offsets

    def decode(self, ids):
        return self.full.decode(ids, skip_special_tokens=False)


class Picture(NamedTuple):
    url: str
    max_pixels: int | float | None = None


class Clip(NamedTuple):
    url: str
    fps: int | float | None = None


def parts(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [("text", content)]
    out = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in ("image_url", "image") or "image_url" in item:
            url = item.get("image_url", item.get("image"))
            spec = url if isinstance(url, dict) else {}
            limit = item.get("max_pixels", spec.get("max_pixels"))
            if limit is None and (spec.get("detail") or item.get("detail")) == "low":
                limit = LOW_DETAIL
            out.append(("image", Picture(spec.get("url") if isinstance(url, dict) else url, limit)))
        elif item.get("type") in ("video_url", "video") or "video_url" in item:
            url = item.get("video_url", item.get("video"))
            spec = url if isinstance(url, dict) else {}
            out.append(("video", Clip(spec.get("url") if isinstance(url, dict) else url, item.get("fps", spec.get("fps")))))
        elif "text" in item:
            out.append(("text", item.get("text") or ""))
    return out


def text_of(content):
    return "".join(v for k, v in parts(content) if k == "text")


def call_args(tc):
    fn = tc.get("function", tc)
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {"arguments": args}
    return fn.get("name", ""), args if isinstance(args, dict) else {"arguments": args}


def message_key(msg, reasoning=True):
    calls = [(n, json.dumps(a, sort_keys=True, ensure_ascii=False)) for n, a in map(call_args, msg.get("tool_calls") or [])]
    r = msg.get("reasoning_content")
    r = r.strip() if reasoning and isinstance(r, str) else ""
    return json.dumps([text_of(msg.get("content")).strip(), calls] + ([r] if r else []), ensure_ascii=False)


class Renderer:
    def __init__(self, tok: Tokens, replay, images=None):
        self.t, self.replay = tok, replay
        self.images = images

    def content(self, content, images=None):
        segs, images = [], images or self.images
        vs, ve = self.t.id["<|vision_start|>"], self.t.id["<|vision_end|>"]
        for kind, val in parts(content):
            if kind in ("image", "video"):
                got = images(val) if images else None
                if got is not None and kind == "image":
                    segs.append([vs, *got, ve])
                    continue
                if got is not None:
                    segs.append([vs] + [x for t, ids in got for x in self.t.full.encode(f"<{t:.1f} seconds>", add_special_tokens=False).ids
                                        + [vs, *ids, ve]] + [ve])
                    continue
                val = NO_VISION if kind == "image" else NO_VIDEO
            if segs and isinstance(segs[-1], str):
                segs[-1] += val
            else:
                segs.append(val)
        if segs and isinstance(segs[0], str):
            segs[0] = segs[0].lstrip()
        if segs and isinstance(segs[-1], str):
            segs[-1] = segs[-1].rstrip()
        return [x for s in segs for x in ([Content(s)] if isinstance(s, str) else s)]

    def render(self, messages, tools=None, enable_thinking=True, effort="xhigh", preserve_thinking=True, think_reply=None,
               images=None):
        T, S = self.t, self.t.id
        ri = EFFORT.get(effort, EFFORT["xhigh"]) if enable_thinking else ""
        pieces, marks = [], []
        start = 0
        if messages and messages[0].get("role") == "system":
            kinds = {k for k, _ in parts(messages[0].get("content"))}
            if "image" in kinds:
                raise ValueError("System message cannot contain images.")
            if "video" in kinds:
                raise ValueError("System message cannot contain videos.")
            sys_text, start = text_of(messages[0].get("content")).strip(), 1
        else:
            sys_text = ""
        if tools:
            pieces += [S["<|im_start|>"], "system\n" + (ri + "\n\n" if ri else "")
                       + "# Tools\n\nYou have access to the following functions:\n\n<tools>"
                       ] + [x for t in tools for x in ("\n", Content(tojson(t)))] + ["\n</tools>" + TOOL_FORMAT]
            marks.append(("tools", len(pieces)))
            if sys_text:
                head, tail = split_static(sys_text)
                pieces += ["\n\n", Content(head)]
                if tail:
                    marks.append(("static", len(pieces)))
                    pieces.append(Content(tail))
            pieces += [S["<|im_end|>"], "\n"]
        elif sys_text or ri:
            pieces += [S["<|im_start|>"], "system\n" + (ri + "\n\n" if ri and sys_text else ri), Content(sys_text),
                       S["<|im_end|>"], "\n"]
        marks.append(("system", len(pieces)))
        last_query = len(messages) - 1
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if m.get("role") == "user":
                c = text_of(m.get("content")).strip()
                if not (c.startswith("<tool_response>") and c.endswith("</tool_response>")):
                    last_query = i
                    break
        for i in range(start, len(messages)):
            m, role = messages[i], messages[i].get("role")
            content = text_of(m.get("content")).strip()
            if role == "system":
                raise ValueError("system message must be first")
            if role == "user":
                marks.append(("messages", len(pieces)))
                pieces += [S["<|im_start|>"], "user\n", *self.content(m.get("content"), images), S["<|im_end|>"], "\n"]
            elif role == "assistant":
                marks.append(("messages", len(pieces)))
                hit = self.replay(message_key(m))
                if hit is None:
                    hit = self.replay(message_key(m, False))
                if hit is not None:
                    thinking, ids = hit
                    pieces += [S["<|im_start|>"], "assistant\n", S["<think>"]]
                    pieces += ["\n"] if thinking else ["\n\n", S["</think>"], "\n\n"]
                    pieces += list(ids) + [S["<|im_end|>"], "\n"]
                    continue
                reasoning = m.get("reasoning_content")
                reasoning = reasoning.strip() if isinstance(reasoning, str) else ""
                pieces += [S["<|im_start|>"], "assistant\n"]
                if preserve_thinking or i > last_query:
                    pieces += [S["<think>"], "\n", Content(reasoning), "\n", S["</think>"], "\n\n"]
                pieces.append(Content(content))
                for j, tc in enumerate(m.get("tool_calls") or []):
                    name, args = call_args(tc)
                    pieces += [("\n\n" if content else "") if j == 0 else "\n", S["<tool_call>"], "\n<function=", Content(name), ">\n"]
                    for k, v in args.items():
                        pieces += ["<parameter=", Content(k), ">\n", Content(v if isinstance(v, str) else tojson(v)), "\n</parameter>\n"]
                    pieces += ["</function>\n", S["</tool_call>"]]
                pieces += [S["<|im_end|>"], "\n"]
            elif role == "tool":
                if i == 0 or messages[i - 1].get("role") != "tool":
                    marks.append(("messages", len(pieces)))
                    pieces += [S["<|im_start|>"], "user"]
                pieces += ["\n", S["<tool_response>"], "\n", *self.content(m.get("content"), images), "\n", S["</tool_response>"]]
                if i == len(messages) - 1 or messages[i + 1].get("role") != "tool":
                    pieces += [S["<|im_end|>"], "\n"]
            else:
                raise ValueError(f"unexpected role {role!r}")
        marks.append(("prompt", len(pieces)))
        pieces += [S["<|im_start|>"], "assistant\n", S["<think>"]]
        pieces += ["\n"] if (enable_thinking if think_reply is None else think_reply) else ["\n\n", S["</think>"], "\n\n"]
        ids, offsets = T.encode(pieces, [i for _, i in marks])
        out = {"tools": None, "static": None, "messages": []}
        for (name, _), off in zip(marks, offsets):
            if name == "messages":
                out["messages"].append(off)
            else:
                out[name] = off
        return ids, out


PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>\s*(?=<parameter=|</function>|$)", re.S)


def parse_call(body, schemas):
    m = re.search(r"<function=([^>\n]+)>", body)
    if not m:
        return None
    name = m.group(1).strip()
    props = ((schemas.get(name) or {}).get("properties")) or {}
    args = {}
    for k, v in PARAM.findall(body[m.end():]):
        k = k.strip()
        typ = (props.get(k) or {}).get("type")
        if typ == "string" or (typ is None and not v.strip()[:1] in ('{', '[')):
            args[k] = v
        else:
            try:
                args[k] = json.loads(v)
            except json.JSONDecodeError:
                args[k] = v
    return name, args


class OutputParser:

    def __init__(self, tok: Tokens, thinking=True, schemas=None):
        self.t, self.s = tok, tok.id
        self.state = "reasoning" if thinking else "content"
        self.pending = []
        self.pending_lp = []
        self.call = None
        self.calls = []
        self.schemas = schemas or {}
        self.content_started = False
        self.reasoning, self.content = [], []
        self.after_think = False

    def _flush(self, out):
        if not self.pending:
            return
        text = self.t.decode(self.pending)
        if text.endswith("�"):
            return
        lp, self.pending, self.pending_lp = self.pending_lp, [], []
        if self.state == "content" and not self.content_started:
            text = text.lstrip("\n")
            if not text:
                return
            self.content_started = True
        if self.state not in ("reasoning", "content"):
            return
        (self.reasoning if self.state == "reasoning" else self.content).append(text)
        out.append((self.state, text, {"token": text, "logprob": sum(x[0] for x in lp), "top_logprobs": lp[0][1]} if lp else None))

    def feed(self, ids, lps=None):
        out = []
        for n, i in enumerate(ids):
            if self.call is not None:
                if i == self.s["</tool_call>"]:
                    parsed = parse_call(self.t.decode(self.call), self.schemas)
                    self.call = None
                    if parsed:
                        self.calls.append(parsed)
                        out.append(("tool_call", parsed))
                else:
                    self.call.append(i)
                continue
            if i == self.s["</think>"] and self.state == "reasoning":
                self._flush(out)
                self.pending, self.state = [], "content"
            elif i == self.s["<tool_call>"]:
                self._flush(out)
                self.pending, self.state, self.call = [], "content", []
            elif i == self.s["<think>"]:
                continue
            else:
                self.pending.append(i)
                if lps is not None:
                    self.pending_lp.append(lps[n])
                self._flush(out)
        return out

    def raw(self):
        return "".join(self.reasoning), "".join(self.content)

    def finish(self):
        out = []
        self._flush(out)
        if self.pending:
            self.pending.append(self.s["<|endoftext|>"])
            text = self.t.decode(self.pending[:-1]).rstrip("\ufffd")
            self.pending = []
            if text:
                (self.reasoning if self.state == "reasoning" else self.content).append(text)
                out.append((self.state if self.state != "tool" else "content", text, None))
        return out

    def text(self):
        return "".join(self.reasoning).strip(), "".join(self.content).strip()


class StopText:

    def __init__(self, stops):
        self.stops = [s for s in stops if s]
        self.held = []
        self.hit = False
        self.matched = None
        self.content = []

    def feed(self, events):
        out = []
        for ev in events:
            if self.hit:
                break
            if ev[0] != "content":
                out += self.flush()
                out.append(ev)
                continue
            self.held.append(ev)
            text = "".join(e[1] for e in self.held)
            found = [(i + len(s), i, s) for s in self.stops if (i := text.find(s)) >= 0]
            if found:
                _, at, self.matched = min(found)
                out += self._take(at, split=True)
                self.held, self.hit = [], True
                break
            maybe = max((k for s in self.stops for k in range(1, min(len(s), len(text) + 1)) if text.endswith(s[:k])), default=0)
            out += self._take(len(text) - maybe, split=False)
        return out

    def flush(self):
        return self._take(sum(len(e[1]) for e in self.held), split=False)

    def _take(self, n, split):
        out = []
        while self.held and (len(self.held[0][1]) <= n or split and n > 0):
            ev = self.held.pop(0)
            if len(ev[1]) > n:
                ev = (ev[0], ev[1][:n], dict(ev[2], token=ev[1][:n]) if len(ev) > 2 and ev[2] else None)
            out.append(ev)
            self.content.append(ev[1])
            n -= len(ev[1])
        return out
