"""Qwen3.8 chat format: prompt rendering straight to token ids, and streaming output parsing.

Rendering follows the model's chat_template.jinja exactly, with two deliberate differences:
  * message content is tokenized without matching special tokens, so a file that contains
    "<|im_end|>" stays text instead of ending the turn;
  * assistant turns that kiln generated are replayed as the exact token ids it produced, so every
    request in an agent loop extends the cached sequence instead of diverging from it.
"""
import functools
import json
import re
from tokenizers import Tokenizer

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
          "<tool_response>", "</tool_response>"]


VOLATILE = re.compile(r"^(You are powered by the model|Here is some useful information about the environment|<env>)", re.M)


def split_static(text):
    """Split a system prompt before its per-session part (OpenCode's model/environment block), at a line
    start so the tokenization is unchanged. The head is identical across projects and days."""
    m = VOLATILE.search(text)
    return (text[:m.start()], text[m.start():]) if m and m.start() > 0 else (text, "")


def tojson(x):
    """Jinja `tojson` as transformers defines it for chat templates."""
    return json.dumps(x, ensure_ascii=False)


class Content(str):
    """Text that came from a message (as opposed to template literals)."""


class Tokens:
    def __init__(self, path):
        self.full = Tokenizer.from_file(path)
        self.added = [t.content for _, t in sorted(self.full.get_added_tokens_decoder().items())]
        self.id = {s: self.full.token_to_id(s) for s in STRUCT}
        self.eos = {self.id["<|im_end|>"], self.id["<|endoftext|>"]}

    def encode_run(self, run):
        """Tokenize adjacent text pieces like the reference tokenizer does, except that special-token strings
        inside Content are cut after their first character so they can never become control tokens.
        Memoized: every request re-sends the whole conversation, but only its new messages need tokenizing."""
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
        """pieces: text (str / Content) or token ids. Returns ids and the token offsets of `marks` (piece indices)."""
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


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
        elif isinstance(item, dict) and item.get("type") in ("image", "image_url", "video"):
            parts.append("[image omitted: this model server is text-only]")
    return "".join(parts)


def call_args(tc):
    """OpenAI tool call -> (name, ordered dict of arguments)."""
    fn = tc.get("function", tc)
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {"arguments": args}
    return fn.get("name", ""), args if isinstance(args, dict) else {"arguments": args}


def message_key(msg):
    """Identity of an assistant turn as the client echoes it back (ignores reasoning and formatting)."""
    calls = [(n, json.dumps(a, sort_keys=True, ensure_ascii=False)) for n, a in map(call_args, msg.get("tool_calls") or [])]
    return json.dumps([text_of(msg.get("content")).strip(), calls], ensure_ascii=False)


class Renderer:
    def __init__(self, tok: Tokens, replay):
        self.t, self.replay = tok, replay       # replay: message_key -> (thinking, generated ids without stop token)

    def render(self, messages, tools=None, enable_thinking=True, effort="xhigh", preserve_thinking=True):
        """Returns (token ids, marks): token offsets of cache points -- "tools" (end of the tool block, shared by
        every session; None without tools), "system" (end of the system turn), "messages" (start of each
        later turn) and "prompt" (start of the generation prompt)."""
        T, S = self.t, self.t.id
        ri = EFFORT.get(effort, EFFORT["xhigh"]) if enable_thinking else ""
        pieces, marks = [], []
        start = 0
        if messages and messages[0].get("role") == "system":
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
                pieces += [S["<|im_start|>"], "user\n", Content(content), S["<|im_end|>"], "\n"]
            elif role == "assistant":
                marks.append(("messages", len(pieces)))
                hit = self.replay(message_key(m))
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
                pieces += ["\n", S["<tool_response>"], "\n", Content(content), "\n", S["</tool_response>"]]
                if i == len(messages) - 1 or messages[i + 1].get("role") != "tool":
                    pieces += [S["<|im_end|>"], "\n"]
            else:
                raise ValueError(f"unexpected role {role!r}")
        marks.append(("prompt", len(pieces)))
        pieces += [S["<|im_start|>"], "assistant\n", S["<think>"]]
        pieces += ["\n\n", S["</think>"], "\n\n"] if not enable_thinking else ["\n"]
        ids, offsets = T.encode(pieces, [i for _, i in marks])
        out = {"tools": None, "static": None, "messages": []}
        for (name, _), off in zip(marks, offsets):
            if name == "messages":
                out["messages"].append(off)
            else:
                out[name] = off
        return ids, out


# ---- output parsing --------------------------------------------------------------------------------

PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>\s*(?=<parameter=|</function>|$)", re.S)


def parse_call(body, schemas):
    """'\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n' -> (name, args dict)."""
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
    """Turns generated token ids into reasoning / content text and tool calls, incrementally."""

    def __init__(self, tok: Tokens, thinking=True, schemas=None):
        self.t, self.s = tok, tok.id
        self.state = "reasoning" if thinking else "content"
        self.pending = []            # undecoded ids (incomplete UTF-8)
        self.call = None             # ids inside <tool_call> ... </tool_call>
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
        self.pending = []
        if self.state == "reasoning":
            self.reasoning.append(text)
            out.append(("reasoning", text))
        elif self.state == "content":
            if not self.content_started:
                text = text.lstrip("\n")
                if not text:
                    return
                self.content_started = True
            self.content.append(text)
            out.append(("content", text))

    def feed(self, ids):
        """Returns a list of events: ("reasoning", text), ("content", text), ("tool_call", (name, args))."""
        out = []
        for i in ids:
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
                self._flush(out)
        return out

    def finish(self):
        out = []
        self._flush(out)
        if self.pending:                          # bytes that never formed valid UTF-8: emit as-is
            self.pending.append(self.s["<|endoftext|>"])
            text = self.t.decode(self.pending[:-1]).rstrip("\ufffd")
            self.pending = []
            if text:
                (self.reasoning if self.state == "reasoning" else self.content).append(text)
                out.append((self.state if self.state != "tool" else "content", text))
        return out

    def text(self):
        return "".join(self.reasoning).strip(), "".join(self.content).strip()
