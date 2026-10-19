"""Two more APIs over the same chat completions: Anthropic's Messages (POST /v1/messages: Claude Code, the Anthropic SDKs)
and OpenAI's Responses (POST /v1/responses: Codex, the newer OpenAI SDKs). A request becomes a chat completions body for
Server.chat, and its events (reasoning and answer text, tool calls, the final info) become the API's response or its
stream of server-sent events. Bad requests raise ValueError (the server answers 400)."""
import json
import time
import uuid

# Anthropic's thinking budget as Chrysalis's reasoning effort: under 2,048 tokens "low", under 8,192 "medium", else
# "xhigh"
BUDGET_EFFORT = ((2048, "low"), (8192, "medium"))


def _id(prefix):
    return prefix + uuid.uuid4().hex[:24]


def _content(parts):
    """Chat content from parts: a string when they are all text (how tool results and plain messages read best)."""
    if all(p["type"] == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


# Anthropic Messages


def _image_url(source):
    """An Anthropic image source as an image_url (base64 as a data URL; a URL as it is, which Chrysalis refuses: it
    reads only pictures sent inline)."""
    if source.get("type") == "base64":
        return f"data:{source.get('media_type') or 'image/png'};base64,{source.get('data') or ''}"
    return source.get("url") or ""


def _blocks(content):
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])


def _parts(blocks, where):
    parts = []
    for b in blocks:
        kind = b.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": b.get("text") or ""})
        elif kind == "image":
            parts.append({"type": "image_url", "image_url": {"url": _image_url(b.get("source") or {})}})
        elif kind == "document":
            parts += _document(b)
        else:
            raise ValueError(f"Chrysalis cannot read a {kind} block in {where}")
    return parts


def _kept(d, *keys):
    """Chrysalis's own fields of a part that it passes through (chrysalis/documents.py page_pictures)."""
    return {k: d[k] for k in keys if k in d}


def _document(b):
    """An Anthropic document block as chat parts: a PDF (base64) as a file part chrysalis/documents.py reads, plain text
    or content blocks as text, with its title and context."""
    src, title = b.get("source") or {}, b.get("title") or "document"
    head = [{"type": "text", "text": f"{b['context']}\n"}] if b.get("context") else []
    if src.get("type") == "base64":
        return head + [{"type": "file", "file": {"file_data": f"data:{src.get('media_type') or 'application/pdf'};base64,{src.get('data') or ''}",
                                                 "filename": title, **_kept(b, "page_pictures")}}]
    if src.get("type") == "text":
        return head + [{"type": "text", "text": f'<document name="{title}">\n{src.get("data") or ""}\n</document>'}]
    if src.get("type") == "content":
        inner = _parts(_blocks(src.get("content")), "a document")
        return head + [{"type": "text", "text": f'<document name="{title}">\n'}] + inner + [{"type": "text", "text": "\n</document>"}]
    raise ValueError("Chrysalis reads documents sent inline (base64, text or content), not by URL or file id")


def messages_body(req):
    """An Anthropic Messages request as a chat completions body. A final assistant message is a prefill: the reply
    goes on from it (continue_final_message)."""
    msgs = []
    system = req.get("system")
    if system:
        msgs.append({"role": "system", "content": system if isinstance(system, str)
                     else "".join(b.get("text") or "" for b in system if b.get("type") == "text")})
    for m in req.get("messages") or []:
        role, blocks = m.get("role"), _blocks(m.get("content"))
        if role == "assistant":
            text, thinking, calls = [], [], []
            for b in blocks:
                kind = b.get("type")
                if kind == "text":
                    text.append(b.get("text") or "")
                elif kind == "thinking":
                    thinking.append(b.get("thinking") or "")
                elif kind == "tool_use":
                    calls.append({"id": b.get("id"), "type": "function",
                                  "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}})
                elif kind != "redacted_thinking":        # another model's encrypted reasoning: nothing Chrysalis can read
                    raise ValueError(f"Chrysalis cannot read a {kind} block in an assistant message")
            msg = {"role": "assistant", "content": "".join(text)}
            if thinking:
                msg["reasoning_content"] = "".join(thinking)
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
        elif role == "user":
            pending = []                                 # tool results become tool messages, in their place
            for b in blocks:
                if b.get("type") == "tool_result":
                    if pending:
                        msgs.append({"role": "user", "content": _content(pending)})
                        pending = []
                    msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                                 "content": _content(_parts(_blocks(b.get("content")), "a tool result"))})
                else:
                    pending += _parts([b], "a user message")
            if pending:
                msgs.append({"role": "user", "content": _content(pending)})
        else:
            raise ValueError(f"unknown role {role!r}")
    body = {"model": req.get("model"), "messages": msgs, "stream": bool(req.get("stream"))}
    if msgs and msgs[-1]["role"] == "assistant":
        body["continue_final_message"] = True
    for k in ("max_tokens", "temperature", "top_p", "top_k", "max_pixels"):
        if req.get(k) is not None:
            body[k] = req[k]
    if req.get("stop_sequences"):
        body["stop"] = req["stop_sequences"]
    tools = []
    for t in req.get("tools") or []:
        if t.get("type") not in (None, "custom"):
            raise ValueError(f"Chrysalis runs only the client's own tools, not Anthropic's {t.get('type')}")
        tools.append({"type": "function", "function": {"name": t.get("name"), "description": t.get("description") or "",
                                                       "parameters": t.get("input_schema") or {"type": "object"},
                                                       **({"strict": True} if t.get("strict") is True else {})}})
    if tools:
        body["tools"] = tools
    choice = req.get("tool_choice") or {}
    if choice.get("type") in ("none", "any"):
        body["tool_choice"] = {"none": "none", "any": "required"}[choice["type"]]
    elif choice.get("type") == "tool":
        body["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}}
    if choice.get("disable_parallel_tool_use"):
        body["parallel_tool_calls"] = False
    thinking = req.get("thinking") or {}
    if thinking.get("type") == "disabled":
        body["enable_thinking"] = False
    elif thinking.get("type") in ("enabled", "adaptive"):
        body["enable_thinking"] = True
        if thinking.get("budget_tokens"):
            body["reasoning_effort"] = next((e for n, e in BUDGET_EFFORT if thinking["budget_tokens"] < n), "xhigh")
            body["thinking_budget"] = thinking["budget_tokens"]
    return body


def _anthropic_usage(info, output_tokens):
    return {"input_tokens": info["prompt_tokens"] - info["cached_tokens"], "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": info["cached_tokens"], "output_tokens": output_tokens}


def _stop_reason(info):
    if info["finish_reason"] == "length":
        return "max_tokens"
    if info["finish_reason"] == "tool_calls":
        return "tool_use"
    return "stop_sequence" if info.get("stop_sequence") is not None else "end_turn"


def messages_stream(events, model):
    """(event name, data) for each server-sent event of a streamed Messages reply. events: Server.chat's."""
    msg_id, index, open_kind = _id("msg_"), -1, None   # the open content block's index and type

    def block_stop():
        return "content_block_stop", {"type": "content_block_stop", "index": index}

    for ev in events:
        kind = ev[0]
        if kind == "start":
            yield "message_start", {"type": "message_start", "message": {
                "id": msg_id, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None,
                "stop_sequence": None, "usage": _anthropic_usage(ev[1], 0)}}
        elif kind in ("reasoning", "content"):
            want = "thinking" if kind == "reasoning" else "text"
            if open_kind != want:
                if open_kind:
                    yield block_stop()
                index, open_kind = index + 1, want
                yield "content_block_start", {"type": "content_block_start", "index": index, "content_block":
                                              {"type": "thinking", "thinking": "", "signature": ""} if want == "thinking"
                                              else {"type": "text", "text": ""}}
            delta = {"type": "thinking_delta", "thinking": ev[1]} if want == "thinking" else {"type": "text_delta", "text": ev[1]}
            yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}
        elif kind == "tool_call":
            if open_kind:
                yield block_stop()
            index, open_kind = index + 1, None
            name, args = ev[1]
            yield "content_block_start", {"type": "content_block_start", "index": index, "content_block":
                                          {"type": "tool_use", "id": _id("toolu_"), "name": name, "input": {}}}
            yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta":
                                          {"type": "input_json_delta", "partial_json": json.dumps(args, ensure_ascii=False)}}
            yield block_stop()
        elif kind == "done":
            if open_kind:
                yield block_stop()
            info = ev[1]
            yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": _stop_reason(info), "stop_sequence": info.get("stop_sequence")},
                                    "usage": {"output_tokens": info["completion_tokens"]}}
            yield "message_stop", {"type": "message_stop"}


def messages_result(events, model):
    """A Messages reply in one piece."""
    content, info = [], None
    for ev in events:
        kind = ev[0]
        if kind in ("reasoning", "content"):
            want = "thinking" if kind == "reasoning" else "text"
            if not content or content[-1]["type"] != want:
                content.append({"type": "thinking", "thinking": "", "signature": ""} if want == "thinking" else {"type": "text", "text": ""})
            content[-1][want] += ev[1]
        elif kind == "tool_call":
            content.append({"type": "tool_use", "id": _id("toolu_"), "name": ev[1][0], "input": ev[1][1]})
        elif kind == "done":
            info = ev[1]
    for b in content:                                    # as the chat completions reply: no blank lines around a part
        if b["type"] in ("thinking", "text"):
            b[b["type"]] = b[b["type"]].strip()
    content = [b for b in content if b["type"] == "tool_use" or b[b["type"]]]
    return {"id": _id("msg_"), "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": _stop_reason(info), "stop_sequence": info.get("stop_sequence"),
            "usage": _anthropic_usage(info, info["completion_tokens"])}


# OpenAI Responses


def _input_parts(content, role):
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        kind = p.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": p.get("text") or ""})
        elif kind == "refusal":
            parts.append({"type": "text", "text": p.get("refusal") or ""})
        elif kind == "input_file":
            if not p.get("file_data"):
                raise ValueError("Chrysalis reads files only as file_data (it has no file store)")
            parts.append({"type": "file", "file": {"file_data": p["file_data"], "filename": p.get("filename"), **_kept(p, "page_pictures")}})
        elif kind == "input_image":
            url = p.get("image_url")
            if not url:
                raise ValueError("Chrysalis reads pictures only as image_url data URLs (it has no file store)")
            parts.append({"type": "image_url", "image_url": {"url": url if isinstance(url, str) else url.get("url"),
                                                             "detail": p.get("detail")}})
        else:
            raise ValueError(f"Chrysalis cannot read {kind} content in a {role} message")
    return _content(parts)


def responses_body(req, history=None):
    """A Responses request as a chat completions body. history: the stored conversation previous_response_id names
    (its instructions are not carried over, as the API says)."""
    msgs = [dict(m) for m in history or []]
    if req.get("instructions"):
        msgs.insert(0, {"role": "system", "content": req["instructions"]})
    items = req.get("input")
    items = [{"role": "user", "content": items}] if isinstance(items, str) else items or []
    turn = None                                          # the assistant message the latest output items make up

    def assistant(fresh=False):
        nonlocal turn
        if turn is None or fresh:
            turn = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}
            msgs.append(turn)
        return turn

    for it in items:
        kind = it.get("type") or "message"
        if kind == "message":
            role = it.get("role")
            if role == "assistant":
                assistant()["content"] += _content([{"type": "text", "text": p if isinstance(p, str) else p.get("text") or p.get("refusal") or ""}
                                                    for p in ([it.get("content")] if isinstance(it.get("content"), str) else it.get("content") or [])])
            elif role in ("user", "system", "developer"):
                turn = None
                msgs.append({"role": "system" if role == "developer" else role, "content": _input_parts(it.get("content"), role)})
            else:
                raise ValueError(f"unknown role {role!r}")
        elif kind == "reasoning":                        # starts a turn unless it opens the one being made
            t = assistant(fresh=turn is not None and bool(turn["content"] or turn["tool_calls"]))
            t["reasoning_content"] += "".join(c.get("text") or "" for c in it.get("content") or []) or \
                "".join(s.get("text") or "" for s in it.get("summary") or [])
        elif kind == "function_call":
            assistant()["tool_calls"].append({"id": it.get("call_id"), "type": "function",
                                              "function": {"name": it.get("name"), "arguments": it.get("arguments") or "{}"}})
        elif kind == "function_call_output":
            turn = None
            out = it.get("output")
            msgs.append({"role": "tool", "tool_call_id": it.get("call_id"),
                         "content": out if isinstance(out, str) else _input_parts(out, "tool")})
        else:
            raise ValueError(f"Chrysalis cannot take {kind} items (it keeps no item store)")
    for m in msgs:
        if m["role"] == "assistant":
            for k in ("reasoning_content", "tool_calls"):
                if not m.get(k):
                    m.pop(k, None)
    body = {"model": req.get("model"), "messages": msgs, "stream": bool(req.get("stream"))}
    if req.get("max_output_tokens") is not None:
        body["max_tokens"] = req["max_output_tokens"]
    for k in ("temperature", "top_p", "seed", "max_pixels"):
        if req.get(k) is not None:
            body[k] = req[k]
    tools = []
    for t in req.get("tools") or []:
        if t.get("type") != "function":
            raise ValueError(f"Chrysalis runs only function tools, not {t.get('type')}")
        tools.append({"type": "function", "function": {"name": t.get("name"), "description": t.get("description") or "",
                                                       "parameters": t.get("parameters") or {"type": "object"},
                                                       **({"strict": True} if t.get("strict") is True else {})}})
    if tools:
        body["tools"] = tools
    choice = req.get("tool_choice")
    if choice in ("none", "required"):
        body["tool_choice"] = choice
    elif isinstance(choice, dict) and choice.get("type") == "function":
        body["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}}
    if req.get("parallel_tool_calls") is False:
        body["parallel_tool_calls"] = False
    effort = (req.get("reasoning") or {}).get("effort")
    if effort == "none":
        body["enable_thinking"] = False
    elif effort:
        body["reasoning_effort"] = effort
    fmt = (req.get("text") or {}).get("format") or {}
    if fmt.get("type") == "json_schema":
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": fmt.get("name"), "schema": fmt.get("schema")}}
    elif fmt.get("type") == "json_object":
        body["response_format"] = {"type": "json_object"}
    return body


class ResponseBuilder:
    """A Responses reply from Server.chat's events: its output items as they grow, and the response object."""

    def __init__(self, req, model, count_tokens):
        self.req, self.model, self.count = req, model, count_tokens
        self.id, self.created = _id("resp_"), int(time.time())
        self.output, self.seq = [], 0
        self.reply = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}   # for the store

    def response(self, status, info=None):
        reasoning = self.reply["reasoning_content"]
        r = {"id": self.id, "object": "response", "created_at": self.created, "status": status, "error": None,
             "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
             "instructions": self.req.get("instructions"), "max_output_tokens": self.req.get("max_output_tokens"),
             "model": self.model, "output": self.output, "parallel_tool_calls": True,
             "previous_response_id": self.req.get("previous_response_id"),
             "reasoning": {"effort": (self.req.get("reasoning") or {}).get("effort"), "summary": None},
             "store": self.req.get("store", True), "temperature": self.req.get("temperature"),
             "text": {"format": (self.req.get("text") or {}).get("format") or {"type": "text"}},
             "tool_choice": self.req.get("tool_choice") or "auto", "tools": self.req.get("tools") or [],
             "top_p": self.req.get("top_p"), "truncation": "disabled", "metadata": self.req.get("metadata") or {}, "usage": None}
        if info:
            r["usage"] = {"input_tokens": info["prompt_tokens"],
                          "input_tokens_details": {"cached_tokens": info["cached_tokens"], "cache_write_tokens": 0},
                          "output_tokens": info["completion_tokens"],
                          "output_tokens_details": {"reasoning_tokens": min(self.count(reasoning), info["completion_tokens"]) if reasoning else 0},
                          "total_tokens": info["prompt_tokens"] + info["completion_tokens"]}
        return r

    def event(self, kind, **data):
        self.seq += 1
        return kind, {"type": kind, "sequence_number": self.seq - 1, **data}

    def events(self, events):
        """(event name, data) for each server-sent event; the last is response.completed (or .incomplete)."""
        yield self.event("response.created", response=self.response("in_progress"))
        yield self.event("response.in_progress", response=self.response("in_progress"))
        item, text = None, []                            # the open reasoning or message item and its text so far

        def close():
            nonlocal item
            if item is None:
                return []
            i, full = len(self.output) - 1, "".join(text)
            if item["type"] == "reasoning":
                part = {"type": "reasoning_text", "text": full}
                done = [self.event("response.reasoning_text.done", item_id=item["id"], output_index=i, content_index=0, text=full)]
            else:
                part = {"type": "output_text", "text": full, "annotations": [], "logprobs": []}
                done = [self.event("response.output_text.done", item_id=item["id"], output_index=i, content_index=0, text=full, logprobs=[])]
            item.update(content=[part], status="completed")
            done += [self.event("response.content_part.done", item_id=item["id"], output_index=i, content_index=0, part=part),
                     self.event("response.output_item.done", output_index=i, item=item)]
            item = None
            return done

        for ev in events:
            kind = ev[0]
            if kind in ("reasoning", "content"):
                want = "reasoning" if kind == "reasoning" else "message"
                if item is None or item["type"] != want:
                    yield from close()
                    item, text = ({"id": _id("rs_"), "type": "reasoning", "summary": [], "content": [], "status": "in_progress"}
                                  if want == "reasoning" else
                                  {"id": _id("msg_"), "type": "message", "role": "assistant", "status": "in_progress", "content": []}), []
                    self.output.append(item)
                    i = len(self.output) - 1
                    yield self.event("response.output_item.added", output_index=i, item=dict(item, content=[]))
                    yield self.event("response.content_part.added", item_id=item["id"], output_index=i, content_index=0,
                                     part={"type": "reasoning_text", "text": ""} if want == "reasoning"
                                     else {"type": "output_text", "text": "", "annotations": [], "logprobs": []})
                text.append(ev[1])
                self.reply["reasoning_content" if want == "reasoning" else "content"] += ev[1]
                i = len(self.output) - 1
                if want == "reasoning":
                    yield self.event("response.reasoning_text.delta", item_id=item["id"], output_index=i, content_index=0, delta=ev[1])
                else:
                    yield self.event("response.output_text.delta", item_id=item["id"], output_index=i, content_index=0, delta=ev[1], logprobs=[])
            elif kind == "tool_call":
                yield from close()
                name, args = ev[1]
                args = json.dumps(args, ensure_ascii=False)
                call = {"id": _id("fc_"), "type": "function_call", "call_id": _id("call_"), "name": name, "arguments": "", "status": "in_progress"}
                self.output.append(call)
                self.reply["tool_calls"].append({"id": call["call_id"], "type": "function", "function": {"name": name, "arguments": args}})
                i = len(self.output) - 1
                yield self.event("response.output_item.added", output_index=i, item=dict(call))
                yield self.event("response.function_call_arguments.delta", item_id=call["id"], output_index=i, delta=args)
                yield self.event("response.function_call_arguments.done", item_id=call["id"], output_index=i, arguments=args)
                call.update(arguments=args, status="completed")
                yield self.event("response.output_item.done", output_index=i, item=call)
            elif kind == "done":
                yield from close()
                status = "incomplete" if ev[1]["finish_reason"] == "length" else "completed"
                yield self.event("response.incomplete" if status == "incomplete" else "response.completed", response=self.response(status, ev[1]))

    def result(self, events):
        """The response object in one piece."""
        last = None
        for name, data in self.events(events):
            last = data
        resp = last["response"]
        for it in resp["output"]:                        # as the chat completions reply: no blank lines around a part
            for part in it.get("content") or []:
                part["text"] = part["text"].strip()
        return resp

    def history(self, body):
        """The conversation after this reply, for a later previous_response_id (the instructions left out)."""
        msgs = [m for i, m in enumerate(body["messages"]) if not (i == 0 and m["role"] == "system" and self.req.get("instructions"))]
        reply = {k: v for k, v in self.reply.items() if v or k == "content"}
        reply["content"] = reply["content"].strip()
        return msgs + [reply]
