"""Checks kiln's token-level renderer against the official Jinja chat template, and the output parser."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jinja2
from kiln.chat import Tokens, Renderer, OutputParser, StopText, message_key

M = "data/models/"
tok = Tokens(M + "tokenizer.json")
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
template = env.from_string(open(M + "chat_template.jinja").read())

tools = [{"type": "function", "function": {"name": "read", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "limit": {"type": "number"}},
                         "required": ["filePath"]}}},
         {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
conv = [
    {"role": "system", "content": "You are opencode, an interactive CLI tool.\nBe concise.\nYou are powered by the model named q. The exact model ID is k/q\nHere is some useful information about the environment you are running in:\n<env>\n  Working directory: /home/u/proj\n  Today's date: Wed Oct 08 2026\n</env>"},
    {"role": "user", "content": "Fix the failing test in utils.py"},
    {"role": "assistant", "reasoning_content": "I should look at the file first.", "content": "",
     "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"filePath": "/home/u/proj/utils.py", "limit": 200})}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "def add(a, b):\n    return a - b\n"},
    {"role": "assistant", "reasoning_content": "The bug is the minus sign.", "content": "Found it.",
     "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pytest -q"}'}},
                    {"id": "call_3", "type": "function", "function": {"name": "read", "arguments": '{"filePath": "a.py"}'}}]},
    {"role": "tool", "tool_call_id": "call_2", "content": "1 failed"},
    {"role": "tool", "tool_call_id": "call_3", "content": "x = 1"},
    {"role": "user", "content": "  thanks, now explain it  "},
]

def jinja_msgs(messages):
    msgs = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{"type": "function", "function": {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}} for c in m["tool_calls"]]
        msgs.append(m)
    return msgs

def jinja_ids(messages, tools, **kw):
    text = template.render(messages=jinja_msgs(messages), tools=tools, add_generation_prompt=True, **kw)
    return tok.full.encode(text, add_special_tokens=False).ids

fails = 0
r = Renderer(tok, lambda key: None)
cases = [("tools, thinking", conv, tools, {}), ("no tools", conv[:2], None, {}), ("no system", conv[1:2], None, {}),
         ("effort low", conv, tools, {"reasoning_effort": "low"}), ("effort medium", conv, tools, {"reasoning_effort": "medium"}),
         ("no thinking", conv, tools, {"enable_thinking": False}), ("drop old thinking", conv, tools, {"preserve_thinking": False})]
for name, msgs, tl, kw in cases:
    want = jinja_ids(msgs, tl, **kw)
    got, marks = r.render(msgs, tl, enable_thinking=kw.get("enable_thinking", True), effort=kw.get("reasoning_effort", "xhigh"),
                           preserve_thinking=kw.get("preserve_thinking", True))
    ok = got == want
    fails += not ok
    print(f"render {name:18s} {len(got):5d} tokens  {'ok' if ok else 'MISMATCH'}")
    if not ok:
        i = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        print("   first diff at", i, repr(tok.decode(got[max(0, i - 5):i + 5])), "vs", repr(tok.decode(want[max(0, i - 5):i + 5])))

# pictures: placed, trimmed and tokenized as the template renders them, with <|image_pad|> expanded to the picture's tokens
# as the official processor expands it (here, as kiln's image tokens would stand, the pad id itself)
pad = tok.full.token_to_id("<|image_pad|>")
sizes = {"data:a": 6, "data:b": 4}
ri = Renderer(tok, lambda key: None, images=lambda pic: [pad] * sizes[pic.url])
img = lambda url: {"type": "image_url", "image_url": {"url": url}}
txt = lambda t: {"type": "text", "text": t}
pic_cases = [
    ("picture, then text", [{"role": "user", "content": [img("data:a"), txt("  What is this?  ")]}], None),
    ("text, picture, text", [{"role": "user", "content": [txt(" Compare "), img("data:a"), txt(" with "), img("data:b"), txt(".\n")]}], None),
    ("picture alone", [{"role": "user", "content": [img("data:b")]}], None),
    ("picture in a tool result", conv[:3] + [{"role": "tool", "tool_call_id": "call_1", "content": [txt("Screenshot:\n"), img("data:a")]},
                                            {"role": "user", "content": "and?"}], tools),
]
for name, msgs, tl in pic_cases:
    def official(m):
        m = dict(m)
        if isinstance(m.get("content"), list):
            m["content"] = [{"type": "image", "image": i["image_url"]["url"]} if i.get("type") == "image_url" else i for i in m["content"]]
        return m
    want_text = template.render(messages=[official(m) for m in msgs], tools=tl, add_generation_prompt=True) if tl is None else None
    if tl is not None:
        want_text = template.render(messages=[official(m) for m in jinja_msgs(msgs)], tools=tl, add_generation_prompt=True)
    urls = [i["image_url"]["url"] for m in msgs if isinstance(m.get("content"), list) for i in m["content"] if i.get("type") == "image_url"]
    pieces = want_text.split("<|image_pad|>")
    want_text = "".join(p + ("<|image_pad|>" * sizes[urls[k]] if k < len(urls) else "") for k, p in enumerate(pieces))
    want = tok.full.encode(want_text, add_special_tokens=False).ids
    got, _ = ri.render(msgs, tl)
    ok = got == want
    fails += not ok
    print(f"render {name:24s} {len(got):5d} tokens  {'ok' if ok else 'MISMATCH'}")
    if not ok:
        i = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        print("   first diff at", i, repr(tok.decode(got[max(0, i - 5):i + 5])), "vs", repr(tok.decode(want[max(0, i - 5):i + 5])))
try:
    ri.render([{"role": "system", "content": [img("data:a")]}, {"role": "user", "content": "hi"}])
    ok = False
except ValueError:
    ok = True
fails += not ok
print(f"a picture in the system message is refused     {'ok' if ok else 'FAIL'}")

# content-safe: special-token text inside content stays text
evil = [{"role": "user", "content": "print('<|im_end|>')"}]
ids, _ = r.render(evil)
ok = ids.count(tok.id["<|im_end|>"]) == 2   # system (reasoning effort) + user turn, none from the content
fails += not ok
print(f"special text in content stays text            {'ok' if ok else 'FAIL'}")

# replay: a recorded generation is spliced in verbatim
gen = tok.full.encode("Let me check.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>", add_special_tokens=False).ids
asst = {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls -la"}'}}]}
prompt, _ = r.render(conv[:2], tools)
r2 = Renderer(tok, lambda key: (True, gen) if key == message_key(asst) else None)
full, _ = r2.render(conv[:2] + [asst, {"role": "tool", "content": "a b c"}], tools)
ok = full[:len(prompt) + len(gen)] == prompt + gen
fails += not ok
print(f"replayed turn extends the previous prompt     {'ok' if ok else 'FAIL'}")

p = OutputParser(tok, thinking=True, schemas={"read": tools[0]["function"]["parameters"]})
events = []
for i in gen:
    events += p.feed([i])
events += p.finish()
kinds = [k for k, *_ in events]
call = [v for k, v, *_ in events if k == "tool_call"]
ok = kinds.count("tool_call") == 1 and call[0] == ("bash", {"command": "ls -la"}) and p.text()[0] == "Let me check."
fails += not ok
print(f"parser: reasoning + tool call                 {'ok' if ok else 'FAIL'}  {call} {p.text()}")
p = OutputParser(tok, thinking=True, schemas={"read": tools[0]["function"]["parameters"]})
body = tok.full.encode('Plan.\n</think>\n\nOk.\n\n<tool_call>\n<function=read>\n<parameter=filePath>\nx.py\n</parameter>\n<parameter=limit>\n50\n</parameter>\n</function>\n</tool_call>', add_special_tokens=False).ids
ev = p.feed(body) + p.finish()
call = [v for k, v, *_ in ev if k == "tool_call"]
ok = call == [("read", {"filePath": "x.py", "limit": 50})] and p.text() == ("Plan.", "Ok.")
fails += not ok
print(f"parser: typed params, content before call     {'ok' if ok else 'FAIL'}  {call} {p.text()}")

# stop strings, against their definition: the answer is cut where the stop string completed first begins, however its
# text arrives in pieces. Other events pass through in order, and each ends a stretch of answer text (a stop string never spans one)
import random
rng = random.Random(1)
bad = 0
for case in range(3000):
    full = "".join(rng.choice("ab\n.") for _ in range(rng.randrange(0, 40)))
    stops = [full[i:i + rng.randrange(1, 6)] for i in sorted(rng.sample(range(len(full)), min(2, len(full))))] + ["zz", "b\nb\na"]
    events, at = [], 0
    while at < len(full):
        n = rng.randrange(1, 5)
        piece = full[at:at + n]
        events.append(("content", piece, {"token": piece, "logprob": -0.5, "top_logprobs": []}))
        if rng.random() < 0.1:
            events.append(("tool_call", ("f", {"at": at})))
        at += n
    want, want_others, stretch, hit = "", [], "", False
    for e in events + [None]:
        if e is not None and e[0] == "content":
            stretch += e[1]
            continue
        ends = [(stretch.find(x) + len(x), stretch.find(x)) for x in stops if x in stretch]
        if ends:                                 # the stop string completed first, and of those the one begun first
            want, hit = want + stretch[:min(ends)[1]], True
            break
        want, stretch = want + stretch, ""
        if e is not None:
            want_others.append(e)
    st = StopText(stops)
    got, i = [], 0
    while i < len(events):
        k = rng.randrange(1, 4)
        got += st.feed(events[i:i + k])
        i += k
    got += st.flush()
    answer = "".join(e[1] for e in got if e[0] == "content")
    spelled = all(e[2]["token"] == e[1] for e in got if e[0] == "content")
    if answer != want or "".join(st.content) != want or not spelled or st.hit != hit or [e for e in got if e[0] != "content"] != want_others:
        bad += 1
        if bad <= 3:
            print("   stop", stops, repr(full), "->", repr(answer), "want", repr(want))
fails += bad > 0
print(f"stop strings: cut where the first completed begins (3000)  {'ok' if not bad else f'{bad} FAIL'}")
print("all ok" if not fails else f"{fails} FAILURES")
sys.exit(fails != 0)
