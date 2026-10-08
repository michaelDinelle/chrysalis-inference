"""Checks kiln's token-level renderer against the official Jinja chat template, and the output parser."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jinja2
from kiln.chat import Tokens, Renderer, OutputParser, message_key

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

def jinja_ids(messages, tools, **kw):
    msgs = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{"type": "function", "function": {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}} for c in m["tool_calls"]]
        msgs.append(m)
    text = template.render(messages=msgs, tools=tools, add_generation_prompt=True, **kw)
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

# output parser
p = OutputParser(tok, thinking=True, schemas={"read": tools[0]["function"]["parameters"]})
events = []
for i in gen + tok.full.encode("", add_special_tokens=False).ids:
    events += p.feed([i])
events += p.finish()
kinds = [k for k, _ in events]
call = [v for k, v in events if k == "tool_call"]
ok = kinds.count("tool_call") == 1 and call[0] == ("bash", {"command": "ls -la"}) and p.text()[0] == "Let me check."
fails += not ok
print(f"parser: reasoning + tool call                 {'ok' if ok else 'FAIL'}  {call} {p.text()}")
p = OutputParser(tok, thinking=True, schemas={"read": tools[0]["function"]["parameters"]})
body = tok.full.encode('Plan.\n</think>\n\nOk.\n\n<tool_call>\n<function=read>\n<parameter=filePath>\nx.py\n</parameter>\n<parameter=limit>\n50\n</parameter>\n</function>\n</tool_call>', add_special_tokens=False).ids
ev = p.feed(body) + p.finish()
call = [v for k, v in ev if k == "tool_call"]
ok = call == [("read", {"filePath": "x.py", "limit": 50})] and p.text() == ("Plan.", "Ok.")
fails += not ok
print(f"parser: typed params, content before call     {'ok' if ok else 'FAIL'}  {call} {p.text()}")
print("all ok" if not fails else f"{fails} FAILURES")
sys.exit(fails != 0)
