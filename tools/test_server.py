"""In-process server checks: a client that disconnects mid-stream (OpenCode's Esc, a Stop button) leaves the
prompt cache consistent with the engine, and the conversation can continue from the partial reply."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kiln.server import Server

srv = Server("data/models/kiln-mix.gguf", "data/models/tokenizer.json")
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{name:44s} {'ok' if ok else 'FAIL'}  {detail}")


def consistent():
    slot = max(srv.cache.slots, key=lambda s: s.used)
    return srv.engine.seq_len(slot.sid) == len(slot.tokens), f"engine {srv.engine.seq_len(slot.sid)} cache {len(slot.tokens)}"


question = {"role": "user", "content": "Count from 1 to 300, separated by spaces."}
gen = srv.chat({"model": "qwen3.8-27b-fast", "messages": [question], "temperature": 0})
partial = ""
for kind, val in gen:
    partial += val if kind == "content" else ""
    if len(partial) > 60:
        break
gen.close()                                   # what the HTTP handler does when the client goes away
check("aborted stream: cache matches engine", *consistent())

partial = partial[:partial.rstrip().rfind(" ")]   # the client keeps what it received; end on a whole number
messages = [question, {"role": "assistant", "content": partial}, {"role": "user", "content": "Stop. What was the last number you wrote?"}]
answer, info = "", None
for kind, val in srv.chat({"model": "qwen3.8-27b-fast", "messages": messages, "temperature": 0}):
    if kind == "content":
        answer += val
    elif kind == "done":
        info = val
last = partial.split()[-1]
# The partial reply re-tokenizes differently from what was generated, so this resumes from the snapshot taken
# where the reply began, not from the aborted slot (whose recurrent state includes tokens the client never saw).
check("follow-up resumes from the cache", info["cached_tokens"] > 0, f"cached {info['cached_tokens']} of {info['prompt_tokens']}")
check("follow-up knows where the reply stopped", last in answer, f"last {last!r}, answer {answer.strip()[:60]!r}")
check("after follow-up: cache matches engine", *consistent())
print("all ok" if not fails else f"{fails} FAILURES")
sys.exit(fails != 0)
