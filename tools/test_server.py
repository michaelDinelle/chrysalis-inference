"""In-process server checks: a client that disconnects mid-stream (OpenCode's Esc, a Stop button) leaves the
prompt cache consistent with the engine, and the conversation can continue from the partial reply. A seed
reproduces a reply, whichever cached state the prompt is served from."""
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


def reply(body):
    text, info = "", None
    for kind, val in srv.chat({"model": "qwen3.8-27b-fast", "max_tokens": 60, **body}):
        if kind == "content":
            text += val
        elif kind == "done":
            info = val
    return text, info


poem = [{"role": "user", "content": "Write a four-line poem about a kiln."}]
a, ia = reply({"messages": poem, "seed": 7})
b, ib = reply({"messages": poem, "seed": 7})
check("same seed: same reply (served from cache)", a == b and ib["cached_tokens"] > 0, f"cached {ib['cached_tokens']}, seed {ib['seed']}")
c, _ = reply({"messages": poem, "seed": 8})
check("another seed: another reply", c != a, repr(c[:40]))
d, idd = reply({"messages": poem})
e, _ = reply({"messages": poem, "seed": idd["seed"]})
check("unseeded reply reports a seed that redraws it", d == e, f"seed {idd['seed']}")
# The second turn first extends the live sequence that still holds the first reply (drawn again just before),
# then is served again from the snapshot taken at its prompt. The presence penalty (1.5 without thinking) must
# count this reply's tokens only, or the two would differ.
reply({"messages": poem, "seed": 7})
turn2 = poem + [{"role": "assistant", "content": a}, {"role": "user", "content": "Now one about a forge."}]
f, i_live = reply({"messages": turn2, "seed": 9})
g, i_snap = reply({"messages": turn2, "seed": 9})
served = ia["prompt_tokens"] < i_live["cached_tokens"] < i_snap["cached_tokens"]   # past the first reply, then a later snapshot
check("second turn: live sequence = snapshot", f == g and served,
      f"cached {i_live['cached_tokens']} then {i_snap['cached_tokens']} of {i_snap['prompt_tokens']}")
print("all ok" if not fails else f"{fails} FAILURES")
sys.exit(fails != 0)
