"""In-process server checks: a client that disconnects mid-stream (OpenCode's Esc, a Stop button) leaves the
prompt cache consistent with the engine, and the conversation can continue from the partial reply. A seed
reproduces a reply, whichever cached state the prompt is served from. Anthropic's Messages API and OpenAI's Responses
API answer their official SDKs (data/venv has them) with what those SDKs' own types accept.
  python tools/test_server.py [--use qwen3.8-flash-next]   (default: the 27B)"""
import atexit, os, shutil, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# shared prefixes are kept on disk: from empty, on disk rather than in a RAM-backed /tmp (0.5 GB), gone afterwards
os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
os.environ["CHRYSALIS_SNAPSHOTS"] = tempfile.mkdtemp(prefix="snapshots-", dir=os.path.join(ROOT, "build"))
atexit.register(shutil.rmtree, os.environ["CHRYSALIS_SNAPSHOTS"], True)
from chrysalis.models import MODELS
from chrysalis.server import Server

MODEL = sys.argv[sys.argv.index("--use") + 1] if "--use" in sys.argv else "qwen3.8-27b"
srv = Server(MODELS[MODEL]["path"], "data/models/tokenizer.json", model=MODEL)
fails = 0


def pin(engine):
    """Flash-Next's cold experts' CPU / GPU split held fixed, as tools_fn/compare_fn.py does: the same numbers on every
    load (its own split settles on timings and its expert pool follows the replies, so a seed would not redraw a reply
    across a reload)."""
    if hasattr(engine.lib, "chrysalis_tune"):
        import ctypes as C
        engine.lib.chrysalis_tune.restype, engine.lib.chrysalis_tune.argtypes = C.c_double, [C.c_char_p, C.c_double]
        engine.lib.chrysalis_tune(b"adapt_left", 0)
        engine.lib.chrysalis_tune(b"split", 1)
        engine.lib.chrysalis_tune(b"gfrac", 0.3)
        engine.lib.chrysalis_tune(b"experts", 0)


pin(srv.engine)


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{name:44s} {'ok' if ok else 'FAIL'}  {detail}")


def consistent():
    slot = max(srv.cache.slots, key=lambda s: s.used)
    return srv.engine.seq_len(slot.sid) == len(slot.tokens), f"engine {srv.engine.seq_len(slot.sid)} cache {len(slot.tokens)}"


question = {"role": "user", "content": "Count from 1 to 300, separated by spaces."}
gen = srv.chat({"model": MODEL + "-fast", "messages": [question], "temperature": 0})
partial = ""
for kind, val, *_ in gen:
    partial += val if kind == "content" else ""
    if len(partial) > 60:
        break
gen.close()                                   # what the HTTP handler does when the client goes away
check("aborted stream: cache matches engine", *consistent())

partial = partial[:partial.rstrip().rfind(" ")]   # the client keeps what it received; end on a whole number
messages = [question, {"role": "assistant", "content": partial}, {"role": "user", "content": "Stop. What was the last number you wrote?"}]
answer, info = "", None
for kind, val, *_ in srv.chat({"model": MODEL + "-fast", "messages": messages, "temperature": 0}):
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
    for kind, val, *_ in srv.chat({"model": MODEL + "-fast", "max_tokens": 60, **body}):
        if kind == "content":
            text += val
        elif kind == "done":
            info = val
    return text, info


poem = [{"role": "user", "content": "Write a four-line poem about a chrysalis."}]
a, ia = reply({"messages": poem, "seed": 7})
b, ib = reply({"messages": poem, "seed": 7})
check("same seed: same reply (served from cache)", a == b and ib["cached_tokens"] > 0, f"cached {ib['cached_tokens']}, seed {ib['seed']}")
# A stop string from the middle of that reply: the same seed writes the same reply up to where the string begins.
stop = a[len(a) // 2:len(a) // 2 + 3]
s_text, s_info = reply({"messages": poem, "seed": 7, "stop": ["\u2603", stop]})
check("stop string: the answer ends where it begins", s_text == a[:a.find(stop)] and s_info["finish_reason"] == "stop",
      f"stop {stop!r}: {s_text[-20:]!r}, {s_info['finish_reason']}")
c, _ = reply({"messages": poem, "seed": 8})
check("another seed: another reply", c != a, repr(c[:40]))
d, idd = reply({"messages": poem})
e, _ = reply({"messages": poem, "seed": idd["seed"]})
check("unseeded reply reports a seed that redraws it", d == e, f"seed {idd['seed']}")
# The second turn first extends the live sequence that still holds the first reply (drawn again just before),
# then is served again from the snapshot taken at its prompt. The presence penalty (1.5 without thinking) must
# count this reply's tokens only, or the two would differ.
reply({"messages": poem, "seed": 7})
turn2 = poem + [{"role": "assistant", "content": a}, {"role": "user", "content": "Now one about a butterfly."}]
f, i_live = reply({"messages": turn2, "seed": 9})
g, i_snap = reply({"messages": turn2, "seed": 9})
served = ia["prompt_tokens"] < i_live["cached_tokens"] < i_snap["cached_tokens"]   # past the first reply, then a later snapshot
check("second turn: live sequence = snapshot", f == g and served,
      f"cached {i_live['cached_tokens']} then {i_snap['cached_tokens']} of {i_snap['prompt_tokens']}")


def run(body):
    """All of a request's events: prefix (continued replies), text with log probs, done."""
    out = {"prefix": None, "content": "", "lp": [], "info": None}
    for ev in srv.chat({"model": MODEL + "-fast", **body}):
        if ev[0] == "prefix":
            out["prefix"] = ev[1]
        elif ev[0] == "content":
            out["content"] += ev[1]
            if len(ev) > 2 and ev[2]:
                out["lp"].append(ev[2])
        elif ev[0] == "done":
            out["info"] = ev[1]
    return out


# Log probs: one entry per piece of text, the alternatives first by probability.
names = [{"role": "user", "content": "Suggest a name for a cat, then say why in one sentence."}]
r = run({"messages": names, "seed": 3, "max_tokens": 10, "logprobs": True, "top_logprobs": 4})
alts = r["lp"][0]["top_logprobs"] if r["lp"] else []
check("log probs: entries spell the reply", "".join(e["token"] for e in r["lp"]) == r["content"] and len(alts) == 4
      and all(alts[i]["logprob"] >= alts[i + 1]["logprob"] for i in range(3)), repr(r["content"][:40]))
# Continue a reply cut at its length limit: Chrysalis restates it and goes on from its exact tokens (a cache hit).
cut = {"role": "assistant", "content": r["content"]}
c = run({"messages": names + [cut], "continue_final_message": True, "seed": 4, "max_tokens": 40})
check("continue: from the reply's own tokens", c["prefix"] == {"reasoning_content": "", "content": r["content"]}
      and c["info"]["cached_tokens"] >= r["info"]["prompt_tokens"] and len(c["content"]) > 0,
      f"cached {c['info']['cached_tokens']} of {c['info']['prompt_tokens']}, +{c['content'][:30]!r}")
# Rewrite from the first token with its second alternative.
alt = alts[1]
w = run({"messages": names + [{"role": "assistant", "content": ""}], "continue_final_message": True, "chrysalis_original": cut,
         "chrysalis_append_token": alt["id"], "seed": 4, "max_tokens": 20})
check("rewrite from an alternative token", w["prefix"]["content"] == alt["token"].lstrip("\n") and len(w["content"]) > 0,
      f"{w['prefix']['content']!r} + {w['content'][:30]!r}")
# An aside (the web UI's draft of the user's next message) leaves the conversation's cache as it was.
convo = names + [{"role": "assistant", "content": r["content"] + c["content"]}]
run({"messages": convo + [{"role": "user", "content": "Write my next message."}], "chrysalis_aside": True, "max_tokens": 20})
n = run({"messages": convo + [{"role": "user", "content": "Another one?"}], "max_tokens": 10})
check("aside: the conversation stays cached", n["info"]["cached_tokens"] >= c["info"]["prompt_tokens"],
      f"cached {n['info']['cached_tokens']} of {n['info']['prompt_tokens']}")


# Parallel requests: two replies written at the same time share passes (chrysalis/server.py Batcher) and come out as each does
# alone (greedy: as many drafts each as alone, the same logits). Flash-Next gives a reply one draft in a shared pass, so
# here its replies alone take one too.
import threading
asks = [[{"role": "user", "content": "Name three rivers in Europe and one fact about each."}],
        [{"role": "user", "content": "Explain in four sentences why the sky is blue."}]]
drafts = srv.n_draft
if "flash-next" in MODEL:
    srv.n_draft = 1
alone = [reply({"messages": q, "temperature": 0, "max_tokens": 120})[0] for q in asks]
shared, both = srv.batcher.shared, [None, None]
writers = [threading.Thread(target=lambda i=i: both.__setitem__(i, reply({"messages": asks[i], "temperature": 0, "max_tokens": 120})[0]))
           for i in range(2)]
for w in writers: w.start()
for w in writers: w.join()
srv.n_draft = drafts
check("parallel requests: together = alone", both == alone and srv.batcher.shared > shared,
      f"{srv.batcher.shared - shared} passes shared, {[len(x) for x in both]} chars")
check("after them: cache matches engine", *consistent())

# A long prompt read while another reply is being written (Server.prefill): after each prefill chunk the GPU goes back to
# the reply for as long as the chunk took, so the reply keeps coming (its longest pause about a chunk's read, where it
# would wait for the whole prompt), and the prompt's own reply is the one it gets read alone (the same chunks).
import time
import numpy as np
chunk = srv.umax or MODELS[MODEL]["chunk"]
lines = [f"Note {i}: shipment {i * 7919 % 10007} left dock {i % 13} at {i % 24:02d}:{i * 7 % 60:02d}." for i in range(1, 20 * chunk)]
while len(srv.tok.full.encode("\n".join(lines), add_special_tokens=False).ids) > 4 * chunk:
    lines = lines[:len(lines) * 9 // 10]
long_ask = [{"role": "user", "content": "\n".join(lines) + "\n\nWhich dock did shipment 7919 leave from? Just the number."}]


def cold():
    """Nothing cached: the next prompt is read from its start."""
    for en in srv.cache.entries:
        srv.engine.free_snapshot(en.snap)
    srv.cache.entries.clear()
    for sl in srv.cache.slots:
        srv.engine.truncate(sl.sid, 0)
        sl.tokens = np.zeros(0, np.int32)


cold()
alone, _ = reply({"messages": long_ask, "temperature": 0, "max_tokens": 12})
cold()
stamps, stop = [], threading.Event()


def writer():
    for kind, *_ in srv.chat({"model": MODEL + "-fast", "messages": [{"role": "user", "content": "Count from 1 to 2000, separated by spaces."}],
                              "temperature": 0, "max_tokens": 6000}):
        stamps.append(time.time()) if kind == "content" else None
        if stop.is_set():
            break


w = threading.Thread(target=writer)
w.start()
while len(stamps) < 20:
    time.sleep(0.01)
t0 = time.time()
during, read = reply({"messages": long_ask, "temperature": 0, "max_tokens": 12})
t1 = t0 + read["prefill_s"]
stop.set()
w.join()
inside = [t for t in stamps if t0 <= t <= t1]
gap = max(np.diff([t0] + inside + [t1]))
check("a long prompt read beside a reply", during == alone and inside and gap < 0.4 * read["prefill_s"],
      f"read in {read['prefill_s']:.1f} s; the reply's longest pause {gap:.2f} s, {len(inside)} pieces meanwhile; {during!r}")
check("after it: cache matches engine", *consistent())


# An answer held to a JSON schema (response_format): property names nobody would pick, a prompt that invites prose. What
# comes back must parse and validate (jsonschema, not the grammar engine Chrysalis uses), with or without reasoning
# first; without the schema the same prompt gets prose.
import json, jsonschema
shape = {"type": "object", "additionalProperties": False, "required": ["zq_city", "temp_c", "tags"],
         "properties": {"zq_city": {"type": "string"}, "temp_c": {"type": "number"},
                        "tags": {"type": "array", "items": {"enum": ["warm", "cold", "wet", "dry"]}, "minItems": 1}}}
prose = [{"role": "user", "content": "Tell me about the weather in Lisbon in July, in a few sentences."}]
held = {"messages": prose, "seed": 3, "max_tokens": 200, "response_format": {"type": "json_schema", "json_schema": {"name": "w", "schema": shape}}}
for name, body in (("no reasoning", held), ("after reasoning", {**held, "model": MODEL, "max_tokens": 1500})):
    r = run(body) if name == "no reasoning" else None
    steps = r["info"]["tokens_per_step"] if r else steps
    if r is None:
        out = {"content": "", "reasoning": "", "info": None}
        for ev in srv.chat(body):
            if ev[0] in ("content", "reasoning"):
                out[ev[0]] += ev[1]
            elif ev[0] == "done":
                out["info"] = ev[1]
        r = out
    try:
        jsonschema.validate(json.loads(r["content"]), shape)
        valid = True
    except (ValueError, jsonschema.ValidationError) as e:
        valid = str(e)[:60]
    check(f"response_format: JSON, {name}", valid is True and r["info"]["finish_reason"] == "stop" and (name == "no reasoning" or r["reasoning"]),
          f"{r['content'].strip()[:70]!r}" if valid is True else valid)
free = run({"messages": prose, "seed": 3, "max_tokens": 60})
check("response_format: without it, prose", not free["content"].lstrip().startswith("{"), repr(free["content"][:40]))
# Its steps verify drafts, the grammar's forced tokens and the MTP head's guesses (Engine.generate_masked): one token a
# step without them, 2.55 with them (measured, the answer above).
check("response_format: steps of several tokens", steps >= 1.5, f"{steps:.2f} tokens a step")
check("after the held answers: cache matches engine", *consistent())


# Tool calls held to their tool's schema (chrysalis/structured.py): a parameter name nobody would guess, an enum, a
# bounded integer. Every call must parse and validate (jsonschema), whether tool_choice demands one on a prompt that
# asks for none ("required"), names a tool the prompt does not want, or the strict tool is called in a free answer. The
# 27B writes that last call right by itself (even asked for 30 days, or for a parameter of another name: measured), so
# there it is the held path that is checked: a call that comes back whole, its values typed.
odd = {"type": "function", "function": {"name": "zq_forecast", "description": "A weather forecast.", "strict": True,
       "parameters": {"type": "object", "additionalProperties": False, "required": ["zq_place", "units", "days"],
                      "properties": {"zq_place": {"type": "string"}, "units": {"enum": ["metric_si", "imperial_us"]},
                                     "days": {"type": "integer", "minimum": 1, "maximum": 14}}}}}
note = {"type": "function", "function": {"name": "save_note", "description": "Save a note for the user.",
        "parameters": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}}}


def answer_of(body):
    out = {"calls": [], "content": "", "reasoning": "", "info": None}
    for ev in srv.chat({"model": MODEL + "-fast", "max_tokens": 300, **body}):
        if ev[0] == "tool_call":
            out["calls"].append(ev[1])
        elif ev[0] in ("content", "reasoning"):
            out[ev[0]] += ev[1]
        elif ev[0] == "done":
            out["info"] = ev[1]
    return out


def valid(call):
    try:
        jsonschema.validate(call[1], {"zq_forecast": odd, "save_note": note}[call[0]]["function"]["parameters"])
        return True
    except (KeyError, jsonschema.ValidationError):
        return False


hi = [{"role": "user", "content": "Hi! How are you doing today?"}]
r = answer_of({"messages": hi, "tools": [odd, note], "tool_choice": "required"})
check("tool_choice required: a valid call", r["calls"] and all(map(valid, r["calls"])) and r["info"]["finish_reason"] == "tool_calls",
      repr(r["calls"])[:80])
milk = [{"role": "user", "content": "Save a note saying I need milk."}]
r = answer_of({"messages": milk, "tools": [odd, note], "tool_choice": {"type": "function", "function": {"name": "zq_forecast"}}})
check("tool_choice naming a tool: it, once, valid", len(r["calls"]) == 1 and r["calls"][0][0] == "zq_forecast" and valid(r["calls"][0]),
      repr(r["calls"])[:80])
paris = [{"role": "user", "content": "What will the weather be in Paris over the next three days, in celsius? Use the forecast tool."}]
r = answer_of({"model": MODEL, "messages": paris, "tools": [odd, note], "max_tokens": 2000})
check("a strict tool in a free answer: valid calls", r["calls"] and all(map(valid, r["calls"])), repr(r["calls"])[:80])
check("after the held calls: cache matches engine", *consistent())

# A thinking budget (thinking_budget; Anthropic's budget_tokens below): the reasoning ends at it with the words Qwen3's
# technical report closes it with, and the answer follows. Unbounded, this problem takes the model past 4,000 tokens.
pell = [{"role": "user", "content": "Find all integer solutions of x^2 - 7y^2 = 1 with 0 < x < 10000. Work it through carefully."}]
r = answer_of({"model": MODEL, "messages": pell, "thinking_budget": 200, "max_tokens": 2500, "seed": 1})
n = len(srv.tok.full.encode(r["reasoning"], add_special_tokens=False).ids)
check("thinking_budget: reasoning closed there, then the answer", n <= 200 + len(srv.close_thinking()) + 4
      and r["reasoning"].rstrip().endswith("directly now.") and len(r["content"]) > 20, f"{n} reasoning tokens; {r['content'][:40]!r}")


# Anthropic's Messages API and OpenAI's Responses API over HTTP, through their official SDKs; every response and every
# streamed event is also validated strictly against the SDKs' own types (the SDKs parse leniently).
import base64, json, threading, time
import anthropic, httpx, openai
from pydantic import TypeAdapter
from chrysalis.server import HTTPServer, make_handler

ready = threading.Event()
ready.set()
httpd = HTTPServer(("127.0.0.1", 0), make_handler({"srv": srv}, ready))
threading.Thread(target=httpd.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{httpd.server_port}"
claude = anthropic.Anthropic(base_url=base, api_key="chrysalis", max_retries=0)
oai = openai.OpenAI(base_url=base + "/v1", api_key="chrysalis", max_retries=0)
weather = {"name": "get_weather", "description": "The current weather in a city.",
           "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
ask = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]


def sse(path, body):
    """A streamed request's events as (name, data), straight off the wire."""
    out = []
    with httpx.stream("POST", base + path, json={**body, "stream": True}, timeout=300) as r:
        name = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                out.append((name, json.loads(line[6:])))
    return out


def strictly(adapter, obj):
    try:
        TypeAdapter(adapter).validate_python(obj)
        return True, ""
    except Exception as e:
        return False, str(e).split("\n")[0][:120]


raw = claude.messages.with_raw_response.create(model=MODEL, max_tokens=40, messages=[{"role": "user", "content": "Say hello in French."}],
                                               thinking={"type": "disabled"})
msg = raw.http_response.json()
ok, why = strictly(anthropic.types.Message, msg)
check("Messages: a reply", ok and msg["content"][0]["type"] == "text" and msg["content"][0]["text"] and msg["stop_reason"] in ("end_turn", "max_tokens"),
      why or repr(msg["content"][0]["text"][:40]))
body = {"model": MODEL, "max_tokens": 300, "tools": [weather], "messages": ask, "thinking": {"type": "enabled", "budget_tokens": 1024}}
evs = sse("/v1/messages", body)
bad = [why for _, d in evs for ok, why in [strictly(anthropic.types.RawMessageStreamEvent, d)] if not ok]
names = [n for n, _ in evs]
kinds = [d["content_block"]["type"] for n, d in evs if n == "content_block_start"]
check("Messages: a stream (thinking, a tool call)", not bad and names[0] == "message_start" and names[-1] == "message_stop"
      and names.count("content_block_start") == names.count("content_block_stop") and kinds[:1] == ["thinking"] and "tool_use" in kinds
      and evs[-2][1]["delta"]["stop_reason"] == "tool_use" and all(n == d["type"] for n, d in evs), (bad[:1] or [kinds])[0])
with claude.messages.stream(**body) as st:            # the SDK puts the same stream back together
    final = st.get_final_message()
use = next((b for b in final.content if b.type == "tool_use"), None)
check("Messages: the SDK assembles the stream", use is not None and use.name == "get_weather" and "paris" in json.dumps(use.input).lower(),
      repr(use and use.input))
after = claude.messages.create(model=MODEL, max_tokens=120, tools=[weather], thinking={"type": "disabled"}, messages=ask + [
    {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in final.content]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "Raining, 11 C, wind 30 km/h."}]}])
text = "".join(b.text for b in after.content if b.type == "text")
check("Messages: a tool result answered", "11" in text or "rain" in text.lower(), repr(text[:60]))
n = claude.messages.count_tokens(model=MODEL, tools=[weather], messages=ask, thinking={"type": "disabled"})
m = claude.messages.create(model=MODEL, max_tokens=1, tools=[weather], messages=ask, thinking={"type": "disabled"})
check("Messages: count_tokens = the prompt it reads", n.input_tokens == m.usage.input_tokens + m.usage.cache_read_input_tokens,
      f"{n.input_tokens} vs {m.usage.input_tokens} + {m.usage.cache_read_input_tokens}")
anyc = claude.messages.create(model=MODEL, max_tokens=200, tools=[weather], tool_choice={"type": "any"}, thinking={"type": "disabled"},
                              messages=[{"role": "user", "content": "Hi! How are you doing today?"}])
use = next((b for b in anyc.content if b.type == "tool_use"), None)
check("Messages: tool_choice any", use is not None and isinstance(use.input.get("city"), str) and anyc.stop_reason == "tool_use",
      repr(use and use.input))
bud = claude.messages.create(model=MODEL, max_tokens=4000, thinking={"type": "enabled", "budget_tokens": 1024}, messages=pell)
thought = "".join(b.thinking for b in bud.content if b.type == "thinking")
n = len(srv.tok.full.encode(thought, add_special_tokens=False).ids)
check("Messages: budget_tokens closes the thinking", n <= 1024 + len(srv.close_thinking()) + 4 and thought.rstrip().endswith("directly now.")
      and any(b.type == "text" and b.text for b in bud.content), f"{n} thinking tokens")
st = claude.messages.create(model=MODEL, max_tokens=60, stop_sequences=["4"], thinking={"type": "disabled"},
                            messages=[{"role": "user", "content": "Count from 1 to 9, separated by spaces."}])
check("Messages: a stop sequence", st.stop_reason == "stop_sequence" and st.stop_sequence == "4" and "4" not in st.content[0].text,
      f"{st.stop_reason} {st.stop_sequence!r} {st.content[0].text!r}")
photo = base64.b64encode(open("data/llama.cpp/tools/mtmd/test-1.jpeg", "rb").read()).decode()
if srv.pictures:
    pic = claude.messages.create(model=MODEL, max_tokens=40, thinking={"type": "disabled"}, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": photo}},
        {"type": "text", "text": "What is this front page's headline? Just the headline."}]}])
    check("Messages: a picture", "moon" in pic.content[0].text.lower(), repr(pic.content[0].text[:60]))
    # The photo read at fewer pixels: the official processor's grids at 512 x 512 pixels (tools/test_vision.py) make it
    # 13 x 18 tokens where 15 x 20, asked for by the image part (detail "low") or by the request (max_pixels).
    url = "data:image/jpeg;base64," + photo

    def read(part, **extra):
        r = oai.chat.completions.create(model=MODEL, max_tokens=40, extra_body={"enable_thinking": False, **extra}, messages=[
            {"role": "user", "content": [part, {"type": "text", "text": "What is this front page's headline? Just the headline."}]}])
        return r.usage.prompt_tokens, r.choices[0].message.content or ""
    (full, _), (low, said), (asked, _) = (read({"type": "image_url", "image_url": {"url": url}}),
                                          read({"type": "image_url", "image_url": {"url": url, "detail": "low"}}),
                                          read({"type": "image_url", "image_url": {"url": url}}, max_pixels=512 * 512))
    check("a picture read at fewer pixels", full - low == 300 - 13 * 18 and asked == low and "moon" in said.lower(),
          f"prompt {full} tokens, {low} at detail low, {asked} at max_pixels 262144; {said[:40]!r}")

    # A PDF (chrysalis/documents.py), its pages as pictures and as their text, through each API's way of attaching one. The code
    # is set in white (only the text layer has it), the vault number drawn as a picture (only the page's image has it).
    from PIL import Image, ImageDraw, ImageFont
    from pdf_fixture import pdf

    page = Image.new("RGB", (1224, 1584), "white")
    ImageDraw.Draw(page).text((150, 600), "VAULT 9314", fill="black", font=ImageFont.load_default(size=120))
    doc = base64.b64encode(pdf([("text", "The secret code is CHRYSALIS-4217."), ("image", page)])).decode()
    q = "What is the secret code in this document, and which vault number does it show? Answer in one short sentence."
    r = oai.chat.completions.create(model=MODEL, max_tokens=60, extra_body={"enable_thinking": False}, messages=[{"role": "user", "content": [
        {"type": "file", "file": {"file_data": "data:application/pdf;base64," + doc, "filename": "vault.pdf"}}, {"type": "text", "text": q}]}])
    said = r.choices[0].message.content or ""
    check("a PDF: its text and its pages' pictures", "4217" in said and "9314" in said, repr(said[:60]))
    m = claude.messages.create(model=MODEL, max_tokens=60, thinking={"type": "disabled"}, messages=[{"role": "user", "content": [
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": doc}, "title": "vault.pdf"},
        {"type": "text", "text": q}]}])
    said = m.content[0].text
    check("Messages: a PDF document", "4217" in said and "9314" in said, repr(said[:60]))
    rr = oai.responses.create(model=MODEL, reasoning={"effort": "none"}, max_output_tokens=60, input=[{"role": "user", "content": [
        {"type": "input_file", "file_data": "data:application/pdf;base64," + doc, "filename": "vault.pdf"}, {"type": "input_text", "text": q}]}])
    check("Responses: a PDF file", "4217" in rr.output_text and "9314" in rr.output_text, repr(rr.output_text[:60]))

    # A video (chrysalis/vision.py): frames taken 2 a second, each pair read as a picture with its time before it. A red circle
    # for two seconds, then a blue square and "CHRYSALIS 58": what came first, and what follows.
    import subprocess
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x360", "-r", "10", "-i", "-",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "-"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    font, frames = ImageFont.load_default(size=48), []
    for i in range(40):
        img = Image.new("RGB", (640, 360), "white")
        d = ImageDraw.Draw(img)
        if i < 20:
            d.ellipse([200 + 4 * i, 100, 320 + 4 * i, 220], fill="red")
        else:
            d.rectangle([100, 120, 220, 240], fill="blue")
            d.text((300, 150), "CHRYSALIS 58", fill="black", font=font)
        frames.append(img.tobytes())
    film, _ = enc.communicate(b"".join(frames))
    r = oai.chat.completions.create(model=MODEL, max_tokens=80, extra_body={"enable_thinking": False}, messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(film).decode()}},
        {"type": "text", "text": "This video shows a shape, then another shape with some text. What color is the first shape, and "
                                 "what does the text say? One short sentence."}]}])
    said = r.choices[0].message.content or ""
    check("a video: what it shows, in order", "red" in said.lower() and "58" in said, repr(said[:70]))

raw = oai.responses.with_raw_response.create(model=MODEL, input="Say hello in French.", reasoning={"effort": "none"}, max_output_tokens=40)
resp = raw.http_response.json()
ok, why = strictly(openai.types.responses.Response, resp)
out = [i for i in resp["output"] if i["type"] == "message"]
check("Responses: a reply", ok and resp["status"] in ("completed", "incomplete") and out and out[0]["content"][0]["text"], why or repr(out and out[0]["content"][0]["text"][:40]))
fn = {"type": "function", "name": "get_weather", "description": weather["description"], "parameters": weather["input_schema"]}
evs = sse("/v1/responses", {"model": MODEL, "input": ask[0]["content"], "tools": [fn], "max_output_tokens": 400})
bad = [why for _, d in evs for ok, why in [strictly(openai.types.responses.ResponseStreamEvent, d)] if not ok]
seqs = [d["sequence_number"] for _, d in evs]
done = evs[-1][1].get("response", {}) if evs else {}
call = next((i for i in done.get("output", []) if i["type"] == "function_call"), None)
check("Responses: a stream (reasoning, a function call)", not bad and evs[0][0] == "response.created" and evs[-1][0] == "response.completed"
      and seqs == list(range(len(seqs))) and done["output"][0]["type"] == "reasoning" and call is not None and "paris" in call["arguments"].lower(),
      (bad[:1] or [[i["type"] for i in done.get("output", [])]])[0])
req = oai.responses.create(model=MODEL, input="Hi! How are you doing today?", tools=[fn], tool_choice="required",
                           reasoning={"effort": "none"}, max_output_tokens=200)
call = next((i for i in req.output if i.type == "function_call"), None)
check("Responses: tool_choice required", call is not None and isinstance(json.loads(call.arguments).get("city"), str), repr(call and call.arguments))
with oai.responses.stream(model=MODEL, input=ask[0]["content"], tools=[fn], max_output_tokens=400) as st:
    final = st.get_final_response()
call = next((i for i in final.output if i.type == "function_call"), None)
check("Responses: the SDK assembles the stream", call is not None and "paris" in call.arguments.lower(), repr(call and call.arguments))
tool_out = {"type": "function_call_output", "call_id": call.call_id, "output": "Raining, 11 C, wind 30 km/h."}
nxt = oai.responses.create(model=MODEL, previous_response_id=final.id, input=[tool_out], tools=[fn], reasoning={"effort": "none"}, max_output_tokens=120)
full = oai.responses.create(model=MODEL, input=[{"role": "user", "content": ask[0]["content"]}] + [i.model_dump(exclude_none=True) for i in final.output]
                            + [tool_out], tools=[fn], reasoning={"effort": "none"}, max_output_tokens=120)
check("Responses: a function result, both ways", ("11" in nxt.output_text or "rain" in nxt.output_text.lower())
      and nxt.usage.input_tokens == full.usage.input_tokens and full.usage.input_tokens_details.cached_tokens > 0,
      f"{nxt.output_text[:40]!r}; prompt {nxt.usage.input_tokens} / {full.usage.input_tokens}, cached {full.usage.input_tokens_details.cached_tokens}")
fmt = oai.responses.create(model=MODEL, input=prose[0]["content"], reasoning={"effort": "none"}, max_output_tokens=200,
                           text={"format": {"type": "json_schema", "name": "w", "schema": shape, "strict": True}})
try:
    jsonschema.validate(json.loads(fmt.output_text), shape)
    valid = True
except (ValueError, jsonschema.ValidationError) as e:
    valid = str(e)[:60]
check("Responses: text.format holds the answer", valid is True and fmt.text.format.type == "json_schema",
      repr(fmt.output_text[:50]) if valid is True else valid)

# /metrics (chrysalis/metrics.py) as Prometheus's own parser reads it: a reply moves the counters by what its usage
# says, a request Chrysalis refuses counts as an error. On Flash-Next the PCIe link Chrysalis reports
# (chrysalis/link.py) bounds the DMA rate the engine measures, and the rate comes near it: a reading of the card's own
# switch (16 GT/s x16) would be twice too fast.
from prometheus_client.parser import text_string_to_metric_families
from chrysalis import link


def scrape():
    return {(s.name, tuple(sorted(s.labels.items()))): s.value
            for f in text_string_to_metric_families(httpx.get(base + "/metrics").text) for s in f.samples}


m0 = scrape()
r = oai.chat.completions.create(model=MODEL + "-fast", messages=[{"role": "user", "content": "Name a colour."}], max_tokens=40)
httpx.post(base + "/v1/chat/completions", json={"messages": []})
m1 = scrape()
moved = lambda name, **labels: m1.get((name, tuple(sorted(labels.items()))), 0) - m0.get((name, tuple(sorted(labels.items()))), 0)
check("metrics: a reply's tokens", moved("chrysalis_completion_tokens_total", model=MODEL) == r.usage.completion_tokens
      and moved("chrysalis_prompt_tokens_total", model=MODEL) == r.usage.prompt_tokens
      and moved("chrysalis_replies_total", model=MODEL, finish=r.choices[0].finish_reason) == 1
      and moved("chrysalis_time_to_first_token_seconds_count", model=MODEL) == 1, f"{r.usage.completion_tokens} tokens")
check("metrics: a refused request", moved("chrysalis_http_errors_total", status="400") == 1)
now, dma = link.status(srv.link_dev), m1.get(("chrysalis_dma_bytes_per_second", ()))
if srv.engine.expert_stats():
    lanes = now and now["gts"] * 1e9 * now["width"] * (0.8 if now["gts"] < 8 else 128 / 130) / 8   # bytes/s, line encoding
    check("metrics: the PCIe link bounds the DMA rate", bool(now and dma and 0.25 * lanes < dma <= lanes),
          f"link {now and now['gts']} GT/s x{now and now['width']} ({lanes and lanes / 1e9:.1f} GB/s), DMA {dma and dma / 1e9:.1f} GB/s")
httpd.shutdown()


# Sleep, as chrysalis/sleep.py calls it on logind's signals: the reply being written ends at its next step, the GPU's
# memory is freed, and after the wake the model is back (a seed draws the same reply from the empty cache).
from chrysalis.sleep import vram_used

# A tool block is a prefix every session shares: kept on disk, the wake below reads it back instead of prefilling it.
tooled = {"messages": [{"role": "user", "content": "What is the weather in Oslo? Answer in one word if you can."}], "seed": 5,
          "tools": [{"type": "function", "function": {"name": weather["name"], "description": weather["description"],
                                                      "parameters": weather["input_schema"]}}]}
def answer(body):
    """A reply's text and tool calls, and its info."""
    out, info = [], None
    for kind, val, *_ in srv.chat({"model": MODEL + "-fast", "max_tokens": 60, **body}):
        if kind in ("content", "tool_call"):
            out.append((kind, val))
        elif kind == "done":
            info = val
    return out, info


t_before, i_before = answer(tooled)
disk = srv.cache.disk
while disk and any(t is None for t in dict(disk.index).values()):   # its file is written by a thread
    time.sleep(0.05)
events = []
writer = threading.Thread(target=lambda: events.extend(srv.chat({"model": MODEL + "-fast", "messages": [question], "temperature": 0})))
writer.start()
while not any(e[0] == "content" for e in events) and writer.is_alive():
    time.sleep(0.01)
held, t0 = vram_used(), time.time()
srv.sleep()
slept = time.time() - t0
writer.join()
freed = vram_used()
check("sleep: the reply ends at its next step", events[-1][0] == "done" and events[-1][1]["finish_reason"] == "length" and slept < 2,
      f"{events[-1][1]['completion_tokens']} tokens, GPU free after {slept:.2f} s")
check("sleep: the GPU's memory is freed", freed < held - 15e9, f"VRAM used {held / 1e9:.1f} -> {freed / 1e9:.1f} GB")
srv.wake()
pin(srv.engine)
again, _ = reply({"messages": poem, "seed": 7})
check("wake: the model is back, a seed redraws its reply", again == a, repr(again[:40]))
if disk:
    t_after, i_after = answer(tooled)
    check("wake: the tool block read back from disk", i_after["cached_tokens"] > 0 and t_after == t_before and t_after,
          f"cached {i_after['cached_tokens']} of {i_after['prompt_tokens']} ({len(os.listdir(disk.dir))} files), {str(t_after)[:50]}")
print("all ok" if not fails else f"{fails} FAILURES")
sys.exit(fails != 0)
