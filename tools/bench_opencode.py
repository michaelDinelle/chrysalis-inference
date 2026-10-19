"""What OpenCode (2.0.25) feels on a running Chrysalis server, replayed from its real requests (tools/opencode2.json):
each new session's title request and its first request go out 8 ms apart, as OpenCode sends them, and an agent
session then appends tool output the way OpenCode does (~25 ms after each reply). Per request: time to first token
seen by the client, the whole turn, and tok/s as OpenCode's footer counts it (tokens / whole turn).
  python tools/bench_opencode.py [--url http://127.0.0.1:8080] [--model qwen3.8-flash-next] [--steps 3] [--out FILE]
Run it right after Chrysalis (re)starts: the first scenario is the cold cache."""
import argparse
import json
import os
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = json.load(open(os.path.join(ROOT, "tools/opencode2.json")))
README = open(os.path.join(ROOT, "README.md")).read()


def system(date="Sat Oct 17 2026", cwd="/home/omarchy", git=False):
    return FIX["system"].replace("{DATE}", date).replace("{CWD}", cwd).replace("{GIT}", "yes" if git else "no")


def post(url, body):
    """Streams one chat completion. Returns timings, usage, and the reply as OpenCode echoes it back."""
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, t_first, usage = time.time(), None, {}
    content, reasoning, calls = [], [], {}
    with urllib.request.urlopen(req) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if t_first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                    t_first = time.time()
                content.append(d.get("content") or "")
                reasoning.append(d.get("reasoning_content") or "")
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc["index"], {"id": tc.get("id"), "type": "function", "function": {"name": "", "arguments": ""}})
                    c["function"]["name"] += (tc.get("function") or {}).get("name") or ""
                    c["function"]["arguments"] += (tc.get("function") or {}).get("arguments") or ""
    t1 = time.time()
    msg = {"role": "assistant", "content": "".join(content)}
    if "".join(reasoning):
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return {"t0": t0, "t1": t1, "ttft": (t_first or t1) - t0, "total": t1 - t0, "tokens": usage.get("completion_tokens", 0),
            "prompt": usage.get("prompt_tokens", 0), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "msg": msg}


def request(model, messages, seed):
    return {"model": model, "messages": messages, "tools": FIX["tools"], "stream": True, "stream_options": {"include_usage": True},
            "max_completion_tokens": FIX["max_completion_tokens"], "seed": seed}


def session_turn(url, model, messages, seed):
    """A new session's first turn: the title request, then 8 ms later the turn itself, side by side."""
    out = {}
    first_user = next(m["content"] for m in messages if m["role"] == "user")

    def title():
        out["title"] = post(url, {"model": model, "stream": True, "stream_options": {"include_usage": True},
                                  "messages": [{"role": "system", "content": FIX["title_system"]}, {"role": "user", "content": first_user}]})
    th = threading.Thread(target=title)
    th.start()
    time.sleep(0.008)
    r = post(url, request(model, messages, seed))
    th.join()
    r["title"] = out["title"]
    return r


def server_recent(url):
    with urllib.request.urlopen(url + "/chrysalis/stats") as r:
        return json.load(r).get("recent", [])


def row(name, r, recent):
    srv = next((x for x in reversed(recent) if x["prompt_tokens"] == r["prompt"] and x["completion_tokens"] == r["tokens"]), {})
    t = r.get("title")
    title = f"{t['t1'] - r['t0']:+5.2f}s" if t else "     -"
    print(f"{name:34s} {r['ttft']:6.2f} {r['total']:6.2f} {r['tokens']:5d} {r['tokens'] / r['total']:8.1f} "
          f"{srv.get('decode_tps', 0):7.1f} {srv.get('prefill_s', 0):7.2f} {r['cached']:6d}/{r['prompt']:<6d} {title}", flush=True)
    return {"name": name, "ttft": r["ttft"], "total": r["total"], "tokens": r["tokens"], "opencode_tps": r["tokens"] / r["total"],
            "decode_tps": srv.get("decode_tps"), "prefill_s": srv.get("prefill_s"), "cached": r["cached"], "prompt": r["prompt"],
            "title_done": (t["t1"] - r["t0"]) if t else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    print(f"{'':34s} {'ttft':>6s} {'turn':>6s} {'gen':>5s} {'OC tok/s':>8s} {'decode':>7s} {'prefill':>7s} {'cached/prompt':>13s} title", flush=True)

    def turn(name, msgs, seed):
        r = session_turn(a.url, a.model, msgs, seed)
        rows.append(row(name, r, server_recent(a.url)))
        return r

    home = system()
    turn("cold start: new session", [{"role": "system", "content": home}, {"role": "user", "content": "hello"}], 1)
    turn("new session, same project", [{"role": "system", "content": home}, {"role": "user", "content": "hey there"}], 2)
    chrysalis = system(cwd="/home/omarchy/chrysalis-inference", git=True)
    first = [{"role": "system", "content": chrysalis}, {"role": "user", "content": "good morning"}]
    r = turn("new session, other project", first, 3)
    turn("new day", [{"role": "system", "content": system(date="Sun Oct 18 2026")}, {"role": "user", "content": "hi"}], 4)
    # an agent session in the Chrysalis project: each step appends ~1K tokens of tool output (OpenCode sends it ~25 ms
    # after the reply) to the conversation Chrysalis just generated
    msgs = first + [r["msg"]]
    chunk = 4000
    for k in range(a.steps):
        text = README[k * chunk:(k + 1) * chunk]
        if r["msg"].get("tool_calls"):
            for c in r["msg"]["tool_calls"]:
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": text})
        else:
            msgs.append({"role": "user", "content": f"Here is part of README.md:\n{text}\nReply with just OK."})
        time.sleep(0.025)
        r = post(a.url, request(a.model, msgs, 10 + k))
        rows.append(row(f"agent step {k + 1} (+{len(text)} chars)", r, server_recent(a.url)))
        msgs.append(r["msg"])
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
