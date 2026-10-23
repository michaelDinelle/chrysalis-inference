"""The same long-context measurement for any OpenAI-compatible server (Strata or Chrysalis), timed by the client:
a ~100K-token prompt (synthetic records, as Strata's standard benchmark builds its long prompts), then a follow-up of
~1K new tokens on the same conversation. Reports time to first token and decode rate (completion tokens over the time
from the first token to the last) for each, with Qwen's thinking defaults.
  python longctx.py URL MODEL OUT.json"""
import json, sys, time, urllib.request

URL, MODEL, OUT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
recs = lambda a, b: "\n".join(f"record{i:06d}: alpha beta gamma delta epsilon zeta eta theta." for i in range(a, b))
BIG = recs(1, 5201) + "\n\nSummarize the structure of these records and explain how to look up a record by its ID."
MORE = recs(5201, 5261) + "\n\nThese records were added since. Which ID is the last one now?"

def ask(messages, seed):
    body = {"model": MODEL, "messages": messages, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 512,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "reasoning_effort": "xhigh", "seed": seed,
            "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; usage = None; text = ""; think = ""
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            ev = json.loads(line[6:])
            if ev.get("usage"): usage = ev["usage"]
            for c in ev.get("choices") or []:
                d = c.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    now = time.time(); first = first or now; last = now
                    text += d.get("content") or ""; think += d.get("reasoning_content") or d.get("reasoning") or ""
    n = (usage or {}).get("completion_tokens", 0)
    return {"ttft_s": round(first - t0, 3), "decode_tps": round((n - 1) / (last - first), 2) if n > 1 and last > first else None,
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": n, "answer": text[-300:], "thinking_chars": len(think)}, text

out = []
for trial in (1, 2):
    tag = f"[trial {trial}]\n"
    msgs = [{"role": "user", "content": tag + BIG}]
    a, said = ask(msgs, trial); print("long", a, flush=True)
    msgs += [{"role": "assistant", "content": said}, {"role": "user", "content": MORE}]
    b, _ = ask(msgs, trial); print("follow-up", b, flush=True)
    out.append({"trial": trial, "long": a, "followup": b})
json.dump(out, open(OUT, "w"), indent=1)
