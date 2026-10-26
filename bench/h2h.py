import argparse, json, statistics, time, urllib.request
from pathlib import Path

SHORT = "Why is the sky blue?"
STORY = "Write a long story about a lighthouse keeper who finds a message in a bottle. Make it detailed."
WIKI = (Path.home() / "models/data/wikitext-2-raw/wiki.train.raw").read_text(encoding="utf-8")
CHARS_PER_TOK = 4.45


def stream(url, model, messages, n, cache=False):
    body = {"model": model, "messages": messages, "max_tokens": n, "temperature": 0, "top_k": 1, "ignore_eos": True,
            "cache_prompt": cache, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; usage = {}; text = ""; content = ""
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                content += d.get("content") or ""
                if piece:
                    last = time.perf_counter()
                    first = first or last
                    text += piece
    ct, pt = usage.get("completion_tokens"), usage.get("prompt_tokens")
    return {"ttft_ms": (first - t0) * 1000 if first else None, "prompt_tokens": pt, "completion_tokens": ct,
            "decode_tok_s": (ct - 1) / (last - first) if ct and ct > 1 and last > first else None,
            "prefill_tok_s": pt / (first - t0) if pt and first else None, "text": text[:80], "content": content}


def doc(rep, tokens):
    start = (rep * 7919 + tokens) * 37 % (len(WIKI) - int(tokens * CHARS_PER_TOK) - 1)
    return f"Document {time.time_ns()}:\n" + WIKI[start:start + int(tokens * CHARS_PER_TOK)] + "\n\nSummarize the document in one line."


def cell(url, model, name, rep):
    if name == "ttft":
        return stream(url, model, [{"role": "user", "content": f"({rep}-{time.time_ns()}) {SHORT}"}], 8)
    if name in ("short", "story"):
        return stream(url, model, [{"role": "user", "content": SHORT if name == "short" else STORY}], 200)
    if name in ("4k", "32k"):
        return stream(url, model, [{"role": "user", "content": doc(rep, 4000 if name == "4k" else 32000)}], 1)
    if name == "turn":
        msgs = [{"role": "user", "content": doc(rep, 4000)}]
        a = stream(url, model, msgs, 16, cache=True)
        msgs += [{"role": "assistant", "content": a.pop("content") or "OK."}, {"role": "user", "content": "Now list three key terms from it."}]
        return stream(url, model, msgs, 8, cache=True)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8190")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="m")
    ap.add_argument("--cells", default="ttft,short,story,4k,32k,turn")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default=str(Path.home() / "models/bench/h2h.jsonl"))
    a = ap.parse_args()
    stream(a.url, a.model, [{"role": "user", "content": "Hi"}], 16)
    out = open(a.out, "a")
    for c in a.cells.split(","):
        rows = []
        for rep in range(a.reps):
            r = {"name": a.name, "cell": c, "rep": rep, **cell(a.url, a.model, c, rep)}
            r.pop("content", None)
            out.write(json.dumps(r) + "\n"); out.flush(); rows.append(r)
        key = {"ttft": "ttft_ms", "turn": "ttft_ms", "short": "decode_tok_s", "story": "decode_tok_s"}.get(c, "prefill_tok_s")
        vals = [r[key] for r in rows if r[key] is not None]
        print(f"{a.name:28s} {c:6s} {key:14s} median {statistics.median(vals):9.1f}  {[round(v, 1) for v in vals]}" if vals
              else f"{a.name} {c}: no value {rows}", flush=True)


if __name__ == "__main__":
    main()
