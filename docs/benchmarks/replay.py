"""Strata's standard benchmark (standard-v1) replayed on any server through Strata's own code: its frozen fixtures and
manifest check (load_corpus), its request body (payload_of: Qwen's thinking defaults, xhigh), its streaming client
(stream_chat) and its scoring (evaluate). Only the check that the server is Strata is left out. Every engine is timed
the same way, by the client: TTFT is the first streamed event, decode is completion tokens after the first over the
time from the first event to the end. Also re-times Strata's saved results.json with the same client clock.
  STRATA_TERMINAL=... python replay.py URL MODEL OUT.json
  STRATA_TERMINAL=... python replay.py --from-strata results.json OUT.json"""
import json, os, statistics, sys
# Strata's installed terminal (its strata_cli package and bench/prompts/standard-v1), e.g. ~/.local/share/strata/terminal/current
sys.path.insert(0, os.environ["STRATA_TERMINAL"])
from strata_cli import bench_standard as B

def client_metrics(ttft, total, out_tokens):
    return {"ttft_s": round(ttft, 3), "total_s": round(total, 3),
            "decode_tps": round((out_tokens - 1) / (total - ttft), 2) if out_tokens > 1 and total > ttft else None}

def summarize(rows):
    out = {}
    for case in ("short", "prompt-4k", "prompt-16k"):
        rs = [r for r in rows if r["phase"] == "performance" and r["case_id"] == case]
        out[case] = {k: statistics.median([r[k] for r in rs if r[k] is not None]) for k in ("ttft_s", "decode_tps", "total_s")}
    sanity = [r for r in rows if r["phase"] == "sanity"]
    out["sanity_passed"] = f"{sum(r['passed'] for r in sanity)}/{len(sanity)}"
    return out

if sys.argv[1] == "--from-strata":
    res = json.load(open(sys.argv[2]))
    rows = []
    for r in res["requests"]:
        if r["phase"] not in ("performance", "sanity"): continue
        m = client_metrics(r["timings"]["ttft_s"], r["timings"]["total_s"], r["tokens"]["output_tokens"])
        rows.append({"id": r["id"], "phase": r["phase"], "case_id": r["case_id"], **m, "out_tokens": r["tokens"]["output_tokens"],
                     "passed": bool((r.get("evaluation") or {}).get("passed"))})
    json.dump({"rows": rows, "summary": summarize(rows)}, open(sys.argv[3], "w"), indent=1)
    print(json.dumps(summarize(rows), indent=1)); sys.exit()

URL, MODEL, OUT = sys.argv[1:4]
corpus, target, rows = B.load_corpus(), B.Target(URL, None), []
for phase in ("warmup", "performance", "sanity"):
    for spec in [s for s in corpus.specs if s["phase"] == phase]:
        raw = B.stream_chat(target, spec, MODEL, timeout_s=B.REQUEST_TIMEOUT_S)
        total = (raw["t_end"] - raw["t0"]) / 1e9
        ttft = raw["events"][0][0] if raw["events"] else total
        n = (raw["usage"] or {}).get("completion_tokens", 0)
        status = "failed" if raw["error"] else "truncated" if raw["finish_reason"] == "length" else "completed"
        calls = [c for _, c in sorted(raw["tool_calls"].items())]
        ev = B.evaluate(spec, status, "".join(raw["content"]), calls) if spec.get("expected") else None
        row = {"id": spec["id"], "phase": phase, "case_id": spec["case_id"], **client_metrics(ttft, total, n), "out_tokens": n,
               "prompt_tokens": (raw["usage"] or {}).get("prompt_tokens"), "status": status, "error": raw["error"],
               "passed": bool(ev and ev["passed"]), "reason": ev and ev["reason"]}
        print(json.dumps(row), flush=True)
        if phase != "warmup": rows.append(row)
json.dump({"corpus_sha256": corpus.sha256, "rows": rows, "summary": summarize(rows)}, open(OUT, "w"), indent=1)
print(json.dumps(summarize(rows), indent=1))
