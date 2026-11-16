import json
import os
import signal
import subprocess
import sys
import time

reply = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hill_lib import DOCS, RIVALS, Sampler, Server, deep, nulls, rivals_here
from measure import PY, ROOT, STRATA_CASES, kl_ref, kl_tokens, p99, tree_supports

name, here = sys.argv[1], rivals_here()


def send(obj):
    reply.write(json.dumps(obj) + "\n")
    reply.flush()


def speed(s, load):
    start = time.monotonic()
    s.clear_peak()
    sampler = Sampler()
    out, profile = {"load_s": load["load_s"]}, {}
    limited0 = (s.stats().get("experts") or {}).get("memory_limited")
    specs = {x["id"]: x for x in s.B.load_corpus().specs}

    def record(case, spec, deep_from=None):
        x0 = s.stats().get("experts") or {}
        raw, stat = s.ask(spec)
        x1 = s.stats().get("experts") or {}
        d = {k: x1[k] - x0.get(k, 0) for k in x1 if isinstance(x1[k], (int, float))}
        steps = max(d.get("decode_steps", 0), 1)
        profile.update({f"{case}.{k}": v for k, v in {
            "ms_per_step": 1000 * stat["tokens_per_step"] / stat["decode_tps"], "tokens_per_step": stat["tokens_per_step"],
            "pool_hits": d.get("decode_picks_in_pool", 0) / max(d.get("decode_picks", 0), 1),
            "cold_picks_per_step": (d.get("decode_picks", 0) - d.get("decode_picks_in_pool", 0)) / steps,
            "moves_per_step": d.get("placed", 0) / steps, "from_file_per_step": d.get("placed_file_tier", 0) / steps,
            "waited_ms_per_step": 1000 * d.get("waited_s", 0) / steps, "prefill_s": stat["prefill_s"]}.items()})
        if deep_from and stat["cached_tokens"] < 0.9 * stat["prompt_tokens"]:
            s.stop()
            sys.exit(f"{name} {case}: the {deep_from} prefix was not restored from the bench snapshots ({stat['cached_tokens']} of {stat['prompt_tokens']} cached)")
        gaps = sorted(b[0] - a[0] for a, b in zip(raw["events"], raw["events"][1:]))
        out.update({f"ttft.{case}": raw["events"][0][0], f"prompt_tps.{case}": (stat["prompt_tokens"] - stat["cached_tokens"]) / stat["prefill_s"],
                    f"decode_tps.{case}": stat["decode_tps"], f"gap_ms.{case}": 1000 * p99(gaps), f"per_pass.{case}": stat["tokens_per_step"]})
        return raw, stat

    for case in ("short", "4k"):
        record(case, {**specs[STRATA_CASES[case]], "max_output_tokens": 512 if here else 64})
    for case in ("16k", "100k"):
        doc, more = DOCS[case]
        raw, _ = record(case, {"id": f"deep-{case}", "messages": deep(doc, more), "max_output_tokens": 64, "seed": 1}, case)
    follow = deep(*DOCS["100k"]) + [{"role": "assistant", "content": "".join(raw["content"])}, {"role": "user", "content": "Which ID is the last one now?"}]
    _, stat = record("followup", {"id": "follow-up", "messages": follow, "max_output_tokens": 64, "seed": 1})
    out["reused.followup"] = stat["cached_tokens"]
    stats = s.stats()
    out["kv_tokens"] = stats["kv_tokens"]
    limited = (stats.get("experts") or {}).get("memory_limited")
    out["mem_limited"] = None if limited is None else load["mem_limited"] + limited - limited0
    out["ram_gb"] = max(load["ram_gb"], s.peak_ram_gb())
    used = sampler.stop()
    out["energy_kj"] = None if used["energy_kj"] is None else load["energy_kj"] + used["energy_kj"]
    out["min_free_gb"] = min(load["min_free_gb"], used["min_free_gb"])
    out["run_s"] = load["run_s"] + time.monotonic() - start
    for rival, r in RIVALS["rivals"].items():
        if r["model"] == name:
            for i, case in enumerate(("short", "4k")):
                out[f"{rival}.ttft.{case}"] = r["ttft"][i] / out[f"ttft.{case}"] if here else None
                out[f"{rival}.decode.{case}"] = out[f"decode_tps.{case}"] / r["tps"][i] if here else None
    s.renew()
    return out, profile


def quality(s, load):
    passed, t0, each = 0, time.monotonic(), []
    for spec in (x for x in s.B.load_corpus().specs if x["phase"] == "sanity"):
        t1 = time.monotonic()
        raw, st = s.ask(spec)
        each.append((round(time.monotonic() - t1, 2), st.get("completion_tokens"), st.get("prompt_tokens")))
        status = "truncated" if raw["finish_reason"] == "length" else "completed"
        passed += bool(s.B.evaluate(spec, status, "".join(raw["content"]), [c for _, c in sorted(raw["tool_calls"].items())])["passed"])
    tokens = kl_tokens()
    t2 = time.monotonic()
    out = {"sanity": passed, **s.renew([tokens, kl_ref(name, tokens)])}
    for rival, r in RIVALS["rivals"].items():
        if r["model"] == name and "kl" in r:
            out[f"{rival}.kl"] = r["kl"] / out["kl_mean"] if here else None
    return out, {"sanity_s": t2 - t0, "kl_s": time.monotonic() - t2, "slowest_prompt_s": max(e[0] for e in each)}


def serve():
    if not tree_supports(ROOT):
        send({"ready": True})
        for line in sys.stdin:
            send(nulls(line.split()[1]))
        return
    prime = subprocess.run([PY, os.path.join(ROOT, "bench/hill_prime.py")], cwd=ROOT, capture_output=True, text=True)
    if prime.returncode:
        sys.exit((prime.stdout + prime.stderr).strip())
    ready = subprocess.run([PY, os.path.join(ROOT, "bench/hill_ready.py"), name], capture_output=True, text=True)
    if ready.returncode:
        sys.exit((ready.stdout + ready.stderr).strip())
    start = time.monotonic()
    sampler = Sampler()
    s = Server(name)
    try:
        load = {"load_s": s.load_s, "ram_gb": s.peak_ram_gb(), "mem_limited": (s.stats().get("experts") or {}).get("memory_limited")}
        load.update(sampler.stop())
        load["run_s"] = time.monotonic() - start
        runs = {f"speed-{name}": speed, f"quality-{name}": quality}
        speed(s, load)
        send({"ready": True})
        for line in sys.stdin:
            words = line.split()
            if not words:
                continue
            if words[0] != "run" or len(words) != 2 or words[1] not in runs:
                sys.exit(f"worker {name}: unknown command {line.strip()!r}")
            t = time.monotonic()
            out, detail = runs[words[1]](s, load)
            out["detail"] = {**detail, "seconds": time.monotonic() - t}
            send(out)
    finally:
        s.stop()


signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
signal.signal(signal.SIGHUP, lambda *_: sys.exit(1))
serve()
