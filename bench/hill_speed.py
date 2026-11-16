import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hill_lib import DOCS, RIVALS, Sampler, Server, deep, rivals_here, unsupported
from measure import ROOT, STRATA_CASES, p99, tree_supports

start = time.monotonic()
name, here = sys.argv[1], rivals_here()
if not tree_supports(ROOT):
    unsupported(f"speed-{name}")
sampler = Sampler()
s = Server(name)
out = {"load_s": s.load_s}
specs = {x["id"]: x for x in s.B.load_corpus().specs}


profile = {}


def record(case, spec, deep_from=None):
    x0 = s.stats().get("experts") or {}
    raw, stat = s.ask(spec)
    x1 = s.stats().get("experts") or {}
    d = {k: x1[k] - x0.get(k, 0) for k in x1 if isinstance(x1[k], (int, float))}
    steps = max(d.get("decode_steps", 0), 1)
    profile[case] = {"ms_per_step": 1000 * stat["tokens_per_step"] / stat["decode_tps"], "tokens_per_step": stat["tokens_per_step"],
                     "pool_hits": d.get("decode_picks_in_pool", 0) / max(d.get("decode_picks", 0), 1),
                     "cold_picks_per_step": (d.get("decode_picks", 0) - d.get("decode_picks_in_pool", 0)) / steps,
                     "moves_per_step": d.get("placed", 0) / steps, "from_file_per_step": d.get("placed_file_tier", 0) / steps,
                     "waited_ms_per_step": 1000 * d.get("waited_s", 0) / steps, "prefill_s": stat["prefill_s"]}
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
out["mem_limited"] = (stats.get("experts") or {}).get("memory_limited")
out["ram_gb"] = s.peak_ram_gb()
out.update(sampler.stop())
out["run_s"] = time.monotonic() - start
s.stop()
for rival, r in RIVALS["rivals"].items():
    if r["model"] == name:
        for i, case in enumerate(("short", "4k")):
            out[f"{rival}.ttft.{case}"] = r["ttft"][i] / out[f"ttft.{case}"] if here else None
            out[f"{rival}.decode.{case}"] = out[f"decode_tps.{case}"] / r["tps"][i] if here else None
json.dump(profile, open(os.path.join(ROOT, "build", "bench", f"profile-{name}.json"), "w"))
print(json.dumps(out))
