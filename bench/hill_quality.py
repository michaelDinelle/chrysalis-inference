import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hill_lib import RIVALS, Server, rivals_here, settle
from measure import BENCH, PY, ROOT, kl_ref, kl_tokens

name = sys.argv[1]
s = Server(name)
passed = 0
for spec in (x for x in s.B.load_corpus().specs if x["phase"] == "sanity"):
    raw, _ = s.ask(spec)
    status = "truncated" if raw["finish_reason"] == "length" else "completed"
    passed += bool(s.B.evaluate(spec, status, "".join(raw["content"]), [c for _, c in sorted(raw["tool_calls"].items())])["passed"])
s.stop()
tokens = kl_tokens()
kl = subprocess.run([PY, os.path.join(ROOT, "bench/kl.py"), name, tokens, kl_ref(name, tokens)], capture_output=True, text=True,
                    env={**os.environ, **BENCH[name]["env"]})
settle()
if kl.returncode:
    sys.exit(f"{name} KL failed:\n{kl.stderr[-2000:]}")
out = {"sanity": passed, **json.loads(kl.stdout.strip().split("\n")[-1])}
here = rivals_here()
for rival, r in RIVALS["rivals"].items():
    if r["model"] == name and "kl" in r:
        out[f"{rival}.kl"] = r["kl"] / out["kl_mean"] if here else None
print(json.dumps(out))
