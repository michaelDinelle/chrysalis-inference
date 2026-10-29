import glob
import hashlib
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hill_lib import DOCS, SNAPS, Server, primed
from measure import PY, ROOT, WORK

for name in ("fn", "27b"):
    folder = os.path.join(SNAPS, name)
    mark = os.path.join(folder, "primed-" + hashlib.sha256(repr(sorted(DOCS.items())).encode()).hexdigest()[:16])
    if os.path.exists(mark):
        continue
    ready = subprocess.run([PY, os.path.join(ROOT, "bench/hill_ready.py"), name], capture_output=True, text=True)
    if ready.returncode:
        sys.exit(f"priming the {name} bench snapshots: {(ready.stdout + ready.stderr).strip()}")
    shutil.rmtree(folder, ignore_errors=True)
    s = Server(name, env=["CHRYSALIS_BENCH_PRIME=1"])
    for case, (doc, _) in DOCS.items():
        s.ask({"id": f"prime-{case}", "messages": primed(doc), "max_output_tokens": 1, "seed": 1})
    seen, still, deadline = None, 0, time.monotonic() + 1800
    while still < 15:
        now = sorted(glob.glob(os.path.join(folder, "*")))
        settled = now == seen and not any(p.endswith(".tmp") for p in now) and sum(p.endswith(".snap") for p in now) >= 2 * len(DOCS)
        still, seen = still + 1 if settled else 0, now
        if time.monotonic() > deadline:
            s.stop()
            sys.exit(f"the {name} bench snapshots were not written")
        time.sleep(0.2)
    s.stop()
    open(mark, "w").close()
if not os.path.exists(os.path.join(WORK, "flash-next-calibration.txt")):
    ready = subprocess.run([PY, os.path.join(ROOT, "bench/hill_ready.py"), "fn"], capture_output=True, text=True)
    if ready.returncode:
        sys.exit(f"warming this tree's Flash-Next calibration: {(ready.stdout + ready.stderr).strip()}")
    Server("fn").stop()
