import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import BACKENDS, MACHINE_BACKEND, ROOT

model = sys.argv[1] if len(sys.argv) > 1 else None
other = {"27b": "flash-next", "fn": "27b"}.get(model)
skip = {os.path.join(ROOT, "engine", b) for b in BACKENDS if b != MACHINE_BACKEND} | {os.path.join(ROOT, "chrysalis", "voice")}
if MACHINE_BACKEND == "metal":
    skip.add(os.path.join(ROOT, "engine", "gpu"))
if other:
    skip |= {os.path.join(ROOT, "engine", MACHINE_BACKEND, other), os.path.join(ROOT, "engine", "gpu", other)}
if model == "27b" or MACHINE_BACKEND == "metal":
    skip |= {os.path.join(ROOT, "engine", n) for n in ("cpu_exact.cpp", "cpu_exact.h", "cpu_exact_isa.h")}
h = hashlib.sha256(f"{MACHINE_BACKEND} {model}".encode())
for top in ("engine", "chrysalis", "Makefile"):
    path = os.path.join(ROOT, top)
    files = [path] if os.path.isfile(path) else []
    for d, dirs, names in os.walk(path):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__" and os.path.join(d, x) not in skip)
        files += [os.path.join(d, n) for n in sorted(names) if os.path.join(d, n) not in skip]
    for f in files:
        h.update(os.path.relpath(f, ROOT).encode() + b"\0")
        h.update(open(f, "rb").read())
        h.update(b"\0")
print(h.hexdigest())
