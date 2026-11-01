import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import BACKENDS, MACHINE_BACKEND, ROOT

others = {os.path.join(ROOT, "engine", b) for b in BACKENDS if b != MACHINE_BACKEND}
h = hashlib.sha256(MACHINE_BACKEND.encode())
for top in ("engine", "chrysalis", "Makefile"):
    path = os.path.join(ROOT, top)
    files = [path] if os.path.isfile(path) else []
    for d, dirs, names in os.walk(path):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__" and os.path.join(d, x) not in others)
        files += [os.path.join(d, n) for n in sorted(names)]
    for f in files:
        h.update(os.path.relpath(f, ROOT).encode() + b"\0")
        h.update(open(f, "rb").read())
        h.update(b"\0")
print(h.hexdigest())
