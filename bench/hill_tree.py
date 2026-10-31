import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import MACHINE_BACKEND, PY, ROOT, tree_supports

if tree_supports(ROOT):
    subprocess.run(["make", "-s", "-j", f"BACKEND={MACHINE_BACKEND}", "engines"], cwd=ROOT, check=True)
    subprocess.run([PY, os.path.join(ROOT, "bench/hill_prime.py")], cwd=ROOT, check=True)
