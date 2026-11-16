import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

tree = os.getcwd()
sys.path.insert(0, os.path.join(tree, "bench"))
from measure import MACHINE_BACKEND, tree_supports


def build():
    if not tree_supports(tree):
        return {"build_s": None, "lib_27b_bytes": None, "lib_fn_bytes": None}
    work = os.path.join(tempfile.gettempdir(), "chrysalis-build")
    shutil.rmtree(work, ignore_errors=True)
    try:
        shutil.copytree(os.path.join(tree, "engine"), os.path.join(work, "engine"))
        shutil.copy2(os.path.join(tree, "Makefile"), work)
        os.mkdir(os.path.join(work, "tests"))
        shutil.copy2(os.path.join(tree, "tests/tests.mk"), os.path.join(work, "tests"))
        os.symlink(os.path.join(tree, "data"), os.path.join(work, "data"))
        t = time.monotonic()
        out = subprocess.run(["make", "-s", "-j", f"BACKEND={MACHINE_BACKEND}", "engines"], cwd=work, capture_output=True, text=True)
        took = time.monotonic() - t
        if out.returncode:
            sys.exit(f"the clean build failed:\n{out.stderr[-2000:]}")
        lib = "dylib" if sys.platform == "darwin" else "so"
        size = lambda name: os.path.getsize(os.path.join(work, "build", f"{name}.{lib}"))
        return {"build_s": took, "lib_27b_bytes": size("libchrysalis"), "lib_fn_bytes": size("libchrysalis_fn")}
    finally:
        shutil.rmtree(work, ignore_errors=True)


print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    if line.split():
        print(json.dumps(build()), flush=True)
