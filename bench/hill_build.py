import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

tree = os.getcwd()
work = tempfile.mkdtemp(prefix="chrysalis-build-")
try:
    shutil.copytree(os.path.join(tree, "engine"), os.path.join(work, "engine"))
    shutil.copy2(os.path.join(tree, "Makefile"), work)
    os.symlink(os.path.join(tree, "data"), os.path.join(work, "data"))
    t = time.monotonic()
    out = subprocess.run(["make", "-s", "-j", "engines"], cwd=work, capture_output=True, text=True)
    took = time.monotonic() - t
    if out.returncode:
        sys.exit(f"the clean build failed:\n{out.stderr[-2000:]}")
    lib = "dylib" if sys.platform == "darwin" else "so"
    size = lambda name: os.path.getsize(os.path.join(work, "build", f"{name}.{lib}"))
    print(json.dumps({"build_s": took, "lib_27b_bytes": size("libchrysalis"), "lib_fn_bytes": size("libchrysalis_fn")}))
finally:
    shutil.rmtree(work, ignore_errors=True)
