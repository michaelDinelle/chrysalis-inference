import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import NVML, free_bytes

NEED_GB = {"linux": {"fn": 52, "27b": 12}, "darwin": {"fn": 30, "27b": 27}}
name = sys.argv[1]
procs = [p.split() for p in subprocess.run(["ps", "-axo", "pid=,args="], capture_output=True, text=True).stdout.splitlines()]
busy = [" ".join(p[1:]) for p in procs if p[2:5] == ["-m", "chrysalis", "serve"] or len(p) > 2 and p[2].endswith("bench/serve.py")]
if busy:
    sys.exit(f"a Chrysalis model server is running: {busy[0]}")
if sys.platform != "darwin":
    for total in glob.glob("/sys/class/drm/card[0-9]*/device/mem_info_vram_total"):
        d = os.path.dirname(total)
        if int(open(total).read()) < 8 << 30:
            continue
        where = os.path.basename(os.path.realpath(d))
        if any(open(s).read().strip() == "connected" for s in glob.glob(os.path.join(d, "drm", "card*", "card*-*", "status"))):
            sys.exit(f"a display is connected to the GPU at {where}: the desktop would share its memory with the bench")
        used = int(open(os.path.join(d, "mem_info_vram_used")).read())
        if used > 256 << 20:
            sys.exit(f"{used / 1e9:.1f} GB of video memory on the GPU at {where} is in use by another program")
    if NVML:
        if NVML.display():
            sys.exit("a display is connected to the NVIDIA GPU: the desktop would share its memory with the bench")
        if NVML.used() > 256 << 20:
            sys.exit(f"{NVML.used() / 1e9:.1f} GB of video memory on the NVIDIA GPU is in use by another program")
    free = free_bytes() / 1e9
else:
    sysctl = lambda key: int(subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout)
    free = sysctl("kern.memorystatus_level") / 100 * sysctl("hw.memsize") / 1e9
need = NEED_GB[sys.platform][name]
if free < need:
    sys.exit(f"the {name} bench needs {need} GB of free memory here and {free:.1f} GB is free")
