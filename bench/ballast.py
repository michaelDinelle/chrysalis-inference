import ctypes
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import MACHINE_BACKEND, ROOT

gb, device = float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0
if MACHINE_BACKEND == "hip":
    lib, pre = ctypes.CDLL(os.path.join(ROOT, "data/rocm/lib/libamdhip64.so")), "hip"
elif MACHINE_BACKEND == "cuda":
    lib, pre = ctypes.CDLL("libcudart.so"), "cuda"
else:
    sys.exit("ballast: this machine's GPU memory is the system's own")
call = lambda name, *args: getattr(lib, pre + name)(*args) and sys.exit(f"ballast: {pre}{name} failed")
call("SetDevice", device)
held, size = ctypes.c_void_p(), ctypes.c_size_t(int(gb * 2 ** 30))
call("Malloc", ctypes.byref(held), size)
call("Memset", held, 0, size)
call("DeviceSynchronize")
print(json.dumps({"ready": True, "held_gb": gb}), flush=True)
sys.stdin.read()
