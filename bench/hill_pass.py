import ctypes as C
import ctypes.util
import glob
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import CASES, MAC, run_pass

RIVALS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rivals.json")))


def rivals_here():
    if MAC:
        return False
    want = RIVALS["machine"]
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    read = lambda p: open(p).read().strip()
    cards = [[read(f"{d}/device"), read(f"{d}/subsystem_device")] for d in glob.glob("/sys/class/drm/card*/device") if os.path.exists(f"{d}/subsystem_device")]
    return cpu == want["cpu"] and want["gpu"] in cards


def gpu_power():
    for total in glob.glob("/sys/class/drm/card[0-9]*/device/mem_info_vram_total"):
        if int(open(total).read()) >= 8 << 30:
            found = glob.glob(os.path.join(os.path.dirname(total), "hwmon", "hwmon*", "power1_average"))
            if found:
                return found[0]
    return None


def free_gb():
    if not MAC:
        return next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")) * 1024 / 1e9
    libc = C.CDLL(ctypes.util.find_library("c"))
    level, total = C.c_int(0), C.c_uint64(0)
    libc.sysctlbyname(b"kern.memorystatus_level", C.byref(level), C.byref(C.c_size_t(4)), None, 0)
    libc.sysctlbyname(b"hw.memsize", C.byref(total), C.byref(C.c_size_t(8)), None, 0)
    return level.value / 100 * total.value / 1e9


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.power, self.joules, self.low, self.done = None if MAC else gpu_power(), 0.0, free_gb(), threading.Event()

    def run(self):
        last = time.monotonic()
        while not self.done.wait(0.1):
            now = time.monotonic()
            if self.power:
                self.joules += int(open(self.power).read()) / 1e6 * (now - last)
            self.low, last = min(self.low, free_gb()), now


name = sys.argv[1]
sampler = Sampler()
sampler.start()
m = run_pass(name, with_kl=True)["metrics"]
sampler.done.set()
sampler.join()
out = {}
for case in CASES:
    out[f"ttft.{case}"] = m[f"ttft.{case}"]
    out[f"prompt_tps.{case}"] = m[f"prompt_tps.{case}"]
    out[f"decode_tps.{case}"] = m[f"decode_tps.{case}"]
    out[f"gap_ms.{case}"] = 1000 * m[f"itl_p99.{case}"]
    out[f"per_pass.{case}"] = m[f"per_pass.{case}"]
for key in ("load_s", "ram_gb", "kv_tokens", "reused.followup", "sanity", "kl_mean", "kl_p999", "kl_max", "top1", "mem_limited"):
    out[key] = m.get(key)
out["energy_kj"] = sampler.joules / 1000 if sampler.power else None
out["min_free_gb"] = sampler.low
here = rivals_here()
for rival, r in RIVALS["rivals"].items():
    if r["model"] != name:
        continue
    for i, case in enumerate(CASES):
        out[f"{rival}.ttft.{case}"] = r["ttft"][i] / m[f"ttft.{case}"] if here else None
        out[f"{rival}.decode.{case}"] = m[f"decode_tps.{case}"] / r["tps"][i] if here else None
    if "kl" in r:
        out[f"{rival}.kl"] = r["kl"] / m["kl_mean"] if here else None
print(json.dumps(out))
