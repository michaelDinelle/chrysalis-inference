import ctypes as C
import ctypes.util
import errno
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import BENCH, MAC, MACHINE_BACKEND, NVML, PY, RECORDS, ROOT, WORK, drop_cache, free_bytes, free_port, http, peak_footprint, scope, \
    server_pid, strata

SNAPS = os.path.join(ROOT, "data/cache/bench-snapshots")
RIVALS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rivals.json")))
DOCS = {"16k": (RECORDS(1, 841), RECORDS(841, 895)), "100k": (RECORDS(1, 5201), RECORDS(5201, 5255))}
QUESTION = "\n\nThese records were added since. Which ID is the last one now? Then describe how the records are laid out."


def primed(doc):
    return [{"role": "user", "content": doc}, {"role": "assistant", "content": "Noted."}, {"role": "user", "content": "Ready?"}]


def deep(doc, more):
    return primed(doc) + [{"role": "assistant", "content": "Yes."}, {"role": "user", "content": more + QUESTION}]


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


def unsupported(instrument):
    spec = json.load(open(os.path.join(ROOT, "hill.json")))
    print(json.dumps({r.get("field", k): None for k, r in spec["rewards"].items() if r["instrument"] == instrument}))
    sys.exit(0)


def free_gb():
    if not MAC:
        return free_bytes() / 1e9
    libc = C.CDLL(ctypes.util.find_library("c"))
    level, total = C.c_int(0), C.c_uint64(0)
    libc.sysctlbyname(b"kern.memorystatus_level", C.byref(level), C.byref(C.c_size_t(4)), None, 0)
    libc.sysctlbyname(b"hw.memsize", C.byref(total), C.byref(C.c_size_t(8)), None, 0)
    return level.value / 100 * total.value / 1e9


def unused_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return sum(int(out.split(k + ":")[1].split()[0].rstrip(".")) for k in ("Pages free", "Pages inactive", "Pages speculative")) * page / 2**30


def settle(before=None):
    deadline = time.monotonic() + 60
    if MAC:
        seen, start = [], time.monotonic()
        while before is not None and time.monotonic() < deadline:
            seen.append(unused_gb())
            if seen[-1] >= before - 1 or time.monotonic() - start >= 3 and len(seen) > 10 and seen[-1] < seen[-11] + 0.25:
                break
            time.sleep(0.2)
        return
    if NVML:
        while NVML.used() > 256 << 20 and time.monotonic() < deadline:
            time.sleep(0.05)
        return
    for total in glob.glob("/sys/class/drm/card[0-9]*/device/mem_info_vram_total"):
        if int(open(total).read()) >= 8 << 30:
            used = os.path.join(os.path.dirname(total), "mem_info_vram_used")
            while int(open(used).read()) > 256 << 20 and time.monotonic() < deadline:
                time.sleep(0.05)


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.power, self.joules, self.low, self.done = None if MAC or NVML else gpu_power(), 0.0, free_gb(), threading.Event()
        self.mj0 = NVML.energy_mj() if NVML else None
        self.start()

    def run(self):
        last = time.monotonic()
        while not self.done.wait(0.1):
            now = time.monotonic()
            if self.power:
                try:
                    self.joules += int(open(self.power).read()) / 1e6 * (now - last)
                except OSError as e:
                    if e.errno != errno.EBUSY:
                        raise
            self.low, last = min(self.low, free_gb()), now

    def stop(self):
        self.done.set()
        self.join()
        if self.mj0 is not None:
            mj = NVML.energy_mj()
            return {"energy_kj": (mj - self.mj0) / 1e6 if mj is not None else None, "min_free_gb": self.low}
        return {"energy_kj": self.joules / 1000 if self.power else None, "min_free_gb": self.low}


class Server:
    def __init__(self, name, env=()):
        self.name, self.use, b = name, BENCH[name]["use"], BENCH[name]
        for f in b["files"]:
            drop_cache(os.path.join(ROOT, f))
        os.makedirs(WORK, exist_ok=True)
        self.before = unused_gb() if MAC else None
        port, self.snap, self.log = free_port(), tempfile.mkdtemp(prefix=f"snap-{name}-", dir=WORK), os.path.join(WORK, f"server-{name}.log")
        env = [f"CHRYSALIS_SNAPSHOTS={self.snap}", f"CHRYSALIS_BENCH_SNAPS={os.path.join(SNAPS, name)}", *env, *(f"{k}={v}" for k, v in b["env"].items())]
        if MAC:
            env.append(f"CHRYSALIS_FN_STATS={os.path.join(self.snap, 'experts.bin')}")
        wrap = scope()
        self.url = f"http://127.0.0.1:{port}"
        t0 = time.monotonic()
        self.srv = subprocess.Popen([*wrap, "env", *env, PY, os.path.join(ROOT, "bench/serve.py"), name, str(port)], stdout=open(self.log, "w"),
                                    stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            if self.srv.poll() is not None:
                self.stop()
                sys.exit(f"the {name} server exited while loading (see {self.log})")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(0.05)
        http(self.url + "/v1/chat/completions", {"model": self.use, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1, "stream": False,
                                                  "chat_template_kwargs": {"enable_thinking": False}})
        self.load_s = time.monotonic() - t0
        self.B = strata()
        self.target = self.B.Target(self.url, None)

    def ask(self, spec):
        raw = self.B.stream_chat(self.target, spec, self.use, timeout_s=self.B.REQUEST_TIMEOUT_S)
        if raw["error"]:
            self.stop()
            sys.exit(f"{self.name} {spec['id']}: {raw['error']}")
        stat = self.stats()["recent"][-1]
        if stat.get("completion_tokens") != (raw["usage"] or {}).get("completion_tokens"):
            self.stop()
            sys.exit(f"{self.name} {spec['id']}: the server's last reply is not this request's")
        return raw, stat

    def stats(self):
        return http(self.url + "/chrysalis/stats")

    def peak_ram_gb(self):
        pid = server_pid(self.srv.pid)
        if MAC:
            return peak_footprint(pid) / 1e9
        return next(int(l.split()[1]) for l in open(f"/proc/{pid}/status") if l.startswith("VmHWM:")) * 1024 / 1e9

    def stop(self):
        try:
            os.killpg(self.srv.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.srv.wait()
        shutil.rmtree(self.snap, ignore_errors=True)
        settle(self.before)
