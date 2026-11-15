import ctypes
import glob
import re
import shutil
import subprocess
import sys
import threading
import time

LOGIND = ["--system", "--dest", "org.freedesktop.login1", "--object-path", "/org/freedesktop/login1"]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("total", ctypes.c_ulonglong), ("reserved", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


_nvml = []


def _nvml_used():
    if not _nvml:
        found = None
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() == 0:
                n, handles = ctypes.c_uint(0), []
                lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
                for i in range(n.value):
                    h = ctypes.c_void_p()
                    if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) == 0:
                        handles.append(h)
                found = (lib, handles) if handles else None
        except (OSError, AttributeError):
            pass
        _nvml.append(found)
    if not _nvml[0]:
        return None
    lib, handles = _nvml[0]
    best = None
    for h in handles:
        info = _NvmlMemory(version=ctypes.sizeof(_NvmlMemory) | 2 << 24)
        if lib.nvmlDeviceGetMemoryInfo_v2(h, ctypes.byref(info)) == 0 and (best is None or info.total > best.total):
            best = info
    return best.used if best else None


def vram_used():
    if sys.platform == "darwin":
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
        found = re.search(r'"In use system memory"=(\d+)', out)
        return int(found.group(1)) if found else None
    cards = [(int(open(f).read()), f.replace("_total", "_used")) for f in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")]
    return int(open(max(cards)[1]).read()) if cards else _nvml_used()


def wait_vram_freed(held, timeout=3.0):
    end, last, still = time.time() + timeout, vram_used(), time.time()
    while last is not None and time.time() < end and (last >= held / 2 or time.time() - still < 0.05):
        time.sleep(0.002)
        now = vram_used()
        if now != last:
            last, still = now, time.time()


class SleepGuard:
    def __init__(self, before, after, log):
        self.before, self.after, self.log = before, after, log
        self.inhibit = None
        self.slept = False

    def start(self):
        if sys.platform == "win32":
            return False
        if not (shutil.which("systemd-inhibit") and shutil.which("gdbus")):
            print("chrysalis: no systemd-inhibit or gdbus: the GPU is not freed before the machine sleeps", file=self.log, flush=True)
            return False
        self.hold()
        mon = subprocess.Popen(["gdbus", "monitor", *LOGIND], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self.watch, args=(mon,), daemon=True).start()
        return True

    def hold(self):
        self.inhibit = subprocess.Popen(["systemd-inhibit", "--what=sleep", "--mode=delay", "--who=Chrysalis Inference",
                                         "--why=Frees the GPU before the machine sleeps", "cat"],
                                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def release(self):
        if self.inhibit:
            self.inhibit.stdin.close()
            self.inhibit.wait()
            self.inhibit = None

    def watch(self, mon):
        for line in mon.stdout:
            if ".PrepareForSleep (true" in line:
                try:
                    self.before()
                    self.slept = True
                finally:
                    self.release()
            elif ".PrepareForSleep (false" in line:
                if not self.inhibit:
                    self.hold()
                if self.slept:
                    self.slept = False
                    self.after()
