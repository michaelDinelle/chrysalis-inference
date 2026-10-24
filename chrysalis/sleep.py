import glob
import shutil
import subprocess
import threading
import time

LOGIND = ["--system", "--dest", "org.freedesktop.login1", "--object-path", "/org/freedesktop/login1"]


def vram_used():
    cards = [(int(open(f).read()), f.replace("_total", "_used")) for f in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")]
    return int(open(max(cards)[1]).read()) if cards else None


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
