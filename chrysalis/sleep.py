"""The GPU is freed before the machine sleeps and the model loads again after it wakes, so the machine never sleeps on
the model's 24 GB of VRAM (amdgpu would evict all of it to system RAM, which Flash-Next has already taken most of).

logind announces a sleep (suspend, hibernate) to everyone holding a delay lock and waits until they let go of it, up to
InhibitDelayMaxSec (15 s here); the lock is open to any user. systemd-inhibit holds it for as long as its child (cat on
a pipe) runs, and gdbus reports logind's PrepareForSleep signal: true before the sleep, false after the wake."""
import glob
import shutil
import subprocess
import threading
import time

LOGIND = ["--system", "--dest", "org.freedesktop.login1", "--object-path", "/org/freedesktop/login1"]


def vram_used():
    """Bytes in use on the amdgpu device with the most VRAM (Chrysalis's), or None without one."""
    cards = [(int(open(f).read()), f.replace("_total", "_used")) for f in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")]
    return int(open(max(cards)[1]).read()) if cards else None


def wait_vram_freed(held, timeout=3.0):
    """After a device reset the driver hands the VRAM back in pieces (measured: the 27B's 25.7 GB down to 0.35 in 0.14 s;
    Flash-Next's to 20.8 GB at once and the rest 90 ms later): wait until less than half of `held` (the bytes in use before
    the reset) remains and that holds still for 50 ms, so the machine sleeps with the memory released."""
    end, last, still = time.time() + timeout, vram_used(), time.time()
    while last is not None and time.time() < end and (last >= held / 2 or time.time() - still < 0.05):
        time.sleep(0.002)
        now = vram_used()
        if now != last:
            last, still = now, time.time()


class SleepGuard:
    def __init__(self, before, after, log):
        """before(): free the GPU (returns once it is free); after(): load the model again."""
        self.before, self.after, self.log = before, after, log
        self.inhibit = None
        self.slept = False                      # before() ran: after() is due at the wake

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
        for line in mon.stdout:                 # "/org/freedesktop/login1: org.freedesktop.login1.Manager.PrepareForSleep (true,)"
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
