import os
import threading
import time

RESERVE = 10 << 30
PIECE = 64 << 20


def mem_free():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemFree:"):
                return int(line.split()[1]) << 10
    return 0


class Warmer:
    def __init__(self, log, busy):
        self.log, self.busy, self.stopping, self.thread = log, busy, threading.Event(), None
        self.cached = {}

    def start(self, paths, own, delay=10.0):
        self.stop()
        self.stopping.clear()
        self.thread = threading.Thread(target=self.run, args=(list(paths), own, delay), daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        if self.thread:
            self.thread.join()
            self.thread = None

    def run(self, paths, own, delay):
        if self.stopping.wait(delay):
            return
        try:
            if mem_free() - RESERVE < sum(os.path.getsize(p) for p in paths):
                fd = os.open(own, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
        except OSError:
            pass
        buf = bytearray(PIECE)
        for path in paths:
            t0, done = time.time(), 0
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                size = os.fstat(fd).st_size
                while done < size and not self.stopping.is_set() and mem_free() > RESERVE + PIECE:
                    if self.busy():
                        self.stopping.wait(0.5)
                        continue
                    n = os.preadv(fd, [buf], done)
                    if n <= 0:
                        break
                    done += n
            finally:
                os.close(fd)
            self.cached[path] = done
            if not self.stopping.is_set() and done:
                print(f"chrysalis: {done / 1e9:.1f} of {size / 1e9:.1f} GB of {os.path.basename(path)} read into the page cache in "
                      f"{time.time() - t0:.0f} s, for a switch to it", file=self.log, flush=True)
