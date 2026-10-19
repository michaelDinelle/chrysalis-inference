"""The other model's file read into the page cache while this one runs, so a switch to it reads RAM instead of the disk.
Flash-Next's load reads all of its file (63 GB, more than this machine's RAM) at the disk's ~4 GB/s: 17 s from nothing
cached, 8 s with 40 GB of it cached (measured). The 27B leaves most of the RAM free, Flash-Next does not (its experts
take 43 GB, and its working set lives in the page cache: the n-gram table, the file tier's experts, which another file
would push out), so only a model whose MODELS entry says "warm" reads the others' files. Reads pause while requests
run (the disk has no I/O scheduler, so priorities would not hold: a shared prefix read back from disk would wait) and
stop where less than RESERVE of free memory would be left: MemAvailable counts the page cache as free, and reading past
what fits would only push out the file's own first pages."""
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
        """busy(): requests are running."""
        self.log, self.busy, self.stopping, self.thread = log, busy, threading.Event(), None
        self.cached = {}                        # path -> bytes read on the last pass

    def start(self, paths, own, delay=10.0):
        """Reads paths into the page cache, one after another, starting `delay` seconds from now (after the load that
        just finished has the disk to itself). own: the running model's file, whose pages leave the cache when the others
        would not fit beside them (its weights are on the GPU: only a reload after a restart or a wake reads it, from the
        disk then: 3.9 s for the 27B instead of 1.5)."""
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
            t0, done = time.time(), 0               # the time includes the pauses
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
