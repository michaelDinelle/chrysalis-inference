import glob
import os


def device():
    best, size = None, 0
    for card in glob.glob("/sys/class/drm/card[0-9]*/device"):
        try:
            vram = int(open(os.path.join(card, "mem_info_vram_total")).read())
        except (OSError, ValueError):
            continue
        if vram > size:
            best, size = os.path.realpath(card), vram
    return best


def sharers(dev):
    nodes = {"/dev/dri/" + os.path.basename(n) for k in ("card*", "renderD*") for n in glob.glob(os.path.join(dev, "drm", k))}
    me, found = os.getpid(), {}
    for proc in glob.glob("/proc/[0-9]*"):
        pid = int(proc[6:])
        if pid == me:
            continue
        try:
            fds = os.listdir(proc + "/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(f"{proc}/fd/{fd}") in nodes:
                    found[pid] = open(proc + "/comm").read().strip()
                    break
            except OSError:
                continue
    return found


def reserve_mb():
    dev = device()
    return 256 + (2048 if dev and sharers(dev) else 0)
