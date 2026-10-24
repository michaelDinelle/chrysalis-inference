import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST = os.path.join(ROOT, "data", "cache", "link.json")
TRY = "CHRYSALIS_LINK_TRY"


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


def render_node(dev):
    nodes = glob.glob(os.path.join(dev, "drm", "renderD*"))
    return "/dev/dri/" + os.path.basename(nodes[0]) if nodes else None


def read(dev):
    try:
        if open(os.path.join(dev, "power", "runtime_status")).read().strip() != "active":
            return None
    except OSError:
        pass
    slowest, p = None, dev
    while os.path.exists(os.path.join(p, "current_link_speed")):
        try:
            gts = float(open(os.path.join(p, "current_link_speed")).read().split()[0])
            width = int(open(os.path.join(p, "current_link_width")).read())
        except (OSError, ValueError, IndexError):
            break
        if slowest is None or gts * width < slowest["gts"] * slowest["width"]:
            slowest = {"gts": gts, "width": width}
        p = os.path.dirname(p)
    return slowest


def best():
    try:
        with open(BEST) as f:
            b = json.load(f)
        return {"gts": float(b["gts"]), "width": int(b["width"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def slower(link, than):
    return bool(link and than) and link["gts"] * link["width"] < than["gts"] * than["width"]


def remember(link):
    b = best()
    if link and (b is None or slower(b, link)):
        os.makedirs(os.path.dirname(BEST), exist_ok=True)
        with open(BEST + ".tmp", "w") as f:
            json.dump(link, f)
        os.replace(BEST + ".tmp", BEST)
        return link
    return b


def status(dev):
    link, b = read(dev) if dev else None, best()
    return link and {**link, "best_gts": b and b["gts"], "best_width": b and b["width"], "slow": slower(link, b)}


def wait_suspended(dev, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if open(os.path.join(dev, "power", "runtime_status")).read().strip() == "suspended":
                return True
        except OSError:
            return False
        time.sleep(0.25)
    return False


def check(dev, tries, log=sys.stderr):
    link = read(dev)
    if link is None:
        return None
    b = remember(link)
    done = int(os.environ.get(TRY, "0"))
    if not slower(link, b):
        if done:
            print(f"chrysalis: PCIe link at {link['gts']:g} GT/s x{link['width']} after {done} "
                  f"{'restart' if done == 1 else 'restarts'}", file=log, flush=True)
        return link
    if done >= tries:
        print(f"chrysalis: PCIe link at {link['gts']:g} GT/s x{link['width']}, slower than this machine's best "
              f"({b['gts']:g} GT/s x{b['width']}): prefill and Flash-Next run slower until Chrysalis restarts", file=log, flush=True)
        return link
    print(f"chrysalis: PCIe link at {link['gts']:g} GT/s x{link['width']}, slower than this machine's best "
          f"({b['gts']:g} GT/s x{b['width']}): restarting to train it again ({done + 1} of {tries})", file=log, flush=True)
    restart(done + 1)


def restart(n):
    os.environ[TRY] = str(n)
    for fd in os.listdir("/proc/self/fd"):
        fd = int(fd)
        if fd > 2:
            try:
                os.set_inheritable(fd, False)
            except OSError:
                pass
    os.execv(sys.executable, [sys.executable] + sys.orig_argv[1:])
