import os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chrysalis.server import Held, Lanes, Turnstile, admit

gpu, lanes = Turnstile(), Lanes(3)
waited, done = {}, {}


def title():
    t = time.time()
    admit(lanes, gpu, True, t)
    held = Held(gpu, main=False, arrived=t)
    held.take()
    done["title"] = time.time()
    held.drop()
    lanes.leave()


def agent(k, work=0.6):
    t = time.time()
    admit(lanes, gpu, False, t)
    waited[k] = time.time() - t
    end = time.time() + work
    while time.time() < end:
        gpu.acquire(True)
        time.sleep(0.01)
        gpu.release(True)
        time.sleep(0.002)
    done[k] = time.time()
    lanes.leave()


threads = [threading.Thread(target=title)] + [threading.Thread(target=agent, args=(k,)) for k in range(3)]
threads[0].start()
time.sleep(0.008)
for t in threads[1:]:
    t.start()
for t in threads:
    t.join(10)
assert all(k in waited for k in range(3)) and "title" in done, f"a request never ran: {sorted(map(str, [*waited, *done]))}"
assert max(waited.values()) < 0.1, f"an agent request waited {max(waited.values()):.2f} s for a lane behind the title"
assert done["title"] >= max(done[k] for k in range(3)), "the title ran before the agent requests were done"
print("a title waiting for the GPU keeps no agent request out of a lane: ok")
