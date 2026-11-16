import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chrysalis.cache import Cache, Disk
from chrysalis.engine import load_async, wake_gpus
from chrysalis.server import Server, serve
from measure import BENCH, MODELS, own_calibration, tuned

class Kept(Disk):
    def save(self, engine, snap, tokens):
        if PRIME:
            super().save(engine, snap, tokens)


PRIME = bool(os.environ.get("CHRYSALIS_BENCH_PRIME"))

if __name__ == "__main__":
    name, port = sys.argv[1], int(sys.argv[2])
    use = BENCH[name]["use"]
    spec = MODELS[use]
    path = os.path.join(ROOT, spec["path"])
    kept = os.environ.get("CHRYSALIS_BENCH_SNAPS")
    if kept:
        identity = f"bench|{os.path.basename(path)}:{os.stat(path).st_size}"
        Server.snapshot_disk = lambda self, n, p: Kept(kept, identity, 32 << 30) if self.engine.persistent else None
        if PRIME:
            checkpoint = Cache.checkpoint
            Cache.checkpoint = lambda self, slot, pinned=False, logits=False, kept=True: checkpoint(self, slot, True, logits, kept)
    own_calibration()
    wake_gpus()
    loading = load_async(path, max_seqs=3, umax=spec["chunk"], warmup=False, lib=os.path.join(ROOT, spec["lib"]))
    pool = ThreadPoolExecutor(1)
    engine = pool.submit(lambda: tuned(loading.result(), name))
    pool.shutdown(wait=False)
    serve(lambda: Server(path, os.path.join(ROOT, "data/models/tokenizer.json"), slots=3, engine=engine, warmup=False, model=use),
          "127.0.0.1", port, model=use)
