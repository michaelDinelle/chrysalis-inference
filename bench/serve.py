import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import chrysalis.server as cs
from chrysalis.cache import Cache, Disk
from chrysalis.engine import load_async, wake_gpus
from chrysalis.server import LRU, Renderer, Server, Sticky, serve
from kl import kl_pass
from measure import BENCH, MODELS, own_calibration, tuned

class Kept(Disk):
    def save(self, engine, snap, tokens):
        if PRIME:
            super().save(engine, snap, tokens)


PRIME = bool(os.environ.get("CHRYSALIS_BENCH_PRIME"))
POOL = os.path.join(cs.SNAPSHOT_DIR, "bench-pool.bin")


def renew(srv, path, kl=None):
    e, spec = srv.engine, MODELS[srv.active]
    srv.gpu.acquire(main=True)
    try:
        for en in srv.cache.entries:
            e.free_snapshot(en.snap)
        for slot in srv.cache.slots:
            e.free_seq(slot.sid)
        out = {}
        for step in ([kl_pass] if kl else []) + [None]:
            if os.path.exists(POOL):
                e.warmup()
                e.pool_warm(POOL)
            if step:
                out = step(e, *kl)
        srv.cache = Cache(e, srv.slots, max_snapshots=spec["snapshots"], disk=srv.snapshot_disk(srv.active, path))
        srv.sticky, srv.sessions, srv.stored, srv.replay, srv.recent = Sticky(), LRU(64), LRU(256), LRU(8192), LRU(64)
        srv.renderer = Renderer(srv.tok, srv.replay.get)
        return out
    finally:
        srv.gpu.release()


def bench_routes(path):
    made = cs.make_handler

    def make_handler(state, ready):
        class Handler(made(state, ready)):
            def do_POST(self):
                if self.path.rstrip("/") != "/bench/renew":
                    return super().do_POST()
                ready.wait()
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                self.send_json(200, renew(state["srv"], path, body.get("kl")))
        return Handler
    cs.make_handler = make_handler


def kept_pool(srv):
    e = srv.engine

    def warmup():
        del e.warmup
        e.warmup()
        e.pool_save(POOL)
    e.warmup = warmup
    return srv


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
    bench_routes(path)
    own_calibration()
    wake_gpus()
    loading = load_async(path, max_seqs=3, umax=spec["chunk"], warmup=False, lib=os.path.join(ROOT, spec["lib"]))
    pool = ThreadPoolExecutor(1)
    engine = pool.submit(lambda: tuned(loading.result(), name))
    pool.shutdown(wait=False)
    serve(lambda: kept_pool(Server(path, os.path.join(ROOT, "data/models/tokenizer.json"), slots=3, engine=engine, warmup=False, model=use)),
          "127.0.0.1", port, model=use)
