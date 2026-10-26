import ctypes as C
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chrysalis.engine import load_async, wake_gpus
from chrysalis.models import MODELS
from chrysalis.server import Server, serve

PIN = [(b"adapt_left", 0), (b"split", 1), (b"gfrac", 0.3), (b"experts", 0)]


def pinned(future):
    engine = future.result()
    tune = engine.lib.chrysalis_tune
    tune.restype, tune.argtypes = C.c_double, [C.c_char_p, C.c_double]
    for key, value in PIN:
        tune(key, value)
    return engine


if __name__ == "__main__":
    use, port = sys.argv[1], int(sys.argv[2])
    spec = MODELS[use]
    path = os.path.join(ROOT, spec["path"])
    wake_gpus()
    engine = load_async(path, max_seqs=3, umax=spec["chunk"], warmup=False, lib=os.path.join(ROOT, spec["lib"]))
    if "--pin" in sys.argv:
        pool = ThreadPoolExecutor(1)
        engine = pool.submit(pinned, engine)
        pool.shutdown(wait=False)
    serve(lambda: Server(path, os.path.join(ROOT, "data/models/tokenizer.json"), slots=3, engine=engine, warmup=False, model=use),
          "127.0.0.1", port, model=use)
