import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chrysalis.engine import load_async, wake_gpus
from chrysalis.server import Server, serve
from measure import BENCH, MODELS, tuned

if __name__ == "__main__":
    name, port = sys.argv[1], int(sys.argv[2])
    use = BENCH[name]["use"]
    spec = MODELS[use]
    path = os.path.join(ROOT, spec["path"])
    wake_gpus()
    loading = load_async(path, max_seqs=3, umax=spec["chunk"], warmup=False, lib=os.path.join(ROOT, spec["lib"]))
    pool = ThreadPoolExecutor(1)
    engine = pool.submit(lambda: tuned(loading.result(), name))
    pool.shutdown(wait=False)
    serve(lambda: Server(path, os.path.join(ROOT, "data/models/tokenizer.json"), slots=3, engine=engine, warmup=False, model=use),
          "127.0.0.1", port, model=use)
