import ctypes as C
import glob
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chrysalis.models import BACKEND, MODELS

WORK = os.path.join(ROOT, "build/bench")
PY = os.path.join(ROOT, "data/venv/bin/python")
STRATA = os.path.join(ROOT, "data/strata")
CASES = ["short", "4k", "16k", "100k", "followup"]
STRATA_CASES = {"short": "performance-short-r1", "4k": "performance-prompt-4k-r1", "16k": "performance-prompt-16k-r1"}
RECORDS = lambda a, b: "\n".join(f"record{i:06d}: alpha beta gamma delta epsilon zeta eta theta." for i in range(a, b))
LONG = "[trial 1]\n" + RECORDS(1, 5201) + "\n\nSummarize the structure of these records and explain how to look up a record by its ID."
MORE = RECORDS(5201, 5261) + "\n\nThese records were added since. Which ID is the last one now?"
MAC = sys.platform == "darwin"
BACKENDS = ("hip", "metal")
FN_FILES = ["data/models/qwen3.8-flash-next/qwen3.8-flash-next.gguf", "data/models/qwen3.8-flash-next/src/Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf",
            "data/models/tokenizer.json"]
ON = {
    "hip": {"fn": dict(env={"CHRYSALIS_FN_SPILL_GB": "1"}, tune=[("spill_spare", 0)],
                       files=FN_FILES + ["data/llama.cpp/build/bin/libggml-cpu.so", "data/llama.cpp/build/bin/libggml-base.so",
                                         "data/cache/hipblaslt.txt", "data/cache/flash-next-calibration.txt"]),
            "27b": dict(env={}, tune=[], files=["data/models/qwen3.8-27b/qwen3.8-27b.gguf", "data/models/tokenizer.json", "data/cache/hipblaslt.txt"])},
    "metal": {"fn": dict(env={"CHRYSALIS_FN_CACHE_GB": "16", "CHRYSALIS_FN_CTX": "131072"}, tune=[], files=FN_FILES),
              "27b": dict(env={}, tune=[], files=["data/models/qwen3.8-27b/qwen3.8-27b.gguf", "data/models/tokenizer.json"])},
}
DIRS = {"fn": "flash-next", "27b": "27b"}


def bench_for(backend):
    out = {
        "fn": dict(use="qwen3.8-flash-next", ref="build/cmp_strata/ref.bin",
                   make_ref=["build/ref_logits_fn", "data/models/qwen3.8-flash-next/src/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"]),
        "27b": dict(use="qwen3.8-27b", ref="build/bench/ref-27b.bin", make_ref=["build/ref_logits", "data/models/qwen3.8-27b/qwen3.8-27b.gguf"]),
    }
    for name, b in out.items():
        b.update(ON[backend][name])
        b["skip"] = tuple(f"engine/{o}/" for o in BACKENDS if o != backend) + tuple(f"engine/{backend}/{d}/" for n, d in DIRS.items() if n != name)
    return out


BENCH = bench_for(BACKEND)


def own_calibration():
    import chrysalis.engine as ce
    os.makedirs(WORK, exist_ok=True)
    for shared in (ce.CALIBRATION, ce.LT_CACHE):
        mine = os.path.join(WORK, os.path.basename(shared))
        if not os.path.exists(mine) and os.path.exists(shared):
            shutil.copy(shared, mine)
    ce.CALIBRATION = os.path.join(WORK, os.path.basename(ce.CALIBRATION))
    os.environ["CHRYSALIS_LT_CACHE"] = os.path.join(WORK, os.path.basename(ce.LT_CACHE))


def tuned(engine, name):
    if BENCH[name]["tune"]:
        tune = engine.lib.chrysalis_tune
        tune.restype, tune.argtypes = C.c_double, [C.c_char_p, C.c_double]
        for key, value in BENCH[name]["tune"]:
            tune(key.encode(), value)
    return engine


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def drop_cache(path):
    if MAC:
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def peak_footprint(pid):
    info = (C.c_uint8 * 512)()
    if C.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(pid, 4, info):
        return None
    return int.from_bytes(bytes(info[240:248]), "little")


def tree_pids(pid):
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        for t in glob.glob(f"/proc/{p}/task/*/children"):
            try:
                todo += [int(c) for c in open(t).read().split()]
            except OSError:
                pass
    return out


def server_pid(pid):
    if MAC:
        return pid
    for p in tree_pids(pid):
        try:
            if b"bench/serve.py" in open(f"/proc/{p}/cmdline", "rb").read():
                return p
        except OSError:
            pass
    return None


def http(url, body=None, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def strata():
    if not os.path.isdir(os.path.join(STRATA, "strata_cli")):
        sys.exit(f"bench: Strata's client is missing at {STRATA} (link it to Strata's terminal package, which has bench/prompts/standard-v1)")
    sys.path.insert(0, STRATA)
    from strata_cli import bench_standard
    return bench_standard


def kl_ref(name, tokens):
    b = BENCH[name]
    ref = os.path.join(ROOT, b["ref"])
    if not os.path.exists(ref):
        os.makedirs(os.path.dirname(ref), exist_ok=True)
        print(f"bench: {name} reference logits for the KL text (llama.cpp)...", flush=True)
        subprocess.run([os.path.join(ROOT, b["make_ref"][0]), os.path.join(ROOT, b["make_ref"][1]), tokens, ref + ".tmp"], check=True)
        os.replace(ref + ".tmp", ref)
    return ref


def kl_tokens():
    from tokenizers import Tokenizer
    import numpy as np
    path = os.path.join(WORK, "kl-tokens.bin")
    tok = Tokenizer.from_file(os.path.join(ROOT, "data/models/tokenizer.json"))
    ids = np.array(tok.encode(open(os.path.join(ROOT, "bench/kl.txt")).read(), add_special_tokens=False).ids, np.int32)
    if not os.path.exists(path) or not np.array_equal(np.fromfile(path, np.int32), ids):
        ids.tofile(path)
    return path


def p99(gaps):
    return gaps[min(len(gaps) - 1, math.ceil(0.99 * len(gaps)) - 1)] if gaps else None


def run_pass(name, with_kl):
    B, b = strata(), BENCH[name]
    use, start = b["use"], time.monotonic()
    for f in b["files"]:
        drop_cache(os.path.join(ROOT, f))
    port, snap = free_port(), tempfile.mkdtemp(prefix=f"snap-{name}-", dir=WORK)
    vram_out, log_path = os.path.join(WORK, f"vram-{name}.txt"), os.path.join(WORK, f"server-{name}.log")
    if os.path.exists(vram_out):
        os.remove(vram_out)
    env = [f"CHRYSALIS_SNAPSHOTS={snap}", *(f"{k}={v}" for k, v in b["env"].items())]
    if MAC:
        cmd = ["env", *env, f"CHRYSALIS_FN_STATS={os.path.join(snap, 'experts.bin')}", PY, os.path.join(ROOT, "bench/serve.py"), name, str(port)]
    else:
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemorySwapMax=0", "env", f"LD_PRELOAD={os.path.join(ROOT, 'build/vram_peak.so')}",
               f"BENCH_VRAM_OUT={vram_out}", *env, PY, os.path.join(ROOT, "bench/serve.py"), name, str(port)]
    url = f"http://127.0.0.1:{port}"
    t0 = time.monotonic()
    srv = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    m, digest, answers = {}, hashlib.sha256(), []
    try:
        while True:
            if srv.poll() is not None:
                sys.exit(f"bench: the {name} server exited while loading (see {log_path})")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(0.05)
        http(url + "/v1/chat/completions", {"model": use, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1, "stream": False,
                                             "chat_template_kwargs": {"enable_thinking": False}})
        m["load_s"] = time.monotonic() - t0
        m["kv_tokens"] = http(url + "/chrysalis/stats")["kv_tokens"]
        target, specs = B.Target(url, None), {s["id"]: s for s in B.load_corpus().specs}

        def ask(spec):
            raw = B.stream_chat(target, spec, use, timeout_s=B.REQUEST_TIMEOUT_S)
            if raw["error"]:
                sys.exit(f"bench: {name} {spec['id']}: {raw['error']}")
            stat = http(url + "/chrysalis/stats")["recent"][-1]
            n = (raw["usage"] or {}).get("completion_tokens")
            if stat.get("completion_tokens") != n:
                sys.exit(f"bench: {name} {spec['id']}: the server's last reply is not this request's")
            content = "".join(raw["content"])
            said = json.dumps([spec["id"], "".join(raw["reasoning"]), content, [[c["name"], c["arguments"]] for _, c in sorted(raw["tool_calls"].items())]]).encode()
            digest.update(said)
            gaps = sorted(b2[0] - a2[0] for a2, b2 in zip(raw["events"], raw["events"][1:]))
            answers.append({"id": spec["id"], "answer": hashlib.sha256(said).hexdigest()[:16], "prompt": stat["prompt_tokens"], "cached": stat["cached_tokens"],
                            "written": stat["completion_tokens"], "decode_tps": stat["decode_tps"], "per_pass": stat["tokens_per_step"],
                            "gap_p99": p99(gaps)})
            return raw, stat, content, gaps

        def record(case, raw, stat, gaps):
            m[f"ttft.{case}"] = raw["events"][0][0]
            m[f"prompt_tps.{case}"] = (stat["prompt_tokens"] - stat["cached_tokens"]) / stat["prefill_s"]
            m[f"decode_tps.{case}"] = stat["decode_tps"]
            m[f"per_pass.{case}"] = stat["tokens_per_step"]
            m[f"pass_rate.{case}"] = stat["decode_tps"] / stat["tokens_per_step"]
            m[f"itl_p99.{case}"] = p99(gaps)

        ask(next(s for s in specs.values() if s["phase"] == "warmup"))
        for case, sid in STRATA_CASES.items():
            raw, stat, _, gaps = ask(specs[sid])
            record(case, raw, stat, gaps)
        passed = 0
        for spec in (s for s in specs.values() if s["phase"] == "sanity"):
            raw, stat, content, _ = ask(spec)
            status = "truncated" if raw["finish_reason"] == "length" else "completed"
            passed += bool(B.evaluate(spec, status, content, [c for _, c in sorted(raw["tool_calls"].items())])["passed"])
        m["sanity"] = passed
        long = {"id": "long-100k", "messages": [{"role": "user", "content": LONG}], "max_output_tokens": 512, "seed": 1}
        raw, stat, said, gaps = ask(long)
        record("100k", raw, stat, gaps)
        more = {"id": "follow-up", "messages": long["messages"] + [{"role": "assistant", "content": said}, {"role": "user", "content": MORE}],
                "max_output_tokens": 512, "seed": 1}
        raw, stat, _, gaps = ask(more)
        record("followup", raw, stat, gaps)
        m["reused.followup"] = stat["cached_tokens"]
        experts = http(url + "/chrysalis/stats").get("experts")
        if experts and "memory_limited" in experts:
            m["mem_limited"] = experts["memory_limited"]
        pid = server_pid(srv.pid)
        if MAC:
            m["ram_gb"] = peak_footprint(pid) / 1e9
        else:
            m["ram_gb"] = next(int(l.split()[1]) for l in open(f"/proc/{pid}/status") if l.startswith("VmHWM:")) * 1024 / 1e9
    finally:
        try:
            os.killpg(srv.pid, signal.SIGINT)
            srv.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(srv.pid, signal.SIGKILL)
            srv.wait()
        except ProcessLookupError:
            pass
        shutil.rmtree(snap, ignore_errors=True)
    if not MAC:
        m["vram_gb"] = int(open(vram_out).read()) / 1e9
    if with_kl:
        tokens = kl_tokens()
        out = subprocess.run([PY, os.path.join(ROOT, "bench/kl.py"), name, tokens, kl_ref(name, tokens)], capture_output=True, text=True,
                             env={**os.environ, **b["env"]})
        if out.returncode:
            sys.exit(f"bench: {name} KL failed:\n{out.stderr[-2000:]}")
        m.update(json.loads(out.stdout.strip().split("\n")[-1]))
    m["pass_s"] = time.monotonic() - start
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "metrics": m, "outputs": digest.hexdigest()[:16], "answers": answers, "load": os.getloadavg()[0]}
