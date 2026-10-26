import glob
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys

from measure import BENCH, CASES, MODELS, ROOT, STRATA, WORK, kl_ref, kl_tokens, run_pass

RESULTS = os.path.join(ROOT, "bench/results.json")
PROTOCOL = ["bench/measure.py", "bench/serve.py", "bench/kl.py", "bench/vram.c", "bench/kl.txt"]
CODE = ["engine", "chrysalis", "Makefile"] + PROTOCOL
SKIP = ("chrysalis/voice/",)


def metric_dirs(name):
    d = {}
    for c in CASES:
        d[f"ttft.{c}"], d[f"prompt_tps.{c}"], d[f"decode_tps.{c}"], d[f"itl_p99.{c}"] = -1, 1, 1, -1
    d.update({"reused.followup": 1, "sanity": 1, "kl_mean": -1, "kl_p999": -1, "kl_max": -1, "top1": 1, "ram_gb": -1,
              "kv_tokens": 1, "vram_gb": 0, "load_s": -1, "pass_s": 0})
    return d


def git(*args, stdin=None):
    return subprocess.run(["git", "-C", ROOT, *args], input=stdin, capture_output=True, text=True, check=True).stdout


def code_pairs(name, tree):
    skip = SKIP + BENCH[name]["skip"]
    if tree == "worktree":
        paths = [p for p in git("ls-files", "-co", "--exclude-standard", "--", *CODE).split("\n") if p and os.path.isfile(os.path.join(ROOT, p))]
        shas = git("hash-object", "--stdin-paths", stdin="\n".join(paths) + "\n").split()
        pairs = list(zip(paths, shas))
    elif tree == "index":
        pairs = [(line.split("\t", 1)[1], line.split()[1]) for line in git("ls-files", "-s", "--", *CODE).split("\n") if line]
    else:
        pairs = [(line.split("\t", 1)[1], line.split()[2]) for line in git("ls-tree", "-r", tree, "--", *CODE).split("\n") if line]
    return sorted((p, s) for p, s in pairs if not p.startswith(skip))


def file_sha(path, cache):
    st = os.stat(path)
    ident = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]
    if cache.get(path, [None])[:4] == ident:
        return cache[path][4]
    h, fd = hashlib.sha256(), os.open(path, os.O_RDONLY)
    try:
        off = 0
        while chunk := os.read(fd, 64 << 20):
            h.update(chunk)
            os.posix_fadvise(fd, off, len(chunk), os.POSIX_FADV_DONTNEED)
            off += len(chunk)
    finally:
        os.close(fd)
    cache[path] = ident + [h.hexdigest()]
    return cache[path][4]


def fingerprint():
    read = lambda p: open(p).read().strip() if os.path.exists(p) else ""
    cpu = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in open("/proc/cpuinfo") if ":" in l) if k.strip() in ("model name", "microcode")}
    cards = [os.path.dirname(p) for p in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"))]
    site = glob.glob(os.path.join(ROOT, "data/venv/lib/python*/site-packages"))
    dists = sorted(os.path.basename(d) for d in glob.glob(os.path.join(site[0], "*.dist-info"))) if site else []
    fw = subprocess.run(["pacman", "-Q", "linux-firmware"], capture_output=True, text=True).stdout.strip() if shutil.which("pacman") else ""
    return {"kernel": os.uname().release, "rocm": read(os.path.join(ROOT, "data/rocm/.info/version")), "firmware": fw,
            "gpu": [[read(f"{c}/{k}") for k in ("vendor", "device", "revision", "vbios_version", "mem_info_vram_total")] for c in cards],
            "cpu": cpu, "ram": read("/proc/meminfo").split("\n")[0], "bios": [read(f"/sys/class/dmi/id/{k}") for k in ("board_name", "bios_version", "bios_date")],
            "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"), "python": dists}


def env_files():
    return [p for p in ("data/rocm/lib/libamdhip64.so", "data/rocm/lib/libhipblaslt.so") if os.path.exists(os.path.join(ROOT, p))] + \
           [os.path.relpath(p, ROOT) for p in sorted(glob.glob(os.path.join(STRATA, "strata_cli/bench_standard.py")))] + \
           [os.path.relpath(p, ROOT) for p in sorted(glob.glob(os.path.join(STRATA, "bench/prompts/standard-v1/MANIFEST.sha256")))]


def keys(tree, names=BENCH):
    os.makedirs(WORK, exist_ok=True)
    cache_path = os.path.join(WORK, "sha.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    env = fingerprint()
    env_id = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:16]
    shared = {p: file_sha(os.path.join(ROOT, p), cache) for p in env_files()}
    out = {}
    for name in names:
        files = {p: file_sha(os.path.join(ROOT, p), cache) for p in BENCH[name]["files"] + [BENCH[name]["ref"]] if os.path.exists(os.path.join(ROOT, p))}
        blob = {"code": code_pairs(name, tree), "files": files, "shared": shared, "env": env_id}
        out[name] = hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]
    json.dump(cache, open(cache_path, "w"))
    return out, env_id, env


def load_results(text=None):
    if text is None:
        text = open(RESULTS).read() if os.path.exists(RESULTS) else ""
    data = json.loads(text) if text.strip() else {}
    data.setdefault("envs", {})
    for name in BENCH:
        data.setdefault(name, {})
    return data


def save_results(data):
    with open(RESULTS + ".tmp", "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(RESULTS + ".tmp", RESULTS)


def betacf(a, b, x):
    tiny, c, d = 1e-300, 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10000):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)), -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1 + num * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return h


def betainc(a, b, x):
    if x <= 0 or x >= 1:
        return max(0.0, min(1.0, x))
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return front * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - front * betacf(b, a, 1 - x) / b


def t_quantile(p, dof):
    cdf = lambda t: 1 - 0.5 * betainc(dof / 2, 0.5, dof / (dof + t * t))
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def noise(data, name, env_id):
    groups = [[p["metrics"] for p in e["passes"]] for e in data[name].values() if e["env"] == env_id and len(e["passes"]) > 1]
    out = {}
    for m in metric_dirs(name):
        ss, dof, seen = 0.0, 0, False
        for g in groups:
            vals = [x[m] for x in g if x.get(m) is not None]
            if len(vals) > 1:
                seen = True
                mu = statistics.fmean(vals)
                ss += sum((v - mu) ** 2 for v in vals)
                dof += len(vals) - 1
        if seen:
            out[m] = (math.sqrt(ss / dof), dof)
    return out


def compare(data, name, parent, child, env_id):
    a, b = data[name][parent]["passes"], data[name][child]["passes"]
    sig, dirs, steps, unknown = noise(data, name, env_id), metric_dirs(name), [], []
    for m, d in dirs.items():
        if d == 0:
            continue
        va, vb = [p["metrics"][m] for p in a if p["metrics"].get(m) is not None], [p["metrics"][m] for p in b if p["metrics"].get(m) is not None]
        if not va or not vb:
            continue
        diff = d * (statistics.fmean(vb) - statistics.fmean(va))
        if m not in sig:
            unknown.append(m)
            continue
        s, dof = sig[m]
        if s == 0:
            if diff < 0:
                steps.append((m, statistics.fmean(va), statistics.fmean(vb), 0.0))
        else:
            thr = t_quantile(0.99, dof) * s * math.sqrt(1 / len(va) + 1 / len(vb))
            if diff < -thr:
                steps.append((m, statistics.fmean(va), statistics.fmean(vb), thr))
    return steps, unknown


def machine_ready(names):
    for p in glob.glob("/proc/[0-9]*/fdinfo/*"):
        try:
            vram = sum(int(l.split()[1]) for l in open(p) if l.startswith("drm-memory-vram:"))
        except (OSError, ValueError, IndexError):
            continue
        if vram > 4 << 20:
            sys.exit(f"bench: process {p.split('/')[2]} holds {vram >> 20} GiB of VRAM; stop its model (the chat's Stop), the bench needs the GPU to itself")
    from chrysalis import link
    from chrysalis.engine import wake_gpus
    wake_gpus()
    st = link.status(link.device())
    if st and st["slow"]:
        sys.exit(f"bench: the GPU's PCIe link is at {st['gts']:g} GT/s x{st['width']}, slower than this machine's best "
                 f"{st['best_gts']:g} GT/s x{st['best_width']}")
    avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) << 10
    if "fn" in names and avail < 48 << 30:
        sys.exit(f"bench: {avail / 2**30:.1f} GiB of RAM is available; Flash-Next needs 48")


def fmt(m, v):
    if v is None:
        return "–"
    if m.startswith(("ttft", "load_s", "pass_s", "itl")):
        return f"{v * 1000:.0f} ms" if m.startswith("itl") else f"{v:.2f} s"
    if m.startswith(("prompt_tps", "decode_tps")):
        return f"{v:,.1f}"
    if m.startswith("kl"):
        return f"{v:.5f}"
    if m == "top1":
        return f"{100 * v:.2f}%"
    if m.endswith("_gb"):
        return f"{v:.2f} GB"
    return f"{v:g}"


def table(data, name, key, env_id):
    e = data[name].get(key)
    if not e:
        print(f"{name}: not measured for this code and machine")
        return
    sig = noise(data, name, env_id)
    mean = lambda m: statistics.fmean(v) if (v := [p["metrics"][m] for p in e["passes"] if p["metrics"].get(m) is not None]) else None
    res = lambda m: (t_quantile(0.99, sig[m][1]) * sig[m][0] * math.sqrt(2) / abs(mean(m)) * 100) if m in sig and sig[m][0] and mean(m) else None
    print(f"\n{BENCH[name]['use']} ({len(e['passes'])} pass{'es' if len(e['passes']) > 1 else ''}, key {key})")
    print("| | " + " | ".join(CASES) + " |\n|---|" + "---:|" * len(CASES))
    for row, label in (("ttft", "Time to first token"), ("prompt_tps", "Prompt tok/s"), ("decode_tps", "Decode tok/s"), ("itl_p99", "Token gap p99")):
        print(f"| {label} | " + " | ".join(fmt(f"{row}.{c}", mean(f"{row}.{c}")) for c in CASES) + " |")
    print("\n| Metric | Value | Resolution |\n|---|---:|---:|")
    for m in ("reused.followup", "sanity", "kl_mean", "kl_p999", "kl_max", "top1", "kv_tokens", "ram_gb", "vram_gb", "load_s"):
        r = res(m)
        print(f"| {m} | {fmt(m, mean(m))} | {'exact' if m in sig and sig[m][0] == 0 else f'{r:.2f}%' if r is not None else '–'} |")
    timed = [res(f"{row}.{c}") for row in ("ttft", "prompt_tps", "decode_tps") for c in CASES]
    timed = [r for r in timed if r is not None]
    exact = [m for m in sig if sig[m][0] == 0]
    outs = {p["outputs"] for p in e["passes"]}
    print("\n| Meta | Value |\n|---|---:|")
    for label, with_kl in (("Time for a first pass (with KL)", True), ("Time for a repeat pass", False)):
        t = [p["metrics"]["pass_s"] for p in e["passes"] if ("kl_mean" in p["metrics"]) == with_kl]
        print(f"| {label} | {fmt('pass_s', statistics.fmean(t)) if t else '–'} |")
    print(f"| Resolution of timed rows (median, worst) | {statistics.median(timed):.2f}%, {max(timed):.2f}% |" if timed else "| Resolution of timed rows | needs the spread |")
    print(f"| Rows that repeat to the bit | {len(exact)} of {len(sig)} |")
    print(f"| Outputs identical across passes | {'yes' if len(outs) == 1 else 'no'} ({len(e['passes'])} passes) |")
    print(f"| Confirmation passes | {sum(1 for x in data[name].values() for p in x['passes'] if p.get('confirm'))} |")


def make(names):
    targets = ["build/vram_peak.so"] + [MODELS[BENCH[n]["use"]]["lib"] for n in names] + [BENCH[n]["make_ref"][0] for n in names]
    subprocess.run(["make", "-s", "-C", ROOT, *targets], check=True)


def cmd_run(argv):
    names = [n for n in BENCH if n in argv] or list(BENCH)
    passes = int(argv[argv.index("--passes") + 1]) if "--passes" in argv else 0
    machine_ready(names)
    make(names)
    tokens = kl_tokens()
    for name in names:
        kl_ref(name, tokens)
    data = load_results()
    have, env_id, env = keys("worktree", names)
    parent, _, _ = keys("HEAD", names)
    data["envs"][env_id] = env
    head = git("rev-parse", "HEAD").strip()
    dirty = bool(git("status", "--porcelain", "--", *CODE).strip())
    verdict = 0
    for name in names:
        key = have[name]
        entry = data[name].setdefault(key, {"env": env_id, "head": head, "dirty": dirty, "passes": []})
        n = passes or (0 if entry["passes"] else 1)
        for i in range(n):
            print(f"bench: {name} pass {len(entry['passes']) + 1}{' (spread)' if passes else ''}", flush=True)
            p = run_pass(name, with_kl=len(entry["passes"]) < 2)
            p["spread"] = bool(passes)
            entry["passes"].append(p)
            save_results(data)
        if not passes and key != parent[name] and parent[name] in data[name]:
            steps, unknown = compare(data, name, parent[name], key, env_id)
            if steps and len(entry["passes"]) == 1 and all(s[3] > 0 for s in steps):
                print(f"bench: {name} looks worse on {', '.join(s[0] for s in steps)}; confirming with a second pass", flush=True)
                p = run_pass(name, with_kl=False)
                p["confirm"] = True
                entry["passes"].append(p)
                save_results(data)
                steps, unknown = compare(data, name, parent[name], key, env_id)
            verdict |= report(name, steps, unknown)
        table(data, name, key, env_id)
    sys.exit(verdict)


def report(name, steps, unknown):
    for m, a, b, thr in steps:
        print(f"bench: {name} {m} stepped back: {a:.6g} -> {b:.6g}" + (f" (beyond its noise, {thr:.3g})" if thr else " (repeats to the bit)"))
    if unknown:
        print(f"bench: {name} has no spread for {', '.join(unknown)}; run `bench.py run --passes 5` on a clean tree")
    return 1 if steps or unknown else 0


def check(child_tree, parent_tree, text, label, parent_measured):
    if text is None:
        return 0
    data = load_results(text)
    ck, env_id, _ = keys(child_tree)
    pk, _, _ = keys(parent_tree)
    bad = 0
    for name in BENCH:
        if ck[name] == pk[name]:
            continue
        if ck[name] not in data[name]:
            print(f"bench: {label}: {name} changed and has no numbers for this code and machine; run `make bench`")
            bad = 1
        elif pk[name] not in data[name] and not parent_measured:
            continue
        elif pk[name] not in data[name]:
            print(f"bench: {label}: {name}'s parent has no numbers for this machine; run `make bench` on it first")
            bad = 1
        else:
            bad |= report(name, *compare(data, name, pk[name], ck[name], env_id))
    return bad


def show_file(rev):
    r = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:bench/results.json"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def instant_share():
    revs = git("rev-list", "--reverse", "HEAD").split()
    first = next((i for i, r in enumerate(revs) if show_file(r) is not None), None)
    if first is None:
        return
    same = total = 0
    for r in revs[first + 1:]:
        parents = git("rev-list", "--parents", "-n", "1", r).split()[1:]
        if parents:
            total += 1
            same += keys(r)[0] == keys(parents[0])[0]
    print(f"\n| Meta | Value |\n|---|---:|\n| Commits settled by the hash alone | {same} of {total} |")


def cmd_check(argv):
    if "--staged" in argv:
        sys.exit(check("index", "HEAD", show_file(""), "staged", show_file("HEAD") is not None))
    bad = 0
    for line in sys.stdin:
        local, remote = line.split()[1], line.split()[3]
        if set(local) == {"0"}:
            continue
        span = f"{remote}..{local}" if set(remote) != {"0"} else local
        for c in git("rev-list", "--reverse", span, "--not", "--remotes").split():
            parents = git("rev-list", "--parents", "-n", "1", c).split()[1:]
            if parents:
                bad |= check(c, parents[0], show_file(c), c[:9], show_file(parents[0]) is not None)
    sys.exit(bad)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "run":
        cmd_run(sys.argv[2:])
    elif cmd == "check":
        cmd_check(sys.argv[2:])
    else:
        data = load_results()
        have, env_id, _ = keys("worktree")
        for name in BENCH:
            table(data, name, have[name], env_id)
        instant_share()
