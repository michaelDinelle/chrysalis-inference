import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
HOME = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "hill-rail", os.path.basename(ROOT))
MACHINES = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "hill", os.path.basename(ROOT) + ".json")
LEDGER = ".hill/ledger.jsonl"
VERDICT = re.compile(r"^(?:👑 )?(PROMOTE|REJECT|TIE|REFUSE|ERROR)\b", re.M)


def git(*args, cwd=ROOT, data=None, env=None, check=True):
    r = subprocess.run(["git", *args], cwd=cwd, input=data, capture_output=True, env={**os.environ, **(env or {})})
    if check and r.returncode:
        sys.exit(f"git {' '.join(args[:3])}: {r.stderr.decode()[-400:]}")
    return r.stdout.decode().strip()


def lines(path):
    return open(path).read().splitlines() if os.path.exists(path) else []


def snapshot(message):
    index = os.path.join(HOME, "index")
    env = {"GIT_INDEX_FILE": index}
    git("read-tree", "HEAD", env=env)
    git("add", "-A", "--", ".", ":!" + LEDGER, env=env)
    tree = git("write-tree", env=env)
    if tree == git("rev-parse", "HEAD^{tree}"):
        sys.exit("hill step: the working tree is HEAD: nothing to judge")
    return tree, git("commit-tree", tree, "-p", "HEAD", "-m", message)


def local(patch, ledger, out):
    wt = os.path.join(HOME, "wt")
    if not os.path.exists(os.path.join(wt, ".git")):
        git("worktree", "prune")
        git("worktree", "add", "-q", "--detach", wt, "HEAD")
        os.symlink(os.path.join(ROOT, "data"), os.path.join(wt, "data"))
        os.makedirs(os.path.join(wt, "build"), exist_ok=True)
    for name in os.listdir(os.path.join(ROOT, "build")) if os.path.isdir(os.path.join(ROOT, "build")) else []:
        src, dst = os.path.join(ROOT, "build", name), os.path.join(wt, "build", name)
        keep = name.startswith(("ref_logits", "cmp", "llama_params")) or name in ("bench", "gpu")
        if keep and not os.path.exists(dst):
            subprocess.run(["cp", "-a", "--reflink=auto", src, dst], check=False)
    git("checkout", "-q", "-f", "--detach", git("rev-parse", "HEAD"), cwd=wt)
    git("clean", "-qfd", "-e", "data", "-e", "build", cwd=wt)
    git("apply", "--binary", "--index", "-", cwd=wt, data=patch)
    open(os.path.join(wt, LEDGER), "w").write("".join(x + "\n" for x in ledger))
    gpu = os.path.join(ROOT, "build", "gpu")
    env = dict(os.environ)
    if os.path.exists(gpu):
        for kv in subprocess.run([gpu], capture_output=True, text=True).stdout.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k] = v
    wrap = ["systemd-run", "--user", "--scope", "-q"] if shutil.which("systemd-run") else []
    wrap += sum((["-E", f"{k}={env[k]}"] for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES") if k in env and wrap), [])
    r = subprocess.run([*wrap, "sh", ".hill/hill", "gate"], cwd=wt, capture_output=True, text=True, env=env)
    out["local"] = (r.stdout + r.stderr, [x for x in lines(os.path.join(wt, LEDGER)) if x not in set(ledger)])


def remote(m, head, patch, ledger, out):
    name = m["name"]
    push = subprocess.run(["git", "push", "-q", m["git"], f"+{head}:refs/heads/incoming"], cwd=ROOT, capture_output=True, text=True)
    if push.returncode:
        out[name] = ("ERROR: push failed: " + push.stderr[-300:], [])
        return
    mark = "====HILL-RAIL===="
    script = f"""{m.get('pre', ':')}
cd {m['dir']} || exit 9
git checkout -q -f --detach incoming && git clean -qfd -e data -e build >/dev/null
base64 -d > .hill/rail.patch <<'RAIL_EOF'
{__import__('base64').b64encode(patch).decode()}
RAIL_EOF
base64 -d > .hill/ledger.jsonl <<'RAIL_EOF'
{__import__('base64').b64encode(''.join(x + chr(10) for x in ledger).encode()).decode()}
RAIL_EOF
cp .hill/ledger.jsonl .hill/rail.before
git apply --binary --index .hill/rail.patch || exit 8
rm -f .hill/rail.patch
sh .hill/hill gate 2>&1
echo {mark}
grep -vxFf .hill/rail.before .hill/ledger.jsonl
rm -f .hill/rail.before
"""
    r = subprocess.run([*m["ssh"], m.get("shell", "bash -s")], input=script, capture_output=True, text=True)
    text, _, new = r.stdout.partition(mark)
    out[name] = (text + r.stderr[-300:], [x for x in new.splitlines() if x.startswith("{")])


def step(message):
    os.makedirs(HOME, exist_ok=True)
    lock = open(os.path.join(HOME, "lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    t0 = time.time()
    head = git("rev-parse", "HEAD")
    tree, cand = snapshot(message)
    patch = subprocess.run(["git", "diff", "--binary", head, cand], cwd=ROOT, capture_output=True, check=True).stdout
    ledger = lines(os.path.join(ROOT, LEDGER))
    machines = json.load(open(MACHINES)) if os.path.exists(MACHINES) else []
    out = {}
    threads = [threading.Thread(target=local, args=(patch, ledger, out))]
    threads += [threading.Thread(target=remote, args=(m, head, patch, ledger, out)) for m in machines]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    verdicts, new = {}, []
    for name, (text, added) in out.items():
        found = VERDICT.findall(text)
        verdicts[name] = found[-1] if found else "ERROR"
        new += [x for x in added if x not in new]
        print(f"== {name}: {verdicts[name]}")
        shown = [x for x in text.splitlines() if "| win |" in x or "| loss |" in x or VERDICT.match(x) or x.startswith("  ")]
        print("\n".join(shown[-24:]) if shown else text[-600:])
    values = set(verdicts.values())
    lands = "PROMOTE" in values and values <= {"PROMOTE", "TIE"}
    summary = ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items()))
    with open(os.path.join(ROOT, LEDGER), "a") as f:
        f.write("".join(x + "\n" for x in new))
    if lands:
        wt = os.path.join(HOME, "wt")
        shutil.copyfile(os.path.join(ROOT, LEDGER), os.path.join(wt, LEDGER))
        git("add", "-A", "--", ".", ":!data", ":!build", cwd=wt)
        c = subprocess.run(["git", "commit", "-q", "-m", f"{message}\n\nHill: {summary}"], cwd=wt, capture_output=True, text=True)
        if c.returncode:
            print("hill step: the commit was refused:\n" + (c.stdout + c.stderr)[-600:])
            lands = False
        else:
            new_head = git("rev-parse", "HEAD", cwd=wt)
            branch = git("symbolic-ref", "-q", "HEAD")
            git("update-ref", branch, new_head, head)
            git("reset", "-q")
            print(f"LANDED {new_head[:7]} ({summary}) in {time.time() - t0:.0f} s")
    if not lands:
        slug = re.sub(r"[^a-z0-9]+", "-", message.lower()).strip("-")[:48] or cand[:7]
        shelved = git("commit-tree", tree, "-p", head, "-m", f"Candidate, not landed: {message}\n\nHill: {summary}")
        git("update-ref", f"refs/candidates/{slug}", shelved)
        env = {"GIT_INDEX_FILE": os.path.join(HOME, "index")}
        git("read-tree", "HEAD", env=env)
        git("add", "-A", "--", ".", ":!" + LEDGER, env=env)
        if git("write-tree", env=env) == tree:
            git("checkout", "-q", "-f", "HEAD", "--", ".", ":!" + LEDGER)
            git("clean", "-qfd", "-e", "data", "-e", "build")
            where = "the tree is back at HEAD"
        else:
            where = "the tree has newer edits and was left alone"
        print(f"SHELVED refs/candidates/{slug} ({summary}) in {time.time() - t0:.0f} s; {where}")
    return 0 if lands else 1


def nxt():
    print(time.strftime("clock %H:%M:%S"))
    spec = json.load(open(os.path.join(ROOT, "hill.json")))
    last, seen = {}, {}
    for x in lines(os.path.join(ROOT, LEDGER)):
        e = json.loads(x)
        if e.get("type") == "record" or e.get("verdict") == "PROMOTE":
            last.setdefault(e.get("machine"), {}).update({k: v for k, v in (e.get("values") or {}).items() if isinstance(v, (int, float))})
            seen[e.get("machine")] = (e.get("at", ""), e.get("label"))
    newest = max((at for at, _ in seen.values()), default="")[:10]
    for ident, values in last.items():
        if seen[ident][0][:10] != newest:
            continue
        label = seen[ident][1]
        behind = sorted((v, k) for k, v in values.items() if any(r in k for r in ("strata", "llamacpp", "mlx")) and 0 < v < 1)
        print(f"{label}: " + ("; ".join(f"{spec['rewards'].get(k, {}).get('label', k)} {spec['rewards'].get(k, {}).get('case', '')} {v:.2f}x".replace("  ", " ")
                                           for v, k in behind[:6]) or "ahead of every rival measured here"))
    shelved = git("for-each-ref", "refs/candidates", "--format=%(refname:short)\t%(contents:subject)")
    if shelved:
        print("shelved:\n" + "\n".join("  " + x[:150] for x in shelved.splitlines()))


if __name__ == "__main__":
    if sys.argv[1] == "next":
        nxt()
    elif sys.argv[1] == "step":
        if len(sys.argv) < 3 or not sys.argv[2].strip():
            sys.exit('usage: hill step "what changed, as the commit message"')
        sys.exit(step(sys.argv[2]))
