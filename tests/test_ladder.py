import hashlib, json, os, random, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))
PY = os.path.join(ROOT, "data/venv/bin/python")


def child(use):
    import numpy as np
    from chrysalis.engine import Engine, Sampling
    from chrysalis.models import MODELS
    spec = MODELS[use]
    eng = Engine(os.path.join(ROOT, spec["path"]), max_seqs=3, umax=spec["chunk"], lib=os.path.join(ROOT, spec["lib"]))
    random.seed(11)
    prompt = [random.randrange(1000, 50000) for _ in range(700)]
    greedy = Sampling(temperature=0.0, top_p=1.0, min_p=0.0, presence_penalty=0.0, top_k=1, n_draft=spec["draft"])
    s = eng.new_seq()
    eng.prefill(s, prompt)
    logits = np.array(eng.last_logits(), np.float32)
    out = []
    while len(out) < 96:
        out += eng.generate(s, greedy, 16)
    print(json.dumps({"tokens": out[:96], "logits": hashlib.sha256(logits.tobytes()).hexdigest(), "top": int(logits.argmax()),
                      "ram_gb": next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmRSS")) / 2 ** 20, "context": eng.capacity}))


def rung(use, hold):
    ballast = None
    if hold:
        ballast = subprocess.Popen([PY, os.path.join(ROOT, "bench/ballast.py"), str(hold)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        if not json.loads(ballast.stdout.readline() or "{}").get("ready"):
            sys.exit(f"the ballast could not hold {hold} GB")
    try:
        run = subprocess.run([PY, os.path.abspath(__file__), "child", use], capture_output=True, text=True)
    finally:
        if ballast:
            ballast.stdin.close()
            ballast.wait()
    if run.returncode:
        return {"error": (run.stderr or run.stdout).strip().splitlines()[-1:]}
    return json.loads(run.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(sys.argv[2])
        sys.exit(0)
    use, holds = sys.argv[1], [float(x) for x in sys.argv[2:]]
    full, fails = rung(use, 0), 0
    print(f"all of the card: context {full.get('context')}, {full.get('ram_gb', 0):.1f} GB of RAM")
    fails += "error" in full
    for hold in holds:
        r = rung(use, hold)
        if "error" in r:
            print(f"{hold:g} GB held by another program: it starts and answers   FAIL  {r['error']}")
            fails += 1
            continue
        took = r["ram_gb"] > full["ram_gb"] + 1 or r["context"] < full["context"]
        same = r["tokens"] == full["tokens"]
        print(f"{hold:g} GB held by another program: it starts and answers   ok  context {r['context']}")
        print(f"{hold:g} GB held: the smaller card changed what the engine keeps  {'ok' if took else 'FAIL'}  {r['ram_gb']:.1f} GB of RAM")
        print(f"{hold:g} GB held: the same 96 tokens as with all of the card      {'ok' if same else 'FAIL'}  logits {'identical' if r['logits'] == full['logits'] else 'differ'}")
        fails += (not took) + (not same)
    print("all ok" if not fails else f"{fails} FAILED")
    sys.exit(bool(fails))
