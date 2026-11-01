import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data/llama.cpp/gguf-py"))

MODEL = os.path.join(ROOT, "data/models/qwen3.8-flash-next/qwen3.8-flash-next.gguf")
STEPS = 400

if len(sys.argv) == 4:
    import chrysalis.engine as ce
    from tokenizers import Tokenizer
    from chrysalis.models import MODELS
    cal, steps, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    ce.CALIBRATION = cal
    spec = MODELS["qwen3.8-flash-next"]
    eng = ce.Engine(os.path.join(ROOT, spec["path"]), max_seqs=1, umax=spec["chunk"], lib=os.path.join(ROOT, spec["lib"]))
    tok = Tokenizer.from_file(os.path.join(ROOT, "data/models/tokenizer.json"))
    ids = tok.encode(open(os.path.join(ROOT, "bench/kl.txt")).read(), add_special_tokens=False).ids[:steps]
    at_load = eng.expert_stats()
    s = eng.new_seq()
    sums = []
    for t in ids:
        eng.prefill(s, [t])
        sums.append(hashlib.sha1(eng.last_logits().tobytes()).hexdigest())
    json.dump({"sums": sums, "at_load": at_load, "stats": eng.expert_stats()}, open(out, "w"))
    sys.exit(0)

from gguf import GGUFReader

TENSORS = GGUFReader(MODEL).tensors
HOT = [(int(t.n_bytes), int(t.shape[-1])) for t in TENSORS if t.name.endswith("_hot.weight")]
HOT_EXPERT = sum(nb for nb, _ in HOT) / sum(ne for _, ne in HOT) * 3
EMBD = next(int(t.n_bytes) for t in TENSORS if t.name == "token_embd.weight")

tmp = tempfile.mkdtemp(prefix="mirror-budget-")
cal = os.path.join(tmp, "flash-next-calibration.txt")
shared = os.path.join(ROOT, "data/cache/flash-next-calibration.txt")
with open(cal, "w") as f:
    for line in open(shared) if os.path.exists(shared) else []:
        if "\t" in line:
            key, values = line.rstrip("\n").split("\t")
            f.write(f"{key}\t{' '.join(values.split()[:7])}\n")
lines = len(open(cal).read().splitlines())


def child(steps, name):
    out = os.path.join(tmp, name + ".json")
    subprocess.run([sys.executable, os.path.abspath(__file__), cal, str(steps), out], check=True)
    return json.load(open(out))


fails = 0
first = child(48, "first")
now = open(cal).read().splitlines()
key, values = now[-1].split("\t")
kept = len(now) > lines and len(values.split()) == 9
print(f"a first process keeps the RAM it measured for the engine on this machine's calibration line  {'ok' if kept else 'FAIL'}", flush=True)
fails += not kept
footprint = int(key.split(" | ")[-1]) + first["at_load"]["mirrored"] * HOT_EXPERT + EMBD
with open(cal, "a") as f:
    f.write(f"{key}\t{' '.join(values.split()[:7])} {footprint + 192 * HOT_EXPERT:.17g} 0\n")
a = child(STEPS, "a")
held = bytearray(1 << 30)
for i in range(0, len(held), 4096):
    held[i] = 1
b = child(STEPS, "b")
del held
same = a["sums"] == b["sums"] and len(a["sums"]) == STEPS
bound = a["stats"]["memory_limited"] > 0 and b["stats"]["memory_limited"] > 0
diverge = next((i for i, (x, y) in enumerate(zip(a["sums"], b["sums"])) if x != y), None)
ok = same and bound and a["stats"]["mirrored"] == b["stats"]["mirrored"]
print(f"fresh processes, the second with 1 GB less free RAM: {STEPS} tokens' logits {'the same to the bit' if same else f'differ from token {diverge}'}, "
      f"the mirror held to the kept budget in both ({a['stats']['memory_limited']} and {b['stats']['memory_limited']} steps limited, "
      f"{a['stats']['mirrored']} and {b['stats']['mirrored']} experts mirrored)  {'ok' if ok else 'FAIL'}", flush=True)
fails += not ok
shutil.rmtree(tmp, ignore_errors=True)
print(f"{fails} FAILURES" if fails else "all ok")
sys.exit(1 if fails else 0)
