import json, math, os, random, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenizers import Tokenizer
from chrysalis.chat import Content, Tokens
from chrysalis.engine import STAGES
from chrysalis.server import DEFAULTS, EOG, RequestError, SAMPLER_LETTERS, SAMPLER_NAMES, Server, sampling_of

tok = Tokens(os.path.join(ROOT, "data/models/tokenizer.json"))
fails = 0


def check(ok, what):
    global fails
    fails += not ok
    print(f"{what:100s} {'ok' if ok else 'FAIL'}")


FLOATS = ["temperature", "dynatemp_range", "dynatemp_exponent", "top_p", "min_p", "top_n_sigma", "xtc_probability", "xtc_threshold",
          "typical_p", "repeat_penalty", "presence_penalty", "frequency_penalty", "dry_multiplier", "dry_base", "mirostat_tau", "mirostat_eta"]
INTS = ["top_k", "repeat_last_n", "dry_allowed_length", "dry_penalty_last_n", "mirostat", "min_keep", "n_probs", "n_predict"]


def ours(body, thinking):
    try:
        sp, bias, breakers, n_probs, n_predict = sampling_of(body, DEFAULTS[thinking], tok, -1)
    except RequestError as e:
        return {"error": str(e)}
    out = {k: getattr(sp, k) for k in FLOATS + INTS[:-2]}
    out.update(n_probs=n_probs, n_predict=-1 if n_predict is None else n_predict, dry_sequence_breakers=list(breakers),
               samplers=[STAGES[sp.order[i]] for i in range(sp.n_order)], ignore_eos=tok.eos <= {t for t, b in bias if b == -math.inf})
    return out


def theirs(bodies, thinking):
    d = DEFAULTS[thinking]
    head = json.dumps({**{k: d[k] for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty")}, "dry_penalty_last_n": -1})
    run = subprocess.run([os.path.join(ROOT, "build/llama_params")], input="\n".join([head] + [json.dumps(b) for b in bodies]) + "\n",
                         capture_output=True, text=True)
    lines = run.stdout.splitlines()
    if run.returncode or len(lines) != len(bodies):
        raise SystemExit(f"build/llama_params failed ({run.returncode}): {run.stderr[-400:]}")
    out = []
    for line in lines:
        j = json.loads(line)
        out.append(j if "error" in j else {**{k: j[k] for k in FLOATS + INTS[:-1] + ["dry_sequence_breakers", "samplers", "ignore_eos"]},
                                            "n_predict": j["n_predict"]})
    return out


def compatible_body(r):
    names = [n for n, v in SAMPLER_NAMES.items() if v in STAGES] + ["Top-K", "TEMP", "Min-P", "NUCLEUS"]
    pools = {
        "float": lambda: r.choice([0, 1, -1, 0.5, 2, 100, r.gauss(0, 2), r.uniform(0, 1), r.uniform(-3, 3), 3, -0.25]),
        "int": lambda: r.choice([0, 1, 2, 3, 5, 8, 64, r.randrange(0, 200), r.randrange(0, 20), -2, 2.0]),
        "junk": lambda: r.choice(["0.5", [1], {"a": 1}, None]),
    }
    body = {}
    for k in FLOATS + INTS:
        if r.random() < 0.15:
            body[k] = pools["junk"]() if r.random() < 0.05 else pools["float" if k in FLOATS else "int"]()
    if isinstance(body.get("repeat_penalty"), (int, float)) and body["repeat_penalty"] <= 0:
        body["repeat_penalty"] = 1.25
    if r.random() < 0.4:
        body["samplers"] = r.choice(["".join(r.choice("edskypmxt") for _ in range(r.randrange(0, 9))),
                                     [r.choice(names) for _ in range(r.randrange(0, 9))], ["top_k", 3]])
    if r.random() < 0.2:
        body["dry_sequence_breakers"] = r.choice([["\n"], ["User:", "\n\n"], [], "x", [1], ["*", "a b"]])
    if r.random() < 0.2:
        body["ignore_eos"] = r.choice([True, False, 1, "yes"])
    return body


DELIBERATE = {
    "true or false for a number: llama.cpp reads 1 or 0, Chrysalis says 400": lambda r: {r.choice(FLOATS + INTS): r.choice([True, False])},
    "a fraction for a whole number: llama.cpp truncates it, Chrysalis says 400": lambda r: {r.choice(INTS): r.choice([2.5, 0.25, 7.75])},
    "a repeat_penalty of 0 or less: llama.cpp's sampler refuses it later, Chrysalis says 400 at once": lambda r: {"repeat_penalty": r.choice([0, -1.5])},
    "samplers neither a list nor a string: llama.cpp ignores it, Chrysalis says 400": lambda r: {"samplers": r.choice([7, {"k": 1}])},
    "an unknown sampler name or letter: llama.cpp drops it, Chrysalis says 400": lambda r: {"samplers": r.choice(["kq", ["top_k", "bogus"]])},
    "infill or adaptive_p: llama.cpp runs them, Chrysalis has neither and says 400": lambda r: {"samplers": r.choice(["ka", ["infill"]])},
    "a window of -1: llama.cpp says 400, Chrysalis reads the whole context": lambda r: {r.choice(["repeat_last_n", "dry_penalty_last_n"]): -1},
}


r = random.Random(5)
bodies = [{}, {"samplers": "kpt"}, {"samplers": ["Top-K", "nucleus", "TEMP", "typ", "minp", "top-n-sigma", "penalties"]},
          {"samplers": "edskypmxt"}, {"samplers": []}, {"samplers": ""}, {"temperature": -1, "top_k": -5, "top_p": 3, "min_p": -1,
          "mirostat": 9, "xtc_probability": 2, "xtc_threshold": -1}, {"dry_base": 0.5}, {"dry_base": 1}, {"n_predict": 0},
          {"n_predict": -1}, {"n_predict": -2}, {"n_probs": 5}, {"min_keep": -1}, {"dry_allowed_length": -1}, {"ignore_eos": True},
          {"dry_sequence_breakers": ["\n\n", "User:"]}, {"dry_sequence_breakers": []}, {"temperature": "0.5"},
          {"temperature": None, "top_k": None}, {"mirostat": 1, "mirostat_tau": 3, "mirostat_eta": 0.2}, {"repeat_last_n": 0},
          {"typical_p": 0.5, "top_n_sigma": 1.5, "dynatemp_range": 0.5, "dynatemp_exponent": 2}] + [compatible_body(r) for _ in range(3000)]
for thinking in (True, False):
    mode = "thinking" if thinking else "fast"
    want = theirs(bodies, thinking)
    same, first = 0, ""
    for body, w in zip(bodies, want):
        g = ours(body, thinking)
        ok = ("error" in g) == ("error" in w) and ("error" in g or g == w)
        same += ok
        if not ok and not first:
            first = f"{json.dumps(body)}\n    llama.cpp: {w}\n    chrysalis: {g}"
    check(same == len(bodies), f"{mode} mode: requests parse as llama.cpp's server parses them ({same}/{len(bodies)})")
    if first:
        print("  first difference:", first)
    errors = sum("error" in w for w in want)
    check(300 < errors < len(bodies) // 2, f"{mode} mode: llama.cpp refuses {errors} of them, so the refusals are compared too")
    bases = [b for b, w in zip(bodies, want) if "error" not in w and "samplers" not in b and "repeat_last_n" not in b and "dry_penalty_last_n" not in b]
    for why, change in DELIBERATE.items():
        trial = [{**r.choice(bases), **change(r)} for _ in range(40)]
        ok = 0
        for body, w in zip(trial, theirs(trial, thinking)):
            g = ours(body, thinking)
            ok += ("error" in w and g.get(next(k for k in ("repeat_last_n", "dry_penalty_last_n") if body.get(k) == -1)) == -1) \
                if why.startswith("a window") else ("error" in g and "error" not in w)
        check(ok == len(trial), f"{mode} mode, on purpose: {why} ({ok}/{len(trial)})")

parse = lambda b: sampling_of(b, DEFAULTS[True], tok, -1)[1]
plain = Tokenizer.from_file(os.path.join(ROOT, "data/models/tokenizer.json"))
plain.encode_special_tokens = True
text, markup = "Hello, world", "say <|im_end|> twice"
check(parse({"logit_bias": {"12": 3, text: False}}) == parse({"logit_bias": [[12, 3], [text, False]]})
      == [(12, 3.0)] + [(t, -math.inf) for t in plain.encode(text, add_special_tokens=False).ids],
      "logit_bias's object and list forms agree; a string biases its tokens, false bans them")
got = [t for t, _ in parse({"logit_bias": [[markup, -1]]})]
check(got == plain.encode(markup, add_special_tokens=False).ids and not tok.eos & set(got),
      "a string in logit_bias is plain text, as llama.cpp reads it: <|im_end|> there is not the end token")
refused = []
for bad in ({"12": True}, [[12]], [[12, "x"]], {str(tok.full.get_vocab_size()): 1}, [[-1, 1]], "12", {"12": None}):
    try:
        parse({"logit_bias": bad})
    except RequestError as e:
        refused.append(str(e))
check(len(refused) == 7 and all(m.startswith("logit_bias") for m in refused), "a malformed logit_bias is a 400 that names it")
check(set(dict(parse({"ignore_eos": True}))) == {tok.full.token_to_id(t) for t in EOG} >= tok.eos,
      "ignore_eos bans the stop tokens and the other end-of-generation tokens llama.cpp bans for this vocabulary")

srv = Server.__new__(Server)
srv.tok = tok
texts = tok.full.decode_batch([[i] for i in range(tok.full.get_vocab_size())], skip_special_tokens=False)
flat = srv.restart_sequences(("\n", ":", "\"", "*"))
singles = {flat[i + 1] for i in range(0, len(flat), 2)}
check(set(flat[0::2]) == {1} and singles == {i for i, t in enumerate(texts) if any(c in t for c in "\n:\"*")},
      f"the default breakers are the {len(singles)} tokens whose text holds \\n, :, \" or *")
flat = srv.restart_sequences(("User:",))
seqs, i = [], 0
while i < len(flat):
    seqs.append(flat[i + 1:i + 1 + flat[i]])
    i += 1 + flat[i]
own = tok.encode([Content("User:")])[0]
check(own in seqs and all(any(texts[s[0]].endswith("User:"[:k]) for k in range(1, 5)) for s in seqs),
      f"a longer breaker restarts at its own tokens {[texts[t] for t in own]} among {len(seqs)} sequences")
print("FAILURES" if fails else "all ok")
sys.exit(fails != 0)
