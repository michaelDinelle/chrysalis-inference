"""The picture path against independent references (GPU, Chrysalis stopped; --use qwen3.8-flash-next: Flash-Next, ~47
GiB of RAM, llama.cpp's reference on the CPU from its Q8_0 source):
  * preprocessing (chrysalis/vision.py) against the official transformers processor: the same grids, pixels within 2
steps
    of 255 on under 1% of them (its torchvision resize rounds differently from Pillow's; unresized pictures match exactly),
    at the official size limit and at smaller ones (max_pixels)
  * the vision tower (engine/vision.hip, through the engine) against transformers' Qwen3_5VisionModel in fp32 with Qwen's
    own weights, on the same patches: each token's cosine similarity and the relative error of all of them
  * what the tower's error does to the language model: llama.cpp's next-token distributions after the photo fed
Chrysalis's
    embeddings against fed the official ones, next to fed the official ones rounded to fp16 (as Chrysalis stores them)
  * the language model with a picture (image tokens, M-RoPE) against llama.cpp fed the same embeddings at the same
    positions: next-token KL divergence and top-1 agreement on the decode, prefill and mixed paths, as in compare_ref.py
  * a video: the frames Chrysalis takes, their pixels, the prompt's layout (each frame pair's time) and M-RoPE positions
    against transformers' video processor and Qwen3.5's get_rope_index, its frame pairs through the tower against
    transformers' vision model
The transformers references run in data/ref-venv (tools/vision_ref.py); they and llama.cpp's logits are kept in
build/vision/ (--fresh makes them again).
  data/venv/bin/python tools/test_vision.py [--fresh] [--use qwen3.8-flash-next]"""
import base64, hashlib, io, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw, ImageFont
from chrysalis import vision
from chrysalis.chat import Renderer, Tokens
from chrysalis.engine import IMG_BASE, IMG_ROWS, Engine
from chrysalis.models import MODELS
from chrysalis.server import Pictures
from chrysalis.sleep import vram_used, wait_vram_freed

USE = sys.argv[sys.argv.index("--use") + 1] if "--use" in sys.argv else "qwen3.8-27b"
FN = USE == "qwen3.8-flash-next"
MODEL, MMPROJ, LIB = MODELS[USE]["path"], MODELS[USE]["vision"], MODELS[USE]["lib"]
D = 2560 if FN else 5120
# llama.cpp: the 27B's own GGUF on the GPU; Flash-Next's Q8_0 source on the CPU (chrysalis-fn.gguf's weights are
# requantized, so there its quantization sets most of the KL, as in tools_fn/compare_fn.py)
REF_LOGITS = (["build_fn/ref_logits_mm_fn", "data/models/flash-next/src/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"] if FN
              else ["build/ref_logits_mm", MODEL])
REF = "data/ref-venv/bin/python"
OUT = "build/vision" + ("/fn" if FN else "")
# Mean KL after a picture (the two largest left out, see stats). The 27B: as text (compare_ref.py). Flash-Next, measured
# 2026-10-18 (decode 0.044, prefill 0.018, mixed 0.038): its requantized weights set most of it, and the picture makes the
# conversation three times as sensitive. Against llama.cpp the decode moves 0.044 here and 0.015 with a description in the
# picture's place; two Chrysalis runs that differ only in rounding (the cold experts' split) 0.0075 and 0.0025.
MAX_KL = {"decode": 0.06, "prefill": 0.03, "mixed": 0.05} if FN else {"decode": 0.004, "prefill": 0.004, "mixed": 0.004}
# The largest KL of a path: the 27B's stay under 0.12; Flash-Next's reach 2.1 at the conversation's most sensitive position
# (with the rows and columns of its M-RoPE swapped: 2.9, which its trimmed mean and top-1 then catch on the mixed path).
MAX_ONE = 4.0 if FN else 1.0
# The tower against fp32, measured 2026-10-18: the 27B's worst token 0.9993, relative error 0.45-1.2% (transformers itself
# in bf16: 0.947, 7.3%). Flash-Next's tower is chaotic at a few tokens of the photo (0.89 and 0.93; any fp16 rounding
# moves them, emulated one rounding at a time; transformers in bf16 takes one to 0.08), its median token 0.999993 and 97%
# of them over 0.998: so it is held at its 5th percentile, and both at their median and relative error (photo: 2.9%).
COS_AT = 0.05 if FN else 0.0                 # the share of tokens that may fall below MIN_COS
MIN_COS, MIN_MEDIAN, MAX_REL = 0.998, 0.9999, 0.04 if FN else 0.02
os.makedirs(OUT, exist_ok=True)
fresh = "--fresh" in sys.argv
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{name:52s} {'ok' if ok else 'FAIL'}  {detail}")


def cached(path, make):
    if fresh or not os.path.exists(path):
        make()
    if not os.path.exists(path):
        raise SystemExit(f"no reference at {path}: see data/ref-venv in AGENTS.md")
    return path


def picture(w, h, seed):
    """Shapes and text on a gradient: a picture whose size needs resizing (or not) and with something to read."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = Image.fromarray(np.stack([(xx * 255 // max(w - 1, 1)), (yy * 255 // max(h - 1, 1)), (xx + yy + seed * 40) % 256], -1).astype(np.uint8))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    for _ in range(6):
        x0, y0 = int(rng.integers(0, w - 10)), int(rng.integers(0, h - 10))
        d.ellipse([x0, y0, x0 + int(rng.integers(8, max(9, w // 3))), y0 + int(rng.integers(8, max(9, h // 3)))], fill=tuple(int(c) for c in rng.integers(0, 256, 3)))
    d.text((w // 10, h // 2), f"CHRYSALIS {seed * 7}", fill=(0, 0, 0))
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


pictures = {"photo 640x480 (no resize)": open("data/llama.cpp/tools/mtmd/test-1.jpeg", "rb").read(),
            "1000x700 (resized)": picture(1000, 700, 1), "120x90 (enlarged to the minimum)": picture(120, 90, 2),
            "300x1800 (tall)": picture(300, 1800, 3)}
def engine():
    e = Engine(MODEL, max_seqs=3, umax=MODELS[USE]["chunk"], lib=LIB)
    if FN:                       # the cold experts' CPU / GPU split fixed, so every run rounds alike (tools_fn/compare_fn.py)
        import ctypes as C
        e.lib.chrysalis_tune.restype, e.lib.chrysalis_tune.argtypes = C.c_double, [C.c_char_p, C.c_double]
        e.lib.chrysalis_tune(b"split", 1)
        e.lib.chrysalis_tune(b"gfrac", 0.3)
        e.lib.chrysalis_tune(b"experts", 0)
    e.vision_open(MMPROJ)
    return e


def preprocessing(name, data, limit=None):
    """chrysalis/vision.py's patches against the official processor's (given the same max_pixels); returns
    Chrysalis's."""
    key = hashlib.sha1(data).hexdigest()[:12]
    src, out = f"{OUT}/{key}.img", f"{OUT}/{key}.proc{limit or ''}.npz"
    open(src, "wb").write(data)
    patches, grid = vision.patches(data) if limit is None else vision.patches(data, limit)
    proc = np.load(cached(out, lambda: subprocess.run([REF, "tools/vision_ref.py", "processor", src, out] + ([str(limit)] if limit else []), check=True)))
    steps = np.abs(patches - proc["patches"]) * 255 / 2 if tuple(proc["grid"]) == grid else None
    check(f"preprocessing, {name}", steps is not None and steps.max() <= 2.001 and (steps > 1e-3).mean() < 0.01,
          f"grid {grid}" + (f", max {steps.max():.2f} steps, {100 * (steps > 1e-3).mean():.2f}% of values differ" if steps is not None else f" vs {tuple(proc['grid'])}"))
    return patches, grid


eng = engine()
for slot, (name, data) in enumerate(pictures.items()):
    key = hashlib.sha1(data).hexdigest()[:12]
    patches, grid = preprocessing(name, data)
    np.savez(f"{OUT}/{key}.patches.npz", patches=patches, grid=np.array(grid))
    ref = np.load(cached(f"{OUT}/{key}.tower.npy", lambda: subprocess.run([REF, "tools/vision_ref.py", "tower", f"{OUT}/{key}.patches.npz",
                                                                       f"{OUT}/{key}.tower.npy"] + (["fn"] if FN else []), check=True)))
    ok = eng.image_encode(slot, patches, grid)
    ours = eng.image_read(slot, len(ref), D) if ok else np.zeros_like(ref)
    cos = (ours * ref).sum(1) / np.maximum(np.linalg.norm(ours, axis=1) * np.linalg.norm(ref, axis=1), 1e-30)
    rel = float(np.linalg.norm(ours - ref) / np.linalg.norm(ref))
    low = float(np.quantile(cos, COS_AT)) if ok else 0.0
    check(f"vision tower, {name}", ok and len(ours) == len(ref) and low >= MIN_COS and np.median(cos) >= MIN_MEDIAN and rel <= MAX_REL,
          f"{len(ref)} tokens, cosine min {cos.min():.5f}" + (f", 5th percentile {low:.5f}" if COS_AT else "") +
          f", median {np.median(cos):.6f}, relative error {rel:.1e}")

# Pictures read at fewer pixels (a request's max_pixels, an image part's detail "low"): the processor given that max_pixels.
preprocessing("photo at detail low", pictures["photo 640x480 (no resize)"], vision.LOW_DETAIL)
preprocessing("1000x700 at 300,000 pixels", pictures["1000x700 (resized)"], 300000)

# The language model on a conversation about the photo: image tokens from slot 0, positions as chrysalis/vision.py lays them out.
tok = Tokens("data/models/tokenizer.json")
photo = pictures["photo 640x480 (no resize)"]
gh, gw = vision.grid(photo)
ids = list(range(IMG_BASE, IMG_BASE + gh * gw // 4))
url = "data:image/jpeg;base64," + base64.b64encode(photo).decode()
conv = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "What is the headline of this front page, and when was it printed?"}]},
        {"role": "assistant", "content": "The headline reads \"MEN WALK ON MOON\". This is The New York Times of Monday, July 21, 1969, "
                                         "printed the morning after Apollo 11's lunar module landed and Neil Armstrong stepped onto the surface."}]
toks, _ = Renderer(tok, lambda k: None, images=lambda u: ids).render(conv, enable_thinking=False)
toks = np.array(toks, np.int32)
img = np.flatnonzero(toks >= IMG_BASE)
a, b = int(img[0]), int(img[-1]) + 1
pos = vision.positions([("text", a), ("image", gh // 2, gw // 2), ("text", len(toks) - b)])
n = len(toks)
emb = eng.image_read(0, b - a, D)
held = vram_used()
eng.shutdown()                                           # llama.cpp needs the VRAM
wait_vram_freed(held)
toks.tofile(f"{OUT}/llm.tokens.i32")
pos.tofile(f"{OUT}/llm.pos.i32")


def llama_logits(embd):
    """llama.cpp's logits for the conversation with these embeddings (kept, by their hash, in OUT)."""
    embd = np.ascontiguousarray(embd, np.float32)
    path = f"{OUT}/llm.ref.{hashlib.sha1(embd.tobytes() + toks.tobytes()).hexdigest()[:12]}.f32"
    if fresh or not os.path.exists(path):
        if FN:
            print("reference: llama.cpp on the Q8_0 source (CPU, weights streamed from disk, a few minutes)...", flush=True)
        embd.tofile(f"{OUT}/llm.embd.f32")
        subprocess.run([*REF_LOGITS, f"{OUT}/llm.tokens.i32", f"{OUT}/llm.pos.i32", f"{OUT}/llm.embd.f32", path],
                       check=True, stderr=subprocess.DEVNULL)
    return np.fromfile(path, np.float32).reshape(n, -1)


def logsoftmax(x):
    x = x.astype(np.float64)
    x -= x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def kl_rows(p_logits, q_logits):
    lp, lq = logsoftmax(p_logits), logsoftmax(q_logits)
    return (np.exp(lp) * (lp - lq)).sum(-1)


ref = llama_logits(emb)                                  # made again whenever Chrysalis's embeddings change
# The tower's error as the language model sees it, on the text after the photo. Its predictions at a few positions are
# on a knife's edge (any change to the embeddings moves them: llama.cpp fed the official embeddings rounded to fp16
# moves by KL 0.4 at one), so the median and the mean of the smallest 90% are compared with that rounding's own.
# Measured 2026-10-18 (Chrysalis's tower / transformers' in bf16 / a planted bug, the position table resampled without aligned
# corners, against the rounding): the 27B's median 1.2x / 10x / 13x and its mean 2.1x / 9x / 14x; Flash-Next's 2.0x / 1.6x
# and 2.0x / 1.7x (its predictions there move with any rounding).
official = np.load(f"{OUT}/{hashlib.sha1(photo).hexdigest()[:12]}.tower.npy")[:b - a]
fp32, fp16 = llama_logits(official)[b - 1:], llama_logits(official.astype(np.float16).astype(np.float32))[b - 1:]
ours_kl, floor_kl = np.sort(kl_rows(fp32, ref[b - 1:])), np.sort(kl_rows(fp32, fp16))
del fp32, fp16                                           # 0.4 GB each, and Flash-Next's engine needs the RAM
m90 = lambda k: float(k[:int(0.9 * len(k))].mean())
check("the tower's error, as the language model sees it", np.median(ours_kl) <= 3 * np.median(floor_kl) and m90(ours_kl) <= 3 * m90(floor_kl),
      f"KL against the official embeddings: median {np.median(ours_kl):.2e} (rounded to fp16: {np.median(floor_kl):.2e}), "
      f"mean of the smallest 90% {m90(ours_kl):.2e} ({m90(floor_kl):.2e})")
eng = engine()
eng.image_encode(0, *vision.patches(photo))
check("the tower gives the same embeddings again", np.array_equal(eng.image_read(0, b - a, D), emb))


# Two positions of this conversation are on a knife's edge: 0.1% noise on the picture's embeddings moves llama.cpp itself
# by KL 0.05 at one of them (measured; Chrysalis differs there by about as much), and the other is a near tie (0.57 / 0.43). So
# the mean leaves out a path's two largest KLs; a wrong position or embedding moves every position, and the largest
# must stay under MAX_ONE. Measured 2026-10-18 without them (27B): decode 0.0011 (text, compare_ref.py: 0.0022 with all
# positions).
def stats(name, pairs, max_kl):
    kls, same = [], []
    for i, ours in pairs:
        lp, lq = logsoftmax(ref[i]), logsoftmax(ours)
        kls.append(float((np.exp(lp) * (lp - lq)).sum()))
        same.append(ref[i].argmax() == ours.argmax())
    keep = np.argsort(kls)[:-2]
    trimmed, top1 = float(np.mean(np.array(kls)[keep])), float(np.mean(np.array(same)[keep]))
    check(f"with a picture: {name}", trimmed <= max_kl and max(kls) < MAX_ONE and top1 >= 0.95,
          f"{len(pairs)} positions; without the largest two: mean KL {trimmed:.5f}, top-1 {100 * top1:.1f}% "
          f"(with them {np.mean(kls):.5f}, {100 * np.mean(same):.1f}%); max {np.max(kls):.4f}")


s = eng.new_seq()
eng.prefill(s, toks[:b].tolist(), pos[:b])
pairs = [(b - 1, eng.last_logits())]
for i in range(b, n):                                    # decode: text after the picture at the sequence's own position
    eng.prefill(s, [int(toks[i])])
    pairs.append((i, eng.last_logits()))
eng.free_seq(s)
stats("decode after it (positions from the engine)", pairs, MAX_KL["decode"])
pairs = []
for cut in (b + 1, b + 2, b + 4, b + 8, b + 16, b + 32, (b + n) // 2, n):   # prefill: one chunk with text, the picture, text
    s = eng.new_seq()
    eng.prefill(s, toks[:cut].tolist(), pos[:cut])
    pairs.append((cut - 1, eng.last_logits()))
    eng.free_seq(s)
stats("prefill through it", pairs, MAX_KL["prefill"])
s = eng.new_seq()
mid = (a + b) // 2                                       # a chunk boundary inside the picture, then decode
eng.prefill(s, toks[:mid].tolist(), pos[:mid])
eng.prefill(s, toks[mid:b + 4].tolist(), pos[mid:b + 4])
pairs = [(b + 3, eng.last_logits())]
for i in range(b + 4, n):
    eng.prefill(s, [int(toks[i])])
    pairs.append((i, eng.last_logits()))
eng.free_seq(s)
stats("prefill split inside it, then decode", pairs, MAX_KL["mixed"])


# A video (chrysalis/vision.py, chrysalis/server.py Pictures): the frames Chrysalis takes, their size and pixels against
# transformers' video processor sampling all of ffmpeg's frames itself (Chrysalis's settings; cap_pixels_per_frame, as
# qwen-vl-utils reads videos); the prompt's tokens against the processor's expansion of <|video_pad|> (each frame pair's
# time) and their M-RoPE positions against Qwen3.5's get_rope_index; the frame pairs through the tower against
# transformers' vision model.
def clip():
    """Four seconds at 10 frames a second: a red circle moving, then a blue square and "CHRYSALIS 58"."""
    path = f"{OUT}/clip.mp4"
    if fresh or not os.path.exists(path):
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x360", "-r", "10",
                                "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        font = ImageFont.load_default(size=48)
        for i in range(40):
            img = Image.new("RGB", (640, 360), "white")
            d = ImageDraw.Draw(img)
            if i < 20:
                d.ellipse([200 + 4 * i, 100, 320 + 4 * i, 220], fill="red")
            else:
                d.rectangle([100, 120, 220, 240], fill="blue")
                d.text((300, 150), "CHRYSALIS 58", fill="black", font=font)
            enc.stdin.write(img.tobytes())
        enc.stdin.close()
        enc.wait()
    return path


path = clip()
data = open(path, "rb").read()
key = hashlib.sha1(data).hexdigest()[:12]


def every_frame():
    total, fps, (vh, vw) = vision.video_info(path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    np.savez(f"{OUT}/{key}.frames.npz", frames=np.frombuffer(raw, np.uint8).reshape(-1, vh, vw, 3), fps=fps)
    subprocess.run([REF, "tools/vision_ref.py", "video", f"{OUT}/{key}.frames.npz", f"{OUT}/{key}.video.npz"], check=True)


ref = np.load(cached(f"{OUT}/{key}.video.npz", every_frame))
pics, seen = Pictures(eng, D, 1 << 30), {}
conv = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(data).decode()}},
                                     {"type": "text", "text": "What happens in this video?"}]}]
vtoks, _ = Renderer(tok, lambda k: None, images=lambda item: pics.tokens(item, seen)).render(conv, enable_thinking=False)
vtoks = np.array(vtoks, np.int32)
video = next(iter(seen.values()))
pairs = vision.pair_patches(video.frames())
grid = (len(pairs),) + video.grid
steps = np.abs(pairs.reshape(-1, 2 * vision.PATCH_DIM) - ref["patches"]) * 255 / 2 if grid == tuple(ref["grid"]) else None
shown = tok.full.decode(np.where(vtoks >= IMG_BASE, tok.full.token_to_id("<|video_pad|>"), vtoks).tolist(), skip_special_tokens=False)
check("video: frames, pixels and prompt", video.indices.tolist() == ref["indices"].tolist() and steps is not None and steps.max() <= 2.001
      and (steps > 1e-3).mean() < 0.01 and f"<|vision_start|>{ref['text']}<|vision_end|>" in shown,
      f"frames {video.indices.tolist()}, grid {grid}" + (f", max {steps.max():.2f} steps on {100 * (steps > 1e-3).mean():.3f}%" if steps is not None else f" vs {tuple(ref['grid'])}"))
kinds = np.where(vtoks >= IMG_BASE, 2, 0).astype(np.int32)
rk = hashlib.sha1(kinds.tobytes() + np.array(grid).tobytes()).hexdigest()[:12]
np.savez(f"{OUT}/{rk}.kinds.npz", types=kinds, grids=np.array([grid]))
rpos = np.load(cached(f"{OUT}/{rk}.rope.npy", lambda: subprocess.run([REF, "tools/vision_ref.py", "rope", f"{OUT}/{rk}.kinds.npz", f"{OUT}/{rk}.rope.npy"], check=True)))
ours = pics.positions(vtoks)
check("video: M-RoPE positions", ours is not None and np.array_equal(ours, rpos), f"{len(vtoks)} tokens, {len(pairs)} frame pairs")
slots = [pic for a, b, pic in pics.runs(vtoks)]
pics.ensure(slots, seen)
ours = np.concatenate([eng.image_read(s_, video.grid[0] * video.grid[1] // 4, D) for s_ in slots])
np.savez(f"{OUT}/{key}.pairs.npz", patches=pairs.reshape(-1, 2 * vision.PATCH_DIM), grid=np.array(grid))
vref = np.load(cached(f"{OUT}/{key}.vtower.npy", lambda: subprocess.run([REF, "tools/vision_ref.py", "tower", f"{OUT}/{key}.pairs.npz",
                                                                       f"{OUT}/{key}.vtower.npy"] + (["fn"] if FN else []), check=True)))
cos = (ours * vref).sum(1) / np.maximum(np.linalg.norm(ours, axis=1) * np.linalg.norm(vref, axis=1), 1e-30) if ours.shape == vref.shape else np.zeros(1)
rel = float(np.linalg.norm(ours - vref) / np.linalg.norm(vref)) if ours.shape == vref.shape else 1.0
low = float(np.quantile(cos, COS_AT))
check("video: frame pairs through the tower", ours.shape == vref.shape and low >= MIN_COS and np.median(cos) >= MIN_MEDIAN and rel <= MAX_REL,
      f"{len(vref)} tokens, cosine min {cos.min():.5f}" + (f", 5th percentile {low:.5f}" if COS_AT else "") +
      f", median {np.median(cos):.6f}, relative error {rel:.1e}")
print("all ok" if not fails else f"{fails} FAILURES")
sys.exit(fails != 0)
