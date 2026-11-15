import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from chrysalis import vision

MEDIA = os.path.join(ROOT, "tests", "media")
failed = 0


def check(name, ok, detail=""):
    global failed
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))


ffmpeg = np.load(os.path.join(MEDIA, "ffmpeg.npz"))
recorded = sorted({k.split("|")[0] for k in ffmpeg.files})
clips = sorted(f for f in os.listdir(MEDIA) if f not in ("ffmpeg.npz", "poppler.json"))
check("every clip in tests/media has ffmpeg's reading of it", recorded == clips, f"{len(recorded)} recorded, {len(clips)} clips")
for name in recorded:
    with open(os.path.join(MEDIA, name), "rb") as f:
        data = f.read()
    try:
        video = vision.Video(data)
        frames, error = video.frames(), None
    except vision.VideoError as e:
        error = str(e)
    if f"{name}|error" in ffmpeg.files:
        check(f"{name}: refused as ffmpeg refused it", error == str(ffmpeg[f"{name}|error"]), repr(error))
        continue
    was = json.loads(str(ffmpeg[f"{name}|meta"]))
    check(f"{name}: the frames ffmpeg picked, at its rate, size and times", error is None and video.indices.tolist() == was["indices"]
          and video.fps == was["fps"] and list(video.shown) == was["shown"] and list(video.size) == was["size"] and video.times == was["times"],
          error or f"frames {video.indices.tolist()} at {video.fps:g} fps, {video.shown[1]}x{video.shown[0]}")
    want = ffmpeg[f"{name}|frames"]
    steps = np.abs(frames.astype(int) - want.astype(int)) if error is None and frames.shape == want.shape else None
    check(f"{name}: pixels within 2 steps of ffmpeg's on under 1% of values", steps is not None and steps.max() <= 2 and (steps > 0).mean() < 0.01,
          f"max {steps.max()} steps on {100 * (steps > 0).mean():.3f}%" if steps is not None else f"{error or frames.shape}")

print("all ok" if not failed else f"{failed} failed")
sys.exit(1 if failed else 0)
