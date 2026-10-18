"""Pictures for Qwen3.8's vision tower: decoding, its official preprocessing (preprocessor_config.json: 16-pixel patches,
merged 2 x 2 into one token each, the picture resized to between 65,536 and 16,777,216 pixels), and the positions its
tokens take in the language model (M-RoPE: a token at row r, column c of an image that starts at position p sits at
(p, p + r, p + c); the text after it goes on from p + max(rows, columns)).

Resizing is Pillow's antialiased bicubic on 8-bit RGB. The official processor resizes with torchvision's native uint8
kernel (a port of Pillow-SIMD on x86 CPUs, a float path elsewhere), which rounds differently: measured on photos, noise
and gradients from 33 x 4000 to 4000 x 3000, the same grids and at most 2 steps of 255 apart on under 1% of the pixels;
pictures that need no resize come out identical. Transparent pictures are laid on white first, as vLLM does (dropping the
alpha channel, as the transformers processor does, shows whatever colour transparent pixels happen to hold)."""
import base64
import hashlib
import io
import math

import numpy as np

PATCH, MERGE = 16, 2
FACTOR = PATCH * MERGE                   # a side is a whole number of merged tokens
MIN_PIXELS, MAX_PIXELS = 65536, 16777216
LOW_DETAIL = 512 * 512                   # OpenAI's detail "low": the picture seen at about 512 x 512 (256 tokens)
PATCH_DIM = 3 * PATCH * PATCH            # colour, row, column (the two identical frames are summed in the weights)


class ImageError(ValueError):
    """A picture kiln cannot read."""


def smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """The official target size (transformers' qwen2_vl smart_resize, Python's round included)."""
    if max(h, w) / min(h, w) > 200:
        raise ImageError(f"a picture can be at most 200 times as long as it is wide (this one is {max(h, w) / min(h, w):.0f})")
    h_bar, w_bar = round(h / factor) * factor, round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar, w_bar = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def image_bytes(url, what="images"):
    """The bytes of an image_url (or video_url): a data URL (kiln fetches nothing from the network)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ImageError(f"{what} must be sent inline, as data: URLs (kiln fetches nothing from the network)")
    head, _, data = url.partition(",")
    try:
        return base64.b64decode(data, validate=False) if head.endswith(";base64") else data.encode("latin-1")
    except (ValueError, UnicodeEncodeError) as e:
        raise ImageError(f"cannot decode the image data: {e}") from None


def digest(data):
    return hashlib.blake2b(data, digest_size=16).digest()


def _open(data):
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ImageError(f"cannot read the picture: {e}") from None
    return ImageOps.exif_transpose(img)  # phone photos stand the way they were taken


def size(data):
    """(height, width) as the picture is shown (EXIF orientation applied), from its header where possible."""
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ImageError(f"cannot read the picture: {e}") from None
    w, h = img.size
    orientation = img.getexif().get(0x0112, 1)
    return (w, h) if orientation in (5, 6, 7, 8) else (h, w)   # turned a quarter: the sides swap


def pixel_limit(*limits, cap=MAX_PIXELS):
    """The most pixels a picture is resized to: the first of `limits` given (a picture's own, then its request's), held
    between MIN_PIXELS and the server's cap; the cap when none is. Raises ImageError for one that is not a number."""
    for limit in limits:
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, (int, float)):
                raise ImageError(f"max_pixels must be a number (got {limit!r})")
            return max(MIN_PIXELS, min(int(cap), int(limit)))
    return int(cap)


def grid(data, max_pixels=MAX_PIXELS):
    """(rows, columns) of patches the picture becomes; its tokens are rows * columns / 4."""
    h, w = smart_resize(*size(data), max_pixels=max_pixels)
    return h // PATCH, w // PATCH


def patches(data, max_pixels=MAX_PIXELS):
    """The picture as the vision tower takes it: float32 [rows * columns][768], each patch its colours, rows and columns
    normalized to [-1, 1], the patches in merge-block order (the four of each merged token together, in reading order;
    the blocks in reading order). Also returns (rows, columns)."""
    from PIL import Image
    img = _open(data)
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    img = img.convert("RGB")
    h, w = smart_resize(img.height, img.width, max_pixels=max_pixels)
    if (h, w) != (img.height, img.width):
        img = img.resize((w, h), Image.Resampling.BICUBIC)
    x = np.asarray(img, dtype=np.float32) * np.float32(1 / 255)
    x = (x - np.float32(0.5)) / np.float32(0.5)
    gh, gw = h // PATCH, w // PATCH
    x = x.reshape(gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH, 3)   # block row, in row, pixel row, block col, ...
    x = x.transpose(0, 3, 1, 4, 6, 2, 5)                                    # block row, block col, in row, in col, c, y, x
    return np.ascontiguousarray(x.reshape(gh * gw, PATCH_DIM)), (gh, gw)


# Videos. Each pair of frames is a temporal patch the tower reads as it reads a picture (it attends within a pair), its
# merged tokens preceded in the prompt by the pair's time ("<1.5 seconds>") and set between <|vision_start|> and
# <|vision_end|> inside the video's own pair of them (transformers' Qwen3VLProcessor); in M-RoPE each pair sits as a
# picture would (Qwen3.5's get_rope_index). Frames are taken 2 a second (4 to 768, transformers' sample_frames) and sized
# within a budget of 24,576 tokens for the whole video, at most 768 a frame pair (transformers' Qwen3VLVideoProcessor with
# cap_pixels_per_frame, qwen-vl-utils' reference behaviour). ffmpeg decodes the frames kiln takes, and Pillow resizes them
# as it does a picture.
VIDEO_FPS, VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES = 2.0, 4, 768
VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, VIDEO_FRAME_TOKENS = 4096, 24576 * FACTOR * FACTOR, 768


class VideoError(ImageError):
    """A video kiln cannot read."""


def video_info(path):
    """(frames, frames per second, height, width as shown) of a video file, by ffprobe (counting its packets)."""
    import json
    import subprocess
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                              "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_packets:stream_side_data=rotation",
                              "-of", "json", path], capture_output=True, check=True, timeout=120).stdout
    except FileNotFoundError:
        raise VideoError("kiln needs ffmpeg (ffprobe) to read videos") from None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise VideoError("cannot read the video") from None
    streams = json.loads(out).get("streams") or []
    if not streams:
        raise VideoError("the file has no video stream")
    st = streams[0]

    def rate(r):
        n, _, d = str(r or "0/1").partition("/")
        return float(n) / float(d or 1) if float(d or 1) else 0.0
    fps = rate(st.get("avg_frame_rate")) or rate(st.get("r_frame_rate"))
    frames = int(st.get("nb_read_packets") or 0)
    if frames < 1 or fps <= 0:
        raise VideoError("cannot tell the video's frames or frame rate")
    turned = any(abs(int(d.get("rotation", 0))) in (90, 270) for d in st.get("side_data_list") or [])
    h, w = int(st["height"]), int(st["width"])
    return frames, fps, (w, h) if turned else (h, w)


def video_indices(frames, fps, rate=VIDEO_FPS):
    """The frames kiln takes: `rate` a second, evenly spread (transformers' sample_frames)."""
    n = min(max(int(frames / fps * rate), VIDEO_MIN_FRAMES), VIDEO_MAX_FRAMES, frames)
    return np.linspace(0, frames - 1, n).round().astype(int)


def video_size(n, h, w):
    """(height, width) of n frames of h x w (transformers' video smart_resize, the budget shared by frames, each capped)."""
    per = max(min(VIDEO_FRAME_TOKENS * FACTOR * FACTOR, VIDEO_MAX_PIXELS // n), int(VIDEO_MIN_PIXELS * 1.05))
    if h < FACTOR or w < FACTOR:
        scale = max(FACTOR / h, FACTOR / w)
        h, w = int(h * scale), int(w * scale)
    if max(h, w) / min(h, w) > 200:
        raise VideoError(f"a video can be at most 200 times as wide as it is tall (this one is {max(h, w) / min(h, w):.0f})")
    h_bar, w_bar, t_bar = round(h / FACTOR) * FACTOR, round(w / FACTOR) * FACTOR, round(n / 2) * 2
    if t_bar * h_bar * w_bar > per * n:
        beta = math.sqrt(n * h * w / (per * n))
        h_bar = max(FACTOR, math.floor(h / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(w / beta / FACTOR) * FACTOR)
    elif t_bar * h_bar * w_bar < VIDEO_MIN_PIXELS:
        beta = math.sqrt(VIDEO_MIN_PIXELS / (n * h * w))
        h_bar, w_bar = math.ceil(h * beta / FACTOR) * FACTOR, math.ceil(w * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def pair_times(indices, fps):
    """Each frame pair's time in seconds: its two frames' mean (transformers' _calculate_timestamps)."""
    idx = list(indices) + [indices[-1]] * (len(indices) % 2)
    t = [i / fps for i in idx]
    return [(t[i] + t[i + 1]) / 2 for i in range(0, len(t), 2)]


class Video:
    """A video file read for the vision tower: frames (taken, their size), its pairs' times, and its frames decoded on
    demand (patches()). data: the file's bytes; rate: frames a second to take."""

    def __init__(self, data, rate=VIDEO_FPS):
        import tempfile
        self.data = data
        with tempfile.NamedTemporaryFile(prefix="kiln-video-") as f:
            f.write(data)
            f.flush()
            total, self.fps, self.shown = video_info(f.name)
        self.indices = video_indices(total, self.fps, rate)
        self.size = video_size(len(self.indices), *self.shown)
        self.times = pair_times(self.indices, self.fps)

    @property
    def grid(self):
        """(rows, columns) of patches of a frame pair; its tokens are rows * columns / 4."""
        return self.size[0] // PATCH, self.size[1] // PATCH

    def frames(self):
        """The frames taken, resized: uint8 [n][height][width][3]."""
        import subprocess
        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(prefix="kiln-video-") as f:
            f.write(self.data)
            f.flush()
            pick = "+".join(f"eq(n,{i})" for i in self.indices)
            try:     # ffmpeg turns a rotated video upright, as video_info reports its size
                raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f.name, "-vf", f"select='{pick}'", "-fps_mode", "passthrough",
                                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True, timeout=600).stdout
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                raise VideoError("cannot decode the video") from None
        h, w = self.shown
        n = len(raw) // (w * h * 3)
        if n != len(self.indices) or n * w * h * 3 != len(raw):
            raise VideoError(f"the video decoded to {len(raw) / (w * h * 3):g} of the {len(self.indices)} frames kiln took")
        out = np.empty((n, self.size[0], self.size[1], 3), np.uint8)
        frames = np.frombuffer(raw, np.uint8).reshape(n, h, w, 3)
        for i in range(n):
            img = Image.fromarray(frames[i])
            out[i] = np.asarray(img if img.size == (self.size[1], self.size[0]) else img.resize((self.size[1], self.size[0]), Image.Resampling.BICUBIC))
        return out


def pair_patches(frames):
    """Frame pairs (uint8 [n][height][width][3], an odd last frame doubled as transformers does) as the vision tower takes
    them: float32 [pairs][rows * columns][1536], each patch colour by colour, the two frames, then rows and columns; the
    patches in merge-block order (as patches() lays a picture's)."""
    if len(frames) % 2:
        frames = np.concatenate([frames, frames[-1:]])
    x = frames.astype(np.float32) * np.float32(1 / 255)
    x = (x - np.float32(0.5)) / np.float32(0.5)
    n, h, w = len(frames) // 2, frames.shape[1], frames.shape[2]
    gh, gw = h // PATCH, w // PATCH
    x = x.reshape(n, 2, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH, 3)   # pair, frame, block row, in row, y, ...
    x = x.transpose(0, 2, 5, 3, 6, 8, 1, 4, 7)                                    # pair, block row, block col, in row, in col, c, frame, y, x
    return np.ascontiguousarray(x.reshape(n, gh * gw, 2 * PATCH_DIM))


def positions(segments):
    """M-RoPE positions of a prompt, int32 [n][3] (t, h, w). segments: ("text", n) or ("image", rows, columns) in order,
    rows and columns of merged tokens."""
    out, p = [], 0
    for seg in segments:
        if seg[0] == "text":
            r = np.arange(p, p + seg[1], dtype=np.int32)
            out.append(np.stack([r, r, r], axis=1))
            p += seg[1]
        else:
            rows, cols = seg[1], seg[2]
            rr, cc = np.divmod(np.arange(rows * cols, dtype=np.int32), cols)
            out.append(np.stack([np.full(rows * cols, p, np.int32), p + rr, p + cc], axis=1))
            p += max(rows, cols)
    return np.concatenate(out) if out else np.zeros((0, 3), np.int32)
