import base64
import hashlib
import io
import math
import struct
import time

import numpy as np

PATCH, MERGE = 16, 2
FACTOR = PATCH * MERGE
MIN_PIXELS, MAX_PIXELS = 65536, 16777216
LOW_DETAIL = 512 * 512
PATCH_DIM = 3 * PATCH * PATCH


class ImageError(ValueError):
    pass


def smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
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
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ImageError(f"{what} must be sent inline, as data: URLs (Chrysalis fetches nothing from the network)")
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
    return ImageOps.exif_transpose(img)


def size(data):
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ImageError(f"cannot read the picture: {e}") from None
    w, h = img.size
    orientation = img.getexif().get(0x0112, 1)
    return (w, h) if orientation in (5, 6, 7, 8) else (h, w)


def pixel_limit(*limits, cap=MAX_PIXELS):
    for limit in limits:
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, (int, float)):
                raise ImageError(f"max_pixels must be a number (got {limit!r})")
            return max(MIN_PIXELS, min(int(cap), int(limit)))
    return int(cap)


def grid(data, max_pixels=MAX_PIXELS):
    h, w = smart_resize(*size(data), max_pixels=max_pixels)
    return h // PATCH, w // PATCH


def patches(data, max_pixels=MAX_PIXELS):
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
    x = x.reshape(gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH, 3)
    x = x.transpose(0, 3, 1, 4, 6, 2, 5)
    return np.ascontiguousarray(x.reshape(gh * gw, PATCH_DIM)), (gh, gw)


VIDEO_FPS, VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES = 2.0, 4, 768
VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, VIDEO_FRAME_TOKENS = 4096, 24576 * FACTOR * FACTOR, 768


class VideoError(ImageError):
    pass


def video_info(data):
    import av
    deadline = time.monotonic() + 120
    try:
        with av.open(io.BytesIO(data)) as c:
            if not c.streams.video:
                raise VideoError("the file has no video stream")
            st = c.streams.video[0]
            fps = float(st.average_rate or 0) or float(st.base_rate or 0)
            frames, turns = 0, None
            for packet in c.demux(st):
                if time.monotonic() > deadline:
                    raise VideoError("cannot read the video")
                frames += packet.size > 0 or packet.dts is not None
                if turns is None:
                    turns = next((upright(f) for f in decoded(packet)), None)
            h, w = st.codec_context.height, st.codec_context.width
    except av.FFmpegError:
        raise VideoError("cannot read the video") from None
    if frames < 1 or fps <= 0:
        raise VideoError("cannot tell the video's frames or frame rate")
    turned = bool(turns) and turns[0][0] == "transpose"
    return frames, fps, (w, h) if turned else (h, w)


def decoded(packet):
    import av
    try:
        return packet.decode()
    except av.FFmpegError:
        return []


def upright(frame):
    import av
    side = frame.side_data.get(av.sidedata.sidedata.Type.DISPLAYMATRIX)
    if side is None:
        return []
    m = struct.unpack("9i", bytes(side))
    sx, sy = math.hypot(m[0], m[3]) / 65536, math.hypot(m[1], m[4]) / 65536
    if not sx or not sy:
        return []
    turn = math.degrees(math.atan2(m[1] / 65536 / sy, m[0] / 65536 / sx))
    theta = math.copysign(math.floor(abs(turn) + 0.5), turn)
    theta -= 360 * math.floor(theta / 360 + 0.9 / 360)
    if abs(theta - 90) < 1:
        return [("transpose", "cclock_flip" if m[3] > 0 else "clock")]
    if abs(theta - 180) < 1:
        return [("hflip", None)] * (m[0] < 0) + [("vflip", None)] * (m[4] < 0)
    if abs(theta - 270) < 1:
        return [("transpose", "clock_flip" if m[3] < 0 else "cclock")]
    if abs(theta) > 1:
        return [("rotate", f"{theta:f}*PI/180")]
    return [("vflip", None)] * (m[4] < 0)


def video_frames(data, keep=None):
    import av
    import av.filter
    deadline, graphs, n = time.monotonic() + 600, {}, 0
    try:
        with av.open(io.BytesIO(data)) as c:
            st = c.streams.video[0]
            st.thread_type = "AUTO"
            for packet in c.demux(st):
                if time.monotonic() > deadline:
                    raise VideoError("cannot decode the video")
                for frame in decoded(packet):
                    if keep is None or n in keep:
                        turns = upright(frame)
                        key = (frame.width, frame.height, frame.format.name, tuple(turns))
                        if key not in graphs:
                            g = av.filter.Graph()
                            node = g.add_buffer(width=frame.width, height=frame.height, format=frame.format, time_base=st.time_base)
                            for name, arg in turns + [("format", "rgb24")]:
                                nxt = g.add(name, arg)
                                node.link_to(nxt)
                                node = nxt
                            node.link_to(g.add("buffersink"))
                            g.configure()
                            graphs[key] = g
                        graphs[key].push(frame)
                        yield graphs[key].pull().to_ndarray()
                    n += 1
    except av.FFmpegError:
        raise VideoError("cannot decode the video") from None


def video_indices(frames, fps, rate=VIDEO_FPS):
    n = min(max(int(frames / fps * rate), VIDEO_MIN_FRAMES), VIDEO_MAX_FRAMES, frames)
    return np.linspace(0, frames - 1, n).round().astype(int)


def video_size(n, h, w):
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
    idx = list(indices) + [indices[-1]] * (len(indices) % 2)
    t = [i / fps for i in idx]
    return [(t[i] + t[i + 1]) / 2 for i in range(0, len(t), 2)]


class Video:

    def __init__(self, data, rate=VIDEO_FPS):
        self.data = data
        total, self.fps, self.shown = video_info(data)
        self.indices = video_indices(total, self.fps, rate)
        self.size = video_size(len(self.indices), *self.shown)
        self.times = pair_times(self.indices, self.fps)

    @property
    def grid(self):
        return self.size[0] // PATCH, self.size[1] // PATCH

    def frames(self):
        from PIL import Image
        out, n = np.empty((len(self.indices), self.size[0], self.size[1], 3), np.uint8), 0
        for frame in video_frames(self.data, set(self.indices.tolist())):
            img = Image.fromarray(frame)
            out[n] = np.asarray(img if img.size == (self.size[1], self.size[0]) else img.resize((self.size[1], self.size[0]), Image.Resampling.BICUBIC))
            n += 1
        if n != len(self.indices):
            raise VideoError(f"the video decoded to {n:g} of the {len(self.indices)} frames Chrysalis took")
        return out


def pair_patches(frames):
    if len(frames) % 2:
        frames = np.concatenate([frames, frames[-1:]])
    x = frames.astype(np.float32) * np.float32(1 / 255)
    x = (x - np.float32(0.5)) / np.float32(0.5)
    n, h, w = len(frames) // 2, frames.shape[1], frames.shape[2]
    gh, gw = h // PATCH, w // PATCH
    x = x.reshape(n, 2, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH, 3)
    x = x.transpose(0, 2, 5, 3, 6, 8, 1, 4, 7)
    return np.ascontiguousarray(x.reshape(n, gh * gw, 2 * PATCH_DIM))


def positions(segments):
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
