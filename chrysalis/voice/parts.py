import collections, ctypes, gc, os, re, shutil, tarfile, tempfile, threading, urllib.request

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "voice")
PARTS = {"vad": {}, "stt": {}, "turn": {}, "tts": {}}
RATE = 16000


def part(kind, name):
    def add(cls):
        cls.name = name
        PARTS[kind][name] = cls
        return cls
    return add


class Missing(RuntimeError):
    pass


SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


def fetch(cls):
    src, dest = cls.SOURCE, os.path.join(DATA, cls.DIR)
    os.makedirs(DATA, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=DATA, prefix=".fetch.")
    try:
        if isinstance(src, str):
            with urllib.request.urlopen(src) as r, tarfile.open(fileobj=r, mode="r|bz2") as t:
                t.extractall(tmp, filter="data")
            (inner,) = os.listdir(tmp)
            shutil.rmtree(dest, ignore_errors=True)
            os.rename(os.path.join(tmp, inner), dest)
        else:
            os.makedirs(dest, exist_ok=True)
            for url, n in src:
                if not os.path.exists(os.path.join(dest, n)):
                    urllib.request.urlretrieve(url, os.path.join(tmp, n))
                    os.replace(os.path.join(tmp, n), os.path.join(dest, n))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def files(cls):
    paths = [os.path.join(DATA, cls.DIR, n) for n in cls.FILES]
    if not all(os.path.exists(p) for p in paths):
        try:
            fetch(cls)
        except OSError as e:
            raise Missing(f"{cls.label} could not be downloaded: {e}") from e
    return paths


def installed(cls):
    return all(os.path.exists(os.path.join(DATA, cls.DIR, n)) for n in cls.FILES)


def session(path, threads, spin=False):
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = threads, 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.add_session_config_entry("session.intra_op.allow_spinning", "1" if spin else "0")
    o.enable_cpu_mem_arena = False
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


try:
    _malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
except (OSError, AttributeError):
    _malloc_trim = None


def trim():
    if _malloc_trim:
        _malloc_trim(0)


DICTS = ("/usr/share/dict/words", "/usr/share/dict/cracklib-small")
_words = None


def words():
    global _words
    if _words is None:
        path = next((p for p in DICTS if os.path.exists(p)), None)
        with open(path or os.devnull, errors="ignore") as f:
            _words = {w.strip().lower() for w in f}
    return _words


def terms(text):
    known, out = words(), []
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9.'+\-]*[A-Za-z0-9]|[A-Za-z]", text):
        parts = [p for p in tok.strip(".'").split("-") if p]
        for p in parts:
            a = re.match(r"[A-Za-z]*", p).group(0)
            digits, caps = bool(re.search(r"\d", p)), bool(re.search(r"[A-Z]", a[1:])) or (a.isupper() and len(a) >= 2)
            unknown = bool(known) and len(a) >= 3 and a.lower() not in known
            if a and (caps or unknown or (digits and len(a) >= 3) or (a.isupper() and digits)):
                t = p if a.isupper() and digits and len(p) <= 5 else a
                if t not in out:
                    out.append(t)
        run = []
        for p in parts + [""]:
            if re.fullmatch(r"[A-Z][a-z]+", p):
                run.append(p)
                continue
            if len(run) >= 2 and " ".join(run) not in out:
                out.append(" ".join(run))
            run = []
    return out


@part("vad", "silero")
class Silero:
    label, DIR, FILES, FRAME = "Silero VAD v6", "silero", ("silero_vad.onnx",), 512
    SOURCE = [("https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad_16k_op15.onnx", "silero_vad.onnx")]

    def __init__(self, threads):
        self.s = session(files(Silero)[0], 1)
        self.sr = np.array(RATE, np.int64)

    def state(self):
        return [np.zeros((2, 1, 128), np.float32), np.zeros((1, 64), np.float32)]

    def prob(self, st, frame):
        x = np.concatenate([st[1], frame[None]], axis=1)
        p, st[0] = self.s.run(None, {"input": x, "state": st[0], "sr": self.sr})
        st[1] = x[:, -64:]
        return float(p[0, 0])

    def warm(self):
        self.prob(self.state(), np.zeros(self.FRAME, np.float32))


@part("stt", "parakeet-110m")
class Parakeet:
    label, DIR = "Parakeet 110M (English)", "parakeet-110m"
    FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
    SOURCE = f"{SHERPA}/asr-models/sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8.tar.bz2"
    HOT = 1.5
    PAUSE, LONGEST = int(0.3 * RATE), 9 * RATE
    EARLY = int(0.1 * RATE)

    def __init__(self, threads):
        import sherpa_onnx
        enc, dec, join, tokens = files(type(self))
        vocab = os.path.join(os.path.dirname(tokens), "terms.vocab")
        if not os.path.exists(vocab):
            with open(tokens) as f, open(vocab + ".tmp", "w") as out:
                for line in f:
                    tok = line.rsplit(" ", 1)[0]
                    out.write(f"{tok}\t{0.0 if tok == '<unk>' else -1.0}\n")
            os.replace(vocab + ".tmp", vocab)
        self.r = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=enc, decoder=dec, joiner=join, tokens=tokens, num_threads=1, model_type="nemo_transducer",
            decoding_method="modified_beam_search", max_active_paths=4, hotwords_score=self.HOT, modeling_unit="bpe",
            bpe_vocab=vocab)
        self.vad = Silero(1)

    def stream(self, terms=""):
        return ParakeetStream(self, terms)

    def warm(self):
        st = self.stream()
        st.feed(np.zeros(RATE, np.float32))
        st.finish()


class ParakeetStream:
    def __init__(self, m, terms=""):
        self.m, self.vst, self.done, self.piece, self.spare = m, m.vad.state(), [], [], np.zeros(0, np.float32)
        self.voiced, self.quiet, self.n, self.terms = False, 0, 0, terms
        self.early = None
        self.told = ""

    def read(self, pcm):
        s = self.m.r.create_stream(self.terms) if self.terms else self.m.r.create_stream()
        s.accept_waveform(RATE, pcm)
        self.m.r.decode_stream(s)
        return s.result.text.strip()

    def decode(self):
        text = self.early if self.early is not None else self.read(np.concatenate(self.piece)) if self.voiced and self.piece else ""
        if self.voiced and text:
            self.done.append(text)
        self.piece, self.voiced, self.quiet, self.n, self.early = [], False, 0, 0, None

    def feed(self, pcm):
        x = np.concatenate([self.spare, pcm])
        F = Silero.FRAME
        for i in range(0, len(x) - F + 1, F):
            f = x[i:i + F]
            self.piece.append(f)
            self.n += F
            if self.m.vad.prob(self.vst, f) >= 0.5:
                self.voiced, self.quiet, self.early = True, 0, None
            else:
                self.quiet += F
            if self.voiced and self.early is None and self.quiet >= self.m.EARLY:
                self.early = self.read(np.concatenate(self.piece))
            if (self.voiced and self.quiet >= self.m.PAUSE) or self.n >= self.m.LONGEST:
                self.decode()
        self.spare = x[len(x) - len(x) % F:]
        now = " ".join(self.done + ([self.early] if self.early else []))
        if now != self.told:
            self.told = now
            return now

    def finish(self):
        if self.spare.size:
            self.piece.append(self.spare)
        self.decode()
        return " ".join(self.done)

    def peek(self, seconds=None):
        x = np.concatenate(self.piece) if self.voiced and self.piece else np.zeros(0, np.float32)
        if self.early is not None and not seconds:
            now = self.early
        else:
            now = self.read(x[-int(seconds * RATE):] if seconds else x) if x.size else ""
        return " ".join(([] if seconds else self.done) + ([now] if now else []))


def mel_filters(n_mels=80, n_fft=400, sr=RATE):
    def mel(f):
        f = np.asarray(f, np.float64)
        return np.where(f < 1000, 3 * f / 200, 15 + np.log(np.maximum(f, 1e-10) / 1000) * 27 / np.log(6.4))

    def hz(m):
        return np.where(m < 15, 200 * m / 3, 1000 * np.exp(np.log(6.4) * (m - 15) / 27))
    edges = hz(np.linspace(mel(0), mel(sr / 2), n_mels + 2))
    bins = np.linspace(0, sr // 2, n_fft // 2 + 1)
    slope = edges[None, :] - bins[:, None]
    down, up = -slope[:, :-2] / np.diff(edges)[:-1], slope[:, 2:] / np.diff(edges)[1:]
    return (np.maximum(0, np.minimum(down, up)) * (2 / (edges[2:] - edges[:-2]))).T


def whisper_features(x, filters):
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    x = np.pad(x, 200, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    spec = np.log10(np.maximum(filters @ power.T, 1e-10))[:, :-1]
    return ((np.maximum(spec, spec.max() - 8) + 4) / 4).astype(np.float32)


@part("turn", "smart-turn")
class SmartTurn:
    label, DIR, FILES, SPAN = "Smart Turn v3.2", "smart-turn", ("smart-turn-v3.2-cpu.onnx",), 8 * RATE
    SOURCE = [("https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx", "smart-turn-v3.2-cpu.onnx")]

    def __init__(self, threads):
        self.s = session(files(SmartTurn)[0], threads)
        self.filters = mel_filters()

    def prob(self, audio):
        x = audio[-self.SPAN:].astype(np.float64)
        x = np.pad(x, (self.SPAN - len(x), 0))
        return float(self.s.run(None, {"input_features": whisper_features(x, self.filters)[None]})[0].reshape(-1)[0])

    def warm(self):
        self.prob(np.zeros(RATE, np.float32))


@part("tts", "kokoro")
class Kokoro:
    label, DIR, FILES, VOICE = "Kokoro 82M", "kokoro", ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin"), "af_heart"
    SOURCE = [("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx", "kokoro-v1.0.int8.onnx"),
              ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", "voices-v1.0.bin")]
    LANGS = {"a": ("en-US", "en-us"), "b": ("en-GB", "en-gb"), "e": ("es", "es"), "f": ("fr", "fr-fr"), "h": ("hi", "hi"),
             "i": ("it", "it"), "j": ("ja", "ja"), "p": ("pt", "pt-br"), "z": ("zh", "cmn")}

    def __init__(self, threads):
        from kokoro_onnx import Kokoro as Model
        model, voices = files(Kokoro)
        self.k, self.rate = Model.from_session(session(model, threads, spin=False), voices), 24000

    @classmethod
    def voices(cls):
        with np.load(os.path.join(DATA, cls.DIR, cls.FILES[1])) as v:
            ids = sorted(v.files)
        return [{"id": i, "label": i[3:].replace("_", " ").title(), "lang": cls.LANGS.get(i[0], ("", ""))[0],
                 "gender": {"f": "female", "m": "male"}.get(i[1])} for i in ids]

    def synth(self, text, voice, speed):
        lang = self.LANGS.get(voice[:1], ("", "en-us"))[1]
        audio = self.k.create(text, voice=voice, speed=speed, lang=lang)[0].astype(np.float32)
        trim()
        return audio

    def warm(self):
        self.synth("Hello there.", self.VOICE, 1.0)
        self.voice_list = self.voices()


DEFAULTS = {"vad": "silero", "stt": "parakeet-110m", "turn": "smart-turn", "tts": "kokoro"}


class Library:

    def __init__(self, threads):
        self.threads, self.lock, self.loaded, self.users = threads, threading.Lock(), {}, collections.Counter()
        self.loading = collections.defaultdict(threading.Lock)

    def catalog(self):
        def entry(kind, cls):
            e = {"name": cls.name, "label": cls.label, "installed": installed(cls), "default": DEFAULTS[kind] == cls.name}
            if kind == "tts":
                e["voices"] = cls.voices() if e["installed"] else []
            return e
        return {kind: [entry(kind, cls) for cls in PARTS[kind].values()] for kind in PARTS}

    def get(self, kind, name):
        if name not in PARTS[kind]:
            raise Missing(f"there is no {kind} named {name!r} (there are: {', '.join(PARTS[kind])})")
        key = kind, name
        with self.lock:
            self.users[key] += 1
            gate = self.loading[key]
        with gate:
            if key not in self.loaded:
                try:
                    p = self.load(kind, name)
                except Exception:
                    self.release(kind, name)
                    raise
                self.loaded[key] = p
            return self.loaded[key]

    def load(self, kind, name):
        p = PARTS[kind][name](self.threads)
        p.warm()
        trim()
        return p

    def release(self, kind, name):
        with self.lock:
            self.users[kind, name] -= 1
            if self.users[kind, name] <= 0:
                del self.users[kind, name]
                if self.loaded.pop((kind, name), None) is not None:
                    gc.collect()
                    trim()
