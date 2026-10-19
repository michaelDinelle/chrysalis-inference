"""The voice service's parts, each kind behind one small interface and named in PARTS (a session picks them by name):
  vad   FRAME (samples at 16 kHz), state() -> a session's own state, prob(state, frame) -> P(speech)
  stt   stream() -> a stream: feed(pcm) -> the whole text so far when it changed, else None; finish() -> final text;
        peek(seconds=None) -> the text of the last seconds (or all) decoded now, apart from the stream
  turn  prob(audio) -> P(the user's turn is over), given the turn's audio up to now
  tts   rate, voices() -> [{id, label, lang, gender}] (a classmethod: it loads nothing), VOICE (the default),
        synth(text, voice, speed) -> float32 mono at rate
A loaded part is shared by the sessions using it; what a session changes lives in its own state or stream. A part's
models are its FILES in data/voice/<DIR>/, fetched by `tools/voice_setup.sh DIR`."""
import collections, ctypes, gc, os, threading, wave

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def files(cls):
    paths = [os.path.join(DATA, cls.DIR, n) for n in cls.FILES]
    for p in paths:
        if not os.path.exists(p):
            raise Missing(f"{cls.label} is not installed ({p} is missing): run tools/voice_setup.sh {cls.DIR}")
    return paths


def installed(cls):
    return all(os.path.exists(os.path.join(DATA, cls.DIR, n)) for n in cls.FILES)


def session(path, threads, spin=False):
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = threads, 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Pool threads that spin between a run's parallel steps take cores Flash-Next's CPU experts use; speech synthesis
    # spins all the same: it is what the user waits for (Kokoro's "Sure!" 173 ms instead of 198 at 4 threads)
    o.add_session_config_entry("session.intra_op.allow_spinning", "1" if spin else "0")
    # ONNX Runtime's arena keeps the largest activations it ever held (Kokoro: +604 MB after a dozen sentences); without
    # it, and with trim() after a run, Kokoro holds +176 MB, ~5% slower
    o.enable_cpu_mem_arena = False
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


try:
    _malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
except (OSError, AttributeError):
    _malloc_trim = None


def trim():
    """Gives the memory a run freed back to the system (glibc keeps it otherwise)."""
    if _malloc_trim:
        _malloc_trim(0)


@part("vad", "silero")
class Silero:
    """Silero VAD v6, 16 kHz: each call sees 512 new samples after the previous call's last 64."""
    label, DIR, FILES, FRAME = "Silero VAD v6", "silero", ("silero_vad.onnx",), 512

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


class Streaming:
    """A sherpa-onnx streaming transducer: it decodes as the audio comes in, so a turn's end leaves one chunk to flush."""

    def __init__(self, threads):
        import sherpa_onnx
        enc, dec, join, tokens = files(type(self))
        self.r = sherpa_onnx.OnlineRecognizer.from_transducer(encoder=enc, decoder=dec, joiner=join, tokens=tokens,
                                                              num_threads=1)

    def stream(self):
        return OnlineStream(self.r)


class OnlineStream:
    # The last chunk decodes only with silence after it, and as much as the encoder takes in: kroko's 1.28 s chunks plus
    # 0.13 s of lookahead (T = 141 frames). sherpa's examples pad 0.66 s, which kept a turn's last word in 5 of 40 chunk
    # alignments; 1.2 s in 38, 1.5 s in all 40
    TAIL = np.zeros(int(1.5 * RATE), np.float32)

    def __init__(self, r):
        self.r, self.s, self.text, self.audio = r, r.create_stream(), "", []

    def decode(self):
        while self.r.is_ready(self.s):
            self.r.decode_stream(self.s)
        return self.r.get_result(self.s).strip()

    def feed(self, pcm):
        self.s.accept_waveform(RATE, pcm)
        self.audio.append(pcm)
        text = self.decode()
        if text != self.text:
            self.text = text
            return text

    def finish(self):
        self.s.accept_waveform(RATE, self.TAIL)
        self.s.input_finished()
        return self.decode()

    def peek(self, seconds=None):
        """A chunk's words come out of the stream only once the chunk is whole (kroko: every 1.28 s): the audio decoded
        again in a stream of its own, flushed at once, has them now."""
        x = np.concatenate(self.audio) if self.audio else np.zeros(0, np.float32)
        s = self.r.create_stream()
        s.accept_waveform(RATE, x[-int(seconds * RATE):] if seconds else x)
        s.accept_waveform(RATE, self.TAIL)
        s.input_finished()
        while self.r.is_ready(s):
            self.r.decode_stream(s)
        return self.r.get_result(s).strip()


@part("stt", "kroko")
class Kroko(Streaming):
    label, DIR, FILES = "Kroko streaming (English)", "kroko", ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")


@part("stt", "nemotron")
class Nemotron(Streaming):
    label, DIR = "Nemotron streaming 0.6B (English)", "nemotron"
    FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


class Offline:
    """An offline sherpa-onnx model fed as a stream: Silero splits the speech at pauses and each finished piece is
    decoded while the user goes on, so a turn's end leaves only its last piece; the pieces so far are the partial."""
    PAUSE, LONGEST = int(0.3 * RATE), 9 * RATE        # Moonshine returns nothing past ~15 s

    def __init__(self, threads):
        self.r, self.vad = self.load(files(type(self))), Silero(1)

    def stream(self):
        return OfflineStream(self)


class OfflineStream:
    def __init__(self, m):
        self.m, self.vst, self.done, self.piece, self.spare = m, m.vad.state(), [], [], np.zeros(0, np.float32)
        self.voiced, self.quiet, self.n = False, 0, 0

    def text(self, pcm):
        s = self.m.r.create_stream()
        s.accept_waveform(RATE, pcm)
        self.m.r.decode_stream(s)
        return s.result.text.strip()

    def decode(self):
        if self.voiced and self.piece and (text := self.text(np.concatenate(self.piece))):
            self.done.append(text)
        self.piece, self.voiced, self.quiet, self.n = [], False, 0, 0

    def feed(self, pcm):
        before, x = len(self.done), np.concatenate([self.spare, pcm])
        F = Silero.FRAME
        for i in range(0, len(x) - F + 1, F):
            f = x[i:i + F]
            self.piece.append(f)
            self.n += F
            if self.m.vad.prob(self.vst, f) >= 0.5:
                self.voiced, self.quiet = True, 0
            else:
                self.quiet += F
            if (self.voiced and self.quiet >= self.m.PAUSE) or self.n >= self.m.LONGEST:
                self.decode()
        self.spare = x[len(x) - len(x) % F:]
        if len(self.done) > before:
            return " ".join(self.done)

    def finish(self):
        if self.spare.size:
            self.piece.append(self.spare)
        self.decode()
        return " ".join(self.done)

    def peek(self, seconds=None):
        """The piece under way decoded now (its words otherwise wait for the next pause), after the pieces so far."""
        x = np.concatenate(self.piece) if self.voiced and self.piece else np.zeros(0, np.float32)
        now = self.text(x[-int(seconds * RATE):] if seconds else x) if x.size else ""
        return " ".join(([] if seconds else self.done) + ([now] if now else []))


@part("stt", "parakeet-110m")
class Parakeet110m(Offline):
    label, DIR = "Parakeet 110M (English)", "parakeet-110m"
    FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")

    def load(self, paths):
        import sherpa_onnx
        enc, dec, join, tokens = paths
        return sherpa_onnx.OfflineRecognizer.from_transducer(encoder=enc, decoder=dec, joiner=join, tokens=tokens,
                                                             num_threads=1, model_type="nemo_transducer")


@part("stt", "parakeet-v3")
class ParakeetV3(Parakeet110m):
    label, DIR = "Parakeet v3 0.6B (25 languages)", "parakeet-v3"


@part("stt", "moonshine-base")
class MoonshineBase(Offline):
    label, DIR, FILES = "Moonshine base (English)", "moonshine-base", ("encoder_model.ort", "decoder_model_merged.ort", "tokens.txt")

    def load(self, paths):
        import sherpa_onnx
        enc, dec, tokens = paths
        return sherpa_onnx.OfflineRecognizer.from_moonshine_v2(encoder=enc, decoder=dec, tokens=tokens, num_threads=1)


def mel_filters(n_mels=80, n_fft=400, sr=RATE):
    """Slaney-scale triangles with Slaney's area normalization: Whisper's (librosa's) filter bank."""
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
    """WhisperFeatureExtractor's log-mel of x, normalized to zero mean and unit variance first (do_normalize)."""
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    x = np.pad(x, 200, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)          # periodic Hann
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    spec = np.log10(np.maximum(filters @ power.T, 1e-10))[:, :-1]
    return ((np.maximum(spec, spec.max() - 8) + 4) / 4).astype(np.float32)


@part("turn", "smart-turn")
class SmartTurn:
    """Smart Turn v3.2: the last 8 s of the turn, zero-padded at the start, as Whisper log-mel features."""
    label, DIR, FILES, SPAN = "Smart Turn v3.2", "smart-turn", ("smart-turn-v3.2-cpu.onnx",), 8 * RATE

    def __init__(self, threads):
        self.s = session(files(SmartTurn)[0], threads)
        self.filters = mel_filters()

    def prob(self, audio):
        x = audio[-self.SPAN:].astype(np.float64)
        x = np.pad(x, (self.SPAN - len(x), 0))
        return float(self.s.run(None, {"input_features": whisper_features(x, self.filters)[None]})[0].reshape(-1)[0])


@part("tts", "kokoro")
class Kokoro:
    label, DIR, FILES, VOICE = "Kokoro 82M", "kokoro", ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin"), "af_heart"
    # a voice id's first letter is its language and its second the voice's gender (af_heart: American English, female)
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


class SherpaTts:
    """A sherpa-onnx OfflineTts model; its voices are speaker numbers, SPEAKERS of them."""
    VOICE, LANG = "0", "en-US"

    def __init__(self, threads):
        import sherpa_onnx
        self.so = sherpa_onnx
        cfg = sherpa_onnx.OfflineTtsModelConfig(num_threads=threads, provider="cpu")
        self.model(cfg, files(type(self)))
        self.t = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=cfg, max_num_sentences=1))
        self.rate = self.t.sample_rate

    @classmethod
    def voices(cls):
        return [{"id": str(i), "label": f"Voice {i + 1}", "lang": cls.LANG, "gender": None} for i in range(cls.SPEAKERS)]

    def config(self, g, voice):
        g.sid = int(voice)

    def synth(self, text, voice, speed):
        g = self.so.GenerationConfig()
        g.speed = speed
        self.config(g, voice)
        audio = np.asarray(self.t.generate(text, g).samples, np.float32)
        trim()
        return audio


@part("tts", "kitten")
class Kitten(SherpaTts):
    label, DIR, FILES, SPEAKERS = "Kitten micro (English)", "kitten", ("model.onnx", "voices.bin", "tokens.txt"), 8

    def model(self, cfg, f):
        cfg.kitten = self.so.OfflineTtsKittenModelConfig(model=f[0], voices=f[1], tokens=f[2],
                                                         data_dir=os.path.join(DATA, self.DIR, "espeak-ng-data"))


@part("tts", "supertonic")
class Supertonic(SherpaTts):
    label, DIR, SPEAKERS, VOICE, LANG, STEPS = "Supertonic 3 (31 languages)", "supertonic", 10, "6", "en", 5
    FILES = ("duration_predictor.int8.onnx", "text_encoder.int8.onnx", "vector_estimator.int8.onnx", "vocoder.int8.onnx",
             "tts.json", "unicode_indexer.bin", "voice.bin")

    def model(self, cfg, f):
        cfg.supertonic = self.so.OfflineTtsSupertonicModelConfig(
            duration_predictor=f[0], text_encoder=f[1], vector_estimator=f[2], vocoder=f[3], tts_json=f[4],
            unicode_indexer=f[5], voice_style=f[6])

    def config(self, g, voice):
        g.sid, g.num_steps = int(voice), self.STEPS
        g.extra["lang"] = self.LANG


@part("tts", "pocket")
class Pocket(SherpaTts):
    """Pocket TTS clones the voice of a reference recording: its voices are the WAV files in its test_wavs."""
    label, DIR, VOICE, STEPS = "Pocket TTS (voice cloning)", "pocket", "bria", 5
    FILES = ("lm_flow.int8.onnx", "lm_main.int8.onnx", "encoder.onnx", "decoder.int8.onnx", "text_conditioner.onnx",
             "vocab.json", "token_scores.json")

    def __init__(self, threads):
        self.refs = {}
        super().__init__(threads)

    @classmethod
    def voices(cls):
        d = os.path.join(DATA, cls.DIR, "test_wavs")
        return [{"id": f[:-4], "label": f[:-4].replace("_", " ").capitalize(), "lang": None, "gender": None}
                for f in sorted(os.listdir(d)) if f.endswith(".wav")]

    def model(self, cfg, f):
        cfg.pocket = self.so.OfflineTtsPocketModelConfig(lm_flow=f[0], lm_main=f[1], encoder=f[2], decoder=f[3],
                                                         text_conditioner=f[4], vocab_json=f[5], token_scores_json=f[6])

    def config(self, g, voice):
        if voice not in self.refs:
            with wave.open(os.path.join(DATA, self.DIR, "test_wavs", voice + ".wav")) as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels())[:, 0]
                self.refs[voice] = (pcm.astype(np.float32) / 32768, w.getframerate())
        g.reference_audio, g.reference_sample_rate = self.refs[voice]
        g.num_steps = self.STEPS


DEFAULTS = {"vad": "silero", "stt": "kroko", "turn": "smart-turn", "tts": "kokoro"}


class Library:
    """The parts sessions use, loaded by name the first time one asks and each warmed up with one run (an ONNX
    session's first run is the slow one); a part no session holds any more is dropped, so a switch frees what it
    leaves. The VAD and the STT keep up with the microphone on a thread each, beside speech synthesis; the turn model
    and synthesis take `threads` each, as one runs while the other waits (a turn's end, then the reply)."""

    def __init__(self, threads):
        self.threads, self.lock, self.loaded, self.users = threads, threading.Lock(), {}, collections.Counter()
        self.loading = collections.defaultdict(threading.Lock)     # one per part: different parts load side by side

    def catalog(self):
        def entry(kind, cls):
            e = {"name": cls.name, "label": cls.label, "installed": installed(cls)}
            if kind == "tts":
                e["voices"] = cls.voices() if e["installed"] else []
            return e
        return {kind: [entry(kind, cls) for cls in PARTS[kind].values()] for kind in PARTS}

    def get(self, kind, name):
        """The part, loaded if it is not yet (call it off the event loop); held until release()."""
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
        p = PARTS[kind][name]({"vad": 1, "stt": 1}.get(kind, self.threads))
        if kind == "vad":
            p.prob(p.state(), np.zeros(p.FRAME, np.float32))
        elif kind == "stt":
            st = p.stream()
            st.feed(np.zeros(RATE, np.float32))
            st.finish()
        elif kind == "turn":
            p.prob(np.zeros(RATE, np.float32))
        else:
            p.synth("Hello there.", p.VOICE, 1.0)
            p.voice_list = type(p).voices()
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
