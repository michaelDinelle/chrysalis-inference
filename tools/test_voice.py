"""The voice service (voice/) on the CPU: its Whisper features against transformers' (data/ref-venv), Smart Turn
through them on a finished and an unfinished question, and sessions over its WebSocket. A spoken question in: speech
on, partials, then its turn and transcript. A markdown reply in: audio pieces in order whose texts carry no markup or
code and whose ranges cover the reply, a short first one, and speech the STT hears as the reply. While a reply plays,
its own echo in the mic: nothing; another voice over it: an interrupt and that voice's turn. A preview, a reply with no
text, a switch to another speech model mid-session (and to one that does not exist). Kokoro voices the questions
(another voice than the replies'); the STT hears the replies.
  data/voice-venv/bin/python tools/test_voice.py       (make test_voice)"""
import json, os, re, struct, subprocess, sys, tempfile, threading, time, wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from websockets.sync.client import connect
from voice.parts import Kokoro, Kroko, SmartTurn, mel_filters, whisper_features
from voice.talk import speakable

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{name:52s} {'ok' if ok else 'FAIL'}  {detail}")


def wer(ref, hyp):
    r, h = (re.sub(r"[^\w' ]", " ", s.lower()).split() for s in (ref, hyp))
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return d[len(h)] / max(1, len(r))


def to16k(a, rate):
    return np.interp(np.arange(0, len(a) / rate, 1 / 16000), np.arange(len(a)) / rate, a).astype(np.float32)


tts, stt = Kokoro(4), Kroko(2)
say = lambda text, voice="am_michael": to16k(tts.synth(text, voice, 1.0), tts.rate)


def hear(x):
    s = stt.stream()
    s.feed(x)
    return s.finish()


with wave.open(os.path.join(ROOT, "data/voice/kroko/test_wavs/0.wav")) as w:
    real = to16k(np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768, w.getframerate())
x = np.pad(real[-128000:].astype(np.float64), (128000 - min(len(real), 128000), 0))
mine = whisper_features(x, mel_filters())
with tempfile.TemporaryDirectory() as d:
    np.save(f"{d}/x.npy", x.astype(np.float32))
    subprocess.run([os.path.join(ROOT, "data/ref-venv/bin/python"), "-c", f"""
import numpy as np
from transformers import WhisperFeatureExtractor
x = np.load("{d}/x.npy")
f = WhisperFeatureExtractor(chunk_length=8)(x, sampling_rate=16000, return_tensors="np", padding="max_length",
                                            max_length=128000, truncation=True, do_normalize=True).input_features[0]
np.save("{d}/ref.npy", f)"""], check=True, capture_output=True)
    ref = np.load(f"{d}/ref.npy")
diff = float(np.abs(ref - mine).max()) if ref.shape == mine.shape else float("inf")
check("Whisper log-mel features match transformers'", diff < 1e-3, f"max |diff| {diff:.2g}, shape {mine.shape}")

turn = SmartTurn(4)
gap = np.zeros(3200, np.float32)
done = turn.prob(np.concatenate([gap, say("Can you tell me what the weather will be like in Denver tomorrow?"), gap]))
cut = turn.prob(np.concatenate([gap, say("So I was wondering whether you could maybe help me with the"), gap]))
check("Smart Turn: a finished question ends the turn", done > 0.5, f"P {done:.3f}")
check("Smart Turn: a sentence cut off mid-clause does not", cut < 0.5, f"P {cut:.3f}")

# A turn's stream starts with its pre-roll, wherever that falls among the STT's chunks, and is finished ~0.3 s after
# the speech: its last word must come out whatever the alignment
last = say("Name a fun weekend activity for two.")
kept = 0
for k in range(24):
    s = stt.stream()
    s.feed(np.concatenate([np.zeros(int((0.4 + k * 0.053) * 16000), np.float32), last, np.zeros(4800, np.float32)]))
    kept += s.finish().lower().rstrip(".").endswith("two")
check("a turn's last word, at any chunk alignment", kept == 24, f"{kept}/24")

svc = subprocess.Popen([sys.executable, "-m", "voice"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
url = svc.stdout.readline().decode().split()[-1]
ws = connect(url, compression=None, max_size=None)


sending = threading.Thread()


def stream(x, tail=1.5, live=False):
    """A recording as the page sends it: 20 ms pieces of int16, after and before silence. live: as it is spoken, from
    a thread of its own, so events() times what arrives meanwhile; returns when x starts, as spoken."""
    global sending
    sending.join() if sending.is_alive() else None
    pcm = (np.clip(np.concatenate([np.zeros(4800, np.float32), x, np.zeros(int(tail * 16000), np.float32)]), -1, 1) * 32767).astype("<i2")
    t = time.monotonic()

    def send():
        for k, i in enumerate(range(0, len(pcm), 320)):
            ws.send(pcm[i:i + 320].tobytes())
            if live:
                time.sleep(max(0.0, t + (k + 1) * 0.02 - time.monotonic()))
    sending = threading.Thread(target=send)
    sending.start()
    if not live:
        sending.join()
    return t + 0.3

def events(until, timeout=15.0):
    """Messages until one passes `until` or the time is up: dicts, and audio as (header, float32 samples). Each one's
    arrival time goes in `arrived`."""
    out, end = [], time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            m = ws.recv(timeout=max(0.01, end - time.monotonic()))
        except TimeoutError:
            break
        if isinstance(m, bytes):
            n = struct.unpack("<I", m[:4])[0]
            m = (json.loads(m[4:4 + n]), np.frombuffer(m[4 + n:], "<i2").astype(np.float32) / 32768)
        else:
            m = json.loads(m)
        out.append(m)
        arrived[id(m)] = time.monotonic()
        if until(m):
            break
    return out


arrived = {}


def kind(m):
    return "audio" if isinstance(m, tuple) else m["type"]


ws.send(json.dumps({"type": "hello"}))
ready = events(lambda m: kind(m) in ("ready", "error"), 120)[-1]
heart = next((v for v in ready.get("voices", []) if v["id"] == "af_heart"), None)
check("hello: ready with the default stack", ready.get("stack") == {"vad": "silero", "stt": "kroko", "turn": "smart-turn", "tts": "kokoro"}
      and ready.get("voice") == "af_heart" and ready.get("sample_rate") == 24000, f"{ready.get('stack')}, {ready.get('load_ms')} ms")
check("its catalog: every part, installed or not, with voices", heart == {"id": "af_heart", "label": "Heart", "lang": "en-US", "gender": "female"}
      and {e["name"]: e["installed"] for e in ready.get("catalog", {}).get("tts", [])}.get("kokoro") is True
      and {"kroko", "parakeet-110m"} <= {e["name"] for e in ready.get("catalog", {}).get("stt", [])}, str(heart))

question = "What's the weather going to be like in Denver tomorrow?"
spoke = stream(say(question), live=True)
got = events(lambda m: kind(m) == "turn")
kinds = [kind(m) for m in got]
t = next((m for m in got if kind(m) == "turn"), {})
check("a question: speech on, partials, then its turn", kinds[:1] == ["speech"] and "partial" in kinds and kinds[-1] == "turn",
      " ".join(dict.fromkeys(kinds)))
check("the turn's transcript is the question", wer(question, t.get("text", "")) <= 0.15, repr(t.get("text")))
first = next((arrived[id(m)] - spoke for m in got if kind(m) == "partial"), float("inf"))
check("its first words show while it is spoken", first < 0.9, f"{first:.2f} s after the speech starts")
check("its wait is the silence, turn model and flush", 200 <= t.get("wait_ms", -1) < 500, f"wait_ms {t.get('wait_ms')}")

reply = ("Sure! Here's the **short** answer: it's about *eighteen degrees* in [Denver](https://example.com/w) right now.\n\n"
         "```python\nprint_banana_value(42)\n```\n\n| Day | High |\n|---|---|\n| Mango | 21 |\n\n## Tomorrow\n- Warmer than today, so a light jacket is enough.\n"
         "- Check https://weather.example.com before you go.\n")
t0 = time.monotonic()
for i in range(0, len(reply), 5):
    ws.send(json.dumps({"type": "say", "id": "r1", "text": reply[i:i + 5]}))
ws.send(json.dumps({"type": "say_end", "id": "r1"}))
got = events(lambda m: kind(m) == "said", 30)
first = next((arrived[id(m)] - t0 for m in got if kind(m) == "audio"), None)
pieces = [m[0] for m in got if kind(m) == "audio"]
texts = [p["text"] for p in pieces]
check("a reply: audio pieces in order, then said", [p["seq"] for p in pieces] == list(range(len(pieces))) and len(pieces) >= 3
      and kind(got[-1]) == "said", f"{len(pieces)} pieces, first after {first and round(first * 1000)} ms")
check("no markup, code or table is read out", not any(re.search(r"[*#`\[\]()|]|https?:|banana|print_|mango", s, re.I) for s in texts) and "a link" in " ".join(texts),
      " | ".join(texts))
check("the pieces' ranges cover the reply", [p["from"] for p in pieces[1:]] == [p["to"] for p in pieces[:-1]]
      and pieces and pieces[0]["from"] == 0 and pieces[-1]["to"] >= len(reply.rstrip()), str([(p["from"], p["to"]) for p in pieces]))
check("the first piece is short", pieces and len(pieces[0]["text"].split()) <= 3, repr(pieces and pieces[0]["text"]))
spoken = np.concatenate([to16k(m[1], m[0]["rate"]) for m in got if kind(m) == "audio"])
want = speakable(reply.replace("```python\nprint_banana_value(42)\n```", ""))
heard = hear(spoken)
check("its speech is heard as the reply", wer(want, heard) <= 0.15, f"WER {wer(want, heard):.2f}: {heard!r}")

ws.send(json.dumps({"type": "playback", "playing": True}))
stream(spoken, tail=1.0)
echo = events(lambda m: kind(m) in ("interrupt", "turn", "partial"), 6)
check("the reply's own echo while it plays: nothing", not echo, str([m for m in echo][:3]))

ws.send(json.dumps({"type": "say", "id": "r2", "text": "Let me tell you about the long history of bridges in Europe, from the "
                                                         "Romans to the present day, one century at a time. "}))
ws.send(json.dumps({"type": "say_end", "id": "r2"}))
events(lambda m: kind(m) == "said", 30)
over = "Actually, could you tell me about penguins instead?"
voice, echo = say(over, "bf_emma"), spoken[16000:]
mix = np.zeros(19200 + len(voice), np.float32)       # the reply through the speakers, a voice over it mid-phrase
mix[:min(len(mix), len(echo))] += 0.5 * echo[:len(mix)]
mix[19200:] += voice
over_at = stream(mix, tail=2.0, live=True) + 1.2    # past the longest silence, should the turn model say "not yet"
got = events(lambda m: kind(m) == "turn", 15)
kinds = [kind(m) for m in got]
t = next((m for m in got if kind(m) == "turn"), {})
check("another voice over the reply interrupts it", "interrupt" in kinds and kinds.index("interrupt") < len(kinds) - 1,
      " ".join(dict.fromkeys(kinds)))
stopped = next((arrived[id(m)] - over_at for m in got if kind(m) == "interrupt"), float("inf"))
check("it interrupts within a second of the voice", stopped < 1.0, f"{stopped:.2f} s after the voice starts")
check("and that voice's words are the turn", "penguins" in t.get("text", "").lower() and wer(over, t.get("text", "")) <= 0.3,
      repr(t.get("text")))

ws.send(json.dumps({"type": "playback", "playing": False}))
ws.send(json.dumps({"type": "say_end", "id": "r3"}))
got = events(lambda m: kind(m) == "said", 10)
check("a reply with no text is said all the same", got and got[-1] == {"type": "said", "id": "r3"}, str(got[-1:]))
ws.send(json.dumps({"type": "preview", "voice": "bf_emma"}))
got = events(lambda m: kind(m) == "said", 15)
sample = [m for m in got if kind(m) == "audio"]
heard = hear(np.concatenate([to16k(m[1], m[0]["rate"]) for m in sample])) if sample else ""
check("a preview: the voice's sample, as audio of its own", sample and sample[0][0]["id"] == "preview" and got[-1] == {"type": "said", "id": "preview"}
      and wer("Hi, I'm Emma. This is how I sound.", heard) <= 0.2, repr(heard))
ws.send(json.dumps({"type": "config", "stack": {"tts": "kitten"}}))
got = events(lambda m: kind(m) in ("ready", "error"), 60)[-1]
check("config switches the speech mid-session", got.get("stack", {}).get("tts") == "kitten" and got.get("voice") == "0"
      and len(got.get("voices", [])) == 8, f"{got.get('stack')}, voice {got.get('voice')}")
ws.send(json.dumps({"type": "config", "stack": {"tts": "no-such-tts"}}))
got = events(lambda m: kind(m) == "ready", 10)
err = next((m for m in got if kind(m) == "error"), {})
check("an unknown part is an error, and the session keeps its own", "no-such-tts" in err.get("message", "")
      and got[-1].get("stack", {}).get("tts") == "kitten", err.get("message", ""))
ws.send(json.dumps({"type": "say", "id": "r4", "text": "This one comes from another model. "}))
ws.send(json.dumps({"type": "say_end", "id": "r4"}))
got = [m for m in events(lambda m: kind(m) == "said", 30) if kind(m) == "audio"]
heard = hear(np.concatenate([to16k(m[1], m[0]["rate"]) for m in got])) if got else ""
check("and speaks with it", wer("This one comes from another model.", heard) <= 0.2, repr(heard))

ws.close()
with open(f"/proc/{svc.pid}/status") as f:
    mem = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in f if l.startswith(("VmRSS", "VmHWM"))}
# Speech that comes before a session has its parts (the user talks as soon as voice mode opens, while a first
# start's models load) is kept and heard once they are in
ws = connect(url, compression=None, max_size=None)
early = "Tell me a joke about penguins."
stream(say(early), tail=1.5)
ws.send(json.dumps({"type": "hello"}))
t = next((m for m in events(lambda m: kind(m) == "turn", 20) if kind(m) == "turn"), {})
check("what is said before the parts load is heard", wer(early, t.get("text", "")) <= 0.2, repr(t.get("text")))

# Headphones (or the browser's echo cancellation): a reply plays and the mic stays quiet, so speech over it is the
# user's and needs no words to stop the reply. A session of its own: the one above has heard its echo.
ws = connect(url, compression=None, max_size=None)
ws.send(json.dumps({"type": "hello"}))
events(lambda m: kind(m) == "ready", 60)
ws.send(json.dumps({"type": "say", "id": "q1", "text": "Here is a long answer about the history of tea, from China to Britain and back. "}))
ws.send(json.dumps({"type": "playback", "playing": True}))
quiet_at = stream(np.zeros(int(2.5 * 16000), np.float32), tail=0.0, live=True)
over_at = stream(say("Wait, what about coffee?", "bf_emma"), tail=1.0, live=True)
got = events(lambda m: kind(m) == "interrupt", 10)
stopped = next((arrived[id(m)] - over_at for m in got if kind(m) == "interrupt"), float("inf"))
check("a quiet mic: talking over the reply stops it at once", stopped < 0.5, f"{stopped:.2f} s after the voice starts")
ws.send(json.dumps({"type": "playback", "playing": False}))
t = next((m for m in events(lambda m: kind(m) == "turn", 10) if kind(m) == "turn"), {})
check("and the words are the turn", "coffee" in t.get("text", "").lower(), repr(t.get("text")))

svc.stdin.close()
try:
    svc.wait(10)
except subprocess.TimeoutExpired:
    svc.kill()
check("the service exits when its stdin ends", svc.returncode == 0, f"exit {svc.returncode}")
print(f"service: loaded in {ready.get('load_ms')} ms, RSS {mem.get('VmRSS')} MB (peak {mem.get('VmHWM')} MB); "
      f"first audio {first and round(first * 1000)} ms after the reply's first text")
print("all ok" if not fails else f"{fails} FAILURES")
sys.exit(fails != 0)
