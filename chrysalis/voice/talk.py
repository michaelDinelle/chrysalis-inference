import asyncio, collections, json, re, struct, sys, time, unicodedata

import numpy as np

from .parts import DEFAULTS, PARTS, RATE, Missing, fetch, terms, trim

PRE_ROLL = int(0.4 * RATE)
START_FRAMES = 3
PAUSED = int(0.2 * RATE)
ECHO_TAIL = 0.3
ECHO_KEEP = 10.0
VAD_ON, TURN_DONE = 0.5, 0.5
SILENCE = int(0.2 * RATE)
LONGEST_SILENCE = int(1.5 * RATE)
PEEK = int(0.3 * RATE)
WINDOW = 3.0
CLEAN = 2 * RATE
BARGE = 8
EARLY = 750
STOP_WORDS = {"stop", "wait", "hold", "pause", "hang", "sorry", "excuse", "hey", "quiet", "shush", "enough"}


def norm(token):
    return re.sub(r"[^\w']", "", token.lower())


def words(text):
    return {w for w in map(norm, text.split()) if w}


def near(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return a[i + 1:] == b[i + 1:] or a[i + 1:] == b[i:] or a[i:] == b[i + 1:]


class Listener:

    def __init__(self, talk):
        self.talk, self.played = talk, 0

    def use(self, vad, stt, turn):
        self.vad, self.stt, self.turn = vad, stt, turn
        self.vst, self.spare, F = vad.state(), np.zeros(0, np.float32), vad.FRAME
        self.pre, self.keep = PRE_ROLL // F + 1, 8 * RATE // F + 1
        self.reset()

    def reset(self):
        self.before, self.stream, self.run, self.talking = collections.deque(maxlen=self.pre), None, 0, False

    def feed(self, pcm):
        out, x, F = [], np.concatenate([self.spare, pcm]), self.vad.FRAME
        for i in range(0, len(x) - F + 1, F):
            self.frame(x[i:i + F], out)
        self.spare = x[len(x) - len(x) % F:]
        return out

    def frame(self, f, out):
        p = self.vad.prob(self.vst, f)
        voiced = p >= (VAD_ON - 0.15 if self.talking else VAD_ON)
        if self.stream is None:
            if self.talk.playing:
                self.played += len(f)
            self.before.append(f)
            self.run = self.run + 1 if voiced else 0
            if self.run >= START_FRAMES:
                self.start(out)
            return
        self.audio.append(f)
        self.since += len(f)
        if self.barge is not None:
            self.barge = self.barge + 1 if voiced else 0
            if self.barge >= BARGE:
                self.barge, heard = None, self.stream.peek().split()
                if heard and not self.novel(heard):
                    self.carries()
                    self.own = None
                else:
                    self.interrupt(out, "the mic clear of the reply: its voiced frames")
        if (text := self.stream.feed(f)) is not None:
            self.heard(text, out, settled=True)
        elif self.since >= PEEK and voiced:
            self.look(out)
        if voiced:
            self.last, self.quiet, self.checked = time.monotonic(), 0, False
            if not self.talking:
                self.talking = True
                self.emit(out, {"type": "speech", "on": True})
            return
        self.quiet += len(f)
        if self.talking and self.quiet >= PAUSED:
            self.talking = False
            self.emit(out, {"type": "speech", "on": False})
        if self.quiet >= SILENCE and not self.checked:
            self.checked = True
            if self.turn.prob(np.concatenate(self.audio)) >= TURN_DONE:
                return self.end(out)
            self.pause(out)
        if self.quiet >= LONGEST_SILENCE:
            self.end(out)

    def start(self, out):
        t = self.talk
        over = t.playing or time.monotonic() < t.echo_until
        clean = over and self.played >= CLEAN
        self.own = None if over and not clean else 0
        self.broke, self.since, self.shown, self.barge = False, 0, [], 0 if clean else None
        self.stream, self.audio = self.stt.stream(t.terms), collections.deque(self.before, maxlen=self.keep)
        self.last, self.quiet, self.checked, self.talking = time.monotonic(), 0, False, True
        self.emit(out, {"type": "speech", "on": True})
        if (text := self.stream.feed(np.concatenate(self.before))) is not None:
            self.heard(text, out, settled=True)

    def emit(self, out, ev):
        if self.own is not None:
            out.append(ev)

    def novel(self, tokens):
        echo = self.talk.echo_words()
        long = [e for e in echo if len(e) >= 4]
        return [i for i, w in enumerate(map(norm, tokens))
                if w and w not in echo and not (len(w) >= 4 and any(near(w, e) for e in long))]

    def carries(self):
        if self.played >= CLEAN:
            print("voice: the mic carries the reply: talking over it takes words, until 2 s of a reply play unheard",
                  file=sys.stderr, flush=True)
        self.played = 0

    def interrupt(self, out, how):
        if not self.broke:
            self.broke = True
            out.append({"type": "interrupt"})
            print(f"voice: talked over the reply ({how})", file=sys.stderr, flush=True)

    def look(self, out):
        self.since = 0
        if self.own is None:
            recent = self.stream.peek(WINDOW).split()[1:-1]
            new = self.novel(recent)
            if len(new) >= 2 or any(norm(recent[i]) in STOP_WORDS for i in new):
                self.heard(self.stream.peek(), out)
            return
        self.heard(self.stream.peek(), out)

    def heard(self, text, out, settled=False):
        tokens = text.split()
        if self.own is None:
            new = self.novel(tokens)
            if not (len(new) >= 2 or any(norm(tokens[i]) in STOP_WORDS for i in new)):
                return
            self.own = new[0]
            self.interrupt(out, "over its echo: words not the reply's")
            if self.talking:
                out.append({"type": "speech", "on": True})
        elif self.talk.awaiting:
            self.interrupt(out, "before it was heard")
        mine = tokens[self.own:]
        if mine and not (len(mine) < len(self.shown) and [norm(w) for w in mine] == [norm(w) for w in self.shown[:len(mine)]]):
            if mine != self.shown or settled:
                self.shown = mine
                out.append({"type": "partial", "text": " ".join(mine), **({"settled": True} if settled else {})})

    def pause(self, out):
        if self.own is None:
            return
        text = " ".join(self.stream.peek().split()[self.own:])
        if any(c.isalnum() for c in text):
            out.append({"type": "pause", "text": text})

    def end(self, out):
        tokens, wait, echo = self.stream.finish().split(), time.monotonic() - self.last, self.own is None
        if self.talking and not echo:
            out.append({"type": "speech", "on": False})
        if echo:
            own = [] if self.talk.playing else [i for i in self.novel(tokens) if len(norm(tokens[i])) >= 3]
            self.own = own[0] if own else len(tokens)
            if not own and any(w for w in map(norm, tokens)):
                self.carries()
        mine = " ".join(tokens[self.own:])
        if not echo and not any(c.isalnum() for c in mine):
            peak = float(np.abs(np.concatenate(self.audio)).max()) if self.audio else 0.0
            print(f"voice: speech with no words in it, peak {20 * np.log10(max(peak, 1e-6)):.0f} dBFS", file=sys.stderr, flush=True)
        self.reset()
        if any(c.isalnum() for c in mine):
            out.append({"type": "turn", "text": mine, "wait_ms": round(wait * 1000)})


ABBREV = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "jr", "sr", "prof", "inc", "ltd", "fig", "approx", "no"}
ENDS = re.compile(r"[.!?]+[\"')\]]*(?=\s)|\n")
CLAUSE = re.compile(r"[,;:][\"')\]]*(?=\s)|\s[—–-]{1,2}(?=\s)")
FENCE = re.compile(r"(?m)^[ \t]*```")
WORD = re.compile(r"[^\W_][\w'’]*")


RAMP = [(1, 3, 8), (4, 8, 16), (6, 20, 40)]
PHRASE = {"is", "are", "was", "were", "will", "would", "can", "could", "should", "has", "have", "had", "of", "to", "in",
          "on", "at", "for", "from", "with", "by", "about", "into", "than", "and", "but", "or", "so", "because", "that",
          "which", "who", "when", "where", "if", "as"}


def cut(text, i, k, ended):
    m = FENCE.search(text, i)
    if m and not text[i:m.start()].strip():
        body = text.find("\n", m.end())
        shut = FENCE.search(text, body + 1) if body >= 0 else None
        if not shut:
            return (len(text), False) if ended else None
        nl = text.find("\n", shut.end())
        return (nl + 1, False) if nl >= 0 else (len(text), False) if ended else None
    limit = m.start() if m else len(text)
    bounds = sorted([(b.end(), True) for b in ENDS.finditer(text, i, limit) if not abbrev(text, b.start())] +
                    [(b.end(), False) for b in CLAUSE.finditer(text, i, limit)])
    least, clause, cap = RAMP[min(k, len(RAMP) - 1)]
    for b, sentence in bounds:
        if len(WORD.findall(text, i, b)) >= (least if sentence else clause):
            return b, True
    ws = list(WORD.finditer(text, i, limit))
    if k == 0:
        for j in range(4, min(len(ws), cap)):
            if ws[j].group().lower() in PHRASE and ws[j].end() < limit:
                return ws[j - 1].end(), True
    if len(ws) > cap:
        return ws[cap - 1].end(), True
    if m or ended:
        return (limit, bool(text[i:limit].strip())) if limit > i else None
    return None


def abbrev(text, at):
    if text[at] != ".":
        return False
    head = text[:at]
    tok = head.split()[-1].lstrip("(\"'[").lower() if head.strip() else ""
    starts_line = not head[:len(head) - len(tok)].strip(" \t") or head[:len(head) - len(tok)].rstrip(" \t").endswith("\n")
    return tok in ABBREV or (len(tok) == 1 and tok.isalpha()) or "." in tok or (tok.isdigit() and starts_line)


def speakable(s):
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"```.*?(```|$)", " ", s, flags=re.S)
    s = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"<https?://[^>]+>|\bhttps?://\S+|\bwww\.\S+", "a link", s)
    s = re.sub(r"`([^`\n]*)`", r"\1", s)
    s = re.sub(r"</?[A-Za-z][^>]*>", " ", s)
    lines = []
    for line in s.split("\n"):
        line = line.strip()
        if re.fullmatch(r"[|:\-\s]*-{3,}[|:\-\s]*|[*_=-]{3,}", line):
            continue
        if line.count("|") >= 2:
            continue
        if line := re.sub(r"^(#{1,6}\s+|>\s*|[-*+•]\s+|\d+[.)]\s+)+", "", line):
            lines.append(line)
    s = " ".join(l if k == len(lines) - 1 or l[-1] in ".!?,;:" else l + "." for k, l in enumerate(lines))
    s = re.sub(r"(\*\*|__|~~|\*)(?=\S)(.+?)(?<=\S)\1", r"\2", s)
    s = re.sub(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)", r"\1", s)
    s = "".join(c for c in s if unicodedata.category(c)[0] in "LNPZ" or c in "$€£¥%+=&@")
    s = re.sub(r"[*#_~^|\\<>{}\[\]]+", " ", s)
    return re.sub(r"\s+", " ", s).strip() if any(c.isalnum() for c in s) else ""


def pause(text):
    return 0.28 if text[-1:] in ".!?" else 0.14 if text[-1:] in ",;:" else 0.04


class Reply:
    def __init__(self, id):
        self.id, self.text, self.told, self.spoken, self.seq, self.pieces = id, "", 0, 0, 0, 0
        self.queue, self.ended, self.dead, self.said, self.task = collections.deque(), False, False, False, None
        self.words = frozenset()


PREVIEW = {"en": "Hi, I'm {}. This is how I sound.", "es": "Hola, soy {}. Así suena mi voz.",
           "fr": "Bonjour, je suis {}. Voici ma voix.", "it": "Ciao, sono {}. Questa è la mia voce.",
           "pt": "Olá, eu sou {}. Esta é a minha voz.", "hi": "नमस्ते, यह मेरी आवाज़ है।", "ja": "こんにちは、これが私の声です。",
           "zh": "你好，这是我的声音。"}


def rss_mb():
    return int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) // 1024


class Talk:

    def __init__(self, lib, send, listen_pool, speak_pool):
        self.lib, self.send, self.lp, self.sp = lib, send, listen_pool, speak_pool
        self.names, self.parts, self.voice, self.speed, self.preview = {}, {}, None, 1.0, 0
        self.sampler = None
        self.playing, self.echo_until, self.awaiting, self.muted = False, 0.0, False, False
        self.terms = ""
        self.opened, self.heard_any = time.monotonic(), False
        self.reply, self.gone, self.dead = None, (frozenset(), 0.0), set()
        self.listener, self.audio = Listener(self), asyncio.Queue()
        self.listen_task = asyncio.create_task(self.listening())

    def echo_words(self):
        w = self.reply.words if self.reply else frozenset()
        return w | self.gone[0] if time.monotonic() < self.gone[1] + ECHO_KEEP else w

    async def listening(self):
        loop = asyncio.get_running_loop()
        early = collections.deque(maxlen=EARLY)
        while True:
            item = await self.audio.get()
            if item is None:
                self.listener.reset()
                early.clear()
                continue
            if isinstance(item, tuple):
                self.listener.use(*item)
                if early:
                    item = np.concatenate(list(early))
                    early.clear()
                else:
                    continue
            elif not hasattr(self.listener, "vad"):
                early.append(item)
                continue
            for ev in await loop.run_in_executor(self.lp, self.listener.feed, item):
                if ev["type"] == "interrupt":
                    self.awaiting = False
                    self.drop(self.reply)
                elif ev["type"] == "turn":
                    self.awaiting = True
                    print(f"voice: turn, {ev['wait_ms']} ms after speech: {ev['text'][:80]!r}", file=sys.stderr, flush=True)
                await self.send(ev)
                if ev["type"] == "turn":
                    trim()

    def hear(self, data):
        if not self.heard_any:
            self.heard_any = True
            print(f"voice: microphone audio {round((time.monotonic() - self.opened) * 1000)} ms after the page connected",
                  file=sys.stderr, flush=True)
        if not self.muted:
            self.audio.put_nowait(np.frombuffer(data, "<i2").astype(np.float32) / 32768)

    async def message(self, m):
        t = m.get("type")
        if t in ("hello", "config"):
            await self.configure(m)
        elif t == "say":
            self.take(str(m["id"]), str(m.get("text", "")), False)
        elif t == "say_end":
            self.take(str(m["id"]), "", True)
        elif t == "cancel":
            self.awaiting = False
            if m.get("id") is None or (self.reply and self.reply.id == str(m["id"])):
                self.drop(self.reply)
            if m.get("id") is None:
                self.preview += 1
            else:
                self.dead.add(str(m["id"]))
        elif t == "playback":
            self.playing = bool(m.get("playing"))
            if self.playing:
                self.awaiting = False
            else:
                self.echo_until = time.monotonic() + ECHO_TAIL
        elif t == "context":
            self.terms = "/".join(terms(str(m.get("text", ""))))
        elif t == "mute":
            self.muted = bool(m.get("on"))
            self.audio.put_nowait(None)
        elif t == "preview":
            await self.sample(m.get("voice") or self.voice, m.get("text"), m.get("tts"))
        elif t == "catalog":
            await self.send({"type": "catalog", "catalog": self.lib.catalog()})
        elif t == "get" and m.get("tts") in PARTS["tts"]:
            try:
                await asyncio.to_thread(fetch, PARTS["tts"][m["tts"]].DIR)
            except OSError as e:
                await self.send({"type": "error", "message": f"{PARTS['tts'][m['tts']].label} could not be downloaded: {e}"})
            await self.send({"type": "catalog", "catalog": self.lib.catalog()})
        else:
            await self.send({"type": "error", "message": f"unknown message type {t!r}"})

    async def configure(self, m):
        want = {**(DEFAULTS if m["type"] == "hello" else {}), **(m.get("stack") or {})}
        if "tts" not in want and "tts" not in self.names:
            want["tts"] = DEFAULTS["tts"]
        if bad := [k for k in want if k not in PARTS]:
            return await self.send({"type": "error", "message": f"no part kind {bad[0]!r} (there are: {', '.join(PARTS)})"})
        loop, t0 = asyncio.get_running_loop(), time.monotonic()
        todo = {k: n for k, n in want.items() if self.names.get(k) != n}
        got = await asyncio.gather(*(loop.run_in_executor(None, self.lib.get, k, n) for k, n in todo.items()), return_exceptions=True)
        if loaded := [n for n, p in zip(todo.values(), got) if not isinstance(p, Exception)]:
            print(f"voice: {', '.join(loaded)} in {(time.monotonic() - t0) * 1000:.0f} ms, {rss_mb()} MB", file=sys.stderr, flush=True)
        for (kind, name), part in zip(todo.items(), got):
            if isinstance(part, Exception):
                msg = str(part) if isinstance(part, Missing) else f"{kind} {name} did not load: {part}"
                await self.send({"type": "error", "message": msg})
                continue
            if kind in self.names:
                self.lib.release(kind, self.names[kind])
            self.names[kind], self.parts[kind] = name, part
        if {"vad", "stt", "turn"} & set(todo) and {"vad", "stt", "turn"} <= set(self.names):
            self.audio.put_nowait((self.parts["vad"], self.parts["stt"], self.parts["turn"]))
        tts = self.parts.get("tts")
        ids = [v["id"] for v in tts.voice_list] if tts else []
        if "voice" in m and m["voice"] not in ids:
            await self.send({"type": "error", "message": f"{self.names.get('tts')} has no voice {m['voice']!r}"})
        self.voice = m["voice"] if m.get("voice") in ids else self.voice if self.voice in ids else tts.VOICE if tts else None
        if "speed" in m:
            self.speed = min(2.0, max(0.5, float(m["speed"])))
        await self.send({"type": "ready", "stack": dict(self.names), "catalog": self.lib.catalog(),
                         "voices": tts.voice_list if tts else [], "voice": self.voice, "speed": self.speed,
                         "sample_rate": tts.rate if tts else None, "load_ms": round((time.monotonic() - t0) * 1000)})

    async def sample(self, voice, text, name=None):
        name = name or self.names.get("tts") or DEFAULTS["tts"]
        tts = self.parts.get("tts") if name == self.names.get("tts") else await self.sampling(name)
        voice = str(voice or (tts.VOICE if tts else ""))
        v = next((x for x in (tts.voice_list if tts else []) if x["id"] == voice), None)
        if not v:
            return tts and await self.send({"type": "error", "message": f"{name} has no voice {voice!r}"})
        text = str(text or PREVIEW.get((v["lang"] or "en")[:2], PREVIEW["en"]).format(v["label"]))
        self.preview += 1
        mine = self.preview
        audio = await asyncio.get_running_loop().run_in_executor(self.sp, tts.synth, speakable(text), voice, self.speed)
        if mine == self.preview:
            await self.send(self.piece("preview", 0, speakable(text), 0, len(text), audio, tts.rate))
            await self.send({"type": "said", "id": "preview"})

    async def sampling(self, name):
        if self.sampler and self.sampler[0] == name:
            return self.sampler[1]
        try:
            part = await asyncio.get_running_loop().run_in_executor(None, self.lib.get, "tts", name)
        except Exception as e:
            return await self.send({"type": "error", "message": str(e) if isinstance(e, Missing) else f"tts {name} did not load: {e}"})
        if self.sampler:
            self.lib.release("tts", self.sampler[0])
        self.sampler = name, part
        return part

    @staticmethod
    def piece(id, seq, text, frm, to, audio, rate):
        head = json.dumps({"id": id, "seq": seq, "text": text, "from": frm, "to": to, "rate": rate,
                           "secs": round(len(audio) / rate, 3)}).encode()
        audio = np.concatenate([audio, np.zeros(int(pause(text) * rate), np.float32)])
        return struct.pack("<I", len(head)) + head + (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()

    def drop(self, r):
        if r and not r.dead:
            r.dead = True
            r.queue.clear()
            self.dead.add(r.id)
        if r and self.reply is r:
            self.gone, self.reply = (r.words, time.monotonic()), None

    def take(self, id, text, end):
        if id in self.dead:
            if end:
                asyncio.create_task(self.send({"type": "said", "id": id}))
            return
        r = self.reply
        if r is None or r.id != id:
            self.drop(r)
            r = self.reply = Reply(id)
        r.text += text
        r.ended |= end
        while (c := cut(r.text, r.told, r.pieces, r.ended)) is not None and c[0] > r.told:
            say = speakable(r.text[r.told:c[0]]) if c[1] else ""
            if say:
                r.queue.append((r.spoken, c[0], say))
                r.spoken, r.pieces = c[0], r.pieces + 1
            r.told = c[0]
        if r.task is None or r.task.done():
            r.task = asyncio.create_task(self.speaking(r))

    async def speaking(self, r):
        loop = asyncio.get_running_loop()
        while r.queue and not r.dead and (tts := self.parts.get("tts")):
            frm, to, say = r.queue.popleft()
            audio = await loop.run_in_executor(self.sp, tts.synth, say, self.voice, self.speed)
            if r.dead:
                return
            r.words = r.words | words(say)
            await self.send(self.piece(r.id, r.seq, say, frm, to, audio, tts.rate))
            r.seq += 1
        if r.ended and not r.dead and r.told >= len(r.text) and not r.said:
            r.said = True
            if r.seq == 0:
                self.awaiting = False
            await self.send({"type": "said", "id": r.id})

    def close(self):
        self.listen_task.cancel()
        if self.reply and self.reply.task:
            self.reply.task.cancel()
        for kind, name in [*self.names.items(), *([("tts", self.sampler[0])] if self.sampler else [])]:
            self.lib.release(kind, name)
