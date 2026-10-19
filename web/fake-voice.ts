// A speech service for the tests, through KILN_VOICE_CMD: voice mode's protocol, played from scripts instead of
// models. The voice a page asks for picks its script: the user says each line once some microphone audio has come
// (speech, partials word by word, then the turn), and the reply's text comes back as tone chunks of three words (or
// up to a clause's end), each lasting 0.12 s a word. "interruptAfter": the user talks over the reply once that many
// chunks were sent and the page plays them; "interruptOnSay": as soon as the reply's first text arrives, before any
// sound. "after": the line waits for the page's first config. "goOn": the next line is said straight after this one.
// "pause": that many frames of silence after the words, which the turn waits out (a pause event at their start, as
// the service sends when its turn model hears more to come), then the turn, or "more" words in the same turn.
// Its catalog has a second speech-to-text and voice model to switch to and one of each not installed (refused, as the
// real service refuses them); a preview speaks a voice's sample as chunks of the reply "preview". Every message, every chunk (with the voice that spoke it), the audio frames
// counted and its own start and exit go to FAKE_VOICE_LOG as JSON lines.
import { appendFileSync } from "node:fs";

interface Line { text: string; interruptAfter?: number; interruptOnSay?: boolean; after?: "config"; goOn?: boolean; pause?: number; more?: string }
const SCRIPTS: Record<string, Line[]> = {
  plain: [{ text: "hello there" }],
  pie: [{ text: "lets do a snake game in pie game" }],
  story: [{ text: "slow story", interruptAfter: 2 }, { text: "and then" }],
  merge: [{ text: "slow first", interruptOnSay: true }, { text: "second part" }],
  join: [{ text: "wait lets do a snake game", goOn: true }, { text: "in pie game" }],
  ahead: [{ text: "what is the capital of France", pause: 60 }],
  onward: [{ text: "slow what is the", pause: 20, more: "capital of France" }],
  checking: [{ text: "lets do a snake game in pie game", pause: 75 }],
  silent: [],
  pick: [{ text: "first" }, { text: "second", after: "config" }],
  notes: [{ text: "first question" }, { text: "second question" }, { text: "third question", after: "config" }],
};
const voice = (id: string, lang: string, gender: string) => ({ id, label: id[0].toUpperCase() + id.slice(1), lang, gender });
const CATALOG = {
  vad: [{ name: "fake-vad", label: "Fake VAD", installed: true }],
  stt: [{ name: "fake-stt", label: "Fake STT", installed: true }, { name: "fake-stt-big", label: "Fake STT Big", installed: true },
    { name: "fake-stt-huge", label: "Fake STT Huge", installed: false }],
  turn: [{ name: "fake-turn", label: "Fake Turn", installed: true }],
  tts: [
    { name: "fake-tts", label: "Fake TTS", installed: true, voices: [...Object.keys(SCRIPTS).map((k) => voice(k, "en-us", "female")), voice("echo", "en-gb", "male")] },
    { name: "other-tts", label: "Other TTS", installed: true, voices: [voice("ana", "es", "female"), voice("bob", "en-us", "male")] },
    { name: "big-tts", label: "Big TTS", installed: false, voices: [voice("zed", "en-us", "male")] },
  ],
} as const;
type Part = keyof typeof CATALOG;
const RATE = 24000, LOG = process.env.FAKE_VOICE_LOG;
const log = (o: object) => { if (LOG) appendFileSync(LOG, `${JSON.stringify(o)}\n`); };

interface Reply { text: string[]; told: number; seq: number; sent: number; ended: boolean; dropped: boolean }
interface Session {                               // n: pid.count, as a service started again counts from 1 and the log outlives it
  n: string; script: Line[]; at: number; line?: Line; listening: boolean; heard: number; frames: number; configs: number;
  stack: Record<Part, string>; voice: string; speed: number;
  replies: Map<string, Reply>; playing: boolean; answer?: string; overtalk?: boolean;
}
let sessions = 0;

function tone(seconds: number) {
  const pcm = new Int16Array(Math.round(seconds * RATE));
  for (let i = 0; i < pcm.length; i++) pcm[i] = Math.round(6000 * Math.sin((2 * Math.PI * 220 * i) / RATE));
  return pcm;
}

function chunk(ws: Bun.ServerWebSocket<Session>, id: string, r: Reply, to: number, spoken = ws.data.voice) {
  const text = r.text.slice(r.told, to).join(""), words = text.trim().split(/\s+/).filter(Boolean).length;
  const header = new TextEncoder().encode(JSON.stringify({ id, seq: r.seq++, text, from: r.told, to, rate: RATE }));
  const pcm = tone(Math.max(0.25, 0.12 * words)), buf = new Uint8Array(4 + header.length + pcm.byteLength);
  new DataView(buf.buffer).setUint32(0, header.length, true);
  buf.set(header, 4);
  buf.set(new Uint8Array(pcm.buffer), 4 + header.length);
  log({ s: ws.data.n, ev: "chunk", id, text, from: r.told, to, voice: spoken, tts: ws.data.stack.tts, speed: ws.data.speed });
  r.told = to;
  r.sent++;
  ws.send(buf);
  overtalk(ws);
}

// The script's talking over: once the page plays the reply and enough of it was sent.
function overtalk(ws: Bun.ServerWebSocket<Session>) {
  const d = ws.data, r = d.answer ? d.replies.get(d.answer) : undefined, after = d.line?.interruptAfter;
  if (!r || !after || !d.playing || r.sent < after || r.dropped || d.overtalk) return;
  d.overtalk = true;
  setTimeout(() => { if (!r.dropped) interrupt(ws); }, 150);
}

// Speaks what is complete: three words, or a clause.
function speak(ws: Bun.ServerWebSocket<Session>, id: string, r: Reply) {
  for (;;) {
    const rest = r.text.slice(r.told).join(""), clause = /[.,!?;:]\s/.exec(rest), three = /^\s*(\S+\s+){3}/.exec(rest);
    const end = clause ? clause.index + 2 : three ? three[0].length : 0;
    if (!end) return;
    chunk(ws, id, r, r.told + Array.from(rest.slice(0, end)).length);
  }
}

function interrupt(ws: Bun.ServerWebSocket<Session>) {
  for (const r of ws.data.replies.values()) r.dropped = true;
  ws.data.answer = undefined;
  ws.data.listening = true;
  ws.data.heard = 0;
  ws.send(JSON.stringify({ type: "interrupt" }));
  log({ s: ws.data.n, ev: "interrupt" });
}

// One microphone frame: the next line of the script is said over the frames that follow, a word each 10 frames.
function hear(ws: Bun.ServerWebSocket<Session>) {
  const d = ws.data;
  if (++d.frames % 10 === 0) log({ s: d.n, ev: "frames", n: d.frames });
  const line = d.script[d.at];
  if (!d.listening || !line || (line.after === "config" && !d.configs)) return;
  const k = ++d.heard, step = 10, first = line.text.split(" "), more = line.more?.split(" ") ?? [];
  const send = (o: object) => (ws.send(JSON.stringify(o)), log({ s: d.n, ev: "sent", t: Date.now(), ...o }));
  // the words from frame `at` on, after `before` (already said in this turn): speech, a partial a word, its end
  const say = (at: number, before: string[], words: string[]) => {
    const w = (k - at) / step - 1;
    if (k === at + step) send({ type: "speech", on: true });
    if ((k - at) % step === 0 && w >= 0 && w < words.length) send({ type: "partial", text: [...before, ...words.slice(0, w + 1)].join(" ") });
    return k === at + step * (words.length + 2);
  };
  const turn = (text: string) => {
    send({ type: "turn", text, wait_ms: 210 });
    d.listening = !!line.goOn;
    d.heard = 0;
    d.line = line;
    d.at++;
  };
  const ended = say(0, [], first), back = step * (first.length + 2) + (line.pause ?? 0);
  if (ended) send({ type: "speech", on: false });
  if (!line.pause) return void (ended && turn(line.text));
  if (ended) send({ type: "pause", text: line.text });
  if (!more.length) return void (k === back && turn(line.text));
  if (k > back && say(back, first, more)) {
    send({ type: "speech", on: false });
    turn([...first, ...more].join(" "));
  }
}

const voicesOf = (tts: string) => CATALOG.tts.find((t) => t.name === tts)?.voices ?? [];
const ready = (ws: Bun.ServerWebSocket<Session>) => ws.send(JSON.stringify({ type: "ready", stack: ws.data.stack, catalog: CATALOG,
  voices: voicesOf(ws.data.stack.tts), voice: ws.data.voice, speed: ws.data.speed, sample_rate: RATE, load_ms: 5 }));

// hello's and config's asks: a part of the stack by name (loaded, then ready again), the voice, the speed. A model
// it does not have, or does not have installed, is refused and the one in use kept.
function ask(ws: Bun.ServerWebSocket<Session>, m: any): boolean {
  const d = ws.data, stack = { ...d.stack };
  for (const [part, name] of Object.entries(m.stack ?? {}) as [Part, string][]) {
    const model = CATALOG[part]?.find((x) => x.name === name);
    if (!model?.installed) {
      ws.send(JSON.stringify({ type: "error", message: `${name} is ${model ? "not installed" : "unknown"}` }));
      return false;
    }
    stack[part] = name;
  }
  const switched = stack.tts !== d.stack.tts || stack.stt !== d.stack.stt;
  d.stack = stack;
  if (m.voice && voicesOf(stack.tts).some((v) => v.id === m.voice)) d.voice = m.voice;
  else if (!voicesOf(stack.tts).some((v) => v.id === d.voice)) d.voice = voicesOf(stack.tts)[0]?.id ?? d.voice;
  if (m.speed) d.speed = m.speed;
  return switched;
}

function message(ws: Bun.ServerWebSocket<Session>, m: any) {
  const d = ws.data;
  log({ s: d.n, ev: "msg", ...m });
  if (m.type === "hello") {
    d.script = SCRIPTS[m.voice] ?? SCRIPTS.plain;
    ask(ws, m);
    ready(ws);
    d.listening = true;
  } else if (m.type === "config") {
    d.configs++;
    if (ask(ws, m)) ready(ws);
  } else if (m.type === "preview") {
    const v = Object.values(CATALOG.tts).flatMap((t) => t.voices).find((x) => x.id === m.voice);
    const r: Reply = { text: Array.from(m.text ?? `This is ${v?.label ?? m.voice}.`), told: 0, seq: 0, sent: 0, ended: true, dropped: false };
    chunk(ws, "preview", r, r.text.length, m.voice);
    ws.send(JSON.stringify({ type: "said", id: "preview" }));
  } else if (m.type === "say") {
    const r = d.replies.get(m.id) ?? { text: [], told: 0, seq: 0, sent: 0, ended: false, dropped: false };
    d.replies.set(m.id, r);
    if (r.dropped) return;
    if (!d.answer) { d.answer = m.id; d.overtalk = false; }
    r.text.push(...Array.from(String(m.text)));
    if (d.line?.interruptOnSay && d.answer === m.id) return interrupt(ws);
    speak(ws, m.id, r);
  } else if (m.type === "say_end") {
    const r = d.replies.get(m.id) ?? { text: [], told: 0, seq: 0, sent: 0, ended: false, dropped: false };
    d.replies.set(m.id, r);
    if (!r.dropped) {
      if (r.told < r.text.length) chunk(ws, m.id, r, r.text.length);
      r.ended = true;
      ws.send(JSON.stringify({ type: "said", id: m.id }));
      if (!r.sent && d.answer === m.id) d.listening = true;
    }
  } else if (m.type === "cancel" && m.id !== undefined) {          // one reply (one spoken ahead, not heard)
    const r = d.replies.get(m.id) ?? { text: [], told: 0, seq: 0, sent: 0, ended: false, dropped: false };
    d.replies.set(m.id, { ...r, dropped: true });
    if (d.answer === m.id) d.answer = undefined;
  } else if (m.type === "cancel") {
    for (const r of d.replies.values()) r.dropped = true;
    Object.assign(d, { answer: undefined, listening: true, heard: 0 });
  } else if (m.type === "playback") {
    d.playing = !!m.playing;
    const r = d.answer ? d.replies.get(d.answer) : undefined;
    overtalk(ws);
    if (!d.playing && r?.ended) { d.answer = undefined; d.listening = true; d.heard = 0; }
  }
}

const server = Bun.serve<Session>({
  hostname: "127.0.0.1",
  port: 0,
  fetch(req, srv) {
    return srv.upgrade(req, { data: { n: `${process.pid}.${++sessions}`, script: [], at: 0, listening: false, heard: 0, frames: 0, configs: 0, replies: new Map(), playing: false,
      stack: { vad: "fake-vad", stt: "fake-stt", turn: "fake-turn", tts: "fake-tts" }, voice: "plain", speed: 1 } })
      ? undefined : new Response("a WebSocket only", { status: 426 });
  },
  websocket: {
    open(ws) { log({ s: ws.data.n, ev: "open" }); },
    message(ws, m) { if (typeof m === "string") message(ws, JSON.parse(m)); else hear(ws); },
    close(ws) { log({ s: ws.data.n, ev: "close" }); },
  },
});
log({ ev: "start" });
console.log(`voice: ws://127.0.0.1:${server.port}`);
for await (const _ of Bun.stdin.stream()) { /* the web server's end of the pipe */ }
log({ ev: "exit" });
process.exit(0);
