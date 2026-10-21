// Voice mode's audio and its line to the speech service: the microphone goes out as 16 kHz int16 frames, the speech
// comes back as chunks played back to back, over one WebSocket (/api/voice, which server.ts hands to whatever stack
// runs behind it). It knows the protocol, not the chat: app.ts says what to speak and is told what was said.
export type Part = "vad" | "stt" | "turn" | "tts";
export interface VoiceInfo { id: string; label: string; lang: string; gender: string }
// A model the service knows: one it can load (installed), or one it would need set up first (hint: how).
export interface ModelInfo { name: string; label: string; installed: boolean; hint?: string; voices?: VoiceInfo[] }
export interface VoiceReady {
  stack: Partial<Record<Part, string>>; catalog: Partial<Record<Part, ModelInfo[]>>; voices: VoiceInfo[]; voice: string; speed: number;
  sample_rate: number; load_ms: number;
}
// What the page asks the service for (hello, config): parts of the stack by name, the voice, its speed.
export interface VoiceAsk { stack?: Partial<Record<Part, string>>; voice?: string; speed?: number }
export const PREVIEW = "preview";                   // the id of a voice's sample, beside any reply
export interface VoiceHandlers {
  ready?(r: VoiceReady): void;
  speech?(on: boolean): void;                       // the user started or paused speaking
  partial?(text: string, settled: boolean): void;   // the turn's words so far (settled: read at a pause, which only more speech changes)
  pause?(text: string): void;                       // the turn waits for more: its words, which are its turn's if no more come
  turn?(text: string, waitMs: number): void;        // the user is done: the turn's words
  interrupt?(): void;                               // the user talks over the reply (its sound has already stopped)
  // a chunk of reply `id` began to sound: its text is [from, to) of the reply (code points), its speech lasts `secs`
  // from `at` (performance.now(), as it reaches the ear)
  spoken?(id: string, from: number, to: number, at: number, secs: number): void;
  audible?(id: string, latencyMs?: number): void;   // the reply's first sound; with a turn before it, the time from the end of that speech
  played?(id: string): void;                        // the reply's last sound ended, or it had nothing to say
  playing?(on: boolean): void;
  error?(message: string): void;
  closed?(): void;                                  // the service's side went away
}
interface Chunk { id: string; seq: number; text: string; from: number; to: number; rate: number; secs?: number }

const LEAD = 0.03;                                  // s from a chunk's arrival to its start when nothing plays: the audio thread's room

export class VoiceClient {
  private ws?: WebSocket;
  private out?: AudioContext;
  private gain?: GainNode;
  private mic?: AudioContext;
  private stream?: MediaStream;
  private outLevel?: AnalyserNode;
  private micLevel?: AnalyserNode;
  private samples = new Float32Array(512);
  private end = 0;                                  // the out context's time the last scheduled chunk ends
  private live = new Map<AudioBufferSourceNode, string>();
  private timers = new Set<number>();
  private said = new Set<string>();                 // replies whose audio has all been sent
  private dropped = new Set<string>();              // replies stopped: chunks still on their way are not played
  private heard = new Set<string>();
  private sent = new Set<string>();
  private turnAt?: { t: number; wait: number };     // the last turn, until a reply answers it
  private answers = new Map<string, { t: number; wait: number }>();
  private shut = false;
  private early: ArrayBuffer[] = [];               // microphone frames from before the socket opened
  private held = new Map<string, ArrayBuffer[]>(); // replies spoken ahead: their chunks, unplayed until let go
  private told?: string;                          // the conversation as last given (context), sent again on open
  playing = false;
  muted = false;
  hearing = false;                                  // the microphone gives audio: from now on what is said is heard

  constructor(private on: VoiceHandlers) {}

  // The socket first (the service loads its models meanwhile), then the microphone: the user hears nothing until the
  // service is ready, and speaks only once the page says it is listening.
  async open(ask: VoiceAsk) {
    this.out = new AudioContext({ latencyHint: "interactive" });
    this.gain = this.out.createGain();
    this.outLevel = this.out.createAnalyser();
    this.outLevel.fftSize = 512;
    this.gain.connect(this.outLevel).connect(this.out.destination);
    // A faint noise (-80 dB, not heard) keeps the output playing between replies: Bluetooth headphones and some Macs'
    // outputs sleep on digital silence and wake on the next sound, swallowing its first syllable
    const hush = this.out.createBuffer(1, this.out.sampleRate, this.out.sampleRate), data = hush.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * 1e-4;
    const keep = this.out.createBufferSource();
    keep.buffer = hush;
    keep.loop = true;
    keep.connect(this.out.destination);
    keep.start();
    void this.out.resume();
    const ws = (this.ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/voice`));
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.send({ type: "hello", ...ask });
      if (this.told !== undefined) this.send({ type: "context", text: this.told });
      for (const f of this.early.splice(0)) ws.send(f);
    };
    ws.onmessage = (e) => (typeof e.data === "string" ? this.event(JSON.parse(e.data)) : this.chunk(e.data));
    ws.onclose = () => { if (!this.shut) this.on.closed?.(); };
    // The microphone's context is made now, in the click that opened voice mode: where audio starts only from a click
    // (Safari) one made after the wait for the microphone may start suspended. Its worklet loads meanwhile.
    let ctx = (this.mic = new AudioContext({ sampleRate: 16000, latencyHint: "interactive" })), src: MediaStreamAudioSourceNode;
    let worklet = ctx.audioWorklet.addModule("/voice-worklet.js");
    void ctx.resume();
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    if (this.shut) return this.release();
    try {
      src = ctx.createMediaStreamSource(this.stream);
    } catch {                                       // Firefox: the microphone only at the device's rate; the worklet resamples
      void ctx.close();
      ctx = this.mic = new AudioContext({ latencyHint: "interactive" });
      worklet = ctx.audioWorklet.addModule("/voice-worklet.js");
      src = ctx.createMediaStreamSource(this.stream);
    }
    await worklet;
    const node = new AudioWorkletNode(ctx, "chrysalis-capture"), sink = ctx.createGain();
    sink.gain.value = 0;                            // a worklet runs only on a path to the destination; it is silent
    this.micLevel = ctx.createAnalyser();
    this.micLevel.fftSize = 512;
    src.connect(node).connect(sink).connect(ctx.destination);
    src.connect(this.micLevel);
    // what is said before the socket opens goes once it does (the service keeps it while its models load), so the
    // first words are heard however soon they come
    node.port.onmessage = (e) => {
      this.hearing = true;
      if (this.muted) return;
      if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(e.data);
      else if (this.ws?.readyState === WebSocket.CONNECTING && this.early.length < 750) this.early.push(e.data);
    };
    void ctx.resume();
  }

  close() {
    this.shut = true;
    this.halt();
    this.ws?.close();
    this.release();
  }

  private release() {
    for (const t of this.stream?.getTracks() ?? []) t.stop();
    void this.mic?.close().catch(() => {});
    void this.out?.close().catch(() => {});
  }

  say(id: string, text: string) {
    if (!this.sent.has(id) && this.turnAt && !this.held.has(id)) { this.answers.set(id, this.turnAt); this.turnAt = undefined; }
    this.sent.add(id);
    this.send({ type: "say", id, text });
  }
  // A reply spoken before it is known to stand (app.ts, Ahead): its audio is kept, not played, until go, and cancel
  // drops it on both sides. Let go, it answers the last turn.
  hold(id: string) { this.held.set(id, []); }
  go(id: string) {
    const kept = this.held.get(id);
    if (!kept) return;
    this.held.delete(id);
    if (this.turnAt) { this.answers.set(id, this.turnAt); this.turnAt = undefined; }
    for (const buf of kept) this.chunk(buf);
    if (this.said.has(id) && ![...this.live.values()].includes(id)) this.on.played?.(id);
  }
  cancel(id: string) {
    this.held.delete(id);
    this.dropped.add(id);
    this.send({ type: "cancel", id });
  }
  done(id: string) { this.send({ type: "say_end", id }); }
  // The conversation's text, for the service to listen for its names and terms from the next turn on.
  context(text: string) {
    if (text === this.told) return;
    this.told = text;
    this.send({ type: "context", text });
  }
  config(ask: VoiceAsk) { this.send({ type: "config", ...ask }); }

  // A voice's sample: a new one replaces the one playing; it is not a reply, so nothing waits for it.
  preview(voice: string, text?: string) {
    for (const [src, id] of this.live) if (id === PREVIEW) { src.onended = null; src.stop(); this.live.delete(src); }
    for (const set of [this.dropped, this.said, this.heard]) set.delete(PREVIEW);
    if (!this.live.size) this.end = 0;
    this.send({ type: "preview", voice, ...(text ? { text } : {}) });
  }

  mute(on: boolean) {
    this.muted = on;
    for (const t of this.stream?.getAudioTracks() ?? []) t.enabled = !on;
    this.send({ type: "mute", on });
  }

  // Barge-in from this side (a click, Esc): silent at once, and the service drops what it had left to say.
  stop() {
    this.halt();
    this.send({ type: "cancel" });
  }

  // The sound now: the microphone's while listening, the speech's while speaking (RMS, 0..1).
  levels() { return { mic: this.rms(this.micLevel), out: this.rms(this.outLevel) }; }
  private rms(a?: AnalyserNode) {
    if (!a) return 0;
    a.getFloatTimeDomainData(this.samples);
    let s = 0;
    for (const x of this.samples) s += x * x;
    return Math.sqrt(s / this.samples.length);
  }

  private send(o: object) { if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(o)); }

  private event(e: any) {
    if (e.type === "ready") this.on.ready?.(e);
    else if (e.type === "speech") this.on.speech?.(!!e.on);
    else if (e.type === "partial") this.on.partial?.(String(e.text ?? ""), e.settled === true);
    else if (e.type === "pause") this.on.pause?.(String(e.text ?? ""));
    else if (e.type === "turn") {
      this.turnAt = { t: performance.now(), wait: Number(e.wait_ms) || 0 };
      this.on.turn?.(String(e.text ?? ""), Number(e.wait_ms) || 0);
    } else if (e.type === "interrupt") {
      this.halt();
      this.on.interrupt?.();
    } else if (e.type === "said") {
      this.said.add(e.id);
      if (![...this.live.values()].includes(e.id) && !this.dropped.has(e.id) && !this.held.has(e.id)) this.on.played?.(e.id);
    } else if (e.type === "error") this.on.error?.(String(e.message ?? "The voice service failed"));
  }

  // [uint32 LE header length][header JSON][int16 LE PCM]: scheduled right after what is already queued, so the
  // chunks play gapless as long as they come faster than they last (a reply's speech is written several times faster).
  private chunk(buf: ArrayBuffer) {
    const n = new DataView(buf).getUint32(0, true), h: Chunk = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
    const ctx = this.out;
    if (!ctx || this.dropped.has(h.id)) return;
    if (this.held.has(h.id)) return void this.held.get(h.id)!.push(buf);
    const pcm = new Int16Array(buf.slice(4 + n)), ab = ctx.createBuffer(1, Math.max(1, pcm.length), h.rate), ch = ab.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const src = ctx.createBufferSource(), now = ctx.currentTime, start = Math.max(now + LEAD, this.end);
    src.buffer = ab;
    src.connect(this.gain!);
    src.start(start);
    this.end = start + ab.duration;
    this.live.set(src, h.id);
    src.onended = () => {
      this.live.delete(src);
      if (this.said.has(h.id) && ![...this.live.values()].includes(h.id)) this.on.played?.(h.id);
      if (!this.live.size) this.setPlaying(false);
    };
    const delay = (start - now) * 1000, audibleAt = performance.now() + delay + (ctx.outputLatency || ctx.baseLatency || 0) * 1000;
    const timer = window.setTimeout(() => {
      this.timers.delete(timer);
      if (!this.heard.has(h.id)) {
        this.heard.add(h.id);
        const a = this.answers.get(h.id);
        this.on.audible?.(h.id, a ? a.wait + audibleAt - a.t : undefined);
      }
      this.setPlaying(true);
      this.on.spoken?.(h.id, h.from, h.to, audibleAt, h.secs ?? ab.duration);
    }, delay);
    this.timers.add(timer);
  }

  private halt() {
    for (const id of this.sent) this.dropped.add(id);
    for (const src of this.live.keys()) {
      src.onended = null;
      try { src.stop(); } catch { /* not started */ }
    }
    this.live.clear();
    for (const t of this.timers) clearTimeout(t);
    this.timers.clear();
    this.end = 0;
    this.setPlaying(false);
  }

  private setPlaying(on: boolean) {
    if (on === this.playing) return;
    this.playing = on;
    this.send({ type: "playback", playing: on });
    this.on.playing?.(on);
  }
}

// The voices there are and a voice's sample, apart from voice mode (a character's voice picker): a socket of its own
// with no microphone, for which the service loads only the voice model a sample needs. It closes when the picker does.
export class VoiceSampler {
  private ws?: WebSocket;
  private out?: AudioContext;
  private sound?: AudioBufferSourceNode;
  private listed: ((c: VoiceReady["catalog"] | null) => void)[] = [];
  previewed = false;                                // a sample was asked for
  constructor(private failed: (message: string) => void) {}

  catalog(): Promise<VoiceReady["catalog"] | null> {
    return new Promise((done) => { this.listed.push(done); this.send({ type: "catalog" }); });
  }
  // in the click that asked: where audio starts only from a click (Safari), a context made later starts suspended
  preview(tts: string, voice: string) {
    this.previewed = true;
    this.out ??= new AudioContext({ latencyHint: "interactive" });
    void this.out.resume();
    this.hush();
    this.send({ type: "preview", tts, voice });
  }
  close() {
    this.hush();
    this.ws?.close();
    void this.out?.close().catch(() => {});
    this.ws = this.out = undefined;
    for (const done of this.listed.splice(0)) done(null);
  }

  private send(o: object) {
    if (!this.ws || this.ws.readyState > WebSocket.OPEN) {
      const ws = (this.ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/voice`));
      ws.binaryType = "arraybuffer";
      ws.onmessage = (e) => {
        if (typeof e.data !== "string") return this.play(e.data);
        const m = JSON.parse(e.data);
        if (m.type === "catalog") for (const done of this.listed.splice(0)) done(m.catalog ?? {});
        else if (m.type === "error") this.failed(m.message);
      };
      ws.onclose = () => { if (this.ws === ws) for (const done of this.listed.splice(0)) done(null); };
    }
    const ws = this.ws, text = JSON.stringify(o);
    if (ws.readyState === WebSocket.OPEN) ws.send(text);
    else ws.addEventListener("open", () => ws.send(text), { once: true });
  }
  private play(buf: ArrayBuffer) {
    const n = new DataView(buf).getUint32(0, true), h: Chunk = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
    if (!this.out || h.id !== PREVIEW) return;
    const pcm = new Int16Array(buf.slice(4 + n)), ab = this.out.createBuffer(1, Math.max(1, pcm.length), h.rate), ch = ab.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    this.hush();
    const src = (this.sound = this.out.createBufferSource());
    src.buffer = ab;
    src.connect(this.out.destination);
    src.start();
  }
  private hush() {
    try { this.sound?.stop(); } catch { /* not started */ }
    this.sound = undefined;
  }
}
