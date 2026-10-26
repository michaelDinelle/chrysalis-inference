export type Part = "vad" | "stt" | "turn" | "tts";
export interface VoiceInfo { id: string; label: string; lang: string; gender: string }
export interface ModelInfo { name: string; label: string; installed: boolean; hint?: string; voices?: VoiceInfo[] }
export interface VoiceReady {
  stack: Partial<Record<Part, string>>; catalog: Partial<Record<Part, ModelInfo[]>>; voices: VoiceInfo[]; voice: string; speed: number;
  sample_rate: number; load_ms: number;
}
export interface VoiceAsk { stack?: Partial<Record<Part, string>>; voice?: string; speed?: number }
export const PREVIEW = "preview";
export interface VoiceHandlers {
  ready?(r: VoiceReady): void;
  speech?(on: boolean): void;
  partial?(text: string, settled: boolean): void;
  pause?(text: string): void;
  turn?(text: string, waitMs: number): void;
  interrupt?(): void;
  spoken?(id: string, from: number, to: number, at: number, secs: number): void;
  audible?(id: string, latencyMs?: number): void;
  played?(id: string): void;
  playing?(on: boolean): void;
  error?(message: string): void;
  closed?(): void;
}
interface Chunk { id: string; seq: number; text: string; from: number; to: number; rate: number; secs?: number }

const LEAD = 0.03;

export class VoiceClient {
  private ws?: WebSocket;
  private out?: AudioContext;
  private gain?: GainNode;
  private mic?: AudioContext;
  private stream?: MediaStream;
  private outLevel?: AnalyserNode;
  private micLevel?: AnalyserNode;
  private samples = new Float32Array(512);
  private end = 0;
  private live = new Map<AudioBufferSourceNode, string>();
  private timers = new Set<number>();
  private said = new Set<string>();
  private dropped = new Set<string>();
  private heard = new Set<string>();
  private sent = new Set<string>();
  private turnAt?: { t: number; wait: number };
  private answers = new Map<string, { t: number; wait: number }>();
  private shut = false;
  private early: ArrayBuffer[] = [];
  private held = new Map<string, ArrayBuffer[]>();
  private told?: string;
  private sample?: () => void;
  playing = false;
  muted = false;
  hearing = false;

  constructor(private on: VoiceHandlers) {}

  async open(ask: VoiceAsk) {
    this.out = new AudioContext({ latencyHint: "interactive" });
    this.gain = this.out.createGain();
    this.outLevel = this.out.createAnalyser();
    this.outLevel.fftSize = 512;
    this.gain.connect(this.outLevel).connect(this.out.destination);
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
    let ctx = (this.mic = new AudioContext({ sampleRate: 16000, latencyHint: "interactive" })), src: MediaStreamAudioSourceNode;
    let worklet = ctx.audioWorklet.addModule("/voice-worklet.js");
    void ctx.resume();
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    if (this.shut) return this.release();
    try {
      src = ctx.createMediaStreamSource(this.stream);
    } catch {
      void ctx.close();
      ctx = this.mic = new AudioContext({ latencyHint: "interactive" });
      worklet = ctx.audioWorklet.addModule("/voice-worklet.js");
      src = ctx.createMediaStreamSource(this.stream);
    }
    await worklet;
    const node = new AudioWorkletNode(ctx, "chrysalis-capture"), sink = ctx.createGain();
    sink.gain.value = 0;
    this.micLevel = ctx.createAnalyser();
    this.micLevel.fftSize = 512;
    src.connect(node).connect(sink).connect(ctx.destination);
    src.connect(this.micLevel);
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
  context(text: string) {
    if (text === this.told) return;
    this.told = text;
    this.send({ type: "context", text });
  }
  config(ask: VoiceAsk) { this.send({ type: "config", ...ask }); }

  preview(voice?: string, tts?: string): Promise<void> {
    this.previewDone();
    for (const [src, id] of this.live) if (id === PREVIEW) { src.onended = null; src.stop(); this.live.delete(src); }
    for (const set of [this.dropped, this.said, this.heard]) set.delete(PREVIEW);
    if (!this.live.size) this.end = 0;
    const heard = new Promise<void>((done) => { this.sample = done; });
    this.send({ type: "preview", ...(voice ? { voice } : {}), ...(tts ? { tts } : {}) });
    return heard;
  }
  private previewDone() {
    const done = this.sample;
    this.sample = undefined;
    done?.();
  }
  private played(id: string) {
    if (id === PREVIEW) this.previewDone();
    this.on.played?.(id);
  }

  mute(on: boolean) {
    this.muted = on;
    for (const t of this.stream?.getAudioTracks() ?? []) t.enabled = !on;
    this.send({ type: "mute", on });
  }

  stop() {
    this.halt();
    this.send({ type: "cancel" });
  }

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
      if (![...this.live.values()].includes(e.id) && !this.dropped.has(e.id) && !this.held.has(e.id)) this.played(e.id);
    } else if (e.type === "error") {
      this.previewDone();
      this.on.error?.(String(e.message ?? "The voice service failed"));
    }
  }

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
      if (this.said.has(h.id) && ![...this.live.values()].includes(h.id)) this.played(h.id);
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
    this.previewDone();
    for (const id of this.sent) this.dropped.add(id);
    for (const src of this.live.keys()) {
      src.onended = null;
      try { src.stop(); } catch {  }
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

export class Speaker {
  private ws?: WebSocket;
  private out?: AudioContext;
  private live = new Set<AudioBufferSourceNode>();
  private end = 0;
  private now?: string;
  private reading?: object;
  private ids = 0;
  private listed: ((c: VoiceReady["catalog"] | null) => void)[] = [];
  private readied: ((r: VoiceReady | null) => void)[] = [];
  private made = new Map<string, () => void>();
  private quiet: (() => void)[] = [];
  private hearing?: { done: () => void; said: boolean };
  previewed = false;
  constructor(private failed: (message: string) => void) {}

  get(tts: string) { this.send({ type: "get", tts }); }
  catalog(): Promise<VoiceReady["catalog"] | null> {
    return new Promise((done) => { this.listed.push(done); this.send({ type: "catalog" }); });
  }
  preview(tts?: string, voice?: string): Promise<void> {
    this.previewed = true;
    this.stop();
    this.sound();
    this.now = PREVIEW;
    const heard = new Promise<void>((done) => { this.hearing = { done, said: false }; });
    this.send({ type: "preview", ...(tts ? { tts } : {}), ...(voice ? { voice } : {}) });
    return heard;
  }
  async read(parts: { text: string; voice?: { tts?: string; name?: string } }[], base: VoiceAsk, done: () => void) {
    this.stop();
    this.sound();
    const run = (this.reading = {});
    this.send({ type: "config", ...base });
    const r = await new Promise<VoiceReady | null>((ok) => this.readied.push(ok));
    let asked = "";
    for (const p of parts) {
      if (this.reading !== run || !r) return;
      const tts = p.voice?.name ? p.voice.tts ?? r.stack.tts : r.stack.tts, name = p.voice?.name ?? r.voice;
      const want: VoiceAsk = { ...(tts ? { stack: { tts } } : {}), ...(name ? { voice: name } : {}), speed: base.speed };
      if (JSON.stringify(want) !== asked) this.send({ type: "config", ...want });
      asked = JSON.stringify(want);
      const id = (this.now = `read-${++this.ids}`);
      const made = new Promise<void>((ok) => this.made.set(id, ok));
      this.send({ type: "say", id, text: p.text });
      this.send({ type: "say_end", id });
      await made;
    }
    if (this.reading !== run) return;
    if (this.live.size) await new Promise<void>((ok) => this.quiet.push(ok));
    if (this.reading === run) { this.reading = this.now = undefined; done(); }
  }
  stop() {
    if (this.now && this.now !== PREVIEW && this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: "cancel", id: this.now }));
    this.reading = this.now = undefined;
    for (const ok of this.made.values()) ok();
    this.made.clear();
    for (const src of this.live) {
      src.onended = null;
      try { src.stop(); } catch {  }
    }
    this.live.clear();
    for (const ok of this.quiet.splice(0)) ok();
    this.end = 0;
    this.heard();
  }
  close() {
    this.stop();
    this.ws?.close();
    void this.out?.close().catch(() => {});
    this.ws = this.out = undefined;
    this.gone();
  }

  private sound() {
    this.out ??= new AudioContext({ latencyHint: "interactive" });
    void this.out.resume();
  }
  private heard() {
    const h = this.hearing;
    this.hearing = undefined;
    h?.done();
  }
  private gone() {
    this.heard();
    for (const done of this.listed.splice(0)) done(null);
    for (const done of this.readied.splice(0)) done(null);
    for (const ok of this.made.values()) ok();
    this.made.clear();
  }
  private send(o: object) {
    if (!this.ws || this.ws.readyState > WebSocket.OPEN) {
      const ws = (this.ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/voice`));
      ws.binaryType = "arraybuffer";
      ws.onmessage = (e) => {
        if (typeof e.data !== "string") return this.play(e.data);
        const m = JSON.parse(e.data);
        if (m.type === "catalog") for (const done of this.listed.splice(0)) done(m.catalog ?? {});
        else if (m.type === "ready") for (const done of this.readied.splice(0)) done(m);
        else if (m.type === "said") {
          this.made.get(m.id)?.();
          this.made.delete(m.id);
          if (m.id === PREVIEW && this.now === PREVIEW && this.hearing) {
            this.hearing.said = true;
            if (!this.live.size) this.heard();
          }
        } else if (m.type === "error") {
          this.failed(m.message);
          this.heard();
        }
      };
      ws.onclose = () => { if (this.ws === ws) this.gone(); };
    }
    const ws = this.ws, text = JSON.stringify(o);
    if (ws.readyState === WebSocket.OPEN) ws.send(text);
    else ws.addEventListener("open", () => ws.send(text), { once: true });
  }
  private play(buf: ArrayBuffer) {
    const n = new DataView(buf).getUint32(0, true), h: Chunk = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
    const ctx = this.out;
    if (!ctx || h.id !== this.now) return;
    const pcm = new Int16Array(buf.slice(4 + n)), ab = ctx.createBuffer(1, Math.max(1, pcm.length), h.rate), ch = ab.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const src = ctx.createBufferSource(), start = Math.max(ctx.currentTime + LEAD, this.end);
    src.buffer = ab;
    src.connect(ctx.destination);
    src.start(start);
    this.end = start + ab.duration;
    this.live.add(src);
    src.onended = () => {
      this.live.delete(src);
      if (this.live.size) return;
      for (const ok of this.quiet.splice(0)) ok();
      if (this.hearing?.said) this.heard();
    };
  }
}
