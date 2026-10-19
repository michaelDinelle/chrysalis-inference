// Voice mode's latency end to end on this machine: questions spoken into headless Chromium's microphone (a WAV made by
// the voice service's own TTS in another voice), through server.ts, the real voice service and the running Chrysalis,
// to the reply's first sound: the service's wait after the speech, then the turn to the first sound (a reply begun
// ahead of the turn, at a pause, has its audio in before it). Two clocks: the page's own figure (the service's wait
// after speech + turn to first sound), and an outside one from where the speech ends in the WAV (capture start + its
// offset) to the first reply buffer's start. The last question trails off, so the turn model waits for more and a reply
// begins at the pause.
//   cd web && bun bench-voice.ts [questions...]     (Chrysalis running; voice mode's settings as a new tab has them)
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import puppeteer from "puppeteer-core";

const ROOT = `${import.meta.dir}/..`, LEAD = 2.5, TAIL = 8;
// The same six questions each run, their subjects drawn anew: Chrysalis keeps a prompt's end with its first token's
// odds, so a question asked again (its reply's prompt, its check's) would come back at once from the cache
const any = (xs: string[]) => xs[Math.floor(Math.random() * xs.length)];
const QUESTIONS = process.argv.slice(2).length ? process.argv.slice(2) : [
  `What's the capital of ${any(["France", "Peru", "Kenya", "Norway", "Japan", "Chile", "Egypt", "Canada", "Poland", "Vietnam", "Ghana", "Spain"])}?`,
  `How many legs does a ${any(["spider", "crab", "beetle", "lobster", "centipede", "ant", "scorpion", "shrimp", "starfish", "mosquito"])} have?`,
  `Give me one quick tip for ${any(["sleeping better", "cooking rice", "learning guitar", "running faster", "saving money", "writing emails", "growing tomatoes"])}.`,
  `What's the difference between a ${any(["virus and a bacterium", "lake and a pond", "frog and a toad", "comet and an asteroid", "violin and a viola", "hurricane and a tornado"])}?`,
  `Can you recommend a good ${any(["sci-fi", "mystery", "fantasy", "history", "horror", "travel", "cooking", "poetry"])} book?`,
  `I was wondering if you could tell me a fun fact about ${any(["octopuses", "penguins", "volcanoes", "honeybees", "glaciers", "owls", "deserts"])} and`];
const dir = mkdtempSync(`${tmpdir()}/chrysalis-bench-voice-`);
writeFileSync(`${dir}/q.json`, JSON.stringify(QUESTIONS));
const synth = Bun.spawnSync([`${ROOT}/data/voice-venv/bin/python`, "-I", "-c", `
import json, sys, wave, numpy as np
sys.path.insert(0, ${JSON.stringify(ROOT)})
from voice.parts import PARTS
tts = PARTS["tts"]["kokoro"](4)
out = []
for i, q in enumerate(json.load(open(sys.argv[1] + "/q.json"))):
    a = tts.synth(q, "am_michael", 1.0)
    pcm = np.concatenate([np.zeros(int(${LEAD} * tts.rate)), a, np.zeros(int(${TAIL} * tts.rate))])
    with wave.open(f"{sys.argv[1]}/q{i}.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(tts.rate); w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    out.append(len(a) / tts.rate)
print(json.dumps(out))
`, dir], { stderr: "inherit" });
const speech: number[] = JSON.parse(synth.stdout.toString().trim().split("\n").at(-1)!);

const web = Bun.spawn([process.execPath, "server.ts"], { cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
  env: { ...process.env, PORT: "0", CHRYSALIS_HOST: "127.0.0.1", NODE_ENV: "production" } });
let out = "";
for (const reader = web.stdout.getReader(), dec = new TextDecoder(); !/chrysalis web: \S+/.test(out); ) {
  const { value, done } = await reader.read();
  if (done) throw new Error(`server.ts exited: ${out}`);
  out += dec.decode(value);
}
const base = out.match(/chrysalis web: (\S+)/)![1].replace(/\/$/, "");

// The outside clock: when the microphone's stream starts (the fake device starts its file there) and when the first
// reply buffer is due to sound, in performance.now() terms.
const HOOKS = () => {
  const w = window as any;
  w.__bench = { gum: 0, audible: 0, turn: 0, wait: 0, pause: 0 };
  const WS = window.WebSocket;
  window.WebSocket = class extends WS {
    constructor(...a: ConstructorParameters<typeof WebSocket>) {
      super(...a);
      this.addEventListener("message", (e) => {
        const b = w.__bench;
        if (typeof e.data === "string") {
          const m = JSON.parse(e.data);
          if (m.type === "turn" && !b.turn) Object.assign(b, { turn: performance.now(), wait: m.wait_ms });
          if (m.type === "pause" && !b.turn) b.pause = performance.now();
        }
      });
    }
  } as typeof WebSocket;
  const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (c) => { const s = await gum(c); w.__bench.gum = performance.now(); return s; };
  const start = AudioBufferSourceNode.prototype.start;
  AudioBufferSourceNode.prototype.start = function (when = 0, ...rest: any[]) {
    const ctx = this.context as AudioContext;
    if (!w.__bench.audible && ctx.sampleRate !== 16000 && !this.loop)          // not the output's keep-alive hush
      w.__bench.audible = performance.now() + Math.max(0, when - ctx.currentTime) * 1000 + (ctx.outputLatency || ctx.baseLatency || 0) * 1000;
    return start.call(this, when, ...rest);
  };
};

const rows: { q: string; page: number; outside: number; heard: string; reply: string }[] = [];
for (const [i, q] of QUESTIONS.entries()) {
  const browser = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: [
    "--no-sandbox", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required",
    `--use-file-for-fake-audio-capture=${dir}/q${i}.wav%noloop`] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });
  await page.evaluateOnNewDocument(HOOKS);
  await page.goto(`${base}/`, { waitUntil: "networkidle0" });
  await page.click("#voice");
  await page.waitForFunction(() => /voice \d/.test(document.querySelector("#thread .msg:last-child .stats")?.textContent ?? ""), { timeout: 60_000 });
  const r = await page.evaluate(() => ({ b: (window as any).__bench, stats: document.querySelector("#thread .msg:last-child .stats")!.textContent!,
    heard: document.querySelector("#thread .msg.user:not(.ghost) .bubble")?.textContent?.trim() ?? "",
    reply: document.querySelector("#thread .msg:last-child .markdown")?.textContent?.trim() ?? "" }));
  const page_s = Number(r.stats.match(/voice (\d+(?:\.\d+)?) s/)![1]), outside = (r.b.audible - (r.b.gum + (LEAD + speech[i]) * 1000)) / 1000;
  rows.push({ q, page: page_s, outside, heard: r.heard, reply: r.reply.slice(0, 70) });
  const b = r.b, ms = (x: number) => String(Math.round(x)).padStart(4);
  console.log(`${page_s.toFixed(2)} s page  ${outside.toFixed(2)} s outside  = wait ${ms(b.wait)} + turn to sound ${ms(b.audible - b.turn)} ms${b.pause ? ` (paused ${ms(b.turn - b.pause)} ms before the turn)` : ""} | heard "${r.heard}" | ${r.reply.slice(0, 50)}`);
  await browser.close();
}
const med = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1];
console.log(`median: ${med(rows.map((r) => r.page)).toFixed(2)} s on the page's clock, ${med(rows.map((r) => r.outside)).toFixed(2)} s outside`);
web.kill();
