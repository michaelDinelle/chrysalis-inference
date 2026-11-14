import { chmodSync, copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import type { Subprocess } from "bun";
import { clean, fold, paint, plain } from "./terminal";
import { MODELS, point } from "./veil";

const { bold, dim, green, yellow, red, gray } = paint;
const ROOT = `${import.meta.dir}/..`, SH = `${ROOT}/chrysalis.sh`;
const API = process.env.CHRYSALIS_URL ?? "http://127.0.0.1:8080", PORT = process.env.PORT ?? "8081";
const WEB_FILE = existsSync(`${import.meta.dir}/server.js`) ? "server.js" : "server.ts";
const ENGINE_CMD: string[] = process.env.CHRYSALIS_ENGINE_CMD ? JSON.parse(process.env.CHRYSALIS_ENGINE_CMD) : [SH, "serve"];
export const MODES = ["local", "network", "insecure"] as const;
export type Mode = (typeof MODES)[number];
export { MODELS };
const SEP = dim(" · "), JSON_BODY = { "Content-Type": "application/json" };

const args = process.argv.slice(2), flag = (name: string) => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined);
export const HAS_MODEL = !args.includes("--no-model");
export interface Ask { key: string; text: string; answer?: (yes: boolean) => void }
export const state = {
  mode: (MODES.find((m) => m === flag("--mode")) ?? "local") as Mode,
  verbose: false,
  incognito: false,
  veiled: 0,
  serving: HAS_MODEL,
  web: undefined as { local: string; network: string[]; api: string[] } | undefined,
  questions: [] as Ask[],
  asks: [] as Ask[],
  opencode: undefined as { state: string; outside: boolean } | undefined,
  engine: { state: "down" as "down" | "loading" | "ready" | "stopped", model: "", progress: 0, step: "loading the weights", since: Date.now(), writing: 0, vram: "", ram: "" },
};
const engine = state.engine;
let quitting = false, version = 0, shown = "";
const listeners = new Set<() => void>();
export const subscribe = (l: () => void) => (listeners.add(l), () => void listeners.delete(l));
export const snapshot = () => version;
function sync() {
  state.serving = !!engineSvc?.proc;
  const now = JSON.stringify(state);
  if (now === shown) return;
  shown = now;
  version++;
  for (const l of listeners) l();
}

export const ui = {
  live: false,
  log: (text: string): void => { process.stdout.write(`${plain(text)}\n`); },
  suspend: (run: () => Promise<unknown>): Promise<unknown> => run(),
  clear: (): void => {},
  close: (): void => {},
};
const note = (text: string, hang = 2) => ui.log(ui.live ? fold(text, (process.stdout.columns || 80) - 3, " ".repeat(hang)).map((l, i) => (i ? l : `  ${l}`)).join("\n") : `  ${text}`);
const count = (x: number | string) => Number(x).toLocaleString("en-US");
let zone = { hour: -1, ms: 0 };
function stamp() {
  const now = new Date();
  if (now.getUTCHours() !== zone.hour) {
    const z = Bun.spawnSync(["date", "+%z"], { stdout: "pipe", stderr: "ignore" }).stdout.toString().trim();
    zone = { hour: now.getUTCHours(), ms: /^[+-]\d{4}$/.test(z) ? (z[0] === "-" ? -1 : 1) * (Number(z.slice(1, 3)) * 60 + Number(z.slice(3))) * 60_000 : -now.getTimezoneOffset() * 60_000 };
  }
  return gray(new Date(now.getTime() + zone.ms).toISOString().slice(11, 19));
}


type Proc = Subprocess<"ignore", "pipe", "pipe">;
class Service {
  proc?: Proc;
  tail: string[] = [];
  private quick = 0;
  private asked = false;
  constructor(readonly name: string, private spawn: () => Proc, private line: (l: string) => boolean, private began: () => void) {}

  start() {
    const t0 = Date.now(), proc = this.spawn();
    this.proc = proc;
    this.asked = false;
    this.tail = [];
    this.began();
    if (state.incognito) dumps(proc, true);
    sync();
    const read = async (stream: ReadableStream<Uint8Array>) => {
      const dec = new TextDecoder();
      let buf = "";
      for await (const chunk of stream) {
        buf += dec.decode(chunk, { stream: true });
        const lines = buf.split(/\r?\n|\r/);
        buf = lines.pop()!;
        for (const l of lines) if (l.trim()) this.seen(l);
      }
      if (buf.trim()) this.seen(buf);
    };
    void Promise.all([read(proc.stdout), read(proc.stderr), proc.exited]).then(() => this.ended(proc, Date.now() - t0));
  }

  private seen(l: string) {
    if (this.line(l)) return;
    if (state.verbose) return ui.log(dim(`  ${this.name}  ${clean(l)}`));
    this.tail.push(l);
    if (this.tail.length > 12) this.tail.shift();
  }

  private ended(proc: Proc, ms: number) {
    if (this.proc !== proc) return;
    this.proc = undefined;
    sync();
    if (this.asked || quitting) return;
    for (const l of this.tail) ui.log(dim(`  ${this.name}  ${clean(l)}`));
    this.tail = [];
    const code = proc.exitCode, how = code === null ? String(proc.signalCode) : `exit ${code}`;
    if (proc.signalCode === "SIGKILL" || code === 137)
      return note(red(`the ${this.name} was killed, most likely by the kernel for memory (journalctl -k shows which): not starting it again, so the desktop keeps its memory · restart starts it once there is room`));
    if (code === 0) return note(`the ${this.name} stopped · restart starts it`);
    this.quick = ms < 120_000 ? this.quick + 1 : 0;
    if (this.quick >= 3) return note(red(`the ${this.name} stopped ${this.quick} times in a row within two minutes of starting (${how}): not starting it again · restart tries once more`));
    note(yellow(`the ${this.name} stopped (${how}): starting it again`));
    setTimeout(() => { if (!this.proc && !this.asked && !quitting) this.start(); }, 3000);
  }

  async stop() {
    this.asked = true;
    const proc = this.proc;
    if (!proc) return;
    proc.kill();
    await proc.exited;
  }

  async restart() {
    await this.stop();
    this.quick = 0;
    this.start();
  }
}

const REQUEST = /^chrysalis: prompt (\d+) \(cached (\d+)\)(?: \| (\d+) pictures encoded)? ttft ([\d.]+)s \| gen (\d+) ([\d.]+) tok\/s [\d.]+\/step \| (\w+) \| slot \d+/;
const STOPPED = /^chrysalis: prompt (\d+) \(cached (\d+)\) \| stopped by the client after (\d+) tokens/;
const FAILED = /^chrysalis: request failed: (.*)/;
const SIZES = /weights ([0-9.]+) GB.*buffers ([0-9.]+) GB.*\(([0-9.]+) GB\)/, COLD = /cold experts ([0-9.]+) GB/;
const LOADING = /^chrysalis: loading (qwen[\w.-]+)$/, SERVING = /^chrysalis: serving (\S+) .*\(context (\d+) tokens\)/;
const SWITCHED = /^chrysalis: switched \S+ -> (\S+) in ([\d.]+) s \(context (\d+) tokens\)/;
const AWAKE = /^chrysalis: awake: (\S+) loaded in ([\d.]+) s/;
const FINISH: Record<string, string> = { length: "cut at its token limit", tool_calls: "a tool call" };

const read = (prompt: string, cached: string) => `${count(prompt)} tokens read${Number(cached) ? dim(` (${count(cached)} cached)`) : ""}`;
function ready(model: string, seconds: string | number, context?: string) {
  const sizes = [context ? `${count(context)} tokens of context` : "", engine.vram ? `${engine.vram} GB VRAM` : "", engine.ram ? `${engine.ram} GB RAM` : ""].filter(Boolean).join(" · ");
  note(`${green("✓")} ${bold(model)} ready in ${seconds} s  ${dim(sizes)}`);
  Object.assign(engine, { state: "ready", model, vram: "", ram: "", step: "loading the weights" });
  engineSvc.tail = [];
  readyAt = Date.now();
}

function engineLine(l: string): boolean {
  let m: RegExpExecArray | null;
  if ((m = REQUEST.exec(l))) {
    if (!state.veiled) note(`${stamp()}  ${read(m[1], m[2])}${m[3] ? `${SEP}${m[3]} pictures` : ""}${SEP}first token ${m[4]} s${SEP}${count(m[5])} written at ${m[6]} tok/s${m[7] === "stop" ? "" : SEP + (FINISH[m[7]] ?? m[7])}`);
    return !state.verbose;
  }
  if ((m = STOPPED.exec(l))) {
    if (!state.veiled) note(`${stamp()}  ${read(m[1], m[2])}${SEP}stopped by the client after ${count(m[3])} written`);
    return !state.verbose;
  }
  if ((m = FAILED.exec(l))) return note(`${stamp()}  ${red(`a request failed: ${clean(m[1])}`)}`), true;
  if ((m = SIZES.exec(l))) {
    engine.vram = (Number(m[1]) + Number(m[2]) + Number(m[3])).toFixed(1);
    engine.ram = COLD.test(l) ? Number(COLD.exec(l)![1]).toFixed(1) : "";
    engine.step = "warming up";
    engine.progress = 1;
    return false;
  }
  if ((m = LOADING.exec(l))) {
    Object.assign(engine, { state: "loading", model: m[1], progress: 0 });
    return false;
  }
  if ((m = SERVING.exec(l))) return ready(m[1], Math.round((Date.now() - engine.since) / 1000), m[2]), !state.verbose;
  if ((m = SWITCHED.exec(l))) return ready(m[1], m[2], m[3]), !state.verbose;
  if ((m = AWAKE.exec(l))) return ready(m[1], m[2]), !state.verbose;
  return false;
}


let engineSvc: Service, webSvc: Service;
engineSvc = new Service("model server", () => Bun.spawn(scoped(ENGINE_CMD), { stdin: "ignore", stdout: "pipe", stderr: "pipe",
  env: { ...process.env, ...(run ? { CHRYSALIS_SNAPSHOTS: `${run}/snapshots`, TMPDIR: `${run}/tmp` } : {}) } }), (l) => { const took = engineLine(l); sync(); return took; },
  () => Object.assign(engine, { state: "down", since: Date.now(), progress: 0, writing: 0 }));
webSvc = new Service("web chat", () => Bun.spawn(scoped([process.execPath, WEB_FILE]), {
  cwd: import.meta.dir, stdin: "ignore", stdout: "pipe", stderr: "pipe",
  env: { ...process.env, NODE_ENV: "production", PORT, CHRYSALIS_HOST: state.mode === "local" ? "127.0.0.1" : "0.0.0.0", CHRYSALIS_INSECURE: state.mode === "insecure" ? "1" : "", CHRYSALIS_MEMORY: run ?? "", CHRYSALIS_KEPT: KEPT },
  ipc: heard,
}), () => false, () => { state.web = undefined; state.questions = []; state.veiled = 0; hidden(false); });

let scope: boolean | undefined;
function scoped(cmd: string[]) {
  if (process.env.CHRYSALIS_ENGINE_CMD) return cmd;
  try { scope ??= Bun.spawnSync(["systemd-run", "--user", "--scope", "-q", "true"], { stdout: "ignore", stderr: "ignore" }).exitCode === 0; } catch { scope = false; }
  return scope ? ["systemd-run", "--user", "--scope", "-q", "-p", "MemorySwapMax=0", ...cmd] : cmd;
}

function heard(m: any) {
  if (m?.up) { state.web = m.up; links(); }
  if (Array.isArray(m?.questions)) {
    const now: Ask[] = m.questions.map((q: Ask) => ({ key: String(q.key), text: clean(String(q.text)) }));
    for (const q of state.questions) if (!now.some((x) => x.key === q.key)) note(dim(`${q.text} · answered elsewhere`));
    state.questions = now;
  }
  if (typeof m?.hidden === "boolean") hidden(m.hidden);
  if (typeof m?.veiled === "number") state.veiled = m.veiled;
  if (typeof m?.from === "string") note(`${stamp()}  ${dim("from")} ${clean(m.from)}`);
  if (typeof m?.note === "string") note(yellow(clean(m.note)));
  sync();
}

export function links() {
  const web = state.web;
  if (!web) return;
  const row = (k: string, v: string, more = "") => note(`${dim(k.padEnd(8))}${v}${more ? `  ${dim(more)}` : ""}`), there = web.network[0];
  ui.log("");
  const code = there && state.mode !== "local" ? scan(there) : undefined;
  if (code) ui.log(`${code}\n`);
  if (state.mode === "insecure" && there) {
    row("chat", there, `${web.local} on this computer`);
    row("API", web.api[0] ?? `${API}/v1`, web.api[0] ? "" : "on this computer only");
    ui.log("");
    note(yellow("Unencrypted: anyone on this network can use the model and read what is sent. Other devices get no voice mode, since browsers give " +
      "a microphone only to encrypted pages, and no OpenCode. network is the encrypted way."));
  } else {
    row("chat", web.local);
    row("API", `${API}/v1`);
    if (state.mode === "network" && there) row("phone", there, code ? "scan the code" : "");
    if (state.mode !== "local" && !there) note(yellow("this computer is on no network now, so only it reaches Chrysalis"));
  }
  ui.log("");
}

let polling = false, readyAt = 0, ticks = 0;
async function poll() {
  if (polling) return;
  polling = true;
  const asked = Date.now();
  try {
    const r = await fetch(`${API}/chrysalis/stats`, { signal: AbortSignal.timeout(900) }), b = await r.json().catch(() => ({}));
    if (asked < readyAt) return;
    if (r.status === 503 || (r.ok && b.loading)) {
      if (engine.state === "ready" || engine.state === "stopped") engine.since = Date.now();
      Object.assign(engine, { state: "loading", model: typeof b.loading === "string" ? b.loading : b.model ?? engine.model, progress: Number(b.progress) || engine.progress });
    } else if (r.ok) {
      Object.assign(engine, { state: b.asleep ? "stopped" : "ready", model: b.model ?? engine.model });
      const metrics = b.asleep ? "" : await fetch(`${API}/metrics`, { signal: AbortSignal.timeout(900) }).then((x) => x.text());
      engine.writing = Number(/^chrysalis_replies_writing (\d+)/m.exec(metrics)?.[1] ?? 0);
    } else engine.state = "down";
  } catch {
    if (asked >= readyAt) engine.state = "down";
  } finally {
    polling = false;
    sync();
  }
}

const HELP: [string, string][] = [
  ["Tab", "the switches: every setting below as a row to move through and set"],
  ["local", "only this computer reaches Chrysalis, the way it starts"],
  ["network", "your network too, encrypted; each phone or computer asks here to be let in"],
  ["insecure", "the network over plain HTTP, like llama.cpp: no certificates, no voice mode, open to anyone on it"],
  ["model [name]", "the model that is loaded; 27b or flash-next loads one, unload frees the GPU"],
  ["restart [part]", "start the model server and the web chat again; model or web for one of them"],
  ["verbose", "also show what the servers print, for this run; again turns it off"],
  ["incognito [end]", "whether incognito chats are open; end forgets them all"],
  ["qr", "the code to scan with a phone, again"],
  ["opencode", "the line that connects OpenCode on another computer"],
  ["links", "the addresses again"],
  ["clear", "clear the screen"],
  ["quit", "stop Chrysalis (Ctrl+C)"],
];
export const WORDS = ["local", "network", "insecure", "model", "restart", "verbose", "incognito", "qr", "opencode", "links", "clear", "quit", "help"];

const here = () => state.web?.local.replace("localhost", "127.0.0.1");
const failure = async (r: Response) => (await r.json().catch(() => null))?.error?.message ?? `it answered ${r.status}`;
const NO_WEB = "the web chat is not running: restart starts it";

export async function setMode(to: Mode) {
  if (to === state.mode) return;
  if (to !== "local" && ui.live) await ui.suspend(() => Bun.spawn([SH, "firewall", to], { stdio: ["inherit", "inherit", "inherit"] }).exited);
  state.mode = to;
  await webSvc.restart();
}

export async function setModel(to: string) {
  if (to === engine.model && engine.state === "ready") return note(`${to} is loaded already`);
  const r = await fetch(`${here()}/api/chrysalis/load`, { method: "POST", body: JSON.stringify({ model: to }), headers: JSON_BODY }).catch(() => null);
  if (!r?.ok) note(red(r ? await failure(r) : NO_WEB));
}

export async function unload() {
  if (engine.state === "stopped") return note("the model is unloaded already");
  const r = await fetch(`${API}/chrysalis/unload`, { method: "POST" }).catch(() => null);
  note(r?.ok ? `${engine.model || "the model"} is unloaded and the GPU is free${SEP}${dim("model loads one again")}` : red("the model server is not running"));
  if (r?.ok) engine.state = "stopped";
}

export function setVerbose(on: boolean) {
  state.verbose = on;
  note(on ? "verbose is on: what the servers and voice print shows here too, for this run" : "verbose is off: requests only");
}

const SHM = "/dev/shm", RUN = "chrysalis-run-", KEPT = process.env.CHRYSALIS_SNAPSHOTS || `${ROOT}/data/cache`;
let run: string | undefined;
const cores = new Map<number, string>();
function wipe(dir: string) {
  Bun.spawnSync(["chmod", "-R", "u+w", dir], { stdout: "ignore", stderr: "ignore" });
  rmSync(dir, { recursive: true, force: true, maxRetries: 3 });
}
function sweep() {
  for (const old of existsSync(SHM) ? readdirSync(SHM) : []) {
    const pid = old.startsWith(RUN) ? Number(old.slice(RUN.length).split("-")[0]) : 0;
    if (!pid) continue;
    try { process.kill(pid, 0); } catch { wipe(`${SHM}/${old}`); }
  }
}
function memory() {
  if (!existsSync(SHM)) return;
  run = mkdtempSync(`${SHM}/${RUN}${process.pid}-`);
  mkdirSync(`${run}/tmp`);
  mkdirSync(`${run}/snapshots`);
  for (const model of MODELS) {
    mkdirSync(`${run}/none/${model}`, { recursive: true });
    if (existsSync(`${KEPT}/${model}/pool.bin`)) copyFileSync(`${KEPT}/${model}/pool.bin`, `${run}/none/${model}/pool.bin`);
    chmodSync(`${run}/none/${model}`, 0o555);
  }
  point(run, KEPT, false);
}
function dumps(proc: { pid: number } | undefined, off: boolean) {
  if (!proc || process.env.CHRYSALIS_ENGINE_CMD) return;
  const limit = (...more: string[]) => { try { return Bun.spawnSync(["prlimit", "--pid", String(proc.pid), ...more], { stdout: "pipe", stderr: "ignore" }).stdout.toString().trim(); } catch { return ""; } };
  if (off) {
    if (!cores.has(proc.pid)) cores.set(proc.pid, limit("--core", "--raw", "--noheadings", "-o", "SOFT"));
    limit("--core=0:");
  } else if (cores.has(proc.pid)) {
    limit(`--core=${cores.get(proc.pid) || "unlimited"}:`);
    cores.delete(proc.pid);
  }
}
function hidden(on: boolean) {
  if (on === state.incognito) return;
  state.incognito = on;
  for (const svc of [engineSvc, webSvc]) dumps(svc.proc, on);
  note(on ? "an incognito chat is open: OpenCode keeps it in memory only, and its requests don't show here" : "the incognito chats are gone");
}
export function forget() {
  if (!state.incognito) return note(dim("no incognito chat is open"));
  webSvc.proc?.send({ forget: true });
}

export async function restart(part: string) {
  if (part !== "web" && HAS_MODEL) await engineSvc.restart();
  if (part !== "model") await webSvc.restart();
}

function scan(url: string): string | undefined {
  try {
    const code = Bun.spawnSync(["qrencode", "-t", "ANSIUTF8", "-m", "2", url], { stdout: "pipe", stderr: "ignore" });
    return code.exitCode === 0 ? code.stdout.toString().trimEnd().split("\n").map((l) => `  ${l}`).join("\n") : undefined;
  } catch {
    return undefined;
  }
}

export function qr() {
  const url = state.web?.network[0];
  if (!url) return note(`a code is for other devices${SEP}${dim("network or insecure puts Chrysalis on your network")}`);
  const code = scan(url);
  if (!code) return note(`qrencode is not installed${SEP}${url}`);
  ui.log(code);
  note(dim(url));
}

export async function opencodeLine() {
  if (state.mode !== "network") return note(`OpenCode on another computer comes in over the encrypted network${SEP}${dim("network first")}`);
  const got = await fetch(`${here()}/api/opencode`).then((r) => r.json()).catch(() => null);
  if (!got?.line) return note(red(NO_WEB));
  note(dim("Run this on the other computer. This terminal then asks to let it in."));
  ui.log(got.line);
}

export async function readOpencode() {
  const got = await fetch(`${here()}/api/threads`).then((r) => r.json()).catch(() => null);
  state.opencode = got && got.state !== "absent" ? { state: String(got.state), outside: !!got.outside } : undefined;
  sync();
}

export async function setOutside(on: boolean) {
  const send = async (method: string) => {
    const r = await fetch(`${here()}/api/threads/outside`, { method, body: "{}", headers: JSON_BODY }).catch(() => null);
    note(r?.ok ? (on ? "OpenCode may use models outside this computer" : "OpenCode is kept to Chrysalis's models") : red(r ? await failure(r) : NO_WEB));
    await readOpencode();
  };
  if (!on) return send("DELETE");
  if (state.asks.some((a) => a.key === "outside")) return;
  state.asks.push({ key: "outside", text: "A model outside this computer gets your prompts, your files and what its tools read here. Allow them?",
    answer: (yes) => void (yes ? send("POST") : note(dim("kept to Chrysalis's models"))) });
}

function answer(yes: boolean) {
  const q = state.asks.shift() ?? state.questions.shift();
  if (!q) return note(dim("nothing is asking"));
  if (q.answer) return q.answer(yes);
  webSvc.proc?.send({ answer: { key: q.key, yes } });
  note(yes ? `${green("✓")} let in` : dim("refused"));
}

async function command(line: string) {
  const [cmd, ...rest] = line.replace(/^\//, "").toLowerCase().split(/\s+/), arg = rest.join(" ");
  if (/^(y|yes|n|no)$/.test(cmd)) return answer(cmd[0] === "y");
  switch (cmd) {
    case "help": case "?":
      ui.log("");
      for (const [k, v] of HELP) if (ui.live || k !== "Tab") note(`${k.padEnd(16)}${dim(v)}`, 18);
      ui.log("");
      note(dim("Requests show as they finish: their sizes and speeds, never their words. Nothing is written to disk."));
      return ui.log("");
    case "local": case "network": case "insecure": return cmd === state.mode ? links() : setMode(cmd);
    case "links": return links();
    case "clear": return ui.clear();
    case "quit": case "exit": case "q": return quit();
    case "verbose": return setVerbose(arg ? arg === "on" : !state.verbose);
    case "incognito": return arg === "end" ? forget() : note(state.incognito ? `an incognito chat is open${SEP}${dim("incognito end forgets them all")}` : `no incognito chat is open${SEP}${dim("the ghost button in the page starts one")}`);
    case "model": {
      if (!arg) return note(`${engine.model || "no model"} is ${engine.state === "stopped" ? "unloaded" : "loaded"}${SEP}${dim("model 27b, model flash-next or model unload")}`);
      if (/^(unload|none|off)$/.test(arg)) return unload();
      const to = /27/.test(arg) ? MODELS[0] : /flash|next|fn/.test(arg) ? MODELS[1] : undefined;
      return to ? setModel(to) : note(`no model ${arg}${SEP}${dim("27b, flash-next or unload")}`);
    }
    case "restart": return arg && arg !== "model" && arg !== "web" ? note("restart, restart model or restart web") : restart(arg);
    case "qr": return qr();
    case "opencode": return opencodeLine();
  }
  note(`no command ${clean(cmd)}${SEP}${dim("help lists them")}`);
}

let queue = Promise.resolve();
export const act = (run: () => unknown) => { queue = queue.then(run).then(() => {}, (e) => note(red(e instanceof Error ? e.message : String(e)))).then(sync); };
export const submit = (line: string) => act(() => command(line));

let getting: Subprocess | undefined;
export function quit() {
  if (quitting) return;
  quitting = true;
  ui.close();
  for (const p of [getting, engineSvc.proc, webSvc.proc]) p?.kill();
  if (run) wipe(run);
  process.exit(0);
}

export function start() {
  for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"] as const) process.on(sig, quit);
  process.on("exit", () => ui.close());
  setInterval(() => { if (ticks++ % (engine.state === "ready" || engine.state === "stopped" ? 4 : 1) === 0) void poll(); }, 250);
  sweep();
  memory();
  ui.log("");
  note(bold("Chrysalis Inference"));
  if (flag("--get")) {
    getting = Bun.spawn([SH, "get", flag("--get")!], { stdin: "ignore", stdout: "ignore", stderr: "ignore" });
    note(dim(`${flag("--get")} is downloading in the background: model has it once it is here`));
  }
  if (HAS_MODEL) engineSvc.start();
  else note(dim("No GPU Chrysalis runs on here: the web chat uses Chrysalis on another computer on your network."));
  webSvc.start();
}
