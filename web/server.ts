import { existsSync, rmSync, statSync, writeFileSync, readFileSync } from "node:fs";
import { homedir, hostname } from "node:os";
import type { BunPlugin, Server, Subprocess } from "bun";
import app from "./index.html";
import { Backends, standard } from "./backend";
import { Devices, keyCookie } from "./devices";
import { advertise, answerName, Link } from "./link";
import { opencodeLine, opencodeManifest, opencodeScript } from "./opencode";
import { tunnel } from "./machines";
import { HOME, MODELS, python } from "./platform";
import { qrSvg as qrCode } from "./qr";
import { appleProfile, connectPage, deviceScript, TRUSTED_GIF } from "./setup";
import * as harness from "./harness";
import { fingerprint, lanAddresses, netName, routeAddress, tlsDir, webTls } from "./tls";
import { assistantTurn, checkpointText, withText, type WireCall } from "./wire";
import { ceiling, CONTINUE_AFTER_BREAK, covers, filled, NUDGE, recentText, saysTooLong, summaryPrompt, summaryRoom, tokens, tokensIn, type Turn } from "./compaction";

const CHRYSALIS = process.env.CHRYSALIS_URL ?? "http://127.0.0.1:8080";
const MODEL_KEY = crypto.randomUUID();
const engineStats = () => fetch(`${CHRYSALIS}/chrysalis/stats`, { signal: AbortSignal.timeout(1000) });
const LIVE = import.meta.path.endsWith(".ts");
const PAGE_FILES = ["index.html", "app.ts", "sampler.ts", "files.ts", "wire.ts", "voice.ts", "voice-worklet.js", "sandbox.js", "heard.ts", "spoken.ts", "ocean.ts", "logo.svg", "style.css", "tokens.css"]
  .map((f) => `${import.meta.dir}/${f}`);
const BUILT = Bun.hash((await Promise.all(PAGE_FILES.map((f) => Bun.file(f).text()))).join("\0")).toString(36);
const BUILD = () => LIVE ? Bun.hash(PAGE_FILES.map((f) => { const at = statSync(f); return `${at.mtimeMs}:${at.size}`; }).join("\0")).toString(36) : BUILT;
for (const sig of ["SIGINT", "SIGTERM"] as const) process.on(sig, () => process.exit(0));
process.on("disconnect", () => process.exit(0));
const tell = (m: object) => void process.send?.(m);

const NAMES = new Set(["", "localhost", hostname().toLowerCase(), `${hostname().toLowerCase()}.local`, netName()]);
const hostOk = (req: Request) => {
  const host = (req.headers.get("host") ?? "").toLowerCase();
  return host.startsWith("[") || /^\d{1,3}(\.\d{1,3}){3}(:\d+)?$/.test(host) || NAMES.has(host.replace(/:\d+$/, ""));
};
type Srv = Server<VoiceSocket>;
type Handler = (req: Request, srv: Srv) => Response | undefined | Promise<Response>;
const open = (handler: Handler) => (req: Request, srv: Srv) => (hostOk(req) ? handler(req, srv) : new Response("unknown host", { status: 403 }));
const SELF_AS_NETWORK = process.env.CHRYSALIS_SELF_AS_NETWORK === "1", OPEN = process.env.CHRYSALIS_INSECURE === "1";
const self = (req: Request, srv: Srv) => {
  const at = srv.requestIP(req)?.address.replace(/^::ffff:/, "") ?? "";
  return at.startsWith("127.") || at === "::1" || (!SELF_AS_NETWORK && lanAddresses().includes(at));
};
const fromNetwork = (req: Request, srv: Srv) => !self(req, srv);
const PAST_WARNING = { Connection: "close" };
const unkept = async (res: Response | undefined | Promise<Response>) => {
  const r = await res;
  if (r && !r.headers.has("Cache-Control")) try { r.headers.set("Cache-Control", "no-store"); } catch { }
  return r as Response;
};
const guarded = (handler: Handler) => open((req, srv) => {
  if (OPEN || !fromNetwork(req, srv)) return unkept(handler(req, srv));
  const found = devices?.find(req);
  if (found && devices!.trusted(found.device)) return unkept(handler(req, srv));
  return Response.json(found ? { error: { message: "this browser has not shown lately that it trusts Chrysalis: its page checks again" }, check: true }
    : { error: { message: "this device is not connected to Chrysalis: its page at / connects it" } }, { status: 401 });
});
const here = (handler: Handler) => guarded((req, srv) => fromNetwork(req, srv)
  ? Response.json({ error: { message: "only on the computer Chrysalis runs on" } }, { status: 403 }) : handler(req, srv));
const KEPT_HERE = "Chrysalis is on the network without encryption, so OpenCode and voice stay on the computer it runs on";
const trusted = (handler: Handler) => guarded((req, srv) => OPEN && fromNetwork(req, srv)
  ? Response.json({ error: { message: KEPT_HERE } }, { status: 403 }) : handler(req, srv));
const ASKS = /\/(chat\/completions|messages|responses)$/;
const arrived = (req: Request, srv: Srv) => {
  if (fromNetwork(req, srv)) tell({ from: devices?.find(req)?.device.name ?? srv.requestIP(req)?.address.replace(/^::ffff:/, "") ?? "the network" });
};
const json = (handler: (req: Request, srv: Srv) => Response | Promise<Response>) => (req: Request, srv: Srv) =>
  (req.headers.get("content-type") ?? "").startsWith("application/json") ? handler(req, srv) : new Response("JSON only", { status: 415 });

const MODEL_FILE = `${tlsDir()}/model`, OPENCODE_OFF = `${tlsDir()}/opencode-off`, OPENCODE_OUTSIDE = `${tlsDir()}/opencode-outside`;
harness.allowOutside(existsSync(OPENCODE_OUTSIDE));
async function pickModel(req: Request): Promise<string | undefined> {
  const model = (await req.json().catch(() => ({})))?.model;
  if (!MODELS.includes(model)) return undefined;
  writeFileSync(MODEL_FILE, model);
  return model;
}
const engineCall = (path: string, body: object = {}) =>
  fetch(`${CHRYSALIS}${path}`, { method: "POST", body: JSON.stringify(body), headers: { "Content-Type": "application/json" } });

interface Job {
  id: string;
  events: (string | null)[];
  folds: { from: number; to: number }[];
  folded: number;
  done: boolean;
  ctrl: AbortController;
  waiters: Set<() => void>;
  asks: Map<string, (d: harness.Answer) => void>;
  thread?: string;
  t0: number; tFirst?: number; tAnswer?: number; tLast?: number;
  unkept?: boolean;
  gen: number;
}
const jobs = new Map<string, Job>(), turns = new Map<string, Job>();
const KEEP_MS = 15 * 60_000;
const RETRIES = 10, RETRY_MS = Number(process.env.CHRYSALIS_RETRY_MS) || 2000, RETRY_MOST_MS = RETRY_MS * 5;
const SUMMARY_THINKING = 2048, WORTH_IT = 4;
const ROOM_MS = 2000;
const FIRST_ONLY = ["continue_final_message", "chrysalis_original", "chrysalis_append_token"];

function startJob(body: string, over?: () => void): Job {
  const job: Job = { id: crypto.randomUUID(), events: [], folds: [], folded: 0, done: false, ctrl: new AbortController(), waiters: new Set(), asks: new Map(),
    t0: Date.now(), gen: 0, unkept: !!over };
  jobs.set(job.id, job);
  const push = (data: string | object) => {
    if (typeof data !== "string" && (data as any).round) fold(job);
    job.events.push(typeof data === "string" ? data : JSON.stringify(data));
    for (const wake of [...job.waiters]) wake();
  };
  push({ started: job.t0 });
  let main: Thread | undefined;
  (async () => {
    try {
      const raw = JSON.parse(body);
      if (raw.chrysalis_thread) {
        const ask = (call: string) => new Promise<harness.Answer>((resolve) => {
          if (job.ctrl.signal.aborted) return resolve("stop");
          job.asks.set(call, (d) => (job.asks.delete(call), resolve(d)));
          job.ctrl.signal.addEventListener("abort", () => (job.asks.delete(call), resolve("stop")), { once: true });
        });
        await harness.ensure(CHRYSALIS);
        const before = raw.chrysalis_thread.session && turns.get(raw.chrysalis_thread.session);
        if (before) {
          before.ctrl.abort();
          await new Promise<void>((over) => {
            const check = () => { if (before.done || before.events.at(-1) === "[DONE]") { before.waiters.delete(check); over(); } };
            before.waiters.add(check);
            check();
          });
        }
        if (raw.chrysalis_thread.again && raw.chrysalis_thread.session) {
          await harness.settled(raw.chrysalis_thread.session, job.ctrl.signal);
          await harness.again(raw.chrysalis_thread.session);
        }
        let from = 0, to = 0;
        const first = (d: any) => {
          if (d?.choices?.[0]?.delta) { const now = Date.now(); job.tFirst ??= now; job.tLast = to = now; from ||= now; if (d.choices[0].delta.content) job.tAnswer ??= now; }
          if (d?.usage) { job.gen += to - from; from = to = 0; }
          push(d);
        };
        await harness.turn(raw.chrysalis_thread, first, job.ctrl.signal, ask, (id) => { job.thread = id; turns.set(id, job); push({ thread: id }); });
        return;
      }
      const req = asSent(raw);
      main = thread(job, req, req.messages ?? [], push, job.ctrl.signal, true);
      await step(main);
    } catch (e) {
      if (!job.ctrl.signal.aborted && !(e instanceof ChrysalisDown || e instanceof Refused || e instanceof ChrysalisBroke || e instanceof CannotGoOn)) console.error(`chrysalis web: reply ${job.id} failed:`, e);
      push(job.ctrl.signal.aborted ? { stopped: true } : { error: { message: e instanceof Error ? e.message : String(e) } });
    } finally {
      if (main) threads.delete(main);
      if (job.thread && turns.get(job.thread) === job) turns.delete(job.thread);
      push({ timing: { start: job.t0, first: job.tFirst, answer: job.tAnswer, last: job.tLast, gen: job.gen } });
      push("[DONE]");
      job.done = true;
      over?.();
      setTimeout(() => jobs.delete(job.id), KEEP_MS);
    }
  })();
  return job;
}

const wordsOnly = (ev: any) => {
  const c = ev?.choices?.[0];
  if (!c || c.finish_reason || ev.usage || ev.chrysalis_prompt) return false;
  const { role, content, reasoning_content, ...rest } = c.delta ?? {};
  return !Object.keys(rest).length;
};
function fold(job: Job) {
  const end = job.events.length, words: number[] = [];
  let reasoning = "", content = "";
  for (let i = job.folded; i < end; i++) {
    const raw = job.events[i];
    if (!raw?.includes('"delta"')) continue;
    const ev = JSON.parse(raw);
    if (!wordsOnly(ev)) continue;
    reasoning += ev.choices[0].delta.reasoning_content ?? "";
    content += ev.choices[0].delta.content ?? "";
    words.push(i);
  }
  job.folded = end + 1;
  if (words.length < 2) return;
  const last = words.pop()!;
  for (const i of words) job.events[i] = null;
  job.events[last] = JSON.stringify({ choices: [{ index: 0, delta: { ...(reasoning ? { reasoning_content: reasoning } : {}), ...(content ? { content } : {}) } }] });
  job.folds.push({ from: words[0], to: last });
}

const connector: harness.Connector = harness;
harness.onWake((session) => { if (!turns.has(session)) startJob(JSON.stringify({ chrysalis_thread: { session, dir: "", text: "", follow: true } })); });

const asSent = (raw: any) => (backends.isChrysalis ? raw : standard(raw));

class ChrysalisDown extends Error {
  message = backends.atService ? "Chrysalis is not running" : `nothing answers at ${new URL(backends.url).origin}: is the backend running?`;
}
interface Out { content: string; reasoning: string; calls: WireCall[]; finish?: string; usage?: { prompt_tokens: number; completion_tokens: number } }
class ChrysalisBroke extends Error {
  message = backends.isChrysalis ? "Chrysalis's reply broke off (Chrysalis's log has why)" : "the backend's reply broke off";
  constructor(readonly partial: Out) { super(); }
}
class Refused extends Error {
  constructor(message: string, readonly status: number, readonly code?: string) { super(message); }
}
class CannotGoOn extends Error {}
const refusal = async (r: Response) => {
  const e = (await r.json().catch(() => null))?.error;
  return new Refused((typeof e === "string" ? e : e?.message) ?? `${backends.isChrysalis ? "Chrysalis" : new URL(backends.url).host} answered ${r.status}`, r.status, e?.code);
};
const memoryFull = (e: unknown) => e instanceof Refused && /KV cache full/.test(e.message);
const tooLong = (e: unknown) => e instanceof Refused && (e.code === "context_length_exceeded" || saysTooLong(e.message));
const passing = (e: unknown) => e instanceof ChrysalisDown || e instanceof ChrysalisBroke || (e instanceof Refused && (e.status >= 500 || e.status === 429 || e.status === 408));
async function round(job: Job, body: object, push: (d: string | object) => void, signal = job.ctrl.signal, timed = true): Promise<Out> {
  const r = await fetch(`${backends.url}/chat/completions`, {
    method: "POST", body: JSON.stringify(body), headers: { ...backends.headers(), ...(job.unkept ? UNKEPT : {}) }, signal,
  }).catch((e) => { throw signal.aborted ? e : new ChrysalisDown(); });
  if (!r.ok || !r.body) throw await refusal(r);
  const out: Out = { content: "", reasoning: "", calls: [] };
  let first: number | undefined, last: number | undefined;
  const reader = r.body.getReader(), decoder = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { value, done } = await reader.read().catch((e) => { throw signal.aborted ? e : new ChrysalisBroke(out); });
      if (done) break;
      buf += decoder.decode(value, { stream: true }).replace(/\r\n?/g, "\n");
      for (let end = buf.indexOf("\n\n"); end >= 0; end = buf.indexOf("\n\n")) {
        let data = buf.slice(0, end).split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(l.startsWith("data: ") ? 6 : 5)).join("\n");
        buf = buf.slice(end + 2);
        if (!data || data === "[DONE]") continue;
        const now = Date.now(), ev = JSON.parse(data), choice = ev.choices?.[0], delta = choice?.delta;
        if (ev.error) throw new Refused(ev.error.message ?? String(ev.error), ev.error.status ?? 500, ev.error.code);
        if (typeof delta?.reasoning === "string" && !backends.isChrysalis) {
          delta.reasoning_content ??= delta.reasoning;
          delete delta.reasoning;
          data = JSON.stringify(ev);
        }
        if ((delta?.content || delta?.reasoning_content || delta?.tool_calls) && timed) {
          job.tFirst ??= now;
          job.tLast = last = now;
          first ??= now;
          if (delta.content) job.tAnswer ??= now;
        }
        if (delta?.chrysalis_prefix) {
          out.content = delta.chrysalis_prefix.content ?? "";
          out.reasoning = delta.chrysalis_prefix.reasoning_content ?? "";
        }
        out.content += delta?.content ?? "";
        out.reasoning += delta?.reasoning_content ?? "";
        for (const c of delta?.tool_calls ?? []) {
          const at = (out.calls[c.index ?? out.calls.length] ??= { id: "", name: "", args: "" });
          if (c.id) at.id = c.id;
          if (c.function?.name) at.name += c.function.name;
          if (c.function?.arguments) at.args += c.function.arguments;
        }
        out.finish = choice?.finish_reason ?? out.finish;
        out.usage = ev.usage ?? out.usage;
        push(data);
      }
    }
  } finally {
    if (first && last) job.gen += last - first;
  }
  return out;
}

interface Thread { job: Job; req: any; messages: Turn[]; task?: Turn; head: number; push: (d: string | object) => void; signal: AbortSignal; timed: boolean;
  counted: number; read: number; settled: boolean; asking: number }
const threads = new Set<Thread>();
function thread(job: Job, req: any, messages: Turn[], push: (d: string | object) => void, signal: AbortSignal, timed: boolean) {
  const t: Thread = { job, req, messages, task: messages.findLast((m) => m.role === "user"), head: messages[0]?.role === "system" ? 1 : 0, push, signal, timed,
    counted: 0, read: 0, settled: false, asking: 0 };
  threads.add(t);
  return t;
}
const add = (t: Thread, ...turns: Turn[]) => { t.messages = [...t.messages, ...turns]; };
const sizeOf = (t: Thread) => t.counted + t.messages.slice(t.read).reduce((n, m) => n + tokensIn(m), 0);
const pause = (ms: number, signal: AbortSignal) => new Promise<void>((done, stop) => {
  const gone = () => (clearTimeout(timer), stop(new DOMException("stopped", "AbortError"))), timer = setTimeout(() => (signal.removeEventListener("abort", gone), done()), ms);
  if (signal.aborted) gone();
  else signal.addEventListener("abort", gone, { once: true });
});
const gap = (n: number) => Math.ceil(Math.min(RETRY_MS * 2 ** n, RETRY_MOST_MS) * (1 + Math.random() * 0.2));

interface Room { window?: number; pool?: number; lanes?: number }
const rooms = new Map<string, { at: number; room: Promise<Room> }>();
async function readRoom(model: string): Promise<Room> {
  if (!backends.isChrysalis) return { window: await backends.windowOf(model) };
  const s = await fetch(`${new URL(backends.url).origin}/chrysalis/stats`, { signal: AbortSignal.timeout(ROOM_MS) }).then((r) => r.json());
  return { window: s.context, pool: s.kv_tokens, lanes: s.cache?.slots?.length };
}
function roomOf(t: Thread): Promise<Room> {
  const model = String(t.req.model ?? ""), seen = rooms.get(model);
  if (seen && Date.now() - seen.at < ROOM_MS) return seen.room;
  const room = readRoom(model).catch(() => seen?.room ?? {});
  rooms.set(model, { at: Date.now(), room });
  return room;
}
async function limitOf(t: Thread) {
  const { window, pool, lanes } = await roomOf(t);
  if (!window) return Infinity;
  if (!pool || !lanes) return ceiling(window);
  const sharing = [...threads];
  const others = sharing.filter((x) => x !== t).map(sizeOf).sort((a, b) => b - a).slice(0, lanes - 1).reduce((a, b) => a + b, 0);
  return ceiling(Math.min(window, Math.max(pool - others, Math.floor(pool / Math.max(1, Math.min(lanes, sharing.length))))));
}
async function patient(t: Thread, send: () => Promise<Out>, keep: boolean): Promise<Out> {
  const asked = async () => {
    t.asking++;
    try { return await send(); } finally { t.asking--; }
  };
  for (let failed = 0, waited = 0; ;) {
    try {
      return await asked();
    } catch (e) {
      if (t.signal.aborted) throw e;
      if (memoryFull(e) && (waited < RETRIES || [...threads].some((x) => x !== t && x.asking))) {
        t.push({ waiting: "Waiting for memory: other replies are using it" });
        await pause(gap(waited++), t.signal);
        continue;
      }
      if (!passing(e) || failed >= RETRIES) throw e;
      if (keep && e instanceof ChrysalisBroke) carry(t, e.partial, CONTINUE_AFTER_BREAK);
      const ms = gap(failed++);
      t.push({ retry: { attempt: failed, ms, message: (e as Error).message } });
      await pause(ms, t.signal);
    }
  }
}
function begun(t: Thread) {
  if (t.req.continue_final_message) t.messages = t.messages.slice(0, -1);
  for (const f of FIRST_ONLY) delete t.req[f];
}
function carry(t: Thread, out: Out, note: string) {
  if (out.content.trim()) {
    begun(t);
    add(t, assistantTurn(out.content, out.reasoning), { role: "user", content: note });
  }
  t.push({ round: true });
}

async function step(t: Thread): Promise<Out> {
  let rejected = false;
  for (;;) {
    if (!t.settled && sizeOf(t) >= await limitOf(t)) await compact(t, false);
    let out: Out;
    try {
      out = await patient(t, () => round(t.job, { ...t.req, messages: t.messages }, t.push, t.signal, t.timed), true);
    } catch (e) {
      if (!tooLong(e)) throw e;
      if (rejected) throw new CannotGoOn("what this reply has to keep whole is larger than the context of the model");
      rejected = true;
      t.settled = false;
      await compact(t, true);
      continue;
    }
    t.settled = false;
    if (out.usage) { t.counted = out.usage.prompt_tokens + out.usage.completion_tokens; t.read = t.messages.length + (t.req.continue_final_message ? 0 : 1); }
    begun(t);
    return out;
  }
}

function fitted(rest: Turn[], fit: number) {
  let used = 0, end = 0;
  for (let i = 0; i < rest.length; i++) {
    used += tokensIn(rest[i]);
    if (rest[i + 1]?.role === "tool") continue;
    if (used > fit && end) break;
    end = i + 1;
  }
  return rest.slice(0, end);
}
async function summarize(t: Thread, messages: Turn[], update: boolean) {
  const { max_tokens, tool_choice, stop, seed, logprobs, top_logprobs, ...req } = t.req;
  for (const f of FIRST_ONLY) delete req[f];
  const last = messages.at(-1)!, ask = (text: string) => last.role === "user" ? [...messages.slice(0, -1), withText(last, text, false)] : [...messages, { role: "user", content: text }];
  const send = (text: string) => patient(t, () => round(t.job, { ...req, ...(backends.isChrysalis ? { thinking_budget: SUMMARY_THINKING } : {}), messages: ask(text) }, () => {}, t.signal, false), false);
  const prompt = summaryPrompt(update);
  let out = await send(prompt);
  if (!filled(out.content) && out.finish !== "length") out = await send(`${prompt}\n\n${NUDGE}`);
  if (out.finish === "length") throw new Refused("the summary ran out of room", 400, "context_length_exceeded");
  if (!filled(out.content)) throw new CannotGoOn("the model did not write a summary of this chat when its context filled up, so it cannot go on");
  return out.content.trim();
}
async function compact(t: Thread, needed: boolean) {
  const m = t.messages, head = m.slice(0, t.head);
  let tail = m.length - 1;
  while (tail > t.head && m[tail].role === "tool") tail--;
  if (m[tail] !== t.task && m[tail]?.role === "user" && m[tail - 1]?.role === "assistant") tail--;
  const first = m[tail] === t.task, old = m.slice(t.head, tail);
  if (!needed && old.reduce((n, x) => n + tokensIn(x), 0) - (first || !t.task ? 0 : tokensIn(t.task)) < await limitOf(t) / WORTH_IT) return;
  if (!old.length) throw new CannotGoOn("this message is larger than the whole context of the model, so there is nothing to summarize");
  t.push({ compacting: true });
  const { window } = await roomOf(t);
  const fixed = head.reduce((n, x) => n + tokensIn(x), 0);
  let fit = window ? summaryRoom(window) : Infinity, rest = old, summary = "";
  while (rest.length) {
    const part = fitted(rest, fit - fixed - tokens(summary));
    const carried = !summary ? part : part[0].role === "user" ? [withText(part[0], checkpointText({ summary, recent: "", first: true }), true), ...part.slice(1)]
      : [{ role: "user", content: checkpointText({ summary, recent: "", first: true }) }, ...part];
    try {
      summary = await summarize(t, [...head, ...carried], !!summary || part.some(covers));
    } catch (e) {
      if (!tooLong(e)) throw e;
      fit = Math.floor(part.reduce((n, x) => n + tokensIn(x), fixed + tokens(summary)) * 0.7);
      if (fitted(rest, fit - fixed - tokens(summary)).length >= part.length) throw new CannotGoOn("one message of this chat is larger than the whole context of the model, so it cannot be summarized");
      continue;
    }
    rest = rest.slice(part.length);
  }
  const limit = await limitOf(t), recent = recentText(old, limit === Infinity ? sizeOf(t) : limit), checkpoint = { summary, recent, first };
  const kept = first || t.req.continue_final_message ? 0 : 1;
  t.messages = [...head, withText(t.task ?? { role: "user" }, checkpointText(checkpoint), first), ...(first ? [] : m.slice(tail))];
  t.counted = t.read = 0;
  t.settled = true;
  t.push({ compacted: { ...checkpoint, kept } });
}

function follow(job: Job, from: number): Response {
  let i = Math.max(0, from), checked = 0, waiter: (() => void) | undefined;
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async pull(ctl) {
      while (i >= job.events.length && !job.done)
        await new Promise<void>((resolve) => { waiter = () => { job.waiters.delete(waiter!); resolve(); }; job.waiters.add(waiter); });
      const inside = job.folds.slice(checked).some((f) => f.from < i && i <= f.to);
      checked = job.folds.length;
      if (inside) {
        ctl.enqueue(encoder.encode(`id: 0\ndata: {"reset":true}\n\n`));
        i = 0;
      }
      if (i < job.events.length) {
        const end = job.events.length;
        ctl.enqueue(encoder.encode(job.events.slice(i, end).flatMap((d, k) => (d === null ? [] : [`id: ${i + k}\ndata: ${d}\n\n`])).join("")));
        i = end;
      } else ctl.close();
    },
    cancel() { if (waiter) job.waiters.delete(waiter); },
  });
  return new Response(stream, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Chrysalis-Job": job.id } });
}

const MERMAID = `${import.meta.dir}/node_modules/mermaid/dist/`;
const vendor = (req: Request) => {
  const file = new URL(req.url).pathname.slice("/vendor/mermaid/".length);
  if (!/^(mermaid\.esm\.min\.mjs|chunks\/mermaid\.esm\.min\/[\w-]+\.mjs)$/.test(file)) return new Response("not found", { status: 404 });
  return new Response(Bun.file(MERMAID + file), { headers: { "Content-Type": "text/javascript", "Cache-Control": "max-age=86400" } });
};

const THREE_DIR = `${import.meta.dir}/node_modules/three/`;
const AFRAME = `${import.meta.dir}/node_modules/aframe/dist/aframe-v1.8.0.min.js`;
const bundles = new Map<string, Promise<string | null>>();
let threeNames: Promise<string[]> | undefined;
function virtual(entry: string, shim?: string[]): BunPlugin {
  return { name: "chrysalis-three", setup(b) {
    b.onResolve({ filter: /^chrysalis:entry$/ }, () => ({ path: "entry", namespace: "chrysalis" }));
    if (shim) b.onResolve({ filter: /^three$/ }, () => ({ path: "three", namespace: "chrysalis" }));
    b.onLoad({ filter: /.*/, namespace: "chrysalis" }, (a) => ({ loader: "js", contents: a.path === "entry" ? entry
      : `const T = window.THREE; export const { ${shim!.join(", ")} } = T; export default T;` }));
  } };
}
function bundled(file: string): Promise<string | null> {
  let made = bundles.get(file);
  if (made) return made;
  const build = async (o: Parameters<typeof Bun.build>[0]) => {
    const out = await Bun.build({ target: "browser", minify: true, ...o });
    return out.success ? await out.outputs[0].text() : null;
  };
  const addon = file.match(/^(addons|classic)\/(.+)$/), path = addon && `${THREE_DIR}examples/jsm/${addon[2]}`;
  if (file === "three.module.js") made = build({ entrypoints: [`${THREE_DIR}build/three.module.js`], format: "esm" });
  else if (file === "three.min.js") made = build({ entrypoints: ["chrysalis:entry"], format: "iife", plugins: [virtual(`import * as T from ${JSON.stringify(`${THREE_DIR}build/three.module.js`)}; window.THREE = Object.assign({}, T);`)] });
  else if (addon?.[1] === "addons") made = build({ entrypoints: [path!], format: "esm", external: ["three"] });
  else if (addon) made = (threeNames ??= import(`${THREE_DIR}build/three.module.js`).then((t) => Object.keys(t))).then((names) =>
    build({ entrypoints: ["chrysalis:entry"], format: "iife", plugins: [virtual(`import * as M from ${JSON.stringify(path)}; Object.assign(window.THREE, M);`, names)] }));
  else return Promise.resolve(null);
  made = made.catch(() => null);
  bundles.set(file, made);
  return made;
}
const vendor3d = async (req: Request) => {
  const path = new URL(req.url).pathname;
  if (path === "/vendor/aframe/aframe.min.js") return new Response(Bun.file(AFRAME), { headers: { "Content-Type": "text/javascript", "Cache-Control": "max-age=86400" } });
  const file = path.slice("/vendor/three/".length);
  const known = /^(three\.module\.js|three\.min\.js|(addons|classic)\/[\w-]+(\/[\w-]+)*\.js)$/.test(file) &&
    (!/^(addons|classic)\//.test(file) || (await Bun.file(`${THREE_DIR}examples/jsm/${file.slice(file.indexOf("/") + 1)}`).exists()));
  const code = known ? await bundled(file) : null;
  return code === null ? new Response("not found", { status: 404 })
    : new Response(code, { headers: { "Content-Type": "text/javascript", "Cache-Control": "max-age=86400" } });
};

const sandboxPage = async () => `<!doctype html><meta charset="utf-8"><script>${await Bun.file(`${import.meta.dir}/sandbox.js`).text()}</script>`;
const SANDBOX = await sandboxPage();
const SANDBOX_CSP = "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' blob:; style-src 'unsafe-inline'; img-src data: blob:; " +
  "font-src data:; media-src data: blob:; form-action 'none'; base-uri 'none'; frame-ancestors 'self'; sandbox allow-scripts allow-modals allow-pointer-lock allow-orientation-lock";

const VOICE_CMD: string[] = process.env.CHRYSALIS_VOICE_CMD ? JSON.parse(process.env.CHRYSALIS_VOICE_CMD) : [python(), "-m", "chrysalis.voice"];
const VOICE_IDLE_MS = Number(process.env.CHRYSALIS_VOICE_IDLE_MS ?? 60_000);
interface VoiceSocket { up?: WebSocket; queue: (string | Buffer<ArrayBuffer>)[]; closed: boolean; machine?: { id: string; name: string; home?: string; kind?: string; line: ReturnType<typeof tunnel> } }
const attached = new Map<string, { close(): void }>();
let speech: { proc: Subprocess<"pipe", "pipe", "inherit">; url: Promise<string> } | undefined;
let voiceSockets = 0, voiceIdle: Timer | undefined;

function speechService(): Promise<string> {
  if (speech) return speech.url;
  const missing = (why: string) => new Error(`The voice service did not start (${why})`);
  let proc: Subprocess<"pipe", "pipe", "inherit">;
  try {
    proc = Bun.spawn(VOICE_CMD, { cwd: HOME, stdin: "pipe", stdout: "pipe", stderr: "inherit" });
  } catch (e) {
    return Promise.reject(missing(e instanceof Error ? e.message : String(e)));
  }
  const url = (async () => {
    const reader = proc.stdout.getReader(), dec = new TextDecoder();
    let out = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) throw missing(`it exited: ${out.trim().slice(-300) || `code ${await proc.exited}`}`);
      out += dec.decode(value, { stream: true });
      const at = /voice: (ws:\/\/\S+)/.exec(out)?.[1];
      if (!at) continue;
      void (async () => { for (let r = await reader.read(); !r.done; r = await reader.read()) process.stdout.write(r.value); })();
      return at;
    }
  })();
  const s = { proc, url };
  speech = s;
  url.catch(() => { if (speech === s) speech = undefined; proc.kill(); });
  void proc.exited.then(() => { if (speech === s) speech = undefined; });
  return url;
}

function voiceClosed() {
  if (--voiceSockets > 0) return;
  clearTimeout(voiceIdle);
  voiceIdle = setTimeout(() => {
    const s = speech;
    if (!s || voiceSockets) return;
    speech = undefined;
    s.proc.stdin.end();
    setTimeout(() => s.proc.kill(), 3000);
  }, VOICE_IDLE_MS);
}

const LAN = process.env.CHRYSALIS_HOST === "0.0.0.0" && !OPEN, dir = tlsDir();
let certs: ReturnType<typeof webTls> | undefined;
const tls = () => (certs ??= webTls(dir));
const devices = new Devices(`${dir}/devices.json`), backends = new Backends(`${dir}/backend.json`, `${CHRYSALIS}/v1`), sea = await deviceScript();
const link = new Link(`${dir}/link.json`, LAN ? tls().link.fingerprint : "", LAN), MACHINE = hostname().replace(/\.local$/, "");
const PAGE = { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" };
const serve = (hostname: string, port: number) => Bun.serve<VoiceSocket>({
  hostname,
  tls: hostname === "127.0.0.1" || OPEN ? undefined : { cert: tls().cert, key: tls().key },
  maxRequestBodySize: 1 << 30,
  port,
  development: process.env.NODE_ENV !== "production",
  routes: {
    "/": { GET: chatPage },
    "/*": { GET: chatFile, HEAD: chatFile },
    "/chrysalis.mobileconfig": { GET: open(() => authority((a) => appleProfile(a), "application/x-apple-aspen-config", "chrysalis.mobileconfig")) },
    "/chrysalis-ca.crt": { GET: open(() => authority((a) => a.ca, "application/octet-stream", "chrysalis-ca.crt")) },
    "/chrysalis-ca-firefox.crt": { GET: open(() => authority((a) => a.caDer, "application/x-x509-ca-cert")) },
    "/opencode": { GET: open((req) => new Response(opencodeScript(new URL(req.url).origin, MACHINE, tls().dnsCa, fingerprint(tls().dnsCa), tls().cert, netName(), existsSync(MODEL_FILE) ? readFileSync(MODEL_FILE, "utf8").trim() || MODELS[0] : MODELS[0]),
      { headers: { "Content-Type": "text/x-shellscript", "Cache-Control": "no-store" } })) },
    "/.well-known/opencode": { GET: open((req) => Response.json(opencodeManifest(new URL(req.url).origin, MACHINE, tls().cert, process.platform !== "darwin"))) },
    "/opencode/ask": {
      POST: open(json(async (req) => {
        const { secret, code } = devices.computerAsk(String((await req.json().catch(() => ({})))?.name ?? ""));
        return new Response(`${secret} ${code}`);
      })),
    },
    "/opencode/ask/:secret": {
      GET: open((req) => {
        const got = devices.collect((req as Request & { params: { secret: string } }).params.secret);
        return !got ? new Response("gone", { status: 404 }) : got.state === "allowed" ? new Response(got.key) : new Response(null, { status: 202 });
      }),
    },
    "/opencode/machine": {
      GET: open((req, srv) => {
        const device = devices.find(req)?.device;
        if (device?.kind !== "computer") return Response.json({ error: { message: "this computer is not let in to Chrysalis here" } }, { status: 401 });
        if (req.headers.get("upgrade")?.toLowerCase() !== "websocket") return Response.json({ name: MACHINE });
        return srv.upgrade(req, { data: { queue: [], closed: false, machine: { id: device.id, name: device.name, line: tunnel(() => {}) } } }) ? undefined : new Response("a WebSocket only", { status: 426 });
      }),
    },
    "/opencode/machine.js": { GET: open(() => new Response(Bun.file(`${import.meta.dir}/machine.js`), { headers: { "Content-Type": "text/javascript", "Cache-Control": "no-store" } })) },
    "/opencode/:key/*": (req: Request, srv: Srv) => {
      const key = (req as Request & { params: { key: string } }).params.key, rest = new URL(req.url).pathname.slice(`/opencode/${MODEL_KEY}`.length);
      if (key !== MODEL_KEY || !self(req, srv)) return new Response("not found", { status: 404 });
      const [, to, path] = /^\/via\/([\w-]+)(\/.*)?$/.exec(rest) ?? [];
      if (to && !harness.outsideAllowed()) return Response.json({ error: { message: "OpenCode here is kept to Chrysalis's models" } }, { status: 403 });
      if (to) return tunedProxy(req, srv, `${Buffer.from(to, "base64url").toString().replace(/\/+$/, "")}${path ?? ""}`);
      if (rest.startsWith("/hidden/v1/")) return veiled(() => tunedProxy(req, srv, `${CHRYSALIS}${rest.slice("/hidden".length)}`, UNKEPT));
      return rest.startsWith("/v1/") ? tunedProxy(req, srv, `${CHRYSALIS}${rest}`) : new Response("not found", { status: 404 });
    },
    "/v1/*": open((req, srv) => OPEN || devices.find(req)?.device.kind === "computer" ? proxy(req, srv)
      : Response.json({ error: { message: "this computer is not let in to Chrysalis here" } }, { status: 401 })),
    "/pair/ask": {
      POST: open(json(async (req) => {
        const found = devices.find(req);
        if (found && devices.trusted(found.device)) return Response.json({ in: true }, { headers: PAST_WARNING });
        const b = await req.json().catch(() => ({})), nonce = String(b?.nonce ?? ""), name = String(b?.name ?? "");
        const ask = devices.ask(nonce, name);
        return ask ? Response.json(ask, { headers: PAST_WARNING })
          : Response.json({ error: { message: "this browser has not shown that it trusts Chrysalis" } }, { status: 403, headers: PAST_WARNING });
      })),
    },
    "/pair/ask/:secret": {
      GET: open((req, srv) => {
        const got = fromNetwork(req, srv) ? devices?.collect((req as Request & { params: { secret: string } }).params.secret) : undefined;
        if (!got) return Response.json({ state: "gone" }, { status: 404, headers: PAST_WARNING });
        return Response.json(got.state === "allowed" ? { state: "allowed" } : got,
          { headers: { ...PAST_WARNING, ...(got.state === "allowed" ? { "Set-Cookie": keyCookie(got.key) } : {}) } });
      }),
    },
    "/check-frame": {
      GET: open((req) => new Response(`<!doctype html><meta charset="utf-8"><script>
const img = new Image();
img.onload = () => parent.postMessage({ chrysalisCheck: true }, location.origin);
img.onerror = () => parent.postMessage({ chrysalisCheck: false }, location.origin);
img.src = ${JSON.stringify(checkUrl(req))} + "?t=" + Date.now();
</script>`, { headers: { ...PAGE, "Content-Security-Policy": `default-src 'none'; script-src 'unsafe-inline'; img-src ${new URL(checkUrl(req)).origin}; ` +
        "frame-ancestors 'self'" } })),
    },
    "/api/pair": {
      GET: here(async () => {
        if (!LAN) return Response.json({ lan: false });
        const connect = await connectUrls();
        return Response.json({ lan: true, connect, qr: qrSvg(connect[0]), api: connect.map((u) => `${u}v1`), waiting: devices.waiting(),
          devices: devices.all(), name: tls().caName, fingerprint: tls().fingerprint });
      }),
    },
    "/api/opencode": {
      GET: guarded(async (req, srv) => LAN
        ? Response.json({ line: opencodeLine(fromNetwork(req, srv) ? new URL(req.url).origin : (await connectUrls())[0].replace(/\/$/, ""), tls().cert) }) : Response.json({})),
    },
    "/api/pair/allow": {
      POST: here(json(async (req) => {
        const name = devices?.allow(String((await req.json().catch(() => ({})))?.code ?? ""));
        return name ? Response.json({ name }) : Response.json({ error: { message: "no device is asking with that code" } }, { status: 404 });
      })),
    },
    "/api/pair/deny": {
      POST: here(json(async (req) => new Response(null, { status: devices?.deny(String((await req.json().catch(() => ({})))?.code ?? "")) ? 204 : 404 }))),
    },
    "/api/pair/:id": { DELETE: trusted((req) => {
      const id = (req as Request & { params: { id: string } }).params.id;
      attached.get(id)?.close();
      return new Response(null, { status: devices?.remove(id) ? 204 : 404 });
    }) },
    "/api/link": {
      GET: here(() => Response.json(link.state())),
      POST: here(json(async (req) => {
        void link.pair(String((await req.json().catch(() => ({})))?.fp ?? ""), MACHINE).then(forwardHere);
        return Response.json(link.state());
      })),
      DELETE: here(() => (link.forget(), new Response(null, { status: 204 }))),
    },
    "/api/link/confirm": {
      POST: here(json(async () => {
        if (!link.confirm()) return Response.json({ error: { message: "no code to confirm" } }, { status: 409 });
        forwardHere();
        return Response.json(link.state());
      })),
    },
    "/vendor/mermaid/*": { GET: guarded(vendor) },
    "/vendor/three/*": { GET: guarded(vendor3d) },
    "/vendor/aframe/*": { GET: guarded(vendor3d) },
    "/sandbox": { GET: guarded(async () => new Response(LIVE ? await sandboxPage() : SANDBOX, { headers: { "Content-Type": "text/html; charset=utf-8", "Content-Security-Policy": SANDBOX_CSP } })) },

    "/api/chat": {
      POST: guarded(json(async (req, srv) => {
        srv.timeout(req, 0);
        const body = await req.text(), sent = (() => { try { return JSON.parse(body); } catch { return undefined; } })();
        if (sent?.chrysalis_thread && OPEN && fromNetwork(req, srv)) return Response.json({ error: { message: KEPT_HERE } }, { status: 403 });
        const hidden = !!sent?.chrysalis_incognito, asked = hidden ? JSON.stringify({ ...sent, chrysalis_incognito: undefined }) : body;
        if (!hidden) arrived(req, srv);
        if (sent?.stream) {
          if (hidden) veil();
          return follow(startJob(asked, hidden ? unveil : undefined), 0);
        }
        const plain = async () => {
          try {
            const r = await fetch(`${backends.url}/chat/completions`, {
              method: "POST", body: backends.isChrysalis ? asked : JSON.stringify(standard(JSON.parse(asked))), headers: { ...backends.headers(), ...(hidden ? UNKEPT : {}) }, signal: req.signal,
            });
            return new Response(r.body, { status: r.status, headers: { "Content-Type": r.headers.get("Content-Type") ?? "application/json" } });
          } catch {
            return Response.json({ error: { message: new ChrysalisDown().message } }, { status: 502 });
          }
        };
        return hidden ? veiled(plain) : plain();
      })),
    },
    "/api/chat/:id": {
      GET: guarded((req, srv) => {
        srv.timeout(req, 0);
        const job = jobs.get((req as Request & { params: { id: string } }).params.id);
        return job ? follow(job, Number(new URL(req.url).searchParams.get("from") ?? 0)) : Response.json({ error: { message: "gone" } }, { status: 404 });
      }),
    },
    "/api/chat/:id/stop": {
      POST: guarded((req) => {
        jobs.get((req as Request & { params: { id: string } }).params.id)?.ctrl.abort();
        return new Response(null, { status: 204 });
      }),
    },
    "/api/chat/:id/tool/:call": {
      POST: guarded(json(async (req) => {
        const { id, call } = (req as Request & { params: { id: string; call: string } }).params;
        const { decision, answer } = (await req.json().catch(() => ({}))) ?? {}, ask = jobs.get(id)?.asks.get(call), filled = !!answer && typeof answer === "object" && !Array.isArray(answer);
        if (!filled && !["once", "always", "deny"].includes(decision)) return Response.json({ error: { message: "decision: once, always or deny, or an answer" } }, { status: 400 });
        if (!ask) return Response.json({ error: { message: "that call is not waiting for an answer" } }, { status: 409 });
        ask(filled ? { answer } : decision);
        return new Response(null, { status: 204 });
      })),
    },

    "/api/threads": {
      GET: trusted(async () => {
        const at = harness.state();
        if (at.state !== "on") return Response.json({ ...at, outside: harness.outsideAllowed(), install: harness.INSTALL, wanted: !existsSync(OPENCODE_OFF) });
        const [threads, models, agents] = await Promise.all([harness.threads(), harness.models(), harness.agents()]);
        return Response.json({ ...at, outside: harness.outsideAllowed(), machines: [{ id: "", name: MACHINE, home: homedir(), kind: process.platform }, ...harness.machines()], name: connector.name, home: homedir(), models, agents, threads: threads.map((t) => ({ ...t, job: turns.get(t.id)?.id })) });
      }),
      DELETE: trusted(() => {
        writeFileSync(OPENCODE_OFF, "");
        harness.stop();
        return new Response(null, { status: 204 });
      }),
      POST: trusted(async () => {
        rmSync(OPENCODE_OFF, { force: true });
        try { await harness.ensure(CHRYSALIS); } catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
        return Response.json(harness.state());
      }),
    },
    "/api/incognito": {
      DELETE: trusted(async () => (await harness.forget(), new Response(null, { status: 204 }))),
    },
    "/api/threads/outside": {
      POST: here(json(async () => {
        writeFileSync(OPENCODE_OUTSIDE, "");
        return outsideNow(true);
      })),
      DELETE: trusted(async () => {
        rmSync(OPENCODE_OUTSIDE, { force: true });
        return outsideNow(false);
      }),
    },
    "/api/threads/folders": {
      GET: trusted(async (req) => {
        try { return Response.json(await harness.folders(new URL(req.url).searchParams.get("path") ?? "", new URL(req.url).searchParams.get("machine") || undefined)); }
        catch { return Response.json({ error: { message: "that folder cannot be read" } }, { status: 404 }); }
      }),
    },
    "/api/threads/:id/say": {
      POST: trusted(json(async (req) => {
        const text = String((await req.json().catch(() => ({})))?.text ?? "");
        if (!text.trim()) return Response.json({ error: { message: "text" } }, { status: 400 });
        try { await harness.say((req as Request & { params: { id: string } }).params.id, text); } catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
        return new Response(null, { status: 204 });
      })),
    },
    "/api/threads/:id/undo": {
      POST: trusted(json(async (req) => {
        try { await harness.undo((req as Request & { params: { id: string } }).params.id, String((await req.json().catch(() => ({})))?.message ?? "")); }
        catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
        return new Response(null, { status: 204 });
      })),
    },
    "/api/threads/:id/move": {
      POST: trusted(json(async (req) => {
        const dir = String((await req.json().catch(() => ({})))?.dir ?? ""), id = (req as Request & { params: { id: string } }).params.id;
        if (!dir || !existsSync(dir)) return Response.json({ error: { message: `There is no folder ${dir} on this computer` } }, { status: 400 });
        if (turns.has(id)) return Response.json({ error: { message: "OpenCode is still working in this chat" } }, { status: 409 });
        try { await harness.move(id, dir); } catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
        return new Response(null, { status: 204 });
      })),
    },
    "/api/threads/:id/rewrite": {
      POST: trusted(json(async (req) => {
        const { drop, edit, last } = (await req.json().catch(() => ({}))) ?? {}, id = (req as Request & { params: { id: string } }).params.id;
        const gone = Array.isArray(drop) ? drop.filter((x: unknown) => typeof x === "string") : [], put = typeof edit?.id === "string" && typeof edit?.text === "string" ? { id: edit.id, text: edit.text } : undefined;
        if (!gone.length && !put && typeof last !== "string") return Response.json({ error: { message: "drop, edit or last" } }, { status: 400 });
        if (turns.has(id)) return Response.json({ error: { message: "OpenCode is still working in this chat" } }, { status: 409 });
        if (typeof last === "string") await harness.settled(id, req.signal);
        try { await harness.rewrite(id, typeof last === "string" ? { last } : { drop: gone, ...(put ? { edit: put } : {}) }); } catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
        return new Response(null, { status: 204 });
      })),
    },
    "/api/threads/commands": {
      GET: trusted(async (req) => {
        const q = new URL(req.url).searchParams, dir = q.get("dir") ?? "", machine = q.get("machine") || undefined;
        try {
          const [commands, skills] = await Promise.all([harness.commands(dir, machine), harness.skills(dir, machine).catch(() => [])]);
          return Response.json({ commands, skills });
        } catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
      }),
    },
    "/api/threads/:id/compact": {
      POST: trusted(json(async (req) => {
        const id = (req as Request & { params: { id: string } }).params.id;
        if (turns.has(id)) return Response.json({ error: { message: "OpenCode is still working in this chat" } }, { status: 409 });
        try { await harness.compact(id); } catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
        return new Response(null, { status: 204 });
      })),
    },
    "/api/threads/:id/skill": {
      POST: trusted(json(async (req) => {
        const id = (req as Request & { params: { id: string } }).params.id, which = String((await req.json().catch(() => ({})))?.id ?? "");
        try { await harness.skill(id, which); } catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
        return new Response(null, { status: 204 });
      })),
    },
    "/api/threads/:id/handoff": {
      POST: trusted(json(async (req) => {
        const id = (req as Request & { params: { id: string } }).params.id, machine = String((await req.json().catch(() => ({})))?.machine ?? "");
        if (turns.has(id)) return Response.json({ error: { message: "OpenCode is still working in this chat" } }, { status: 409 });
        try { return Response.json(await harness.handoff(id, machine || undefined)); }
        catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
      })),
    },
    "/api/threads/:id/fork": {
      POST: trusted(json(async (req) => {
        const before = (await req.json().catch(() => ({})))?.before;
        try { return Response.json({ id: await harness.fork((req as Request & { params: { id: string } }).params.id, typeof before === "string" ? before : undefined) }); }
        catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
      })),
    },
    "/api/threads/:id": {
      GET: trusted(async (req) => {
        const id = (req as Request & { params: { id: string } }).params.id;
        try { return Response.json({ ...(await harness.transcript(id)), job: turns.get(id)?.id }); }
        catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
      }),
      PATCH: trusted(json(async (req) => {
        const { title, archived, scene } = (await req.json().catch(() => ({}))) ?? {};
        try { await harness.change((req as Request & { params: { id: string } }).params.id, { ...(String(title ?? "").trim() ? { title: String(title).trim() } : {}), ...(typeof archived === "boolean" ? { archived } : {}),
          ...(scene === null || Array.isArray(scene?.cast) ? { scene } : {}) }); }
        catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
        return new Response(null, { status: 204 });
      })),
      DELETE: trusted(async (req) => {
        const id = (req as Request & { params: { id: string } }).params.id;
        turns.get(id)?.ctrl.abort();
        try { await harness.remove(id); } catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
        return new Response(null, { status: 204 });
      }),
    },

    "/api/voice": trusted((req, srv) => {
      let same = false;
      try { same = new URL(req.headers.get("origin") ?? "").host === req.headers.get("host"); } catch {  }
      if (!same) return new Response("only Chrysalis's own pages", { status: 403 });
      return srv.upgrade(req, { data: { queue: [], closed: false } }) ? undefined : new Response("a WebSocket only", { status: 426 });
    }),
    "/voice-worklet.js": { GET: guarded(() => new Response(Bun.file(`${import.meta.dir}/voice-worklet.js`), { headers: { "Content-Type": "text/javascript" } })) },

    "/api/status": guarded(async (req, srv) => {
      const backend = await backends.about();
      const state = backend.listed ? { state: backend.listed.ok ? "online" : "offline", build: BUILD() } : await (async () => {
        try {
          const r = await engineStats();
          if (r.status === 503) {
            const b = await r.json().catch(() => ({}));
            return { state: "loading", build: BUILD(), loading: b.model ?? undefined, progress: b.progress ?? undefined };
          }
          if (!r.ok) throw new Error("loading");
          const stats = await r.json();
          return { state: stats.asleep ? "offline" : "online", build: BUILD(), ...stats };
        } catch {
          return { state: "offline", build: BUILD() };
        }
      })(), net = fromNetwork(req, srv), found = devices.find(req);
      const res = Response.json({ ...state, backend, here: !net, ...(net ? { machine: MACHINE } : {}), ...(found && devices!.stale(found.device) ? { check: true } : {}),
        ...(LAN && !net ? { waiting: devices.waiting().length } : {}),
        ...(!net && (state.state === "offline" || link.name || link.pairing) ? { link: link.state() } : {}) });
      if (found) res.headers.append("Set-Cookie", keyCookie(found.key));
      return res;
    }),
    "/api/backend": {
      POST: here(json(async (req) => {
        const b = await req.json().catch(() => ({}));
        const r = await backends.set(String(b?.url ?? ""), b?.confirmRemote === true);
        return r.ok ? Response.json(await backends.about()) : Response.json({ error: { message: r.error }, ...(r.remote ? { remote: r.remote } : {}) }, { status: r.remote ? 409 : 400 });
      })),
    },
    "/api/chrysalis/start": { POST: guarded(async (req) => {
      if (process.env.CHRYSALIS_ENGINE_HERE === "0" && !link.state().linked)
        return Response.json({ error: { message: "No model runs on this computer and none is linked: Chrysalis's terminal says why no GPU was found" } }, { status: 409 });
      const model = (await pickModel(req)) ?? (await engineStats().then((r) => r.json()).catch(() => ({}))).model;
      if (model) void engineCall("/chrysalis/load", { model }).catch(() => {});
      return new Response(null, { status: 202 });
    }) },
    "/api/chrysalis/load": {
      POST: guarded(async (req, srv) => {
        srv.timeout(req, 0);
        const model = await pickModel(req);
        if (!model) return Response.json({ error: { message: "unknown model" } }, { status: 400 });
        try {
          const r = await fetch(`${CHRYSALIS}/chrysalis/load`, { method: "POST", body: JSON.stringify({ model }), headers: { "Content-Type": "application/json" } });
          return new Response(r.body, { status: r.status, headers: { "Content-Type": "application/json" } });
        } catch {
          return Response.json({ error: { message: "Chrysalis is not running" } }, { status: 502 });
        }
      }),
    },
  },
  websocket: {
    open(ws) {
      const m = ws.data.machine;
      if (m) {
        m.line = tunnel((text) => { if (!ws.data.closed) ws.send(text); });
        attached.get(m.id)?.close();
        attached.set(m.id, ws);
        return;
      }
      voiceSockets++;
      clearTimeout(voiceIdle);
      speechService().then((url) => {
        if (ws.data.closed) return;
        const up = new WebSocket(url);
        up.binaryType = "arraybuffer";
        ws.data.up = up;
        up.onopen = () => { for (const m of ws.data.queue.splice(0)) up.send(m); };
        up.onmessage = (e) => ws.send(e.data);
        up.onclose = () => ws.close();
      }, (e: Error) => {
        ws.send(JSON.stringify({ type: "error", message: e.message }));
        ws.close();
      });
    },
    message(ws, m) {
      const machine = ws.data.machine;
      if (machine) {
        let got: any;
        try { got = JSON.parse(String(m)); } catch { return; }
        if (got.hello) {
          machine.home = String(got.hello.home ?? "");
          machine.kind = String(got.hello.kind ?? "");
          if (String(got.hello.name ?? "").trim()) machine.name = String(got.hello.name).trim().slice(0, 60);
          void harness.awayConfig(CHRYSALIS).then((config) => ws.send(JSON.stringify({ config })));
        } else if (got.ready) {
          void harness.attach({ id: machine.id, name: machine.name, home: machine.home ?? "", kind: machine.kind ?? "" }, machine.line.send).catch((e) => {
            console.error(`chrysalis web: ${machine.name} could not be run from here: ${(e as Error).message}`);
            ws.close();
          });
        } else if (got.failed) console.error(`chrysalis web: ${machine.name}: ${got.failed}`);
        else machine.line.got(got);
        return;
      }
      const up = ws.data.up;
      if (up?.readyState === WebSocket.OPEN) up.send(m);
      else ws.data.queue.push(m);
    },
    close(ws) {
      ws.data.closed = true;
      const machine = ws.data.machine;
      if (machine) {
        machine.line.closed();
        if (attached.get(machine.id) === ws) { attached.delete(machine.id); harness.detach(machine.id); }
        return;
      }
      ws.data.up?.close();
      voiceClosed();
    },
  },
});

const letIn = (req: Request, srv: Srv) => {
  if (OPEN || self(req, srv)) return true;
  const found = devices.find(req);
  return !!found && devices.trusted(found.device);
};
const bundle = Bun.serve({ hostname: "127.0.0.1", port: 0, routes: { "/": app }, fetch: () => new Response("not found", { status: 404 }),
  development: LIVE && { hmr: false } });
async function local(req: Request): Promise<Response> {
  const url = new URL(req.url), r = await fetch(`http://127.0.0.1:${bundle.port}${url.pathname}${url.search}`, { method: req.method });
  const back = new Headers(r.headers);
  for (const h of ["content-encoding", "content-length", "transfer-encoding", "connection"]) back.delete(h);
  if (url.pathname === "/") back.set("Cache-Control", "no-store");
  return new Response(r.body, { status: r.status, headers: back });
}
const chatPage = open((req, srv) => letIn(req, srv) ? local(req) : new Response(connectPage(tls(), checkUrl(req), sea), { headers: { ...PAGE, ...PAST_WARNING,
  "Content-Security-Policy":
  `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src ${new URL(checkUrl(req)).origin} data:; connect-src 'self'; ` +
  "base-uri 'none'; form-action 'none'; frame-ancestors 'none'" } }));
const chatFile = open((req, srv) => letIn(req, srv) ? local(req) : req.headers.get("sec-fetch-mode") === "navigate"
  ? new Response(null, { status: 302, headers: { Location: "/", "Cache-Control": "no-store", ...PAST_WARNING } }) : new Response("not found", { status: 404, headers: PAST_WARNING }));
const authority = (body: (a: ReturnType<typeof webTls>) => Uint8Array<ArrayBuffer> | string, type: string, name?: string) =>
  new Response(body(tls()), { headers: { "Content-Type": type, "Cache-Control": "no-store", ...PAST_WARNING, ...(name ? { "Content-Disposition": `attachment; filename="${name}"` } : {}) } });

const server = serve("127.0.0.1", Number(process.env.PORT ?? 8081));
harness.routeModel(`http://127.0.0.1:${server.port}/opencode/${MODEL_KEY}`, (u) => {
  try {
    const at = new URL(u);
    return ["127.0.0.1", "localhost", "[::1]", ...lanAddresses()].includes(at.hostname) && [server.port, linkPort].includes(Number(at.port)) && at.pathname.startsWith("/v1");
  } catch { return false; }
});
const secure = LAN || OPEN ? lanAddresses().map((ip) => serve(ip, server.port!)) : [];
if (process.env.CHRYSALIS_OPENCODE_ON) void harness.ensure(CHRYSALIS).catch((e) => tell({ note: `OpenCode did not start again: ${(e as Error).message}` }));
let checkPort = Number(process.env.CHRYSALIS_CHECK_PORT ?? 8083);
const checks = (LAN ? lanAddresses() : []).map((ip) => {
  const s = Bun.serve({ hostname: ip, port: checkPort, tls: tls().check, fetch: checkFetch });
  checkPort = s.port!;
  return s;
});
const checkUrl = (req: Request) => `https://${new URL(req.url).hostname}:${checkPort}/check.gif`;
let linkPort = Number(process.env.CHRYSALIS_LINK_PORT ?? 8082);
const LINKED = ["/chrysalis/stats", "/chrysalis/load", "/health", "/v1/health"];
const links = (LAN ? lanAddresses() : []).map((ip) => {
  const s = Bun.serve({ hostname: ip, port: linkPort, tls: { cert: tls().link.cert, key: tls().link.key }, maxRequestBodySize: 1 << 30, development: false,
    routes: {
      "/link/ca": { GET: () => new Response(tls().link.ca, { headers: { "Content-Type": "application/x-pem-file" } }) },
      "/link/ask": {
        POST: json(async (req) => {
          const b = await req.json().catch(() => ({})), asked = devices.linkAsk(String(b?.name ?? ""), String(b?.commit ?? ""));
          return asked ? Response.json(asked) : Response.json({ error: { message: "a commit: 64 hex digits" } }, { status: 400 });
        }),
      },
      "/link/reveal": {
        POST: json(async (req) => {
          const b = await req.json().catch(() => ({}));
          return new Response(null, { status: devices.linkReveal(String(b?.secret ?? ""), String(b?.nonce ?? ""), tls().link.fingerprint) ? 204 : 404 });
        }),
      },
      "/link/ask/:secret": { GET: (req) => { const got = devices.collect(req.params.secret); return Response.json(got ?? { state: "gone" }, { status: got ? 200 : 404 }); } },
    },
    fetch(req, srv) {
      const found = devices.find(req), path = new URL(req.url).pathname;
      if (found?.device.kind !== "computer") return Response.json({ error: { message: "this computer is not linked to Chrysalis here" } }, { status: 401 });
      return path.startsWith("/v1/") || LINKED.includes(path) ? proxy(req, srv as unknown as Srv) : new Response("not found", { status: 404 });
    },
  });
  linkPort = s.port!;
  return s;
});
let engineSeen = 0;
const seeEngine = () => void engineStats().then((r) => { if (r.ok || r.status === 503) engineSeen = Date.now(); }, () => {});
if (links.length) {
  seeEngine();
  setInterval(seeEngine, 10_000);
  advertise(MACHINE, linkPort, tls().link.fingerprint, () => !link.name && Date.now() - engineSeen < 60_000);
  answerName(netName(), lanAddresses);
}
let forwarder: Server<undefined> | undefined;
function forwardHere() {
  if (forwarder || !link.name) return;
  const at = new URL(CHRYSALIS);
  try {
    forwarder = Bun.serve({ hostname: at.hostname, port: Number(at.port || 80), maxRequestBodySize: 1 << 30, development: false,
      fetch: (req, srv) => (srv.timeout(req, 0), link.forward(req)) });
  } catch (e) {
    console.log(`chrysalis link: ${CHRYSALIS} is taken, so Chrysalis runs here (${(e as Error).message})`);
    tell({ note: `${CHRYSALIS} is taken, so the model runs here and not on ${link.name}` });
  }
}
forwardHere();
const spaced = (code: string) => `${code.slice(0, 3)} ${code.slice(3)}`;
const answers = new Map<string, (yes: boolean) => void>(), answered = new Set<string>();
let told = "[]";
if (process.send) {
  process.on("message", (m: any) => {
    if (m?.forget) return void harness.forget();
    const key = String(m?.answer?.key ?? ""), answer = answers.get(key);
    if (!answer) return;
    answered.add(key);
    answer(!!m.answer.yes);
  });
  setInterval(() => {
    const open: { key: string; text: string }[] = [];
    const question = (key: string, text: string, answer: (yes: boolean) => void) => {
      if (answered.has(key)) return;
      answers.set(key, answer);
      open.push({ key, text });
    };
    answers.clear();
    for (const a of LAN ? devices.waiting() : []) if (!a.allowed)
      question(`ask ${a.code}`, `${a.name} asks to use ${a.kind === "computer" ? "this computer's Chrysalis" : "this computer through Chrysalis, as you do"} · ${spaced(a.code)}`,
        (yes) => void (yes ? devices.allow(a.code) : devices.deny(a.code)));
    const p = link.pairing;
    if (p?.state === "waiting" && p.code && !p.confirmed)
      question(`link ${p.code}`, `${p.name}'s code is ${spaced(p.code)}: the same as on ${p.name}?`, (yes) => yes ? (link.confirm(), forwardHere()) : link.forget());
    const now = JSON.stringify(open);
    if (now !== told) tell({ questions: open });
    told = now;
  }, 1000);
}
const declined = new Set<string>();
if (process.env.CHRYSALIS_ENGINE_HERE === "0") setInterval(() => {
  const s = link.state(), to = s.found.find((f) => !declined.has(f.fp));
  if (!s.linked && !link.pairing && to) void link.pair(to.fp, MACHINE).then(() => {
    if (!link.pairing && !link.state().linked) declined.add(to.fp);
    forwardHere();
  });
}, 2000);
const API_PORT = process.env.CHRYSALIS_API_PORT ?? (new URL(CHRYSALIS).port || "80"), apis = new Map<string, number>();
for (const ip of OPEN ? lanAddresses() : []) {
  try {
    apis.set(ip, Bun.serve({ hostname: ip, port: Number(API_PORT), maxRequestBodySize: 1 << 30, development: false,
      fetch(req, srv) {
        const path = new URL(req.url).pathname;
        return hostOk(req) && (path.startsWith("/v1/") || LINKED.includes(path)) ? proxy(req, srv as unknown as Srv) : new Response("not found", { status: 404 });
      } }).port!);
  } catch (e) {
    tell({ note: `the API is not on http://${ip}:${API_PORT}, which something else holds (${(e as Error).message})` });
  }
}
const network = LAN || OPEN ? (await connectUrls()).map((u) => u.replace(/\/$/, "")) : [];
tell({ up: { local: `http://localhost:${server.port}`, network, api: network.flatMap((u) => { const ip = new URL(u).hostname; return apis.has(ip) ? [`http://${ip}:${apis.get(ip)}/v1`] : []; }) } });
if (!process.send) {
  console.log(`chrysalis web: ${server.url}`);
  for (const s of secure) console.log(`chrysalis web: ${s.url}`);
  for (const s of checks) console.log(`chrysalis check: ${s.url}`);
  for (const s of links) console.log(`chrysalis link: ${s.url}`);
  for (const [ip, port] of apis) console.log(`chrysalis api: http://${ip}:${port}`);
}

function checkFetch(req: Request): Response {
  if (!hostOk(req)) return new Response("unknown host", { status: 403 });
  const url = new URL(req.url);
  if (url.pathname !== "/check.gif") return new Response("not found", { status: 404 });
  devices.check(req, url.searchParams.get("n"));
  return new Response(TRUSTED_GIF, { headers: { "Content-Type": "image/gif", "Cache-Control": "no-store" } });
}

async function tuned(req: Request, body: string | undefined): Promise<string | undefined> {
  const session = req.headers.get("x-opencode-session");
  if (!body || !session || req.method !== "POST") return body;
  let sent: Record<string, unknown>;
  try { sent = JSON.parse(body); } catch { return body; }
  if (!Array.isArray(sent?.messages)) return body;
  const want = await harness.tuningOf(session).catch(() => undefined);
  if (!want) return body;
  const { max_tokens, ...rest } = want, cap = "max_completion_tokens" in sent ? "max_completion_tokens" : "max_tokens";
  return JSON.stringify({ ...sent, ...rest, ...(max_tokens ? { [cap]: max_tokens } : {}) });
}
const UNKEPT = { "X-Chrysalis-Unkept": "1" };
let veils = 0;
function unveil() {
  veils--;
  tell({ veiled: veils });
}
function veil() {
  veils++;
  tell({ veiled: veils });
}
async function veiled(answer: () => Response | Promise<Response>): Promise<Response> {
  veil();
  let r: Response;
  try { r = await answer(); } catch (e) { unveil(); throw e; }
  if (!r.body) return unveil(), r;
  const reader = r.body.getReader();
  let over = false;
  const end = () => { if (!over) { over = true; unveil(); } };
  return new Response(new ReadableStream<Uint8Array>({
    async pull(ctl) {
      const { value, done } = await reader.read().catch((e) => { end(); throw e; });
      if (done) return end(), ctl.close();
      ctl.enqueue(value);
    },
    cancel(why) { end(); return reader.cancel(why); },
  }), { status: r.status, headers: r.headers });
}
harness.onHidden((on) => tell({ hidden: on }));
async function outsideNow(on: boolean): Promise<Response> {
  if (harness.outsideAllowed() === on) return new Response(null, { status: 204 });
  harness.allowOutside(on);
  if (harness.state().state !== "on") return new Response(null, { status: 204 });
  for (const job of new Set(turns.values())) job.ctrl.abort();
  harness.stop();
  try { await harness.ensure(CHRYSALIS); } catch (e) { return Response.json({ error: { message: (e as Error).message } }, { status: 502 }); }
  return new Response(null, { status: 204 });
}
async function tunedProxy(req: Request, srv: Srv, to: string, more: Record<string, string> = {}): Promise<Response> {
  srv.timeout(req, 0);
  const url = new URL(req.url), headers = new Headers(req.headers);
  for (const h of ["host", "connection", "content-length"]) headers.delete(h);
  for (const [k, v] of Object.entries(more)) headers.set(k, v);
  const body = await tuned(req, req.method === "GET" || req.method === "HEAD" ? undefined : await req.text());
  try {
    const r = await fetch(`${to}${url.search}`, { method: req.method, headers, signal: req.signal, body });
    const back = new Headers(r.headers);
    for (const h of ["content-encoding", "content-length", "transfer-encoding", "connection"]) back.delete(h);
    return new Response(r.body, { status: r.status, headers: back });
  } catch {
    return Response.json({ error: { message: `${new URL(to).host} could not be reached` } }, { status: 502 });
  }
}
async function proxy(req: Request, srv: Srv): Promise<Response> {
  srv.timeout(req, 0);
  if (req.method === "POST" && ASKS.test(new URL(req.url).pathname)) arrived(req, srv);
  const url = new URL(req.url), headers = new Headers(req.headers);
  for (const h of ["host", "cookie", "authorization", "x-api-key", "connection", "content-length"]) headers.delete(h);
  try {
    const r = await fetch(`${CHRYSALIS}${url.pathname}${url.search}`, { method: req.method, headers, signal: req.signal,
      body: await tuned(req, req.method === "GET" || req.method === "HEAD" ? undefined : await req.text()) });
    const back = new Headers(r.headers);
    for (const h of ["content-encoding", "content-length", "transfer-encoding", "connection"]) back.delete(h);
    return new Response(r.body, { status: r.status, headers: back });
  } catch {
    return Response.json({ error: { message: "Chrysalis is not running" } }, { status: 502 });
  }
}
async function connectUrls(): Promise<string[]> {
  const first = await routeAddress();
  return lanAddresses().sort((a, b) => Number(b === first) - Number(a === first)).map((ip) => `${OPEN ? "http" : "https"}://${ip}:${server.port}/`);
}

const qrs = new Map<string, string>();
function qrSvg(url: string): string {
  if (!qrs.has(url)) qrs.set(url, qrCode(url));
  return qrs.get(url)!;
}
