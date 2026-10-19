// kiln web UI: serves the chat app, proxies the kiln server's OpenAI API (same origin, so no CORS) and runs the tools
// replies call (tools.ts). It stores nothing on disk: conversations stay in the browser tab, replies being written stay
// in RAM (see Job). `bun server.ts` for development, NODE_ENV=production for a minified bundle.
import { hostname } from "node:os";
import type { Server, Subprocess } from "bun";
import app from "./index.html";
import * as registry from "./tools";
import { lanAddresses, lanTls } from "./tls";
import { assistantTurn, DESCRIPTIONS, toolTurn, unnoted, type ToolChoice, type WireCall } from "./wire";

const KILN = process.env.KILN_URL ?? "http://127.0.0.1:8080";
// Changes with the page's code. Open tabs compare it on every status poll and reload to pick up an update
// (their chats are in sessionStorage, so a reload keeps them).
const BUILD = Bun.hash((await Promise.all(["index.html", "app.ts", "sampler.ts", "files.ts", "wire.ts", "voice.ts", "voice-worklet.js", "style.css"].map((f) => Bun.file(`${import.meta.dir}/${f}`).text()))).join("\0")).toString(36);
await registry.reload();
for (const sig of ["SIGINT", "SIGTERM"] as const) process.on(sig, () => void registry.closeAll().finally(() => process.exit(0)));   // MCP servers' processes go too

// DNS rebinding guard: a page from another site that points its own name at this machine still sends that name as
// Host. Requests addressed to an IP address or to this machine's names are fine.
const NAMES = new Set(["", "localhost", hostname().toLowerCase(), `${hostname().toLowerCase()}.local`]);
const hostOk = (req: Request) => {
  const host = (req.headers.get("host") ?? "").toLowerCase();
  return host.startsWith("[") || /^\d{1,3}(\.\d{1,3}){3}(:\d+)?$/.test(host) || NAMES.has(host.replace(/:\d+$/, ""));
};
type Srv = Server<VoiceSocket>;
const guarded = (handler: (req: Request, srv: Srv) => Response | undefined | Promise<Response>) =>
  (req: Request, srv: Srv) => (hostOk(req) ? handler(req, srv) : new Response("unknown host", { status: 403 }));
// A form on another site can post to this server's address, but not as JSON without asking first (CORS), which it is
// never allowed: so what starts tools or lets one run takes JSON only.
const json = (handler: (req: Request, srv: Srv) => Response | Promise<Response>) => (req: Request, srv: Srv) =>
  (req.headers.get("content-type") ?? "").startsWith("application/json") ? handler(req, srv) : new Response("JSON only", { status: 415 });

const unit = (...args: string[]) => Bun.spawnSync(["systemctl", "--user", ...args, "kiln"]).stdout.toString().trim();
// The model picker's choice, kept in the systemd user environment: kiln.sh serve loads it when kiln (re)starts.
const MODELS = ["qwen3.8-27b", "qwen3.8-flash-next"];
async function pickModel(req: Request): Promise<string | undefined> {
  const model = (await req.json().catch(() => ({})))?.model;
  if (!MODELS.includes(model)) return undefined;
  Bun.spawnSync(["systemctl", "--user", "set-environment", `KILN_MODEL=${model}`]);
  return model;
}

// A reply belongs to this server, not to the page that asked for it: the request to kiln runs to its end whatever
// the page does (reloads, closes, sleeps with the laptop, loses Wi-Fi, opens another chat), and any page can follow it
// again from where it left off: every event has a position. Only an explicit stop ends a reply early. A reply stays
// in RAM until 15 minutes after it ends, so a page that comes back in time collects the rest; nothing goes to disk.
// With tools, a reply is rounds: kiln's answer ends in tool calls, this server runs them and asks kiln again with
// their results, so a reply with tools goes on while no page is open, and its events are all of its rounds.
interface Job {
  id: string;
  events: string[];                                 // kiln's SSE data payloads and the tool events, the timings, [DONE]
  done: boolean;
  ctrl: AbortController;
  waiters: Set<() => void>;
  asks: Map<string, (d: Decision) => void>;         // tool calls waiting for the user, by call id
  t0: number; tFirst?: number; tAnswer?: number; tLast?: number;   // ms: start, first token, first answer token, last
  gen: number;                                      // ms between each round's first token and its last, summed
  held?: Promise<void>; go?: () => void;            // a voice reply begun ahead (HELD_MS): it runs no tool until go
}
type Decision = "once" | "always" | "deny";
const jobs = new Map<string, Job>();
const KEEP_MS = 15 * 60_000;
const MAX_ROUNDS = 24;
// A voice reply begun before its question was settled (kiln_voice.ahead) runs no tool until the page says it stands
// (/go): it may yet be dropped, and a tool can do things. One the page neither confirms nor stops is stopped.
const HELD_MS = 60_000;
// Fields of a request that only make sense for its first round (a continued or rewritten reply).
const FIRST_ONLY = ["continue_final_message", "kiln_original", "kiln_append_token"];

function startJob(body: string): Job {
  const job: Job = { id: crypto.randomUUID(), events: [], done: false, ctrl: new AbortController(), waiters: new Set(), asks: new Map(), t0: Date.now(), gen: 0 };
  jobs.set(job.id, job);
  const push = (data: string | object) => {
    job.events.push(typeof data === "string" ? data : JSON.stringify(data));
    for (const wake of [...job.waiters]) wake();
  };
  (async () => {
    try {
      const { req, web, said, tool, ahead, tools, notes } = await asSent(JSON.parse(body));
      if (ahead) {
        job.held = new Promise((go) => {
          const timer = setTimeout(() => job.ctrl.abort(), HELD_MS);
          job.go = () => (clearTimeout(timer), go());
          job.ctrl.signal.addEventListener("abort", () => (clearTimeout(timer), go()));
        });
      }
      if (!web && !said) return void await round(job, JSON.stringify(req), push);
      for (const note of notes) push({ note });
      const allow = new Set(web?.allow ?? []);
      let messages: object[] = req.messages ?? [];
      const route = said && tools.length && web?.route ? ROUTE : undefined;
      let called = false, force = false, preamble: string | undefined;
      for (let k = 0; ; k++) {
        if (k) push({ round: k });
        // a voice turn that needs a tool (the page's check, or the user asked for a search) starts with the call
        const first = k === 0 && route && (tool || route.asks.test(unnoted(textOf((messages.findLast((m: any) => m.role === "user") as any)?.content))));
        if (first) preamble = route.preambles[messages.length % route.preambles.length];
        const out = first || force ? await forced(job, req, messages, push, preamble) : await round(job, JSON.stringify({ ...req, messages }), push);
        // the reply said what it would do and did not: the next round does it
        if (out && !job.ctrl.signal.aborted && route && !out.calls.length && !called && !force && !first && route.announces.test(out.content)) {
          force = true;
          preamble = undefined;
          for (const f of FIRST_ONLY) delete req[f];
          messages = [...messages, assistantTurn(out.content, out.reasoning)];
          continue;
        }
        force = false;
        called ||= !!out?.calls.length;
        if (!out || !out.calls.length || web?.loop === false || job.ctrl.signal.aborted) break;
        await job.held;
        if (job.ctrl.signal.aborted) throw new DOMException("stopped", "AbortError");
        const results = k + 1 < MAX_ROUNDS ? await Promise.all(out.calls.map((c) => call(job, c, tools, allow, web ?? { tools: {} }, push)))
          : out.calls.map((c) => {
            const result = `Not run: this reply reached its limit of ${MAX_ROUNDS} rounds of tool calls.`;
            push({ tool: { id: c.id, state: "stopped", result } });
            return result;
          });
        if (job.ctrl.signal.aborted) throw new DOMException("stopped", "AbortError");
        if (k + 1 >= MAX_ROUNDS) { push({ note: `Stopped after ${MAX_ROUNDS} rounds of tool calls: Continue carries on.` }); break; }
        if (k === 0 && req.continue_final_message) messages = messages.slice(0, -1);   // the prefill, now the whole turn
        for (const f of FIRST_ONLY) delete req[f];
        messages = [...messages, assistantTurn(out.content, out.reasoning, out.calls), ...out.calls.map((c, i) => toolTurn(c.id, results[i]))];
      }
    } catch (e) {
      if (!job.ctrl.signal.aborted && !(e instanceof KilnDown)) console.error(`kiln web: reply ${job.id} failed:`, e);
      push(job.ctrl.signal.aborted ? { stopped: true } : { error: { message: e instanceof KilnDown ? "kiln is not running" : e instanceof Error ? e.message : String(e) } });
    } finally {
      push({ timing: { start: job.t0, first: job.tFirst, answer: job.tAnswer, last: job.tLast, gen: job.gen } });
      push("[DONE]");
      job.done = true;
      setTimeout(() => jobs.delete(job.id), KEEP_MS);
    }
  })();
  return job;
}

// A reply's request as kiln is sent it: the page's own fields (kiln_web, kiln_voice) taken off, and the chat's tools
// (kiln_web's choice), each with kiln's description where it has one (wire.ts). The reply's job and the page's view of
// what the model is sent (/api/preview) both build it here, so the view is what kiln gets.
async function asSent(raw: any) {
  const req = { ...raw }, web: ToolChoice | undefined = req.kiln_web, said = !!req.kiln_voice, tool = req.kiln_voice?.tool === true;
  const ahead = req.kiln_voice?.ahead === true;
  delete req.kiln_web;
  delete req.kiln_voice;
  const { tools, notes } = web ? await registry.select(web.tools ?? {}) : { tools: [] as registry.Named[], notes: [] as string[] };
  if (tools.length) req.tools = tools.map((n) => registry.openai(n, DESCRIPTIONS[n.tool.name]));
  return { req, web, said, tool, ahead, tools, notes };
}

class KilnDown extends Error {}
// kiln took the request and its answer broke off (its log says why: an exception in kiln drops the connection)
class KilnBroke extends Error { message = "kiln's reply broke off (kiln's log has why)"; }
// One request to kiln, its events passed on as they come: what it wrote, and the tool calls it ended with.
async function round(job: Job, body: string, push: (d: string | object) => void, signal = job.ctrl.signal) {
  const r = await fetch(`${KILN}/v1/chat/completions`, {
    method: "POST", body, headers: { "Content-Type": "application/json" }, signal,
  }).catch((e) => { throw signal.aborted ? e : new KilnDown(); });
  if (!r.ok || !r.body) {
    push({ error: { message: (await r.json().catch(() => null))?.error?.message ?? `kiln answered ${r.status}` } });
    return null;
  }
  const out = { content: "", reasoning: "", calls: [] as WireCall[] };
  let first: number | undefined, last: number | undefined;
  const reader = r.body.getReader(), decoder = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { value, done } = await reader.read().catch((e) => { throw signal.aborted ? e : new KilnBroke(); });
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      for (let end = buf.indexOf("\n\n"); end >= 0; end = buf.indexOf("\n\n")) {
        const data = buf.slice(0, end).split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n");
        buf = buf.slice(end + 2);
        if (!data || data === "[DONE]") continue;
        const now = Date.now(), delta = data.includes('"delta"') ? JSON.parse(data).choices?.[0]?.delta : undefined;
        if (delta?.content || delta?.reasoning_content || delta?.tool_calls) {
          job.tFirst ??= now;
          job.tLast = last = now;
          first ??= now;
          if (delta.content) job.tAnswer ??= now;
        }
        if (delta?.kiln_prefix) {
          out.content = delta.kiln_prefix.content ?? "";
          out.reasoning = delta.kiln_prefix.reasoning_content ?? "";
        }
        out.content += delta?.content ?? "";
        out.reasoning += delta?.reasoning_content ?? "";
        for (const c of delta?.tool_calls ?? []) {
          const at = (out.calls[c.index ?? out.calls.length] ??= { id: "", name: "", args: "" });
          if (c.id) at.id = c.id;
          if (c.function?.name) at.name += c.function.name;
          if (c.function?.arguments) at.args += c.function.arguments;
        }
        push(data);
      }
    }
  } finally {
    if (first && last) job.gen += last - first;
  }
  return out;
}

// A voice turn (kiln_voice): whether its reply runs a tool is settled before the reply, not left to its words. The page
// asks beside the conversation (app.ts, heardFor: with what speech recognition misheard, at temperature 0) and says so
// here (kiln_voice.tool); a user who asks for a search (the route's asks) gets one whatever it said. A call is held to
// the tools' grammar from its first token (tool_choice "required"), after the spoken line that says what is happening
// (one of the route's preambles): a model with thinking off, told its reply is spoken, says "I'm searching for that
// now" and ends its turn often enough (and then copies that turn: 1 call in 6 after one such turn in the history), and
// may report results it never got. A reply that announces a tool it did not call (its announces, after "I'll", "let
// me", ...) is made to call one.
type Out = NonNullable<Awaited<ReturnType<typeof round>>>;
// A list of phrases as a pattern for whole words, whatever the spacing between them.
function phrasePattern(phrases: string[], before = ""): RegExp {
  return new RegExp(`\\b${before}(${phrases.map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+")).join("|")})\\b`, "i");
}
interface Routing { preambles: string[]; asks: RegExp; announces: RegExp }
const ROUTE: Routing = {
  preambles: ["Let me look that up.", "Let me check.", "One moment, I'll look."],
  asks: phrasePattern(["search", "google", "look it up", "look that up", "look this up", "look them up", "look up", "browse", "fetch", "online",
    "on the web", "on the internet", "open the link", "open this link", "open that link", "open the page", "open this page"]),
  announces: phrasePattern(["search", "searching", "look that up", "look it up", "look this up", "look them up", "look those up", "look up",
    "looking up", "looking that up", "looking it up", "look for", "looking for", "look into", "looking into", "check on", "check for",
    "check online", "check the", "check that", "check it", "check this", "checking", "find out", "find that", "find the", "find it",
    "find a", "find some", "finding", "pull up", "pulling up", "browse", "browsing", "fetch", "fetching", "open the link",
    "open that link", "open the page", "opening"], "(I['’]?ll|I will|I['’]?m going to|I am going to|let me|I['’]?m|I am)\\s+(now\\s+|just\\s+|quickly\\s+)?"),
};
const FORCE = { tool_choice: "required", logprobs: false, top_logprobs: undefined };   // kiln holds no grammar with logprobs
const textOf = (c: unknown) => (typeof c === "string" ? c : Array.isArray(c) ? c.map((p: any) => p?.text ?? "").join(" ") : "");

async function forced(job: Job, req: any, messages: object[], push: (d: string | object) => void, preamble?: string): Promise<Out | null> {
  if (preamble) {
    push({ choices: [{ index: 0, delta: { content: preamble } }] });
    job.tFirst ??= Date.now();
    job.tAnswer ??= Date.now();
  }
  const out = await round(job, JSON.stringify({ ...req, messages, ...FORCE }), push);
  if (out && preamble) out.content = preamble + out.content;
  return out;
}

// One tool call, from the model's arguments to the result it is given: refused when the chat did not turn the tool on,
// asked about when it is not read-only (unless the user allowed it for good), stopped by Stop. Its events carry what
// the page shows of it; the result also goes to the model as its tool message.
async function call(job: Job, c: WireCall, tools: registry.Named[], allow: Set<string>, web: ToolChoice, push: (d: object) => void): Promise<string> {
  const n = tools.find((t) => t.fn === c.name);
  const about = { id: c.id, title: n?.tool.title ?? c.name, source: n?.source.label, readOnly: n?.tool.readOnly };
  const end = (state: string, result: string, ms?: number) => (push({ tool: { ...about, state, result, ms } }), result);
  if (!n) return end("failed", `Error: there is no tool named "${c.name}" in this chat (the user has not turned it on). The tools you can call are: ${tools.map((t) => t.fn).join(", ") || "none"}.`);
  let args: Record<string, unknown>;
  try {
    args = c.args.trim() ? JSON.parse(c.args) : {};
    if (!args || typeof args !== "object" || Array.isArray(args)) throw new Error("not an object");
  } catch {
    return end("failed", `Error: the arguments are not a JSON object: ${c.args.slice(0, 200)}`);
  }
  if (!n.tool.readOnly && !allow.has(n.fn)) {
    push({ tool: { ...about, state: "approval" } });
    const d = await new Promise<Decision | "stop">((resolve) => {
      if (job.ctrl.signal.aborted) return resolve("stop");
      job.asks.set(c.id, resolve);
      job.ctrl.signal.addEventListener("abort", () => resolve("stop"), { once: true });
    });
    job.asks.delete(c.id);
    if (d === "stop") return end("stopped", "Not run: the user stopped the reply before this call was allowed.");
    if (d === "deny") return end("declined", "The user declined this call: it did not run. Do not try it again unless they ask for it.");
    if (d === "always") allow.add(n.fn);
  }
  push({ tool: { ...about, state: "running" } });
  const t = Date.now();
  try {
    const result = await registry.run(n, args, job.ctrl.signal, n.source.kind === "fetch" ? web.fetchChars : undefined);
    return end("done", result, Date.now() - t);
  } catch (e) {
    if (job.ctrl.signal.aborted) return end("stopped", "Stopped: the user stopped the reply while this call ran.", Date.now() - t);
    return end("failed", `Error: ${e instanceof Error ? e.message : String(e)}`, Date.now() - t);
  }
}

// A job's events from position `from` on, live until it ends. A page that goes away just stops listening.
function follow(job: Job, from: number): Response {
  let i = Math.max(0, from), waiter: (() => void) | undefined;
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async pull(ctl) {
      while (i >= job.events.length && !job.done)
        await new Promise<void>((resolve) => { waiter = () => { job.waiters.delete(waiter!); resolve(); }; job.waiters.add(waiter); });
      if (i < job.events.length) {
        ctl.enqueue(encoder.encode(job.events.slice(i).map((d, k) => `id: ${i + k}\ndata: ${d}\n\n`).join("")));
        i = job.events.length;
      } else ctl.close();
    },
    cancel() { if (waiter) job.waiters.delete(waiter); },
  });
  return new Response(stream, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Kiln-Job": job.id } });
}

// Mermaid's ES module build (the page imports it the first time it draws a diagram): its entry and the chunks it
// loads, straight from the package and nothing else from node_modules.
const MERMAID = `${import.meta.dir}/node_modules/mermaid/dist/`;
const vendor = (req: Request) => {
  const file = new URL(req.url).pathname.slice("/vendor/mermaid/".length);
  if (!/^(mermaid\.esm\.min\.mjs|chunks\/mermaid\.esm\.min\/[\w-]+\.mjs)$/.test(file)) return new Response("not found", { status: 404 });
  return new Response(Bun.file(MERMAID + file), { headers: { "Content-Type": "text/javascript", "Cache-Control": "max-age=86400" } });
};

// The frame an HTML page from a reply runs in. Two walls: the page's iframe is sandboxed (allow-scripts only: an opaque
// origin, so no cookies, storage or access to the chat), and this policy lets its own inline code run but nothing
// load from or talk to anywhere, kiln included. The chat posts the page in, which replaces this document.
const SANDBOX = `<!doctype html><meta charset="utf-8"><script>
addEventListener("message", (e) => {
  if (e.source !== parent || typeof e.data?.kilnPage !== "string") return;
  document.open(); document.write(e.data.kilnPage); document.close();
}, { once: true });
parent.postMessage({ kilnReady: true }, "*");
</script>`;
const SANDBOX_CSP = "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'; img-src data: blob:; " +
  "font-src data:; media-src data: blob:; form-action 'none'; base-uri 'none'; frame-ancestors 'self'; sandbox allow-scripts allow-modals";

// Voice mode's speech service: a process of its own (voice/ in data/voice-venv, or any stack that speaks the same
// protocol: KILN_VOICE_CMD, a JSON argv), started by the first voice socket and stopped KILN_VOICE_IDLE_MS after the
// last one closes, so voice mode holds no memory while nobody uses it. It prints `voice: ws://...` once it listens and
// exits when its stdin closes (this server gone, or the idle time up). Each page's socket is proxied to it verbatim.
const ROOT = `${import.meta.dir}/..`;
const VOICE_CMD: string[] = process.env.KILN_VOICE_CMD ? JSON.parse(process.env.KILN_VOICE_CMD) : [`${ROOT}/data/voice-venv/bin/python`, "-m", "voice"];
const VOICE_IDLE_MS = Number(process.env.KILN_VOICE_IDLE_MS ?? 60_000);
interface VoiceSocket { up?: WebSocket; queue: (string | Buffer<ArrayBuffer>)[]; closed: boolean }
let speech: { proc: Subprocess<"pipe", "pipe", "inherit">; url: Promise<string> } | undefined;
let voiceSockets = 0, voiceIdle: Timer | undefined;

function speechService(): Promise<string> {
  if (speech) return speech.url;
  const missing = (why: string) => new Error(`The voice service did not start (${why}): run \`make voice\` to install it`);
  let proc: Subprocess<"pipe", "pipe", "inherit">;
  try {
    proc = Bun.spawn(VOICE_CMD, { cwd: ROOT, stdin: "pipe", stdout: "pipe", stderr: "inherit" });
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
      void (async () => { for (let r = await reader.read(); !r.done; r = await reader.read()) process.stdout.write(r.value); })();   // its log, and a pipe that never fills
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
    setTimeout(() => s.proc.kill(), 3000);             // one that does not watch its stdin
  }, VOICE_IDLE_MS);
}

// With `kiln start --lan` (KILN_HOST 0.0.0.0) this machine keeps plain HTTP on 127.0.0.1 and the network gets HTTPS on
// the same port (each of this machine's addresses: ufw already lets the port in): browsers give the microphone only to
// secure pages, which localhost is and http://<address> is not (tls.ts).
const LAN = process.env.KILN_HOST === "0.0.0.0", tls = LAN ? lanTls() : undefined;
const serve = (hostname: string, port: number, secure?: { cert: string; key: string }) => Bun.serve<VoiceSocket>({
  hostname,
  ...(secure ? { tls: secure } : {}),
  maxRequestBodySize: 1 << 30,                              // a chat resends its files: a 64 MB video is ~85 MB of base64
  port,
  development: process.env.NODE_ENV !== "production",
  routes: {
    "/": app,
    // the authority behind the network's certificate, for a device to install once and then trust the page
    "/kiln-ca.crt": { GET: guarded(() => tls ? new Response(tls.ca, { headers: { "Content-Type": "application/x-x509-ca-cert" } })
      : new Response("not on the network (kiln start --lan)", { status: 404 })) },
    "/vendor/mermaid/*": { GET: guarded(vendor) },
    "/sandbox": { GET: guarded(() => new Response(SANDBOX, { headers: { "Content-Type": "text/html; charset=utf-8", "Content-Security-Policy": SANDBOX_CSP } })) },

    // A streamed reply starts a job and follows it (its id in X-Kiln-Job); other requests (session titles) pass
    // through. Nothing arrives before the first token, which takes a while for a long prompt: no idle timeout. JSON
    // only: a reply can run tools, which another site's page must not get to ask for.
    "/api/chat": {
      POST: guarded(json(async (req, srv) => {
        srv.timeout(req, 0);
        const body = await req.text();
        if ((() => { try { return JSON.parse(body).stream; } catch { return false; } })()) return follow(startJob(body), 0);
        try {
          const r = await fetch(`${KILN}/v1/chat/completions`, {
            method: "POST", body, headers: { "Content-Type": "application/json" }, signal: req.signal,
          });
          return new Response(r.body, { status: r.status, headers: { "Content-Type": r.headers.get("Content-Type") ?? "application/json" } });
        } catch {
          return Response.json({ error: { message: "kiln is not running" } }, { status: 502 });
        }
      })),
    },
    // What the model is sent (Settings): the next reply's request as kiln would get it, without asking kiln. JSON only,
    // as /api/chat: it starts the chat's MCP servers to list their tools.
    "/api/preview": {
      POST: guarded(json(async (req) => {
        const { req: body, said } = await asSent(await req.json().catch(() => ({})));
        return Response.json({ body, voice: said });
      })),
    },
    // a reply again, from event `from` on (a page that reloaded, woke up or came back to the chat)
    "/api/chat/:id": {
      GET: guarded((req, srv) => {
        srv.timeout(req, 0);
        const job = jobs.get((req as Request & { params: { id: string } }).params.id);
        return job ? follow(job, Number(new URL(req.url).searchParams.get("from") ?? 0)) : Response.json({ error: { message: "gone" } }, { status: 404 });
      }),
    },
    // the Stop button: kiln stops, the reply ends with a "stopped" event
    "/api/chat/:id/stop": {
      POST: guarded((req) => {
        jobs.get((req as Request & { params: { id: string } }).params.id)?.ctrl.abort();
        return new Response(null, { status: 204 });
      }),
    },
    // a voice reply begun ahead stands: its tools may run
    "/api/chat/:id/go": {
      POST: guarded((req) => {
        jobs.get((req as Request & { params: { id: string } }).params.id)?.go?.();
        return new Response(null, { status: 204 });
      }),
    },
    // the user's answer to a tool call that asks first (any page following the reply may give it)
    "/api/chat/:id/tool/:call": {
      POST: guarded(json(async (req) => {
        const { id, call } = (req as Request & { params: { id: string; call: string } }).params;
        const decision = (await req.json().catch(() => ({})))?.decision, ask = jobs.get(id)?.asks.get(call);
        if (!["once", "always", "deny"].includes(decision)) return Response.json({ error: { message: "decision: once, always or deny" } }, { status: 400 });
        if (!ask) return Response.json({ error: { message: "that call is not waiting for an answer" } }, { status: 409 });
        ask(decision);
        return new Response(null, { status: 204 });
      })),
    },

    // The tool sources (Fetch URL, the config file's MCP servers) and their tools. The page can start one, retry one
    // and reload the file; what a server runs comes from the file only (tools.ts).
    "/api/tools": { GET: guarded(() => Response.json(registry.catalog())) },
    "/api/tools/reload": { POST: guarded(json(async () => (await registry.reload(), Response.json(registry.catalog())))) },
    "/api/tools/connect": {
      POST: guarded(json(async (req) => {
        const s = registry.source(String((await req.json().catch(() => ({})))?.source));
        if (!s) return Response.json({ error: { message: "no such tool source" } }, { status: 404 });
        void s.connect();
        return Response.json(registry.catalog());
      })),
    },

    // Voice mode: the page's socket to the speech service. A WebSocket is not held to CORS, so another site's page
    // could open one to this address and drive the service: only this server's own pages may.
    "/api/voice": guarded((req, srv) => {
      let same = false;
      try { same = new URL(req.headers.get("origin") ?? "").host === req.headers.get("host"); } catch { /* no origin, or "null" */ }
      if (!same) return new Response("only kiln's own pages", { status: 403 });
      return srv.upgrade(req, { data: { queue: [], closed: false } }) ? undefined : new Response("a WebSocket only", { status: 426 });
    }),
    "/voice-worklet.js": { GET: guarded(() => new Response(Bun.file(`${import.meta.dir}/voice-worklet.js`), { headers: { "Content-Type": "text/javascript" } })) },

    // online: serving; loading: the systemd unit runs but the model is still loading (kiln answers 503 until it has
    // loaded, or does not listen yet); offline: stopped
    "/api/status": guarded(async () => {
      try {
        const r = await fetch(`${KILN}/kiln/stats`);
        if (r.status === 503) {                         // listening, still loading its first model
          const b = await r.json().catch(() => ({}));
          return Response.json({ state: "loading", build: BUILD, loading: b.model ?? undefined, progress: b.progress ?? undefined });
        }
        if (!r.ok) throw new Error("loading");
        return Response.json({ state: "online", build: BUILD, ...(await r.json()) });
      } catch {
        return Response.json({ state: unit("is-active") === "active" ? "loading" : "offline", build: BUILD });
      }
    }),
    "/api/kiln/start": { POST: guarded(async (req) => (await pickModel(req), unit("start", "--no-block"), new Response(null, { status: 202 }))) },
    // switch the running kiln to another model; answers once it has loaded (up to ~20 s), the page polls meanwhile
    "/api/kiln/load": {
      POST: guarded(async (req, srv) => {
        srv.timeout(req, 0);
        const model = await pickModel(req);
        if (!model) return Response.json({ error: { message: "unknown model" } }, { status: 400 });
        try {
          const r = await fetch(`${KILN}/kiln/load`, { method: "POST", body: JSON.stringify({ model }), headers: { "Content-Type": "application/json" } });
          return new Response(r.body, { status: r.status, headers: { "Content-Type": "application/json" } });
        } catch {
          return Response.json({ error: { message: "kiln is not running" } }, { status: 502 });
        }
      }),
    },
    // stopping takes ~40 ms: wait for it, so the page can show the new state right away
    "/api/kiln/stop": { POST: guarded(() => (unit("stop"), new Response(null, { status: 204 }))) },
  },
  // the page's frames go to the service as they come (queued until its socket opens), and its frames come back
  websocket: {
    open(ws) {
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
      const up = ws.data.up;
      if (up?.readyState === WebSocket.OPEN) up.send(m);
      else ws.data.queue.push(m);
    },
    close(ws) {
      ws.data.closed = true;
      ws.data.up?.close();
      voiceClosed();
    },
  },
});

const server = serve(LAN ? "127.0.0.1" : process.env.KILN_HOST ?? "127.0.0.1", Number(process.env.PORT ?? 8081));
// only the certificate and key: Bun takes a `ca` there as the authority clients must show certificates of
const secure = tls ? lanAddresses().map((ip) => serve(ip, server.port!, { cert: tls.cert, key: tls.key })) : [];
console.log(`kiln web: ${server.url}`);
for (const s of secure) console.log(`kiln web: ${s.url}`);
