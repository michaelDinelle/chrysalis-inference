// Chrysalis web UI: serves the chat app, sends its replies' requests to the backend (Chrysalis, or another with OpenAI's
// API: backend.ts; from the same origin, so no CORS), runs the tools replies call (tools.ts) and proxies Chrysalis's API
// for the network. Conversations stay in the browser tab, replies being written stay in RAM (see
// Job): on disk are only its certificates (tls.ts), the browsers and devices let in (devices.ts) and the backend (backend.ts). `bun server.ts` for development, NODE_ENV=production for a minified bundle.
import { hostname } from "node:os";
import type { BunPlugin, Server, Subprocess } from "bun";
import app from "./index.html";
import { Backends, standard } from "./backend";
import { Devices, keyCookie } from "./devices";
import { appleProfile, connectPage, deviceScript, TRUSTED_GIF } from "./setup";
import { SUBAGENT_SYSTEM } from "./tool-agent";
import * as registry from "./tools";
import { lanAddresses, routeAddress, webTls } from "./tls";
import { assistantTurn, DESCRIPTIONS, toolTurn, unnoted, type ToolChoice, type WireCall } from "./wire";

const CHRYSALIS = process.env.CHRYSALIS_URL ?? "http://127.0.0.1:8080";
// Changes with the page's code. Open tabs compare it on every status poll and reload to pick up an update
// (their chats are in sessionStorage, so a reload keeps them).
const BUILD = Bun.hash((await Promise.all(["index.html", "app.ts", "sampler.ts", "files.ts", "wire.ts", "voice.ts", "voice-worklet.js", "sandbox.js", "heard.ts", "spoken.ts", "ocean.ts", "theme.ts",
  "logo.svg", "logo-thin.svg", "style.css", "tokens.css"].map((f) => Bun.file(`${import.meta.dir}/${f}`).text()))).join("\0")).toString(36);
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
type Handler = (req: Request, srv: Srv) => Response | undefined | Promise<Response>;
const open = (handler: Handler) => (req: Request, srv: Srv) => (hostOk(req) ? handler(req, srv) : new Response("unknown host", { status: 403 }));
// From the network only the devices this computer let in (devices.ts) get Chrysalis's API, the chat's, the tools and
// voice, and a browser only while it shows it trusts Chrysalis (the trust check). This machine needs no key and lets
// devices in (what runs here reaches Chrysalis anyway), at 127.0.0.1 and at its own network addresses, but its browsers
// get the chat as any other does: once they show they trust Chrysalis, with no code to allow. Any other browser gets the
// connect page at / (connectPage), and the chat's own page checks again for one that has not shown it lately
// (check: true). (CHRYSALIS_SELF_AS_NETWORK: the tests' browsers here at a network address stand in for other devices.)
const SELF_AS_NETWORK = process.env.CHRYSALIS_SELF_AS_NETWORK === "1";
const self = (req: Request, srv: Srv) => {
  const at = srv.requestIP(req)?.address.replace(/^::ffff:/, "") ?? "";
  return at.startsWith("127.") || at === "::1" || (!SELF_AS_NETWORK && lanAddresses().includes(at));
};
const fromNetwork = (req: Request, srv: Srv) => !self(req, srv);
// What a browser gets before it trusts Chrysalis comes over connections past its warning, which a browser keeps using
// (and marks the page not secure) after it does: each one closes, so the chat opens on a connection of its own.
const PAST_WARNING = { Connection: "close" };
const guarded = (handler: Handler) => open((req, srv) => {
  if (!fromNetwork(req, srv)) return handler(req, srv);
  const found = devices?.find(req);
  if (found && devices!.trusted(found.device)) return handler(req, srv);
  return Response.json(found ? { error: { message: "this browser has not shown lately that it trusts Chrysalis: its page checks again" }, check: true }
    : { error: { message: "this device is not connected to Chrysalis: its page at / connects it" } }, { status: 401 });
});
// what only this computer's own screen does: let devices in
const here = (handler: Handler) => guarded((req, srv) => fromNetwork(req, srv)
  ? Response.json({ error: { message: "only on the computer Chrysalis runs on" } }, { status: 403 }) : handler(req, srv));
// A form on another site can post to this server's address, but not as JSON without asking first (CORS), which it is
// never allowed: so what starts tools or lets one run takes JSON only.
const json = (handler: (req: Request, srv: Srv) => Response | Promise<Response>) => (req: Request, srv: Srv) =>
  (req.headers.get("content-type") ?? "").startsWith("application/json") ? handler(req, srv) : new Response("JSON only", { status: 415 });

const unit = (...args: string[]) => Bun.spawnSync(["systemctl", "--user", ...args, "chrysalis"]).stdout.toString().trim();
// The model picker's choice, kept in the systemd user environment: chrysalis.sh serve loads it when Chrysalis
// (re)starts.
const MODELS = ["qwen3.8-27b", "qwen3.8-flash-next"];
async function pickModel(req: Request): Promise<string | undefined> {
  const model = (await req.json().catch(() => ({})))?.model;
  if (!MODELS.includes(model)) return undefined;
  Bun.spawnSync(["systemctl", "--user", "set-environment", `CHRYSALIS_MODEL=${model}`]);
  return model;
}

// A reply belongs to this server, not to the page that asked for it: the request to Chrysalis runs to its end whatever
// the page does (reloads, closes, sleeps with the laptop, loses Wi-Fi, opens another chat), and any page can follow it
// again from where it left off: every event has a position. Only an explicit stop ends a reply early. A reply stays
// in RAM until 15 minutes after it ends, so a page that comes back in time collects the rest; nothing goes to disk.
// With tools, a reply is rounds: Chrysalis's answer ends in tool calls, this server runs them and asks Chrysalis again
// with their results, so a reply with tools goes on while no page is open, and its events are all of its rounds.
interface Job {
  id: string;
  events: string[];                                 // Chrysalis's SSE data payloads and the tool events, the timings, [DONE]
  done: boolean;
  ctrl: AbortController;
  waiters: Set<() => void>;
  asks: Map<string, (d: Decision) => void>;         // tool calls waiting for the user, by call id
  agents: Map<string, AbortController>;             // sub-agents at work, by their call's id: each can be stopped alone
  t0: number; tFirst?: number; tAnswer?: number; tLast?: number;   // ms: start, first token, first answer token, last
  gen: number;                                      // ms between each round's first token and its last, summed
  held?: Promise<void>; go?: () => void;            // a voice reply begun ahead (HELD_MS): it runs no tool until go
}
type Decision = "once" | "always" | "deny";
const jobs = new Map<string, Job>();
const KEEP_MS = 15 * 60_000;
const MAX_ROUNDS = 24;
// A voice reply begun before its question was settled (chrysalis_voice.ahead) runs no tool until the page says it
// stands (/go): it may yet be dropped, and a tool can do things. One the page neither confirms nor stops is stopped.
const HELD_MS = 60_000;
// Fields of a request that only make sense for its first round (a continued or rewritten reply).
const FIRST_ONLY = ["continue_final_message", "chrysalis_original", "chrysalis_append_token"];

function startJob(body: string): Job {
  const job: Job = { id: crypto.randomUUID(), events: [], done: false, ctrl: new AbortController(), waiters: new Set(), asks: new Map(),
    agents: new Map(), t0: Date.now(), gen: 0 };
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
      if (!web && !said) return void await round(job, req, push);
      for (const note of notes) push({ note });
      const allow = new Set(web?.allow ?? []);
      let messages: object[] = req.messages ?? [];
      const route = said && tools.length && web?.route && backends.isChrysalis ? ROUTE : undefined;
      let called = false, force = false, preamble: string | undefined;
      for (let k = 0; ; k++) {
        if (k) push({ round: k });
        // a voice turn that needs a tool (the page's check, or the user asked for a search) starts with the call
        const first = k === 0 && route && (tool || route.asks.test(unnoted(textOf((messages.findLast((m: any) => m.role === "user") as any)?.content))));
        if (first) preamble = route.preambles[messages.length % route.preambles.length];
        const out = first || force ? await forced(job, req, messages, push, preamble) : await round(job, { ...req, messages }, push);
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
        const results = k + 1 < MAX_ROUNDS ? await Promise.all(out.calls.map((c) => call(job, c, tools, allow, web ?? { tools: {} }, push, req)))
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
      if (!job.ctrl.signal.aborted && !(e instanceof ChrysalisDown)) console.error(`chrysalis web: reply ${job.id} failed:`, e);
      push(job.ctrl.signal.aborted ? { stopped: true } : { error: { message: e instanceof ChrysalisDown ? "Chrysalis is not running" : e instanceof Error ? e.message : String(e) } });
    } finally {
      push({ timing: { start: job.t0, first: job.tFirst, answer: job.tAnswer, last: job.tLast, gen: job.gen } });
      push("[DONE]");
      job.done = true;
      setTimeout(() => jobs.delete(job.id), KEEP_MS);
    }
  })();
  return job;
}

// A reply's request as the backend is sent it: the page's own fields (chrysalis_web, chrysalis_voice) taken off, the
// chat's tools (chrysalis_web's choice), each with Chrysalis's description where it has one (wire.ts), and to any
// backend but Chrysalis only the standard fields (backend.ts). The reply's job and the page's view of what the model is
// sent (/api/preview) both build it here, so the view is what the backend gets.
async function asSent(raw: any) {
  const req = { ...raw }, web: ToolChoice | undefined = req.chrysalis_web, said = !!req.chrysalis_voice, tool = req.chrysalis_voice?.tool === true;
  const ahead = req.chrysalis_voice?.ahead === true;
  delete req.chrysalis_web;
  delete req.chrysalis_voice;
  const picked = web ? await registry.select(web.tools ?? {}) : { tools: [] as registry.Named[], notes: [] as string[] };
  // a spoken reply stays quick: no sub-agents in voice mode
  const tools = said ? picked.tools.filter((n) => n.source.kind !== "agent") : picked.tools, notes = picked.notes;
  if (tools.length) req.tools = tools.map((n) => registry.openai(n, DESCRIPTIONS[n.tool.name]));
  return { req: backends.isChrysalis ? req : standard(req), web, said, tool, ahead, tools, notes };
}

class ChrysalisDown extends Error {
  message = backends.atService ? "Chrysalis is not running" : `nothing answers at ${new URL(backends.url).origin}: is the backend running?`;
}
// The backend took the request and its answer broke off (Chrysalis's log says why: an exception in Chrysalis drops the
// connection)
class ChrysalisBroke extends Error { message = backends.isChrysalis ? "Chrysalis's reply broke off (Chrysalis's log has why)" : "the backend's reply broke off"; }
// A backend's error, which some send as a string and most as OpenAI's {message}
const errorOf = async (r: Response) => {
  const e = (await r.json().catch(() => null))?.error;
  return (typeof e === "string" ? e : e?.message) ?? `${backends.isChrysalis ? "Chrysalis" : new URL(backends.url).host} answered ${r.status}`;
};
// One request to the backend, its events passed on as they come: what it wrote, and the tool calls it ended with.
// timed: its tokens count in the reply's timings (a sub-agent's do not: they are not the reply's own words).
async function round(job: Job, body: object, push: (d: string | object) => void, signal = job.ctrl.signal, timed = true) {
  const r = await fetch(`${backends.url}/chat/completions`, {
    method: "POST", body: JSON.stringify(body), headers: backends.headers(), signal,
  }).catch((e) => { throw signal.aborted ? e : new ChrysalisDown(); });
  if (!r.ok || !r.body) {
    push({ error: { message: await errorOf(r) } });
    return null;
  }
  const out = { content: "", reasoning: "", calls: [] as WireCall[] };
  let first: number | undefined, last: number | undefined;
  const reader = r.body.getReader(), decoder = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { value, done } = await reader.read().catch((e) => { throw signal.aborted ? e : new ChrysalisBroke(); });
      if (done) break;
      buf += decoder.decode(value, { stream: true }).replace(/\r\n?/g, "\n");   // SSE also allows CRLF and CR
      for (let end = buf.indexOf("\n\n"); end >= 0; end = buf.indexOf("\n\n")) {
        let data = buf.slice(0, end).split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(l.startsWith("data: ") ? 6 : 5)).join("\n");
        buf = buf.slice(end + 2);
        if (!data || data === "[DONE]") continue;
        const now = Date.now(), ev = data.includes('"delta"') ? JSON.parse(data) : undefined, delta = ev?.choices?.[0]?.delta;
        // Ollama and vLLM send a reply's thinking as reasoning, llama.cpp, LM Studio, KoboldCpp and Chrysalis as reasoning_content
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
        push(data);
      }
    }
  } finally {
    if (first && last) job.gen += last - first;
  }
  return out;
}

// A voice turn (chrysalis_voice): whether its reply runs a tool is settled before the reply, not left to its words. The
// page asks beside the conversation (app.ts, heardFor: with what speech recognition misheard, at temperature 0) and
// says so here (chrysalis_voice.tool); a user who asks for a search (the route's asks) gets one whatever it said. A
// call is held to the tools' grammar from its first token (tool_choice "required"), after the spoken line that says
// what is happening (one of the route's preambles): a model with thinking off, told its reply is spoken, says "I'm
// searching for that now" and ends its turn often enough (and then copies that turn: 1 call in 6 after one such turn in
// the history), and may report results it never got. A reply that announces a tool it did not call (its announces,
// after "I'll", "let me", ...) is made to call one. Chrysalis's alone: llama.cpp, LM Studio, Ollama and KoboldCpp
// answered a thinking model's tool_choice "required" with text.
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
const FORCE = { tool_choice: "required", logprobs: false, top_logprobs: undefined };   // Chrysalis holds no grammar with logprobs
const textOf = (c: unknown) => (typeof c === "string" ? c : Array.isArray(c) ? c.map((p: any) => p?.text ?? "").join(" ") : "");

async function forced(job: Job, req: any, messages: object[], push: (d: string | object) => void, preamble?: string): Promise<Out | null> {
  if (preamble) {
    push({ choices: [{ index: 0, delta: { content: preamble } }] });
    job.tFirst ??= Date.now();
    job.tAnswer ??= Date.now();
  }
  const out = await round(job, { ...req, messages, ...FORCE }, push);
  if (out && preamble) out.content = preamble + out.content;
  return out;
}

// One tool call, from the model's arguments to the result it is given: refused when the chat did not turn the tool on,
// asked about when it is not read-only (unless the user allowed it for good), stopped by Stop (signal: the reply's, or a
// sub-agent's). Its events carry what the page shows of it; the result also goes to the model as its tool message. req:
// the reply's request, which a sub-agent's are made from.
async function call(job: Job, c: WireCall, tools: registry.Named[], allow: Set<string>, web: ToolChoice, push: (d: object) => void, req: any,
  signal = job.ctrl.signal): Promise<string> {
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
      if (signal.aborted) return resolve("stop");
      job.asks.set(c.id, resolve);
      signal.addEventListener("abort", () => resolve("stop"), { once: true });
    });
    job.asks.delete(c.id);
    if (d === "stop") return end("stopped", "Not run: the user stopped the reply before this call was allowed.");
    if (d === "deny") return end("declined", "The user declined this call: it did not run. Do not try it again unless they ask for it.");
    if (d === "always") allow.add(n.fn);
  }
  push({ tool: { ...about, state: "running" } });
  const t = Date.now();
  if (n.source.kind === "agent") {
    const [state, result] = await subagent(job, c, args, tools.filter((x) => x.source.kind !== "agent"), allow, web, push, req, signal);
    return end(state, result, Date.now() - t);
  }
  try {
    const result = await registry.run(n, args, signal, n.source.kind === "fetch" ? web.fetchChars : undefined);
    return end("done", result, Date.now() - t);
  } catch (e) {
    if (signal.aborted) return end("stopped", "Stopped: the user stopped the reply while this call ran.", Date.now() - t);
    return end("failed", `Error: ${e instanceof Error ? e.message : String(e)}`, Date.now() - t);
  }
}

// A sub-agent (tool-agent.ts): a conversation of its own on the task the reply handed it, with the chat's other tools,
// run to its end as a reply's rounds are (its calls ask, run and stop as the reply's do). Its events reach the page
// inside the call's (agent: {id, ev}); only its final answer goes back to the reply, as the call's result, so what it
// read stays out of the reply's context. It can be stopped alone (/api/chat/:id/agent/:call/stop), and then gives back
// what it had written.
async function subagent(job: Job, c: WireCall, args: Record<string, unknown>, tools: registry.Named[], allow: Set<string>,
  web: ToolChoice, push: (d: object) => void, req: any, outer: AbortSignal): Promise<["done" | "stopped" | "failed", string]> {
  const task = typeof args.prompt === "string" ? args.prompt.trim() : "";
  if (!task) return ["failed", "Error: the call needs a prompt: the task, with everything the sub-agent needs to know."];
  const ctrl = new AbortController(), signal = AbortSignal.any([outer, ctrl.signal]);
  job.agents.set(c.id, ctrl);
  let failure: string | undefined;
  const inner = (d: string | object) => {
    const ev = typeof d === "string" ? JSON.parse(d) : d;
    if (ev?.error?.message) failure = ev.error.message;
    push({ agent: { id: c.id, ev } });
  };
  // the reply's model and settings, not its messages, tools, seed or log probs (only the page's token view reads those)
  const base = { ...req };
  for (const f of ["messages", "tools", "tool_choice", "seed", "logprobs", "top_logprobs", ...FIRST_ONLY]) delete base[f];
  if (tools.length) base.tools = tools.map((n) => registry.openai(n, DESCRIPTIONS[n.tool.name]));
  let messages: object[] = [{ role: "system", content: SUBAGENT_SYSTEM }, { role: "user", content: task }], said = "";
  try {
    for (let k = 0; ; k++) {
      if (k) inner({ round: k });
      const out = await round(job, { ...base, messages }, inner, signal, false);
      if (!out) return ["failed", `Error: the sub-agent could not go on: ${failure ?? "Chrysalis refused its request"}`];
      said = out.content.trim();
      if (!out.calls.length) return ["done", said || "The sub-agent finished without writing an answer."];
      if (k + 1 >= MAX_ROUNDS) {
        inner({ note: `Stopped after ${MAX_ROUNDS} rounds of tool calls.` });
        return ["done", `${said}\n\n[The sub-agent stopped here: it reached its limit of ${MAX_ROUNDS} rounds of tool calls.]`.trim()];
      }
      const results = await Promise.all(out.calls.map((x) => call(job, x, tools, allow, web, inner, req, signal)));
      if (signal.aborted) throw new DOMException("stopped", "AbortError");
      messages = [...messages, assistantTurn(out.content, out.reasoning, out.calls), ...out.calls.map((x, i) => toolTurn(x.id, results[i]))];
    }
  } catch (e) {
    if (signal.aborted) {
      const by = outer.aborted ? "the user stopped the reply" : "the user stopped it";
      return ["stopped", `Stopped before it finished: ${by}.${said ? ` What it had written last:\n\n${said}` : ""}`];
    }
    return ["failed", `Error: the sub-agent could not go on: ${e instanceof ChrysalisDown ? "Chrysalis is not running" : e instanceof Error ? e.message : String(e)}`];
  } finally {
    job.agents.delete(c.id);
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
  return new Response(stream, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Chrysalis-Job": job.id } });
}

// Mermaid's ES module build (the page imports it the first time it draws a diagram): its entry and the chunks it
// loads, straight from the package and nothing else from node_modules.
const MERMAID = `${import.meta.dir}/node_modules/mermaid/dist/`;
const vendor = (req: Request) => {
  const file = new URL(req.url).pathname.slice("/vendor/mermaid/".length);
  if (!/^(mermaid\.esm\.min\.mjs|chunks\/mermaid\.esm\.min\/[\w-]+\.mjs)$/.test(file)) return new Response("not found", { status: 404 });
  return new Response(Bun.file(MERMAID + file), { headers: { "Content-Type": "text/javascript", "Cache-Control": "max-age=86400" } });
};

// three.js and A-Frame for pages, which reach nothing: the chat hands a page the code it loads from their usual
// addresses (app.ts, libsOf), from here. three's files import one another, so each is bundled into one: three itself,
// an addon (examples/jsm) that imports "three" as the page's import map has it, and the classic builds older pages load
// (a global THREE, and addons that add to it). A-Frame's build has three inside.
const THREE_DIR = `${import.meta.dir}/node_modules/three/`;
const AFRAME = `${import.meta.dir}/node_modules/aframe/dist/aframe-v1.8.0.min.js`;
const bundles = new Map<string, Promise<string | null>>();
let threeNames: Promise<string[]> | undefined;
// A module made here (the entry of a classic build), and "three" as the global THREE where an addon is made classic
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

// The frame an HTML page from a reply runs in. Two walls: the page's iframe is sandboxed (allow-scripts only: an opaque
// origin, so no cookies, storage or access to the chat), and this policy lets its own inline code run but nothing
// load from or talk to anywhere, Chrysalis included. The chat posts the page in (sandbox.js).
const SANDBOX = `<!doctype html><meta charset="utf-8"><script>${await Bun.file(`${import.meta.dir}/sandbox.js`).text()}</script>`;
// (blob: scripts: the libraries the chat hands a page, which it loads from blob: addresses it makes itself; such an address
// holds only what the page could run anyway)
const SANDBOX_CSP = "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' blob:; style-src 'unsafe-inline'; img-src data: blob:; " +
  "font-src data:; media-src data: blob:; form-action 'none'; base-uri 'none'; frame-ancestors 'self'; sandbox allow-scripts allow-modals";

// Voice mode's speech service: a process of its own (voice/ in data/voice-venv, or any stack that speaks the same
// protocol: CHRYSALIS_VOICE_CMD, a JSON argv), started by the first voice socket and stopped CHRYSALIS_VOICE_IDLE_MS
// after the last one closes, so voice mode holds no memory while nobody uses it. It prints `voice: ws://...` once it
// listens and exits when its stdin closes (this server gone, or the idle time up). Each page's socket is proxied to it
// verbatim.
const ROOT = `${import.meta.dir}/..`;
const VOICE_CMD: string[] = process.env.CHRYSALIS_VOICE_CMD ? JSON.parse(process.env.CHRYSALIS_VOICE_CMD) : [`${ROOT}/data/voice-venv/bin/python`, "-m", "voice"];
const VOICE_IDLE_MS = Number(process.env.CHRYSALIS_VOICE_IDLE_MS ?? 60_000);
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

// HTTPS always (tls.ts), on 127.0.0.1, and with `chrysalis start --lan` (CHRYSALIS_HOST 0.0.0.0) at each of this machine's
// network addresses on the same port too (ufw already lets the port in)
const LAN = process.env.CHRYSALIS_HOST === "0.0.0.0", tls = webTls();
const devices = new Devices(`${tls.dir}/devices.json`), backends = new Backends(`${tls.dir}/backend.json`, `${CHRYSALIS}/v1`), sea = await deviceScript();
const PAGE = { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" };
// only the certificate and key: Bun takes a `ca` there as the authority clients must show certificates of
const serve = (hostname: string, port: number) => Bun.serve<VoiceSocket>({
  hostname,
  tls: { cert: tls.cert, key: tls.key },
  maxRequestBodySize: 1 << 30,                              // a chat resends its files: a 64 MB video is ~85 MB of base64
  port,
  development: process.env.NODE_ENV !== "production",
  routes: {
    // the chat's page and its files, or the connect page (setup.ts)
    "/": { GET: chatPage },
    "/*": { GET: chatFile, HEAD: chatFile },
    // the authority, as each kind of device takes it (the connect page's guides): over HTTPS, past the warning the device
    // shows until it trusts it, as Chromium holds back a plain-HTTP page's downloads. PEM for the rest: apps' clients
    // (NODE_EXTRA_CA_CERTS, SSL_CERT_FILE) read only that, and the systems' installers read it too.
    "/chrysalis.mobileconfig": { GET: open(() => authority((a) => appleProfile(a), "application/x-apple-aspen-config", "chrysalis.mobileconfig")) },
    "/chrysalis-ca.crt": { GET: open(() => authority((a) => a.ca, "application/octet-stream", "chrysalis-ca.crt")) },
    "/chrysalis-ca-firefox.crt": { GET: open(() => authority((a) => a.caDer, "application/x-x509-ca-cert")) },   // Firefox asks to trust it
    // the connect page asks, once its nonce came through the trust check: the code it shows, and the secret it waits with
    // (a browser let in that has just shown it trusts Chrysalis again goes in)
    "/pair/ask": {
      POST: open(json(async (req, srv) => {
        const found = devices.find(req);
        if (found && devices.trusted(found.device)) return Response.json({ in: true }, { headers: PAST_WARNING });
        const b = await req.json().catch(() => ({})), nonce = String(b?.nonce ?? ""), name = String(b?.name ?? "");
        const untrusted = () => Response.json({ error: { message: "this browser has not shown that it trusts Chrysalis" } }, { status: 403, headers: PAST_WARNING });
        if (self(req, srv)) {                           // this machine's browser: in, with no code
          const key = devices.here(nonce, `${name} (this computer)`);
          return key ? Response.json({ in: true }, { headers: { ...PAST_WARNING, "Set-Cookie": keyCookie(key) } }) : untrusted();
        }
        const ask = devices.ask(nonce, name);
        return ask ? Response.json(ask, { headers: PAST_WARNING }) : untrusted();
      })),
    },
    // the asking page's answer: waiting, or allowed (its key, once), or gone (denied, or out of time)
    "/pair/ask/:secret": {
      GET: open((req, srv) => {
        const got = fromNetwork(req, srv) ? devices?.collect((req as Request & { params: { secret: string } }).params.secret) : undefined;
        if (!got) return Response.json({ state: "gone" }, { status: 404, headers: PAST_WARNING });
        return Response.json(got.state === "allowed" ? { state: "allowed" } : got,
          { headers: { ...PAST_WARNING, ...(got.state === "allowed" ? { "Set-Cookie": keyCookie(got.key) } : {}) } });
      }),
    },
    // The chat page's own trust check: its policy lets images in from itself only (a reply's markdown must not reach
    // anywhere), so it checks in a frame of its own whose policy lets the check's image in, and is told how it went.
    "/check-frame": {
      GET: open((req) => new Response(`<!doctype html><meta charset="utf-8"><script>
const img = new Image();
img.onload = () => parent.postMessage({ chrysalisCheck: true }, location.origin);
img.onerror = () => parent.postMessage({ chrysalisCheck: false }, location.origin);
img.src = ${JSON.stringify(checkUrl(req))} + "?t=" + Date.now();
</script>`, { headers: { ...PAGE, "Content-Security-Policy": `default-src 'none'; script-src 'unsafe-inline'; img-src ${new URL(checkUrl(req)).origin}; ` +
        "frame-ancestors 'self'" } })),
    },
    // Connect a device (the page here, chrysalis.sh): where devices connect (a QR code of it, when qrencode is
    // installed), the devices asking to join and those let in, and allowing or denying one by its code
    "/api/pair": {
      GET: here(async () => {
        if (!LAN) return Response.json({ lan: false });
        const connect = await connectUrls();
        return Response.json({ lan: true, connect, qr: qrSvg(connect[0]), api: connect.map((u) => `${u}v1`), waiting: devices.waiting(),
          devices: devices.all(), name: tls.caName, fingerprint: tls.fingerprint });
      }),
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
    // a key for an app on another computer (Connect a device, `chrysalis key`), shown once
    "/api/pair/key": {
      POST: here(json(async (req) => {
        if (!LAN) return Response.json({ error: { message: "not on the network (chrysalis start --lan)" } }, { status: 409 });
        const name = String((await req.json().catch(() => ({})))?.name ?? "").trim();
        return Response.json(devices.add(name || "An app", "app"));
      })),
    },
    "/api/pair/:id": { DELETE: here((req) => new Response(null, { status: devices?.remove((req as Request & { params: { id: string } }).params.id) ? 204 : 404 })) },
    "/v1/*": guarded(proxy),
    "/vendor/mermaid/*": { GET: guarded(vendor) },
    "/vendor/three/*": { GET: guarded(vendor3d) },
    "/vendor/aframe/*": { GET: guarded(vendor3d) },
    "/sandbox": { GET: guarded(() => new Response(SANDBOX, { headers: { "Content-Type": "text/html; charset=utf-8", "Content-Security-Policy": SANDBOX_CSP } })) },

    // A streamed reply starts a job and follows it (its id in X-Chrysalis-Job); other requests (session titles) pass
    // through. Nothing arrives before the first token, which takes a while for a long prompt: no idle timeout. JSON
    // only: a reply can run tools, which another site's page must not get to ask for.
    "/api/chat": {
      POST: guarded(json(async (req, srv) => {
        srv.timeout(req, 0);
        const body = await req.text();
        if ((() => { try { return JSON.parse(body).stream; } catch { return false; } })()) return follow(startJob(body), 0);
        try {
          const r = await fetch(`${backends.url}/chat/completions`, {
            method: "POST", body: backends.isChrysalis ? body : JSON.stringify(standard(JSON.parse(body))), headers: backends.headers(), signal: req.signal,
          });
          return new Response(r.body, { status: r.status, headers: { "Content-Type": r.headers.get("Content-Type") ?? "application/json" } });
        } catch {
          return Response.json({ error: { message: new ChrysalisDown().message } }, { status: 502 });
        }
      })),
    },
    // What the model is sent (Settings): the next reply's request as Chrysalis would get it, without asking Chrysalis.
    // JSON only, as /api/chat: it starts the chat's MCP servers to list their tools.
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
    // the Stop button: Chrysalis stops, the reply ends with a "stopped" event
    "/api/chat/:id/stop": {
      POST: guarded((req) => {
        jobs.get((req as Request & { params: { id: string } }).params.id)?.ctrl.abort();
        return new Response(null, { status: 204 });
      }),
    },
    // one sub-agent stops (Stop in its card): the reply goes on with what it had written
    "/api/chat/:id/agent/:call/stop": {
      POST: guarded((req) => {
        const { id, call } = (req as Request & { params: { id: string; call: string } }).params;
        jobs.get(id)?.agents.get(call)?.abort();
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
      if (!same) return new Response("only Chrysalis's own pages", { status: 403 });
      return srv.upgrade(req, { data: { queue: [], closed: false } }) ? undefined : new Response("a WebSocket only", { status: 426 });
    }),
    "/voice-worklet.js": { GET: guarded(() => new Response(Bun.file(`${import.meta.dir}/voice-worklet.js`), { headers: { "Content-Type": "text/javascript" } })) },

    // online: serving; loading: the systemd unit runs but the model is still loading (Chrysalis answers 503 until it
    // has loaded, or does not listen yet); offline: stopped
    // A device's key is set again here, which every open page polls: its 400 days count from its last visit. A browser
    // past half its time since its trust check is told to check again (check), before the gate has to; this machine's
    // page hears how many devices are asking to join (waiting).
    "/api/status": guarded(async (req, srv) => {
      const backend = await backends.about();
      const state = backend.listed ? { state: backend.listed.ok ? "online" : "offline", build: BUILD } : await (async () => {
        try {
          const r = await fetch(`${CHRYSALIS}/chrysalis/stats`);
          if (r.status === 503) {                       // listening, still loading its first model
            const b = await r.json().catch(() => ({}));
            return { state: "loading", build: BUILD, loading: b.model ?? undefined, progress: b.progress ?? undefined };
          }
          if (!r.ok) throw new Error("loading");
          return { state: "online", build: BUILD, ...(await r.json()) };
        } catch {
          return { state: unit("is-active") === "active" ? "loading" : "offline", build: BUILD };
        }
      })(), net = fromNetwork(req, srv), found = devices.find(req);
      const res = Response.json({ ...state, backend, here: !net, ...(found && devices!.stale(found.device) ? { check: true } : {}),
        ...(LAN && !net ? { waiting: devices.waiting().length } : {}) });
      if (found) res.headers.append("Set-Cookie", keyCookie(found.key));
      return res;
    }),
    // Where replies come from (backend.ts): set on this computer's own screen only, checked before it is kept
    "/api/backend": {
      POST: here(json(async (req) => {
        const b = await req.json().catch(() => ({}));
        const r = await backends.set(String(b?.url ?? ""), typeof b?.key === "string" ? b.key : undefined, b?.confirmRemote === true);
        return r.ok ? Response.json(await backends.about()) : Response.json({ error: { message: r.error }, ...(r.remote ? { remote: r.remote } : {}) }, { status: r.remote ? 409 : 400 });
      })),
    },
    "/api/chrysalis/start": { POST: guarded(async (req) => (await pickModel(req), unit("start", "--no-block"), new Response(null, { status: 202 }))) },
    // switch the running Chrysalis to another model; answers once it has loaded (up to ~20 s), the page polls meanwhile
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
    // stopping takes ~40 ms: wait for it, so the page can show the new state right away
    "/api/chrysalis/stop": { POST: guarded(() => (unit("stop"), new Response(null, { status: 204 }))) },
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

// The chat's page and its files (Bun's bundle of index.html): only for a browser let in that has shown lately it trusts
// Chrysalis (this machine's too). Any other browser gets the connect page in the chat's place, before any of the chat
// loads, and none of its files. Bun serves a bundle only as a route of a server's own, so it has one, on 127.0.0.1 at a
// port of its own, which these two ask (its files are no secret to what runs here).
const letIn = (req: Request) => {
  const found = devices.find(req);
  return !!found && devices.trusted(found.device);
};
const bundle = Bun.serve({ hostname: "127.0.0.1", port: 0, routes: { "/": app }, fetch: () => new Response("not found", { status: 404 }),
  development: process.env.NODE_ENV !== "production" && { hmr: false } });   // no socket of its own to reload pages through
async function local(req: Request): Promise<Response> {
  const url = new URL(req.url), r = await fetch(`http://127.0.0.1:${bundle.port}${url.pathname}${url.search}`, { method: req.method });
  const back = new Headers(r.headers);
  for (const h of ["content-encoding", "content-length", "transfer-encoding", "connection"]) back.delete(h);   // fetch undid them
  if (url.pathname === "/") back.set("Cache-Control", "no-store");                // a device removed keeps no copy to open
  return new Response(r.body, { status: r.status, headers: back });
}
const chatPage = open((req, srv) => letIn(req) ? local(req) : new Response(connectPage(tls, checkUrl(req), sea, self(req, srv)), { headers: { ...PAGE, ...PAST_WARNING,
  "Content-Security-Policy":
  `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src ${new URL(checkUrl(req)).origin} data:; connect-src 'self'; ` +
  "base-uri 'none'; form-action 'none'; frame-ancestors 'none'" } }));
// any other address opened in the browser goes to the connect page
const chatFile = open((req) => letIn(req) ? local(req) : req.headers.get("sec-fetch-mode") === "navigate"
  ? new Response(null, { status: 302, headers: { Location: "/", "Cache-Control": "no-store", ...PAST_WARNING } }) : new Response("not found", { status: 404, headers: PAST_WARNING }));
const authority = (body: (a: typeof tls) => Uint8Array<ArrayBuffer> | string, type: string, name?: string) =>
  new Response(body(tls), { headers: { "Content-Type": type, "Cache-Control": "no-store", ...PAST_WARNING, ...(name ? { "Content-Disposition": `attachment; filename="${name}"` } : {}) } });

const server = serve("127.0.0.1", Number(process.env.PORT ?? 8081));
const secure = LAN ? lanAddresses().map((ip) => serve(ip, server.port!)) : [];
// The trust check (devices.ts): HTTPS on a port of its own at each address, with a certificate of its own made at this
// start. Only a browser that trusts the authority gets a request through: the image marks its key, and a connect page's
// nonce, as having come through.
let checkPort = Number(process.env.CHRYSALIS_CHECK_PORT ?? 8083);
const checks = ["127.0.0.1", ...(LAN ? lanAddresses() : [])].map((ip) => {
  const s = Bun.serve({ hostname: ip, port: checkPort, tls: tls.check, fetch: checkFetch });
  checkPort = s.port!;
  return s;
});
const checkUrl = (req: Request) => `https://${new URL(req.url).hostname}:${checkPort}/check.gif`;
console.log(`chrysalis web: ${server.url}`);
for (const s of secure) console.log(`chrysalis web: ${s.url}`);
for (const s of checks) console.log(`chrysalis check: ${s.url}`);

function checkFetch(req: Request): Response {
  if (!hostOk(req)) return new Response("unknown host", { status: 403 });
  const url = new URL(req.url);
  if (url.pathname !== "/check.gif") return new Response("not found", { status: 404 });
  devices.check(req, url.searchParams.get("n"));
  return new Response(TRUSTED_GIF, { headers: { "Content-Type": "image/gif", "Cache-Control": "no-store" } });
}

// Chrysalis's API (OpenAI's, Anthropic's, the Responses API) for the network: Chrysalis itself listens on this machine
// only, so a device reaches it here, over HTTPS, with its key (guarded). A reply streams through as it comes, and an app
// that hangs up stops it (its request's signal ends Chrysalis's).
async function proxy(req: Request, srv: Srv): Promise<Response> {
  srv.timeout(req, 0);                                            // a long prompt is read before anything comes back
  const url = new URL(req.url), headers = new Headers(req.headers);
  for (const h of ["host", "cookie", "authorization", "x-api-key", "connection", "content-length"]) headers.delete(h);
  try {
    const r = await fetch(`${CHRYSALIS}${url.pathname}${url.search}`, { method: req.method, headers, signal: req.signal,
      body: req.method === "GET" || req.method === "HEAD" ? undefined : await req.arrayBuffer() });
    const back = new Headers(r.headers);
    for (const h of ["content-encoding", "content-length", "transfer-encoding", "connection"]) back.delete(h);   // fetch undid them
    return new Response(r.body, { status: r.status, headers: back });
  } catch {
    return Response.json({ error: { message: "Chrysalis is not running" } }, { status: 502 });
  }
}
// The chat at each address, the one other devices reach this machine at first
async function connectUrls(): Promise<string[]> {
  const first = await routeAddress();
  return lanAddresses().sort((a, b) => Number(b === first) - Number(a === first)).map((ip) => `https://${ip}:${server.port}/`);
}

// A QR code of a URL as SVG, by qrencode where it is installed (once per URL: the dialog asks every 2 s)
const qrs = new Map<string, string | undefined>();
function qrSvg(url: string): string | undefined {
  if (qrs.has(url)) return qrs.get(url);
  try {
    const r = Bun.spawnSync(["qrencode", "-t", "SVG", "-m", "2", "-o", "-", url], { stdout: "pipe", stderr: "ignore" });
    qrs.set(url, r.exitCode === 0 ? r.stdout.toString() : undefined);
    return qrs.get(url);
  } catch {
    return undefined;
  }
}
