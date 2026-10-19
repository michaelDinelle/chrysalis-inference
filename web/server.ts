// kiln web UI: serves the chat app, proxies the kiln server's OpenAI API (same origin, so no CORS) and runs the tools
// replies call (tools.ts). It stores nothing on disk: conversations stay in the browser tab, replies being written stay
// in RAM (see Job). `bun server.ts` for development, NODE_ENV=production for a minified bundle.
import { hostname } from "node:os";
import type { Server } from "bun";
import app from "./index.html";
import * as registry from "./tools";
import { assistantTurn, toolTurn, type ToolChoice, type WireCall } from "./wire";

const KILN = process.env.KILN_URL ?? "http://127.0.0.1:8080";
// Changes with the page's code. Open tabs compare it on every status poll and reload to pick up an update
// (their chats are in sessionStorage, so a reload keeps them).
const BUILD = Bun.hash((await Promise.all(["index.html", "app.ts", "sampler.ts", "files.ts", "wire.ts", "style.css"].map((f) => Bun.file(`${import.meta.dir}/${f}`).text()))).join("\0")).toString(36);
await registry.reload();
for (const sig of ["SIGINT", "SIGTERM"] as const) process.on(sig, () => void registry.closeAll().finally(() => process.exit(0)));   // MCP servers' processes go too

// DNS rebinding guard: a page from another site that points its own name at this machine still sends that name as
// Host. Requests addressed to an IP address or to this machine's names are fine.
const NAMES = new Set(["", "localhost", hostname().toLowerCase(), `${hostname().toLowerCase()}.local`]);
const hostOk = (req: Request) => {
  const host = (req.headers.get("host") ?? "").toLowerCase();
  return host.startsWith("[") || /^\d{1,3}(\.\d{1,3}){3}(:\d+)?$/.test(host) || NAMES.has(host.replace(/:\d+$/, ""));
};
const guarded = (handler: (req: Request, srv: Server<undefined>) => Response | Promise<Response>) =>
  (req: Request, srv: Server<undefined>) => (hostOk(req) ? handler(req, srv) : new Response("unknown host", { status: 403 }));
// A form on another site can post to this server's address, but not as JSON without asking first (CORS), which it is
// never allowed: so what starts tools or lets one run takes JSON only.
const json = (handler: (req: Request, srv: Server<undefined>) => Response | Promise<Response>) => (req: Request, srv: Server<undefined>) =>
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
}
type Decision = "once" | "always" | "deny";
const jobs = new Map<string, Job>();
const KEEP_MS = 15 * 60_000;
const MAX_ROUNDS = 24;
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
      const req = JSON.parse(body), web: ToolChoice | undefined = req.kiln_web;
      delete req.kiln_web;
      if (!web) return void await round(job, JSON.stringify(req), push);
      const { tools, notes } = await registry.select(web.tools ?? {});
      for (const note of notes) push({ note });
      if (tools.length) req.tools = tools.map(registry.openai);
      const allow = new Set(web.allow ?? []);
      let messages: object[] = req.messages ?? [];
      for (let k = 0; ; k++) {
        if (k) push({ round: k });
        const out = await round(job, JSON.stringify({ ...req, messages }), push);
        if (!out || !out.calls.length || web.loop === false || job.ctrl.signal.aborted) break;
        const results = k + 1 < MAX_ROUNDS ? await Promise.all(out.calls.map((c) => call(job, c, tools, allow, web, push)))
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
      push(job.ctrl.signal.aborted ? { stopped: true } : { error: { message: e instanceof KilnDown || !(e instanceof Error) ? "kiln is not running" : e.message } });
    } finally {
      push({ timing: { start: job.t0, first: job.tFirst, answer: job.tAnswer, last: job.tLast, gen: job.gen } });
      push("[DONE]");
      job.done = true;
      setTimeout(() => jobs.delete(job.id), KEEP_MS);
    }
  })();
  return job;
}

class KilnDown extends Error {}
// One request to kiln, its events passed on as they come: what it wrote, and the tool calls it ended with.
async function round(job: Job, body: string, push: (d: string | object) => void) {
  const r = await fetch(`${KILN}/v1/chat/completions`, {
    method: "POST", body, headers: { "Content-Type": "application/json" }, signal: job.ctrl.signal,
  }).catch((e) => { throw job.ctrl.signal.aborted ? e : new KilnDown(); });
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
      const { value, done } = await reader.read();
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
  } catch (e) {
    throw job.ctrl.signal.aborted ? e : new KilnDown();
  } finally {
    if (first && last) job.gen += last - first;
  }
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

const server = Bun.serve({
  hostname: process.env.KILN_HOST ?? "127.0.0.1",          // 0.0.0.0 after `kiln start --lan`
  maxRequestBodySize: 1 << 30,                              // a chat resends its files: a 64 MB video is ~85 MB of base64
  port: Number(process.env.PORT ?? 8081),
  development: process.env.NODE_ENV !== "production",
  routes: {
    "/": app,
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
});

console.log(`kiln web: ${server.url}`);
