// kiln web UI: serves the chat app and proxies the kiln server's OpenAI API (same origin, so no CORS). It stores
// nothing on disk: conversations stay in the browser tab, replies being written stay in RAM (see Job). `bun server.ts`
// for development, NODE_ENV=production for a minified bundle.
import { hostname } from "node:os";
import type { Server } from "bun";
import app from "./index.html";

const KILN = process.env.KILN_URL ?? "http://127.0.0.1:8080";
// Changes with the page's code. Open tabs compare it on every status poll and reload to pick up an update
// (their chats are in sessionStorage, so a reload keeps them).
const BUILD = Bun.hash((await Promise.all(["index.html", "app.ts", "style.css"].map((f) => Bun.file(`${import.meta.dir}/${f}`).text()))).join("\0")).toString(36);

// DNS rebinding guard: a page from another site that points its own name at this machine still sends that name as
// Host. Requests addressed to an IP address or to this machine's names are fine.
const NAMES = new Set(["", "localhost", hostname().toLowerCase(), `${hostname().toLowerCase()}.local`]);
const hostOk = (req: Request) => {
  const host = (req.headers.get("host") ?? "").toLowerCase();
  return host.startsWith("[") || /^\d{1,3}(\.\d{1,3}){3}(:\d+)?$/.test(host) || NAMES.has(host.replace(/:\d+$/, ""));
};
const guarded = (handler: (req: Request, srv: Server<undefined>) => Response | Promise<Response>) =>
  (req: Request, srv: Server<undefined>) => (hostOk(req) ? handler(req, srv) : new Response("unknown host", { status: 403 }));

const unit = (...args: string[]) => Bun.spawnSync(["systemctl", "--user", ...args, "kiln"]).stdout.toString().trim();
// The model picker's choice, kept in the systemd user environment: kiln.sh serve loads it when kiln (re)starts.
const MODELS = ["qwen3.8-27b", "qwen3.8-flash-next"];
async function pickModel(req: Request): Promise<string | undefined> {
  const model = (await req.json().catch(() => ({})))?.model;
  if (!MODELS.includes(model)) return undefined;
  Bun.spawnSync(["systemctl", "--user", "set-environment", `KILN_MODEL=${model}`]);
  return model;
}

// ---- replies as jobs -------------------------------------------------------------------------------------------
// A reply belongs to this server, not to the page that asked for it: the request to kiln runs to its end whatever
// the page does (reloads, closes, sleeps with the laptop, loses Wi-Fi, opens another chat), and any page can follow it
// again from where it left off: every event has a position. Only an explicit stop ends a reply early. A reply stays
// in RAM until 15 minutes after it ends, so a page that comes back in time collects the rest; nothing goes to disk.
interface Job {
  id: string;
  events: string[];                                 // kiln's SSE data payloads, then one with the timings, then [DONE]
  done: boolean;
  ctrl: AbortController;
  waiters: Set<() => void>;
  t0: number; tFirst?: number; tAnswer?: number; tLast?: number;   // ms: start, first token, first answer token, last
}
const jobs = new Map<string, Job>();
const KEEP_MS = 15 * 60_000;

function startJob(body: string): Job {
  const job: Job = { id: crypto.randomUUID(), events: [], done: false, ctrl: new AbortController(), waiters: new Set(), t0: Date.now() };
  jobs.set(job.id, job);
  const push = (data: string) => {
    job.events.push(data);
    for (const wake of [...job.waiters]) wake();
  };
  (async () => {
    try {
      const r = await fetch(`${KILN}/v1/chat/completions`, {
        method: "POST", body, headers: { "Content-Type": "application/json" }, signal: job.ctrl.signal,
      });
      if (!r.ok || !r.body) {
        push(JSON.stringify({ error: { message: (await r.json().catch(() => null))?.error?.message ?? `kiln answered ${r.status}` } }));
        return;
      }
      const reader = r.body.getReader(), decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        for (let end = buf.indexOf("\n\n"); end >= 0; end = buf.indexOf("\n\n")) {
          const data = buf.slice(0, end).split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n");
          buf = buf.slice(end + 2);
          if (!data || data === "[DONE]") continue;
          const now = Date.now(), delta = data.includes('"delta"') ? JSON.parse(data).choices?.[0]?.delta : undefined;
          if (delta?.content || delta?.reasoning_content) {
            job.tFirst ??= now;
            job.tLast = now;
            if (delta.content) job.tAnswer ??= now;
          }
          push(data);
        }
      }
    } catch (e) {
      push(JSON.stringify(job.ctrl.signal.aborted ? { stopped: true } : { error: { message: "kiln is not running" } }));
    } finally {
      push(JSON.stringify({ timing: { start: job.t0, first: job.tFirst, answer: job.tAnswer, last: job.tLast } }));
      push("[DONE]");
      job.done = true;
      setTimeout(() => jobs.delete(job.id), KEEP_MS);
    }
  })();
  return job;
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

const server = Bun.serve({
  hostname: process.env.KILN_HOST ?? "127.0.0.1",          // 0.0.0.0 after `kiln start --lan`
  port: Number(process.env.PORT ?? 8081),
  development: process.env.NODE_ENV !== "production",
  routes: {
    "/": app,

    // A streamed reply starts a job and follows it (its id in X-Kiln-Job); other requests (session titles) pass
    // through. Nothing arrives before the first token, which takes a while for a long prompt: no idle timeout.
    "/api/chat": {
      POST: guarded(async (req, srv) => {
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
      }),
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
