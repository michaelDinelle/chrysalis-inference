// kiln web UI: serves the chat app and proxies the kiln server's OpenAI API (same origin, so no CORS). It stores
// nothing: conversations stay in the browser tab. `bun server.ts` for development, NODE_ENV=production for a
// minified bundle.
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

const server = Bun.serve({
  hostname: process.env.KILN_HOST ?? "127.0.0.1",          // 0.0.0.0 after `kiln start --lan`
  port: Number(process.env.PORT ?? 8081),
  development: process.env.NODE_ENV !== "production",
  routes: {
    "/": app,

    // Streams kiln's server-sent events through. Nothing arrives before the first token, which takes a while
    // for a long prompt, so this request has no idle timeout. Closing the tab aborts it and kiln stops.
    "/api/chat": {
      POST: guarded(async (req, srv) => {
        srv.timeout(req, 0);
        try {
          const r = await fetch(`${KILN}/v1/chat/completions`, {
            method: "POST", body: await req.text(), headers: { "Content-Type": "application/json" }, signal: req.signal,
          });
          return new Response(r.body, {
            status: r.status,
            headers: { "Content-Type": r.headers.get("Content-Type") ?? "application/json", "Cache-Control": "no-cache" },
          });
        } catch {
          return Response.json({ error: { message: "kiln is not running" } }, { status: 502 });
        }
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
