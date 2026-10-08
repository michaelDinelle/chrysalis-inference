// kiln web UI: serves the chat app and proxies the kiln server's OpenAI API (same origin, so no CORS). It stores
// nothing: conversations stay in the browser tab. `bun server.ts` for development, NODE_ENV=production for a
// minified bundle.
import app from "./index.html";

const KILN = process.env.KILN_URL ?? "http://127.0.0.1:8080";

const unit = (...args: string[]) => Bun.spawnSync(["systemctl", "--user", ...args, "kiln"]).stdout.toString().trim();

const server = Bun.serve({
  hostname: process.env.HOST ?? "127.0.0.1",
  port: Number(process.env.PORT ?? 8081),
  development: process.env.NODE_ENV !== "production",
  routes: {
    "/": app,

    // Streams kiln's server-sent events through. Nothing arrives before the first token, which takes a while
    // for a long prompt, so this request has no idle timeout. Closing the tab aborts it and kiln stops.
    "/api/chat": {
      async POST(req, srv) {
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
      },
    },

    // online: serving; loading: the systemd unit runs but the model is still loading; offline: stopped
    "/api/status": async () => {
      try {
        return Response.json({ state: "online", ...(await (await fetch(`${KILN}/kiln/stats`)).json()) });
      } catch {
        return Response.json({ state: unit("is-active") === "active" ? "loading" : "offline" });
      }
    },
    "/api/kiln/start": { POST: () => (unit("start", "--no-block"), new Response(null, { status: 202 })) },
    "/api/kiln/stop": { POST: () => (unit("stop", "--no-block"), new Response(null, { status: 202 })) },
  },
});

console.log(`kiln web: ${server.url}`);
