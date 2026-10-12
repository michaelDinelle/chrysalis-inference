// kiln web UI: serves the chat app and proxies the kiln server's OpenAI API (same origin, so no CORS). It stores
// nothing: conversations stay in the browser tab. `bun server.ts` for development, NODE_ENV=production for a
// minified bundle.
import app from "./index.html";

const KILN = process.env.KILN_URL ?? "http://127.0.0.1:8080";
// Changes with the page's code. Open tabs compare it on every status poll and reload to pick up an update
// (their chats are in sessionStorage, so a reload keeps them).
const BUILD = Bun.hash((await Promise.all(["index.html", "app.ts", "style.css"].map((f) => Bun.file(`${import.meta.dir}/${f}`).text()))).join("\0")).toString(36);

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

    // online: serving; loading: the systemd unit runs but the model is still loading (kiln answers 503 until it has
    // loaded, or does not listen yet); offline: stopped
    "/api/status": async () => {
      try {
        const r = await fetch(`${KILN}/kiln/stats`);
        if (!r.ok) throw new Error("loading");
        return Response.json({ state: "online", build: BUILD, ...(await r.json()) });
      } catch {
        return Response.json({ state: unit("is-active") === "active" ? "loading" : "offline", build: BUILD });
      }
    },
    "/api/kiln/start": { POST: () => (unit("start", "--no-block"), new Response(null, { status: 202 })) },
    // stopping takes ~40 ms: wait for it, so the page can show the new state right away
    "/api/kiln/stop": { POST: () => (unit("stop"), new Response(null, { status: 204 })) },
  },
});

console.log(`kiln web: ${server.url}`);
