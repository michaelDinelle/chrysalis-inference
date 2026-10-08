// kiln web UI: serves the chat app, proxies the kiln server's OpenAI API (same origin, so no CORS) and keeps
// conversations in SQLite. `bun server.ts` for development, NODE_ENV=production for a minified bundle.
import { Database } from "bun:sqlite";
import app from "./index.html";

const KILN = process.env.KILN_URL ?? "http://127.0.0.1:8080";
const db = new Database(`${import.meta.dir}/../data/web.db`, { create: true });
db.run("PRAGMA journal_mode = WAL");
db.run("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, updated INTEGER NOT NULL, data TEXT NOT NULL)");

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

    "/api/conversations": {
      GET: () => Response.json(db.query("SELECT id, title, updated FROM conversations ORDER BY updated DESC").all()),
    },
    "/api/conversations/:id": {
      GET(req) {
        const row = db.query<{ data: string }, [string]>("SELECT data FROM conversations WHERE id = ?").get(req.params.id);
        return row ? new Response(row.data, { headers: { "Content-Type": "application/json" } }) : Response.json(null, { status: 404 });
      },
      async PUT(req) {
        const c = await req.json();
        db.query("INSERT OR REPLACE INTO conversations (id, title, updated, data) VALUES (?, ?, ?, ?)")
          .run(req.params.id, String(c.title), Number(c.updated), JSON.stringify(c));
        return new Response(null, { status: 204 });
      },
      DELETE(req) {
        db.query("DELETE FROM conversations WHERE id = ?").run(req.params.id);
        return new Response(null, { status: 204 });
      },
    },
  },
});

console.log(`kiln web: ${server.url}`);
