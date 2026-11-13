import { writeFileSync } from "node:fs";

const port = Number(process.argv[process.argv.indexOf("--port") + 1]), to = process.env.FAKE_AWAY_URL!;
const theirs = `Basic ${Buffer.from(`opencode:${process.env.FAKE_AWAY_SECRET}`).toString("base64")}`, ours = `Basic ${Buffer.from(`opencode:${process.env.OPENCODE_PASSWORD}`).toString("base64")}`;
writeFileSync(process.env.FAKE_AWAY_CONFIG!, process.env.OPENCODE_CONFIG_CONTENT ?? "");
Bun.serve({
  hostname: "127.0.0.1", port, idleTimeout: 0,
  fetch(req) {
    if (req.headers.get("authorization") !== ours) return new Response("", { status: 401 });
    const at = new URL(req.url), headers = new Headers(req.headers);
    headers.set("authorization", theirs);
    headers.delete("host");
    return fetch(to + at.pathname + at.search, { method: req.method, headers, body: req.method === "GET" || req.method === "HEAD" ? undefined : req.body });
  },
});
