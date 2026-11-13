const fs = require("fs"), os = require("os"), path = require("path");
const [origin, keyFile] = process.argv.slice(2);
const named = os.platform() === "darwin" ? Bun.spawnSync(["scutil", "--get", "ComputerName"]).stdout.toString().trim() : "", name = named || os.hostname().replace(/\.local$/, "");
const key = fs.readFileSync(keyFile, "utf8").trim(), bin = process.env.CHRYSALIS_OPENCODE ?? process.execPath;
const password = crypto.randomUUID() + crypto.randomUUID(), auth = "Basic " + Buffer.from("opencode:" + password).toString("base64");
let child, base;

function freePort() {
  const s = Bun.listen({ hostname: "127.0.0.1", port: 0, socket: { data() {} } }), port = s.port;
  s.stop(true);
  return port;
}
async function start(config) {
  if (child && child.exitCode === null) return;
  const port = freePort(), { BUN_BE_BUN: _, OPENCODE_SERVER_PASSWORD: __, ...env } = process.env;
  config.providers.chrysalis.settings = { baseURL: origin + "/v1", apiKey: key };
  child = Bun.spawn([bin, "serve", "--hostname", "127.0.0.1", "--port", String(port)], { stdin: "ignore", stdout: "ignore", stderr: "ignore",
    env: { ...env, OPENCODE_PASSWORD: password, OPENCODE_CONFIG_CONTENT: JSON.stringify(config) } });
  base = "http://127.0.0.1:" + port;
  for (const deadline = Date.now() + 30000; ; await Bun.sleep(150)) {
    if (await fetch(base + "/api/info", { headers: { authorization: auth } }).then((r) => r.ok, () => false)) return;
    if (Date.now() > deadline || child.exitCode !== null) throw new Error("OpenCode did not start");
  }
}
function folders(at) {
  at = path.resolve(at || os.homedir());
  const dirs = fs.readdirSync(at, { withFileTypes: true }).filter((e) => e.isDirectory() && !e.name.startsWith(".")).map((e) => e.name).sort((a, b) => a.localeCompare(b));
  return { path: at, up: path.dirname(at) === at ? undefined : path.dirname(at), home: os.homedir(), dirs: dirs.map((d) => ({ name: d, path: path.join(at, d) })) };
}
async function answer(ws, m, live) {
  const say = (o) => { if (ws.readyState === 1) ws.send(JSON.stringify({ id: m.id, ...o })); }, ctrl = new AbortController();
  live.set(m.id, ctrl);
  try {
    const at = new URL(m.path, base);
    if (at.pathname === "/chrysalis/folders") {
      say({ status: 200 });
      say({ chunk: JSON.stringify(folders(at.searchParams.get("path") ?? "")) });
      return say({ end: true });
    }
    const r = await fetch(base + m.path, { method: m.method, headers: { ...m.headers, authorization: auth }, body: m.body, signal: ctrl.signal });
    say({ status: r.status });
    if (r.body) {
      const text = new TextDecoder();
      for await (const part of r.body) say({ chunk: text.decode(part, { stream: true }) });
    }
    say({ end: true });
  } catch (e) {
    say({ error: e && e.message ? e.message : String(e) });
  } finally {
    live.delete(m.id);
  }
}
process.on("exit", () => child && child.kill());
for (const sig of ["SIGTERM", "SIGINT"]) process.on(sig, () => process.exit(0));
for (;;) {
  const known = await fetch(origin + "/opencode/machine", { headers: { authorization: "Bearer " + key } }).then((r) => r.status, () => 0);
  if (known === 401) process.exit(0);
  if (known === 200) await new Promise((done) => {
    const ws = new WebSocket(origin.replace(/^http/, "ws") + "/opencode/machine", { headers: { authorization: "Bearer " + key } }), live = new Map();
    ws.onopen = () => ws.send(JSON.stringify({ hello: { name, home: os.homedir(), kind: os.platform() } }));
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.config) start(m.config).then(() => ws.send(JSON.stringify({ ready: true })), (err) => { ws.send(JSON.stringify({ failed: err.message })); ws.close(); });
      else if (m.cancel) live.get(m.id)?.abort();
      else if (m.method) void answer(ws, m, live);
    };
    ws.onclose = () => { for (const c of live.values()) c.abort(); done(); };
    ws.onerror = () => {};
  });
  await Bun.sleep(2000);
}
