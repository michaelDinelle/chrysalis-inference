import { expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname } from "node:path";
import { memoryRoot } from "./platform";
import { butterfly, plain, progress } from "./terminal";
import { lanAddresses } from "./tls";
import puppeteer, { type Page } from "puppeteer-core";

const json = { "Content-Type": "application/json" };
const UP = "\x1b[A", DOWN = "\x1b[B", RIGHT = "\x1b[C";
const freePort = () => { const s = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response() }), port = s.port!; s.stop(true); return port; };
const refused = (url: string, init: object = {}) => fetch(url, init).then((r) => `answered ${r.status}`, () => "refused");

function chrysalis(o: { pty?: boolean; mode?: string; args?: string[]; env?: Record<string, string> } = {}) {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-cli-`), port = freePort(), engine = freePort();
  const cmd = [process.execPath, "cli.ts", ...(o.mode ? ["--mode", o.mode] : []), ...(o.args ?? [])];
  const proc = Bun.spawn(o.pty ? ["python3", "-c", "import pty, sys; pty.spawn(sys.argv[1:])", ...cmd] : cmd, {
    cwd: import.meta.dir, stdin: "pipe", stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: String(port), CHRYSALIS_URL: `http://127.0.0.1:${engine}`, FAKE_ENGINE_PORT: String(engine), HOME: dir, BUN_RUNTIME_TRANSPILER_CACHE_PATH: "0", CHRYSALIS_TLS_DIR: `${dir}/tls`,
      CHRYSALIS_ENGINE_CMD: JSON.stringify([process.execPath, `${import.meta.dir}/fake-engine.ts`]), CHRYSALIS_CHECK_PORT: "0", CHRYSALIS_LINK_PORT: "0",
      CHRYSALIS_API_PORT: "0", CHRYSALIS_SELF_AS_NETWORK: "1", XDG_DATA_HOME: `${dir}/.local/share`, XDG_STATE_HOME: `${dir}/.local/state`, CHRYSALIS_SNAPSHOTS: `${dir}/cache`, ...o.env },
  });
  const reader = proc.stdout.getReader(), dec = new TextDecoder();
  let raw = "";
  void (async () => { for (let r = await reader.read(); !r.done; r = await reader.read()) raw += dec.decode(r.value, { stream: true }); })();
  const out = (from = 0) => plain(raw.slice(from));
  const shown = async (text: string | RegExp, from = 0) => {
    for (const end = Date.now() + 15_000; ; await Bun.sleep(50)) {
      const got = typeof text === "string" ? (out(from).includes(text) ? [text] : null) : text.exec(out(from));
      if (got) return got;
      if (Date.now() > end) throw new Error(`the terminal never showed ${text}: ${out(from)}`);
    }
  };
  return {
    proc, dir, port, here: `http://127.0.0.1:${port}`, engine: `http://127.0.0.1:${engine}`, out, shown, mark: () => raw.length,
    type: (keys: string) => { proc.stdin.write(keys); proc.stdin.flush(); },
    press: async (...keys: string[]) => { for (const k of keys) { proc.stdin.write(k); proc.stdin.flush(); await Bun.sleep(40); } },
    reply: (to: string, words: string) => fetch(`${to}/v1/chat/completions`, { method: "POST", headers: json, body: JSON.stringify({ messages: [{ role: "user", content: words }] }) }),
    stop: async () => { proc.kill(); await proc.exited; rmSync(dir, { recursive: true, force: true }); },
  };
}
function decoded(rows: boolean[][]) {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-qr-`), big = rows.flatMap((r) => Array(4).fill(r.flatMap((on) => Array(4).fill(on ? "1" : "0")).join(" ")));
  writeFileSync(`${dir}/code.pbm`, `P1\n${big[0].split(" ").length} ${big.length}\n${big.join("\n")}\n`);
  try { return Bun.spawnSync(["zbarimg", "-q", "--raw", `${dir}/code.pbm`], { stdout: "pipe", stderr: "pipe" }).stdout.toString().trim(); }
  finally { rmSync(dir, { recursive: true, force: true }); }
}
const drawn = (lines: string[]) => lines.flatMap((l) => [[...l.slice(2)].map((c) => c === " " || c === "▄"), [...l.slice(2)].map((c) => c === " " || c === "▀")]);
const painted = (svg: string) => {
  const n = Number(/viewBox="0 0 (\d+)/.exec(svg)![1]), rows = Array.from({ length: n }, () => Array(n).fill(false));
  for (const [, x, y] of svg.matchAll(/M(\d+) (\d+)h1v1h-1z/g)) rows[Number(y)][Number(x)] = true;
  return rows;
};
const files = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap((f) => (f.isDirectory() ? files(`${dir}/${f.name}`) : f.isFile() ? [`${dir}/${f.name}`] : []));
function family(root: number) {
  const kids = new Map<number, number[]>();
  for (const l of Bun.spawnSync(["ps", "-A", "-o", "pid=,ppid="], { stdout: "pipe" }).stdout.toString().trim().split("\n")) {
    const [pid, ppid] = l.trim().split(/\s+/).map(Number);
    kids.set(ppid, [...(kids.get(ppid) ?? []), pid]);
  }
  const tree = [root];
  for (let i = 0; i < tree.length; i++) tree.push(...(kids.get(tree[i]) ?? []));
  return tree;
}
function listening(root: number) {
  const tree = family(root);
  const out = Bun.spawnSync(["lsof", "-nP", "-a", "-p", tree.join(","), "-i"], { stdout: "pipe", stderr: "pipe" }).stdout.toString();
  return out.split("\n").flatMap((l) => {
    const m = /\s(TCP|UDP) (\S+)(?: \((\w+)\))?\s*$/.exec(l);
    return !m || (m[1] === "TCP" ? m[3] !== "LISTEN" : m[2].includes("->")) ? [] : [`${m[1]} ${m[2]}`];
  });
}

test("the terminal shows each request to the model as it finishes, by its sizes and speeds and never its words, and nothing else the servers print until verbose is typed; none of it reaches a file; a request from an incognito chat stays off the terminal and tells the model server to keep it out of its prompt cache on disk, which the next ordinary request writes again", async () => {
  const c = chrysalis();
  try {
    await c.shown("qwen3.8-flash-next ready in");
    await c.shown("257,920 tokens of context · 24.6 GB VRAM · 42.6 GB RAM");
    await c.shown(/chat\s+http:\/\/localhost:\d+/);
    expect((await c.reply(c.engine, "the words of the prompt")).status).toBe(200);
    await c.shown("14,982 tokens read (14,272 cached) · first token 1.08 s · 1,088 written at 42.6 tok/s\n");
    await c.reply(c.engine, "long");
    await c.shown("1,088 written at 42.6 tok/s · cut at its token limit");
    await c.reply(c.engine, "leave");
    await c.shown("14,982 tokens read (14,272 cached) · stopped by the client after 12 written");
    await c.reply(c.engine, "fail");
    await c.shown("a request failed: ValueError('boom')");
    expect((await fetch(`${c.here}/`)).status).toBe(200);
    expect(c.out()).not.toMatch(/CPU experts|warmed up|slot 1|index\.html|words of the prompt|heard/);

    c.type("verbose\n");
    await c.shown("verbose is on");
    let at = c.mark();
    c.type("restart model\n");
    await c.shown("CPU experts on 16 threads", at);
    await c.shown("warmed up in 0.4 s", at);
    await c.shown("ready in", at);
    await c.reply(c.engine, "cold");
    await c.shown("14,982 tokens read · first token 1.08 s", at);
    await c.shown("| stop | slot 1", at);

    c.type("verbose\n");
    await c.shown("verbose is off");
    at = c.mark();
    await c.reply(c.engine, "again");
    await c.shown("1,088 written at 42.6 tok/s", at);
    expect(c.out(at)).not.toContain("slot 1");
    expect(files(c.dir).filter((f) => /14982|14,982|tokens read|slot 1/.test(readFileSync(f, "latin1")))).toEqual([]);

    const asked = async (words: string, more: object = {}) => (await (await fetch(`${c.here}/api/chat`, { method: "POST", headers: json, body: JSON.stringify({ messages: [{ role: "user", content: words }], ...more }) })).json()).choices[0].message.content;
    at = c.mark();
    expect(await asked("probe hidden", { chrysalis_incognito: true })).toBe("unkept");
    await fetch(`${c.engine}/chrysalis/load`, { method: "POST", body: JSON.stringify({ model: "qwen3.8-flash-next" }) });
    await c.shown("qwen3.8-flash-next ready in", at);
    expect(c.out(at)).not.toContain("tokens read");
    at = c.mark();
    expect(await asked("probe seen")).toBe("written");
    await c.shown("14,982 tokens read", at);
    expect(readdirSync(`${c.dir}/cache/qwen3.8-flash-next`)).toEqual(["seen.snap"]);
  } finally {
    await c.stop();
  }
}, 60_000);

test("where no GPU was found, the terminal says why in the launcher's own words, starts no model server and still serves the chat, which is refused a start at once instead of waiting for one", async () => {
  const why = "this computer has no GPU that AMD's ROCm can use", c = chrysalis({ args: ["--no-model", why] });
  try {
    await c.shown(`No GPU was found: ${why}`);
    await c.shown(/chat\s+http:\/\/localhost:\d+/);
    expect(c.out()).not.toContain("ready in");
    expect(await refused(`${c.engine}/health`)).toBe("refused");
    const start = await fetch(`${c.here}/api/chrysalis/start`, { method: "POST", headers: json, body: "{}" });
    expect([start.status, (await start.json()).error.message]).toEqual([409, "No model runs on this computer and none is linked: Chrysalis's terminal says why no GPU was found"]);
  } finally {
    await c.stop();
  }
}, 60_000);

test("a download's progress is read from the line the downloader draws: the newest of what it has said, before it starts, as it goes and when it is done", () => {
  const bar = (tail: string) => `\r\x1b[K  qwen3.8-27b          \x1b[35m━━━━━━━━━\x1b[2m━━━━━━━━━━━━━━━━━━━\x1b[0m  ${tail}`;
  expect(progress("Warning: You are sending unauthenticated requests to the HF Hub.")).toBeUndefined();
  expect(progress(bar(" 0.0 / 16.8 GB  starting"))).toEqual({ done: 0, total: 16.8e9, rate: 0 });
  expect(progress(bar(" 0.0 / 16.8 GB  starting") + bar(" 5.7 / 16.8 GB  110 MB/s, 1:40 left"))).toEqual({ done: 5.7e9, total: 16.8e9, rate: 110e6 });
  expect(progress(bar(" 5.7 / 16.8 GB  110 MB/s, 1:40 left") + bar("16.8 / 16.8 GB  done\n"))).toEqual({ done: 16.8e9, total: 16.8e9, rate: 0 });
});

test("the butterfly is the logo in characters: as wide as asked, its two sides alike, and the same drawing in every frame: only a glint crosses it", () => {
  const full = butterfly(40, 14, 0), ink = (lines: string[]) => lines.join("").replace(/ /g, "").length;
  expect([full.length, Math.max(...full.map((l) => l.length)) <= 40, ink(full) > 120]).toEqual([14, true, true]);
  const sides = full.map((l) => l.padEnd(40)).reduce((n, l) => [n[0] + l.slice(0, 20).replace(/ /g, "").length, n[1] + l.slice(20).replace(/ /g, "").length], [0, 0]);
  expect(Math.abs(sides[0] - sides[1]) / (sides[0] + sides[1])).toBeLessThan(0.1);
  expect(butterfly(40, 14, 3).join("\n")).not.toBe(full.join("\n"));
  expect(Array.from({ length: 80 }, (_, i) => ink(butterfly(40, 14, i * 0.11)) === ink(full)).every(Boolean)).toBe(true);
  expect(Array.from({ length: 80 }, (_, i) => butterfly(40, 14, i * 0.11).join("\n").replace(/%/g, "@")).every((frame) => frame === full.join("\n").replace(/%/g, "@"))).toBe(true);
});

test("with no model here the terminal starts none and downloads none: it says where to pick one, a chat is refused with the same words, and only a model picked in the page's menu reaches the terminal, which then begins its download", async () => {
  const empty = mkdtempSync(`${tmpdir()}/chrysalis-none-`), c = chrysalis({ env: { CHRYSALIS_ENGINE_CMD: "", CHRYSALIS_HOME: empty } });
  try {
    await c.shown("No model is downloaded yet");
    await c.shown(/chat\s+http:\/\/localhost:\d+/);
    const status = async () => (await fetch(`${c.here}/api/status`)).json();
    for (const end = Date.now() + 10_000; !(await status()).got && Date.now() < end; ) await Bun.sleep(100);
    expect([(await status()).got, (await status()).state]).toEqual([[], "offline"]);
    const chat = await fetch(`${c.here}/api/chrysalis/start`, { method: "POST", headers: json, body: JSON.stringify({ model: "qwen3.8-27b" }) });
    expect([chat.status, (await chat.json()).error.message.startsWith("No model is downloaded yet")]).toEqual([409, true]);
    expect(c.out()).not.toContain("is downloading");
    const pick = await fetch(`${c.here}/api/chrysalis/start`, { method: "POST", headers: json, body: JSON.stringify({ model: "qwen3.8-27b", get: true }) });
    expect(pick.status).toBe(202);
    await c.shown("qwen3.8-27b is downloading");
  } finally {
    await c.stop();
    rmSync(empty, { recursive: true, force: true });
  }
}, 60_000);

test("the switches are on screen from the start with the arrows live and Model highlighted: with no model here, Enter downloads the 27B, and right then Enter the other one, with no key pressed to open anything; Uninstall is a row there too, and asks before it removes anything", async () => {
  for (const [keys, model] of [[["\r"], "qwen3.8-27b"], [[RIGHT, "\r"], "qwen3.8-flash-next"]] as const) {
    const empty = mkdtempSync(`${tmpdir()}/chrysalis-pick-`), c = chrysalis({ pty: true, env: { CHRYSALIS_ENGINE_CMD: "", CHRYSALIS_HOME: empty, LINES: "50", COLUMNS: "120" } });
    try {
      await c.shown(/Model\s+○ Qwen3.8-27B · download\s+○ Qwen3.8-Flash-Next · download/);
      await c.shown("Enter sets it");
      await c.shown("no model yet");
      expect(c.out()).not.toContain("is downloading");
      await c.shown(/Uninstall\s+remove Chrysalis, its models and its settings/);
      await c.press(UP, UP, "\r");
      await c.shown("Remove Chrysalis from this computer, with its models, its caches and its settings?");
      c.type("n\r");
      await c.shown("nothing was removed");
      expect(c.out()).not.toContain("is downloading");
      await c.press(DOWN, DOWN, ...keys);
      await c.shown(`${model} is downloading`);
    } finally {
      await c.stop();
      rmSync(empty, { recursive: true, force: true });
    }
  }
}, 90_000);

test("a download can be paused, resumed and cancelled from the page: pausing stops it and keeps its place, resuming carries on, and cancelling removes what had arrived and leaves no download behind", async () => {
  const home = mkdtempSync(`${tmpdir()}/chrysalis-pause-`), part = `${home}/data/models/.cache/huggingface/download/qwen3.8-27b/weights.incomplete`;
  const c = chrysalis({ env: { CHRYSALIS_ENGINE_CMD: "", CHRYSALIS_HOME: home, CHRYSALIS_GET_CMD: JSON.stringify([process.execPath, `${import.meta.dir}/fake-get.ts`]) } });
  const say = (path: string, body: object) => fetch(`${c.here}${path}`, { method: "POST", headers: json, body: JSON.stringify(body) }).then((r) => r.status);
  const download = async () => (await (await fetch(`${c.here}/api/status`)).json()).download;
  const until = async (ok: (d: any) => boolean) => { for (const end = Date.now() + 10_000; Date.now() < end; await Bun.sleep(50)) { const d = await download(); if (ok(d)) return d; } throw new Error(`the download never got there: ${JSON.stringify(await download())}`); };
  try {
    await c.shown("No model is downloaded yet");
    await c.shown(/chat\s+http:\/\/localhost:\d+/);
    expect(await say("/api/chrysalis/download", { how: "pause" })).toBe(409);
    for (const end = Date.now() + 10_000; !(await (await fetch(`${c.here}/api/status`)).json()).got && Date.now() < end; ) await Bun.sleep(100);
    expect(await say("/api/chrysalis/start", { model: "qwen3.8-27b", get: true })).toBe(202);
    const going = await until((d) => d?.done > 1e9 && !d.paused);
    expect([going.model, going.total, going.rate]).toEqual(["qwen3.8-27b", 16.8e9, 100e6]);
    expect(await say("/api/chrysalis/download", { how: "pause" })).toBe(202);
    const paused = await until((d) => d?.paused === true);
    await c.shown("qwen3.8-27b's download is paused");
    await Bun.sleep(500);
    expect([(await download()).done, paused.done >= going.done, existsSync(part)]).toEqual([paused.done, true, true]);
    expect(await say("/api/chrysalis/download", { how: "resume" })).toBe(202);
    await until((d) => d && !d.paused && d.done > 0);
    expect(await say("/api/chrysalis/download", { how: "cancel" })).toBe(202);
    await until((d) => d === undefined);
    await c.shown("download is cancelled and what had arrived is removed");
    expect(existsSync(part)).toBe(false);
  } finally {
    await c.stop();
    rmSync(home, { recursive: true, force: true });
  }
}, 60_000);

test("in local mode every port the terminal, the web chat and what they start listen on is on 127.0.0.1 alone, while the model is unloaded and the page asks for the status and for computers to link to as well", async () => {
  const c = chrysalis({ env: { CHRYSALIS_FIND_MS: "60000" } });
  try {
    await c.shown("ready in");
    await c.shown(/chat\s+http:\/\/localhost:\d+/);
    c.type("model unload\n");
    await c.shown("is unloaded and the GPU is free");
    expect((await (await fetch(`${c.here}/api/status`)).json()).state).toBe("offline");
    expect((await fetch(`${c.here}/api/link`)).status).toBe(200);
    const open = listening(c.proc.pid);
    expect(open).toContain(`TCP 127.0.0.1:${c.port}`);
    expect(open.filter((s) => !/ (127\.[\d.]+|\[::1\]):\d+$/.test(s))).toEqual([]);
  } finally {
    await c.stop();
  }
}, 60_000);

test("insecure puts the chat and the API on the network over plain HTTP with no certificate made and nobody to let in, warns that it is unencrypted and without voice, says who asks, and keeps OpenCode and voice on this computer; local takes it off the network again", async () => {
  const ip = lanAddresses()[0];
  expect(ip).toBeTruthy();
  const c = chrysalis();
  try {
    await c.shown(/chat\s+http:\/\/localhost:\d+/);
    expect(await refused(`http://${ip}:${c.port}/`)).toBe("refused");
    const at = c.mark();
    c.type("insecure\n");
    const chat = (await c.shown(/chat\s+(http:\/\/[\d.]+:\d+)/, at))[1], api = (await c.shown(/API\s+(http:\/\/[\d.]+:\d+)\/v1/, at))[1];
    expect(new URL(chat).hostname).not.toBe("127.0.0.1");
    expect(new URL(api).hostname).toBe(new URL(chat).hostname);
    await c.shown("Unencrypted: anyone on this network can use the model and read what is sent.", at);
    expect(c.out(at).replace(/\s+/g, " ")).toContain("Other devices get no voice mode");

    const page = await fetch(`${chat}/`), html = await page.text();
    expect([page.status, html.includes("Connect to Chrysalis"), /src="\/[^"]+\.js"/.test(html)]).toEqual([200, false, true]);
    expect(await refused(`${chat.replace("http:", "https:")}/`, { tls: { rejectUnauthorized: false } })).toBe("refused");
    expect(files(c.dir).filter((f) => /\.(pem|key)$/.test(f))).toEqual([]);

    const said = await c.reply(api, "over the network");
    expect([said.status, (await said.json()).choices[0].message.content]).toEqual([200, "heard over the network"]);
    await c.shown(`from ${new URL(chat).hostname}`, at);
    await c.shown("14,982 tokens read (14,272 cached)", at);
    expect((await (await fetch(`${api}/v1/models`)).json()).data.map((m: { id: string }) => m.id)).toContain("qwen3.8-27b");
    expect((await fetch(`${api}/metrics`)).status).toBe(404);
    expect((await fetch(`${chat}/api/chat`, { method: "POST", headers: json, body: JSON.stringify({ messages: [{ role: "user", content: "in the chat" }] }) })).status).toBe(200);

    expect([(await fetch(`${chat}/api/threads`)).status, (await fetch(`${c.here}/api/threads`)).status]).toEqual([403, 200]);
    expect((await fetch(`${chat}/api/threads`, { method: "POST" })).status).toBe(403);
    expect((await fetch(`${chat}/api/chat`, { method: "POST", headers: json, body: JSON.stringify({ stream: true, chrysalis_thread: { session: "s", dir: "/", text: "run" } }) })).status).toBe(403);
    expect((await fetch(`${chat}/api/voice`, { headers: { Origin: chat } })).status).toBe(403);
    expect((await fetch(`${c.here}/api/voice`, { headers: { Origin: c.here } })).status).toBe(426);

    const then = c.mark();
    c.type("local\n");
    await c.shown(/chat\s+http:\/\/localhost:\d+/, then);
    expect([await refused(`${chat}/`), await refused(`${api}/v1/chat/completions`, { method: "POST" })]).toEqual(["refused", "refused"]);
    expect((await fetch(`${c.here}/`)).status).toBe(200);
  } finally {
    await c.stop();
  }
}, 60_000);

test("in a terminal on the network, the code a phone scans shows first, above the addresses, and a reader takes that address from it, as from the page's; every device asking shows above the prompt; one answered on the page leaves, so y lets in the one still asking; Tab opens the switches, where arrows and Enter turn verbose on and unload the model; Tab completes a command, and Ctrl+C stops the model server and the web chat", async () => {
  const c = chrysalis({ pty: true, mode: "network" });
  try {
    const net = (await c.shown(/phone\s+(https:\/\/[\d.]+:\d+)/))[1];
    const top = c.out().split(/\r?\n/), from = top.findIndex((l) => /^  [█▀▄ ]{20,}$/.test(l)), to = top.findIndex((l, i) => i > from && !/^  [█▀▄ ]{20,}$/.test(l));
    expect([from >= 0, to - from > 10, decoded(drawn(top.slice(from, to)))]).toEqual([true, true, net]);
    expect(from).toBeLessThan(top.findIndex((l) => /^\s+chat\s+http/.test(l)));
    const pair = await (await fetch(`${c.here}/api/pair`)).json();
    expect([pair.connect[0].replace(/\/$/, ""), decoded(painted(pair.qr))]).toEqual([net, `${net}/`]);
    const ask = async (name: string) => (await (await fetch(`${net}/opencode/ask`, { method: "POST", headers: json, body: JSON.stringify({ name }), tls: { rejectUnauthorized: false } })).text()).split(" ");
    const collect = (secret: string) => fetch(`${net}/opencode/ask/${secret}`, { tls: { rejectUnauthorized: false } }).then((r) => r.status);
    const [firstSecret, first] = await ask("first"), [secondSecret, second] = await ask("second");
    await c.shown(`first asks to use this computer's Chrysalis · ${first.slice(0, 3)} ${first.slice(3)}`);
    await c.shown(`second asks to use this computer's Chrysalis · ${second.slice(0, 3)} ${second.slice(3)}`);
    expect(await collect(secondSecret)).toBe(202);
    expect((await fetch(`${c.here}/api/pair/deny`, { method: "POST", headers: json, body: JSON.stringify({ code: first }) })).status).toBe(204);
    await c.shown(`${first.slice(0, 3)} ${first.slice(3)} · answered elsewhere`);
    c.type("y\r");
    await c.shown("let in");
    expect([await collect(secondSecret), await collect(firstSecret)]).toEqual([200, 404]);

    await c.shown("Enter sets it");
    await c.shown(/Shows\s+● requests only\s+○ everything/);
    await c.press(DOWN, RIGHT, "\r");
    await c.shown("verbose is on");
    await c.shown(/Shows\s+○ requests only\s+● everything/);
    expect((await (await fetch(`${c.engine}/chrysalis/stats`)).json()).asleep).toBeUndefined();
    await c.shown(/Model\s+○ Qwen3.8-27B\s+● Qwen3.8-Flash-Next\s+○ unload\s/);
    await c.press(UP, RIGHT, "\r");
    await c.shown("qwen3.8-flash-next is unloaded and the GPU is free");
    await c.shown(/Model\s+○ Qwen3.8-27B\s+○ Qwen3.8-Flash-Next\s+● unloaded/);
    expect((await (await fetch(`${c.engine}/chrysalis/stats`)).json()).asleep).toBe(true);

    const at = c.mark();
    c.type("lin\t\r");
    await c.shown(/chat\s+http:\/\/localhost/, at);
    c.type("\x03");
    await c.proc.exited;
    expect([await refused(`${c.here}/`), await refused(`${c.engine}/chrysalis/stats`)]).toEqual(["refused", "refused"]);
  } finally {
    await c.stop();
  }
}, 60_000);

const holding = (dir: string, text: string) => (existsSync(dir) ? files(dir) : []).filter((f) => statSync(f).size < 1 << 20 && readFileSync(f, "latin1").includes(text));
test("an incognito thread runs on a second OpenCode that keeps its database in memory, with its tools, beside the first, which is never stopped: it is in no list and in no file, the model server is told to keep its requests out of the prompt cache on disk, and the ordinary store and cache are untouched; deleting it, incognito end in the terminal, stopping Chrysalis or closing its terminal window ends that OpenCode and deletes its folder", async () => {
  const bin = `${tmpdir()}/chrysalis-cli-opencode-${process.pid}`, pids = `${tmpdir()}/chrysalis-cli-opencode-${process.pid}.pids`;
  writeFileSync(bin, `#!/bin/sh\nexec ${process.execPath} ${import.meta.dir}/fake-opencode-bin.ts "$@"\n`);
  chmodSync(bin, 0o755);
  const c = chrysalis({ env: { CHRYSALIS_OPENCODE: bin, FAKE_OPENCODE_PIDS: pids } }), work = `${c.dir}/work`, root = memoryRoot();
  mkdirSync(work);
  const started = () => (existsSync(pids) ? readFileSync(pids, "utf8").trim().split("\n") : []).map((l) => { const [pid, db, data] = l.split(" "); return { pid: Number(pid), db, data }; });
  const up = (pid: number) => { try { process.kill(pid, 0); return true; } catch { return false; } };
  const stores = () => started().filter((o) => o.db === ":memory:" && up(o.pid) && existsSync(dirname(o.data))).map((o) => dirname(o.data));
  const held = () => readdirSync(root).filter((d) => d.startsWith(`chrysalis-run-${c.proc.pid}-`)).map((d) => `${root}/${d}`);
  const anywhere = (text: string) => [c.dir, ...held(), ...started().map((o) => dirname(o.data))].flatMap((d) => holding(d, text));
  const until = async (ok: () => unknown) => { for (const end = Date.now() + 15_000; Date.now() < end; await Bun.sleep(50)) if (await Promise.resolve().then(ok).catch(() => false)) return true; return false; };
  const connect = () => until(async () => (await fetch(`${c.here}/api/threads`, { method: "POST" })).status === 200);
  const listed = async () => ((await (await fetch(`${c.here}/api/threads`)).json()).threads ?? []).map((t: any) => t.title).sort();
  const say = async (text: string, hidden = false) => {
    const r = await fetch(`${c.here}/api/chat`, { method: "POST", headers: json, body: JSON.stringify({ stream: true, chrysalis_thread: { dir: work, text, title: text, model: "chrysalis/qwen3.8-27b", ...(hidden ? { hidden: true } : {}) } }) }), sent = await r.text();
    return [r.status, sent.includes('"error"'), sent.trimEnd().endsWith("[DONE]"), /"thread":"([^"]+)"/.exec(sent)?.[1] ?? "", sent] as const;
  };
  const engine = async () => (await (await fetch(`${c.engine}/chrysalis/env`)).json()).pid;
  try {
    await c.shown("ready in");
    const model = await engine();
    expect(await connect()).toBe(true);
    expect((await say("an ordinary errand")).slice(0, 3)).toEqual([200, false, true]);
    expect([await listed(), stores(), started().map((o) => o.db)]).toEqual([["an ordinary errand"], [], ["file"]]);

    let at = c.mark();
    const secret = await say("the secret plan", true);
    expect(secret.slice(0, 3)).toEqual([200, false, true]);
    await c.shown("an incognito chat is open", at);
    expect([await listed(), stores().length, await engine(), started().map((o) => o.db)]).toEqual([["an ordinary errand"], 1, model, ["file", ":memory:"]]);
    const read = await (await fetch(`${c.here}/api/threads/${secret[3]}`)).json();
    expect([read.thread.hidden, read.messages[0].content]).toEqual([true, "the secret plan"]);
    expect([anywhere("the secret plan"), holding(stores()[0], "an ordinary errand"), holding(c.dir, "an ordinary errand").length > 0]).toEqual([[], [], true]);
    expect((await say("an ordinary errand, again")).slice(0, 3)).toEqual([200, false, true]);
    expect(holding(stores()[0], "an ordinary errand, again")).toEqual([]);
    const [hidden, ordinary] = [await say("probe incognito", true), await say("probe ordinary")];
    const words = (sent: string) => [...sent.matchAll(/^data: (\{.*\})$/gm)].map((m) => JSON.parse(m[1]).choices?.[0]?.delta?.content ?? "").join("");
    expect([words(hidden[4]).includes("The model said: unkept."), words(ordinary[4]).includes("The model said: written.")]).toEqual([true, true]);
    expect(readdirSync(`${c.dir}/cache/qwen3.8-flash-next`)).toEqual(["ordinary.snap"]);
    expect(anywhere("probe incognito")).toEqual([]);

    at = c.mark();
    const store = stores()[0];
    expect((await fetch(`${c.here}/api/threads/${secret[3]}`, { method: "DELETE" })).status).toBe(204);
    expect((await fetch(`${c.here}/api/threads/${hidden[3]}`, { method: "DELETE" })).status).toBe(204);
    await c.shown("the incognito chats are gone", at);
    expect([await listed(), stores(), existsSync(store)]).toEqual([["an ordinary errand", "an ordinary errand, again", "probe ordinary"], [], false]);

    at = c.mark();
    expect((await say("a second secret", true)).slice(0, 3)).toEqual([200, false, true]);
    await c.shown("an incognito chat is open", at);
    const second = stores()[0];
    c.type("incognito end\n");
    await c.shown("the incognito chats are gone", at);
    expect([(await listed()).length, stores(), existsSync(second)]).toEqual([3, [], false]);

    expect((await say("a third secret", true)).slice(0, 3)).toEqual([200, false, true]);
    const third = stores()[0];
    expect(third).toBeTruthy();
    for (const pid of family(c.proc.pid)) try { process.kill(pid, "SIGHUP"); } catch {  }
    await c.proc.exited;
    expect(await until(() => !existsSync(third) && !started().some((o) => up(o.pid)))).toBe(true);
    expect([held(), anywhere("a third secret")]).toEqual([[], []]);
  } finally {
    await c.stop();
    rmSync(bin, { force: true });
    rmSync(pids, { force: true });
  }
}, 90_000);

test("OpenCode connected before network and then local, or before restart web, runs again in the web chat that starts after, and a new thread reaches the model there", async () => {
  const bin = `${tmpdir()}/chrysalis-cli-again-${process.pid}`, pids = `${tmpdir()}/chrysalis-cli-again-${process.pid}.pids`;
  writeFileSync(bin, `#!/bin/sh\nexec ${process.execPath} ${import.meta.dir}/fake-opencode-bin.ts "$@"\n`);
  chmodSync(bin, 0o755);
  const c = chrysalis({ env: { CHRYSALIS_OPENCODE: bin, FAKE_OPENCODE_PIDS: pids } }), work = `${c.dir}/work`;
  mkdirSync(work);
  const opencode = async () => (await (await fetch(`${c.here}/api/threads`)).json()).state;
  const say = async (text: string) => {
    const t0 = Date.now(), r = await fetch(`${c.here}/api/chat`, { method: "POST", headers: json, body: JSON.stringify({ stream: true, chrysalis_thread: { dir: work, text, title: text, model: "chrysalis/qwen3.8-27b" } }) }), sent = await r.text();
    return [r.status, sent.includes('"error"'), sent.trimEnd().endsWith("[DONE]"), Date.now() - t0 < 10_000] as const;
  };
  const back = async (cmd: string) => {
    const at = c.mark();
    c.type(`${cmd}\n`);
    await c.shown(/chat\s+http:\/\/localhost:\d+/, at);
    for (const end = Date.now() + 15_000; Date.now() < end; await Bun.sleep(50)) if ((await opencode().catch(() => "")) === "on") return "on";
    return opencode();
  };
  try {
    await c.shown(/chat\s+http:\/\/localhost:\d+/);
    expect((await fetch(`${c.here}/api/threads`, { method: "POST" })).status).toBe(200);
    expect(await say("before the switch")).toEqual([200, false, true, true]);
    const at = c.mark();
    c.type("network\n");
    await c.shown(/phone\s+https:/, at);
    expect(await back("local")).toBe("on");
    expect(await say("after the switch")).toEqual([200, false, true, true]);
    expect(await back("restart web")).toBe("on");
    expect(await say("after the restart")).toEqual([200, false, true, true]);
  } finally {
    await c.stop();
    rmSync(bin, { force: true });
    rmSync(pids, { force: true });
  }
}, 90_000);

test("OpenCode, the ordinary one and the incognito one, ends with the web server that started it, even when the web server is killed outright, and nothing of the incognito one is left in memory", async () => {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-kept-`), bin = `${dir}/opencode`, pids = `${dir}/pids`, work = `${dir}/work`;
  writeFileSync(bin, `#!/bin/sh\nexec ${process.execPath} ${import.meta.dir}/fake-opencode-bin.ts "$@"\n`);
  chmodSync(bin, 0o755);
  mkdirSync(work);
  const proc = Bun.spawn([process.execPath, "server.ts"], { cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", CHRYSALIS_URL: `http://127.0.0.1:${freePort()}`, CHRYSALIS_HOST: "127.0.0.1", NODE_ENV: "production", CHRYSALIS_TLS_DIR: `${dir}/tls`, HOME: dir,
      XDG_DATA_HOME: `${dir}/share`, XDG_STATE_HOME: `${dir}/state`, CHRYSALIS_OPENCODE: bin, FAKE_OPENCODE_PIDS: pids } });
  const alive = () => (existsSync(pids) ? readFileSync(pids, "utf8").trim().split("\n").map((l) => Number(l.split(" ")[0])) : []).filter((pid) => { try { process.kill(pid, 0); return true; } catch { return false; } });
  try {
    const reader = proc.stdout.getReader(), dec = new TextDecoder();
    let out = "";
    while (!/chrysalis web: \S+/.test(out)) {
      const { value, done } = await reader.read();
      if (done) throw new Error(`server.ts exited: ${out}`);
      out += dec.decode(value);
    }
    const here = /chrysalis web: (\S+)/.exec(out)![1].replace(/\/$/, "");
    expect((await fetch(`${here}/api/threads`, { method: "POST" })).status).toBe(200);
    const hidden = await fetch(`${here}/api/chat`, { method: "POST", headers: json, body: JSON.stringify({ stream: true, chrysalis_thread: { dir: work, text: "a secret", title: "a secret", model: "chrysalis/qwen3.8-27b", hidden: true } }) });
    expect([hidden.status, (await hidden.text()).trimEnd().endsWith("[DONE]")]).toEqual([200, true]);
    expect(alive().length).toBe(2);
    const left = () => [...new Set([memoryRoot(), tmpdir()].map((d) => realpathSync(d)))].flatMap((d) => readdirSync(d).filter((n) => n.startsWith(`chrysalis-run-${proc.pid}-`) || n === `chrysalis-ram-${proc.pid}`));
    expect(left().length).toBe(1);
    proc.kill("SIGKILL");
    await proc.exited;
    for (const end = Date.now() + 15_000; (alive().length || left().length) && Date.now() < end; ) await Bun.sleep(50);
    expect([alive(), left()]).toEqual([[], []]);
  } finally {
    proc.kill();
    for (const pid of alive()) process.kill(pid);
    rmSync(dir, { recursive: true, force: true });
  }
}, 60_000);

test("with no model here yet, OpenCode still connects at once and is given Chrysalis's models, so its threads list and open; a model is needed only to send", async () => {
  const empty = mkdtempSync(`${tmpdir()}/chrysalis-oc-none-`), bin = `${empty}/opencode`;
  writeFileSync(bin, `#!/bin/sh\nexec ${process.execPath} ${import.meta.dir}/fake-opencode-bin.ts "$@"\n`);
  chmodSync(bin, 0o755);
  const c = chrysalis({ env: { CHRYSALIS_ENGINE_CMD: "", CHRYSALIS_HOME: empty, CHRYSALIS_OPENCODE: bin } });
  try {
    await c.shown("No model is downloaded yet");
    await c.shown(/chat\s+http:\/\/localhost:\d+/);
    const started = await Promise.race([fetch(`${c.here}/api/threads`, { method: "POST" }).then((r) => r.status), Bun.sleep(15_000).then(() => "never")]);
    expect(started).toBe(200);
    const on = await (await fetch(`${c.here}/api/threads`)).json();
    expect([on.state, on.models.map((m: { id: string }) => m.id)]).toEqual(["on", ["chrysalis/qwen3.8-27b", "chrysalis/qwen3.8-flash-next"]]);
  } finally {
    await c.stop();
    rmSync(empty, { recursive: true, force: true });
  }
}, 60_000);

test("once OpenCode has been connected, the next start of Chrysalis starts it at once, so its threads are there when the page opens; disconnecting it stops that", async () => {
  const empty = mkdtempSync(`${tmpdir()}/chrysalis-oc-early-`), bin = `${empty}/opencode`, tls = `${empty}/tls`, pids = `${empty}/pids`;
  writeFileSync(bin, `#!/bin/sh\nexec ${process.execPath} ${import.meta.dir}/fake-opencode-bin.ts "$@"\n`);
  chmodSync(bin, 0o755);
  const env = { CHRYSALIS_ENGINE_CMD: "", CHRYSALIS_HOME: empty, CHRYSALIS_OPENCODE: bin, CHRYSALIS_TLS_DIR: tls, FAKE_OPENCODE_PIDS: pids };
  const state = async (c: ReturnType<typeof chrysalis>) => (await (await fetch(`${c.here}/api/threads`)).json()).state;
  const until = async (c: ReturnType<typeof chrysalis>, want: string) => { for (const end = Date.now() + 15_000; Date.now() < end; await Bun.sleep(100)) if ((await state(c)) === want) return want; return state(c); };
  try {
    const first = chrysalis({ env });
    await first.shown(/chat\s+http:\/\/localhost:\d+/);
    await Bun.sleep(1500);
    expect(await state(first)).toBe("off");
    expect((await fetch(`${first.here}/api/threads`, { method: "POST" })).status).toBe(200);
    await first.stop();
    const second = chrysalis({ env });
    await second.shown(/chat\s+http:\/\/localhost:\d+/);
    expect(await until(second, "on")).toBe("on");
    expect((await fetch(`${second.here}/api/threads`, { method: "DELETE" })).status).toBe(204);
    await second.stop();
    const third = chrysalis({ env });
    await third.shown(/chat\s+http:\/\/localhost:\d+/);
    await Bun.sleep(1500);
    expect(await state(third)).toBe("off");
    await third.stop();
  } finally {
    rmSync(empty, { recursive: true, force: true });
  }
}, 90_000);

test("OpenCode started from the page before the model server answers waits for it and is given every model it lists, so the first message of a chat has Chrysalis's models", async () => {
  const engine = freePort(), bin = `${tmpdir()}/chrysalis-cli-opencode-early-${process.pid}`;
  writeFileSync(bin, `#!/bin/sh\nexec ${process.execPath} ${import.meta.dir}/fake-opencode-bin.ts "$@"\n`);
  chmodSync(bin, 0o755);
  let asked!: () => void;
  const models = new Promise<void>((r) => (asked = r));
  const before = Bun.listen({ hostname: "127.0.0.1", port: engine, socket: { data(s, d) { if (d.toString().startsWith("GET /v1/models")) asked(); s.end(); } } });
  const c = chrysalis({ env: { CHRYSALIS_URL: `http://127.0.0.1:${engine}`, FAKE_ENGINE_PORT: String(engine), CHRYSALIS_OPENCODE: bin } });
  try {
    await c.shown(/chat\s+http:\/\/localhost:\d+/);
    const started = fetch(`${c.here}/api/threads`, { method: "POST" });
    await models;
    before.stop(true);
    expect((await started).status).toBe(200);
    expect((await (await fetch(`${c.here}/api/threads`)).json()).models.map((m: { id: string }) => m.id)).toEqual(["chrysalis/qwen3.8-27b", "chrysalis/qwen3.8-flash-next"]);
  } finally {
    before.stop(true);
    await c.stop();
    rmSync(bin, { force: true });
  }
}, 60_000);

test("every model state reads the same in the terminal, the server, a chat and a thread, and sending from each does what it says: nothing downloaded refuses with where to get one, a download under way shows in both buttons and is waited for, a model downloaded but not running loads on the first message from a chat or a thread, and a thread on another model than the one loaded switches to it", async () => {
  const home = mkdtempSync(`${tmpdir()}/chrysalis-states-`), work = `${home}/work`, bin = `${home}/opencode`, finish = `${home}/finish`;
  mkdirSync(work);
  writeFileSync(bin, `#!/bin/sh\nexec ${process.execPath} ${import.meta.dir}/fake-opencode-bin.ts "$@"\n`);
  chmodSync(bin, 0o755);
  const env = { CHRYSALIS_HOME: home, CHRYSALIS_OPENCODE: bin, CHRYSALIS_GET_CMD: JSON.stringify([process.execPath, `${import.meta.dir}/fake-get.ts`]), FAKE_GET_FINISH: finish };
  const browser = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: ["--no-sandbox"] });
  let c = chrysalis({ env });
  const status = async () => (await fetch(`${c.here}/api/status`)).json();
  const start = async (body: object) => { const r = await fetch(`${c.here}/api/chrysalis/start`, { method: "POST", headers: json, body: JSON.stringify(body) }); return [r.status, r.status === 409 ? (await r.json()).error.message.split(":")[0] : ""]; };
  const reach = async (what: string, ok: (s: any) => boolean) => { for (const end = Date.now() + 15_000; ; await Bun.sleep(50)) { const s = await status(); if (ok(s)) return s; if (Date.now() > end) throw new Error(`the server never reached ${what}: ${JSON.stringify(s)}`); } };
  const button = (page: Page) => page.$eval("#model-btn", (b) => [b.querySelector(".model-name")!.textContent, b.querySelector(".dot")?.className.replace("dot ", ""), b.classList.contains("getting") ? (b.classList.contains("paused") ? "paused" : "getting") : "", (b as HTMLElement).title.split(/[:,]/)[0]]);
  const shows = async (page: Page, want: (string | undefined)[]) => {
    await page.bringToFront();
    for (const end = Date.now() + 15_000; ; await Bun.sleep(50)) { const got = await button(page); if (JSON.stringify(got) === JSON.stringify(want)) return; if (Date.now() > end) expect(got).toEqual(want); }
  };
  const said = (page: Page) => page.$$eval("#thread > article", (a) => (a.at(-1) as HTMLElement | undefined)?.innerText ?? "");
  const open = async () => { const page = await browser.newPage(); await page.setViewport({ width: 1200, height: 900 }); await page.goto(`${c.here}/`, { waitUntil: "load" }); return page; };
  const thread = async (page: Page) => {
    await page.click("#opencode");
    await page.waitForSelector("#oc-pop .invite-head");
    await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#oc-pop .invite-card")].find((b) => b.textContent === "Connect")?.click());
    await page.waitForFunction(() => /OpenCode 2/.test(document.querySelector("#oc-pop .invite-head")?.textContent ?? ""));
    await page.keyboard.press("Escape");
    await page.waitForSelector("#folder-btn:not([hidden])");
    await page.click("#folder-btn");
    await page.waitForSelector("#oc-pop .oc-path");
    await page.$eval("#oc-pop .oc-path", (e, v) => { (e as HTMLInputElement).value = v; (e as HTMLInputElement).focus(); }, work);
    await page.keyboard.press("Enter");
    await page.waitForFunction((want) => document.querySelector("#oc-pop .oc-use")?.getAttribute("title") === want, {}, work);
    await page.click("#oc-pop .oc-use");
    await page.waitForFunction((want) => document.querySelector("#folder-btn")?.getAttribute("title") === `Working in ${want}`, {}, work);
  };
  const choose = async (page: Page, name: string) => {
    await page.bringToFront();
    await page.click("#model-btn");
    await page.waitForSelector("#model-pop:not([hidden]) .model-item");
    await page.evaluate((name) => [...document.querySelectorAll<HTMLElement>("#model-pop .model-item")].find((i) => i.querySelector(".model-label")!.textContent!.toLowerCase().endsWith(name))!.click(), name);
  };
  const type = async (page: Page, words: string) => { await page.bringToFront(); await page.type("#input", words); await page.keyboard.press("Enter"); };
  try {
    await c.shown("No model is downloaded yet");
    await c.shown(/chat\s+http:\/\/localhost:\d+/);
    await reach("nothing here", (s) => JSON.stringify(s.got) === "[]" && s.state === "offline");
    expect([await start({}), await start({ model: "qwen3.8-27b" }), await start({ model: "qwen3.8-flash-next" })]).toEqual(Array(3).fill([409, "No model is downloaded yet"]));
    const chat = await open();
    await shows(chat, ["Download a model", "offline", "", "No model is on this computer yet"]);
    await type(chat, "hello");
    await chat.waitForFunction(() => /No model is downloaded yet/.test((document.querySelector("#thread > article:last-child") as HTMLElement | null)?.innerText ?? ""));
    const page = await open();
    await thread(page);
    await shows(page, ["Download a model", "offline", "", "Qwen3.8-27B is not on this computer yet"]);
    await type(page, "hello");
    await page.waitForFunction(() => /No model is downloaded yet/.test((document.querySelector("#thread > article:last-child") as HTMLElement | null)?.innerText ?? ""));
    expect(c.out()).not.toContain("is downloading");

    await choose(page, "27b");
    await c.shown("qwen3.8-27b is downloading");
    await reach("the 27B downloading", (s) => s.download?.model === "qwen3.8-27b" && !s.download.paused);
    await shows(page, ["27B", "loading", "getting", "chrysalis/qwen3.8-27b"]);
    await shows(chat, ["27B", "loading", "getting", "Downloading Qwen3.8-27B"]);
    expect([await start({ model: "qwen3.8-flash-next" }), await start({ model: "qwen3.8-27b" })]).toEqual([[409, "No model is downloaded yet"], [202, ""]]);
    expect((await fetch(`${c.here}/api/chrysalis/download`, { method: "POST", headers: json, body: JSON.stringify({ how: "pause" }) })).status).toBe(202);
    await reach("the download paused", (s) => s.download?.paused === true);
    await shows(page, ["27B", "loading", "paused", "chrysalis/qwen3.8-27b"]);
    await shows(chat, ["27B", "loading", "paused", "Paused Qwen3.8-27B"]);
    expect((await fetch(`${c.here}/api/chrysalis/download`, { method: "POST", headers: json, body: JSON.stringify({ how: "resume" }) })).status).toBe(202);
    await reach("the download resumed", (s) => s.download && !s.download.paused);
    writeFileSync(finish, "");
    await reach("the 27B loaded once it arrived", (s) => s.state === "online" && !s.loading && s.model === "qwen3.8-27b" && !s.download && JSON.stringify(s.got) === '["qwen3.8-27b"]');
    await shows(page, ["27B", "online", "", "chrysalis/qwen3.8-27b"]);
    await shows(chat, ["27B", "online", "", "Qwen3.8-27B"]);

    await choose(page, "flash-next");
    await reach("Flash-Next downloaded and loaded once picked", (s) => s.state === "online" && !s.loading && s.model === "qwen3.8-flash-next" && !s.download && JSON.stringify(s.got) === '["qwen3.8-27b","qwen3.8-flash-next"]');
    await shows(page, ["Flash-Next", "online", "", "chrysalis/qwen3.8-flash-next"]);
    await choose(page, "27b");
    await shows(page, ["27B", "offline", "", "Qwen3.8-27B is not loaded"]);
    await type(page, "which model");
    await reach("the 27B loaded again for the thread's message", (s) => s.state === "online" && !s.loading && s.model === "qwen3.8-27b");
    await shows(page, ["27B", "online", "", "chrysalis/qwen3.8-27b"]);
    await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send");
    expect(await said(page)).not.toMatch(/not downloaded|No model/);

    await c.stop();
    c = chrysalis({ env });
    await c.shown(/chat\s+http:\/\/localhost:\d+/);
    await reach("both here and none running", (s) => JSON.stringify(s.got) === '["qwen3.8-27b","qwen3.8-flash-next"]' && s.state === "offline");
    const again = await open();
    await thread(again);
    await shows(again, ["27B", "offline", "", "Qwen3.8-27B is not loaded"]);
    await type(again, "hello again");
    await reach("the 27B loaded for the thread's first message", (s) => s.state === "online" && !s.loading && s.model === "qwen3.8-27b");
    await shows(again, ["27B", "online", "", "chrysalis/qwen3.8-27b"]);
    expect(await said(again)).not.toMatch(/not downloaded|No model/);
  } finally {
    await browser.close();
    await c.stop();
    rmSync(home, { recursive: true, force: true });
  }
}, 180_000);
