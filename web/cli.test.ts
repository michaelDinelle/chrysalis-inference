import { expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname } from "node:path";
import { memoryRoot } from "./platform";
import { plain } from "./terminal";
import { lanAddresses } from "./tls";

const json = { "Content-Type": "application/json" };
const freePort = () => { const s = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response() }), port = s.port!; s.stop(true); return port; };
const refused = (url: string, init: object = {}) => fetch(url, init).then((r) => `answered ${r.status}`, () => "refused");

function chrysalis(o: { pty?: boolean; mode?: string; env?: Record<string, string> } = {}) {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-cli-`), port = freePort(), engine = freePort();
  const cmd = [process.execPath, "cli.ts", ...(o.mode ? ["--mode", o.mode] : [])];
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
    expect((await fetch(`${api}/v1/models`)).status).toBe(404);
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

const UP = "\x1b[A", DOWN = "\x1b[B", RIGHT = "\x1b[C";
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

    await c.press("\t");
    await c.shown("Enter sets it");
    await c.shown(/Shows\s+● requests only\s+○ everything/);
    await c.press(DOWN, DOWN, RIGHT, "\r");
    await c.shown("verbose is on");
    await c.shown(/Shows\s+○ requests only\s+● everything/);
    expect((await (await fetch(`${c.engine}/chrysalis/stats`)).json()).asleep).toBeUndefined();
    await c.shown(/Model\s+○ Qwen3.8-27B\s+● Qwen3.8-Flash-Next\s+○ unload\s/);
    await c.press(UP, RIGHT, "\r");
    await c.shown("qwen3.8-flash-next is unloaded and the GPU is free");
    await c.shown(/Model\s+○ Qwen3.8-27B\s+○ Qwen3.8-Flash-Next\s+● unloaded/);
    expect((await (await fetch(`${c.engine}/chrysalis/stats`)).json()).asleep).toBe(true);
    await c.press("\t");

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
test("an incognito thread runs on a second OpenCode that keeps its database in memory, with its tools, beside the first, which is never stopped: it is in no list and in no file, the model server is told to keep its requests out of the prompt cache on disk, and the ordinary store and cache are untouched; deleting it, incognito end in the terminal, or stopping Chrysalis ends that OpenCode and deletes its folder", async () => {
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
    c.proc.kill();
    await c.proc.exited;
    expect(await until(() => !existsSync(third) && !started().some((o) => up(o.pid)))).toBe(true);
    expect([held(), anywhere("a third secret")]).toEqual([[], []]);
  } finally {
    await c.stop();
    rmSync(bin, { force: true });
    rmSync(pids, { force: true });
  }
}, 90_000);

test("OpenCode, the ordinary one and the incognito one, ends with the web server that started it, even when the web server is killed outright", async () => {
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
    proc.kill("SIGKILL");
    await proc.exited;
    for (const end = Date.now() + 15_000; alive().length && Date.now() < end; ) await Bun.sleep(50);
    expect(alive()).toEqual([]);
  } finally {
    proc.kill();
    for (const pid of alive()) process.kill(pid);
    rmSync(dir, { recursive: true, force: true });
  }
}, 60_000);
