// The web server (server.ts) with the backends people run, for real: llama.cpp, Ollama, LM Studio, KoboldCpp and vLLM,
// each installed in data/backends (README: Rebuilding) and started here in turn on the CPU, with the same small
// model, Qwen3.5-2B (pictures, tools, thinking that turns off). Each is connected by its address as a person pastes it,
// and asked what the page asks: a reply with its thinking apart from its answer, one with thinking off, a tool call run
// on a real MCP server (fake-mcp.ts) and answered from its result, a picture. llama.cpp runs with a key: one it refuses,
// and the key never shown to a page. The page itself, in headless Chromium, connects to llama.cpp and chats with it.
// `make test_backends`; a backend that is not installed fails its tests.
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { deflateSync } from "node:zlib";
import puppeteer from "puppeteer-core";

process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";          // the web server's own authority, made for this run
const DATA = `${import.meta.dir}/../data`, B = `${DATA}/backends`, GGUF = `${B}/models/qwen3.5-2b-gguf`, KEY = "chrysalis-test-key";
const LMS = `${process.env.HOME}/.lmstudio/bin/lms`;
const MINUTES = (n: number) => n * 60_000;

interface Spec { name: string; port: number; model: string; ready: string; start(): Bun.Subprocess | undefined; stop?(): void; key?: string;
  thinkingOff: "none" | "stub" }
const spawn = (cmd: string[], env: Record<string, string> = {}) =>
  Bun.spawn(cmd, { cwd: tmpdir(), env: { ...process.env, ...env }, stdout: "ignore", stderr: "ignore" });
const BACKENDS: Spec[] = [
  { name: "llama.cpp", port: 18180, model: `${GGUF}/Qwen3.5-2B-Q8_0.gguf`, ready: "/health", key: KEY, thinkingOff: "none",
    start: () => spawn([`${DATA}/llama.cpp/build/bin/llama-server`, "-m", `${GGUF}/Qwen3.5-2B-Q8_0.gguf`, "--mmproj", `${GGUF}/mmproj-Qwen3.5-2B-BF16.gguf`,
      "--no-mmproj-offload", "-ngl", "0", "-c", "16384", "--host", "127.0.0.1", "--port", "18180", "--api-key", KEY]) },
  { name: "Ollama", port: 18181, model: "qwen3.5:2b", ready: "/", thinkingOff: "none",
    start: () => spawn([`${B}/ollama/bin/ollama`, "serve"], { OLLAMA_HOST: "127.0.0.1:18181", OLLAMA_MODELS: `${B}/ollama-models` }) },
  // LM Studio's own service runs its server: started and stopped through its CLI
  { name: "LM Studio", port: 18182, model: "qwen3.5-2b", ready: "/v1/models", thinkingOff: "none",
    start: () => {
      for (const args of [["server", "stop"], ["unload", "--all"], ["load", "qwen3.5-2b", "--gpu", "off", "-c", "16384", "-y"], ["server", "start", "--port", "18182"]])
        Bun.spawnSync([LMS, ...args]);
      return undefined;
    },
    stop: () => { Bun.spawnSync([LMS, "server", "stop"]); Bun.spawnSync([LMS, "unload", "--all"]); } },
  // KoboldCpp and vLLM think as the model's chat template does unless they are started to (Qwen3.5-2B's does not, and
  // vLLM then files the answer as thinking); no reasoning_effort turns it on. KoboldCpp writes "(Reasoning budget
  // exceeded)" as the thinking of a reply sent reasoning_effort "none".
  { name: "KoboldCpp", port: 18183, model: "koboldcpp/Qwen3.5-2B-Q8_0", ready: "/v1/models", thinkingOff: "stub",
    start: () => spawn([`${B}/koboldcpp/koboldcpp`, "--model", `${GGUF}/Qwen3.5-2B-Q8_0.gguf`, "--mmproj", `${GGUF}/mmproj-Qwen3.5-2B-BF16.gguf`,
      "--contextsize", "16384", "--host", "127.0.0.1", "--port", "18183", "--jinja", "--jinjathink", "true", "--skiplauncher", "--quiet"]) },
  // the flags Qwen's model card gives for vLLM's tool calls and thinking; no profiling of the vision tower, which takes
  // over 10 minutes on the CPU
  { name: "vLLM", port: 18184, model: "qwen3.5-2b", ready: "/v1/models", thinkingOff: "none",
    start: () => spawn([`${B}/vllm/venv/bin/vllm`, "serve", `${B}/models/qwen3.5-2b-hf`, "--served-model-name", "qwen3.5-2b", "--host", "127.0.0.1",
      "--port", "18184", "--max-model-len", "16384", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
      "--default-chat-template-kwargs", '{"enable_thinking": true}', "--skip-mm-profiling"], { VLLM_CPU_KVCACHE_SPACE: "4" }) },
];

// a 64 x 64 PNG, all red
const RED = (() => {
  const crc = (b: Uint8Array) => { let c = ~0; for (const x of b) { c ^= x; for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1)); } return ~c >>> 0; };
  const chunk = (type: string, data: Uint8Array) => {
    const out = new Uint8Array(12 + data.length), v = new DataView(out.buffer);
    v.setUint32(0, data.length); out.set(new TextEncoder().encode(type), 4); out.set(data, 8); v.setUint32(8 + data.length, crc(out.subarray(4, 8 + data.length)));
    return out;
  };
  const ihdr = new Uint8Array(13), v = new DataView(ihdr.buffer);
  v.setUint32(0, 64); v.setUint32(4, 64); ihdr.set([8, 2, 0, 0, 0], 8);
  const row = [0, ...Array.from({ length: 64 }, () => [220, 20, 20]).flat()];
  const raw = new Uint8Array(Array.from({ length: 64 }, () => row).flat());
  return `data:image/png;base64,${Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", new Uint8Array())]).toString("base64")}`;
})();

const dir = mkdtempSync(`${tmpdir()}/chrysalis-backends-`), tls = mkdtempSync(`${tmpdir()}/chrysalis-tls-`);
writeFileSync(`${dir}/mcp.json`, JSON.stringify({ mcpServers: { fake: { command: process.execPath, args: ["fake-mcp.ts"], cwd: import.meta.dir } } }));
const json = { "Content-Type": "application/json" };
let web: Bun.Subprocess<"ignore", "pipe", "inherit">, base = "", check = "";
beforeAll(async () => {
  web = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    // no Chrysalis: nothing listens at its address
    env: { ...process.env, PORT: "0", CHRYSALIS_URL: "http://127.0.0.1:9", CHRYSALIS_HOST: "127.0.0.1", NODE_ENV: "production",
      CHRYSALIS_MCP_CONFIG: `${dir}/mcp.json`, CHRYSALIS_TLS_DIR: tls, CHRYSALIS_CHECK_PORT: "0" },
  });
  const reader = web.stdout.getReader(), dec = new TextDecoder();
  let out = "";
  while (!/chrysalis check: \S+/.test(out)) {
    const { value, done } = await reader.read();
    if (done) throw new Error(`server.ts exited: ${out}`);
    out += dec.decode(value);
  }
  base = out.match(/chrysalis web: (\S+)/)![1].replace(/\/$/, "");
  check = out.match(/chrysalis check: (\S+)/)![1].replace(/\/$/, "");
});
afterAll(() => web?.kill());

const connect = (url: string, key?: string) => fetch(`${base}/api/backend`, { method: "POST", headers: json, body: JSON.stringify({ url, ...(key ? { key } : {}) }) });
// A streamed reply as the page asks for one (its job's events), gathered: thinking, answer, tool events, rounds, usage
async function reply(body: object) {
  const r = await fetch(`${base}/api/chat`, { method: "POST", headers: json, body: JSON.stringify({ stream: true, stream_options: { include_usage: true }, ...body }) });
  expect(r.status).toBe(200);
  const got = { reasoning: "", content: "", tools: [] as any[], rounds: 0, usage: undefined as any, timing: false, error: undefined as string | undefined, done: false };
  const reader = r.body!.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    for (let end = buf.indexOf("\n\n"); end >= 0; end = buf.indexOf("\n\n")) {
      const data = buf.slice(0, end).split("\n").find((l) => l.startsWith("data: "))?.slice(6) ?? "";
      buf = buf.slice(end + 2);
      if (data === "[DONE]") { got.done = true; continue; }
      const ev = JSON.parse(data), d = ev.choices?.[0]?.delta ?? {};
      got.reasoning += d.reasoning_content ?? "";
      got.content += d.content ?? "";
      if (d.reasoning !== undefined) throw new Error("a reply's thinking reached the page as reasoning, not reasoning_content");
      if (ev.tool?.state === "done" || ev.tool?.state === "failed") got.tools.push(ev.tool);
      if (ev.round) got.rounds++;
      if (ev.usage) got.usage = ev.usage;
      if (ev.timing) got.timing = true;
      if (ev.error) got.error = ev.error.message;
    }
  }
  return got;
}

for (const b of BACKENDS) {
  describe(b.name, () => {
    let proc: Bun.Subprocess | undefined;
    beforeAll(async () => {
      proc = b.start();
      const deadline = Date.now() + MINUTES(15);
      for (;;) {
        const r = await fetch(`http://127.0.0.1:${b.port}${b.ready}`, { headers: b.key ? { Authorization: `Bearer ${b.key}` } : {} }).catch(() => null);
        if (r?.ok) break;
        if (proc && proc.exitCode !== null) throw new Error(`${b.name} exited (${proc.exitCode})`);
        if (Date.now() > deadline) throw new Error(`${b.name} did not start`);
        await Bun.sleep(1000);
      }
      expect((await connect(`localhost:${b.port}`, b.key)).status).toBe(200);   // each test can run alone
    }, MINUTES(16));
    afterAll(() => { proc?.kill(); b.stop?.(); });

    if (b.key) {
      test(`${b.name}: a key it refuses, or none, is said so; its key reaches it and no page, and is kept readable by this user only`, async () => {
        expect((await (await connect(`localhost:${b.port}`)).json()).error.message).toBe(`http://localhost:${b.port} wants a key`);
        expect((await (await connect(`localhost:${b.port}`, "not-the-key")).json()).error.message).toBe(`http://localhost:${b.port} refused the key`);
        const ok = await connect(`localhost:${b.port}`, b.key);
        expect(ok.status).toBe(200);
        const status = await (await fetch(`${base}/api/status`)).text();
        expect([status.includes(b.key!), JSON.parse(status).backend.key]).toEqual([false, true]);
        expect(statSync(`${tls}/backend.json`).mode & 0o777).toBe(0o600);
        expect(JSON.parse(readFileSync(`${tls}/backend.json`, "utf8")).key).toBe(b.key);
      });
    }

    test(`${b.name}: connected by its address as pasted (no http://, no /v1), it is named, and lists its model`, async () => {
      const r = await connect(`localhost:${b.port}`, b.key);
      const about = await r.json();
      expect([r.status, about.url, about.name, about.chrysalis, about.remote]).toEqual([200, `http://localhost:${b.port}/v1`, b.name, false, false]);
      const status = await (await fetch(`${base}/api/status`)).json();
      expect([status.state, status.backend.listed.models.map((m: any) => m.id)]).toEqual(["online", expect.arrayContaining([b.model])]);
    });

    test(`${b.name}: a reply streams its thinking apart from its answer, with its usage and timings`, async () => {
      const got = await reply({ model: b.model, temperature: 0, messages: [{ role: "user", content: "What is 17 + 25? Answer with the number only." }] });
      expect([got.error, got.reasoning.length > 20, got.content.includes("42"), got.usage?.completion_tokens > 0, got.timing, got.done])
        .toEqual([undefined, true, true, true, true, true]);
    }, MINUTES(10));

    test(`${b.name}: thinking off (reasoning_effort "none", the page's Off) answers without thinking`, async () => {
      const got = await reply({ model: b.model, temperature: 0, reasoning_effort: "none", messages: [{ role: "user", content: "What is 17 + 25? Answer with the number only." }] });
      expect([got.error, got.content.includes("42")]).toEqual([undefined, true]);
      expect(b.thinkingOff === "stub" ? got.reasoning.trim() : got.reasoning).toBe(b.thinkingOff === "stub" ? "(Reasoning budget exceeded)\nTime to respond now." : "");
    }, MINUTES(10));

    test(`${b.name}: a tool call runs on an MCP server, and the answer comes from its result in a second round`, async () => {
      const got = await reply({ model: b.model, temperature: 0, reasoning_effort: "none", chrysalis_web: { tools: { "mcp:fake": {} } }, messages: [
        { role: "user", content: "Call the echo tool with the text butterfly-7, then tell me exactly what it returned." }] });
      expect([got.error, got.tools.map((t) => [t.state, t.result]), got.rounds, got.content.includes("butterfly-7")])
        .toEqual([undefined, [["done", "echo: butterfly-7"]], 1, true]);
    }, MINUTES(10));

    test(`${b.name}: a picture reaches the model`, async () => {
      const got = await reply({ model: b.model, temperature: 0, reasoning_effort: "none", messages: [{ role: "user", content: [
        { type: "image_url", image_url: { url: RED } }, { type: "text", text: "What single color fills this picture? Answer with one word." }] }] });
      expect([got.error, /red/i.test(got.content)]).toEqual([undefined, true]);
    }, MINUTES(10));

    if (b.name === "llama.cpp") {
      test("the page, in a browser: the top bar shows the address it is connected to; llama.cpp's pasted there (and its key) connects it, and a reply shows its thinking and its answer", async () => {
        const nonce = crypto.randomUUID().replaceAll("-", "");
        await fetch(`${check}/check.gif?n=${nonce}`);
        const asked = await fetch(`${base}/pair/ask`, { method: "POST", headers: json, body: JSON.stringify({ nonce, name: "Linux · Chrome" }) });
        const cookie = asked.headers.get("set-cookie")!.split(";")[0].split("=");
        expect((await connect("127.0.0.1:9")).status).toBe(200);       // back on Chrysalis's address: the page connects llama.cpp itself
        const browser = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true,
          args: ["--no-sandbox", "--ignore-certificate-errors"] });
        try {
          const ctx = await browser.createBrowserContext();
          await ctx.setCookie({ name: cookie[0], value: cookie[1], domain: "127.0.0.1", path: "/", secure: true, httpOnly: true, sameSite: "Strict" });
          const page = await ctx.newPage();
          page.on("pageerror", (e) => { throw e; });
          // (it polls 4 times a second meanwhile: Chrysalis's address answers nothing, while this machine's Chrysalis unit runs)
          await page.goto(`${base}/`, { waitUntil: "load" });
          await page.waitForSelector("#backend:not([hidden])");
          expect(await page.$eval("#backend-url", (e) => (e as HTMLInputElement).value)).toBe("http://127.0.0.1:9/v1");   // where it is connected
          await page.click("#backend-url", { count: 3 });
          await page.type("#backend-url", `localhost:${b.port}`);
          await page.type("#backend-key", b.key!);
          await page.keyboard.press("Enter");
          await page.waitForFunction((url) => (document.querySelector("#backend-url") as HTMLInputElement).value === url
            && (document.querySelector("#backend-go") as HTMLElement).hidden && document.querySelector("#backend-dot")!.classList.contains("online"),
          { timeout: 20_000 }, `http://localhost:${b.port}/v1`);
          expect(await page.$eval("#model-btn", (e) => e.textContent)).toBe("Qwen3.5-2B-Q8_0");
          await page.type("#input", "What is 17 + 25? Answer with the number only.");
          await page.keyboard.press("Enter");
          await page.waitForSelector(".msg.assistant .stats", { timeout: MINUTES(8) });
          const shown = await page.$eval(".msg.assistant", (m) => ({
            thought: m.querySelector("details.thinking .thought")?.textContent ?? "", answer: m.querySelector(".markdown")?.textContent ?? "",
            error: m.querySelector(".error")?.textContent ?? "" }));
          expect([shown.error, shown.thought.length > 20, shown.answer.includes("42")]).toEqual(["", true, true]);
        } finally {
          await browser.close();
        }
      }, MINUTES(10));
    }
  });
}

test("every backend this checks is installed", () => {
  const missing = [[`${DATA}/llama.cpp/build/bin/llama-server`, "llama.cpp"], [`${B}/ollama/bin/ollama`, "Ollama"], [LMS, "LM Studio"],
    [`${B}/koboldcpp/koboldcpp`, "KoboldCpp"], [`${B}/vllm/venv/bin/vllm`, "vLLM"], [`${GGUF}/Qwen3.5-2B-Q8_0.gguf`, "the model"]]
    .filter(([path]) => !existsSync(path)).map(([, name]) => name);
  expect(missing).toEqual([]);
});
