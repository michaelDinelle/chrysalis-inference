import { afterAll, beforeAll, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, rmSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import puppeteer, { type Browser, type Page } from "puppeteer-core";
import { ARTIFACTS_PROMPT } from "./artifacts";
import { cardPng } from "./cast";

const RUN = mkdtempSync(`${tmpdir()}/chrysalis-test-`);
import { deflateSync } from "node:zlib";

process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";
const CHAT_ARGS = ["--no-sandbox", "--ignore-certificate-errors"];

const PANDA = '```svg panda\n<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">\n  <rect id="background" width="100" height="100" fill="#ffffff"/>\n' +
  '  <ellipse id="coat" cx="50" cy="60" rx="30" ry="25" fill="#ffffff" stroke="#222222"/>\n  <g id="burger">\n    <rect x="35" y="50" width="30" height="10" fill="#8b4513"/>\n  </g>\n</svg>\n```\n\nHere he is.';
const RED_COAT = '```edit panda\n<ellipse id="coat" cx="50" cy="60" rx="30" ry="25" fill="#d32f2f" stroke="#222222"/>\n<<<<<<< SEARCH\nfill="#ffffff"/>\n' +
  '=======\nfill="#fff3cd"/>\n>>>>>>> REPLACE\n```\n\nRed now.';
const COUNTER = '```html counter\n<!doctype html>\n<html><head><title>Counter</title>\n<style>\n  body { font: 16px sans-serif; margin: 0; padding: 16px; }\n' +
  '  #inc { background: #eeeeee; padding: 8px 14px; }\n  .card { padding: 12px; }\n</style></head>\n<body>\n<h1 id="title">Clicks</h1>\n' +
  '<section class="card" id="card">\n  <p id="count">0</p>\n  <button id="inc">Add</button>\n</section>\n<p id="note">Press the button.</p>\n' +
  '<script>\nlet n = 0;\ndocument.getElementById("inc").onclick = () => { n += 1; document.getElementById("count").textContent = n; };\n' +
  'document.body.dataset.ran = "yes";\n</script>\n</body></html>\n```\n\nA counter.';
const PAPER = '```latex paper\n\\documentclass{article}\n\\title{Rods}\n\\begin{document}\n\\maketitle\n\\section{Heat}\nThe rod cools as\n' +
  '\\begin{equation}\n  u_t = -\\alpha u_{xx}\n\\end{equation}\nwhich is Fourier\'s law.\n\nA second paragraph.\n\\end{document}\n```\n\nThe paper.';
const CUBE = '```html cube\n<!doctype html>\n<html><head><title>Cube</title>\n<script type="importmap">{"imports":{"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>\n' +
  '</head><body style="margin:0">\n<script type="module">\nimport * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";\n' +
  'import { OrbitControls } from "three/addons/controls/OrbitControls.js";\nimport { RoundedBoxGeometry } from "https://cdn.jsdelivr.net/npm/three@0.158.0/examples/jsm/geometries/RoundedBoxGeometry.js";\n' +
  'const renderer = new THREE.WebGLRenderer({ preserveDrawingBuffer: true });\nrenderer.setSize(300, 200);\ndocument.body.append(renderer.domElement);\n' +
  'const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, 1.5, 0.1, 100);\ncamera.position.z = 3;\n' +
  'scene.add(new THREE.Mesh(new RoundedBoxGeometry(1, 1, 1, 4, 0.1), new THREE.MeshNormalMaterial()));\nconst controls = new OrbitControls(camera, renderer.domElement);\n' +
  'renderer.render(scene, camera);\ndocument.body.dataset.three = THREE.REVISION + " " + (controls.target instanceof THREE.Vector3);\n</script>\n</body></html>\n```';
const OLD_CUBE = '```html old\n<!doctype html>\n<html><head><title>Old</title>\n<style>body { margin: 0; overflow: hidden; }\n' +
  '#title { position: absolute; top: 10px; width: 100%; text-align: center; color: #c33; }</style>\n' +
  '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js" integrity="sha512-dLxUelApnYxpLt6K2iomGngnHO83iUvZytA3YjDUCjT0HDOHKXnVYdf3hU4JjM8uEhxf9nD1/ey98U3t2vZ0qQ==" crossorigin="anonymous"></script>\n' +
  '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>\n</head><body>\n<div id="title">Old</div>\n<script>\n' +
  'const renderer = new THREE.WebGLRenderer();\nrenderer.setSize(innerWidth, innerHeight);\ndocument.body.append(renderer.domElement);\n' +
  'addEventListener("resize", () => renderer.setSize(innerWidth, innerHeight));\nconst camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);\nconst controls = new THREE.OrbitControls(camera, renderer.domElement);\n' +
  'document.body.dataset.three = THREE.REVISION + " " + (controls.object === camera);\n</script>\n</body></html>\n```';
const SCENE = '```aframe scene\n<!doctype html>\n<html><head><title>Scene</title>\n<script src="https://aframe.io/releases/1.5.0/aframe.min.js"></script>\n</head>\n<body style="margin:0">\n' +
  '<a-scene id="world" embedded style="width: 100%; height: 300px" vr-mode-ui="enabled: false" loading-screen="enabled: false">\n' +
  '  <a-box id="box" position="0 1.4 -3" color="#4CC3D9"></a-box>\n  <a-sky id="sky" color="#ECECEC"></a-sky>\n</a-scene>\n' +
  '<script>window.started = Math.random();</script>\n</body></html>\n```';
const HALL = '```html\nhall\n<!doctype html>\n<html><head><title>Hall</title></head><body><h1 id="sign">Welcome</h1></body></html>\n';
const GATE = '```html id="gate"\n<!doctype html>\n<html><head><title>Gate</title></head><body><p id="note">Open</p></body></html>\n```\n\nThe gate.';
const REPLIES: Record<string, string> = {
  "good morning mirela": "*She smiles.* Good morning.",
  "the kettle": '*She smiles.* "Good morning," *she says.* "Did you *sleep* well?"\nThe kettle sings.\n"It\'s still open\nAnd «bonjour» and `say "hi"` in code.',
  "steady speech": '*She looks up.* "You came back," *she says, and smiles at the window, where the snow is still falling.* "I kept the lamp lit for you, all night long."',
  "write the letter": '```markdown letter\nShe wrote "come home"\nand signed it.\n```\n\nA letter.',
  "a toast for the hive": "Mirela: *She raises a cup.* To the bees.\nTomas: To the bees!",
  "the same speaker": "Mirela: One.\nMirela: Two.\n\n*She hums.*\nTomas: Three.",
  "a duet": "Vox: *She sings.* La la.\nAda: Hum.\n\nHum hum.",
  "hall on its line": HALL,
  "gate as an attribute": GATE,
  "rename the sign": '```edit hall\n<h1 id="sign">Goodbye</h1>\n```\n\nRenamed.',
  "a page with no name": '```html\n<!doctype html>\n<html><head><title>Room</title></head><body><p id="lamp">Lamp</p></body></html>\n```\n\nA room.',
  "three cube": CUBE,
  "old cube": OLD_CUBE,
  "a scene": SCENE,
  "move the box": '```edit scene\n<a-box id="box" position="1 1.4 -3" color="#4CC3D9"></a-box>\n```\n\nMoved.',
  "write the paper": PAPER,
  "fix the sign": '```edit paper\n<<<<<<< SEARCH\n  u_t = -\\alpha u_{xx}\n=======\n  u_t = \\alpha u_{xx}\n>>>>>>> REPLACE\n```\n\nFixed.',
  "write the notes": '```markdown notes\n# Plan\n\nFirst we read.\n\n- milk\n- eggs\n\nThen we cook $x^2$.\n```\n\nNotes.',
  "write the script": '```python calc\ndef area(r):\n    return 3.14 * r * r\n\nprint(area(2))\n```\n\nA script.',
  "counter page": COUNTER,
  "blue button": '```edit counter\n<<<<<<< SEARCH\n#inc { background: #eeeeee;\n=======\n#inc { background: #0000ff;\n>>>>>>> REPLACE\n' +
    '<<<<<<< SEARCH\n<p id="count">0</p>\n=======\n<p id="count" class="big">0</p>\n>>>>>>> REPLACE\n```\n\nBlue.',
  "count by two": '```edit counter\n<<<<<<< SEARCH\nn += 1;\n=======\nn += 2;\n>>>>>>> REPLACE\n```\n\nBy two.',
  "draw the panda": PANDA,
  "red coat": RED_COAT,
  "broken edit": '```edit panda\n<<<<<<< SEARCH\nfill="#00ff00"\n=======\nfill="#0000ff"\n>>>>>>> REPLACE\n```',
  "did not apply": '```edit panda\n<<<<<<< SEARCH\n<rect x="35" y="50" width="30" height="10" fill="#8b4513"/>\n=======\n<rect x="35" y="50" width="30" height="10" fill="#0000ff"/>\n>>>>>>> REPLACE\n```',
  "make this orange": '```edit panda\n<g id="burger">\n  <rect x="35" y="50" width="30" height="10" fill="#ff8c00"/>\n</g>\n```',
  "capital of France": "Paris.",
  math: "Solve $ax^2 + bx + c = 0$:\n\n$$\nx = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\n$$\n\nAlso \\(e^{i\\pi} + 1 = 0\\) and\n\n\\[\n\\int_0^1 x\\,dx\n\\]\n\n" +
    "It costs $5 and $10. Escaped: \\$x\\$. Code: `$HOME` stays.\n",
  diagram: "```mermaid\nflowchart LR\n  A[Ask] --> B{Cached?}\n  B -->|yes| C[Append]\n  B -->|no| D[Prefill]\n```\n\n" +
    "```mermaid\nsequenceDiagram\n  User->>Chrysalis: question\n  Chrysalis-->>User: answer\n```\n\n```mermaid\nflowchart TD\n  A -->\n  not mermaid ((\n```\n",
  page: "```html\n<!DOCTYPE html>\n<html><head><title>Probe</title></head><body>\n<button id=\"b\" onclick=\"this.textContent = Number(this.textContent) + 1\">0</button>\n" +
    "<pre id=\"out\">running</pre>\n<script>\nconst out = [];\ntry { out.push(\"parent \" + parent.document.title); } catch { out.push(\"parent blocked\"); }\n" +
    "try { localStorage.setItem(\"k\", \"1\"); out.push(\"storage open\"); } catch { out.push(\"storage blocked\"); }\n" +
    "try { document.cookie = \"a=1\"; out.push(document.cookie ? \"cookie open\" : \"cookie blocked\"); } catch { out.push(\"cookie blocked\"); }\n" +
    "const img = new Image(); img.onload = () => out.push(\"image open\"); img.onerror = () => out.push(\"image blocked\"); img.src = \"PROBE_URL/probe.png\";\n" +
    "fetch(\"PROBE_URL/probe\", { method: \"POST\", mode: \"no-cors\" }).then(() => out.push(\"fetch open\"), () => out.push(\"fetch blocked\"))\n" +
    "  .finally(() => setTimeout(() => { document.getElementById(\"out\").textContent = out.sort().join(\",\"); }, 300));\n" +
    "</script>\n</body></html>\n```\n\nAfter the page.\n",
};
const enc = new TextEncoder();
const png = (() => {
  const crc = (b: Uint8Array) => { let c = ~0; for (const x of b) { c ^= x; for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1)); } return ~c >>> 0; };
  const chunk = (type: string, data: Uint8Array) => {
    const out = new Uint8Array(12 + data.length), v = new DataView(out.buffer);
    v.setUint32(0, data.length); out.set(enc.encode(type), 4); out.set(data, 8); v.setUint32(8 + data.length, crc(out.subarray(4, 8 + data.length)));
    return out;
  };
  const ihdr = new Uint8Array(13), v = new DataView(ihdr.buffer);
  v.setUint32(0, 2); v.setUint32(4, 2); ihdr.set([8, 2, 0, 0, 0], 8);
  const rows = new Uint8Array([0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 0, 0]);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(rows)), chunk("IEND", new Uint8Array())]);
})();

const TOOL_ASKS: Record<string, { name: string; args: string }> = {
  "tool echo": { name: "fake__echo", args: '{"text":  "hello"}' },
  "tool note": { name: "fake__write_note", args: '{"text": "n1"}' },
};
const AGENT_TASKS = [{ description: "Echo a word", prompt: "tool echo" }, { description: "Write a note", prompt: "tool note" }];
let lastStream: any = null, probes = 0, callIds = 0, chrysalisStopped = 0;
const streams: any[] = [], streamAt: number[] = [];
const checks: any[] = [];
let loaded = "qwen3.8-27b";
let loading: Promise<void> | null = null;
const fake = Bun.serve({
  hostname: "127.0.0.1",
  port: 0,
  async fetch(req) {
    const path = new URL(req.url).pathname;
    if (path.startsWith("/probe")) return (probes++, new Response(path.endsWith(".png") ? png : "", { headers: { "Content-Type": "image/png" } }));
    if (path === "/chrysalis/stats") return Response.json({ model: loaded, context: 262144, kv_tokens: 276000, cache: { free_kv_tokens: 270000 } });
    if (path === "/chrysalis/load") {
      const { model } = await req.json();
      await (loading = Bun.sleep(3000).then(() => { loaded = model; }));
      return Response.json({ model });
    }
    if (path !== "/v1/chat/completions") return new Response("not found", { status: 404 });
    const body = await req.json();
    const prefix = body.continue_final_message ? body.messages.at(-1) : undefined;
    if (!body.stream && /^You fix speech recognition errors/.test(body.messages[0]?.content)) {
      checks.push(body);
      const heard = String(body.messages.at(-1).content).split("Transcript: ").at(-1)!;
      await Bun.sleep(heard.startsWith("wait") ? 2000 : heard.includes("pie game") ? 1000 : 0);
      return Response.json({ choices: [{ message: { content: heard.includes("pie game") ? "pie game => Pygame\ntool: no" : "-" }, finish_reason: "stop" }] });
    }
    if (!body.stream) return Response.json({ choices: [{ message: { content: "A title" } }] });
    lastStream = body;
    streams.push(body);
    streamAt.push(Date.now());
    req.signal.addEventListener("abort", () => chrysalisStopped++);
    const at = body.messages.findLastIndex((m: any) => m.role === "user"), last = body.messages[at];
    const said = !last ? "" : (typeof last.content === "string" ? last.content : last.content.map((p: any) => p.text ?? "").join("")).replace(/^\[[^\]]*\]\n\n/, "");
    const round = body.messages.slice(at).filter((m: any) => m.role === "assistant" && m !== prefix).length, ask = TOOL_ASKS[said];
    const agents = said === "agents" && round === 0 && body.tools?.some((t: any) => t.function.name === "subagent");
    const calls = agents ? AGENT_TASKS.map((a) => ({ name: "subagent", args: JSON.stringify(a) })) : ask && body.tools && round === 0 ? [ask] : [];
    const next = /\[(.+) next\.\]$/.exec(said)?.[1];
    const think = agents ? "Two separate tasks: each goes to a sub-agent." : next ? `${next} has the floor.` : "";
    const text = agents ? "" : next ? `${next}: I go on.` : calls.length ? "Let me check." : ask ? `The tool said: ${body.messages.at(-1).content}` : said === "agents"
      ? `The sub-agents said: ${body.messages.slice(at).filter((m: any) => m.role === "tool").map((m: any) => m.content).join(" | ")}` : said.startsWith("slow")
      ? Array.from({ length: 40 }, (_, k) => `w${k} `).join("") : Object.entries(REPLIES).find(([k]) => said.includes(k))?.[1] ?? `You said: ${said}`;
    const step = said.startsWith("slow") ? 4 : 9, gap = said.startsWith("slow") ? 60 : said.startsWith("steady") ? 25 : 5;
    let i = 0, j = 0, restated = !prefix, read = false;
    return new Response(new ReadableStream({
      async pull(ctl) {
        await Bun.sleep(gap);
        if (!restated) {
          restated = true;
          return ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { chrysalis_prefix: { reasoning_content: prefix.reasoning_content || "\n", content: prefix.content } } }] })}\n\n`));
        }
        if (!read) { read = true; return ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: {} }], chrysalis_prompt: { prompt_tokens: 10, cached_tokens: 0 } })}\n\n`)); }
        if (j < think.length) { ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { reasoning_content: think.slice(j, j + step) } }] })}\n\n`)); j += step; return; }
        if (i < text.length) { ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: text.slice(i, i + step) } }] })}\n\n`)); i += step; return; }
        if (calls.length) {
          ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { tool_calls: calls.map((c, k) => ({ index: k, id: `call_${++callIds}`, type: "function",
            function: { name: c.name, arguments: c.args } })) } }] })}\n\n`));
          ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "tool_calls" }] })}\n\n`));
          ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 } })}\n\n`));
          ctl.enqueue(enc.encode("data: [DONE]\n\n"));
          return ctl.close();
        }
        ctl.enqueue(enc.encode(`data: ${JSON.stringify({ seed: 1, choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`));
        ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 } })}\n\n`));
        ctl.enqueue(enc.encode("data: [DONE]\n\n"));
        ctl.close();
      },
    }), { headers: { "Content-Type": "text/event-stream" } });
  },
});

let web: Bun.Subprocess<"ignore", "pipe", "inherit">, base = "", browser: Browser;
const dir = mkdtempSync(`${RUN}/web-`), LOG = `${dir}/mcp.log`, VOICE_LOG = `${dir}/voice.log`;
writeFileSync(`${dir}/mcp.json`, JSON.stringify({ mcpServers: {
  fake: { command: process.execPath, args: ["fake-mcp.ts"], cwd: import.meta.dir, env: { FAKE_MCP_LOG: LOG } },
  broken: { command: "/nonexistent/chrysalis-test-mcp" },
} }));
beforeAll(async () => {
  web = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", CHRYSALIS_URL: `http://127.0.0.1:${fake.port}`, CHRYSALIS_HOST: "127.0.0.1", NODE_ENV: "production", CHRYSALIS_MCP_CONFIG: `${dir}/mcp.json`,
      CHRYSALIS_VOICE_CMD: JSON.stringify([process.execPath, `${import.meta.dir}/fake-voice.ts`]), FAKE_VOICE_LOG: VOICE_LOG, CHRYSALIS_VOICE_IDLE_MS: "1500",
      CHRYSALIS_TLS_DIR: mkdtempSync(`${RUN}/tls-`), CHRYSALIS_CHECK_PORT: "0" },
  });
  const reader = web.stdout.getReader(), dec = new TextDecoder();
  let out = "";
  while (!/chrysalis web: \S+/.test(out)) {
    const { value, done } = await reader.read();
    if (done) throw new Error(`server.ts exited: ${out}`);
    out += dec.decode(value);
  }
  base = out.match(/chrysalis web: (\S+)/)![1].replace(/\/$/, "");
  REPLIES.page = REPLIES.page.replaceAll("PROBE_URL", `http://127.0.0.1:${fake.port}`);
  browser = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true,
    args: [...CHAT_ARGS, "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"] });
}, 30_000);
afterAll(async () => { await browser?.close(); web.kill(); fake.stop(true); rmSync(RUN, { recursive: true, force: true }); });

const chatContext = (b = browser) => b.createBrowserContext();
async function open(): Promise<Page> {
  const page = await (await chatContext()).newPage();
  await page.setViewport({ width: 1200, height: 900 });
  page.on("pageerror", (e) => { throw e; });
  await page.goto(`${base}/`, { waitUntil: "networkidle0" });
  return page;
}
async function send(page: Page, text: string) {
  const done = await page.$$eval(".msg .stats", (s) => s.length);
  await page.type("#input", text);
  await page.keyboard.press("Enter");
  await page.waitForFunction((n) => document.querySelectorAll(".msg .stats").length > n, { timeout: 20_000 }, done);
}

test("the context in the message box counts up while a reply is written (the prompt once read, then each token) and ends at the reply's own count", async () => {
  const page = await open();
  const used = async () => parseInt(await page.$eval("#meter > span", (s) => s.textContent ?? ""));
  await page.type("#input", "slow count");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /w3 /.test(document.querySelector(".msg.assistant")?.textContent ?? ""));
  const early = await used();
  await page.waitForFunction(() => /w20 /.test(document.querySelector(".msg.assistant")?.textContent ?? ""));
  const later = await used();
  expect([early > 10, later > early]).toEqual([true, true]);
  await page.waitForSelector(".msg.assistant .stats");
  expect(await used()).toBe(30);
  await page.close();
}, 30_000);

async function still(page: Page) {
  await page.waitForSelector("#shelf .art-canvas");
  await page.waitForFunction(() => new Promise<boolean>((done) => {
    const at = () => JSON.stringify(document.querySelector("#shelf .art-canvas")!.getBoundingClientRect()), was = at();
    requestAnimationFrame(() => requestAnimationFrame(() => done(at() === was)));
  }));
}
const fill = (page: Page, id: string, attr = "fill") => page.$eval(".art-canvas .svg-paper", (p, id, attr) =>
  p.shadowRoot!.querySelector(`[id="${id}"]`)?.getAttribute(attr as string) ?? null, id, attr);

test("an artifact is changed in place: each reply that makes a version has a card; the panel shows the latest, the edit's changes (a whole element by its id, a search) painted into the picture as it was, steps back to the version before, opens the version a card names, and Compare shows both side by side", async () => {
  const page = await open();
  await send(page, "draw the panda");
  await send(page, "red coat");
  expect(await page.$$eval(".art-card", (c) => c.map((x) => [x.closest("article")!.dataset.i, x.querySelector(".art-name")!.textContent, x.querySelector(".art-sub")!.textContent])))
    .toEqual([["1", "panda", "SVG · v1 · now v2"], ["3", "panda", "SVG · v2 · 2 changes"]]);
  await still(page);
  expect(await page.$eval("#shelf .art-canvas .art-label", (l) => l.textContent)).toBe("panda · v2 · 2 changes");
  expect([await fill(page, "coat"), await fill(page, "background"), await fill(page, "burger")]).toEqual(["#d32f2f", "#fff3cd", null]);
  expect(await page.$eval(".art-canvas .svg-paper", (p) => p.shadowRoot!.querySelector("#burger rect")?.getAttribute("fill"))).toBe("#8b4513");
  await page.click(".art-canvas [data-art=prev]");
  expect([await page.$eval(".art-canvas .art-label", (l) => l.textContent), await fill(page, "coat")]).toEqual(["panda · v1", "#ffffff"]);
  await page.click(".art-canvas [data-art=next]");
  expect(await fill(page, "coat")).toBe("#d32f2f");
  await page.click('.art-card[data-v="1"]');
  expect([await page.$eval(".art-canvas .art-label", (l) => l.textContent), await page.$$eval(".art-card.on", (c) => c.map((x) => (x as HTMLElement).dataset.v))])
    .toEqual(["panda · v1", ["1"]]);
  await page.click('.art-card[data-v="2"]');
  await page.click("#shelf-compare");
  const sides = () => page.$$eval("#shelf-stage > .art-canvas", (s) => s.map((x) => [x.querySelector(".art-label")!.textContent,
    x.querySelector(".svg-paper")!.shadowRoot!.querySelector("#coat")?.getAttribute("fill")]));
  expect(await sides()).toEqual([["panda · v1", "#ffffff"], ["panda · v2 · 2 changes", "#d32f2f"]]);
  await page.close();
}, 30_000);

test("the canvas's tools change the picture by hand, with no model: the paint bucket fills the shape clicked, a selection dragged moves, a corner dragged resizes, Delete removes it and Undo brings it back; each change is the artifact's next version, and the next question carries it whole, which the model's edit then changes", async () => {
  const page = await open();
  await send(page, "draw the panda");
  await still(page);
  const n0 = streams.length;
  const inPaper = (sel: string, fy = 0.5) => page.$eval(".art-canvas .svg-paper", (p, sel, fy) => {
    const r = p.shadowRoot!.querySelector(sel as string)!.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height * (fy as number) };
  }, sel, fy);
  const tools = (sel: string) => page.click(`.art-canvas .art-tools ${sel}`);
  const caption = () => page.$eval(".art-canvas .art-label", (l) => l.textContent);
  const coat = await inPaper("#coat", 0.85);
  await page.mouse.click(coat.x, coat.y);
  await page.keyboard.press("Delete");
  expect([await page.$eval("#pointing", (e) => e.textContent), await fill(page, "coat"), await caption()]).toEqual(["Pointing at panda · #coat", "#ffffff", "panda · v1"]);
  await page.keyboard.press("Escape");
  await page.click(".art-canvas [data-art=edit]");
  await page.$eval(".art-canvas .art-tools input", (i) => {
    (i as HTMLInputElement).value = "#00aa00";
    i.dispatchEvent(new Event("input"));
  });
  const coatNow = await inPaper("#coat", 0.85);
  await page.mouse.click(coatNow.x, coatNow.y);
  expect([await fill(page, "coat"), await caption()]).toEqual(["#00aa00", "panda · v2 · your edit, not sent yet"]);
  await tools("[data-tool=select]");
  const burger = await inPaper("#burger rect");
  await page.mouse.click(burger.x, burger.y);
  expect(await page.$eval("#pointing", (e) => e.textContent)).toBe("Pointing at panda · #burger");
  const box = await inPaper(".sel");
  await page.mouse.move(box.x, box.y);
  await page.mouse.down();
  await page.mouse.move(box.x + 10, box.y, { steps: 3 });
  await page.mouse.move(box.x + 20, box.y, { steps: 3 });
  await page.mouse.up();
  const shifted = await fill(page, "burger", "transform");
  expect(shifted).toMatch(/^translate\(\d+(\.\d+)? 0\)$/);
  const se = await inPaper(".sel i[data-h=se]"), nw = await inPaper(".sel i[data-h=nw]");
  await page.mouse.move(se.x, se.y);
  await page.mouse.down();
  await page.mouse.move(nw.x + (se.x - nw.x) * 1.5, nw.y + (se.y - nw.y) * 1.5, { steps: 4 });
  await page.mouse.up();
  const moved = await fill(page, "burger", "transform");
  expect(moved).toMatch(/^translate\(-?\d+(\.\d+)? -?\d+(\.\d+)?\) scale\(1\.(4|5)\d*\)$/);
  await page.keyboard.press("Delete");
  expect(await page.$eval(".art-canvas .svg-paper", (p) => p.shadowRoot!.querySelector("#burger"))).toBeNull();
  await tools("[data-do=undo]");
  expect([await fill(page, "burger", "transform"), await caption()]).toEqual([moved, "panda · v2 · your edit, not sent yet"]);
  expect(streams.length).toBe(n0);
  await send(page, "red coat");
  const asked = lastStream.messages.at(-1).content as string;
  expect([asked.includes("The user changed panda by hand since your last reply. As it is now:"), asked.includes('fill="#00aa00"'),
    asked.includes(`transform="${moved}"`)]).toEqual([true, true, true]);
  expect([await fill(page, "coat"), await fill(page, "burger", "transform"), await caption()]).toEqual(["#d32f2f", moved, "panda · v3 · 2 changes"]);
  expect(await page.$$eval(".edit-chip", (c) => c.map((x) => x.textContent))).toEqual(["you changed panda"]);
  await page.close();
}, 30_000);

const pageFrame = async (page: Page) => (await page.waitForSelector(".art-canvas iframe.page-frame:not(.next)"))!.contentFrame();
async function inPage(page: Page, sel: string) {
  for (let k = 0; ; k++) {
    const box = (await (await page.$(".art-canvas iframe.page-frame:not(.next)"))!.boundingBox())!;
    const r = await (await pageFrame(page)).$eval(sel, (e) => { const r = e.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; });
    if (r.y < box.height - 2 || k === 40) return { x: box.x + 1 + r.x, y: box.y + 1 + r.y };
    await Bun.sleep(50);
  }
}
async function pageTool(page: Page, sel: string) {
  if (!(await page.$(".art-canvas .art-tools.on"))) await page.click(".art-canvas [data-art=edit]");
  await page.click(`.art-canvas .art-tools ${sel}`);
  const mode = { select: "select", text: "text", fill: "bucket", done: "use" }[sel.match(/data-(?:tool|do)=(\w+)/)?.[1] ?? ""];
  if (mode) await (await pageFrame(page)).waitForFunction((t) => (window as any).__chrysalis.mode === t, {}, mode);
}
const pickColor = (page: Page, color: string) => page.$eval(".art-canvas .art-tools input", (i, color) => {
  (i as HTMLInputElement).value = color as string;
  i.dispatchEvent(new Event("input"));
}, color);

test("a page artifact is drawn as it is written, its script waiting, and runs once whole; an edit to its look goes into the running page, which keeps what it holds, and one to its script runs it anew", async () => {
  const page = await open();
  const done = await page.$$eval(".msg .stats", (s) => s.length);
  await page.type("#input", "steady counter page");
  await page.keyboard.press("Enter");
  const frame = await pageFrame(page);
  await frame.waitForSelector("#title", { timeout: 10_000 });
  expect([await page.$$eval(".msg .stats", (s) => s.length), await frame.$eval("body", (b) => b.dataset.ran ?? "")]).toEqual([done, ""]);
  expect(await page.$eval(".art-canvas", (f) => f.classList.contains("drawing"))).toBe(true);
  await page.waitForFunction((n) => document.querySelectorAll(".msg .stats").length > n, { timeout: 20_000 }, done);
  await frame.waitForFunction(() => document.body.dataset.ran === "yes");
  await still(page);
  for (let k = 0; k < 3; k++) await frame.click("#inc");
  expect(await frame.$eval("#count", (c) => c.textContent)).toBe("3");
  await send(page, "blue button");
  await frame.waitForFunction(() => getComputedStyle(document.getElementById("inc")!).backgroundColor === "rgb(0, 0, 255)");
  expect([await frame.$eval("#count", (c) => [c.className, c.textContent]), await page.$$eval(".art-canvas iframe", (f) => f.length)]).toEqual([["big", "3"], 1]);
  expect(await page.$eval(".art-canvas .art-label", (l) => l.textContent)).toBe("counter · v2 · 2 changes");
  await send(page, "count by two");
  await page.waitForFunction(() => document.querySelectorAll(".art-canvas iframe").length === 1 && !document.querySelector(".art-canvas iframe.next"), { timeout: 10_000 });
  const anew = await pageFrame(page);
  await anew.waitForFunction(() => document.body.dataset.ran === "yes");
  expect(await anew.$eval("#count", (c) => c.textContent)).toBe("0");
  await anew.click("#inc");
  expect([await anew.$eval("#count", (c) => c.textContent), await anew.$eval("#inc", (b) => getComputedStyle(b).backgroundColor)]).toEqual(["2", "rgb(0, 0, 255)"]);
  expect(await page.$eval(".art-canvas .art-label", (l) => l.textContent)).toBe("counter · v3 · 1 change");
  await page.close();
}, 40_000);

test("pages labelled as models also label them, the id alone on the block's first line (its fence left open as the reply ends) or as an attribute, are artifacts by that id, and one with none is \"page\": each runs as a page with the tools, and an edit by its id changes it", async () => {
  const page = await open(), shown = (label: string) => page.waitForFunction((label) => document.querySelector("#shelf .art-canvas .art-label")?.textContent === label, { timeout: 10_000 }, label);
  await send(page, "a hall on its line");
  await shown("hall · v1");
  const sign = (text: string) => pageFrame(page).then((f) => f.waitForFunction((text) => document.querySelector("#sign")?.textContent === text, { timeout: 10_000 }, text));
  await sign("Welcome");
  await send(page, "rename the sign");
  await sign("Goodbye");
  await send(page, "a gate as an attribute");
  await shown("gate · v1");
  expect(await page.$$eval(".art-card", (c) => c.map((x) => `${x.querySelector(".art-name")!.textContent}: ${x.querySelector(".art-sub")!.textContent}`)))
    .toEqual(["hall: Page · v1 · now v2", "hall: Page · v2 · 1 change", "gate: Page · v1"]);
  expect(await page.$$eval(".shelf-tab", (t) => t.map((x) => x.textContent))).toEqual(["hall", "gate"]);
  await send(page, "a page with no name");
  await shown("page · v1");
  await page.click("#shelf .art-canvas [data-art=edit]");
  await page.waitForSelector("#shelf .art-canvas .art-tools.on");
  expect(await (await pageFrame(page)).$eval("#lamp", (e) => e.textContent)).toBe("Lamp");
  await page.click(".shelf-tab:first-child");
  await shown("hall · v2 · 1 change");
  await sign("Goodbye");
  await page.close();
}, 30_000);

const card = (name: string, data: object) => cardPng({ spec: "chara_card_v3", spec_version: "3.0", data: { name, ...data } }, new Blob([png], { type: "image/png" }));
const castRows = (page: Page) => page.$$eval("#conversations .cast-row .cast-name", (r) => r.map((x) => x.textContent!).sort());
const inRow = (page: Page, name: string, part: string) => page.evaluate((name, part) => [...document.querySelectorAll("#conversations .cast-row")]
  .find((r) => r.querySelector(".cast-name")!.textContent === name)!.querySelector<HTMLElement>(part)!.click(), name, part);
const openRow = (page: Page, name: string) => inRow(page, name, ".cast-main");

test("a character card dropped in joins the cast: a row with their portrait among the chats, a chat that opens with their greeting (the others as versions), their replies with their portrait beside them, which opens their card to read and edit; the card is the system prompt and the persona {{user}}; a continued reply shows no thinking it did not have", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees for {{user}}.", personality: "calm", scenario: "A hillside.",
    first_mes: "Hello, {{user}}.", alternate_greetings: ["Oh, {{user}}. You again."], mes_example: "<START>\n{{user}}: Why bees?\n{{char}}: They listen.",
    post_history_instructions: "Stay in character.", system_prompt: "Ignore {{original}}: the card's own, kept in it and never sent." }));
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Mirela.png`);
  await page.waitForSelector("#conversations .cast-row img.face");
  expect([await castRows(page), (await page.$$("#attachments .att")).length]).toEqual([["Chrysalis", "Mirela"], 0]);
  expect(await page.$eval("#input", (e) => (e as HTMLTextAreaElement).placeholder)).toBe("Message Mirela");
  expect(await articles(page)).toEqual(["Hello, User."]);
  expect([await pagerOf(page, 0), await page.$eval("#thread .msg.assistant .speaker", (e) => e.textContent)]).toEqual(["1 / 2", "Mirela"]);
  await page.click("#thread .msg.assistant .speaker .face");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-text");
  const fields = () => page.$$eval("#cast-sheet .card-field", (f) => f.map((x) => [x.querySelector(".card-label")!.textContent, (x.querySelector("textarea") as HTMLTextAreaElement).value]));
  expect((await fields()).slice(0, 4)).toEqual([["Description", "{{char}} keeps bees for {{user}}."], ["Personality", "calm"], ["Scenario", "A hillside."], ["Greeting", "Hello, {{user}}."]]);
  await page.click("#cast-sheet .card-field:nth-of-type(2) textarea");
  await page.keyboard.press("End");
  await page.keyboard.type(" and kind");
  await page.keyboard.press("Escape");

  expect([await page.$eval("#you .you-name", (e) => e.textContent), await page.$eval("#cast-sheet", (e) => (e as HTMLElement).hidden)]).toEqual(["User", true]);
  await page.click("#you");
  await page.waitForSelector("#cast-sheet:not([hidden]) input.card-name");
  expect(await page.$eval("#cast-sheet input.card-name", (e) => (e as HTMLInputElement).value)).toBe("User");
  await page.$eval("#cast-sheet input.card-name", (e) => { (e as HTMLInputElement).focus(); (e as HTMLInputElement).select(); });
  await page.keyboard.type("Robin");
  await page.keyboard.press("Escape");
  await send(page, "good morning mirela");
  expect([await page.$eval("#you .you-name", (e) => e.textContent), await page.$eval("#thread .msg.user .speaker", (e) => e.textContent)]).toEqual(["Robin", "Robin"]);
  const one = lastStream.messages;
  expect(one[0].content).toBe("Write Mirela's next reply in a fictional chat between Mirela and Robin.\n\nMirela keeps bees for Robin.\n\nMirela's personality: calm and kind" +
    "\n\nScenario: A hillside.\n\nHow Mirela talks:\nRobin: Why bees?\nMirela: They listen.");
  expect(one.slice(1)).toEqual([{ role: "user", content: "[The scene begins.]" }, { role: "assistant", content: "Hello, User." },
    { role: "user", content: "good morning mirela\n\n[Stay in character.]" }]);
  expect(await page.$$eval("#thread .msg.assistant .speaker", (s) => s.map((x) => x.textContent))).toEqual(["Mirela", "Mirela"]);
  await page.$eval('#thread > article[data-i="2"] [data-act="cont"]', (b) => (b as HTMLElement).click());
  await page.waitForFunction(() => /Good morning\..*Good morning\./.test(document.querySelector('#thread > article[data-i="2"] .markdown')?.textContent ?? ""));
  await untilIdle(page);
  expect([lastStream.continue_final_message, await page.$eval('#thread > article[data-i="2"] .thinking', (t) => (t as HTMLElement).hidden)]).toEqual([true, true]);
  await page.close();
}, 40_000);

test("the user is a character like any other: a portrait opens its card and closes it again (another's switches to theirs); the user's card has every field a character's has, sent as theirs are, their greeting the first message of a new chat (in the box), their voice the one their messages are read in; their messages and the replies alike in bubbles, each from its writer's side", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", first_mes: "Hello, {{user}}." }));
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Mirela.png`);
  await page.waitForFunction(() => document.querySelector("#thread .speaker")?.textContent === "Mirela");
  const shown = () => page.$eval("#cast-sheet", (s) => (s as HTMLElement).hidden ? "" : s.querySelector<HTMLInputElement>("input.card-name")!.value);
  const steps: string[] = [];
  for (const click of [() => page.click("#thread .msg.assistant .speaker .face"), () => page.click("#you"), () => page.click("#you"),
    () => inRow(page, "Mirela", ".cast-pic"), () => inRow(page, "Mirela", ".cast-pic")]) { await click(); steps.push(await shown()); }
  expect(steps).toEqual(["Mirela", "User", "", "Mirela", ""]);
  const parts = () => page.$eval("#cast-sheet", (s) => [...s.querySelectorAll(".card-pic, .card-voice, .card-label")].map((x) => x.classList.contains("card-pic") ? "picture" : x.classList.contains("card-voice") ? "voice" : x.textContent));
  await inRow(page, "Mirela", ".cast-pic");
  const hers = await parts();
  await page.click("#you");
  expect(await parts()).toEqual(hers);
  await pickVoice(page, "#cast-sheet .card-voice .voice-name", "other-tts/bob");
  for (const [label, text] of [["Personality", "curious"], ["Greeting", "Hi, {{char}}."], ["How they talk", "{{user}}: Hm."], ["After the chat", "Keep it short."]]) {
    await page.evaluate((label) => {
      const f = [...document.querySelectorAll("#cast-sheet .card-field")].find((x) => x.querySelector(".card-label")!.textContent === label)!;
      f.closest("details")?.setAttribute("open", "");
      f.querySelector("textarea")!.focus();
    }, label);
    await page.keyboard.type(text);
  }
  await page.keyboard.press("Escape");
  await inRow(page, "Mirela", ".cast-new");
  expect(await page.$eval("#input", (e) => (e as HTMLTextAreaElement).value)).toBe("Hi, Mirela.");
  const done = await page.$$eval(".msg .stats", (s) => s.length);
  await page.focus("#input");
  await page.keyboard.press("Enter");
  await page.waitForFunction((n) => document.querySelectorAll(".msg .stats").length > n, { timeout: 20_000 }, done);
  expect(lastStream.messages[0].content).toBe("Write Mirela's next reply in a fictional chat between Mirela and User.\n\nMirela keeps bees.\n\n" +
    "User's personality: curious\n\nHow User talks:\nUser: Hm.");
  expect(lastStream.messages.at(-1)).toEqual({ role: "user", content: "Hi, Mirela.\n\n[Keep it short.]" });
  const look = await page.evaluate(() => {
    const at = (s: string) => document.querySelector(s)!.getBoundingClientRect(), mine = at(".msg.user .bubble"), hers = at(".msg.assistant > .markdown");
    const style = (s: string) => { const c = getComputedStyle(document.querySelector(s)!); return [c.borderTopWidth, c.backgroundColor, c.paddingLeft]; };
    return { sides: [hers.left - at(".msg.assistant .speaker .face").right, at(".msg.user .speaker .face").left - mine.right],
      bubbles: [style(".msg.user .bubble"), style(".msg.assistant > .markdown")] };
  });
  expect([look.sides[0] > 0 && look.sides[0] === look.sides[1], look.bubbles[0][0], look.bubbles[1]]).toEqual([true, "1px", look.bubbles[0]]);
  const n = voiceLog().length, button = '#thread > article[data-i="1"] [data-act="speak"]';
  await page.click(button);
  await page.waitForFunction((b) => !document.querySelector(`${b}.on`), { timeout: 10_000 }, button);
  const said = voiceLog().slice(n).filter((e) => e.ev === "chunk" || e.type === "say");
  expect([said.find((e) => e.type === "say")?.text, [...new Set(said.filter((e) => e.ev === "chunk").map((e) => `${e.tts}/${e.voice}`))]]).toEqual(["Hi, Mirela.", ["other-tts/bob"]]);
  await page.close();
}, 40_000);

function pngText(b: Uint8Array) {
  const v = new DataView(b.buffer, b.byteOffset), out: Record<string, string> = {};
  let idat = "";
  for (let at = 8; at + 12 <= b.length;) {
    const n = v.getUint32(at), type = String.fromCharCode(...b.subarray(at + 4, at + 8)), data = b.subarray(at + 8, at + 8 + n);
    if (type === "tEXt") { const z = data.indexOf(0); out[String.fromCharCode(...data.subarray(0, z))] = String.fromCharCode(...data.subarray(z + 1)); }
    if (type === "IDAT") idat += Buffer.from(data).toString("hex");
    at += 12 + n;
  }
  return { text: out, idat };
}

test("a card's portrait is the card's file, named for them and as it is now: a click saves it, the PNG any app reads with the picture as given", async () => {
  const page = await open();
  const saved = mkdtempSync(`${RUN}/saved-`), cdp = await page.createCDPSession();
  await cdp.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: saved, browserContextId: page.browserContext().id });
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", first_mes: "Hello, {{user}}." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mirela.png`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 2);
  await inRow(page, "Mirela", ".cast-pic");
  await page.click("#cast-sheet .card-field textarea");
  await page.keyboard.press("End");
  await page.keyboard.type(" Twelve hives.");
  await Bun.sleep(600);
  await page.click("#cast-sheet a.card-file");
  for (const t = Date.now(); !existsSync(`${saved}/Mirela.png`); await Bun.sleep(50)) if (Date.now() - t > 10_000) throw new Error(`not saved: ${readdirSync(saved)}`);
  await Bun.sleep(200);
  const got = pngText(readFileSync(`${saved}/Mirela.png`));
  const v3 = JSON.parse(Buffer.from(got.text.ccv3 ?? "", "base64").toString()), v2 = JSON.parse(Buffer.from(got.text.chara ?? "", "base64").toString());
  expect([readdirSync(saved), v3.data.name, v3.data.description, v2.data.description]).toEqual([["Mirela.png"], "Mirela", "{{char}} keeps bees. Twelve hives.", "{{char}} keeps bees. Twelve hives."]);
  expect(got.idat).toBe(pngText(png).idat);
  expect(await page.$eval("#cast-sheet a.card-file", (a) => [(a as HTMLAnchorElement).download, a.querySelector("img")!.src === (a as HTMLAnchorElement).href]))
    .toEqual(["Mirela.png", true]);
  await page.close();
}, 30_000);

test("the conversations are folders, Chrysalis's and each character's: a folder opened stays open while another's chat is, a closed one says how many chats it holds, and both stay as left after a reload", async () => {
  const page = await open();
  await send(page, "one for chrysalis");
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", first_mes: "Hello, {{user}}." }));
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Mirela.png`);
  await page.waitForFunction(() => document.querySelector("#thread .speaker")?.textContent === "Mirela");
  await send(page, "good morning mirela");
  const folders = () => page.$$eval("#conversations .folder", (f) => f.map((x) => [x.querySelector(".cast-name")!.textContent,
    x.classList.contains("open") ? `open: ${x.querySelectorAll(".conv").length}` : `closed: ${x.querySelector(".cast-count")?.textContent}`]));
  expect(await folders()).toEqual([["Mirela", "open: 1"], ["Chrysalis", "open: 1"]]);
  await openRow(page, "Chrysalis");
  expect(await folders()).toEqual([["Mirela", "open: 1"], ["Chrysalis", "closed: 1"]]);
  await openRow(page, "Chrysalis");
  await page.click("#conversations .folder:first-child .conv");
  expect(await folders()).toEqual([["Mirela", "open: 1"], ["Chrysalis", "open: 1"]]);
  await openRow(page, "Mirela");
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#conversations .folder");
  expect(await folders()).toEqual([["Mirela", "closed: 1"], ["Chrysalis", "open: 1"]]);
  await page.close();
}, 30_000);

test("a character deleted takes their chats with them (none become Chrysalis's), Undo brings it all back, and a character deleted in another tab takes their chats here too", async () => {
  const page = await open();
  await send(page, "one for chrysalis");
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", first_mes: "Hello, {{user}}." }));
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Mirela.png`);
  await page.waitForFunction(() => document.querySelector("#thread .speaker")?.textContent === "Mirela");
  await send(page, "good morning mirela");
  const kept = () => page.evaluate(() => Object.values(JSON.parse(sessionStorage.getItem("chats")!)).map((c: any) => c.who ?? "chrysalis").sort());
  const folders = () => page.$$eval("#conversations .folder", (f) => f.map((x) => `${x.querySelector(".cast-name")!.textContent}: ${x.querySelectorAll(".conv").length || x.querySelector(".cast-count")?.textContent}`));
  const mirela = (await kept()).find((w) => w !== "chrysalis")!;
  await inRow(page, "Mirela", ".cast-pic");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-acts .icon");
  await page.click("#cast-sheet .card-acts .icon");
  expect(await page.$eval("#cast-sheet .card-acts .icon", (b) => b.textContent)).toBe("Delete, with 1 chat?");
  await page.click("#cast-sheet .card-acts .icon");
  await page.waitForFunction(() => /^Mirela deleted, with 1 chat/.test(document.querySelector("#toast")?.textContent ?? ""));
  expect([await kept(), await folders(), await page.$$eval("#thread .speaker", (s) => s.length)]).toEqual([["chrysalis"], ["Chrysalis: 1"], 0]);
  await page.click("#toast .btn");
  await page.waitForFunction(() => document.querySelectorAll("#conversations .folder").length === 2);
  expect((await kept()).sort()).toEqual(["chrysalis", mirela].sort());
  await page.evaluate((id) => new Promise<void>((done) => {
    const open = indexedDB.open("chrysalis-cast", 1);
    open.onsuccess = () => { const t = open.result.transaction("people", "readwrite"); t.objectStore("people").delete(id); t.oncomplete = () => { open.result.close(); done(); }; };
  }), mirela);
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#conversations .folder");
  expect([await kept(), await folders()]).toEqual([["chrysalis"], ["Chrysalis: 1"]]);
  await page.close();
}, 30_000);

test("a group whose file names characters with no card here yet shows them all (their initials, in its face and on its card) and takes each in as their card comes", async () => {
  const page = await open();
  await Bun.write(`${dir}/Ada.png`, await card("Ada", { description: "{{char}} hums." }));
  await Bun.write(`${dir}/Bea.png`, await card("Bea", { description: "{{char}} drums." }));
  writeFileSync(`${dir}/Duo.json`, JSON.stringify({ spec: "chrysalis_group", name: "Duo", members: ["Ada", "Bea"] }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Ada.png`, `${dir}/Duo.json`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .folder").length === 3);
  const tiles = () => page.evaluate(() => [...[...document.querySelectorAll("#conversations .cast-row")].find((r) => r.querySelector(".cast-name")!.textContent === "Duo")!
    .querySelectorAll(".face.mosaic > *")].map((f) => f.tagName === "IMG" ? "picture" : f.textContent));
  expect(await tiles()).toEqual(["picture", "B"]);
  await inRow(page, "Duo", ".cast-pic");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-member.waiting");
  expect(await page.$$eval("#cast-sheet .strips > *", (s) => s.length)).toBe(2);
  await page.keyboard.press("Escape");
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Bea.png`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .folder").length === 4);
  expect(await tiles()).toEqual(["picture", "picture"]);
  await openRow(page, "Duo");
  await page.waitForSelector("#cast-next:not([hidden])");
  expect(await page.$$eval("#cast-next button[title$=' next']", (b) => b.map((x) => (x as HTMLElement).title))).toEqual(["Ada next", "Bea next"]);
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#conversations .folder");
  expect(await tiles()).toEqual(["picture", "picture"]);
  await page.close();
}, 30_000);

test("a group's replies are the scene, each paragraph its speaker's and one going on under the same face; a face has the next reply be theirs, thought through first; let them talk takes turns; cards dropped on a chat join it once (a chat with one character becomes a group, her replies hers); the cast stays after a reload", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", scenario: "A hillside.", first_mes: "Hello, {{user}}." }));
  await Bun.write(`${dir}/Tomas.png`, await card("Tomas", { description: "{{char}} sells honey.", first_mes: "Honey?" }));
  await Bun.write(`${dir}/Ida.png`, await card("Ida", { description: "{{char}} paints hives." }));
  writeFileSync(`${dir}/Hive.json`, JSON.stringify({ spec: "chrysalis_group", name: "Hive", members: ["Mirela", "Tomas"], scenario: "Market day." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mirela.png`, `${dir}/Tomas.png`, `${dir}/Hive.json`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 4);
  await openRow(page, "Hive");
  await page.waitForSelector("#cast-next:not([hidden])");
  await send(page, "a toast for the hive");
  const lines = (i: number) => page.$$eval(`#thread > article[data-i="${i}"] p.line`, (p) => p.map((x) => [x.querySelector(".who")?.textContent ?? "", x.classList.contains("same")]));
  expect(await lines(1)).toEqual([["Mirela", false], ["Tomas", false]]);
  expect(await page.$$eval("#thread p.line", (p) => p.map((x) => x.textContent))).toEqual(["MirelaShe raises a cup. To the bees.", "TomasTo the bees!"]);
  expect([lastStream.stop, lastStream.messages[0].content.includes("Scenario: Market day."), lastStream.messages[0].content.includes("A hillside.")])
    .toEqual([["\nUser:", "\n**User"], true, false]);
  await send(page, "the same speaker");
  expect(await lines(3)).toEqual([["Mirela", false], ["", true], ["", true], ["Tomas", false]]);

  await page.click('#cast-next button[title="Tomas next"]');
  await untilIdle(page);
  expect([lastStream.continue_final_message, lastStream.reasoning_effort !== undefined, lastStream.messages.at(-1)])
    .toEqual([undefined, true, { role: "user", content: "[Tomas next.]" }]);
  expect([await lines(4), await page.$eval('#thread > article[data-i="4"] .thought', (t) => t.textContent)]).toEqual([[["Tomas", false]], "Tomas has the floor."]);
  const n0 = streams.length;
  await page.click("#cast-next .talk");
  await page.waitForFunction(() => !!document.querySelector("#cast-next .talk:not(.on)")
    && document.querySelector("#send")?.getAttribute("aria-label") === "Send", { timeout: 20_000 });
  expect(streams.slice(n0).map((b) => b.messages.at(-1))).toEqual(["Mirela", "Tomas", "Mirela", "Tomas"].map((n) => ({ role: "user", content: `[${n} next.]` })));

  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Ida.png`);
  await page.waitForFunction(() => document.querySelectorAll("#cast-next button[title$=' next']").length === 3);
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Mirela.png`);
  await page.waitForFunction(() => /Mirela is in this chat already/.test(document.querySelector("#toast")?.textContent ?? ""));
  expect([await castRows(page), (await page.$$("#attachments .att")).length]).toEqual([["Chrysalis", "Hive", "Ida", "Mirela", "Tomas"], 0]);
  await openRow(page, "Mirela");
  await page.waitForFunction(() => document.querySelector("#thread .speaker")?.textContent === "Mirela");
  await page.evaluate(() => {
    const row = [...document.querySelectorAll("#conversations .cast-row")].find((r) => r.querySelector(".cast-name")!.textContent === "Tomas")!;
    const dt = new DataTransfer(), at = document.getElementById("thread")!;
    row.dispatchEvent(new DragEvent("dragstart", { dataTransfer: dt, bubbles: true }));
    for (const type of ["dragenter", "dragover", "drop"]) at.dispatchEvent(new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true }));
  });
  await page.waitForSelector("#cast-next:not([hidden])");
  expect([await castRows(page), await page.$$eval("#cast-next button[title$=' next']", (b) => b.map((x) => (x as HTMLElement).title)), await lines(0)])
    .toEqual([["Chrysalis", "Hive", "Ida", "Mirela", "Mirela and Tomas", "Tomas"], ["Mirela next", "Tomas next"], [[`Mirela`, false]]]);

  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#conversations .cast-row");
  expect(await castRows(page)).toEqual(["Chrysalis", "Hive", "Ida", "Mirela", "Mirela and Tomas", "Tomas"]);
  await page.close();
}, 60_000);

test("a page's tools change it by hand, with no model, in the page as it runs: Select points at a part, Text retypes it, the paint bucket fills a background, Delete takes a part out and Undo brings it back; the next question carries the page whole and the part pointed at", async () => {
  const page = await open();
  await send(page, "counter page");
  const frame = await pageFrame(page);
  await frame.waitForFunction(() => document.body.dataset.ran === "yes");
  await still(page);
  const n0 = streams.length, caption = () => page.$eval(".art-canvas .art-label", (l) => l.textContent);
  await frame.click("#inc");
  await frame.click("#inc");
  await pageTool(page, "[data-tool=text]");
  let at = await inPage(page, "#title");
  await page.mouse.click(at.x, at.y);
  await page.keyboard.press("End");
  await page.keyboard.type(" so far");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /your edit/.test(document.querySelector(".art-canvas .art-label")?.textContent ?? ""));
  expect([await frame.$eval("#title", (t) => t.textContent), await caption()]).toEqual(["Clicks so far", "counter · v2 · your edit, not sent yet"]);
  await pickColor(page, "#00aa00");
  await (await pageFrame(page)).waitForFunction(() => (window as any).__chrysalis.mode === "bucket");
  at = await inPage(page, "#card");
  await page.mouse.click(at.x, at.y - 30);
  await frame.waitForFunction(() => getComputedStyle(document.getElementById("card")!).backgroundColor === "rgb(0, 170, 0)");
  await pageTool(page, "[data-tool=select]");
  at = await inPage(page, "#note");
  await page.mouse.click(at.x, at.y);
  await page.waitForFunction(() => document.getElementById("pointing")!.textContent === "Pointing at counter · #note");
  await page.keyboard.press("Delete");
  await frame.waitForFunction(() => !document.getElementById("note"));
  await page.click(".art-canvas .art-tools [data-do=undo]");
  await frame.waitForFunction(() => !!document.getElementById("note"));
  expect([await frame.$eval("#count", (c) => c.textContent), await frame.$eval("#title", (t) => t.textContent), streams.length]).toEqual(["2", "Clicks so far", n0]);
  at = await inPage(page, "#note");
  await page.mouse.click(at.x, at.y);
  await page.waitForFunction(() => document.getElementById("pointing")!.textContent === "Pointing at counter · #note");
  await send(page, "capital of France");
  const asked = lastStream.messages.at(-1).content as string;
  expect([asked.includes("The user changed counter by hand since your last reply. As it is now:"), asked.includes('<h1 id="title">Clicks so far</h1>'),
    asked.includes('<section class="card" id="card" style="background: #00aa00">'), asked.includes('<p id="note">Press the button.</p>'),
    asked.includes('The user is pointing at the part #note of the artifact counter:\n<p id="note">Press the button.</p>')]).toEqual([true, true, true, true, true]);
  await page.close();
}, 40_000);

const docRoot = (page: Page) => page.$eval(".art-canvas .doc-paper", (p) => !!p.shadowRoot);
const inDoc = (page: Page, text: string) => page.$eval(".art-canvas .doc-paper", (p, text) => {
  const el = [...p.shadowRoot!.querySelectorAll<HTMLElement>("[data-at]")].findLast((e) => e.textContent!.includes(text as string))!;
  const r = el.getBoundingClientRect();
  return { x: r.x + Math.min(r.width / 2, 40), y: r.y + r.height / 2 };
}, text);

test("a LaTeX artifact reads as a document; pointing at a paragraph quotes its source, a double click retypes it there with no model, and the model's edit shows its changes as a diff", async () => {
  const page = await open();
  await send(page, "write the paper");
  await still(page);
  expect(await docRoot(page)).toBe(true);
  const shown = () => page.$eval(".art-canvas .doc-paper", (p) => p.shadowRoot!.querySelector(".doc-body")!.textContent!.replace(/\s+/g, " "));
  expect(await shown()).toMatch(/Rods.*1\s*Heat.*The rod cools as.*\(1\).*which is Fourier’s law\..*A second paragraph\./);
  let at = await inDoc(page, "A second paragraph");
  await page.mouse.click(at.x, at.y);
  expect(await page.$eval("#pointing", (e) => e.textContent)).toBe("Pointing at paper · “A second paragraph.”");
  await send(page, "capital of France");
  expect(lastStream.messages.at(-1).content).toBe("[The user is pointing at this part of the artifact paper:\nA second paragraph.]\n\ncapital of France");
  const n0 = streams.length;
  await page.click(".art-canvas [data-art=edit]");
  await still(page);
  at = await inDoc(page, "A second paragraph");
  await page.mouse.click(at.x, at.y, { count: 2 });
  await page.waitForFunction(() => !!document.querySelector(".art-canvas .doc-paper")!.shadowRoot!.querySelector("textarea.retype"));
  expect(await page.$eval(".art-canvas .doc-paper", (p) => p.shadowRoot!.querySelector("textarea")!.value)).toBe("A second paragraph.");
  await page.keyboard.press("End");
  await page.keyboard.type(" It has \\textbf{bold}.");
  await page.keyboard.down("Control");
  await page.keyboard.press("Enter");
  await page.keyboard.up("Control");
  await page.waitForFunction(() => /bold/.test(document.querySelector(".art-canvas .doc-paper")!.shadowRoot!.querySelector(".doc-body")!.textContent!));
  expect([await page.$eval(".art-canvas .doc-paper", (p) => p.shadowRoot!.querySelector("strong")?.textContent), streams.length,
    await page.$eval(".art-canvas .art-label", (l) => l.textContent)]).toEqual(["bold", n0, "paper · v2 · your edit, not sent yet"]);
  await send(page, "fix the sign");
  expect(await page.$eval(".art-canvas .art-label", (l) => l.textContent)).toBe("paper · v3 · 1 change");
  await page.click(".art-canvas [data-art=changes]");
  expect(await page.$$eval(".art-canvas .art-changes :is(.diff-add, .diff-del)", (s) => s.map((x) => x.textContent)))
    .toEqual(["-   u_t = -\\alpha u_{xx}", "+   u_t = \\alpha u_{xx}"]);
  expect(await page.$eval(".art-canvas .art-changes .diff-count", (c) => c.textContent)).toBe("+1 −1");
  await page.close();
}, 40_000);

test("a Markdown artifact reads as a document whose blocks the tools move and delete by hand: Delete takes the list out, Undo brings it back, Move up puts the last paragraph above it", async () => {
  const page = await open();
  await send(page, "write the notes");
  await still(page);
  const blocks = () => page.$eval(".art-canvas .doc-paper", (p) => [...p.shadowRoot!.querySelectorAll(".md > [data-at]")].map((b) => b.textContent!.replace(/\s+/g, " ").trim()));
  expect(await blocks()).toEqual(["Plan", "First we read.", "milk eggs", expect.stringMatching(/^Then we cook x\s*2/)]);
  const pick = async (text: string) => {
    const at = await inDoc(page, text);
    await page.mouse.click(at.x, at.y);
  };
  await page.click(".art-canvas [data-art=edit]");
  await still(page);
  await pick("milk");
  await page.keyboard.press("Delete");
  await page.waitForFunction(() => document.querySelector(".art-canvas .doc-paper")!.shadowRoot!.querySelectorAll(".md > [data-at]").length === 3);
  await page.keyboard.down("Control");
  await page.keyboard.press("z");
  await page.keyboard.up("Control");
  await page.waitForFunction(() => document.querySelector(".art-canvas .doc-paper")!.shadowRoot!.querySelectorAll(".md > [data-at]").length === 4);
  await pick("Then we cook");
  await page.click(".art-canvas .art-tools [data-do=up]");
  expect((await blocks()).slice(2).map((b) => b.slice(0, 12))).toEqual(["Then we cook", "milk eggs"]);
  await send(page, "capital of France");
  expect(lastStream.messages.at(-1).content).toContain("# Plan\n\nFirst we read.\n\nThen we cook $x^2$.\n\n- milk\n- eggs");
  await page.close();
}, 30_000);

test("code is changed by hand where it is shown: Edit, type, Done; the change is the artifact's next version, and Changes shows it", async () => {
  const page = await open();
  await send(page, "write the script");
  await page.click(".art-canvas [data-art=edit]");
  await page.waitForFunction(() => document.activeElement?.matches(".art-canvas code"));
  await page.keyboard.down("Control");
  await page.keyboard.press("End");
  await page.keyboard.up("Control");
  await page.keyboard.type("\nprint(area(3))");
  await page.click(".art-canvas [data-art=edit]");
  await page.waitForFunction(() => /your edit/.test(document.querySelector(".art-canvas .art-label")?.textContent ?? ""));
  expect(await page.$eval(".art-canvas pre code", (c) => c.textContent)).toBe("def area(r):\n    return 3.14 * r * r\n\nprint(area(2))\nprint(area(3))");
  await page.click(".art-canvas [data-art=changes]");
  expect(await page.$$eval(".art-canvas .art-changes :is(.diff-add, .diff-del)", (s) => s.map((x) => x.textContent)))
    .toEqual(["+ print(area(3))"]);
  await page.close();
}, 30_000);

test("3D pages load three.js and A-Frame from their usual addresses, which they cannot reach: three as a module with addons by an import map's names and by address, and the classic build older pages load; each fills the panel", async () => {
  const page = await open();
  for (const [ask, id, want] of [["three cube", "cube", "186 true"], ["steady old cube", "old", "186 true"]]) {
    await send(page, ask);
    await page.waitForFunction((id) => document.querySelector("#shelf .art-canvas")?.getAttribute("data-art") === id, { timeout: 10_000 }, id);
    const frame = await pageFrame(page);
    await frame.waitForFunction(() => !!document.body?.dataset.three, { timeout: 10_000 });
    expect(await frame.$eval("body", (b) => b.dataset.three)).toBe(want);
    expect(await page.$eval("#shelf .art-canvas iframe.page-frame:not(.next)", (f) => f.getBoundingClientRect().bottom > innerHeight - 40 && (f as HTMLElement).offsetHeight > 500)).toBe(true);
  }
  await page.click(".shelf-tab:first-child");
  await page.waitForFunction(() => document.querySelector("#shelf .art-canvas")?.getAttribute("data-art") === "cube");
  const frame = await pageFrame(page);
  await frame.waitForFunction(() => !!document.body?.dataset.three, { timeout: 10_000 });
  const lit = await frame.$eval("canvas", (c) => {
    const g = (c as HTMLCanvasElement).getContext("webgl2")!, px = new Uint8Array(4);
    g.readPixels(150, 100, 1, 1, g.RGBA, g.UNSIGNED_BYTE, px);
    return px[0] + px[1] + px[2];
  });
  expect(lit).toBeGreaterThan(0);
  expect(probes).toBe(0);
  await page.close();
}, 40_000);

test("an A-Frame scene is pointed into: the entity under the pointer is the one its camera sees there; the paint bucket colors it and the model's edit by id moves it, in the scene as it runs", async () => {
  const page = await open();
  await send(page, "a scene");
  let frame = await pageFrame(page);
  await frame.waitForFunction(() => (document.querySelector("a-scene") as any)?.renderStarted, { timeout: 15_000 });
  await still(page);
  const started = await frame.evaluate(() => (window as any).started);
  const boxAt = async () => {
    const box = (await (await page.$(".art-canvas iframe.page-frame:not(.next)"))!.boundingBox())!;
    const p = await frame.evaluate(() => {
      const scene = document.querySelector("a-scene") as any, T = (window as any).AFRAME.THREE, c = scene.canvas.getBoundingClientRect();
      const v = scene.querySelector("#box").object3D.getWorldPosition(new T.Vector3()).project(scene.camera);
      return { x: c.left + ((v.x + 1) / 2) * c.width, y: c.top + ((1 - v.y) / 2) * c.height };
    });
    return { x: box.x + 1 + p.x, y: box.y + 1 + p.y };
  };
  await pageTool(page, "[data-tool=select]");
  let at = await boxAt();
  await page.mouse.move(at.x, at.y);
  await page.mouse.click(at.x, at.y);
  await page.waitForFunction(() => document.getElementById("pointing")!.textContent === "Pointing at scene · #box");
  await page.mouse.click(at.x, at.y - 120);
  await page.waitForFunction(() => document.getElementById("pointing")!.textContent === "Pointing at scene · #sky");
  await pickColor(page, "#ff8800");
  await frame.waitForFunction(() => (window as any).__chrysalis.mode === "bucket");
  at = await boxAt();
  await page.mouse.click(at.x, at.y);
  await frame.waitForFunction(() => (document.querySelector("#box") as any).getAttribute("material").color === "#ff8800");
  await pageTool(page, "[data-do=done]");
  await send(page, "move the box");
  frame = await pageFrame(page);
  await frame.waitForFunction(() => (document.querySelector("#box") as any).object3D.position.x === 1);
  expect([await frame.evaluate(() => (window as any).started), await frame.evaluate(() => (document.querySelector("#box") as any).getAttribute("material").color)])
    .toEqual([started, "#4CC3D9"]);
  expect(probes).toBe(0);
  await page.close();
}, 60_000);

test("an edit that does not apply in full is asked for again, once, with the artifact as it is now, and its answer applies", async () => {
  const page = await open();
  await send(page, "draw the panda");
  await send(page, "broken edit");
  await page.waitForFunction(() => document.querySelectorAll(".msg.assistant .stats").length >= 3, { timeout: 20_000 });
  await Bun.sleep(300);
  const asked = streams.at(-1).messages.at(-1).content as string;
  expect([asked.includes("its SEARCH text is not in it"), asked.includes('```svg panda\n<svg viewBox="0 0 100 100"')]).toEqual([true, true]);
  expect(await page.$$eval(".msg.user.fix", (m) => m.length)).toBe(1);
  expect(await page.$$eval(".art-fail", (f) => f.map((x) => x.textContent).filter(Boolean))).toEqual(["Not applied: its SEARCH text is not in it"]);
  expect(await page.$eval(".art-canvas[data-v='3'] .svg-paper", (p) => p.shadowRoot!.querySelector("#burger rect")?.getAttribute("fill"))).toBe("#0000ff");
  await Bun.sleep(500);
  expect(await page.$$eval(".msg", (m) => m.length)).toBe(6);
  await page.close();
}, 30_000);

test("an artifact being written can be left for another: a tab chosen in the panel while it is written stays chosen until the reply ends", async () => {
  const page = await open(), shown = () => page.$eval("#shelf .art-canvas", (c) => c.getAttribute("data-art"));
  await send(page, "draw the panda");
  const done = await page.$$eval(".msg .stats", (s) => s.length);
  await page.type("#input", "steady counter page");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector("#shelf .art-canvas")?.getAttribute("data-art") === "counter" && !!document.querySelector(".art-card.live"),
    { timeout: 10_000 });
  await page.click(".shelf-tab:first-child");
  await Bun.sleep(500);
  expect([await shown(), await page.$$eval(".art-card.live", (c) => c.length)]).toEqual(["panda", 1]);
  await page.waitForFunction((n) => document.querySelectorAll(".msg .stats").length > n, { timeout: 20_000 }, done);
  expect(await shown()).toBe("panda");
  await page.close();
}, 30_000);

test("a chat whose reply is being written says Working, and for how long, where the list shows its age; once written it shows its age again", async () => {
  const page = await open();
  await page.type("#input", "slow count");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /^Working[1-9]\d*s$/.test(document.querySelector(".conv .conv-status")?.textContent ?? ""), { timeout: 10_000 });
  expect(await page.$(".conv .conv-ago")).toBeNull();
  await page.waitForSelector(".msg.assistant .stats", { timeout: 20_000 });
  await page.waitForFunction(() => !document.querySelector(".conv .conv-status") && document.querySelector(".conv .conv-ago")?.textContent === "now");
  await page.close();
}, 30_000);

test("pointing: the part under the pointer glows, a click keeps it in the message box, and the question tells the model that part as it is now", async () => {
  const page = await open();
  await send(page, "draw the panda");
  await still(page);
  const at = await page.$eval(".svg-fig[data-art] .svg-paper", (p) => {
    const r = p.shadowRoot!.querySelector("#burger rect")!.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  });
  const glow = () => page.$eval(".svg-fig[data-art] .svg-paper", (p) => p.shadowRoot!.querySelector("style.point")?.textContent ?? "");
  await page.mouse.move(at.x, at.y);
  expect(await glow()).toContain('[id="burger"] { filter');
  await page.mouse.click(at.x, at.y);
  expect(await page.$eval("#pointing", (e) => e.textContent)).toBe("Pointing at panda · #burger");
  await send(page, "make this orange");
  expect(lastStream.messages.at(-1).content).toBe('[The user is pointing at the part #burger of the artifact panda:\n<g id="burger">\n    ' +
    '<rect x="35" y="50" width="30" height="10" fill="#8b4513"/>\n  </g>]\n\nmake this orange');
  expect([await page.$eval("#pointing", (e) => (e as HTMLElement).hidden), await page.$eval(".point-chip", (e) => e.textContent)]).toEqual([true, "panda · #burger"]);
  expect(await page.$eval(".art-canvas[data-v='2'] .svg-paper", (p) => p.shadowRoot!.querySelector("#burger rect")?.getAttribute("fill"))).toBe("#ff8c00");
  await page.close();
}, 30_000);

test("TeX is typeset; prices, escaped dollars and code stay text", async () => {
  const page = await open();
  await send(page, "some math");
  const body = await page.$eval(".msg.assistant .markdown", (b) => ({
    math: [...b.querySelectorAll(".katex")].map((k) => k.querySelector("annotation")?.textContent ?? ""),
    display: b.querySelectorAll(".katex-display").length,
    text: [...b.querySelectorAll("p")].map((p) => p.textContent).join("\n"),
  }));
  expect(body.math).toEqual(["ax^2 + bx + c = 0", "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}", "e^{i\\pi} + 1 = 0", "\\int_0^1 x\\,dx"]);
  expect(body.display).toBe(2);
  expect(body.text).toContain("It costs $5 and $10. Escaped: $x$. Code: $HOME stays.");
  await page.close();
}, 30_000);

test("a chat reads as SillyTavern sets it: speech in quotes in its orange, the marks kept (one left open to its line's end), italics muted but a quote's inside one, a line break where one was written, code as written; the user's own text the same, its code and HTML as typed; a document's text as written", async () => {
  const page = await open();
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
  await send(page, '*knocks* "Is **the kettle** on?" and `ls "x"` <b>as typed</b>');
  const look = (sel: string) => page.$eval(sel, (b) => {
    const color = (e: Element | null) => e && getComputedStyle(e).color;
    return {
      quotes: [...b.querySelectorAll("q")].map((q) => q.textContent),
      marks: [...b.querySelectorAll("q")].map((q) => getComputedStyle(q, "::before").content + " " + getComputedStyle(q, "::after").content),
      breaks: b.querySelectorAll("br").length,
      text: color(b), quote: color(b.querySelector("q")), em: color(b.querySelector(":not(q) > em")), quoteEm: color(b.querySelector("q em")),
      html: b.innerHTML, said: b.textContent,
    };
  });
  const reply = await look(".msg.assistant .markdown"), mine = await look(".msg.user .bubble");
  expect(reply.quotes).toEqual(['"Good morning,"', '"Did you sleep well?"', '"It\'s still open', "«bonjour»"]);
  expect(reply.marks).toEqual(Array(4).fill("none none"));
  expect([reply.breaks, reply.quote, reply.quoteEm]).toEqual([3, "rgb(225, 138, 36)", "rgb(225, 138, 36)"]);
  expect([reply.em === reply.text, reply.em === reply.quote]).toEqual([false, false]);
  expect(reply.said).toContain('well?"\nThe kettle sings.\n"It');
  expect(mine.html).toBe('<em>knocks</em> <q>"Is <strong>the kettle</strong> on?"</q> and `ls "x"` &lt;b&gt;as typed&lt;/b&gt;');
  expect([mine.quote, mine.em, mine.marks]).toEqual([reply.quote, reply.em, ["none none"]]);
  await send(page, "write the letter");
  expect(await page.$eval(".art-canvas .doc-paper", (p) => {
    const md = p.shadowRoot!.querySelector(".md")!;
    return [md.querySelectorAll("q, br").length, md.textContent!.replace(/\s+/g, " ").trim()];
  })).toEqual([0, 'She wrote "come home" and signed it.']);
  await page.close();
}, 30_000);

test("speech and italics are set as they are written: a quote in its color from its opening mark, italics from their first word, and no asterisk shown on the way", async () => {
  const page = await open();
  await page.evaluate(() => {
    const seen: string[] = (window as any).seen = [];
    new MutationObserver(() => { const m = document.querySelector(".msg.assistant .markdown"); if (m) seen.push(m.innerHTML); })
      .observe(document.getElementById("thread")!, { childList: true, subtree: true, characterData: true });
  });
  await send(page, "steady speech");
  const seen: string[] = await page.evaluate(() => (window as any).seen);
  expect(seen.length).toBeGreaterThan(10);
  expect(seen.filter((h) => h.replace(/<[^>]*>/g, "").includes("*"))).toEqual([]);
  expect(seen.some((h) => /<q>"I kept[^"]*<\/q>/.test(h))).toBe(true);
  expect(seen.some((h) => /<em>she says(?!, and smiles at the window, where the snow is still falling\.<)[^<]*<\/em>/.test(h))).toBe(true);
  expect(seen.at(-1)).toContain('<q>"I kept the lamp lit for you, all night long."</q>');
  await page.close();
}, 30_000);

test("Mermaid blocks become diagrams; one that does not parse shows its source", async () => {
  const page = await open();
  await send(page, "a diagram");
  await page.waitForFunction(() => [...document.querySelectorAll(".svg-fig")].every((f) => f.querySelector(".svg-paper")!.shadowRoot!.querySelector("svg") || f.classList.contains("failed")), { timeout: 20_000 });
  const figs = await page.$$eval(".svg-fig", (fs) => fs.map((f) => ({
    state: f.querySelector(".svg-state")!.textContent,
    labels: [...(f.querySelector(".svg-paper")!.shadowRoot!.querySelectorAll("text, tspan"))].map((t) => t.textContent).join(" "),
    code: f.classList.contains("failed") ? f.querySelector(".svg-code code")!.textContent : "",
  })));
  expect(figs.map((f) => f.state)).toEqual(["Mermaid · flowchart", "Mermaid · sequence diagram", "Couldn't draw this diagram"]);
  for (const word of ["Ask", "Cached?", "Append", "Prefill"]) expect(figs[0].labels).toContain(word);
  expect(figs[1].labels).toContain("question");
  expect(figs[2].code).toContain("not mermaid ((");
  await page.close();
}, 30_000);

test("an HTML page runs in its sandbox and reaches nothing", async () => {
  const page = await open();
  await send(page, "a page");
  const frame = await (await page.waitForSelector("iframe.page-frame"))!.contentFrame();
  await frame.waitForFunction(() => document.getElementById("out")?.textContent !== "running", { timeout: 10_000 });
  expect(await frame.$eval("#out", (o) => o.textContent)).toBe("cookie blocked,fetch blocked,image blocked,parent blocked,storage blocked");
  expect(probes).toBe(0);
  await frame.click("#b");
  await frame.click("#b");
  expect(await frame.$eval("#b", (b) => b.textContent)).toBe("2");
  await page.close();
}, 30_000);

test("attached files go to Chrysalis as images, PDFs, videos and fenced text, stay after a reload, and travel in an export", async () => {
  const pdf = Buffer.from("%PDF-1.4\n%chrysalis\n"), mp4 = Buffer.from([0, 0, 0, 24, 102, 116, 121, 112, 109, 112, 52, 50]);
  writeFileSync(`${dir}/dot.png`, png);
  writeFileSync(`${dir}/fence.md`, "Has ``` inside.\n");
  writeFileSync(`${dir}/blob.bin`, new Uint8Array([0, 159, 146, 150, 255]));
  writeFileSync(`${dir}/doc.pdf`, pdf);
  writeFileSync(`${dir}/clip.mp4`, mp4);
  const page = await open();
  const cdp = await page.createCDPSession();
  await cdp.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: dir, browserContextId: page.browserContext().id });
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/dot.png`, `${dir}/fence.md`, `${dir}/blob.bin`, `${dir}/doc.pdf`, `${dir}/clip.mp4`);
  await page.waitForFunction(() => document.querySelectorAll("#attachments .att").length === 4);
  expect(await page.$eval("#toast", (t) => t.textContent)).toContain("blob.bin");
  await send(page, "look");
  const content = lastStream.messages.at(-1).content;
  const bytes = (url: string) => Buffer.from(url.split(",")[1], "base64");
  expect(content.map((p: any) => p.type)).toEqual(["image_url", "text", "file", "video_url", "text"]);
  expect(bytes(content[0].image_url.url).equals(png)).toBe(true);
  expect(content[1].text).toBe("fence.md\n````md\nHas ``` inside.\n````\n\n");
  expect(content[2].file.filename).toBe("doc.pdf");
  expect(bytes(content[2].file.file_data).equals(pdf)).toBe(true);
  expect(bytes(content[3].video_url.url).equals(mp4)).toBe(true);
  expect(content[4].text).toBe("look");
  await page.reload({ waitUntil: "networkidle0" });
  await page.waitForFunction(() => (document.querySelector(".msg-files img") as HTMLImageElement | null)?.naturalWidth === 2);
  await send(page, "again");
  expect(lastStream.messages.find((m: any) => m.role === "user").content).toEqual(content);
  await page.click("#export-all");
  const file = await (async () => { for (let k = 0; k < 50; k++) { const f = readdirSync(dir).find((x) => x.startsWith("chrysalis-chats-")); if (f) return f; await Bun.sleep(100); } })();
  const other = await open();
  await (await other.$("input#import-file"))!.uploadFile(`${dir}/${file}`);
  await other.waitForFunction(() => (document.querySelector(".msg-files img") as HTMLImageElement | null)?.naturalWidth === 2);
  await send(other, "and once more");
  expect(lastStream.messages.find((m: any) => m.role === "user").content).toEqual(content);
  expect(JSON.parse(readFileSync(`${dir}/${file}`, "utf8")).conversations[0].messages.length).toBe(4);
  unlinkSync(`${dir}/${file}`);
  await page.close();
  await other.close();
}, 60_000);

const pop = "#tools-pop";
const sw = (src: string, tool = "") => `${pop} [data-do="${tool ? "tool" : "src"}"][data-src="${src}"]${tool ? `[data-tool="${tool}"]` : ""}`;
const checked = (page: Page, sel: string) => page.$eval(sel, (b) => b.getAttribute("aria-checked"));
const badge = (page: Page) => page.$eval("#tools-btn", (b) => [b.querySelector(".tools-n")?.textContent ?? "", b.getAttribute("aria-label")]);
const subOf = (page: Page, src: string) => page.$eval(`${pop} [data-src="${src}"]`, (b) => b.closest(".tp-src")!.querySelector(".tp-sub")!.textContent);
async function toolsOn(page: Page, src = "mcp:fake") {
  await page.click("#tools-btn");
  await page.waitForSelector(sw(src));
  if ((await checked(page, sw(src))) !== "true") await page.click(sw(src));
  await page.waitForFunction((s) => document.querySelector(`#tools-pop [data-src="${s}"]`)?.closest(".tp-src")?.querySelector(".tp-sub")?.textContent?.includes("tools on"),
    { timeout: 20_000 }, src);
  await page.keyboard.press("Escape");
}
async function download(page: Page, into: string) {
  const cdp = await page.createCDPSession();
  await cdp.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: into, browserContextId: page.browserContext().id });
  await page.click("#export-all");
  for (let k = 0; k < 50; k++) { const f = readdirSync(into).find((x) => x.startsWith("chrysalis-chats-")); if (f) return `${into}/${f}`; await Bun.sleep(100); }
  throw new Error("no export");
}

test("the Tools popover turns servers and single tools on and off for this chat, and each chat keeps its own", async () => {
  const page = await open();
  expect(await badge(page)).toEqual(["", "Tools: off in this chat"]);
  await page.click("#tools-btn");
  await page.waitForSelector(sw("mcp:fake"));
  const rows = await page.$$eval(`${pop} .tp-src`, (r) => r.map((x) => [x.querySelector(".tp-name")!.textContent, x.querySelector(".tp-sub")!.textContent]));
  expect(rows).toEqual([["Fetch URL", "Reads web pages at public addresses"], ["Sub-agents", "Hands tasks to agents that work side by side"],
    ["fake", "Not started"], ["broken", "Not started"]]);
  await page.click(sw("mcp:fake"));
  await page.waitForFunction(() => document.querySelector('#tools-pop [data-src="mcp:fake"]')?.closest(".tp-src")?.querySelector(".tp-sub")?.textContent === "5 of 5 tools on",
    { timeout: 20_000 });
  expect(await badge(page)).toEqual(["5", "Tools: 5 on in this chat"]);
  await page.click(`${pop} [data-do="more"][data-src="mcp:fake"]`);
  const tools = await page.$$eval(`${pop} .tp-tool`, (r) => r.map((x) => [x.querySelector(".tp-tname")!.textContent, x.querySelector(".tp-tag")!.textContent]));
  expect(tools).toEqual([["big", "read-only"], ["Echo", "read-only"], ["get.weather", "read-only"], ["slow", "read-only"], ["write_note", "asks first"]]);
  await page.click(sw("mcp:fake", "write_note"));
  expect(await checked(page, sw("mcp:fake", "write_note"))).toBe("false");
  expect(await subOf(page, "mcp:fake")).toBe("4 of 5 tools on");
  await page.click(sw("fetch"));
  expect(await badge(page)).toEqual(["5", "Tools: 5 on in this chat"]);
  const pageLength = () => page.$$eval(`${pop} .tp-src`, (r) => r.map((x) => x.querySelector(".knob label")?.textContent ?? "").filter(Boolean));
  expect(await pageLength()).toEqual(["Page length"]);
  await page.click(sw("fetch"));
  expect(await pageLength()).toEqual([]);
  await page.click(sw("fetch"));
  await page.click(sw("mcp:broken"));
  await page.waitForSelector(`${pop} .tp-err [data-do="retry"][data-src="mcp:broken"]`, { timeout: 20_000 });
  expect(await page.$eval(`${pop} .tp-err`, (e) => e.textContent)).toMatch(/ENOENT|nonexistent|not found/i);
  await page.click(sw("mcp:broken"));
  await page.keyboard.press("Escape");
  expect(await page.$eval(pop, (p) => (p as HTMLElement).hidden)).toBe(true);
  expect(await page.evaluate(() => document.activeElement?.id)).toBe("tools-btn");

  await send(page, "hi");
  expect(lastStream.tools.map((t: any) => t.function.name)).toEqual(["fake__big", "fake__echo", "fake__get_weather", "fake__slow", "fetch_url"]);
  expect(lastStream.messages[0].content).toBe("You are Qwen3.8-27B, running locally in Chrysalis Inference. The tools offered here are the only ones in this chat, " +
    `and there are no files or projects to explore. Use a tool when it helps; otherwise answer directly.\n\n${ARTIFACTS_PROMPT}`);
  const first = await page.evaluate(() => location.hash);

  await page.click("#new-chat");
  expect(await badge(page)).toEqual(["5", "Tools: 5 on in this chat"]);
  await page.click("#tools-btn");
  await page.waitForSelector(sw("fetch"));
  await page.click(sw("fetch"));
  await page.click(sw("mcp:fake"));
  await page.keyboard.press("Escape");
  expect(await badge(page)).toEqual(["", "Tools: off in this chat"]);
  await send(page, "no tools");
  expect(lastStream.tools).toBeUndefined();
  expect(lastStream.messages[0].content).toBe(`You are Qwen3.8-27B, running locally in Chrysalis Inference. There are no tools or files in this chat. Answer directly.\n\n${ARTIFACTS_PROMPT}`);

  await page.goto(`${base}/${first}`, { waitUntil: "networkidle0" });
  await page.waitForFunction(() => document.querySelector("#tools-btn .tools-n")?.textContent === "5", { timeout: 10_000 });
  await page.close();
}, 60_000);

test("a tool call is a card where it happened; it opens to its arguments and result, and the history goes back byte for byte", async () => {
  const page = await open();
  await toolsOn(page);
  await send(page, "tool echo");
  const shape = await page.$eval(".msg.assistant", (a) => ({
    order: [...a.querySelectorAll(".round > .markdown, .round .tool-card, :scope > .markdown")].map((x) => x.classList.contains("tool-card") ? "card" : x.textContent!.trim()),
    card: [".tool-name", ".tool-src", ".tool-args", ".tool-state"].map((s) => a.querySelector(`.tool-card ${s}`)!.textContent),
    open: (a.querySelector(".tool-body") as HTMLElement).hidden,
  }));
  expect(shape.order).toEqual(["Let me check.", "card", "The tool said: echo: hello"]);
  expect(shape.card.slice(0, 3)).toEqual(["Echo", "fake", "hello"]);
  expect(shape.card[3]).toMatch(/^done in \d/);
  expect(shape.open).toBe(true);
  await page.click(".tool-card .tool-head");
  const body = await page.$eval(".tool-card .tool-body", (b) => ({ hidden: (b as HTMLElement).hidden, text: [...b.children].map((x) => x.textContent!.trim()) }));
  expect(body.hidden).toBe(false);
  expect(body.text).toEqual(["Arguments", '{\n  "text": "hello"\n}', "Result", "echo: hello"]);
  expect(await page.$eval(".tool-card .tool-head", (b) => b.getAttribute("aria-expanded"))).toBe("true");

  const round1 = streams.at(-1);
  await send(page, "and then");
  const sent = lastStream.messages;
  expect(JSON.stringify(sent.slice(0, round1.messages.length))).toBe(JSON.stringify(round1.messages));
  expect(sent.slice(round1.messages.length)).toEqual([{ role: "assistant", content: "The tool said: echo: hello" }, { role: "user", content: "and then" }]);
  expect(round1.messages.at(-2).tool_calls[0].function.arguments).toBe('{"text":  "hello"}');
  await page.close();
}, 60_000);

test("a tool that asks first shows the call with Allow once, Always allow and Deny, and does as answered", async () => {
  writeFileSync(LOG, "");
  const page = await open();
  await toolsOn(page);
  const answer = async (button: string) => {
    const before = await page.$$eval(".msg .stats", (s) => s.length);
    await page.type("#input", "tool note");
    await page.keyboard.press("Enter");
    await page.waitForSelector('.tool-card[data-state="approval"] .tool-ask:not([hidden])', { timeout: 20_000 });
    const ask = await page.$eval('.tool-card[data-state="approval"]', (c) => ({ args: c.querySelector(".tool-ask pre")!.textContent,
      buttons: [...c.querySelectorAll(".tool-ask button")].map((b) => b.textContent), state: c.querySelector(".tool-state")!.textContent }));
    expect(ask).toEqual({ args: '{\n  "text": "n1"\n}', buttons: ["Deny", "Always allow", "Allow once"], state: "asks to run" });
    await page.click(`.tool-card[data-state="approval"] [data-decision="${button}"]`);
    await page.waitForFunction((n) => document.querySelectorAll(".msg .stats").length > n, { timeout: 20_000 }, before);
    return page.$$eval(".msg.assistant .tool-card", (c) => c.at(-1)!.getAttribute("data-state"));
  };
  expect(await answer("once")).toBe("done");
  expect(readFileSync(LOG, "utf8").match(/wrote n1/g)?.length).toBe(1);
  expect(await answer("deny")).toBe("declined");
  expect(streams.at(-1).messages.at(-1).content).toContain("The user declined this call");
  expect(readFileSync(LOG, "utf8").match(/wrote n1/g)?.length).toBe(1);
  expect(await answer("always")).toBe("done");
  await send(page, "tool note");
  expect(await page.$$eval(".msg.assistant .tool-card", (c) => c.at(-1)!.getAttribute("data-state"))).toBe("done");
  expect(readFileSync(LOG, "utf8").match(/wrote n1/g)?.length).toBe(3);
  await page.click("#tools-btn");
  await page.waitForSelector("#tools-pop .tp-allow .chip");
  expect(await page.$$eval("#tools-pop .tp-allow .chip", (c) => c.map((x) => x.textContent))).toEqual(["fake__write_note"]);
  await page.click("#tools-pop .tp-allow .chip");
  await page.waitForFunction(() => !document.querySelector("#tools-pop .tp-allow"));
  await page.keyboard.press("Escape");
  await page.type("#input", "tool note");
  await page.keyboard.press("Enter");
  await page.waitForSelector('.tool-card[data-state="approval"] .tool-ask:not([hidden])', { timeout: 20_000 });
  await page.close();
}, 90_000);

test("sub-agents: two tasks handed off at once get a card each that says what it is doing, a call that asks first asks in its card, one can be stopped alone, and only their answers go back to the reply", async () => {
  writeFileSync(LOG, "");
  const page = await open();
  await toolsOn(page);
  await page.click("#tools-btn");
  await page.waitForSelector(sw("agents"));
  await page.click(sw("agents"));
  expect(await checked(page, sw("agents"))).toBe("true");
  await page.keyboard.press("Escape");
  const cards = () => page.evaluate(() => [...[...document.querySelectorAll(".msg.assistant")].at(-1)!.querySelectorAll(".agent-card")].map((c) => ({
    label: c.querySelector(".tool-name")!.textContent, now: c.querySelector(".agent-now")!.textContent, meta: c.querySelector(".agent-meta")!.textContent,
    state: c.getAttribute("data-state") })));
  const ask = async () => {
    await page.type("#input", "agents");
    await page.keyboard.press("Enter");
    await page.waitForSelector('.agent-card .tool-card[data-state="approval"] .tool-ask:not([hidden])', { timeout: 20_000 });
  };
  const done = (n: number) => page.waitForFunction((k) => document.querySelectorAll(".msg .stats").length > k, { timeout: 20_000 }, n);
  const results = () => { const m = streams.at(-1).messages; return m.slice(m.findLastIndex((x: any) => x.role === "user")).filter((x: any) => x.role === "tool").map((x: any) => x.content); };

  await ask();
  const asking = await cards();
  expect(asking.map((c) => c.label)).toEqual(["Echo a word", "Write a note"]);
  expect(await page.evaluate(() => {
    const t = [...[...document.querySelectorAll(".msg.assistant")].at(-1)!.querySelectorAll("details.thinking")].find((d) => !d.closest(".round, .agent-card"))!;
    return [t.querySelector("summary")!.textContent, t.classList.contains("live")];
  })).toEqual(["Thoughts", false]);
  expect(asking[1]).toMatchObject({ state: "running" });
  expect(asking[1].now).toStartWith("Asks to run ");
  await page.click('.agent-card:has(.tool-card[data-state="approval"]) .agent-stop:not([hidden])');
  await done(0);
  expect((await cards()).map((c) => [c.state, c.now])).toEqual([["done", "The tool said: echo: hello"], ["stopped", "Stopped before it finished"]]);
  const [echoed, stopped] = results();
  expect(echoed).toBe("The tool said: echo: hello");
  expect(stopped).toStartWith("Stopped before it finished: the user stopped it.");
  expect(readFileSync(LOG, "utf8")).not.toContain("wrote n1");

  const before = await page.$$eval(".msg .stats", (s) => s.length);
  await ask();
  await page.click('.agent-card .tool-card[data-state="approval"] [data-decision="once"]');
  await done(before);
  const end = await cards();
  expect(end.map((c) => c.state)).toEqual(["done", "done"]);
  expect(end.map((c) => c.meta)).toEqual(["1 tool · 40 tokens", "1 tool · 40 tokens"]);
  expect(readFileSync(LOG, "utf8").match(/wrote n1/g)?.length).toBe(1);
  expect(results()).toEqual(end.map((c) => c.now));
  const subs = streams.filter((b) => /^You are a sub-agent/.test(b.messages[0].content)).slice(-4);
  expect(subs.map((b) => [b.messages[1].content, b.tools.map((t: any) => t.function.name).includes("subagent"), "logprobs" in b]).sort()).toEqual([
    ["tool echo", false, false], ["tool echo", false, false], ["tool note", false, false], ["tool note", false, false]]);
  const last = await page.evaluate(() => (([...document.querySelectorAll(".msg.assistant")].at(-1) as HTMLElement).querySelector(":scope > .markdown") as HTMLElement).innerText);
  expect(last).toBe(`The sub-agents said: ${results().join(" | ")}`);

  await page.evaluate(() => ([...[...document.querySelectorAll(".msg.assistant")].at(-1)!.querySelectorAll(".agent-card .agent-head")][0] as HTMLElement).click());
  const body = await page.evaluate(() => {
    const b = [...[...document.querySelectorAll(".msg.assistant")].at(-1)!.querySelectorAll(".agent-card")][0].querySelector(".agent-body") as HTMLElement;
    return { hidden: b.hidden, task: b.querySelector(".agent-task")!.textContent, tool: b.querySelector(".tool-card .tool-name")!.textContent,
      answer: (b.querySelector(":scope > .markdown") as HTMLElement).innerText.trim() };
  });
  expect(body).toEqual({ hidden: false, task: "tool echo", tool: "Echo", answer: "The tool said: echo: hello" });
  await page.reload({ waitUntil: "networkidle0" });
  expect(await cards()).toEqual(end);
  await page.close();
}, 90_000);

const articles = (page: Page) => page.$$eval("#thread > article", (a) => a.map((x) => (x.querySelector(".bubble, .markdown") as HTMLElement)?.innerText.trim()));
const pagerOf = (page: Page, i: number) => page.$eval(`#thread > article[data-i="${i}"]`, (a) => a.querySelector(".pager-n")?.textContent ?? "");
async function untilIdle(page: Page) {
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send", { timeout: 20_000 });
}

test("a model picked while a reply is written is the one the next question asks for, though Chrysalis has the other until it switches", async () => {
  const page = await open();
  try {
    await page.type("#input", "slow one");
    await page.keyboard.press("Enter");
    await page.click("#new-chat");
    await page.click("#model-btn");
    await page.waitForSelector("#model-pop .model-item");
    await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#model-pop .model-item")].find((b) => /Flash/.test(b.textContent ?? ""))!.click());
    await Bun.sleep(1000);
    const n = streams.length;
    await page.type("#input", "which model");
    await page.keyboard.press("Enter");
    await untilIdle(page);
    expect(streams.slice(n).filter((b) => b.messages.at(-1).content === "which model").map((b) => b.model)).toEqual(["qwen3.8-flash-next"]);
  } finally {
    await loading;
    loaded = "qwen3.8-27b";
    await page.close();
  }
}, 30_000);

test("a question is kept as it is sent, before its reply begins: the chat left at once stays in the list, and after a reload it holds the question and says its reply never began", async () => {
  const page = await open();
  let held = true;
  await page.setRequestInterception(true);
  page.on("request", (r) => { if (held && !(r.method() === "POST" && new URL(r.url()).pathname === "/api/chat")) void r.continue(); });
  await page.type("#input", "a held question");
  await page.keyboard.press("Enter");
  await page.click("#new-chat");
  expect(await page.$$eval(".conv .conv-title", (t) => t.map((x) => x.textContent))).toEqual(["a held question"]);
  await page.reload({ waitUntil: "load" });
  await page.click(".conv");
  await page.waitForSelector("#thread > article.msg.assistant .error");
  expect(await articles(page)).toEqual(["a held question", "The page was left before this reply began: regenerate it"]);
  held = false;
  await page.setRequestInterception(false);
  await page.hover('#thread > article[data-i="1"]');
  await page.click('#thread > article[data-i="1"] [data-act="retry"]');
  await untilIdle(page);
  expect(await articles(page)).toEqual(["a held question", "You said: a held question"]);
  await page.close();
}, 30_000);

test("regenerate and edit make versions; the pager switches them and shows each one's own continuation", async () => {
  const page = await open();
  await send(page, "one");
  await send(page, "two");
  expect(await pagerOf(page, 1)).toBe("");
  await page.hover('#thread > article[data-i="1"]');
  await page.click('#thread > article[data-i="1"] [data-act="retry"]');
  await untilIdle(page);
  expect(await articles(page)).toEqual(["one", "You said: one"]);
  expect(await pagerOf(page, 1)).toBe("2 / 2");
  await page.click('#thread > article[data-i="1"] [data-act="prev"]');
  expect(await articles(page)).toEqual(["one", "You said: one", "two", "You said: two"]);
  expect(await pagerOf(page, 1)).toBe("1 / 2");
  expect(await page.evaluate(() => (document.activeElement as HTMLElement)?.dataset.act)).toBe("next");

  await page.hover('#thread > article[data-i="0"]');
  await page.click('#thread > article[data-i="0"] [data-act="edit"]');
  await page.waitForSelector(".editing textarea");
  await page.$eval(".editing textarea", (t) => { (t as HTMLTextAreaElement).value = ""; });
  await page.type(".editing textarea", "uno");
  await page.keyboard.down("Control"); await page.keyboard.press("Enter"); await page.keyboard.up("Control");
  await untilIdle(page);
  expect(await articles(page)).toEqual(["uno", "You said: uno"]);
  expect(await pagerOf(page, 0)).toBe("2 / 2");
  await page.focus('#thread > article[data-i="0"] [data-act="prev"]');
  await page.keyboard.press("Enter");
  expect(await articles(page)).toEqual(["one", "You said: one", "two", "You said: two"]);
  expect([await pagerOf(page, 0), await pagerOf(page, 1)]).toEqual(["1 / 2", "1 / 2"]);

  await page.reload({ waitUntil: "networkidle0" });
  expect(await articles(page)).toEqual(["one", "You said: one", "two", "You said: two"]);
  await send(page, "three");
  expect(lastStream.messages.filter((m: any) => m.role !== "system").map((m: any) => m.content))
    .toEqual(["one", "You said: one", "two", "You said: two", "three"]);

  await page.hover('#thread > article[data-i="0"]');
  await page.click('#thread > article[data-i="0"] [data-act="trash"]');
  expect(await articles(page)).toEqual(["uno", "You said: uno"]);
  expect(await pagerOf(page, 0)).toBe("");
  await page.click("#toast .btn");
  expect(await articles(page)).toEqual(["one", "You said: one", "two", "You said: two", "three", "You said: three"]);
  expect(await pagerOf(page, 0)).toBe("1 / 2");
  await page.close();
}, 60_000);

test("a reply being written in a version that is not on screen goes on, and shows when its version does", async () => {
  const page = await open();
  await send(page, "slow one");
  const reply = () => page.$eval('#thread > article[data-i="1"] .markdown', (b) => (b as HTMLElement).innerText.trim());
  const whole = await reply();
  await page.hover('#thread > article[data-i="1"]');
  await page.click('#thread > article[data-i="1"] [data-act="retry"]');
  await page.waitForFunction(() => /w3/.test(document.querySelector('#thread > article[data-i="1"] .markdown')?.textContent ?? ""), { timeout: 10_000 });
  await page.click('#thread > article[data-i="1"] [data-act="prev"]');
  expect(await reply()).toBe(whole);
  await Bun.sleep(1200);
  expect(await reply()).toBe(whole);
  await page.click('#thread > article[data-i="1"] [data-act="next"]');
  expect((await reply()).split(" ").length).toBeGreaterThan(15);
  await page.waitForFunction(() => document.querySelector('#thread > article[data-i="1"] .stats')?.textContent?.includes("tok/s"), { timeout: 20_000 });
  expect(await reply()).toBe(whole);
  expect(await pagerOf(page, 1)).toBe("2 / 2");
  await page.close();
}, 60_000);

test("export and import keep versions and tool rounds, and the imported chat resends the same history; an older file still imports", async () => {
  const into = mkdtempSync(`${dir}/export-`);
  const page = await open();
  await toolsOn(page);
  await send(page, "one");
  await page.hover('#thread > article[data-i="1"]');
  await page.click('#thread > article[data-i="1"] [data-act="retry"]');
  await untilIdle(page);
  await send(page, "tool echo");
  const file = await download(page, into);
  const saved = JSON.parse(readFileSync(file, "utf8")).conversations[0];
  expect(saved.messages[1].versions.others.length).toBe(1);
  expect(saved.messages[3].rounds[0].calls[0]).toMatchObject({ name: "fake__echo", args: '{"text":  "hello"}', result: "echo: hello", state: "done" });
  expect(JSON.stringify(saved)).not.toContain('"job"');
  await send(page, "after");
  const here = lastStream.messages;

  const other = await open();
  await (await other.$("input#import-file"))!.uploadFile(file);
  await other.waitForSelector(".tool-card");
  expect(await pagerOf(other, 1)).toBe("2 / 2");
  expect(await other.$eval("#tools-btn", (b) => b.classList.contains("on"))).toBe(true);
  await send(other, "after");
  expect(JSON.stringify(lastStream.messages)).toBe(JSON.stringify(here));
  await other.click('#thread > article[data-i="1"] [data-act="prev"]');
  expect((await articles(other)).length).toBe(2);

  const old = `${into}/old.json`;
  writeFileSync(old, JSON.stringify({ chrysalis: "chats", version: 1, exported: "2026-01-01T00:00:00.000Z", conversations: [{ id: "x", title: "Old", updated: 1, effort: "low",
    messages: [{ role: "user", content: "old q" }, { role: "assistant", content: "old a", reasoning: "hm", stats: { ttft: 1, tps: 2, tokens: 3, prompt: 4, cached: 0 } }] }] }));
  const third = await open();
  await (await third.$("input#import-file"))!.uploadFile(old);
  await third.waitForFunction(() => document.querySelectorAll("#thread > article").length === 2);
  expect(await third.$$eval(".pager", (p) => p.length)).toBe(0);
  await send(third, "next");
  expect(lastStream.messages).toEqual([{ role: "system", content: `You are Qwen3.8-27B, running locally in Chrysalis Inference. There are no tools or files in this chat. Answer directly.\n\n${ARTIFACTS_PROMPT}` }, { role: "user", content: "old q" },
    { role: "assistant", content: "old a", reasoning_content: "hm" }, { role: "user", content: "next" }]);
  for (const p of [page, other, third]) await p.close();
}, 90_000);

const noteShown = async (page: Page) => {
  await openSettings(page);
  const note = await page.$eval("#voice-prompt", (e) => (e as HTMLTextAreaElement).value);
  await closeSettings(page);
  return note;
};
const sent = (b: any) => b.messages;
const voiceLog = (): any[] => { try { return readFileSync(VOICE_LOG, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l)); } catch { return []; } };
const voicePages: Page[] = [];
async function voicePage(script: string, o: { system?: string; first?: (page: Page) => Promise<void>; keepPointer?: boolean } = {}) {
  const page = await (await chatContext()).newPage();
  voicePages.push(page);
  await page.setViewport({ width: 1600, height: 900 });
  page.on("pageerror", (e) => { throw e; });
  await page.evaluateOnNewDocument((v, system) => {
    if (!sessionStorage.getItem("settings2")) sessionStorage.setItem("settings2", JSON.stringify({ system, think: {}, fast: {}, voice: { name: v }, artifacts: false }));
  }, script, o.system ?? "");
  await page.goto(`${base}/`, { waitUntil: "networkidle0" });
  await o.first?.(page);
  await (o.keepPointer ? page.$eval("#voice", (b) => (b as HTMLElement).click()) : page.click("#voice"));
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 10_000 });
  const s = voiceLog().filter((e) => e.ev === "msg" && e.type === "hello" && e.voice === script).at(-1).s;
  return { page, session: () => voiceLog().filter((e) => e.s === s) };
}
async function pickVoice(page: Page, trigger: string, id: string) {
  await page.click(trigger);
  await page.waitForSelector(`#voice-menu:popover-open .voice-chip[data-voice="${id}"]`, { timeout: 10_000 });
  await page.click(`#voice-menu .voice-chip[data-voice="${id}"] > span`);
  await page.waitForFunction(() => !document.querySelector("#voice-menu:popover-open"));
}
const replyText = (page: Page, i: number) => page.$eval(`#thread > article[data-i="${i}"] .markdown`, (el) => el.textContent?.trim() ?? "");

test("a character's voice is picked and heard from their card with voice mode never opened (the voice service lists its voices and speaks the sample, with no hello: no microphone, no speech recognition), Escape closing the voice menu before the card, and voice mode speaks with it", async () => {
  const page = await open();
  await Bun.write(`${dir}/Vox.png`, await cardPng({ spec: "chara_card_v2", spec_version: "2.0", data: { name: "Vox", description: "{{char}} sings.", first_mes: "La." } },
    new Blob([png], { type: "image/png" })));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Vox.png`);
  await page.waitForSelector("#conversations .cast-row");
  const n = voiceLog().length;
  await inRow(page, "Vox", ".cast-pic");
  await pickVoice(page, "#cast-sheet .card-voice .voice-name", "other-tts/bob");
  await page.click("#cast-sheet .card-voice .voice-play");
  const until = async (ok: () => boolean) => { for (let t = Date.now(); !ok(); await Bun.sleep(50)) if (Date.now() - t > 10_000) throw new Error("timed out"); };
  await until(() => voiceLog().slice(n).some((e) => e.type === "preview"));
  const s = voiceLog().slice(n).find((e) => e.type === "catalog").s, own = () => voiceLog().filter((e) => e.s === s);
  expect(own().filter((e) => e.ev === "msg").map((e) => [e.type, e.tts, e.voice])).toEqual([["catalog", undefined, undefined], ["preview", "other-tts", "bob"]]);
  await until(() => own().some((e) => e.ev === "chunk"));
  await page.click("#cast-sheet .card-voice .voice-name");
  await page.waitForSelector("#voice-menu:popover-open");
  await page.keyboard.press("Escape");
  await page.waitForFunction(() => !document.querySelector("#voice-menu:popover-open"));
  expect(await page.$eval("#cast-sheet", (s) => (s as HTMLElement).hidden)).toBe(false);
  await page.keyboard.press("Escape");
  await until(() => own().some((e) => e.ev === "close"));
  const m = voiceLog().length;
  await page.click("#voice");
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 10_000 });
  const asked = voiceLog().slice(m).filter((e) => e.ev === "msg" && (e.type === "hello" || e.type === "config"));
  expect(asked.some((e) => e.voice === "bob" && (e.stack?.tts === "other-tts" || asked.some((x) => x.stack?.tts === "other-tts")))).toBe(true);
  await page.click("#send");
  await page.close();
}, 30_000);

test("a reply is read aloud from its speaker button with voice mode never opened (no hello: no microphone, no speech recognition): a group's paragraphs each in their speaker's voice, without their names; the button shows it, and stops it", async () => {
  const page = await open();
  await Bun.write(`${dir}/Vox.png`, await card("Vox", { description: "{{char}} sings.", extensions: { chrysalis: { voice: { tts: "other-tts", name: "bob" } } } }));
  await Bun.write(`${dir}/Ada.png`, await card("Ada", { description: "{{char}} hums." }));
  writeFileSync(`${dir}/Duet.json`, JSON.stringify({ spec: "chrysalis_group", name: "Duet", members: ["Vox", "Ada"] }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Vox.png`, `${dir}/Ada.png`, `${dir}/Duet.json`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 4);
  await openRow(page, "Duet");
  await page.waitForSelector("#cast-next:not([hidden])");
  await send(page, "a duet");
  const n = voiceLog().length, button = '#thread > article[data-i="1"] [data-act="speak"]';
  await page.click(button);
  await page.waitForSelector(`${button}.on`);
  await page.waitForFunction((b) => !document.querySelector(`${b}.on`), { timeout: 10_000 }, button);
  const s = voiceLog().slice(n).find((e) => e.ev === "msg").s, mine = voiceLog().filter((e) => e.s === s);
  expect(mine.filter((e) => e.ev === "msg").map((e) => e.type)).toEqual(["config", "config", "say", "say_end", "config", "say", "say_end"]);
  const says = mine.filter((e) => e.type === "say");
  expect(says.map((e) => e.text)).toEqual(["*She sings.* La la.", "Hum.\n\nHum hum."]);
  expect(says.map((x) => [...new Set(mine.filter((e) => e.ev === "chunk" && e.id === x.id).map((e) => `${e.tts}/${e.voice}`))]))
    .toEqual([["other-tts/bob"], ["fake-tts/plain"]]);
  await page.click(button);
  await page.waitForSelector(`${button}.on`);
  await page.click(button);
  expect(await page.$(`${button}.on`)).toBeNull();
  const until = async (ok: () => boolean) => { for (let t = Date.now(); !ok(); await Bun.sleep(50)) if (Date.now() - t > 10_000) throw new Error("timed out"); };
  await until(() => voiceLog().filter((e) => e.s === s).some((e) => e.type === "cancel"));
  await page.close();
}, 30_000);

test("voice mode: what is said shows as it is said, becomes the question, and the reply is spoken as it is written", async () => {
  const n0 = streams.length, c0 = checks.length;
  const { page, session } = await voicePage("plain");
  expect(await page.$eval(".composer-box", (el) => el.getBoundingClientRect().height > 0 && !!(el as HTMLElement).offsetParent)).toBe(true);
  expect(await page.$eval("#input", (el) => (el as HTMLTextAreaElement).offsetHeight > 0)).toBe(true);
  expect(await page.$eval("#send", (el) => el.getAttribute("aria-label"))).toBe("End voice mode");
  expect(await page.$eval("#voice-pill", (el) => (el as HTMLElement).title)).toMatch(/^Listening/);
  await page.waitForFunction(() => document.querySelector(".msg.ghost .bubble")?.textContent === "hello", { timeout: 10_000 });
  await page.waitForFunction(() => document.querySelector('#thread > article[data-i="0"] .bubble')?.textContent === "hello there", { timeout: 10_000 });
  expect(await page.$(".msg.ghost")).toBeNull();
  expect(await page.$('#thread > article[data-i="0"] .said')).not.toBeNull();
  await page.waitForFunction(() => /^voice \d\.\d\d s/.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  const note = await noteShown(page);
  expect(sent(lastStream)).toEqual([{ role: "user", content: `[${note}]\n\nhello there` }]);
  expect(lastStream.continue_final_message).toBeUndefined();
  const check = checks.findLast((b) => b.messages.at(-1).content === "Transcript: hello there");
  expect([check?.temperature, check?.stream, check?.model, check?.messages.length]).toEqual([0, undefined, "qwen3.8-27b-fast", 2]);
  expect(checks.slice(c0).map((b) => b.messages.at(-1).content)).toEqual(["Transcript: hello there"]);
  expect(await page.$eval("#energy", (e) => e.getAttribute("aria-valuetext"))).toBe("Max");
  expect([lastStream.model, lastStream.reasoning_effort]).toEqual(["qwen3.8-27b", "xhigh"]);
  const mine = streams.slice(n0), warm = mine.findIndex((b) => b.max_tokens === 1);
  expect(mine[warm]?.messages).toEqual([{ role: "user", content: "Hi" }]);
  expect(mine[warm]?.model).toBe(lastStream.model);
  expect(warm).toBeLessThan(mine.findIndex((b) => sent(b).at(-1)?.content.endsWith("hello there")));
  expect(await replyText(page, 1)).toBe("You said: hello there");
  expect(await page.$eval('#thread > article[data-i="0"] .msg-note summary', (e) => e.textContent)).toBe("Voice mode note");
  expect(await page.$eval('#thread > article[data-i="0"] .msg-note-text', (e) => e.checkVisibility())).toBe(false);
  await page.click('#thread > article[data-i="0"] .msg-note summary');
  expect(await page.$eval('#thread > article[data-i="0"] .msg-note-text', (e) => [e.textContent, e.checkVisibility()])).toEqual([note, true]);
  const log = session(), says = log.filter((e) => e.type === "say"), id = says[0].id;
  expect(says.every((e) => e.id === id)).toBe(true);
  expect(says.map((e) => e.text).join("")).toBe("You said: hello there");
  expect(log.findIndex((e) => e.type === "say_end" && e.id === id)).toBeGreaterThan(log.findLastIndex((e) => e.type === "say"));
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 10_000 });
  expect(session().filter((e) => e.type === "playback").map((e) => e.playing)).toEqual([true, false]);
  const told = session().filter((e) => e.type === "context").map((e) => e.text);
  expect(told[0]).toMatch(/^Qwen3\.8-27B\n/);
  expect(told.at(-1)).toMatch(/hello there\nYou said: hello there$/);
  await page.close();
}, 30_000);

test("voice mode: a question said with the pointer on a part of an artifact points at it (\"make this orange\")", async () => {
  const { page } = await voicePage("point", { keepPointer: true, first: async (page) => {
    await send(page, "draw the panda");
    await still(page);
    const at = await page.$eval(".svg-fig[data-art] .svg-paper", (p) => {
      const r = p.shadowRoot!.querySelector("#burger rect")!.getBoundingClientRect();
      return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
    });
    await page.mouse.move(at.x, at.y);
  } });
  await page.waitForFunction(() => document.querySelector('#thread > article[data-i="2"] .bubble')?.textContent === "make this orange", { timeout: 10_000 });
  await page.waitForSelector('#thread > article[data-i="3"] .stats', { timeout: 10_000 });
  expect(sent(lastStream).at(-1).content).toContain('The user is pointing at the part #burger of the artifact panda:\n<g id="burger">');
  expect(await page.$eval('#thread > article[data-i="2"] .point-chip', (e) => e.textContent)).toBe("panda · #burger");
  expect(await page.$eval(".art-canvas[data-v='2'] .svg-paper", (p) => p.shadowRoot!.querySelector("#burger rect")?.getAttribute("fill"))).toBe("#ff8c00");
}, 30_000);

test("voice mode: talking over a reply silences it, stops Chrysalis's request, and the chat keeps only what was heard", async () => {
  const stopped = chrysalisStopped, { page, session } = await voicePage("story");
  await page.waitForFunction(() => document.querySelector('#thread > article[data-i="0"] .bubble')?.textContent === "slow story", { timeout: 10_000 });
  for (const t = Date.now(); !session().some((e) => e.ev === "interrupt") && Date.now() - t < 10_000; ) await Bun.sleep(10);
  const talked = Date.now();
  for (; chrysalisStopped === stopped && Date.now() - talked < 3000; ) await Bun.sleep(10);
  expect(chrysalisStopped).toBe(stopped + 1);
  expect(Date.now() - talked).toBeLessThan(500);
  await page.waitForFunction(() => /interrupted/.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  const kept = await replyText(page, 1), chunks = session().filter((e) => e.ev === "chunk"), full = Array.from({ length: 40 }, (_, k) => `w${k}`).join(" ");
  expect(kept.length).toBeGreaterThan(0);
  expect(kept.length).toBeLessThan(full.length);
  expect(full.startsWith(kept)).toBe(true);
  const before = chunks.map((c) => c.text).join("").trimEnd(), ends = chunks.map((c) => chunks.filter((x) => x.to <= c.to).map((x) => x.text).join("").trimEnd());
  expect([full[kept.length] === " ", kept.length <= before.length, ends.includes(kept)]).toEqual([true, true, false]);
  await page.waitForFunction(() => document.querySelector('#thread > article[data-i="2"] .bubble')?.textContent === "and then", { timeout: 10_000 });
  await page.waitForFunction(() => /voice/.test(document.querySelector('#thread > article[data-i="3"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  expect(sent(lastStream).filter((m: any) => m.role === "assistant").map((m: any) => m.content)).toEqual([`${kept} — [cut off: the user talked over this reply]`]);
  expect(checks.findLast((b) => b.messages.at(-1).content.endsWith("Transcript: and then"))?.messages.at(-1).content)
    .toBe(`The user said before: slow story\nThe assistant said: ${kept}\nTranscript: and then`);
  await page.close();
}, 30_000);

test("voice mode: a reply's words not heard yet are grey, and turn their own color as they are spoken", async () => {
  const { page } = await voicePage("long");
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Speaking", { timeout: 10_000 });
  const heard = () => page.evaluate(() => {
    const all = document.querySelector('#thread > article[data-i="1"] .markdown')!.textContent!.trim();
    const range = (CSS as any).highlights.get("voice-unspoken")?.values().next().value as Range | undefined;
    const grey = range?.toString().trim() ?? null, mid = range && getComputedStyle(range.startContainer.parentElement!).color;
    return { all, grey, said: grey === null ? null : all.slice(0, all.length - grey.length).trim(), mid };
  });
  await Bun.sleep(500);
  const a = await heard();
  await Bun.sleep(700);
  const b = await heard();
  expect(a.grey && b.grey && a.said && b.said).toBeTruthy();
  expect([b.all.startsWith(a.said!), b.all.startsWith(b.said!), b.said!.length > a.said!.length]).toEqual([true, true, true]);
  expect(/^(w\d+ )*w\d+$/.test(b.said!)).toBe(true);
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 15_000 });
  expect(await page.evaluate(() => (CSS as any).highlights.has("voice-unspoken"))).toBe(false);
  await page.close();
}, 30_000);

test("voice mode: a reply talked over before it made a sound is taken out, and the next words join its question", async () => {
  const { page } = await voicePage("merge");
  await page.waitForFunction(() => document.querySelector('#thread > article[data-i="0"] .bubble')?.textContent === "slow first second part", { timeout: 15_000 });
  await page.waitForFunction(() => /^w0 w1/.test(document.querySelector('#thread > article[data-i="1"] .markdown')?.textContent ?? ""), { timeout: 10_000 });
  expect(await page.$$eval("#thread > article.msg:not(.ghost)", (els) => els.length)).toBe(2);
  expect(sent(lastStream).filter((m: any) => m.role !== "system")).toEqual([{ role: "user", content: `[${await noteShown(page)}]\n\nslow first second part` }]);
  await page.close();
}, 30_000);

test("voice mode's note goes before the first question said and again when it changes, the typing note after it, and the system prompt never changes", async () => {
  const n0 = streams.length, CHANGED = "Speak like a pirate.";
  const { page, session } = await voicePage("notes", { system: "Be brief.", first: (p) => send(p, "typed first") });
  const note = await noteShown(page);
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="5"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  await openSettings(page);
  await page.$eval("#voice-prompt", (el, v) => { (el as HTMLTextAreaElement).value = v; el.dispatchEvent(new Event("input")); }, CHANGED);
  await page.$eval("#knob-voice_speed", (el) => { (el as HTMLInputElement).value = "1.2"; el.dispatchEvent(new Event("change")); });
  await closeSettings(page);
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="7"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  await page.click("#send");
  await send(page, "typed again");
  const asked = streams.slice(n0).filter((b) => b.max_tokens !== 1);
  expect(asked.map((b) => sent(b).at(-1).content)).toEqual(["typed first", `[${note}]\n\nfirst question`, "second question",
    `[${CHANGED}]\n\nthird question`, "[The user is typing again and reads your replies.]\n\ntyped again"]);
  expect(asked.every((b) => JSON.stringify(b.messages[0]) === JSON.stringify({ role: "system", content: "Be brief." }))).toBe(true);
  expect(session().some((e) => e.type === "config")).toBe(true);
  expect(await page.$$eval(".msg-note summary", (els) => els.map((e) => e.textContent))).toEqual(["Voice mode note", "Voice mode note", "Note to the model"]);
  await page.close();
}, 45_000);

test("What the model is sent shows the next request as Chrysalis gets it: the tools as JSON, the one system prompt, every message; an edited prompt goes as written, and Chrysalis's card's scenario after it", async () => {
  const page = await open();
  await page.setViewport({ width: 1600, height: 900 });
  await page.click("#tools-btn");
  await page.waitForSelector(sw("mcp:fake"));
  await page.click(sw("mcp:fake"));
  await page.waitForFunction(() => document.querySelector('#tools-pop [data-src="mcp:fake"]')?.closest(".tp-src")?.querySelector(".tp-sub")?.textContent === "5 of 5 tools on",
    { timeout: 20_000 });
  await page.click(sw("fetch"));
  await page.keyboard.press("Escape");
  await send(page, "first");
  await openSettings(page, "chat");
  await page.click("#sent-open");
  await page.waitForFunction(() => (document.querySelector("#sent-view") as HTMLDialogElement).open && !!document.querySelector("#sent-text")?.textContent);
  const shown = await page.$eval("#sent-text", (e) => e.textContent!);
  await page.keyboard.press("Escape");
  await closeSettings(page);
  await send(page, "second");
  const next = lastStream;
  expect(shown).toContain(JSON.stringify(next.tools, null, 2));
  let at = 0;
  for (const m of next.messages.slice(0, -1)) {
    const k = shown.indexOf(`── ${m.role}\n${m.content}`, at);
    expect([m.role, k]).not.toEqual([m.role, -1]);
    at = k;
  }
  expect(next.messages[0].content).toContain("The tools offered here are the only ones");
  await page.$eval("#system-input", (el) => { (el as HTMLTextAreaElement).value = "Be terse, {{MODEL}}."; el.dispatchEvent(new Event("input")); });
  await send(page, "third");
  expect(lastStream.messages[0]).toEqual({ role: "system", content: `Be terse, Qwen3.8-27B.\n\n${ARTIFACTS_PROMPT}` });
  await page.click("#thread > article.msg.assistant .speaker");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-field textarea");
  const typeIn = async (label: string, text: string) => {
    const at = await page.$$eval("#cast-sheet .card-field", (f, l) => f.findIndex((x) => x.querySelector(".card-label")!.textContent === l), label);
    await page.type(`#cast-sheet .card-field:nth-of-type(${at + 1}) textarea`, text);
  };
  await typeIn("Description", "{{char}} tends a greenhouse.");
  await typeIn("Scenario", "A rainy afternoon.");
  await page.keyboard.press("Escape");
  await send(page, "fourth");
  expect(lastStream.messages[0]).toEqual({ role: "system", content: `Be terse, Qwen3.8-27B.\n\nChrysalis tends a greenhouse.\n\nScenario: A rainy afternoon.\n\n${ARTIFACTS_PROMPT}` });
  await page.click("#you");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-text");
  await page.type("#cast-sheet .card-text", "{{user}} keeps bees.");
  await page.keyboard.press("Escape");
  await send(page, "fifth");
  expect(lastStream.messages[0]).toEqual({ role: "system", content: `Be terse, Qwen3.8-27B.\n\nChrysalis tends a greenhouse.\n\nScenario: A rainy afternoon.\n\nUser keeps bees.\n\n${ARTIFACTS_PROMPT}` });
  await page.click("#thread > article.msg.assistant .speaker");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-field textarea");
  await typeIn("Greeting", "Hello, {{user}}.");
  await page.keyboard.press("Escape");
  await page.click("#new-chat");
  expect(await articles(page)).toEqual(["Hello, User."]);
  await send(page, "sixth");
  expect(lastStream.messages.slice(1)).toEqual([{ role: "user", content: "[The scene begins.]" }, { role: "assistant", content: "Hello, User." }, { role: "user", content: "sixth" }]);
  await page.close();
}, 60_000);

test("the theme is the system's until Settings chooses one, which overrides it and is kept across a reload; System follows the system's scheme as it changes", async () => {
  const page = await open();
  const bg = async (want: "dark" | "light") => {
    const lum = () => page.evaluate(() => {
      const [r, g, b] = getComputedStyle(document.body).backgroundColor.match(/[\d.]+/g)!.map(Number);
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    });
    let v = await lum();
    for (const end = Date.now() + 2000; (want === "dark" ? v >= 60 : v <= 200) && Date.now() < end; await Bun.sleep(25)) v = await lum();
    return v;
  };
  const scheme = (value: "dark" | "light") => page.emulateMediaFeatures([{ name: "prefers-color-scheme", value }]);
  await scheme("dark");
  expect(await bg("dark")).toBeLessThan(60);
  await scheme("light");
  expect(await bg("light")).toBeGreaterThan(200);
  await openSettings(page);
  await page.click('#theme-pick [data-theme="dark"]');
  expect(await bg("dark")).toBeLessThan(60);
  await page.reload({ waitUntil: "networkidle0" });
  expect(await bg("dark")).toBeLessThan(60);
  expect(await page.$eval('#theme-pick [aria-checked="true"]', (b) => (b as HTMLElement).dataset.theme)).toBe("dark");
  await openSettings(page);
  await page.click('#theme-pick [data-theme="system"]');
  expect(await bg("light")).toBeGreaterThan(200);
  await scheme("dark");
  expect(await bg("dark")).toBeLessThan(60);
  await page.close();
}, 30_000);

async function openSettings(page: Page, sec = "chat") {
  if (!(await page.$eval("#settings", (d) => (d as HTMLDialogElement).open))) await page.click("#settings-open");
  await page.click(`.set-nav [data-sec="${sec}"]`);
  await page.waitForFunction((sec) => { const r = document.querySelector(`.set-sec[data-sec="${sec}"]`)!.getBoundingClientRect(); return r.width > 0 && r.right <= innerWidth; }, {}, sec);
}
async function closeSettings(page: Page) {
  await page.click("#settings-close");
  await page.waitForFunction(() => !(document.getElementById("settings") as HTMLDialogElement).open);
}
const stored = (page: Page) => page.evaluate(() => JSON.parse(sessionStorage.getItem("settings2") ?? "{}").voice);
const settled = async (ok: () => boolean, ms = 5000) => { for (const t = Date.now(); !ok() && Date.now() - t < ms; ) await Bun.sleep(25); return ok(); };

test("voice picker: a voice model not downloaded yet is downloaded from the picker, and then offers its voices", async () => {
  const { page } = await voicePage("pick");
  await page.click("#voice-pick");
  await page.waitForSelector('#voice-menu:popover-open [data-get="big-tts"]');
  expect(await page.$$eval('#voice-menu [data-voice^="big-tts/"]', (els) => els.length)).toBe(0);
  await page.click('#voice-menu [data-get="big-tts"]');
  await page.waitForSelector('#voice-menu [data-voice="big-tts/zed"]', { timeout: 10_000 });
  expect(await page.$$eval("#voice-menu [data-get]", (els) => els.length)).toBe(0);
});

test("voice picker: the voice models' voices by language, each with its gender; a pick switches model and voice for the next sentence, and Settings shows the same choice", async () => {
  const { page, session } = await voicePage("pick");
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  expect(new Set(session().filter((e) => e.ev === "chunk").map((e) => `${e.tts}/${e.voice}`))).toEqual(new Set(["fake-tts/pick"]));
  expect(await page.$eval("#voice-pick .voice-pick-name", (el) => el.textContent)).toBe("Pick");
  await page.click("#voice-pick");
  await page.waitForSelector("#voice-menu:popover-open");
  const rows = await page.$$eval("#voice-menu > *", (els) => els.map((e) => [e.className, e.classList.contains("voice-chips")
    ? [...e.children].map((c) => (c as HTMLElement).dataset.voice?.split("/")[1] ?? (c as HTMLElement).dataset.get ?? "") : e.textContent]));
  const fake = ["plain", "pie", "story", "long", "merge", "join", "ahead", "onward", "checking", "silent", "pick", "notes", "point"];
  expect(rows.slice(1)).toEqual([["voice-chips voice-top", ["", "big-tts"]], ["voice-model", "Fake TTS"], ["voice-group", "American English"], ["voice-chips", fake],
    ["voice-group", "British English"], ["voice-chips", ["echo"]], ["voice-model", "Other TTS"], ["voice-group", "Spanish"], ["voice-chips", ["ana"]],
    ["voice-group", "American English"], ["voice-chips", ["bob"]]]);
  expect(await page.$$eval('#voice-menu .voice-chip[data-voice^="other-tts/"]', (els) => els.map((e) => (e as HTMLElement).title))).toEqual(["Spanish · Female", "American English · Male"]);
  expect(await page.$$eval('#voice-menu .voice-chip[aria-checked="true"]', (els) => els.map((e) => (e as HTMLElement).dataset.voice))).toEqual(["fake-tts/pick"]);
  await page.click('#voice-menu .voice-chip[data-voice="other-tts/bob"] > span');
  const configs = () => session().filter((e) => e.type === "config").map(({ stack, voice, speed }) => ({ stack, voice, speed }));
  expect(await settled(() => configs().length === 1)).toBe(true);
  expect(configs()).toEqual([{ stack: { tts: "other-tts" }, voice: "bob", speed: undefined }]);
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="3"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  const log = session(), after = log.slice(log.findIndex((e) => e.type === "config")).filter((e) => e.ev === "chunk");
  expect(after.length).toBeGreaterThan(0);
  expect(new Set(after.map((e) => `${e.tts}/${e.voice}`))).toEqual(new Set(["other-tts/bob"]));
  expect(await page.$eval("#voice-pick .voice-pick-name", (el) => el.textContent)).toBe("Bob");
  expect(await stored(page)).toEqual({ tts: "other-tts", name: "bob" });
  const chosen = async (id: string) => {
    await page.click("#voice-settings");
    await page.waitForFunction((id) => document.querySelector('#voice-menu:popover-open .voice-chip[aria-checked="true"]')?.getAttribute("data-voice") === id, {}, id);
  };
  await openSettings(page);
  await chosen("other-tts/bob");
  await page.click('#voice-menu .voice-chip[data-voice="other-tts/ana"] > span');
  await closeSettings(page);
  await page.waitForFunction(() => document.querySelector("#voice-pick .voice-pick-name")?.textContent === "Ana");
  expect(await settled(() => configs().length === 2)).toBe(true);
  expect(configs()[1]).toEqual({ stack: undefined, voice: "ana", speed: undefined });
  await page.click("#voice-pick");
  await page.click('#voice-menu .voice-chip[data-voice="fake-tts/echo"] > span');
  await page.waitForSelector("#toast:not([hidden]) .btn");
  await openSettings(page);
  await chosen("fake-tts/echo");
  await page.keyboard.press("Escape");
  await closeSettings(page);
  await page.click("#toast .btn");
  await page.waitForFunction(() => document.querySelector("#voice-pick .voice-pick-name")?.textContent === "Ana");
  await openSettings(page);
  await chosen("other-tts/ana");
  await page.keyboard.press("Escape");
  await closeSettings(page);
  expect(await settled(() => configs().length === 4)).toBe(true);
  expect(configs().slice(2)).toEqual([{ stack: { tts: "fake-tts" }, voice: "echo", speed: undefined }, { stack: { tts: "other-tts" }, voice: "ana", speed: undefined }]);
  expect(await stored(page)).toEqual({ tts: "other-tts", name: "ana" });
  await page.close();
}, 40_000);

test("Settings lists the voices in a menu from its voice button, over the dialog and moving nothing in it; a chip's button plays its sample and shows it playing until it has been heard, a chip picks the voice, and the menu goes with the dialog", async () => {
  const page = await open();
  await openSettings(page);
  const layout = () => page.$$eval("#settings, #settings *:not(#voice-menu, #voice-menu *)", (els) => els.flatMap((e) => e instanceof HTMLElement ? [[e.offsetLeft, e.offsetTop, e.offsetWidth, e.offsetHeight].join()] : []).join(";"));
  const still = await layout();
  await page.click("#voice-settings");
  await page.waitForSelector('#voice-menu:popover-open .voice-chip[data-voice="other-tts/bob"]', { timeout: 10_000 });
  expect(await layout()).toBe(still);
  const before = voiceLog().length;
  await page.click('#voice-menu .voice-chip[data-voice="other-tts/bob"] .voice-play');
  await page.waitForSelector('#voice-menu .voice-chip[data-voice="other-tts/bob"] .voice-play.playing');
  await page.waitForFunction(() => !document.querySelector("#voice-menu .voice-play.playing"), { timeout: 10_000 });
  const heard = voiceLog().slice(before);
  expect([heard.filter((e) => e.type === "preview").map((e) => [e.tts, e.voice]), heard.filter((e) => e.ev === "chunk").map((e) => e.id)])
    .toEqual([[["other-tts", "bob"]], ["preview"]]);
  expect(await page.$("#voice-menu:popover-open")).not.toBeNull();
  await page.click('#voice-menu .voice-chip[data-voice="other-tts/bob"] > span');
  await page.waitForFunction(() => !document.querySelector("#voice-menu:popover-open"));
  expect([await page.$eval("#voice-settings .voice-pick-name", (b) => b.textContent), await stored(page)]).toEqual(["Bob", { tts: "other-tts", name: "bob" }]);
  await page.click("#voice-settings");
  await page.waitForSelector("#voice-menu:popover-open");
  await page.keyboard.down("Control");
  await page.keyboard.press(",");
  await page.keyboard.up("Control");
  await page.waitForFunction(() => !(document.getElementById("settings") as HTMLDialogElement).open);
  await page.waitForFunction(() => !document.querySelector("#voice-menu:popover-open"), { timeout: 2000 });
  await page.close();
}, 30_000);

test("voice picker: a voice's play button plays its sample, apart from any reply or turn, and the voice's button closes the menu again", async () => {
  const { page, session } = await voicePage("silent");
  await page.click("#voice-pick");
  await page.waitForSelector('#voice-menu:popover-open .voice-chip[data-voice="other-tts/ana"]');
  await page.click('#voice-menu .voice-chip[data-voice="other-tts/ana"] .voice-play');
  expect(await settled(() => session().filter((e) => e.type === "playback").length === 2)).toBe(true);
  const log = session();
  expect(log.filter((e) => e.type === "preview").map((e) => e.voice)).toEqual(["ana"]);
  expect(log.filter((e) => e.ev === "chunk").map((e) => [e.id, e.voice, e.text])).toEqual([["preview", "ana", "This is Ana."]]);
  expect(log.filter((e) => e.type === "playback").map((e) => e.playing)).toEqual([true, false]);
  expect(log.some((e) => e.type === "say" || e.type === "config")).toBe(false);
  expect(await page.$$eval("#thread > article", (els) => els.length)).toBe(0);
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 3000 });
  expect(await page.$eval("#voice-pick .voice-pick-name", (el) => el.textContent)).toBe("Silent");
  expect(await page.$("#voice-menu:popover-open")).not.toBeNull();
  await page.click("#voice-pick");
  await Bun.sleep(300);
  expect(await page.$("#voice-menu:popover-open")).toBeNull();
  await page.close();
}, 30_000);

test("voice mode: mute stops the microphone's audio; ending it closes the socket, and the service stops once idle", async () => {
  for (const p of voicePages.splice(0)) if (!p.isClosed()) await p.close();
  const { page, session } = await voicePage("silent");
  const frames = () => session().filter((e) => e.ev === "frames").at(-1)?.n ?? 0;
  for (let t = Date.now(); frames() < 20 && Date.now() - t < 5000; ) await Bun.sleep(50);
  expect(frames()).toBeGreaterThanOrEqual(20);
  await page.click("#voice-mute");
  await Bun.sleep(300);
  const muted = frames();
  await Bun.sleep(700);
  expect(frames()).toBe(muted);
  expect(session().some((e) => e.type === "mute" && e.on === true)).toBe(true);
  await page.click("#voice-mute");
  for (let t = Date.now(); frames() < muted + 20 && Date.now() - t < 5000; ) await Bun.sleep(50);
  expect(frames()).toBeGreaterThanOrEqual(muted + 20);
  await page.click("#send");
  expect(await page.$eval("#voice-pill", (el) => (el as HTMLElement).hidden)).toBe(true);
  expect(await page.$eval(".composer-box", (el) => getComputedStyle(el).display)).not.toBe("none");
  for (let t = Date.now(); !session().some((e) => e.ev === "close") && Date.now() - t < 3000; ) await Bun.sleep(50);
  expect(session().some((e) => e.ev === "close")).toBe(true);
  const starts = voiceLog().filter((e) => e.ev === "start").length;
  for (let t = Date.now(); voiceLog().filter((e) => e.ev === "exit").length < starts && Date.now() - t < 6000; ) await Bun.sleep(100);
  expect(voiceLog().filter((e) => e.ev === "exit").length).toBe(starts);
  await page.close();
}, 30_000);

async function pastWarning(page: Page, firefox: boolean) {
  if (firefox) {
    await page.waitForSelector("net-error-card >>> #advanced-button", { timeout: 10_000 });
    await Bun.sleep(1000);
    await page.click("net-error-card >>> #advanced-button");
    await Bun.sleep(1500);
    await page.click("net-error-card >>> #exception-button");
  } else {
    await page.waitForSelector("#details-button", { timeout: 10_000 });
    await page.click("#details-button");
    await page.click("#proceed-link");
  }
}
function trusting(home: string, tlsDir: string) {
  const profile = `${home}/.mozilla/firefox/test.default`, chromium = `${home}/.pki/nssdb`;
  for (const db of [profile, chromium]) {
    mkdirSync(db, { recursive: true });
    expect(Bun.spawnSync(["certutil", "-N", "--empty-password", "-d", `sql:${db}`]).exitCode).toBe(0);
    expect(Bun.spawnSync(["certutil", "-A", "-d", `sql:${db}`, "-n", "Chrysalis", "-t", "C,,", "-i", `${tlsDir}/ca.pem`]).exitCode).toBe(0);
  }
  return profile;
}

test("past the chat's warning on the network, a device gets the connect page and none of the chat: it offers each kind of device the way its system trusts the authority, Chromium downloads it, and nothing reaches the server; trusting the authority, Chromium and Firefox each show their own code on the same page, which this computer's Connect a device allows, and the chat opens there", async () => {
  const tlsDir = mkdtempSync(`${RUN}/tls-`);
  const start = async (ports = { PORT: "0", CHRYSALIS_CHECK_PORT: "0" }) => {
    const proc = Bun.spawn([process.execPath, "server.ts"], {
      cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
      env: { ...process.env, ...ports, CHRYSALIS_URL: `http://127.0.0.1:${fake.port}`, CHRYSALIS_HOST: "0.0.0.0", NODE_ENV: "production", CHRYSALIS_LINK_PORT: "0",
        CHRYSALIS_SELF_AS_NETWORK: "1",
        CHRYSALIS_MCP_CONFIG: `${dir}/mcp.json`, CHRYSALIS_TLS_DIR: tlsDir },
    });
    let out = "";
    for (const reader = proc.stdout.getReader(), dec = new TextDecoder(); !/chrysalis check: \S+/.test(out); ) {
      const { value, done } = await reader.read();
      if (done) throw new Error(`server.ts exited: ${out}`);
      out += dec.decode(value);
    }
    return { proc, out };
  };
  let { proc: lan, out } = await start();
  const others: Browser[] = [];
  try {
    const here = out.match(/chrysalis web: (\S+)/)![1], chat = out.match(/chrysalis web: (https:\/\/(?!127\.)\S+)/)![1];
    const launch = (firefox: boolean, home = "", args: string[] = []) => puppeteer.launch(firefox
      ? { browser: "firefox", executablePath: process.env.FIREFOX ?? "/usr/bin/firefox", headless: true, userDataDir: home || mkdtempSync(`${RUN}/fx-`) }
      : { executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: ["--no-sandbox", ...args], ...(home ? { env: { ...process.env, HOME: home } } : {}) });
    const plain = await launch(false, mkdtempSync(`${RUN}/untrusting-`));
    others.push(plain);
    const UA = {
      iphone: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
      android: "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36",
      firefox: "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
      mac: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
      windows: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
    };
    const offered: Record<string, string | null> = {};
    const setUp = (page: Page) => page.waitForFunction(() => document.getElementById("setup")?.hidden === false, { timeout: 15_000 });
    for (const [kind, ua] of Object.entries(UA)) {
      const page = await (await plain.createBrowserContext()).newPage();
      await page.setUserAgent(ua);
      await page.goto(chat).catch(() => {});
      await pastWarning(page, false);
      await setUp(page);
      offered[kind] = await page.$eval(".go", (a) => a.getAttribute("href"));
      while (!(await page.$eval("#on", (e) => (e as HTMLElement).hidden))) await page.click("#on");
      expect(await page.$eval("#count", (e) => e.textContent)).toMatch(/ · Trust$/);
      if (kind === "android") {
        await Bun.sleep(2000);
        expect(page.url()).toBe(chat);
        const stage = await page.$eval("#stage", (e) => { const b = e.getBoundingClientRect(); return { x: b.x, y: b.y, width: b.width, height: b.height }; });
        expect([(await look(page, stage)).pink > 1500, await page.$eval("#stage", (e) => e.childElementCount)]).toEqual([true, 0]);
      }
      await page.close();
    }
    expect(offered).toEqual({ iphone: "/chrysalis.mobileconfig", android: "/chrysalis-ca.crt", firefox: "/chrysalis-ca-firefox.crt",
      mac: "/chrysalis.mobileconfig", windows: "/chrysalis-ca.crt" });
    const waiting = async () => (await (await fetch(`${here}api/pair`)).json()).waiting;
    {
      const page = await (await plain.createBrowserContext()).newPage();
      await page.goto(chat).catch(() => {});
      await pastWarning(page, false);
      await setUp(page);
      await page.setRequestInterception(true);
      page.on("request", (r) => { if (!r.url().includes("/check.gif")) void r.continue(); });
      await page.reload({ waitUntil: "domcontentloaded" });
      await page.waitForFunction(() => /Cannot reach/.test(document.getElementById("status")!.textContent!), { timeout: 10_000 });
      expect([await page.$eval("#status", (e) => (e as HTMLElement).hidden), await page.$eval("#sub", (e) => e.textContent)])
        .toEqual([false, expect.stringContaining("chrysalis.sh --host 0.0.0.0")]);
      await page.close();
    }

    for (const firefox of [false, true]) {
      const home = firefox ? "" : mkdtempSync(`${RUN}/untrusting-`);
      const b = await launch(firefox, home, ["--disable-3d-apis"]);
      others.push(b);
      const page = await b.newPage();
      await page.goto(chat).catch(() => {});
      await pastWarning(page, firefox);
      await setUp(page);
      expect([page.url(), await page.$eval(".go", (a) => a.getAttribute("href")), await page.$("#input")])
        .toEqual([chat, firefox ? "/chrysalis-ca-firefox.crt" : "/chrysalis-ca.crt", null]);
      if (!firefox) {
        await Bun.sleep(1500);
        const stage = await page.$eval("#stage", (e) => { const b = e.getBoundingClientRect(); return { x: b.x, y: b.y, width: b.width, height: b.height }; });
        expect([(await look(page, stage)).pink > 1500, await page.$eval("#stage", (e) => e.childElementCount)]).toEqual([true, 0]);
        await (await b.target().createCDPSession()).send("Browser.setDownloadBehavior", { behavior: "default" });
        await page.click(".go");
        const kept = `${home}/Downloads/chrysalis-ca.crt`;
        for (let t = Date.now(); !existsSync(kept) && Date.now() - t < 5000; ) await Bun.sleep(100);
        expect(existsSync(kept) && readFileSync(kept, "utf8")).toBe(readFileSync(`${tlsDir}/ca.pem`, "utf8"));
        await page.waitForFunction(() => document.getElementById("go")!.hidden, { timeout: 3000 });
        expect(await page.$eval("#now", (e) => [e.querySelector(".copyable code")?.textContent, !!e.querySelector(".copy")]))
          .toEqual(["chrome://certificate-manager/localcerts/usercerts", true]);
      }
      expect(await waiting()).toEqual([]);
    }

    const local = await (await browser.createBrowserContext()).newPage();
    await local.goto(here, { waitUntil: "load" });
    await local.waitForSelector("#input", { timeout: 15_000 });
    await local.waitForSelector("#connect:not([hidden])");
    await local.click("#connect");
    await local.waitForSelector("#connect-view .dev-qr");
    expect(await local.$eval("#connect-view .dev-qr", (img) => (img as HTMLImageElement).naturalWidth > 0)).toBe(true);
    let chrome: Page | undefined;
    for (const [firefox, name] of [[false, "Linux · Chrome"], [true, "Linux · Firefox"]] as const) {
      const home = mkdtempSync(`${RUN}/trusts-`), profile = trusting(home, tlsDir);
      const b = await launch(firefox, firefox ? profile : home);
      others.push(b);
      const page = await b.newPage();
      await page.goto(chat, { waitUntil: "load" });
      await page.waitForFunction(() => !(document.getElementById("code") as HTMLElement).hidden, { timeout: 10_000 });
      expect(await page.$eval("#setup", (e) => (e as HTMLElement).hidden)).toBe(true);
      const code = await page.$eval("#code", (e) => e.textContent!);
      expect(code).toMatch(/^\d{3} \d{3}$/);
      await local.waitForFunction((c) => [...document.querySelectorAll("#connect-view .dev-ask-code")].some((e) => e.textContent === c), { timeout: 5000 }, code);
      expect(await local.$$eval("#connect-view .dev-ask .dev-name", (n) => n.map((e) => e.textContent))).toEqual([name]);
      await local.evaluate((c) => [...document.querySelectorAll("#connect-view .dev-ask")]
        .find((e) => e.querySelector(".dev-ask-code")!.textContent === c)!.querySelector<HTMLButtonElement>(".btn.primary")!.click(), code);
      await page.waitForFunction((c) => location.href === c, { timeout: 15_000 }, chat);
      await page.waitForSelector("#input");
      await Bun.sleep(1000);
      expect([page.url(), await page.$eval("#connect", (e) => e.checkVisibility())]).toEqual([chat, false]);
      await local.waitForFunction((n) => (document.querySelector("#connect-view .dev-joined")?.textContent ?? "").includes(`${n} is connected`), { timeout: 5000 }, name);
      if (!firefox) chrome = page;
    }
    expect(await local.$$eval("#connect-view .dev-item .dev-name", (n) => n.map((e) => e.textContent)))
      .toEqual(["Linux · Chrome", "Linux · Firefox"]);

    await chrome!.evaluate(() => { (window as any).stayed = true; });
    const port = (re: RegExp) => new URL(out.match(re)![1]).port;
    const ports = { PORT: port(/chrysalis web: (\S+)/), CHRYSALIS_CHECK_PORT: port(/chrysalis check: (\S+)/) };
    lan.kill();
    await lan.exited;
    ({ proc: lan, out } = await start(ports));
    await Bun.sleep(9000);
    expect(await chrome!.evaluate(async () => [(window as any).stayed, location.pathname, (await fetch("/api/status")).status])).toEqual([true, "/", 200]);
  } finally {
    for (const b of others) await b.close().catch(() => {});
    lan.kill();
  }
}, 120_000);

const spokenOf = (log: any[]) => {
  const said = new Map<string, string>();
  for (const e of log.filter((x) => x.type === "say")) said.set(e.id, (said.get(e.id) ?? "") + e.text);
  return { said, cancelled: new Set(log.filter((x) => x.type === "cancel" && x.id !== undefined).map((x) => x.id)) };
};
const turnAt = (log: any[]) => log.find((e) => e.ev === "sent" && e.type === "turn")?.t;

test("voice mode: a question shows grey as it was heard until its check is in, then white with the misheard words fixed (the heard ones on hover); the reply begun on the words as heard is dropped unheard, and the fixed question is answered", async () => {
  const n0 = streams.length, c0 = checks.length, { page, session } = await voicePage("pie");
  const q = '#thread > article[data-i="0"]';
  await page.waitForFunction((q) => document.querySelector(`${q} .bubble`)?.textContent === "lets do a snake game in pie game", { timeout: 10_000 }, q);
  expect(await page.$eval(q, (e) => [e.classList.contains("unsettled"), getComputedStyle(e.querySelector(".bubble")!).color]))
    .toEqual([true, await page.evaluate(() => { const d = document.createElement("div"); d.style.color = "var(--muted)"; document.body.append(d);
      const c = getComputedStyle(d).color; d.remove(); return c; })]);
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  expect(await page.$eval(`${q} .bubble`, (e) => e.textContent)).toBe("lets do a snake game in Pygame");
  expect(await page.$eval(`${q} .bubble .fixed`, (e) => [e.textContent, (e as HTMLElement).title])).toEqual(["Pygame", 'Speech recognition heard "pie game"']);
  expect(await page.$eval(q, (e) => e.classList.contains("unsettled"))).toBe(false);
  expect(await page.$$eval("#thread > article.msg", (els) => els.length)).toBe(2);
  expect(sent(lastStream).at(-1).content).toMatch(/\n\nlets do a snake game in Pygame$/);
  expect(JSON.stringify(lastStream)).not.toMatch(/=>|pie game/);
  expect(await replyText(page, 1)).toBe("You said: lets do a snake game in Pygame");
  const replies = streams.slice(n0).filter((b) => b.max_tokens !== 1).map((b) => sent(b).at(-1).content.split("\n\n").at(-1));
  expect(replies).toEqual(["lets do a snake game in pie game", "lets do a snake game in Pygame"]);
  expect(checks.slice(c0).map((b) => b.messages.at(-1).content)).toEqual(["Transcript: lets do a snake game in pie game"]);
  const { said, cancelled } = spokenOf(session());
  expect([...said].filter(([id]) => !cancelled.has(id)).map(([, t]) => t)).toEqual(["You said: lets do a snake game in Pygame"]);
  expect([...said].filter(([id]) => cancelled.has(id)).map(([, t]) => t)).toEqual(["You said: lets do a snake game in pie game"]);
  const log = session();
  expect(log.findIndex((e) => e.type === "playback" && e.playing)).toBeGreaterThan(log.findIndex((e) => e.type === "cancel"));
  await page.close();
}, 30_000);

test("voice mode: words said while a question is still being checked join it, and the two are answered once", async () => {
  const n0 = streams.length, { page, session } = await voicePage("join");
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 20_000 });
  expect(await page.$eval('#thread > article[data-i="0"] .bubble', (e) => e.textContent)).toBe("wait lets do a snake game in Pygame");
  expect(await page.$$eval("#thread > article.msg:not(.ghost)", (els) => els.length)).toBe(2);
  expect(await replyText(page, 1)).toBe("You said: wait lets do a snake game in Pygame");
  const replies = streams.slice(n0).filter((b) => b.max_tokens !== 1);
  expect(sent(replies.at(-1)).at(-1).content.split("\n\n").at(-1)).toBe("wait lets do a snake game in Pygame");
  const { said, cancelled } = spokenOf(session());
  expect([...said].filter(([id]) => !cancelled.has(id)).map(([, t]) => t)).toEqual(["You said: wait lets do a snake game in Pygame"]);
  await page.close();
}, 30_000);

test("voice mode: a reply begins at a pause the turn waits out, unseen and unheard until the turn ends with its words, and then is heard at once", async () => {
  const n0 = streams.length, { page, session } = await voicePage("ahead");
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  const log = session(), turned = turnAt(log), mine = streams.slice(n0).map((b, i) => [b, streamAt[n0 + i]]).filter(([b]) => b.max_tokens !== 1);
  expect(mine.length).toBe(1);
  expect(mine[0][1]).toBeLessThan(turned);
  const played = log.findIndex((e) => e.type === "playback" && e.playing), turnIx = log.findIndex((e) => e.ev === "sent" && e.type === "turn");
  expect(played).toBeGreaterThan(turnIx);
  expect(log.slice(0, turnIx).some((e) => e.ev === "chunk")).toBe(true);
  expect(await replyText(page, 1)).toBe("Paris.");
  expect(await page.$$eval("#thread > article.msg", (els) => els.length)).toBe(2);
  await page.close();
}, 30_000);

test("voice mode: a reply begun at a pause is dropped unheard when the user goes on, and the whole question is answered", async () => {
  const n0 = streams.length, stopped = chrysalisStopped, { page, session } = await voicePage("onward");
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  const replies = streams.slice(n0).filter((b) => b.max_tokens !== 1).map((b) => sent(b).at(-1).content.split("\n\n").at(-1));
  expect(replies).toEqual(["slow what is the", "slow what is the capital of France"]);
  expect(chrysalisStopped).toBe(stopped + 1);
  expect(await page.$eval('#thread > article[data-i="0"] .bubble', (e) => e.textContent)).toBe("slow what is the capital of France");
  expect(await page.$$eval("#thread > article.msg", (els) => els.length)).toBe(2);
  const log = session(), { said, cancelled } = spokenOf(log);
  expect([...said.keys()].map((id) => cancelled.has(id))).toEqual([true, false]);
  expect(log.findIndex((e) => e.type === "cancel")).toBeLessThan(log.findIndex((e) => e.ev === "sent" && e.type === "turn"));
  await page.close();
}, 30_000);

test("voice mode: the words are checked while they are said: in a pause they turn their own color with the misheard ones fixed", async () => {
  const { page } = await voicePage("checking");
  await page.waitForFunction(() => document.querySelector(".msg.ghost .bubble .checked .fixed")?.textContent === "Pygame", { timeout: 10_000 });
  const ghost = await page.$eval(".msg.ghost .bubble", (b) => [b.textContent, getComputedStyle(b.querySelector(".checked")!).color, getComputedStyle(b).color]);
  const color = (v: string) => page.evaluate((v) => { const d = document.createElement("div"); d.style.color = `var(${v})`; document.body.append(d);
    const c = getComputedStyle(d).color; d.remove(); return c; }, v);
  expect(ghost).toEqual(["lets do a snake game in Pygame", await color("--text"), await color("--muted")]);
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  expect(await replyText(page, 1)).toBe("You said: lets do a snake game in Pygame");
  await page.close();
}, 30_000);

test("voice mode: it says it is listening once the microphone gives audio, not when the service is ready", async () => {
  const page = await (await chatContext()).newPage();
  await page.evaluateOnNewDocument(() => {
    sessionStorage.setItem("settings2", JSON.stringify({ system: "", think: {}, fast: {}, voice: { name: "silent" }, artifacts: false }));
    const ask = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (c) => { await new Promise((r) => setTimeout(r, 1500)); return ask(c); };
  });
  await page.goto(`${base}/`, { waitUntil: "networkidle0" });
  const hellos = () => voiceLog().filter((e) => e.ev === "msg" && e.type === "hello").length, n0 = hellos();
  await page.click("#voice");
  for (const t = Date.now(); hellos() === n0 && Date.now() - t < 1000; ) await Bun.sleep(20);
  await Bun.sleep(300);
  expect([hellos() > n0, await page.$eval("#voice-status", (e) => e.textContent)]).toEqual([true, "Starting voice…"]);
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 5000 });
  await page.close();
}, 30_000);

test("voice mode: the thinking bar is the chat's, one setting for spoken and written replies", async () => {
  const { page } = await voicePage("silent");
  const bar = () => page.$eval("#energy", (e) => [e.getAttribute("aria-valuetext"), (e as HTMLElement).title]);
  expect(await bar()).toEqual(["Max", "Thinking: Max"]);
  await page.focus("#energy");
  await page.keyboard.press("ArrowLeft");
  expect(await bar()).toEqual(["Medium", "Thinking: Medium"]);
  await openSettings(page);
  expect(await page.$(".voice-think")).toBeNull();
  await closeSettings(page);
  await page.click("#send");
  expect(await bar()).toEqual(["Medium", "Thinking: Medium"]);
  await page.close();
}, 30_000);

async function look(page: Page, clip: { x: number; y: number; width: number; height: number }, before = "", like = "") {
  const png = Buffer.from(await page.screenshot({ clip, encoding: "binary", captureBeyondViewport: false })).toString("base64");
  const counts = await page.evaluate(async (src, prev, like) => {
    const read = async (s: string) => {
      const img = new Image();
      img.src = `data:image/png;base64,${s}`;
      await img.decode();
      const cv = document.createElement("canvas");
      cv.width = img.width; cv.height = img.height;
      const ctx = cv.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      return ctx.getImageData(0, 0, cv.width, cv.height).data;
    };
    const d = await read(src), b = prev ? await read(prev) : null;
    const probe = document.body.appendChild(document.createElement("i"));
    probe.style.color = like || "transparent";
    const [lr, lg, lb] = getComputedStyle(probe).color.match(/[\d.]+/g)!.map(Number);
    probe.remove();
    let pink = 0, tint = 0, lit = 0, close = 0, changed = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i] > 90 && d[i] - d[i + 1] > 50) pink++;
      if (d[i] - d[i + 1] > 12 && d[i] > d[i + 2]) tint++;
      if (d[i] + d[i + 1] + d[i + 2] > 100) lit++;
      if (like && Math.abs(d[i] - lr) + Math.abs(d[i + 1] - lg) + Math.abs(d[i + 2] - lb) < 60) close++;
      if (b && Math.abs(d[i] - b[i]) + Math.abs(d[i + 1] - b[i + 1]) + Math.abs(d[i + 2] - b[i + 2]) > 30) changed++;
    }
    return { pink, tint, lit, close, changed };
  }, png, before, like);
  return { png, ...counts };
}

test("the sea drawn by the CPU, in a browser without WebGL, is the sea WebGL2 draws: the same water, the same butterfly, in a window and on a phone", async () => {
  const shots: Record<string, Awaited<ReturnType<typeof look>>> = {};
  for (const [name, flag] of [["gpu", "--enable-unsafe-swiftshader"], ["cpu", "--disable-3d-apis"]] as const) {
    const b = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: [...CHAT_ARGS, flag] });
    try {
      for (const [size, width, height] of [["window", 1280, 800], ["phone", 390, 844]] as const) {
        const page = await (await chatContext(b)).newPage();
        page.on("pageerror", (e) => { throw e; });
        await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "reduce" }, { name: "prefers-color-scheme", value: "dark" }]);
        await page.setViewport({ width, height });
        await page.goto(`${base}/`, { waitUntil: "load" });
        expect(await page.evaluate(() => !!document.createElement("canvas").getContext("webgl2"))).toBe(name === "gpu");
        await Bun.sleep(2000);
        const sea = await page.$eval("#sea", (e) => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
        shots[`${name} ${size}`] = await look(page, sea, shots[`gpu ${size}`]?.png ?? "");
        await page.close();
      }
    } finally {
      await b.close();
    }
  }
  for (const size of ["window", "phone"]) {
    const gpu = shots[`gpu ${size}`], cpu = shots[`cpu ${size}`];
    expect(gpu.lit).toBeGreaterThan(size === "window" ? 20_000 : 5_000);
    expect(gpu.pink).toBeGreaterThan(size === "window" ? 10_000 : 3_000);
    expect(cpu.changed / cpu.lit).toBeLessThan(0.01);
    expect(Math.abs(cpu.pink - gpu.pink) / gpu.pink).toBeLessThan(0.02);
  }
}, 90_000);

test("a new chat is the butterfly in a sea under the message box too, never blanked by a side panel sliding shut; a click on the water makes ripples and keeps the message box focused, and with reduced motion draws nothing; a pointer resting on the water draws it no more often than the water alone does", async () => {
  const gl = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: [...CHAT_ARGS, "--enable-unsafe-swiftshader"] });
  try {
    for (const motion of ["no-preference", "reduce"] as const) {
      const page = await (await chatContext(gl)).newPage();
      page.on("pageerror", (e) => { throw e; });
      await page.evaluateOnNewDocument(() => {
        const get = HTMLCanvasElement.prototype.getContext;
        (window as any).seaFrames = 0;
        HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, kind: string, ...rest: unknown[]) {
          const c = (get as any).call(this, kind, ...rest);
          if (kind === "webgl2" && c && !c.counted) {
            const draw = c.drawArrays.bind(c);
            c.counted = c.drawArrays = (...a: unknown[]) => { if (c.getParameter(c.FRAMEBUFFER_BINDING)) (window as any).seaFrames++; return draw(...a); };
          }
          return c;
        } as typeof get;
      });
      await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }, { name: "prefers-reduced-motion", value: motion }]);
      await page.setViewport({ width: 1200, height: 900 });
      await page.goto(`${base}/`, { waitUntil: "networkidle0" });
      await page.waitForSelector(".sea > canvas.ocean");
      const rect = async (sel: string) => (await (await page.$(sel))!.boundingBox())!;
      let sea = await rect(".sea"), box = await rect(".composer-box");
      const middle = { x: box.x + box.width / 2 - 150, y: (sea.y + box.y) / 2 - 100, width: 300, height: 200 };
      let pink = 0;
      for (const end = Date.now() + 5000; pink <= 800 && Date.now() < end; await Bun.sleep(100)) pink = (await look(page, middle)).pink;
      expect(pink).toBeGreaterThan(800);

      const open = sea.x, left = () => page.$eval(".sea", (e) => e.getBoundingClientRect().left);
      if (motion !== "reduce") await page.addStyleTag({ content: ":root { --pane-ease: 4s linear !important; }" });
      await page.click("#conversations-toggle");
      if (motion !== "reduce") {
        for (let k = 0; k < 2; k++) {
          await Bun.sleep(400);
          const from = await left(), { lit } = await look(page, { x: open + 20, y: sea.y + 40, width: 200, height: 200 }), to = await left();
          expect([from > 0, to < open, to < from]).toEqual([true, true, true]);
          expect(lit).toBeGreaterThan(500);
        }
      }
      await page.waitForFunction(() => document.querySelector(".sea")!.getBoundingClientRect().left === 0);
      sea = await rect(".sea"); box = await rect(".composer-box");
      const beside = { x: sea.x + 40, y: box.y, width: box.x - sea.x - 80, height: box.height };
      expect((await look(page, beside)).lit).toBeGreaterThan(150);
      expect(await page.evaluate((x, y) => document.elementFromPoint(x, y)?.className, beside.x + 20, box.y + box.height / 2)).toBe("ocean");

      await page.focus("#input");
      const at = { x: sea.x + 90, y: (sea.y + box.y) / 2 }, near = { x: at.x - 70, y: at.y - 80, width: 140, height: 160 };
      const far = { ...near, x: sea.x + sea.width - 90 - 70 };
      await page.mouse.move(at.x, at.y);
      await Bun.sleep(1500);
      const near0 = await look(page, near), far0 = await look(page, far);
      await page.mouse.down();
      await page.mouse.up();
      await Bun.sleep(400);
      const near1 = await look(page, near, near0.png), far1 = await look(page, far, far0.png);
      expect(await page.evaluate(() => document.activeElement?.id)).toBe("input");
      if (motion === "reduce") expect([near1.changed, far1.changed]).toEqual([0, 0]);
      else {
        expect(near1.changed).toBeGreaterThan(2 * far1.changed + 500);
        const frames = async () => { const n = await page.evaluate(() => (window as any).seaFrames); await Bun.sleep(2000); return await page.evaluate(() => (window as any).seaFrames) - n; };
        await Bun.sleep(5200);
        const resting = await frames();
        await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
        await Bun.sleep(300);
        const away = await frames();
        expect(away).toBeGreaterThan(10);
        expect(resting).toBeLessThan(away * 1.5);
      }
      await page.close();
    }
  } finally {
    await gl.close();
  }
}, 90_000);

test("the sea stays behind a chat, the butterfly sunk into it and every click the chat's, and a new chat raises it again; Settings takes the sea away from chats (remembered) and leaves it in a new chat; it follows the theme", async () => {
  const gl = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: [...CHAT_ARGS, "--enable-unsafe-swiftshader"] });
  try {
    const page = await (await chatContext(gl)).newPage();
    page.on("pageerror", (e) => { throw e; });
    await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
    await page.setViewport({ width: 1600, height: 900 });
    await page.goto(`${base}/`, { waitUntil: "networkidle0" });
    await page.waitForSelector(".sea > canvas.ocean");
    const rect = async (sel: string) => (await (await page.$(sel))!.boundingBox())!;
    const sea = await rect(".sea"), box = await rect(".composer-box");
    const middle = { x: box.x + box.width / 2 - 150, y: (sea.y + box.y) / 2 - 100, width: 300, height: 200 };
    const risen = async (least = 800) => {
      let pink = 0;
      for (const end = Date.now() + 5000; pink <= least && Date.now() < end; await Bun.sleep(100)) pink = (await look(page, middle)).pink;
      return pink;
    };
    const full = await risen();
    expect(full).toBeGreaterThan(800);

    await send(page, "capital of France");
    await Bun.sleep(1500);
    const column = await rect("#thread"), margin = { x: sea.x + 40, y: sea.y + 60, width: column.x - sea.x - 80, height: 300 };
    const sunk = await look(page, middle);
    expect(sunk.pink).toBeLessThan(50);
    expect(sunk.tint).toBeGreaterThan(300);
    expect((await look(page, margin)).lit).toBeGreaterThan(150);
    const hit = (x: number, y: number) => page.evaluate((x, y) => document.elementFromPoint(x, y)?.closest("#scroller, .sea")?.id, x, y);
    const reply = await rect(".msg.assistant .markdown");
    expect([await hit(reply.x + 5, reply.y + reply.height / 2), await hit(margin.x + 20, margin.y + 20)]).toEqual(["scroller", "scroller"]);

    await page.click("#new-chat");
    const rising = (await look(page, middle)).pink;
    expect([rising < 0.7 * full, await risen(0.9 * full) > 0.9 * full]).toEqual([true, true]);

    await page.click(".conv");
    await openSettings(page);
    await page.click("#sea-switch");
    await closeSettings(page);
    await page.waitForFunction((x) => document.getElementById("thread")!.getBoundingClientRect().x === x, {}, column.x);
    expect((await look(page, margin)).lit).toBe(0);
    await page.click("#new-chat");
    expect(await risen()).toBeGreaterThan(800);
    await page.reload({ waitUntil: "networkidle0" });
    expect(await page.$eval("#sea-switch", (b) => b.getAttribute("aria-checked"))).toBe("false");

    const cores = async () => (await look(page, middle, "", "var(--accent-hi)")).close;
    await openSettings(page);
    await page.click('#theme-pick [data-theme="light"]');
    await closeSettings(page);
    await Bun.sleep(600);
    expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe("light");
    expect(await cores()).toBeGreaterThan(100);
    await page.close();
  } finally {
    await gl.close();
  }
}, 60_000);
