import { afterAll, beforeAll, beforeEach, expect } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, rmSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import puppeteer, { type Browser, type BrowserContext, type Page } from "puppeteer-core";
import { ARTIFACTS_PROMPT } from "./artifacts";
import { cardPng } from "./cast";
import { fakeOpenCode } from "./fake-opencode";

import { deflateSync } from "node:zlib";

export const RUN = mkdtempSync(`${tmpdir()}/chrysalis-test-`);
process.on("exit", () => rmSync(RUN, { recursive: true, force: true }));

process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";
export const CHAT_ARGS = ["--no-sandbox", "--ignore-certificate-errors"];
export const BROWSER_ARGS = ["--force-prefers-reduced-motion", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"];

export const PANDA = '```svg panda\n<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">\n  <rect id="background" width="100" height="100" fill="#ffffff"/>\n' +
  '  <ellipse id="coat" cx="50" cy="60" rx="30" ry="25" fill="#ffffff" stroke="#222222"/>\n  <g id="burger">\n    <rect x="35" y="50" width="30" height="10" fill="#8b4513"/>\n  </g>\n</svg>\n```\n\nHere he is.';
export const RED_COAT = '```edit panda\n<ellipse id="coat" cx="50" cy="60" rx="30" ry="25" fill="#d32f2f" stroke="#222222"/>\n<<<<<<< SEARCH\nfill="#ffffff"/>\n' +
  '=======\nfill="#fff3cd"/>\n>>>>>>> REPLACE\n```\n\nRed now.';
export const COUNTER = '```html counter\n<!doctype html>\n<html><head><title>Counter</title>\n<style>\n  body { font: 16px sans-serif; margin: 0; padding: 16px; }\n' +
  '  #inc { background: #eeeeee; padding: 8px 14px; }\n  .card { padding: 12px; }\n</style></head>\n<body>\n<h1 id="title">Clicks</h1>\n' +
  '<section class="card" id="card">\n  <p id="count">0</p>\n  <button id="inc">Add</button>\n</section>\n<p id="note">Press the button.</p>\n' +
  '<script>\nlet n = 0;\ndocument.getElementById("inc").onclick = () => { n += 1; document.getElementById("count").textContent = n; };\n' +
  'document.body.dataset.ran = "yes";\n</script>\n</body></html>\n```\n\nA counter.';
export const PAPER = '```latex paper\n\\documentclass{article}\n\\title{Rods}\n\\begin{document}\n\\maketitle\n\\section{Heat}\nThe rod cools as\n' +
  '\\begin{equation}\n  u_t = -\\alpha u_{xx}\n\\end{equation}\nwhich is Fourier\'s law.\n\nA second paragraph.\n\\end{document}\n```\n\nThe paper.';
export const CUBE = '```html cube\n<!doctype html>\n<html><head><title>Cube</title>\n<script type="importmap">{"imports":{"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>\n' +
  '</head><body style="margin:0">\n<script type="module">\nimport * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";\n' +
  'import { OrbitControls } from "three/addons/controls/OrbitControls.js";\nimport { RoundedBoxGeometry } from "https://cdn.jsdelivr.net/npm/three@0.158.0/examples/jsm/geometries/RoundedBoxGeometry.js";\n' +
  'const renderer = new THREE.WebGLRenderer({ preserveDrawingBuffer: true });\nrenderer.setSize(300, 200);\ndocument.body.append(renderer.domElement);\n' +
  'const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, 1.5, 0.1, 100);\ncamera.position.z = 3;\n' +
  'scene.add(new THREE.Mesh(new RoundedBoxGeometry(1, 1, 1, 4, 0.1), new THREE.MeshNormalMaterial()));\nconst controls = new OrbitControls(camera, renderer.domElement);\n' +
  'renderer.render(scene, camera);\ndocument.body.dataset.three = THREE.REVISION + " " + (controls.target instanceof THREE.Vector3);\n</script>\n</body></html>\n```';
export const OLD_CUBE = '```html old\n<!doctype html>\n<html><head><title>Old</title>\n<style>body { margin: 0; overflow: hidden; }\n' +
  '#title { position: absolute; top: 10px; width: 100%; text-align: center; color: #c33; }</style>\n' +
  '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js" integrity="sha512-dLxUelApnYxpLt6K2iomGngnHO83iUvZytA3YjDUCjT0HDOHKXnVYdf3hU4JjM8uEhxf9nD1/ey98U3t2vZ0qQ==" crossorigin="anonymous"></script>\n' +
  '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>\n</head><body>\n<div id="title">Old</div>\n<script>\n' +
  'const renderer = new THREE.WebGLRenderer();\nrenderer.setSize(innerWidth, innerHeight);\ndocument.body.append(renderer.domElement);\n' +
  'addEventListener("resize", () => renderer.setSize(innerWidth, innerHeight));\nconst camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);\nconst controls = new THREE.OrbitControls(camera, renderer.domElement);\n' +
  'document.body.dataset.three = THREE.REVISION + " " + (controls.object === camera);\n</script>\n</body></html>\n```';
export const SCENE = '```aframe scene\n<!doctype html>\n<html><head><title>Scene</title>\n<script src="https://aframe.io/releases/1.5.0/aframe.min.js"></script>\n</head>\n<body style="margin:0">\n' +
  '<a-scene id="world" embedded style="width: 100%; height: 300px" vr-mode-ui="enabled: false" loading-screen="enabled: false">\n' +
  '  <a-box id="box" position="0 1.4 -3" color="#4CC3D9"></a-box>\n  <a-sky id="sky" color="#ECECEC"></a-sky>\n</a-scene>\n' +
  '<script>window.started = Math.random();</script>\n</body></html>\n```';
export const HALL = '```html\nhall\n<!doctype html>\n<html><head><title>Hall</title></head><body><h1 id="sign">Welcome</h1></body></html>\n';
export const GATE = '```html id="gate"\n<!doctype html>\n<html><head><title>Gate</title></head><body><p id="note">Open</p></body></html>\n```\n\nThe gate.';
export const REPLIES: Record<string, string> = {
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
  "a small table": "The fruit:\n\n| Fruit | Weight | Color |\n| --- | ---: | :-: |\n| Apple | 180 g | Red |\n| Banana | 120 g | Yellow |\n\nThat is all.",
  "only a table": "| Fruit | Color |\n| --- | --- |\n| Apple | Red |\n",
  "a wide table": `| ${Array.from({ length: 12 }, (_, i) => `Column number ${i}`).join(" | ")} |\n|${" --- |".repeat(12)}\n| ${Array.from({ length: 12 }, (_, i) => `a fairly long value ${i}`).join(" | ")} |\n`,
  "a long reply": Array(40).fill("Every word of this reply takes its place on a long line.").join(" "),
  "a long link": `| Name | Where |\n| --- | --- |\n| Docs | https://example.com/${"long-path-".repeat(30)} |\n`,
  "a bare table": "Here they are:\nFruit       | Color      | Typical Season\nApple       | Red/Green  | Fall\nBanana      | Yellow     | **Summer**\nGrape       | Purple     | Summer\n\nThat is all.",
  "pipes but no table": "One | two\nThree | four\n\n- a | b\n- c | d\n- e | f\n\nx | y | z\nno pipes here\nd | e | f\n\n```\nq | r\ns | t\nu | v\n```",
  math: "Solve $ax^2 + bx + c = 0$:\n\n$$\nx = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\n$$\n\nAlso \\(e^{i\\pi} + 1 = 0\\) and\n\n\\[\n\\int_0^1 x\\,dx\n\\]\n\n" +
    "It costs $5 and $10. Escaped: \\$x\\$. Code: `$HOME` stays.\n",
  diagram: "```mermaid\nflowchart LR\n  A[Ask] --> B{Cached?}\n  B -->|yes| C[Append]\n  B -->|no| D[Prefill]\n```\n\n" +
    "```mermaid\nsequenceDiagram\n  User->>Chrysalis: question\n  Chrysalis-->>User: answer\n```\n\n```mermaid\nflowchart TD\n  A -->\n  not mermaid ((\n```\n",
  "script fails in the browser": "The quote on that line is never closed.",
  "snapped script": "```html\n<!DOCTYPE html>\n<html><head><title>Snapped</title></head><body>\n<p id=\"p\">before</p>\n<script>\nconst ok = 1;\nconst bad = 'rgb(1,2,3)`;\n" +
    "document.getElementById(\"p\").textContent = \"after\";\n</script>\n</body></html>\n```\n",
  "leaving now": "```html\n<!DOCTYPE html>\n<html><head><title>Leaving</title></head><body>\n<p id=\"note\">here</p>\n<script>\nlocation.href = \"PROBE_URL/probe?left\";\n</script>\n</body></html>\n```\n\nAfter the page.\n",
  page: "```html\n<!DOCTYPE html>\n<html><head><title>Probe</title></head><body>\n<button id=\"b\" onclick=\"this.textContent = Number(this.textContent) + 1\">0</button>\n" +
    "<button id=\"lock\" onclick=\"this.requestPointerLock()\">hold</button><button id=\"full\" onclick=\"this.requestFullscreen().then(() => { document.body.dataset.full = 'yes'; }, (e) => { document.body.dataset.full = e.message; })\">fill</button>\n<pre id=\"out\">running</pre>\n<script>\nconst out = [];\n" +
    "document.addEventListener(\"pointerlockchange\", () => { document.body.dataset.lock = document.pointerLockElement ? \"held\" : \"free\"; });\n" +
    "document.addEventListener(\"pointerlockerror\", () => { document.body.dataset.lock = \"refused\"; });\ntry { out.push(\"parent \" + parent.document.title); } catch { out.push(\"parent blocked\"); }\n" +
    "try { localStorage.setItem(\"k\", \"1\"); out.push(localStorage.getItem(\"k\") === \"1\" && localStorage.length === 1 ? \"storage its own\" : \"storage broken\"); } catch { out.push(\"storage blocked\"); }\n" +
    "try { document.cookie = \"a=1\"; out.push(document.cookie === \"a=1\" ? \"cookie its own\" : \"cookie broken\"); } catch { out.push(\"cookie blocked\"); }\n" +
    "const img = new Image(); img.onload = () => out.push(\"image open\"); img.onerror = () => out.push(\"image blocked\"); img.src = \"PROBE_URL/probe.png\";\n" +
    "fetch(\"PROBE_URL/probe\", { method: \"POST\", mode: \"no-cors\" }).then(() => out.push(\"fetch open\"), () => out.push(\"fetch blocked\"))\n" +
    "  .finally(() => setTimeout(() => { document.getElementById(\"out\").textContent = out.sort().join(\",\"); }, 300));\n" +
    "</script>\n</body></html>\n```\n\nAfter the page.\n",
};
export const enc = new TextEncoder();
export const png = (() => {
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

export const SUMMARY = "## Objective\n- keep chatting\n\n## Findings\n- the word is lighthouse\n\n## Next Move\n1. answer the next message";
export const CONTEXT_FULL = () => Response.json({ error: { message: "the request (300000 tokens) exceeds the available context size (262144 tokens), try increasing it", type: "invalid_request_error", code: "context_length_exceeded" } }, { status: 400 });
export let lastStream: any = null, probes = 0, chrysalisStopped = 0, loading: Promise<void> | null = null;
export const streams: any[] = [], streamAt: number[] = [], checks: any[] = [];
export const fakes = { memoryRefusals: 0, naps: 0, loaded: "qwen3.8-27b", hurry: false };
export let fake: ReturnType<typeof Bun.serve>;
const serveFake = () => Bun.serve({
  hostname: "127.0.0.1",
  port: 0,
  async fetch(req) {
    const path = new URL(req.url).pathname;
    if (path.startsWith("/probe")) return (probes++, new Response(path.endsWith(".png") ? png : "", { headers: { "Content-Type": "image/png" } }));
    if (path === "/chrysalis/stats") return Response.json({ model: fakes.loaded, context: 262144, kv_tokens: 276000, cache: { free_kv_tokens: 270000 } });
    if (path === "/chrysalis/load") {
      const { model } = await req.json();
      await (loading = Bun.sleep(1000).then(() => { fakes.loaded = model; }));
      return Response.json({ model });
    }
    if (path !== "/v1/chat/completions") return new Response("not found", { status: 404 });
    const body = await req.json();
    const prefix = body.continue_final_message ? body.messages.at(-1) : undefined;
    if (!body.stream && /^You fix speech recognition errors/.test(body.messages[0]?.content)) {
      checks.push(body);
      const heard = String(body.messages.at(-1).content).split("Transcript: ").at(-1)!;
      await Bun.sleep(heard.startsWith("wait") ? 2000 : heard.includes("pie game") ? 1000 : 0);
      return Response.json({ choices: [{ message: { content: heard.includes("pie game") ? "pie game => Pygame" : "-" }, finish_reason: "stop" }] });
    }
    if (!body.stream) return Response.json({ choices: [{ message: { content: "A title" } }] });
    lastStream = body;
    streams.push(body);
    streamAt.push(Date.now());
    req.signal.addEventListener("abort", () => chrysalisStopped++);
    const at = body.messages.findLastIndex((m: any) => m.role === "user"), last = body.messages[at];
    const said = !last ? "" : (typeof last.content === "string" ? last.content : last.content.map((p: any) => p.text ?? "").join("")).replace(/^\[[^\]]*\]\n\n/, "").split("\n\n<conversation-checkpoint>")[0]
      .replace(/^<conversation-checkpoint>[\s\S]*<\/conversation-checkpoint>\n\n/, "");
    const summary = /structured summary|Update the existing checkpoint/.test(said), covered = JSON.stringify(body.messages).includes("<conversation-checkpoint>");
    const round = body.messages.slice(at).filter((m: any) => m.role === "assistant" && m !== prefix).length;
    if (!summary && !covered && said.startsWith("overflow now")) return CONTEXT_FULL();
    if (said === "overflow again" && !JSON.stringify(body.messages).includes("still lighthouse")) return CONTEXT_FULL();
    if (said === "needs memory" && fakes.memoryRefusals-- > 0) return Response.json({ error: { message: "the request exceeds the available context size (KV cache full), try increasing it", code: "context_length_exceeded" } }, { status: 400 });
    if (said === "chrysalis naps" && fakes.naps-- > 0) return Response.json({ error: { message: "the machine is going to sleep; Chrysalis loads the model again when it wakes", code: "unavailable" } }, { status: 503 });
    const next = /\[(.+) next\.\]$/.exec(said)?.[1];
    const think = summary ? "" : next ? `${next} has the floor.` : "";
    const slowly = said.includes("slowly") && !summary && (!said.startsWith("tool") || round === 0);
    const text = summary ? (said.includes("Update the existing checkpoint") ? SUMMARY.replace("is lighthouse", "is still lighthouse") : SUMMARY) : slowly ? Array.from({ length: 100 }, (_, k) => `w${k} `).join("") : next ? `${next}: I go on.` : said.startsWith("slow")
      ? Array.from({ length: 40 }, (_, k) => `w${k} `).join("") : Object.entries(REPLIES).find(([k]) => said.includes(k))?.[1] ?? `You said: ${said}`;
    const step = said.startsWith("slow") || slowly ? 4 : 9, gap = said.startsWith("slow") || slowly ? 60 : said.startsWith("steady") ? 25 : 5;
    let i = 0, j = 0, restated = !prefix, read = false;
    return new Response(new ReadableStream({
      async pull(ctl) {
        await Bun.sleep(fakes.hurry ? 0 : gap);
        if (!restated) {
          restated = true;
          return ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { chrysalis_prefix: { reasoning_content: prefix.reasoning_content || "\n", content: prefix.content } } }] })}\n\n`));
        }
        if (!read) { read = true; return ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: {} }], chrysalis_prompt: { prompt_tokens: 10, cached_tokens: 0 } })}\n\n`)); }
        if (j < think.length) { ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { reasoning_content: think.slice(j, j + step) } }] })}\n\n`)); j += step; return; }
        if (i < text.length) { ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: text.slice(i, i + step) } }] })}\n\n`)); i += step; return; }
        ctl.enqueue(enc.encode(`data: ${JSON.stringify({ seed: 1, choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`));
        ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 } })}\n\n`));
        ctl.enqueue(enc.encode("data: [DONE]\n\n"));
        ctl.close();
      },
    }), { headers: { "Content-Type": "text/event-stream" } });
  },
});

export let web: Bun.Subprocess<"ignore", "pipe", "inherit">, base = "", browser: Browser;
export const dir = mkdtempSync(`${RUN}/web-`), LOG = `${dir}/mcp.log`, VOICE_LOG = `${dir}/voice.log`;
const PAGE = REPLIES.page, LEAVING = REPLIES["leaving now"];
export const opencode = fakeOpenCode("test-secret");
export const home = mkdtempSync(`${RUN}/home-`);
async function start() {
  const tls = mkdtempSync(`${RUN}/tls-`);
  writeFileSync(`${tls}/opencode-off`, "");
  fake = serveFake();
  web = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", CHRYSALIS_URL: `http://127.0.0.1:${fake.port}`, CHRYSALIS_HOST: "127.0.0.1", NODE_ENV: "production",
      CHRYSALIS_OPENCODE_URL: `http://127.0.0.1:${opencode.server.port}`, CHRYSALIS_OPENCODE_PASSWORD: "test-secret",
      CHRYSALIS_VOICE_CMD: JSON.stringify([process.execPath, `${import.meta.dir}/fake-voice.ts`]), FAKE_VOICE_LOG: VOICE_LOG, CHRYSALIS_VOICE_IDLE_MS: "1500",
      CHRYSALIS_TLS_DIR: tls, CHRYSALIS_CHECK_PORT: "0", CHRYSALIS_RETRY_MS: "400", HOME: home },
  });
  const reader = web.stdout.getReader(), dec = new TextDecoder();
  let out = "";
  while (!/chrysalis web: \S+/.test(out)) {
    const { value, done } = await reader.read();
    if (done) throw new Error(`server.ts exited: ${out}`);
    out += dec.decode(value);
  }
  base = out.match(/chrysalis web: (\S+)/)![1].replace(/\/$/, "");
  REPLIES.page = PAGE.replaceAll("PROBE_URL", `http://127.0.0.1:${fake.port}`);
  REPLIES["leaving now"] = LEAVING.replaceAll("PROBE_URL", `http://127.0.0.1:${fake.port}`);
  browser = process.env.CHROMIUM_WS ? await puppeteer.connect({ browserWSEndpoint: process.env.CHROMIUM_WS })
    : await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: [...CHAT_ARGS, ...BROWSER_ARGS] });
}
export function harness() {
  beforeAll(start, 30_000);
  beforeEach(() => { fakes.hurry = false; });
  afterAll(async () => {
    for (const c of contexts.splice(0)) await c.close().catch(() => {});
    await (process.env.CHROMIUM_WS ? browser.disconnect() : browser.close());
    web.kill();
    await web.exited;
    fake.stop(true);
  });
}

export const idle = async (page: Page, went: Promise<unknown>) => { await went; await page.waitForSelector("#app:not([aria-busy])"); await page.waitForNetworkIdle({ idleTime: 50 }); };
const contexts: BrowserContext[] = [];
export const chatContext = async (b = browser) => {
  const c = await b.createBrowserContext();
  if (b === browser) contexts.push(c);
  return c;
};
export async function open(): Promise<Page> {
  const page = await (await chatContext()).newPage();
  await page.setViewport({ width: 1200, height: 900 });
  page.on("pageerror", (e) => { throw e; });
  await idle(page, page.goto(`${base}/`));
  return page;
}
export async function send(page: Page, text: string) {
  const done = await page.$$eval(".msg .stats", (s) => s.length);
  await page.type("#input", text);
  await page.keyboard.press("Enter");
  await page.waitForFunction((n) => document.querySelectorAll(".msg .stats").length > n, { timeout: 20_000 }, done);
}

export async function still(page: Page) {
  await page.waitForSelector("#shelf .art-canvas");
  await page.waitForFunction(() => new Promise<boolean>((done) => {
    const at = () => JSON.stringify(document.querySelector("#shelf .art-canvas")!.getBoundingClientRect()), was = at();
    requestAnimationFrame(() => requestAnimationFrame(() => done(at() === was)));
  }));
}

export const fill = (page: Page, id: string, attr = "fill") => page.$eval(".art-canvas .svg-paper", (p, id, attr) =>
  p.shadowRoot!.querySelector(`[id="${id}"]`)?.getAttribute(attr as string) ?? null, id, attr);

export const pageFrame = async (page: Page) => (await page.waitForSelector(".art-canvas iframe.page-frame:not(.next)"))!.contentFrame();

export async function inPage(page: Page, sel: string) {
  for (let k = 0; ; k++) {
    const box = (await (await page.$(".art-canvas iframe.page-frame:not(.next)"))!.boundingBox())!;
    const r = await (await pageFrame(page)).$eval(sel, (e) => { const r = e.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; });
    if (r.y < box.height - 2 || k === 40) return { x: box.x + 1 + r.x, y: box.y + 1 + r.y };
    await Bun.sleep(50);
  }
}

export async function pageTool(page: Page, sel: string) {
  if (!(await page.$(".art-canvas .art-tools.on"))) await page.click(".art-canvas [data-art=edit]");
  await page.click(`.art-canvas .art-tools ${sel}`);
  const mode = { select: "select", text: "text", fill: "bucket", done: "use" }[sel.match(/data-(?:tool|do)=(\w+)/)?.[1] ?? ""];
  if (mode) await (await pageFrame(page)).waitForFunction((t) => (window as any).__chrysalis.mode === t, {}, mode);
}

export const pickColor = (page: Page, color: string) => page.$eval(".art-canvas .art-tools input", (i, color) => {
  (i as HTMLInputElement).value = color as string;
  i.dispatchEvent(new Event("input"));
}, color);

export const card = (name: string, data: object) => cardPng({ spec: "chara_card_v3", spec_version: "3.0", data: { name, ...data } }, new Blob([png], { type: "image/png" }));

export const castRows = (page: Page) => page.$$eval("#conversations .cast-row .cast-name", (r) => r.map((x) => x.textContent!).sort());

export const inRow = (page: Page, name: string, part: string) => page.evaluate((name, part) => [...document.querySelectorAll("#conversations .cast-row")]
  .find((r) => r.querySelector(".cast-name")!.textContent === name)!.querySelector<HTMLElement>(part)!.click(), name, part);

export const openRow = (page: Page, name: string) => inRow(page, name, ".cast-main");

export function pngText(b: Uint8Array) {
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

export const docRoot = (page: Page) => page.$eval(".art-canvas .doc-paper", (p) => !!p.shadowRoot);

export const inDoc = (page: Page, text: string) => page.$eval(".art-canvas .doc-paper", (p, text) => {
  const el = [...p.shadowRoot!.querySelectorAll<HTMLElement>("[data-at]")].findLast((e) => e.textContent!.includes(text as string))!;
  const r = el.getBoundingClientRect();
  return { x: r.x + Math.min(r.width / 2, 40), y: r.y + r.height / 2 };
}, text);

export const checked = (page: Page, sel: string) => page.$eval(sel, (b) => b.getAttribute("aria-checked"));

export async function download(page: Page, into: string) {
  const cdp = await page.createCDPSession();
  await cdp.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: into, browserContextId: page.browserContext().id });
  await page.click("#export-all");
  for (let k = 0; k < 50; k++) { const f = readdirSync(into).find((x) => x.startsWith("chrysalis-chats-")); if (f) return `${into}/${f}`; await Bun.sleep(100); }
  throw new Error("no export");
}

export const articles = (page: Page) => page.$$eval("#thread > article", (a) => a.map((x) => (x.querySelector(".bubble, .markdown") as HTMLElement)?.innerText.trim()));

export const pagerOf = (page: Page, i: number) => page.$eval(`#thread > article[data-i="${i}"]`, (a) => a.querySelector(".pager-n")?.textContent ?? "");

export async function untilIdle(page: Page) {
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send", { timeout: 20_000 });
}

export const noteShown = async (page: Page) => {
  await openSettings(page);
  const note = await page.$eval("#voice-prompt", (e) => (e as HTMLTextAreaElement).value);
  await closeSettings(page);
  return note;
};

export const sent = (b: any) => b.messages;

export const voiceLog = (): any[] => { try { return readFileSync(VOICE_LOG, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l)); } catch { return []; } };

export const voicePages: Page[] = [];

export async function voicePage(script: string, o: { system?: string; first?: (page: Page) => Promise<void>; keepPointer?: boolean } = {}) {
  const page = await (await chatContext()).newPage();
  voicePages.push(page);
  await page.setViewport({ width: 1600, height: 900 });
  page.on("pageerror", (e) => { throw e; });
  await page.evaluateOnNewDocument((v, system) => {
    if (!sessionStorage.getItem("settings2")) sessionStorage.setItem("settings2", JSON.stringify({ system, think: {}, fast: {}, voice: { name: v }, artifacts: false }));
  }, script, o.system ?? "");
  await idle(page, page.goto(`${base}/`));
  await o.first?.(page);
  await (o.keepPointer ? page.$eval("#voice", (b) => (b as HTMLElement).click()) : page.click("#voice"));
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 10_000 });
  const s = voiceLog().filter((e) => e.ev === "msg" && e.type === "hello" && e.voice === script).at(-1).s;
  return { page, session: () => voiceLog().filter((e) => e.s === s) };
}

export async function pickVoice(page: Page, trigger: string, id: string) {
  await page.click(trigger);
  await page.waitForSelector(`#voice-menu:popover-open .voice-chip[data-voice="${id}"]`, { timeout: 10_000 });
  await page.click(`#voice-menu .voice-chip[data-voice="${id}"] > span`);
  await page.waitForFunction(() => !document.querySelector("#voice-menu:popover-open"));
}

export const replyText = (page: Page, i: number) => page.$eval(`#thread > article[data-i="${i}"] .markdown`, (el) => el.textContent?.trim() ?? "");

export async function openSettings(page: Page, sec = "chat") {
  if (!(await page.$eval("#settings", (d) => (d as HTMLDialogElement).open))) await page.click("#settings-open");
  await page.click(`.set-nav [data-sec="${sec}"]`);
  await page.waitForFunction((sec) => { const r = document.querySelector(`.set-sec[data-sec="${sec}"]`)!.getBoundingClientRect(); return r.width > 0 && r.right <= innerWidth; }, {}, sec);
}

export async function closeSettings(page: Page) {
  await page.click("#settings-close");
  await page.waitForFunction(() => !(document.getElementById("settings") as HTMLDialogElement).open);
}

export const stored = (page: Page) => page.evaluate(() => JSON.parse(sessionStorage.getItem("settings2") ?? "{}").voice);

export const settled = async (ok: () => boolean, ms = 5000) => { for (const t = Date.now(); !ok() && Date.now() - t < ms; ) await Bun.sleep(25); return ok(); };

export const SIZES = [[1920, 1080], [1440, 900], [1210, 834], [834, 1210], [390, 844], [320, 640]] as const;

export const composerFaults = (page: Page) => page.evaluate(() => {
  const box = document.querySelector(".composer-box")!.getBoundingClientRect(), name = (e: Element) => e.id || e.className;
  const shown = [...document.querySelectorAll(".composer-row button, .composer-row .think-ctl, .composer-row .voice-pill")].filter((e) => e.getBoundingClientRect().width > 0);
  const out = shown.filter((e) => { const r = e.getBoundingClientRect(); return r.left < box.left || r.right > box.right || r.top < box.top || r.bottom > box.bottom; });
  const blank = shown.filter((e) => e.matches("button") && !e.textContent!.trim() && !e.querySelector("svg, i"));
  return { out: out.map(name), blank: blank.map(name), sideways: document.documentElement.scrollWidth > innerWidth };
});

export const shown = (page: Page, sel: string) => page.$eval(sel, (e) => e.getBoundingClientRect().width > 0);

export const CLEAN = { out: [], blank: [], sideways: false };

export async function resize(page: Page, width: number, height: number) {
  await page.setViewport({ width, height });
  await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
}

export async function pastWarning(page: Page, firefox: boolean) {
  if (firefox) {
    await page.waitForSelector("net-error-card >>> #advanced-button", { timeout: 10_000 });
    while (!(await page.$("net-error-card >>> #exception-button:not([disabled])"))) {
      await page.click("net-error-card >>> #advanced-button").catch(() => {});
      await page.waitForSelector("net-error-card >>> #exception-button:not([disabled])", { timeout: 500 }).catch(() => {});
    }
    await page.click("net-error-card >>> #exception-button");
  } else {
    await page.waitForSelector("#details-button", { timeout: 10_000 });
    await page.click("#details-button");
    await page.click("#proceed-link");
  }
}

export function trusting(home: string, tlsDir: string) {
  const profile = `${home}/.mozilla/firefox/test.default`, chromium = `${home}/.pki/nssdb`;
  for (const db of [profile, chromium]) {
    mkdirSync(db, { recursive: true });
    expect(Bun.spawnSync(["certutil", "-N", "--empty-password", "-d", `sql:${db}`]).exitCode).toBe(0);
    expect(Bun.spawnSync(["certutil", "-A", "-d", `sql:${db}`, "-n", "Chrysalis", "-t", "C,,", "-i", `${tlsDir}/ca.pem`]).exitCode).toBe(0);
  }
  return profile;
}

export const spokenOf = (log: any[]) => {
  const said = new Map<string, string>();
  for (const e of log.filter((x) => x.type === "say")) said.set(e.id, (said.get(e.id) ?? "") + e.text);
  return { said, cancelled: new Set(log.filter((x) => x.type === "cancel" && x.id !== undefined).map((x) => x.id)) };
};

export const turnAt = (log: any[]) => log.find((e) => e.ev === "sent" && e.type === "turn")?.t;

type Clip = { x: number; y: number; width: number; height: number };
const counted = (page: Page, png: string, before: string, like: string, clips: Clip[] | null) => page.evaluate(async (src, prev, like, clips) => {
  const read = async (s: string) => {
    const img = new Image();
    img.src = `data:image/png;base64,${s}`;
    await img.decode();
    const cv = document.createElement("canvas");
    cv.width = img.width; cv.height = img.height;
    const ctx = cv.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    return { d: ctx.getImageData(0, 0, cv.width, cv.height).data, w: img.width, h: img.height };
  };
  const { d, w, h } = await read(src), b = prev ? (await read(prev)).d : null;
  const probe = document.body.appendChild(document.createElement("i"));
  probe.style.color = like || "transparent";
  const [lr, lg, lb] = getComputedStyle(probe).color.match(/[\d.]+/g)!.map(Number);
  probe.remove();
  return (clips ?? [{ x: 0, y: 0, width: w, height: h }]).map((c) => {
    let pink = 0, tint = 0, lit = 0, close = 0, changed = 0;
    for (let y = Math.round(c.y); y < Math.round(c.y + c.height); y++) for (let x = Math.round(c.x); x < Math.round(c.x + c.width); x++) {
      const i = (y * w + x) * 4;
      if (d[i] > 90 && d[i] - d[i + 1] > 50) pink++;
      if (d[i] - d[i + 1] > 12 && d[i] > d[i + 2]) tint++;
      if (d[i] + d[i + 1] + d[i + 2] > 100) lit++;
      if (like && Math.abs(d[i] - lr) + Math.abs(d[i + 1] - lg) + Math.abs(d[i + 2] - lb) < 60) close++;
      if (b && Math.abs(d[i] - b[i]) + Math.abs(d[i + 1] - b[i + 1]) + Math.abs(d[i + 2] - b[i + 2]) > 30) changed++;
    }
    return { pink, tint, lit, close, changed };
  });
}, png, before, like, clips);
const shot = async (page: Page, clip?: Clip) => Buffer.from(await page.screenshot({ clip, encoding: "binary", captureBeyondViewport: false })).toString("base64");
export async function look(page: Page, clip: Clip, before = "", like = "") {
  const png = await shot(page, clip);
  return { png, ...(await counted(page, png, before, like, null))[0] };
}
export async function looks(page: Page, clips: Clip[], before = "") {
  const png = await shot(page);
  return { png, parts: await counted(page, png, before, "", clips) };
}

export async function onNetwork() {
  const tlsDir = mkdtempSync(`${RUN}/tls-`), others: Browser[] = [];
  const start = async (ports = { PORT: "0", CHRYSALIS_CHECK_PORT: "0" }) => {
    const proc = Bun.spawn([process.execPath, "server.ts"], {
      cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
      env: { ...process.env, ...ports, CHRYSALIS_URL: `http://127.0.0.1:${fake.port}`, CHRYSALIS_HOST: "0.0.0.0", NODE_ENV: "production", CHRYSALIS_LINK_PORT: "0",
        CHRYSALIS_SELF_AS_NETWORK: "1",
        CHRYSALIS_TLS_DIR: tlsDir },
    });
    let out = "";
    for (const reader = proc.stdout.getReader(), dec = new TextDecoder(); !/chrysalis check: \S+/.test(out); ) {
      const { value, done } = await reader.read();
      if (done) throw new Error(`server.ts exited: ${out}`);
      out += dec.decode(value);
    }
    return { proc, out };
  };
  let { proc, out } = await start();
  const here = out.match(/chrysalis web: (\S+)/)![1], chat = out.match(/chrysalis web: (https:\/\/(?!127\.)\S+)/)![1];
  return {
    tlsDir, here, chat,
    launch: async (firefox: boolean, home = "", args: string[] = []) => {
      const b = await puppeteer.launch(firefox
        ? { browser: "firefox", executablePath: process.env.FIREFOX ?? "/usr/bin/firefox", headless: true, userDataDir: home || mkdtempSync(`${RUN}/fx-`),
          extraPrefsFirefox: { "security.dialog_enable_delay": 0 } }
        : { executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: ["--no-sandbox", ...args], ...(home ? { env: { ...process.env, HOME: home } } : {}) });
      others.push(b);
      return b;
    },
    waiting: async () => (await (await fetch(`${here}api/pair`)).json()).waiting,
    restart: async () => {
      const port = (re: RegExp) => new URL(out.match(re)![1]).port;
      const ports = { PORT: port(/chrysalis web: (\S+)/), CHRYSALIS_CHECK_PORT: port(/chrysalis check: (\S+)/) };
      proc.kill();
      await proc.exited;
      ({ proc, out } = await start(ports));
    },
    stop: async () => {
      for (const b of others) await b.close().catch(() => {});
      proc.kill();
    },
  };
}
export const setUpShown = (page: Page) => page.waitForFunction(() => document.getElementById("setup")?.hidden === false, { timeout: 15_000 });
export const stageDrawn = async (page: Page) => {
  const stage = await page.$eval("#stage", (e) => { const b = e.getBoundingClientRect(); return { x: b.x, y: b.y, width: b.width, height: b.height }; });
  for (const end = Date.now() + 5000; Date.now() < end; await Bun.sleep(100)) if ((await look(page, stage)).pink > 1500) return true;
  return false;
};

export const work = `${dir}/site`;
mkdirSync(work, { recursive: true });

export async function fresh(page: Page) {
  await page.click("#new-chat");
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#palette[open] .palette-item")].find((b) => b.querySelector(".palette-title")!.textContent === "No project")?.click());
}
export async function outside(page: Page, on: boolean) {
  expect(await page.evaluate((on) => fetch("/api/threads/outside", on ? { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" } : { method: "DELETE" }).then((r) => r.status), on)).toBe(204);
  if (on) await page.reload();
}
export async function connect(page: Page) {
  await page.click("#opencode");
  await page.waitForSelector("#oc-pop .invite-head");
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#oc-pop .invite-card")].find((b) => b.textContent === "Connect")?.click());
  await page.waitForFunction(() => /OpenCode 2/.test(document.querySelector("#oc-pop .invite-head")?.textContent ?? ""));
}
export async function openFolder(page: Page, path: string) {
  await page.keyboard.press("Escape");
  await fresh(page);
  await page.waitForSelector("#folder-btn:not([hidden])");
  await page.click("#folder-btn");
  await page.waitForSelector("#oc-pop .oc-path");
  await page.$eval("#oc-pop .oc-path", (e, v) => { (e as HTMLInputElement).value = v; (e as HTMLInputElement).focus(); }, path);
  await page.keyboard.press("Enter");
  await page.waitForFunction((want) => document.querySelector("#oc-pop .oc-use")?.getAttribute("title") === want, {}, path);
  await page.click("#oc-pop .oc-use");
  await page.waitForFunction((want) => document.querySelector("#folder-btn")?.getAttribute("title") === `Working in ${want}`, {}, path);
}
export async function menu(page: Page, title: string, item: string) {
  await page.evaluate((title) => [...document.querySelectorAll<HTMLElement>("#conversations a.conv")].find((a) => a.querySelector(".conv-title")?.textContent === title)!
    .dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true, clientX: 60, clientY: 300 })), title);
  await page.waitForSelector("#row-menu:not([hidden])");
  await page.evaluate((item) => [...document.querySelectorAll<HTMLElement>("#row-menu button")].find((b) => b.textContent!.startsWith(item))!.click(), item);
}
export const cards = (page: Page) => page.$$eval("#thread .tool-card", (c) => c.map((x) => [x.querySelector(".tool-name")!.textContent, x.querySelector(".tool-src")!.textContent, (x as HTMLElement).dataset.state]));
export const lastText = (page: Page) => page.$eval("#thread > article.assistant:last-child > .markdown", (e) => (e as HTMLElement).innerText.trim());
