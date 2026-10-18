// The page in a real browser (headless Chromium) against a fake kiln that streams canned replies: TeX typeset and
// prices left alone, Mermaid diagrams drawn, a reply's HTML page run in its sandbox where it reaches nothing, attached
// files (pictures, PDFs, videos, texts) sent as kiln expects them and kept across a reload, and chats carried through
// export and import.
import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import puppeteer, { type Browser, type Page } from "puppeteer-core";
import { deflateSync } from "node:zlib";

const REPLIES: Record<string, string> = {
  math: "Solve $ax^2 + bx + c = 0$:\n\n$$\nx = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\n$$\n\nAlso \\(e^{i\\pi} + 1 = 0\\) and\n\n\\[\n\\int_0^1 x\\,dx\n\\]\n\n" +
    "It costs $5 and $10. Escaped: \\$x\\$. Code: `$HOME` stays.\n",
  diagram: "```mermaid\nflowchart LR\n  A[Ask] --> B{Cached?}\n  B -->|yes| C[Append]\n  B -->|no| D[Prefill]\n```\n\n" +
    "```mermaid\nsequenceDiagram\n  User->>kiln: question\n  kiln-->>User: answer\n```\n\n```mermaid\nflowchart TD\n  A -->\n  not mermaid ((\n```\n",
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
// a 2 x 2 PNG, red / blue
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

let lastStream: any = null, probes = 0;              // probes: requests a sandboxed page got through to a server
const fake = Bun.serve({
  hostname: "127.0.0.1",
  port: 0,
  async fetch(req) {
    const path = new URL(req.url).pathname;
    if (path.startsWith("/probe")) return (probes++, new Response(path.endsWith(".png") ? png : "", { headers: { "Content-Type": "image/png" } }));
    if (path === "/kiln/stats") return Response.json({ model: "qwen3.8-27b", context: 262144, kv_tokens: 276000, cache: { free_kv_tokens: 270000 } });
    if (path !== "/v1/chat/completions") return new Response("not found", { status: 404 });
    const body = await req.json();
    if (!body.stream) return Response.json({ choices: [{ message: { content: "A title" } }] });
    lastStream = body;
    const last = body.messages.at(-1), said = typeof last.content === "string" ? last.content : last.content.map((p: any) => p.text ?? "").join("");
    const text = Object.entries(REPLIES).find(([k]) => said.includes(k))?.[1] ?? `You said: ${said}`;
    let i = 0;
    return new Response(new ReadableStream({
      async pull(ctl) {
        await Bun.sleep(5);
        if (i < text.length) { ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: text.slice(i, i + 9) } }] })}\n\n`)); i += 9; return; }
        ctl.enqueue(enc.encode(`data: ${JSON.stringify({ seed: 1, choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`));
        ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 } })}\n\n`));
        ctl.enqueue(enc.encode("data: [DONE]\n\n"));
        ctl.close();
      },
    }), { headers: { "Content-Type": "text/event-stream" } });
  },
});

let web: Bun.Subprocess<"ignore", "pipe", "inherit">, base = "", browser: Browser;
const dir = mkdtempSync(`${tmpdir()}/kiln-web-`);
beforeAll(async () => {
  web = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", KILN_URL: `http://127.0.0.1:${fake.port}`, KILN_HOST: "127.0.0.1", NODE_ENV: "production" },
  });
  const reader = web.stdout.getReader(), dec = new TextDecoder();
  let out = "";
  while (!/kiln web: \S+/.test(out)) {
    const { value, done } = await reader.read();
    if (done) throw new Error(`server.ts exited: ${out}`);
    out += dec.decode(value);
  }
  base = out.match(/kiln web: (\S+)/)![1].replace(/\/$/, "");
  REPLIES.page = REPLIES.page.replaceAll("PROBE_URL", `http://127.0.0.1:${fake.port}`);
  browser = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: ["--no-sandbox"] });
}, 30_000);
afterAll(async () => { await browser?.close(); web.kill(); fake.stop(true); });

async function open(): Promise<Page> {
  const page = await browser.newPage();
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

test("attached files go to kiln as images, PDFs, videos and fenced text, stay after a reload, and travel in an export", async () => {
  const pdf = Buffer.from("%PDF-1.4\n%kiln\n"), mp4 = Buffer.from([0, 0, 0, 24, 102, 116, 121, 112, 109, 112, 52, 50]);
  writeFileSync(`${dir}/dot.png`, png);
  writeFileSync(`${dir}/fence.md`, "Has ``` inside.\n");
  writeFileSync(`${dir}/blob.bin`, new Uint8Array([0, 159, 146, 150, 255]));
  writeFileSync(`${dir}/doc.pdf`, pdf);
  writeFileSync(`${dir}/clip.mp4`, mp4);
  const page = await open();
  const cdp = await page.createCDPSession();
  await cdp.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: dir });
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
  await page.click("#export");
  const file = await (async () => { for (let k = 0; k < 50; k++) { const f = readdirSync(dir).find((x) => x.endsWith(".kiln.json")); if (f) return f; await Bun.sleep(100); } })();
  const other = await open();                            // another tab: its own conversations and files
  await (await other.$("input#import-file"))!.uploadFile(`${dir}/${file}`);
  await other.waitForFunction(() => (document.querySelector(".msg-files img") as HTMLImageElement | null)?.naturalWidth === 2);
  await send(other, "and once more");
  expect(lastStream.messages.find((m: any) => m.role === "user").content).toEqual(content);
  expect(JSON.parse(readFileSync(`${dir}/${file}`, "utf8")).conversations[0].messages.length).toBe(4);
  await page.close();
  await other.close();
}, 60_000);
