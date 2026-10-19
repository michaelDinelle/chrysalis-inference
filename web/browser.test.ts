// The page in a real browser (headless Chromium) against a fake Chrysalis that streams canned replies: TeX typeset and
// prices left alone, Mermaid diagrams drawn, a reply's HTML page run in its sandbox where it reaches nothing, attached
// files (pictures, PDFs, videos, texts) sent as Chrysalis expects them and kept across a reload, chats carried through
// export and import, versions of messages and replies, tools (an MCP server, fake-mcp.ts, that server.ts runs), voice
// mode against a scripted speech service (fake-voice.ts) with Chromium's fake microphone, and a new chat's sea.
import { afterAll, beforeAll, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import puppeteer, { type Browser, type Page } from "puppeteer-core";
import { lanAddresses } from "./tls";
import { deflateSync } from "node:zlib";

const REPLIES: Record<string, string> = {
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

// Questions that make the fake Chrysalis call a tool (in its first round, then answer once the result is in), or write slowly.
const TOOL_ASKS: Record<string, { name: string; args: string }> = {
  "tool echo": { name: "fake__echo", args: '{"text":  "hello"}' },
  "tool note": { name: "fake__write_note", args: '{"text": "n1"}' },
};
let lastStream: any = null, probes = 0, callIds = 0, chrysalisStopped = 0;  // probes: requests a sandboxed page got through to a server
const streams: any[] = [], streamAt: number[] = [];  // every streamed request Chrysalis got, and when
const checks: any[] = [];                            // every check of what speech recognition heard (app.ts CHECK)
const fake = Bun.serve({
  hostname: "127.0.0.1",
  port: 0,
  async fetch(req) {
    const path = new URL(req.url).pathname;
    if (path.startsWith("/probe")) return (probes++, new Response(path.endsWith(".png") ? png : "", { headers: { "Content-Type": "image/png" } }));
    if (path === "/chrysalis/stats") return Response.json({ model: "qwen3.8-27b", context: 262144, kv_tokens: 276000, cache: { free_kv_tokens: 270000 } });
    if (path !== "/v1/chat/completions") return new Response("not found", { status: 404 });
    const body = await req.json();
    const prefix = body.continue_final_message ? body.messages.at(-1) : undefined;
    if (!body.stream && /^You fix speech recognition errors/.test(body.messages[0]?.content)) {
      checks.push(body);
      // "pie game" was misheard; its check takes 1 s (the turn ends first), and one of words that begin "wait" 2 s
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
    // what the user said, without the app's note to the model before it (voice mode's)
    const said = (typeof last.content === "string" ? last.content : last.content.map((p: any) => p.text ?? "").join("")).replace(/^\[[^\]]*\]\n\n/, "");
    const round = body.messages.slice(at).filter((m: any) => m.role === "assistant" && m !== prefix).length, ask = TOOL_ASKS[said];
    const call = ask && body.tools && round === 0 ? ask : undefined;
    const text = call ? "Let me check." : ask ? `The tool said: ${body.messages.at(-1).content}` : said.startsWith("slow")
      ? Array.from({ length: 40 }, (_, k) => `w${k} `).join("") : Object.entries(REPLIES).find(([k]) => said.includes(k))?.[1] ?? `You said: ${said}`;
    const step = said.startsWith("slow") ? 4 : 9, gap = said.startsWith("slow") ? 60 : 5;
    let i = 0, restated = !prefix, read = false;
    return new Response(new ReadableStream({
      async pull(ctl) {
        await Bun.sleep(gap);
        // a continued reply restates itself first, as Chrysalis does, and the prompt's length comes once it is read
        if (!restated) { restated = true; return ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { chrysalis_prefix: { content: prefix.content } } }] })}\n\n`)); }
        if (!read) { read = true; return ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: {} }], chrysalis_prompt: { prompt_tokens: 10, cached_tokens: 0 } })}\n\n`)); }
        if (i < text.length) { ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: text.slice(i, i + step) } }] })}\n\n`)); i += step; return; }
        if (call) {
          ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: `call_${++callIds}`, type: "function", function: { name: call.name, arguments: call.args } }] } }] })}\n\n`));
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
const dir = mkdtempSync(`${tmpdir()}/chrysalis-web-`), LOG = `${dir}/mcp.log`, VOICE_LOG = `${dir}/voice.log`;
writeFileSync(`${dir}/mcp.json`, JSON.stringify({ mcpServers: {
  fake: { command: process.execPath, args: ["fake-mcp.ts"], cwd: import.meta.dir, env: { FAKE_MCP_LOG: LOG } },
  broken: { command: "/nonexistent/chrysalis-test-mcp" },
} }));
beforeAll(async () => {
  web = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", CHRYSALIS_URL: `http://127.0.0.1:${fake.port}`, CHRYSALIS_HOST: "127.0.0.1", NODE_ENV: "production", CHRYSALIS_MCP_CONFIG: `${dir}/mcp.json`,
      CHRYSALIS_VOICE_CMD: JSON.stringify([process.execPath, `${import.meta.dir}/fake-voice.ts`]), FAKE_VOICE_LOG: VOICE_LOG, CHRYSALIS_VOICE_IDLE_MS: "1500" },
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
    args: ["--no-sandbox", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"] });
}, 30_000);
afterAll(async () => { await browser?.close(); web.kill(); fake.stop(true); });

// Each page in a browser context of its own: what one test leaves in localStorage (the last chat's tools) stays there.
async function open(): Promise<Page> {
  const page = await (await browser.createBrowserContext()).newPage();
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
  expect([early > 10, later > early]).toEqual([true, true]);    // the fake's prompt is 10 tokens
  await page.waitForSelector(".msg.assistant .stats");
  expect(await used()).toBe(30);                                // its usage: 10 read, 20 written
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
  await page.click("#export");
  const file = await (async () => { for (let k = 0; k < 50; k++) { const f = readdirSync(dir).find((x) => x.endsWith(".chrysalis.json")); if (f) return f; await Bun.sleep(100); } })();
  const other = await open();                            // another tab: its own conversations and files
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
  await page.click("#export");
  for (let k = 0; k < 50; k++) { const f = readdirSync(into).find((x) => x.endsWith(".chrysalis.json")); if (f) return `${into}/${f}`; await Bun.sleep(100); }
  throw new Error("no export");
}

test("the Tools popover turns servers and single tools on and off for this chat, and each chat keeps its own", async () => {
  const page = await open();
  expect(await badge(page)).toEqual(["", "Tools: off in this chat"]);
  await page.click("#tools-btn");
  await page.waitForSelector(sw("mcp:fake"));
  const rows = await page.$$eval(`${pop} .tp-src`, (r) => r.map((x) => [x.querySelector(".tp-name")!.textContent, x.querySelector(".tp-sub")!.textContent]));
  expect(rows).toEqual([["Fetch URL", "Reads web pages at public addresses"], ["fake", "Not started"], ["broken", "Not started"]]);   // as the config file lists them
  await page.click(sw("mcp:fake"));                                       // starts the server, every tool on
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
  await page.click(sw("mcp:broken"));                                     // a server that cannot start says why, with Retry
  await page.waitForSelector(`${pop} .tp-err [data-do="retry"][data-src="mcp:broken"]`, { timeout: 20_000 });
  expect(await page.$eval(`${pop} .tp-err`, (e) => e.textContent)).toMatch(/ENOENT|nonexistent|not found/i);
  await page.click(sw("mcp:broken"));
  await page.keyboard.press("Escape");
  expect(await page.$eval(pop, (p) => (p as HTMLElement).hidden)).toBe(true);
  expect(await page.evaluate(() => document.activeElement?.id)).toBe("tools-btn");

  await send(page, "hi");
  expect(lastStream.tools.map((t: any) => t.function.name)).toEqual(["fake__big", "fake__echo", "fake__get_weather", "fake__slow", "fetch_url"]);
  expect(lastStream.messages[0].content).toBe("You are Qwen3.8-27B, running locally in Chrysalis Inference. The tools offered here are the only ones in this chat, " +
    "and there are no files or projects to explore. Use a tool when it helps; otherwise answer directly.");
  const first = await page.evaluate(() => location.hash);

  await page.click("#new-chat");                                          // a new chat starts from the last choice
  expect(await badge(page)).toEqual(["5", "Tools: 5 on in this chat"]);
  await page.click("#tools-btn");
  await page.waitForSelector(sw("fetch"));
  await page.click(sw("fetch"));
  await page.click(sw("mcp:fake"));
  await page.keyboard.press("Escape");
  expect(await badge(page)).toEqual(["", "Tools: off in this chat"]);
  await send(page, "no tools");
  expect(lastStream.tools).toBeUndefined();
  expect(lastStream.messages[0].content).toBe("You are Qwen3.8-27B, running locally in Chrysalis Inference. There are no tools or files in this chat. Answer directly.");

  await page.goto(`${base}/${first}`, { waitUntil: "networkidle0" });   // the first chat, after a reload: its own choice
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

  const round1 = streams.at(-1);                                          // what the tool's round sent Chrysalis
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
  await send(page, "tool note");                                          // allowed now: it runs without asking
  expect(await page.$$eval(".msg.assistant .tool-card", (c) => c.at(-1)!.getAttribute("data-state"))).toBe("done");
  expect(readFileSync(LOG, "utf8").match(/wrote n1/g)?.length).toBe(3);
  await page.keyboard.down("Control"); await page.keyboard.press(","); await page.keyboard.up("Control");
  await page.waitForSelector("#allow-list .chip");
  expect(await page.$$eval("#allow-list .chip", (c) => c.map((x) => x.textContent))).toEqual(["fake__write_note"]);
  await page.close();
}, 90_000);

const articles = (page: Page) => page.$$eval("#thread > article", (a) => a.map((x) => (x.querySelector(".bubble, .markdown") as HTMLElement)?.innerText.trim()));
const pagerOf = (page: Page, i: number) => page.$eval(`#thread > article[data-i="${i}"]`, (a) => a.querySelector(".pager-n")?.textContent ?? "");
async function untilIdle(page: Page) {
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send", { timeout: 20_000 });
}

test("regenerate and edit make versions; the pager switches them and shows each one's own continuation", async () => {
  const page = await open();
  await send(page, "one");
  await send(page, "two");
  expect(await pagerOf(page, 1)).toBe("");                                // one version: no pager
  await page.hover('#thread > article[data-i="1"]');
  await page.click('#thread > article[data-i="1"] [data-act="retry"]');   // a new version of the first reply
  await untilIdle(page);
  expect(await articles(page)).toEqual(["one", "You said: one"]);
  expect(await pagerOf(page, 1)).toBe("2 / 2");
  await page.click('#thread > article[data-i="1"] [data-act="prev"]');
  expect(await articles(page)).toEqual(["one", "You said: one", "two", "You said: two"]);
  expect(await pagerOf(page, 1)).toBe("1 / 2");
  expect(await page.evaluate(() => (document.activeElement as HTMLElement)?.dataset.act)).toBe("next");   // the focus stays on the pager

  await page.hover('#thread > article[data-i="0"]');
  await page.click('#thread > article[data-i="0"] [data-act="edit"]');
  await page.waitForSelector(".editing textarea");
  await page.$eval(".editing textarea", (t) => { (t as HTMLTextAreaElement).value = ""; });
  await page.type(".editing textarea", "uno");
  await page.keyboard.down("Control"); await page.keyboard.press("Enter"); await page.keyboard.up("Control");
  await untilIdle(page);
  expect(await articles(page)).toEqual(["uno", "You said: uno"]);
  expect(await pagerOf(page, 0)).toBe("2 / 2");
  await page.focus('#thread > article[data-i="0"] [data-act="prev"]');    // by keyboard
  await page.keyboard.press("Enter");
  expect(await articles(page)).toEqual(["one", "You said: one", "two", "You said: two"]);
  expect([await pagerOf(page, 0), await pagerOf(page, 1)]).toEqual(["1 / 2", "1 / 2"]);

  await page.reload({ waitUntil: "networkidle0" });                       // the tree and what is shown are kept
  expect(await articles(page)).toEqual(["one", "You said: one", "two", "You said: two"]);
  await send(page, "three");                                              // only the path on screen goes to Chrysalis
  expect(lastStream.messages.filter((m: any) => m.role !== "system").map((m: any) => m.content))
    .toEqual(["one", "You said: one", "two", "You said: two", "three"]);

  await page.hover('#thread > article[data-i="0"]');                      // deleting a version shows the next one
  await page.click('#thread > article[data-i="0"] [data-act="trash"]');
  expect(await articles(page)).toEqual(["uno", "You said: uno"]);
  expect(await pagerOf(page, 0)).toBe("");
  await page.click("#toast .btn");                                        // Undo
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
  await page.click('#thread > article[data-i="1"] [data-act="prev"]');    // away from it while it is written
  expect(await reply()).toBe(whole);
  await Bun.sleep(1200);
  expect(await reply()).toBe(whole);
  await page.click('#thread > article[data-i="1"] [data-act="next"]');    // back: it has gone on meanwhile
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

  const old = `${into}/old.json`;                                         // from before versions and tools
  writeFileSync(old, JSON.stringify({ chrysalis: "chats", version: 1, exported: "2026-01-01T00:00:00.000Z", conversations: [{ id: "x", title: "Old", updated: 1, effort: "low",
    messages: [{ role: "user", content: "old q" }, { role: "assistant", content: "old a", reasoning: "hm", stats: { ttft: 1, tps: 2, tokens: 3, prompt: 4, cached: 0 } }] }] }));
  const third = await open();
  await (await third.$("input#import-file"))!.uploadFile(old);
  await third.waitForFunction(() => document.querySelectorAll("#thread > article").length === 2);
  expect(await third.$$eval(".pager", (p) => p.length)).toBe(0);
  await send(third, "next");
  expect(lastStream.messages).toEqual([{ role: "system", content: "You are Qwen3.8-27B, running locally in Chrysalis Inference. There are no tools or files in this chat. Answer directly." }, { role: "user", content: "old q" },
    { role: "assistant", content: "old a", reasoning_content: "hm" }, { role: "user", content: "next" }]);
  for (const p of [page, other, third]) await p.close();
}, 90_000);

// Voice mode. Each page asks the fake speech service for a voice whose script it plays (fake-voice.ts); its log has
// what the page sent it, session by session.
// the voice note as Settings shows it
const noteShown = async (page: Page) => (await openSettings(page), page.$eval("#voice-prompt", (e) => (e as HTMLTextAreaElement).value));
const sent = (b: any) => b.messages;
const voiceLog = (): any[] => { try { return readFileSync(VOICE_LOG, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l)); } catch { return []; } };
const voicePages: Page[] = [];
async function voicePage(script: string, o: { system?: string; first?: (page: Page) => Promise<void> } = {}) {
  const page = await (await browser.createBrowserContext()).newPage();
  voicePages.push(page);
  await page.setViewport({ width: 1600, height: 900 });            // Settings beside the chat (from 1460 px), so it stays open while the dock is used
  page.on("pageerror", (e) => { throw e; });
  await page.evaluateOnNewDocument((v, system) => {
    if (!sessionStorage.getItem("settings2")) sessionStorage.setItem("settings2", JSON.stringify({ system, think: {}, fast: {}, voice: { name: v } }));
  }, script, o.system ?? "");
  await page.goto(`${base}/`, { waitUntil: "networkidle0" });
  await o.first?.(page);
  await page.click("#voice");
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 10_000 });
  const s = voiceLog().filter((e) => e.ev === "msg" && e.type === "hello" && e.voice === script).at(-1).s;
  return { page, session: () => voiceLog().filter((e) => e.s === s) };
}
const replyText = (page: Page, i: number) => page.$eval(`#thread > article[data-i="${i}"] .markdown`, (el) => el.textContent?.trim() ?? "");

test("voice mode: what is said shows as it is said, becomes the question, and the reply is spoken as it is written", async () => {
  const n0 = streams.length, c0 = checks.length;
  const { page, session } = await voicePage("plain");
  // the message box stays (it can still take typing), with the pill in its row and Send become "End voice mode"
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
  expect(sent(lastStream)).toEqual([{ role: "user", content: `[${note}]\n\nhello there` }]);   // the note, before what was said
  // what was heard was checked first, at temperature 0 in a conversation of its own, and nothing of it is in the reply's
  expect(lastStream.continue_final_message).toBeUndefined();
  const check = checks.findLast((b) => b.messages.at(-1).content === "Transcript: hello there");
  expect([check?.temperature, check?.stream, check?.model, check?.messages.length]).toEqual([0, undefined, "qwen3.8-27b-fast", 2]);
  // the words checked are those speech recognition settled ("hello" was a look's, as it was said)
  expect(checks.slice(c0).map((b) => b.messages.at(-1).content)).toEqual(["Transcript: hello there"]);
  // it thinks as the chat's bar says (a new chat's: Max)
  expect(await page.$eval("#energy", (e) => e.getAttribute("aria-valuetext"))).toBe("Max");
  expect([lastStream.model, lastStream.reasoning_effort]).toEqual(["qwen3.8-27b", "xhigh"]);
  // opening voice mode had Chrysalis read the chat first (a new one's question stands in), so the first turn is an
  // append
  const mine = streams.slice(n0), warm = mine.findIndex((b) => b.max_tokens === 1);
  expect(mine[warm]?.messages).toEqual([{ role: "user", content: "Hi" }]);
  expect(mine[warm]?.model).toBe(lastStream.model);                 // as the reply it reads ahead for
  expect(warm).toBeLessThan(mine.findIndex((b) => sent(b).at(-1)?.content.endsWith("hello there")));
  expect(await replyText(page, 1)).toBe("You said: hello there");
  expect(await page.$eval('#thread > article[data-i="0"] .msg-note summary', (e) => e.textContent)).toBe("Voice mode note");
  expect(await page.$eval('#thread > article[data-i="0"] .msg-note-text', (e) => e.checkVisibility())).toBe(false);   // closed until asked
  await page.click('#thread > article[data-i="0"] .msg-note summary');
  expect(await page.$eval('#thread > article[data-i="0"] .msg-note-text', (e) => [e.textContent, e.checkVisibility()])).toEqual([note, true]);
  const log = session(), says = log.filter((e) => e.type === "say"), id = says[0].id;
  expect(says.every((e) => e.id === id)).toBe(true);
  expect(says.map((e) => e.text).join("")).toBe("You said: hello there");
  expect(log.findIndex((e) => e.type === "say_end" && e.id === id)).toBeGreaterThan(log.findLastIndex((e) => e.type === "say"));
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 10_000 });
  expect(session().filter((e) => e.type === "playback").map((e) => e.playing)).toEqual([true, false]);
  // speech recognition is told the conversation, for its names: the model's, then the chat as it grows
  const told = session().filter((e) => e.type === "context").map((e) => e.text);
  expect(told[0]).toMatch(/^Qwen3\.8-27B\n/);
  expect(told.at(-1)).toMatch(/hello there\nYou said: hello there$/);
  await page.close();
}, 30_000);

test("voice mode: talking over a reply silences it, stops Chrysalis's request, and the chat keeps only what was heard", async () => {
  const stopped = chrysalisStopped, { page, session } = await voicePage("story");
  await page.waitForFunction(() => document.querySelector('#thread > article[data-i="0"] .bubble')?.textContent === "slow story", { timeout: 10_000 });
  for (const t = Date.now(); !session().some((e) => e.ev === "interrupt") && Date.now() - t < 10_000; ) await Bun.sleep(10);
  const talked = Date.now();                       // Chrysalis's request stops with the speech, not when the next words come in
  for (; chrysalisStopped === stopped && Date.now() - talked < 3000; ) await Bun.sleep(10);
  expect(chrysalisStopped).toBe(stopped + 1);
  expect(Date.now() - talked).toBeLessThan(500);
  await page.waitForFunction(() => /interrupted/.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  const kept = await replyText(page, 1), chunks = session().filter((e) => e.ev === "chunk"), full = Array.from({ length: 40 }, (_, k) => `w${k}`).join(" ");
  expect(kept.length).toBeGreaterThan(0);
  expect(kept.length).toBeLessThan(full.length);
  expect(full.startsWith(kept)).toBe(true);
  // cut at the word it was on (whole words), within the pieces sent before the user talked
  const before = chunks.map((c) => c.text).join("").trimEnd(), ends = chunks.map((c) => chunks.filter((x) => x.to <= c.to).map((x) => x.text).join("").trimEnd());
  expect([full[kept.length] === " ", kept.length <= before.length, ends.includes(kept)]).toEqual([true, true, false]);   // not at a piece's end
  await page.waitForFunction(() => document.querySelector('#thread > article[data-i="2"] .bubble')?.textContent === "and then", { timeout: 10_000 });
  await page.waitForFunction(() => /voice/.test(document.querySelector('#thread > article[data-i="3"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  // the kept reply as Chrysalis is sent it again: only what was heard
  expect(sent(lastStream).filter((m: any) => m.role === "assistant").map((m: any) => m.content)).toEqual([`${kept} — [cut off: the user talked over this reply]`]);
  // the next question's check has the one before it and what the reply said of it
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
  await Bun.sleep(500);                                           // into its first pieces, past the output's latency
  const a = await heard();
  await Bun.sleep(700);
  const b = await heard();
  expect(a.grey && b.grey && a.said && b.said).toBeTruthy();
  expect([b.all.startsWith(a.said!), b.all.startsWith(b.said!), b.said!.length > a.said!.length]).toEqual([true, true, true]);
  expect(/^(w\d+ )*w\d+$/.test(b.said!)).toBe(true);                         // whole words
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 15_000 });
  expect(await page.evaluate(() => (CSS as any).highlights.has("voice-unspoken"))).toBe(false);   // all of it heard
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
  await page.$eval("#knob-voice_speed", (el) => { (el as HTMLInputElement).value = "1.2"; el.dispatchEvent(new Event("change")); });   // the script's next line waits for a config
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="7"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  await page.click("#send");
  await send(page, "typed again");
  const asked = streams.slice(n0).filter((b) => b.max_tokens !== 1);   // not the warm-up when voice mode opened
  expect(asked.map((b) => sent(b).at(-1).content)).toEqual(["typed first", `[${note}]\n\nfirst question`, "second question",
    `[${CHANGED}]\n\nthird question`, "[The user is typing again and reads your replies.]\n\ntyped again"]);
  expect(asked.every((b) => JSON.stringify(b.messages[0]) === JSON.stringify({ role: "system", content: "Be brief." }))).toBe(true);
  expect(session().some((e) => e.type === "config")).toBe(true);
  expect(await page.$$eval(".msg-note summary", (els) => els.map((e) => e.textContent))).toEqual(["Voice mode note", "Voice mode note", "Note to the model"]);
  await page.close();
}, 45_000);

test("What the model is sent shows the next request as Chrysalis gets it: the tools as JSON, the one system prompt, every message; an edited prompt goes as written", async () => {
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
  await openSettings(page);
  await page.click("#sent-open");
  await page.waitForFunction(() => (document.querySelector("#sent-view") as HTMLDialogElement).open && !!document.querySelector("#sent-text")?.textContent);
  const shown = await page.$eval("#sent-text", (e) => e.textContent!);
  await page.keyboard.press("Escape");
  await send(page, "second");
  const next = lastStream;                                       // what Chrysalis got next: all of the view, then the new question
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
  expect(lastStream.messages[0]).toEqual({ role: "system", content: "Be terse, Qwen3.8-27B." });
  await page.close();
}, 60_000);

test("the theme is the system's until Settings chooses one, which overrides it and is kept across a reload; System follows the system's scheme as it changes", async () => {
  const page = await open();
  // the page's background, by its luminance, once it has settled (a change of the system's scheme reaches the page on
  // its next frame): dark under 60, light over 200
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
  expect(await bg("dark")).toBeLessThan(60);                     // chosen over the system's light
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

async function openSettings(page: Page) {
  await page.click("#settings-toggle");
  await page.waitForFunction(() => { const r = document.querySelector("#voice-knobs")!.getBoundingClientRect(); return r.width > 0 && r.right <= innerWidth; });
}
const stored = (page: Page) => page.evaluate(() => JSON.parse(sessionStorage.getItem("settings2") ?? "{}").voice);
const settled = async (ok: () => boolean, ms = 5000) => { for (const t = Date.now(); !ok() && Date.now() - t < ms; ) await Bun.sleep(25); return ok(); };

test("voice picker: the voice models' voices by language and gender; a pick switches model and voice for the next sentence, and Settings shows the same choice", async () => {
  const { page, session } = await voicePage("pick");
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  expect(new Set(session().filter((e) => e.ev === "chunk").map((e) => `${e.tts}/${e.voice}`))).toEqual(new Set(["fake-tts/pick"]));
  expect(await page.$eval("#voice-pick .voice-pick-name", (el) => el.textContent)).toBe("Pick");
  await page.click("#voice-pick");
  const rows = await page.$$eval("#voice-list > *", (els) => els.map((e) => [e.className, e.textContent]));
  expect(rows.filter(([c]) => c.startsWith("voice-model"))).toEqual([["voice-model", "Fake TTS"], ["voice-model", "Other TTS"], ["voice-model missing", "Big TTStools/voice_setup.sh big-tts"]]);
  expect(rows.filter(([c]) => c === "voice-group").map(([, t]) => t)).toEqual(["American English · Female", "British English · Male", "Spanish · Female", "American English · Male"]);
  expect(await page.$$eval("#voice-pop .voice-item", (els) => els.map((e) => (e as HTMLElement).dataset.voice))).toEqual(
    ["plain", "pie", "story", "long", "merge", "join", "ahead", "onward", "checking", "silent", "pick", "notes", "echo"].map((v) => `fake-tts/${v}`).concat(["other-tts/ana", "other-tts/bob"]));
  expect(await page.$$eval('#voice-pop .voice-item[aria-selected="true"]', (els) => els.map((e) => (e as HTMLElement).dataset.voice))).toEqual(["fake-tts/pick"]);
  await page.click('#voice-pop [data-voice="other-tts/bob"]');
  const configs = () => session().filter((e) => e.type === "config").map(({ stack, voice, speed }) => ({ stack, voice, speed }));
  expect(await settled(() => configs().length === 1)).toBe(true);
  expect(configs()).toEqual([{ stack: { tts: "other-tts" }, voice: "bob", speed: undefined }]);
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="3"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  const log = session(), after = log.slice(log.findIndex((e) => e.type === "config")).filter((e) => e.ev === "chunk");
  expect(after.length).toBeGreaterThan(0);
  expect(new Set(after.map((e) => `${e.tts}/${e.voice}`))).toEqual(new Set(["other-tts/bob"]));
  // one value, two views: the dock's picker and Settings show it, and Settings changing it is the dock's change too
  expect(await page.$eval("#voice-pick .voice-pick-name", (el) => el.textContent)).toBe("Bob");
  expect(await stored(page)).toEqual({ tts: "other-tts", name: "bob" });
  await openSettings(page);
  await page.waitForFunction(() => (document.querySelector("#voice-name") as HTMLSelectElement)?.value === JSON.stringify(["other-tts", "bob"]));
  await page.select("#voice-name", JSON.stringify(["other-tts", "ana"]));
  await page.waitForFunction(() => document.querySelector("#voice-pick .voice-pick-name")?.textContent === "Ana");
  expect(await settled(() => configs().length === 2)).toBe(true);
  expect(configs()[1]).toEqual({ stack: undefined, voice: "ana", speed: undefined });
  // a pick from the dock is undone like any setting
  await page.click("#voice-pick");
  await page.click('#voice-pop [data-voice="fake-tts/echo"]');
  await page.waitForFunction(() => (document.querySelector("#voice-name") as HTMLSelectElement)?.value === JSON.stringify(["fake-tts", "echo"]));
  await page.click("#toast .btn");
  await page.waitForFunction(() => document.querySelector("#voice-pick .voice-pick-name")?.textContent === "Ana" && (document.querySelector("#voice-name") as HTMLSelectElement)?.value === JSON.stringify(["other-tts", "ana"]));
  expect(await settled(() => configs().length === 4)).toBe(true);
  expect(configs().slice(2)).toEqual([{ stack: { tts: "fake-tts" }, voice: "echo", speed: undefined }, { stack: { tts: "other-tts" }, voice: "ana", speed: undefined }]);
  expect(await stored(page)).toEqual({ tts: "other-tts", name: "ana" });
  await page.close();
}, 40_000);

test("voice picker: a voice's play button plays its sample, apart from any reply or turn", async () => {
  const { page, session } = await voicePage("silent");
  await page.click("#voice-pick");
  await page.click('#voice-pop .voice-row:has([data-voice="other-tts/ana"]) .voice-play');
  expect(await settled(() => session().filter((e) => e.type === "playback").length === 2)).toBe(true);
  const log = session();
  expect(log.filter((e) => e.type === "preview").map((e) => e.voice)).toEqual(["ana"]);
  expect(log.filter((e) => e.ev === "chunk").map((e) => [e.id, e.voice, e.text])).toEqual([["preview", "ana", "This is Ana."]]);
  expect(log.filter((e) => e.type === "playback").map((e) => e.playing)).toEqual([true, false]);
  expect(log.some((e) => e.type === "say" || e.type === "config")).toBe(false);
  expect(await page.$$eval("#thread > article", (els) => els.length)).toBe(0);
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 3000 });   // (drawn on the next frame)
  expect(await page.$eval("#voice-pick .voice-pick-name", (el) => el.textContent)).toBe("Silent");
  expect(await page.$eval("#voice-pop", (el) => (el as HTMLElement).hidden)).toBe(false);
  await page.close();
}, 30_000);

test("voice mode: mute stops the microphone's audio; ending it closes the socket, and the service stops once idle", async () => {
  for (const p of voicePages.splice(0)) if (!p.isClosed()) await p.close();   // earlier tests' voice sockets, open if they failed
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

test("voice mode on a page that is not secure (plain HTTP at this machine's address) says where the secure page is, and starts nothing", async () => {
  const ip = lanAddresses()[0], log = `${dir}/voice-insecure.log`;
  expect(ip).toBeDefined();
  const insecure = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", CHRYSALIS_URL: `http://127.0.0.1:${fake.port}`, CHRYSALIS_HOST: ip, NODE_ENV: "production", CHRYSALIS_MCP_CONFIG: `${dir}/mcp.json`,
      CHRYSALIS_VOICE_CMD: JSON.stringify([process.execPath, `${import.meta.dir}/fake-voice.ts`]), FAKE_VOICE_LOG: log },
  });
  try {
    let out = "";
    for (const reader = insecure.stdout.getReader(), dec = new TextDecoder(); !/chrysalis web: \S+/.test(out); ) {
      const { value, done } = await reader.read();
      if (done) throw new Error(`server.ts exited: ${out}`);
      out += dec.decode(value);
    }
    const url = out.match(/chrysalis web: (\S+)/)![1].replace(/\/$/, "");
    const page = await (await browser.createBrowserContext()).newPage();
    await page.goto(`${url}/`, { waitUntil: "networkidle0" });
    expect(await page.evaluate(() => isSecureContext)).toBe(false);
    await page.click("#voice");
    await page.waitForFunction(() => /secure page/.test(document.querySelector("#voice-status")?.textContent ?? ""), { timeout: 5000 });
    expect(await page.$eval("#voice-status", (e) => e.textContent)).toContain(`https://${new URL(url).host}`);
    await Bun.sleep(300);
    expect(existsSync(log)).toBe(false);                          // no socket, so no service
    await page.close();
  } finally {
    insecure.kill();
  }
});

// The service's log of a session: the replies the page asked it to speak (by id, their text) and the ones it cancelled.
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
  // the reply begun at the turn, on the words as heard: its speech cancelled, unheard (the page held it)
  const replies = streams.slice(n0).filter((b) => b.max_tokens !== 1).map((b) => sent(b).at(-1).content.split("\n\n").at(-1));
  expect(replies).toEqual(["lets do a snake game in pie game", "lets do a snake game in Pygame"]);
  expect(checks.slice(c0).map((b) => b.messages.at(-1).content)).toEqual(["Transcript: lets do a snake game in pie game"]);   // once
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
  // the replies begun before the question stood were cancelled unheard
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
  expect(mine[0][1]).toBeLessThan(turned);                           // Chrysalis was asked before the turn ended
  // the reply was shown and heard only after: its first playback after the turn, and its text all spoken by then
  const played = log.findIndex((e) => e.type === "playback" && e.playing), turnIx = log.findIndex((e) => e.ev === "sent" && e.type === "turn");
  expect(played).toBeGreaterThan(turnIx);
  expect(log.slice(0, turnIx).some((e) => e.ev === "chunk")).toBe(true);   // spoken ahead, held by the page
  expect(await replyText(page, 1)).toBe("Paris.");
  expect(await page.$$eval("#thread > article.msg", (els) => els.length)).toBe(2);
  await page.close();
}, 30_000);

test("voice mode: a reply begun at a pause is dropped unheard when the user goes on, and the whole question is answered", async () => {
  const n0 = streams.length, stopped = chrysalisStopped, { page, session } = await voicePage("onward");
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  const replies = streams.slice(n0).filter((b) => b.max_tokens !== 1).map((b) => sent(b).at(-1).content.split("\n\n").at(-1));
  expect(replies).toEqual(["slow what is the", "slow what is the capital of France"]);
  expect(chrysalisStopped).toBe(stopped + 1);                            // still being written (slowly) when the user went on
  expect(await page.$eval('#thread > article[data-i="0"] .bubble', (e) => e.textContent)).toBe("slow what is the capital of France");
  expect(await page.$$eval("#thread > article.msg", (els) => els.length)).toBe(2);
  // the pause's reply was spoken into the page's hold and cancelled there; the question's is the one heard
  const log = session(), { said, cancelled } = spokenOf(log);
  expect([...said.keys()].map((id) => cancelled.has(id))).toEqual([true, false]);
  // dropped as the user went on, not at the turn
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
  const page = await (await browser.createBrowserContext()).newPage();
  await page.evaluateOnNewDocument(() => {
    sessionStorage.setItem("settings2", JSON.stringify({ system: "", think: {}, fast: {}, voice: { name: "silent" } }));
    const ask = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);   // a phone's microphone, slow to open
    navigator.mediaDevices.getUserMedia = async (c) => { await new Promise((r) => setTimeout(r, 1500)); return ask(c); };
  });
  await page.goto(`${base}/`, { waitUntil: "networkidle0" });
  const hellos = () => voiceLog().filter((e) => e.ev === "msg" && e.type === "hello").length, n0 = hellos();
  await page.click("#voice");
  for (const t = Date.now(); hellos() === n0 && Date.now() - t < 1000; ) await Bun.sleep(20);
  await Bun.sleep(300);                                          // the service is ready (the fake loads nothing); the microphone is not
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
  expect(await page.$(".voice-think")).toBeNull();                // no thinking of voice mode's own
  await page.click("#send");                                    // ■: ends voice mode
  expect(await bar()).toEqual(["Medium", "Thinking: Medium"]);
  await page.close();
}, 30_000);

// A part of the page's screenshot, counted: its pink pixels (the butterfly is drawn in the accent, the sea in a grey),
// its faintly pink ones (any pink tint, however dim), its lit ones (anything drawn on the dark page), those close to
// `like` (a CSS color) and those that differ from `before` (an earlier shot of the same part). The PNGs are decoded by
// the page itself (an image drawn into a 2D canvas), so the test needs no image library.
async function look(page: Page, clip: { x: number; y: number; width: number; height: number }, before = "", like = "") {
  // (not beyond the viewport: that resizes the window for the shot, which ends a panel's slide at once)
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
      if (d[i] + d[i + 1] + d[i + 2] > 100) lit++;                   // the page is #121010
      if (like && Math.abs(d[i] - lr) + Math.abs(d[i + 1] - lg) + Math.abs(d[i + 2] - lb) < 60) close++;
      if (b && Math.abs(d[i] - b[i]) + Math.abs(d[i + 1] - b[i + 1]) + Math.abs(d[i + 2] - b[i + 2]) > 30) changed++;
    }
    return { pink, tint, lit, close, changed };
  }, png, before, like);
  return { png, ...counts };
}

// The sea needs WebGL2, which headless Chromium here gets only from SwiftShader on request: a browser of its own, so the
// other tests run without its frames (costly when rendered in software) and with the plain logo in a new chat.
test("a new chat is the butterfly in a sea under the message box too, never blanked by a side panel sliding shut; a click on the water makes ripples and keeps the message box focused, and with reduced motion draws nothing", async () => {
  const gl = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: ["--no-sandbox", "--enable-unsafe-swiftshader"] });
  try {
    for (const motion of ["no-preference", "reduce"] as const) {
      const page = await (await gl.createBrowserContext()).newPage();     // its own storage: the panel starts open
      page.on("pageerror", (e) => { throw e; });
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

      // the conversations panel closes: slowed down, so that shots land while the chat widens (reduced motion has no slide)
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
      // the sea beside the message box, where a click reaches the water
      const beside = { x: sea.x + 40, y: box.y, width: box.x - sea.x - 80, height: box.height };
      expect((await look(page, beside)).lit).toBeGreaterThan(150);
      expect(await page.evaluate((x, y) => document.elementFromPoint(x, y)?.className, beside.x + 20, box.y + box.height / 2)).toBe("ocean");

      // a click in the open water left of the butterfly, against the same stretch of water on the right: the current
      // drifts in both, the ripples only near the click. The pointer rests there first, so its light is already lit.
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
      else expect(near1.changed).toBeGreaterThan(2 * far1.changed + 500);
      await page.close();
    }
  } finally {
    await gl.close();
  }
}, 90_000);

test("the sea stays behind a chat, the butterfly sunk into it and every click the chat's, and a new chat raises it again; Settings takes the sea away from chats (remembered) and leaves it in a new chat; it follows the theme", async () => {
  const gl = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: ["--no-sandbox", "--enable-unsafe-swiftshader"] });
  try {
    const page = await (await gl.createBrowserContext()).newPage();
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
    await Bun.sleep(1500);                                         // the butterfly sinks into the water
    const column = await rect("#thread"), margin = { x: sea.x + 40, y: sea.y + 60, width: column.x - sea.x - 80, height: 300 };
    const sunk = await look(page, middle);
    expect(sunk.pink).toBeLessThan(50);                            // its bold glyphs gone,
    expect(sunk.tint).toBeGreaterThan(300);                        // its shape faintly pink in the water
    expect((await look(page, margin)).lit).toBeGreaterThan(150);
    const hit = (x: number, y: number) => page.evaluate((x, y) => document.elementFromPoint(x, y)?.closest("#scroller, .sea")?.id, x, y);
    const reply = await rect(".msg.assistant .markdown");
    expect([await hit(reply.x + 5, reply.y + reply.height / 2), await hit(margin.x + 20, margin.y + 20)]).toEqual(["scroller", "scroller"]);

    // a new chat raises it out of the water: part of the way in the first shot after the click, all of it soon after
    await page.click("#new-chat");
    const rising = (await look(page, middle)).pink;
    expect([rising < 0.7 * full, await risen(0.9 * full) > 0.9 * full]).toEqual([true, true]);

    await page.click(".conv");
    await page.click("#settings-toggle");                         // (the chat narrows while Settings is open)
    await page.click("#sea-switch");
    await page.click("#settings-toggle");
    await page.waitForFunction((x) => document.getElementById("thread")!.getBoundingClientRect().x === x, {}, column.x);
    expect((await look(page, margin)).lit).toBe(0);
    await page.click("#new-chat");
    expect(await risen()).toBeGreaterThan(800);
    await page.reload({ waitUntil: "networkidle0" });
    expect(await page.$eval("#sea-switch", (b) => b.getAttribute("aria-checked"))).toBe("false");

    // the light theme: the butterfly's bright cores in the light theme's own (a deep pink where the dark's is near white)
    const cores = async () => (await look(page, middle, "", "var(--accent-hi)")).close;
    await openSettings(page);
    await page.click('#theme-pick [data-theme="light"]');
    await page.click("#settings-toggle");
    await Bun.sleep(600);
    expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe("light");
    expect(await cores()).toBeGreaterThan(100);
    await page.close();
  } finally {
    await gl.close();
  }
}, 60_000);
