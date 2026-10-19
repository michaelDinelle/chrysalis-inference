// The page in a real browser (headless Chromium) against a fake kiln that streams canned replies: TeX typeset and
// prices left alone, Mermaid diagrams drawn, a reply's HTML page run in its sandbox where it reaches nothing, attached
// files (pictures, PDFs, videos, texts) sent as kiln expects them and kept across a reload, chats carried through
// export and import, versions of messages and replies, and tools (an MCP server, fake-mcp.ts, that server.ts runs).
import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from "node:fs";
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

// Questions that make the fake kiln call a tool (in its first round, then answer once the result is in), or write slowly.
const TOOL_ASKS: Record<string, { name: string; args: string }> = {
  "tool echo": { name: "fake__echo", args: '{"text":  "hello"}' },
  "tool note": { name: "fake__write_note", args: '{"text": "n1"}' },
};
let lastStream: any = null, probes = 0, callIds = 0;  // probes: requests a sandboxed page got through to a server
const streams: any[] = [];                           // every streamed request kiln got
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
    streams.push(body);
    const at = body.messages.findLastIndex((m: any) => m.role === "user"), last = body.messages[at];
    const said = typeof last.content === "string" ? last.content : last.content.map((p: any) => p.text ?? "").join("");
    const round = body.messages.slice(at).filter((m: any) => m.role === "assistant").length, ask = TOOL_ASKS[said];
    const call = ask && body.tools && round === 0 ? ask : undefined;
    const text = call ? "Let me check." : ask ? `The tool said: ${body.messages.at(-1).content}` : said.startsWith("slow")
      ? Array.from({ length: 40 }, (_, k) => `w${k} `).join("") : Object.entries(REPLIES).find(([k]) => said.includes(k))?.[1] ?? `You said: ${said}`;
    const step = said.startsWith("slow") ? 4 : 9, gap = said.startsWith("slow") ? 60 : 5;
    let i = 0;
    return new Response(new ReadableStream({
      async pull(ctl) {
        await Bun.sleep(gap);
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
const dir = mkdtempSync(`${tmpdir()}/kiln-web-`), LOG = `${dir}/mcp.log`;
writeFileSync(`${dir}/mcp.json`, JSON.stringify({ mcpServers: {
  fake: { command: process.execPath, args: ["fake-mcp.ts"], cwd: import.meta.dir, env: { FAKE_MCP_LOG: LOG } },
  broken: { command: "/nonexistent/kiln-test-mcp" },
} }));
beforeAll(async () => {
  web = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", KILN_URL: `http://127.0.0.1:${fake.port}`, KILN_HOST: "127.0.0.1", NODE_ENV: "production", KILN_MCP_CONFIG: `${dir}/mcp.json` },
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
  const file = await (async () => { for (let k = 0; k < 50; k++) { const f = readdirSync(dir).find((x) => x.endsWith(".kiln.json")); if (f) return f; await Bun.sleep(100); } })();
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
  for (let k = 0; k < 50; k++) { const f = readdirSync(into).find((x) => x.endsWith(".kiln.json")); if (f) return `${into}/${f}`; await Bun.sleep(100); }
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
  expect(lastStream.messages[0].content).toContain("The tools offered here are the only ones");
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
  expect(lastStream.messages[0].content).toBe("There are no tools or files in this chat. Answer directly.");

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

  const round1 = streams.at(-1);                                          // what the tool's round sent kiln
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
  await send(page, "three");                                              // only the path on screen goes to kiln
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
  writeFileSync(old, JSON.stringify({ kiln: "chats", version: 1, exported: "2026-01-01T00:00:00.000Z", conversations: [{ id: "x", title: "Old", updated: 1, effort: "low",
    messages: [{ role: "user", content: "old q" }, { role: "assistant", content: "old a", reasoning: "hm", stats: { ttft: 1, tps: 2, tokens: 3, prompt: 4, cached: 0 } }] }] }));
  const third = await open();
  await (await third.$("input#import-file"))!.uploadFile(old);
  await third.waitForFunction(() => document.querySelectorAll("#thread > article").length === 2);
  expect(await third.$$eval(".pager", (p) => p.length)).toBe(0);
  await send(third, "next");
  expect(lastStream.messages).toEqual([{ role: "system", content: "There are no tools or files in this chat. Answer directly." }, { role: "user", content: "old q" },
    { role: "assistant", content: "old a", reasoning_content: "hm" }, { role: "user", content: "next" }]);
  for (const p of [page, other, third]) await p.close();
}, 90_000);
