import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { deflateSync } from "node:zlib";
import puppeteer, { type Browser, type BrowserContext, type Page } from "puppeteer-core";
import { partCode } from "./fake-mcp";

const RUN = mkdtempSync(`${tmpdir()}/chrysalis-chaos-`);
process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";
const CHRYSALIS = "http://127.0.0.1:8080", MIN = 60_000;
const CHAT_ARGS = ["--no-sandbox", "--ignore-certificate-errors"];

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
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", new Uint8Array())]);
})();

let web: Bun.Subprocess<"ignore", "pipe", "pipe">, base = "", browser: Browser;
const problems: string[] = [];
const serverLog: string[] = [];

beforeAll(async () => {
  const tls = mkdtempSync(`${RUN}/tls-`), dir = mkdtempSync(`${RUN}/mcp-`);
  writeFileSync(`${dir}/mcp.json`, JSON.stringify({ mcpServers: { fake: { command: process.execPath, args: ["fake-mcp.ts"], cwd: import.meta.dir, env: { FAKE_MCP_PART_CHARS: "38000" } } } }));
  web = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "pipe",
    env: { ...process.env, PORT: "0", CHRYSALIS_URL: CHRYSALIS, CHRYSALIS_HOST: "127.0.0.1", NODE_ENV: "production",
      CHRYSALIS_MCP_CONFIG: `${dir}/mcp.json`, CHRYSALIS_TLS_DIR: tls },
  });
  void (async () => {
    const r = web.stderr.getReader(), dec = new TextDecoder();
    for (;;) {
      const { value, done } = await r.read();
      if (done) return;
      for (const l of dec.decode(value).split("\n").filter(Boolean)) { serverLog.push(l); if (/error|exception|unhandled/i.test(l)) problems.push(`web server: ${l}`); }
    }
  })();
  const reader = web.stdout.getReader(), dec = new TextDecoder();
  let out = "";
  while (!/chrysalis web: \S+/.test(out)) {
    const { value, done } = await reader.read();
    if (done) throw new Error(`server.ts exited: ${out}`);
    out += dec.decode(value);
  }
  base = out.match(/chrysalis web: (\S+)/)![1].replace(/\/$/, "");
  browser = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: CHAT_ARGS, protocolTimeout: 30 * MIN,
    userDataDir: mkdtempSync(`${RUN}/chromium-`) });
}, MIN);
afterAll(async () => { await browser?.close(); web?.kill(); rmSync(RUN, { recursive: true, force: true }); });

const inArtifact = (stack: string) => {
  const frames = stack.split("\n").filter((l) => /^\s+at /.test(l));
  return frames.length > 0 && frames.every((l) => !/https?:/.test(l) || /\/sandbox:\d+:\d+\)?$/.test(l));
};

async function context(): Promise<BrowserContext> {
  return browser.createBrowserContext();
}
async function open(ctx?: BrowserContext): Promise<Page> {
  const page = await (ctx ?? await context()).newPage();
  await page.setViewport({ width: 1280, height: 900 });
  page.on("pageerror", (e) => { if (!inArtifact((e as Error).stack ?? "")) problems.push(`page: ${(e as Error).message}`); });
  page.on("console", (m) => {
    const from = m.location()?.url ?? "";
    if (m.type() === "error" && !/\/sandbox$|^blob:/.test(from) && !/^Permissions policy violation: xr-spatial-tracking/.test(m.text())) problems.push(`console: ${m.text()}`);
  });
  await page.goto(`${base}/`, { waitUntil: "load" });
  await page.waitForSelector("#input");
  return page;
}
async function thinking(page: Page, level: string) {
  for (let k = 0; k < 4; k++) {
    const now = await page.$eval("#energy", (e) => e.getAttribute("aria-valuetext"));
    if (now === level) return;
    await page.focus("#energy");
    await page.keyboard.press(["Off", "Low", "Medium", "Max"].indexOf(level) < ["Off", "Low", "Medium", "Max"].indexOf(now ?? "") ? "ArrowLeft" : "ArrowRight");
  }
  expect(await page.$eval("#energy", (e) => e.getAttribute("aria-valuetext"))).toBe(level);
}
async function ask(page: Page, text: string) {
  await page.click("#input");
  await page.$eval("#input", (t) => { (t as HTMLTextAreaElement).value = ""; });
  await page.type("#input", text);
  await page.keyboard.press("Enter");
}
function replyText(): string {
  const arts = document.querySelectorAll("#thread > article.msg.assistant"), body = arts[arts.length - 1]?.querySelector(":scope > .markdown");
  let out = "";
  const walk = (n: Node) => {
    if (n.nodeType === Node.TEXT_NODE) { out += n.textContent; return; }
    if (n instanceof Element) {
      if (n.matches("figcaption, style, script, .code-head, .svg-code, .art-card, .svg-bar, .art-tools")) return;
      if (n.shadowRoot) walk(n.shadowRoot);
      if (/^(P|DIV|LI|H\d|PRE|TR|BR)$/.test(n.tagName)) out += " ";
    }
    for (const c of n.childNodes) walk(c);
  };
  if (body) walk(body);
  if (body?.querySelector(".art-card.on")) for (const canvas of document.querySelectorAll("#shelf-stage > .art-canvas")) walk(canvas);
  return out.replace(/\s+/g, " ").trim();
}
function openLastCard() {
  const arts = document.querySelectorAll("#thread > article.msg.assistant"), cards = arts[arts.length - 1]?.querySelectorAll<HTMLElement>(":scope > .markdown .art-card");
  const card = cards?.[cards.length - 1];
  if (card && !card.classList.contains("on")) card.click();
  return !!card;
}
async function lastReply(page: Page) {
  if (await page.evaluate(openLastCard)) await page.waitForFunction(`(${replyText.toString()})().length > 0`, { timeout: 10_000 });
  return page.evaluate(replyText);
}
const reload = (page: Page) => page.reload({ waitUntil: "load" });
async function until(page: Page, what: string, wait: () => Promise<unknown>) {
  try { await wait(); } catch (e) {
    const dump = process.env.CHAOS_DUMP;
    if (dump) {
      const name = `${dump}/${Date.now()}-${what.replace(/\W+/g, "-")}`;
      await page.screenshot({ path: `${name}.png` }).catch(() => {});
      writeFileSync(`${name}.txt`, await page.evaluate(() => `${location.href}\n${document.body.innerText}`).catch((x) => String(x)));
    }
    throw new Error(`${what}: ${(e as Error).message}`);
  }
}
const longer = (page: Page, chars: number, timeout = 3 * MIN) => until(page, `a reply of ${chars} characters`,
  () => page.waitForFunction(`(${replyText.toString()})().length >= ${chars}`, { timeout }));
const writing = (page: Page, chars = 80) => longer(page, chars);
const idle = (page: Page, timeout = 5 * MIN) => until(page, "the reply's end",
  () => page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send", { timeout }));
const answered = async (page: Page, timeout = 5 * MIN) => {
  await until(page, "the reply's start", () => page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") !== "Send", { timeout: MIN }));
  await idle(page, timeout);
};
const attended = async (page: Page, wait: () => Promise<void>) => {
  const watch = setInterval(() => void page.evaluate(() => [...document.querySelectorAll<HTMLElement>('.tool-card[data-state="approval"]')].map((c) => {
    c.querySelector<HTMLElement>('[data-decision="once"]')?.click();
    return c.querySelector(".tool-name")?.textContent;
  })).then((asked) => { if (asked.length) console.log(`allowed: ${asked.join(", ")}`); }).catch(() => {}), 2000);
  await wait().finally(() => clearInterval(watch));
};
const openChat = (page: Page, k: number) => page.evaluate((i) => (document.querySelectorAll(".conv")[i] as HTMLElement).click(), k);

interface Logged { gen: number; tps: number; finish: string; stopped: boolean }
async function chrysalisReplies(since: number): Promise<Logged[]> {
  const recent: { at: number; completion_tokens: number; decode_tps?: number; finish_reason: string }[] =
    (await (await fetch(`${CHRYSALIS}/chrysalis/stats`)).json()).recent ?? [];
  const got: Logged[] = recent.filter((x) => x.at * 1000 >= since)
    .map((x) => ({ gen: x.completion_tokens, tps: x.decode_tps ?? 0, finish: x.finish_reason, stopped: x.finish_reason === "client" }));
  return got;
}
async function experts() {
  return (await (await fetch(`${CHRYSALIS}/chrysalis/stats`)).json()).experts as { mirrored: number; mirror_cap: number; decode_picks: number; decode_picks_in_pool: number };
}
async function healthy(label: string) {
  const x = await experts();
  expect(x.mirrored).toBeLessThanOrEqual(x.mirror_cap);
  const seen = problems.splice(0);
  if (seen.length) console.log(`${label}:\n  ${seen.join("\n  ")}`);
  expect(seen).toEqual([]);
}
const tpsOf = (r: Logged[]) => r.filter((x) => !x.stopped && x.gen >= 200).map((x) => x.tps);

test("a reply is followed through a reload, a new chat and back, and a second reload as it ends", async () => {
  const t0 = Date.now(), page = await open();
  await thinking(page, "Off");
  await ask(page, "Write a 500-word story about a lighthouse keeper and a storm.");
  await writing(page, 300);
  const before = await lastReply(page);
  await reload(page);
  await longer(page, before.length + 1, MIN);
  await idle(page);
  const after = await lastReply(page);
  expect(after.startsWith(before.slice(0, 200))).toBe(true);
  expect(after.length).toBeGreaterThan(1500);

  await ask(page, "Now write a 400-word story about a fox in the snow.");
  await writing(page, 200);
  await page.click("#new-chat");
  await page.type("#input", "a draft in another chat");
  await page.$eval(".conv:has(.conv-status)", (c) => (c as HTMLElement).click());
  await writing(page, 400);
  await reload(page);
  await idle(page);
  expect((await lastReply(page)).length).toBeGreaterThan(1200);
  expect(await page.$$eval("#thread .msg.assistant", (a) => a.length)).toBe(2);
  const r = await chrysalisReplies(t0);
  console.log("reload and switch:", r.map((x) => `${x.gen} tokens ${x.tps} tok/s ${x.finish}`).join("; "));
  expect(r.filter((x) => x.finish === "stop").length).toBeGreaterThanOrEqual(2);
  await healthy("reload and switch");
  await page.close();
}, 10 * MIN);

test("three chats write at once, a fourth waits its turn, and another tab writes beside them; every reply ends whole", async () => {
  const t0 = Date.now(), ctx = await context(), page = await open(ctx);
  const topics = ["a robot learning to paint", "a city under the sea", "the last tree on Mars", "a clockmaker's apprentice"];
  for (const [i, t] of topics.entries()) {
    if (i) await page.click("#new-chat");
    await thinking(page, "Off");
    await ask(page, `Write a 300-word story about ${t}.`);
    await page.waitForFunction((n) => document.querySelectorAll(".conv").length >= n, { timeout: 30_000 }, i + 1);
  }
  const other = await open(ctx);
  await thinking(other, "Off");
  await ask(other, "Explain in about 250 words how a refrigerator works.");
  await page.waitForFunction(() => !document.querySelector(".conv:has(.conv-status)"), { timeout: 10 * MIN, polling: 1000 });
  await idle(other, 10 * MIN);
  const n = await page.$$eval(".conv", (c) => c.length);
  expect(n).toBe(4);
  for (let k = 0; k < n; k++) {
    await openChat(page, k);
    await page.waitForFunction((i) => document.querySelectorAll(".conv")[i]?.classList.contains("active")
      && document.querySelectorAll("#thread > article.msg.assistant").length === 1, { timeout: 10_000 }, k);
    expect((await lastReply(page)).length).toBeGreaterThan(800);
  }
  expect((await lastReply(other)).length).toBeGreaterThan(800);
  const r = await chrysalisReplies(t0);
  console.log("five at once:", r.map((x) => `${x.gen} tokens ${x.tps} tok/s ${x.finish}`).join("; "));
  expect(r.filter((x) => x.finish === "stop" && x.gen > 150).length).toBe(5);
  await healthy("five at once");
  await ctx.close();
}, 15 * MIN);

test("Stop, regenerate and edit while replies are written leave every version whole or cut where it was stopped", async () => {
  const page = await open();
  await thinking(page, "Off");
  await ask(page, "Write a 600-word essay about the history of bridges.");
  await writing(page, 200);
  await page.keyboard.press("Escape");
  await idle(page, 30_000);
  const cut = await lastReply(page);
  expect(cut.length).toBeGreaterThan(200);
  await page.hover("#thread > article[data-i='1']");
  await page.click("#thread > article[data-i='1'] [data-act='retry']");
  await writing(page, 150);
  await page.click("#send");
  await idle(page, 30_000);
  expect(await page.$eval("#thread > article[data-i='1'] .pager-n", (p) => p.textContent)).toBe("2 / 2");
  await page.hover("#thread > article[data-i='0']");
  await page.click("#thread > article[data-i='0'] [data-act='edit']");
  await page.waitForSelector(".editing textarea");
  await page.$eval(".editing textarea", (t) => { (t as HTMLTextAreaElement).value = ""; });
  await page.type(".editing textarea", "In two sentences, what is a suspension bridge?");
  await page.keyboard.down("Control"); await page.keyboard.press("Enter"); await page.keyboard.up("Control");
  await idle(page);
  expect(await page.$eval("#thread > article[data-i='0'] .pager-n", (p) => p.textContent)).toBe("2 / 2");
  expect((await lastReply(page)).length).toBeGreaterThan(40);
  await page.click("#thread > article[data-i='0'] [data-act='prev']");
  await page.waitForFunction(() => document.querySelector("#thread > article[data-i='1'] .pager-n")?.textContent === "2 / 2", { timeout: 5000 });
  await healthy("stop, regenerate, edit");
  await page.close();
}, 10 * MIN);

test("a chat deleted while it is written stops its reply at Chrysalis, and the chats beside it go on", async () => {
  const t0 = Date.now(), page = await open();
  await thinking(page, "Off");
  await ask(page, "Write a 700-word story about a dragon who collects teacups.");
  await writing(page, 100);
  const doomed = await page.$eval(".conv.active", (c) => c.getAttribute("href"));
  await page.click("#new-chat");
  await thinking(page, "Off");
  await ask(page, "Write a 300-word story about a lost umbrella.");
  await page.waitForFunction(() => document.querySelectorAll(".conv:has(.conv-status)").length === 2, { timeout: 30_000 });
  await page.evaluate((href) => {
    const del = document.querySelector<HTMLElement>(`.conv[href="${href}"] .conv-del`)!;
    del.click(); del.click();
  }, doomed);
  await page.waitForFunction(() => document.querySelectorAll(".conv").length === 1, { timeout: 10_000 });
  await idle(page);
  expect((await lastReply(page)).length).toBeGreaterThan(800);
  await Bun.sleep(1500);
  const r = await chrysalisReplies(t0);
  console.log("deleted mid-reply:", r.map((x) => `${x.gen} tokens ${x.finish}`).join("; "));
  expect(r.some((x) => x.stopped)).toBe(true);
  await healthy("delete mid-reply");
  await page.close();
}, 10 * MIN);

test("settings, panels, the window's size and the models list changed while a reply is written: it ends whole and the settings stay", async () => {
  const page = await open();
  await thinking(page, "Off");
  await ask(page, "Write a 500-word story about a library that only opens at night.");
  await writing(page, 100);
  await page.click("#settings-open");
  await page.waitForFunction(() => {
    const r = document.querySelector("#sea-switch")?.getBoundingClientRect();
    return !!r && r.width > 0 && r.right <= innerWidth && !document.getAnimations().some((a) => a.playState === "running" && (a.effect as KeyframeEffect)?.target instanceof Element && (a.effect as KeyframeEffect).target!.closest("#settings"));
  }, { timeout: 10_000 });
  await page.$eval("#sea-switch", (b) => b.scrollIntoView({ block: "center" }));
  await page.click("#sea-switch");
  await page.$eval("#artifacts-switch", (b) => b.scrollIntoView({ block: "center" }));
  await page.click("#artifacts-switch");
  await page.click("#artifacts-switch");
  await page.click("#settings-close");
  await page.click("#tools-btn");
  await page.keyboard.press("Escape");
  await page.click("#model-btn");
  await page.keyboard.press("Escape");
  await page.click("#conversations-toggle");
  await page.click("#conversations-toggle");
  await page.setViewport({ width: 390, height: 800 });
  await Bun.sleep(500);
  await page.setViewport({ width: 1280, height: 900 });
  await page.focus("#energy");
  await page.keyboard.press("ArrowRight");
  await idle(page);
  expect((await lastReply(page)).length).toBeGreaterThan(1500);
  const settings = () => page.evaluate(() => document.querySelector("#sea-switch")?.getAttribute("aria-checked") ?? "");
  const was = await settings();
  await reload(page);
  expect(await settings()).toEqual(was);
  await healthy("settings mid-reply");
  await page.close();
}, 10 * MIN);

test("tools: a call followed through a reload, one that asks first answered after a reload, and a slow one stopped", async () => {
  const page = await open();
  await thinking(page, "Off");
  await page.click("#tools-btn");
  const fake = '#tools-pop [data-do="src"][data-src="mcp:fake"]';
  await page.waitForSelector(fake);
  if ((await page.$eval(fake, (b) => b.getAttribute("aria-checked"))) !== "true") await page.click(fake);
  await page.waitForFunction(() => document.querySelector('#tools-pop [data-src="mcp:fake"]')?.closest(".tp-src")?.querySelector(".tp-sub")?.textContent?.includes("tools on"), { timeout: 30_000 });
  await page.keyboard.press("Escape");
  await ask(page, "Call the echo tool with the text 'chaos monkey', then tell me exactly what it returned.");
  await page.waitForSelector(".tool-card", { timeout: 2 * MIN });
  await reload(page);
  await idle(page);
  expect(await page.$$eval(".tool-card", (c) => c.length)).toBeGreaterThan(0);
  expect((await lastReply(page)).toLowerCase()).toContain("chaos monkey");

  await ask(page, "Save a note with the write_note tool saying 'buy milk'.");
  await page.waitForSelector('[data-act="decide"][data-decision="once"]', { timeout: 2 * MIN });
  await reload(page);
  await page.waitForSelector('[data-act="decide"][data-decision="once"]', { timeout: 30_000 });
  await page.click('[data-act="decide"][data-decision="once"]');
  await idle(page);
  expect(await page.$$eval(".tool-card", (c) => c.map((x) => x.textContent ?? "").join(" "))).toMatch(/done/i);

  await ask(page, "Call the slow tool with ms set to 20000.");
  await page.waitForFunction(() => [...document.querySelectorAll(".tool-card")].some((c) => /running/.test(c.textContent ?? "")), { timeout: 2 * MIN });
  await page.keyboard.press("Escape");
  await idle(page, 30_000);
  expect(await page.$$eval(".tool-card", (c) => c.map((x) => x.textContent ?? "").join(" "))).toMatch(/stopped/i);
  await healthy("tools");
  await page.close();
}, 15 * MIN);

test("a 3D page asked for and handled while it is written: Edit, clicks and keys in it, the shelf, a reload; it runs whole at the column's size", async () => {
  const page = await open();
  await thinking(page, "Off");
  await ask(page, "Make a three.js scene of a vampire countess's bedroom as a single HTML page.");
  await until(page, "the page's frame", () => page.waitForSelector("iframe.page-frame", { timeout: 4 * MIN }));
  const named = !!(await page.$(".art-canvas iframe.page-frame"));
  const frame = await page.$("iframe.page-frame");
  const box = (await frame!.boundingBox())!;
  expect(box.height).toBeGreaterThan(300);
  await page.click(".art-canvas [data-art=edit]").catch(() => {});
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await page.keyboard.press("Delete");
  await page.keyboard.down("Control"); await page.keyboard.press("z"); await page.keyboard.up("Control");
  if (named) await page.click("#artifacts-toggle");
  await page.click("#shelf-close").catch(() => {});
  await reload(page);
  await idle(page, 6 * MIN);
  const f = await page.waitForSelector("iframe.page-frame:not(.next)", { timeout: 30_000 });
  await page.waitForFunction(() => (document.querySelector("iframe.page-frame:not(.next)") as HTMLElement).offsetHeight > 300, { timeout: 30_000 });
  const inner = (await f!.contentFrame())!;
  await inner.waitForFunction(() => !!document.querySelector("canvas"), { timeout: 30_000 });
  console.log(`3D page: ${named ? "an artifact" : "a page with no name"}`);
  if (named) {
    await page.click(".art-canvas [data-art=edit]");
    await page.waitForSelector(".art-canvas .art-tools", { timeout: 5000 });
    await page.keyboard.press("Escape");
    await page.keyboard.press("Escape");
  }
  await healthy("3D page");
  await page.close();
}, 10 * MIN);

test("the controls hammered: Enter held on an empty box, a question sent twice fast, Esc and the send button pressed over and over, a very long paste", async () => {
  const page = await open();
  await thinking(page, "Off");
  await page.click("#input");
  for (let k = 0; k < 10; k++) await page.keyboard.press("Enter");
  expect(await page.$$eval("#thread > article", (a) => a.length)).toBe(0);
  await page.type("#input", "Count from 1 to 40, one number per line.");
  await page.keyboard.press("Enter");
  await page.keyboard.press("Enter");
  await writing(page, 20);
  expect(await page.$$eval("#thread .msg.user", (a) => a.length)).toBe(1);
  await idle(page);
  await ask(page, "Write a 500-word story about a snail who wins a race.");
  await writing(page, 50);
  for (let k = 0; k < 6; k++) await page.keyboard.press("Escape");
  for (let k = 0; k < 4; k++) await page.click("#send").catch(() => {});
  await idle(page, 30_000);
  const long = "The quick brown fox jumps over the lazy dog. ".repeat(1500);
  await page.$eval("#input", (t, v) => { (t as HTMLTextAreaElement).value = v; t.dispatchEvent(new Event("input", { bubbles: true })); }, long + "\nHow many times does 'fox' appear above? Just guess.");
  await page.keyboard.press("Enter");
  await Bun.sleep(1500);
  await page.keyboard.press("Escape");
  await idle(page, MIN);
  await ask(page, "Say hello in five languages.");
  await idle(page);
  expect((await lastReply(page)).length).toBeGreaterThan(20);
  await healthy("hammered");
  await page.close();
}, 10 * MIN);

test("two chats' replies are written whole while another program takes the memory (all but 1.5 GB of what is free, 8 GB at most), and Chrysalis gives back the RAM it borrowed", async () => {
  const run = async (label: string) => {
    const t0 = Date.now(), page = await open();
    await thinking(page, "Off");
    await ask(page, "Write a Python module implementing a thread-safe LRU cache with tests.");
    await page.click("#new-chat");
    await thinking(page, "Off");
    await ask(page, "Write a 600-word story about a pianist who goes deaf.");
    await page.waitForFunction(() => !document.querySelector(".conv:has(.conv-status)"), { timeout: 10 * MIN, polling: 1000 });
    await page.close();
    const r = await chrysalisReplies(t0);
    console.log(`${label}:`, r.map((x) => `${x.gen} tokens ${x.tps} tok/s`).join("; "));
    return tpsOf(r);
  };
  const calm = await run("no other program");
  const available = () => +readFileSync("/proc/meminfo", "utf8").match(/MemAvailable:\s+(\d+)/)![1] * 1024;
  for (let k = 0; k < 120 && available() < 2.5e9; k++) await Bun.sleep(1000);
  const free = available(), take = Math.min(8e9, free - 1.5e9);
  expect(free).toBeGreaterThan(2.5e9);
  const hog = Bun.spawn(["systemd-run", "--user", "--scope", "-q", "python3", "-c",
    `import time\nb = bytearray(${Math.round(take)})\nfor i in range(0, len(b), 4096): b[i] = 1\nprint('held', flush=True)\ntime.sleep(900)`], { stdout: "pipe" });
  try {
    const r = hog.stdout.getReader();
    await r.read();
    const squeezed = await run(`${(take / 1e9).toFixed(1)} GB held by another program`);
    const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;
    expect([calm.length, squeezed.length]).toEqual([2, 2]);
    console.log(`decode with the memory taken: ${(100 * mean(squeezed) / mean(calm)).toFixed(0)}% of its speed without`);
    const back = available(), { mirrored } = await experts();
    console.log(`with ${(take / 1e9).toFixed(1)} GB taken: ${(back / 1e9).toFixed(1)} GB available, ${mirrored} experts still mirrored`);
    expect(back > 2.5e9 || mirrored === 0).toBe(true);
  } finally {
    hog.kill();
    await hog.exited;
  }
  await healthy("memory pressure");
}, 25 * MIN);

test("Chrysalis restarted under a reply: the reply says it was cut off, nothing hangs, and the next one is written once it is back", async () => {
  const page = await open();
  await thinking(page, "Off");
  await ask(page, "Write a 700-word story about a train that never stops.");
  await writing(page, 200);
  await fetch(`${base}/api/chrysalis/stop`, { method: "POST" });
  await fetch(`${base}/api/chrysalis/start`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  await idle(page, 2 * MIN);
  const deadline = Date.now() + 3 * MIN;
  while ((await fetch(`${CHRYSALIS}/v1/models`).then((r) => r.ok, () => false)) === false) {
    if (Date.now() > deadline) throw new Error("Chrysalis did not come back");
    await Bun.sleep(2000);
  }
  await ask(page, "In one sentence, what is a train?");
  await idle(page, 3 * MIN);
  expect((await lastReply(page)).length).toBeGreaterThan(20);
  problems.splice(0).forEach((p) => console.log("while Chrysalis restarted:", p));
  await ask(page, "In one sentence, what is a boat?");
  await idle(page, 2 * MIN);
  await healthy("restart");
  await page.close();
}, 15 * MIN);

test("exported and imported while a reply is written, and a picture asked about in the imported chat", async () => {
  const ctx = await context(), page = await open(ctx), into = mkdtempSync(`${RUN}/export-`);
  await thinking(page, "Off");
  await ask(page, "Write a 400-word story about a kite.");
  await writing(page, 100);
  const cdp = await page.createCDPSession();
  await cdp.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: into, browserContextId: ctx.id });
  await page.click("#export-all");
  let file = "";
  for (let k = 0; k < 50 && !file; k++) { file = readdirSync(into).find((x) => x.startsWith("chrysalis-chats-")) ?? ""; await Bun.sleep(100); }
  expect(file).not.toBe("");
  await idle(page);
  const other = await open(ctx);
  await (await other.$("input#import-file"))!.uploadFile(`${into}/${file}`);
  await other.waitForFunction(() => document.querySelectorAll("#thread > article.msg.assistant").length === 1, { timeout: 10_000 });
  const png = `${into}/red.png`;
  await Bun.write(png, RED);
  await thinking(other, "Off");
  await (await other.$("input#attach-file"))!.uploadFile(png);
  await other.waitForSelector("#attachments .att", { timeout: 10_000 });
  await ask(other, "What color is this picture? One word.");
  await idle(other);
  expect((await lastReply(other)).toLowerCase()).toContain("red");
  await healthy("export, import, picture");
  await ctx.close();
}, 10 * MIN);

test("a long conversation keeps its speed: eight questions in one chat, each answered whole", async () => {
  const t0 = Date.now(), page = await open();
  await thinking(page, "Off");
  const qs = ["Write a Python function that parses ISO 8601 dates.", "Add error handling to it.", "Now write pytest tests for it.",
    "Explain the tests in plain words.", "Rewrite the function in TypeScript.", "What are the edge cases we missed?",
    "Write a short poem about dates and times.", "Summarize this whole conversation in five bullet points."];
  for (const q of qs) {
    await ask(page, q);
    await idle(page);
    expect((await lastReply(page)).length).toBeGreaterThan(40);
  }
  const fixes = await page.$$eval("#thread .msg.user.fix", (a) => a.length);
  expect(await page.$$eval("#thread .msg.assistant", (a) => a.length)).toBe(qs.length + fixes);
  const r = (await chrysalisReplies(t0)).filter((x) => x.finish === "stop");
  console.log("eight turns:", r.map((x) => `${x.gen} tokens ${x.tps} tok/s`).join("; "));
  expect(r.length).toBeGreaterThanOrEqual(qs.length);
  await healthy("long conversation");
  await page.close();
}, 20 * MIN);

test("sub-agents: three tasks handed off at once in one reply while another chat writes", async () => {
  const page = await open();
  await thinking(page, "Off");
  await page.click("#tools-btn");
  const agents = '#tools-pop [data-do="src"][data-src="agents"]';
  await page.waitForSelector(agents);
  if ((await page.$eval(agents, (b) => b.getAttribute("aria-checked"))) !== "true") await page.click(agents);
  await page.keyboard.press("Escape");
  await ask(page, "Use three sub-agents at once: one writes a haiku about rain, one a haiku about fire, one a haiku about wind. Then show all three.");
  await page.click("#new-chat");
  await thinking(page, "Off");
  await ask(page, "Write a 300-word story about a lighthouse.");
  await page.waitForFunction(() => !document.querySelector(".conv:has(.conv-status)"), { timeout: 10 * MIN, polling: 1000 });
  await page.$eval(".conv:not(.active)", (c) => (c as HTMLElement).click());
  await page.waitForSelector(".agent-head", { timeout: 10_000 });
  const cards = await page.$$eval(".agent-head", (c) => c.length);
  console.log(`sub-agent cards: ${cards}`);
  expect(cards).toBeGreaterThan(0);
  expect((await lastReply(page)).length).toBeGreaterThan(40);
  await healthy("sub-agents");
  await page.close();
}, 15 * MIN);

test("sub-agents: three long readings at once outgrow the memory they share, compact, and still report every code exactly", async () => {
  const page = await open(), parts = 20, books = ["alpha", "beta", "gamma"];
  await thinking(page, "Off");
  await page.click("#tools-btn");
  for (const src of ["agents", "mcp:fake"]) {
    const on = `#tools-pop [data-do="src"][data-src="${src}"]`;
    await page.waitForSelector(on);
    if ((await page.$eval(on, (b) => b.getAttribute("aria-checked"))) !== "true") await page.click(on);
  }
  await page.keyboard.press("Escape");
  await ask(page, `Use three sub-agents at once, in one message. Give the first the book "alpha", the second the book "beta" and the third the book "gamma". Tell each one: read parts 1 to ${parts} of its book with the read_part tool, `
    + "four parts per message, and report every part's code in a list. It must read every part: the codes cannot be guessed. Then show me the three lists.");
  await attended(page, () => answered(page, 25 * MIN));
  const cards = await page.$$eval(".agent-card", (c) => c.map((x) => x.getAttribute("data-state")));
  console.log(`sub-agents: ${cards.join(", ")}`);
  expect(cards.length).toBeGreaterThanOrEqual(3);
  expect(cards.every((s) => s === "done")).toBe(true);
  const reply = await lastReply(page);
  expect(books.flatMap((b) => Array.from({ length: parts }, (_, i) => partCode(b, i + 1))).filter((c) => !reply.includes(c))).toEqual([]);
  for (const head of await page.$$(".agent-head")) await head.click();
  const compacted = await page.$$eval(".agent-log .checkpoint", (c) => c.length);
  console.log(`compactions: ${compacted}`);
  expect(compacted).toBeGreaterThan(0);
  await healthy("sub-agents past their memory");
  await page.close();
}, 30 * MIN);

test("one reply reads past the whole context, compacts, and the chat goes on from what it kept, through a reload", async () => {
  const page = await open(), parts = 40, book = "delta";
  await thinking(page, "Off");
  await page.click("#tools-btn");
  const on = '#tools-pop [data-do="src"][data-src="mcp:fake"]';
  await page.waitForSelector(on);
  if ((await page.$eval(on, (b) => b.getAttribute("aria-checked"))) !== "true") await page.click(on);
  await page.keyboard.press("Escape");
  await ask(page, `Read parts 1 to ${parts} of the book "${book}" with the read_part tool, four parts per message, and report every part's code in a list. Read every part: the codes cannot be guessed. Do not use sub-agents.`);
  await attended(page, () => answered(page, 20 * MIN));
  const compacted = await page.$$eval(".checkpoint", (c) => c.length);
  console.log(`compactions: ${compacted}`);
  expect(compacted).toBeGreaterThan(0);
  const list = await lastReply(page);
  expect(Array.from({ length: parts }, (_, i) => partCode(book, i + 1)).filter((c) => !list.includes(c))).toEqual([]);
  await ask(page, "What was the code of part 3? Answer with the code alone.");
  await answered(page, 3 * MIN);
  expect(await lastReply(page)).toContain(partCode(book, 3));
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector(".checkpoint", { timeout: 20_000 });
  await ask(page, "And the code of part 38? Answer with the code alone.");
  await answered(page, 3 * MIN);
  expect(await lastReply(page)).toContain(partCode(book, 38));
  await healthy("a reply past the whole context");
  await page.close();
}, 30 * MIN);

test("a tab closed while its reply is written: the reply runs out at Chrysalis, and a new tab starts clean", async () => {
  const t0 = Date.now(), ctx = await context(), page = await open(ctx);
  await thinking(page, "Off");
  await ask(page, "Write a 300-word story about a paper boat.");
  await writing(page, 100);
  await page.close();
  const fresh = await open(ctx);
  expect(await fresh.$$eval(".conv", (c) => c.length)).toBe(0);
  await thinking(fresh, "Off");
  await ask(fresh, "In one sentence, what is a paper boat?");
  await idle(fresh, 3 * MIN);
  expect((await lastReply(fresh)).length).toBeGreaterThan(20);
  for (let k = 0; k < 60 && !(await chrysalisReplies(t0)).some((x) => x.finish === "stop" && x.gen > 250); k++) await Bun.sleep(1000);
  expect((await chrysalisReplies(t0)).some((x) => x.finish === "stop" && x.gen > 250)).toBe(true);
  await healthy("closed tab");
  await ctx.close();
}, 10 * MIN);

test("the model switched to the 27B while a Flash-Next reply is written, then back: every reply ends, from the model asked", async () => {
  const page = await open();
  await thinking(page, "Off");
  await ask(page, "Write a 400-word story about a moth and a lamp.");
  await writing(page, 100);
  const pick = async (label: RegExp) => {
    await page.click("#model-btn");
    await page.waitForSelector("#model-pop .model-item");
    const items = await page.$$("#model-pop .model-item");
    for (const it of items) if (label.test(await it.evaluate((e) => e.textContent ?? ""))) { await it.click(); return; }
    throw new Error(`no model ${label}`);
  };
  await page.click("#new-chat");
  await pick(/27B/);
  await thinking(page, "Off");
  await ask(page, "In two sentences, what is a moth?");
  await idle(page, 10 * MIN);
  expect((await lastReply(page)).length).toBeGreaterThan(20);
  const model = async () => (await (await fetch(`${CHRYSALIS}/chrysalis/stats`)).json()).model;
  expect(await model()).toBe("qwen3.8-27b");
  await page.$eval(".conv:not(.active)", (c) => (c as HTMLElement).click());
  await page.waitForFunction(() => document.querySelectorAll("#thread > article.msg.assistant").length === 1, { timeout: 10_000 });
  expect((await lastReply(page)).length).toBeGreaterThan(100);
  await pick(/Flash/);
  await thinking(page, "Off");
  await ask(page, "In two sentences, what is a lamp?");
  await idle(page, 5 * MIN);
  expect(await model()).toBe("qwen3.8-flash-next");
  expect((await lastReply(page)).length).toBeGreaterThan(20);
  const x = await experts();
  expect(x.mirrored).toBeLessThanOrEqual(x.mirror_cap);
  problems.splice(0).forEach((p) => console.log("while the model switched:", p));
  await page.close();
}, 20 * MIN);
