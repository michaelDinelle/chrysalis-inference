import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { ARTIFACTS_PROMPT } from "./artifacts";
import { harness, articles, card, dir, download, fakes, idle, lastStream, loading, open, pagerOf, pop, send, sent, still, streams, toolsOn, untilIdle } from "./browser-harness";

harness();

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
    fakes.hurry = true;
    await untilIdle(page);
    expect(streams.slice(n).filter((b) => b.messages.at(-1).content === "which model").map((b) => b.model)).toEqual(["qwen3.8-flash-next"]);
  } finally {
    await loading;
    fakes.loaded = "qwen3.8-27b";
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

  await idle(page, page.reload());
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
  fakes.hurry = true;
  await send(page, "slow one");
  fakes.hurry = false;
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
  fakes.hurry = true;
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
