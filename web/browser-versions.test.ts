import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { ARTIFACTS_PROMPT } from "./artifacts";
import { harness, articles, card, dir, download, fakes, idle, lastStream, loading, open, pagerOf, send, sent, still, streams, untilIdle, fresh } from "./browser-harness";

harness();

test("a model picked while a reply is written is the one the next question asks for, though Chrysalis has the other until it switches", async () => {
  const page = await open();
  try {
    await page.type("#input", "slow one");
    await page.keyboard.press("Enter");
    await fresh(page);
    await page.click("#model-btn");
    await page.waitForSelector("#model-pop .model-item");
    await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#model-pop .model-item")].find((b) => /Flash/.test(b.textContent ?? ""))!.click());
    while (!loading) await Bun.sleep(10);
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
  await fresh(page);
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

test("swiping the latest reply: its pager shows from the first version and its next arrow writes a new one; with the message box empty the arrow keys go between versions and past the last to a new one, and move the caret once there is text; the right arrow while a reply is being written starts that reply over; a drag across the reply on a touch screen does the same", async () => {
  const page = await open();
  await send(page, "one");
  await send(page, "two");
  expect([await pagerOf(page, 1), await pagerOf(page, 3)]).toEqual(["", "1 / 1"]);
  await page.click('#thread > article[data-i="3"] [data-act="next"]');
  await untilIdle(page);
  expect([await articles(page), await pagerOf(page, 3)]).toEqual([["one", "You said: one", "two", "You said: two"], "2 / 2"]);
  expect(lastStream.messages.filter((m: any) => m.role !== "system").map((m: any) => m.content)).toEqual(["one", "You said: one", "two"]);

  await page.focus("#input");
  await page.keyboard.press("ArrowLeft");
  expect(await pagerOf(page, 3)).toBe("1 / 2");
  await page.keyboard.press("ArrowRight");
  expect(await pagerOf(page, 3)).toBe("2 / 2");
  await page.keyboard.press("ArrowRight");
  await untilIdle(page);
  expect(await pagerOf(page, 3)).toBe("3 / 3");
  expect(await page.evaluate(() => document.activeElement?.id)).toBe("input");
  await page.type("#input", "ab");
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.type("c");
  expect([await pagerOf(page, 3), await page.$eval("#input", (t) => (t as HTMLTextAreaElement).value)]).toEqual(["3 / 3", "acb"]);

  const box = (await (await page.$('#thread > article[data-i="3"] .markdown'))!.boundingBox())!, y = box.y + box.height / 2;
  const drag = async (from: number, to: number, down = 0) => {
    await page.touchscreen.touchStart(from, y);
    await page.touchscreen.touchMove(to, y + down);
    await page.touchscreen.touchEnd();
  };
  await drag(box.x + 80, box.x + 260);
  expect(await pagerOf(page, 3)).toBe("2 / 3");
  await drag(box.x + 80, box.x + 120);
  await drag(box.x + 80, box.x + 180, 120);
  expect(await pagerOf(page, 3)).toBe("2 / 3");
  await drag(box.x + 260, box.x + 80);
  expect(await pagerOf(page, 3)).toBe("3 / 3");
  await drag(box.x + 260, box.x + 80);
  await untilIdle(page);
  expect(await pagerOf(page, 3)).toBe("4 / 4");

  await page.$eval("#input", (t) => { (t as HTMLTextAreaElement).value = ""; });
  await page.type("#input", "slow three");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /w3/.test(document.querySelector('#thread > article[data-i="5"] .markdown')?.textContent ?? ""), { timeout: 10_000 });
  const begun = streams.length;
  await page.keyboard.press("ArrowRight");
  await page.waitForFunction(() => !/w3/.test(document.querySelector('#thread > article[data-i="5"] .markdown')?.textContent ?? ""), { timeout: 10_000 });
  fakes.hurry = true;
  await untilIdle(page);
  fakes.hurry = false;
  expect([await pagerOf(page, 5), streams.length - begun, (await articles(page)).length]).toEqual(["1 / 1", 1, 6]);
  await page.close();
}, 60_000);
