import { expect, test } from "bun:test";
import type { Page } from "puppeteer-core";
import { card, connect, dir, fresh, harness, open, opencode, send } from "./browser-harness";

harness();

const rowOf = (page: Page, title: string) => page.evaluate((title) => {
  const a = [...document.querySelectorAll<HTMLElement>("#conversations a.conv")].find((x) => x.querySelector(".conv-title")!.textContent === title)!, r = a.getBoundingClientRect();
  return { x: r.x, y: r.y, w: r.width, h: r.height };
}, title);
const titles = (page: Page, sel: string) => page.$$eval(`#conversations ${sel} .conv-title`, (t) => t.map((x) => x.textContent));

test("a thread's row is dragged as T3 Code's is: it lifts after six pixels and stays under the pointer on the vertical alone, no other row moves, at the top it pins where a line shows and pins reorder, below them it unpins, dropped on a character it becomes her chat in OpenCode, and dragged down to Settled it is settled and dragged back up it is active again; New thread's palette lists projects and characters in rows and starts a chat with the one picked, and the filter's list narrows the sidebar to one", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mira.png`, await card("Mira", { description: "{{char}} keeps bees." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mira.png`);
  await page.waitForFunction(() => [...document.querySelectorAll("#conversations .cast-row .cast-name")].some((n) => n.textContent === "Mira"));
  await connect(page);
  for (const text of ["first jar", "second jar", "third jar"]) {
    await fresh(page);
    await send(page, text);
  }
  await fresh(page);
  expect([await page.$("#conversations .folder.pinned"), await page.$eval("#side-pins", (e) => e.textContent), await page.$eval("#side-shelf", (e) => e.textContent)]).toEqual([null, "Pinned", "Settled"]);
  const others = () => page.$$eval("#conversations a.conv, #conversations .cast-row, #side-pins, #side-list, #side-shelf", (x) => x.map((e) => { const r = e.getBoundingClientRect(); return [Math.round(r.x), Math.round(r.y), Math.round(r.height)]; }));

  const drag = async (title: string, to: (at: { x: number; y: number; w: number; h: number }) => Promise<number> | number, during?: () => Promise<void>) => {
    const at = await rowOf(page, title), x = at.x + at.w / 2, y = at.y + at.h / 2;
    await page.mouse.move(x, y);
    await page.mouse.down();
    await page.mouse.move(x + 4, y + 3);
    expect(await page.$eval("#conversations", (l) => l.classList.contains("sorting"))).toBe(false);
    await page.mouse.move(x + 4, y + 9);
    expect(await page.$eval("#conversations", (l) => l.classList.contains("sorting"))).toBe(true);
    const goal = await to(at);
    await page.mouse.move(x + 40, goal, { steps: 12 });
    await during?.();
    await page.mouse.up();
    await page.waitForFunction(() => !document.querySelector("#conversations.sorting"));
    return { x, y, goal };
  };
  const top = async () => (await page.$eval("#conversations", (l) => l.getBoundingClientRect().top)) + 14;
  const dragged = () => page.$eval(".drag-ghost", (a) => { const r = a.getBoundingClientRect(); return { x: r.x, mid: r.y + r.height / 2, verb: a.querySelector(".conv-verb:not([hidden])")?.textContent ?? "" }; });

  let first = await rowOf(page, "first jar"), goal = 0;
  const still = await others();
  await drag("first jar", async () => (goal = await top()), async () => {
    const now = await dragged();
    expect([now.x, Math.abs(now.mid - (goal + 14 + first.h / 2)) < 3, now.verb, await page.$$eval("#side-pins, #side-shelf", (x) => x.map((e) => e.classList.contains("target"))), await others(),
      await page.evaluate(() => { const [a, l, b] = ["#side-pins", "#side-list", "#side-shelf"].map((q) => document.querySelector(q)!.getBoundingClientRect()); return a.bottom <= l.top && b.top >= l.bottom; })])
      .toEqual([first.x, true, "Pin", [true, false], still, true]);
  });
  expect([await titles(page, ".folder.pinned"), (await titles(page, ".folder:not(.pinned)")).includes("first jar")]).toEqual([["first jar"], false]);

  const pinnedAt = await rowOf(page, "first jar");
  await drag("third jar", () => pinnedAt.y + 6, async () => {
    expect([Math.abs((await page.$eval(".drag-bar", (b) => b.getBoundingClientRect().top)) - pinnedAt.y) < 4, (await rowOf(page, "first jar")).y]).toEqual([true, pinnedAt.y]);
  });
  expect(await titles(page, ".folder.pinned")).toEqual(["third jar", "first jar"]);
  first = await rowOf(page, "first jar");
  await drag("third jar", () => first.y + first.h - 4);
  expect(await titles(page, ".folder.pinned")).toEqual(["first jar", "third jar"]);

  const second = await rowOf(page, "second jar");
  await drag("first jar", () => second.y + second.h / 2, async () => expect((await dragged()).verb).toBe("Unpin"));
  expect(await titles(page, ".folder.pinned")).toEqual(["third jar"]);

  const mira = () => page.evaluate(() => { const r = [...document.querySelectorAll<HTMLElement>("#conversations .folder")].find((f) => f.querySelector(".cast-name")?.textContent === "Mira")!.getBoundingClientRect(); return r.y + r.height / 2; });
  const session = opencode.log.prompts.find((p) => p.text === "second jar")!.session;
  await drag("second jar", mira, async () => {
    expect([(await dragged()).verb, await page.$eval("#conversations .folder.drop .cast-name", (n) => n.textContent)]).toEqual(["Mira", "Mira"]);
  });
  await page.waitForFunction(() => [...document.querySelectorAll<HTMLElement>("#conversations .folder")].find((f) => f.querySelector(".cast-name")?.textContent === "Mira")?.querySelector(".conv-title")?.textContent === "second jar");
  expect((opencode.sessions.get(session)!.metadata as { chrysalis?: { cast: string[] } }).chrysalis?.cast).toEqual(["Mira"]);

  expect(await page.$("#conversations .folder.settled")).toBeNull();
  const mid = (sel: string) => page.$eval(sel, (e) => { const r = e.getBoundingClientRect(); return r.y + r.height / 2; });
  await drag("first jar", () => page.$eval("#conversations", (l) => l.getBoundingClientRect().bottom - 4), async () => expect([(await dragged()).verb, await page.$eval("#side-shelf", (e) => e.classList.contains("target"))]).toEqual(["Settle", true]));
  expect(await page.$(".drag-ghost, .drag-bar, #side-shelf.target")).toBeNull();
  await page.waitForSelector("#conversations .folder.settled .settled-head");
  expect([await page.$eval("#conversations .folder.settled .cast-count", (c) => c.textContent), (await titles(page, ".folder:not(.settled)")).includes("first jar"), !!opencode.sessions.get(opencode.log.prompts.find((p) => p.text === "first jar")!.session)!.archived])
    .toEqual(["1", false, true]);
  await page.click("#conversations .folder.settled .settled-head");
  await page.waitForFunction(() => document.querySelector("#conversations .folder.settled .conv-title")?.textContent === "first jar");
  await drag("first jar", () => mid("#conversations .folder[data-who] .cast-row"), async () => expect((await dragged()).verb).toBe("Un-settle"));
  await page.waitForFunction(() => !document.querySelector("#conversations .folder.settled"));
  expect((await titles(page, ".folder:not(.settled)")).includes("first jar")).toBe(true);

  const third = await rowOf(page, "third jar");
  await page.mouse.move(third.x + 20, third.y + 20);
  await page.mouse.down();
  await page.mouse.move(third.x + 20, third.y + 200, { steps: 6 });
  await page.keyboard.press("Escape");
  await page.mouse.up();
  expect([await page.$("#conversations.sorting, .drag-ghost"), await titles(page, ".folder.pinned")]).toEqual([null, ["third jar"]]);

  await page.click("#new-chat");
  await page.waitForSelector("#palette[open] .palette-item");
  const stacked = (sel: string) => page.$$eval(sel, (x) => { const r = x.map((e) => e.getBoundingClientRect()); return r.every((b, k) => !k || (b.top >= r[k - 1].bottom && b.left === r[0].left)); });
  expect([await page.$$eval("#palette .palette-label", (x) => x.map((e) => e.textContent)), await page.$$eval("#palette .palette-title", (x) => x.map((e) => e.textContent)), await stacked("#palette .palette-item"),
    await page.$eval("#palette", (d) => Math.round(d.getBoundingClientRect().width)), await page.$eval("#palette .palette-foot", (f) => f.getBoundingClientRect().top >= document.querySelector("#palette .palette-list")!.getBoundingClientRect().bottom)])
    .toEqual([["Projects", "Characters"], ["No project", "Mira"], true, 576, true]);
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => !document.querySelector("#palette[open]") && document.querySelector<HTMLTextAreaElement>("#input")?.placeholder === "Message Mira");
  await page.click("#side-scope");
  await page.waitForSelector("#scope-pop:not([hidden]) .scope-item");
  expect([await page.$$eval("#scope-pop .scope-name", (x) => x.map((e) => e.textContent)), await stacked("#scope-pop .scope-item"), await page.$eval("#scope-pop", (d) => Math.round(d.getBoundingClientRect().width))]).toEqual([["All projects", "Chrysalis", "Mira"], true, 288]);
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#scope-pop .scope-item")].find((b) => b.textContent === "Mira")!.click());
  expect(await page.$$eval("#side-list .folder .cast-name", (x) => x.map((e) => e.textContent))).toEqual(["Mira"]);
  await page.close();
}, 90_000);
