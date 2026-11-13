import { expect, test } from "bun:test";
import type { Page } from "puppeteer-core";
import { card, connect, dir, harness, home, inRow, lastStream, lastText, menu, open, openFolder, opencode, send, streams, work } from "./browser-harness";

harness();

test("settling puts a chat or a thread away on a shelf at the bottom, Undo and the shelf's own button bring it back, and it stays settled after a reload", async () => {
  const page = await open();
  await send(page, "good morning mirela");
  await connect(page);
  await openFolder(page, work);
  await send(page, "count the bees please");
  const titles = (sel: string) => page.$$eval(`#conversations ${sel} .conv-title`, (t) => t.map((x) => x.textContent));
  const settle = (title: string, sel = ".folder:not(.settled)") => page.evaluate((title, sel) => [...document.querySelectorAll(`#conversations ${sel} a.conv`)]
    .find((a) => a.querySelector(".conv-title")!.textContent === title)!.querySelector<HTMLElement>(".conv-settle")!.click(), title, sel);
  const mine = (await titles(".folder:not(.settled)")).filter((t) => t === "A title" || t === "count the bees please");
  expect(mine.sort()).toEqual(["A title", "count the bees please"]);
  await settle("A title");
  expect([await titles(".folder:not(.settled)").then((t) => t.includes("A title")), await page.$eval("#conversations .settled-head", (h) => h.textContent), await page.$eval("#toast span", (t) => t.textContent)])
    .toEqual([false, "Settled1", "A title is settled"]);
  await page.click("#toast button");
  expect([await titles(".folder:not(.settled)").then((t) => t.includes("A title")), await page.$("#conversations .settled-head")]).toEqual([true, null]);
  await settle("A title");
  await settle("count the bees please");
  await page.click("#conversations .settled-head");
  expect((await titles(".folder.settled")).sort()).toEqual(["A title", "count the bees please"]);
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#conversations .settled-head");
  await connect(page);
  await page.keyboard.press("Escape");
  await page.waitForFunction(() => document.querySelectorAll("#conversations .folder.settled a.conv").length === 2);
  await settle("count the bees please", ".folder.settled");
  await page.waitForFunction(() => document.querySelectorAll("#conversations .folder.settled a.conv").length === 1);
  expect([await titles(".folder.settled"), (await titles(".folder:not(.settled)")).includes("count the bees please")]).toEqual([["A title"], true]);
  await page.close();
}, 60_000);

test("in a thread a page is a file OpenCode writes: its write step carries the page as an artifact, an edit of the file is its next version, which the panel draws, and both come back from OpenCode after a reload", async () => {
  const page = await open();
  await connect(page);
  await openFolder(page, work);
  await send(page, "make a page file");
  const arts = () => page.$$eval("#thread .tool-card", (c) => c.map((x) => [x.querySelector(".tool-name")!.textContent, x.querySelector(".art-card .art-name")?.textContent ?? null,
    x.querySelector(".art-card .art-sub")?.textContent ?? null]));
  await page.waitForFunction(() => document.querySelectorAll("#thread .tool-card .art-card").length === 2);
  const want = [["Write", "counter.html", "Page · v1 · now v2"], ["Edit", "counter.html", "Page · v2 · 1 change"]];
  expect(await arts()).toEqual(want);
  await page.click("#thread .tool-card:last-child .art-card");
  await page.waitForSelector(".shelf-stage .art-canvas iframe.page-frame");
  const title = async () => (await (await page.waitForSelector(".shelf-stage .art-canvas iframe.page-frame:not(.next)"))!.contentFrame())!.$eval("#title", (e) => e.textContent);
  await page.waitForFunction(() => document.querySelector(".shelf-stage .art-label")?.textContent?.startsWith("counter.html · v2"));
  expect(await title()).toBe("Taps");
  await page.reload();
  await page.waitForFunction(() => document.querySelectorAll("#thread .tool-card .art-card").length === 2, { timeout: 10_000 });
  expect(await arts()).toEqual(want);
  await page.close();
}, 30_000);

test("a page OpenCode is still working on says so until its turn ends: its step cards and the panel show it being written, its script waits, and the version the turn ends with is the one that runs", async () => {
  const page = await open();
  await connect(page);
  await openFolder(page, work);
  await page.type("#input", "make a page file until released");
  await page.keyboard.press("Enter");
  const subs = () => page.$$eval("#thread .tool-card .art-card .art-sub", (c) => c.map((x) => x.textContent));
  await page.waitForFunction(() => document.querySelectorAll("#thread .tool-card .art-card").length === 2);
  expect(await subs()).toEqual(["Page · v1 · Writing the page…", "Page · v2 · Changing, 1 so far…"]);
  await page.click("#thread .tool-card:last-child .art-card");
  await page.waitForSelector(".shelf-stage .page-fig.drawing .svg-bar > .page-writing");
  opencode.release();
  await page.waitForFunction(() => !document.querySelector(".shelf-stage .page-fig.drawing") && document.querySelectorAll("#thread .tool-card .art-card:not(.live)").length === 2, { timeout: 10_000 });
  expect(await subs()).toEqual(["Page · v1 · now v2", "Page · v2 · 1 change"]);
  const frame = (await (await page.waitForSelector(".shelf-stage .art-canvas iframe.page-frame:not(.next)"))!.contentFrame())!;
  expect(await frame.$eval("#title", (e) => e.textContent)).toBe("Taps");
  await page.close();
}, 30_000);

test("what OpenCode does shows in the thread without Chrysalis doing it: a sub-agent is a card that says what it is at and what it asks is answered there and reaches the sub-agent, one left in the background keeps the thread working until its report is written, a slash command and the agent picked go to OpenCode, a thread deleted in the sidebar is deleted in OpenCode, and the OpenCode button disconnects it (folders gone, also after a reload) and connects it again", async () => {
  const page = await open(), last = () => opencode.log.prompts.at(-1)!;
  const sub = () => page.$eval('#thread .tool-card:has(.tool-name)', () => [...document.querySelectorAll<HTMLElement>("#thread .tool-card")].filter((c) => c.querySelector(".tool-name")!.textContent === "Subagent")
    .map((c) => [c.dataset.state, c.querySelector(".tool-state")!.textContent!.replace(/[\d.]+ s/, "N s")]).at(-1));
  await connect(page);
  await openFolder(page, work);
  await page.type("#input", "count the jars with a helper");
  await page.keyboard.press("Enter");
  await page.waitForSelector(".tool-ask:not([hidden]) [data-decision]");
  expect([await sub(), await page.$eval(".tool-ask:not([hidden]) pre", (e) => e.textContent!.replace(/\s+/g, " ")), await page.$$eval("#conversations .conv-status", (s) => s.map((x) => x.textContent))])
    .toEqual([["running", "running · N s · Shell"], '{ "shell": [ "ls jars" ] }', ["Approval"]]);
  await page.click('.tool-ask:not([hidden]) [data-decision="once"]');
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send");
  expect([opencode.log.replies.at(-1), (await sub())![0], await lastText(page)]).toEqual(["to the helper: once", "done", "The file says hello and honey. You asked: count the jars with a helper"]);

  await page.type("#input", "count them again with a helper in the background");
  await page.keyboard.press("Enter");
  await page.waitForSelector(".tool-ask:not([hidden]) [data-decision]");
  expect(await page.$eval("#thread > article.assistant:last-child", (e) => (e as HTMLElement).innerText)).toContain("You asked: count them again with a helper");
  expect(await page.$eval("#send", (e) => e.getAttribute("aria-label"))).toBe("Stop");
  await page.click('.tool-ask:not([hidden]) [data-decision="once"]');
  await page.waitForFunction(() => document.querySelector<HTMLElement>("#thread > article.assistant:last-child > .markdown")?.innerText.trim() === "The helper counted 12 jars."
    && document.querySelector("#send")?.getAttribute("aria-label") === "Send", { timeout: 20_000 });
  await page.reload({ waitUntil: "load" });
  await page.waitForFunction(() => document.querySelectorAll("#thread > article").length === 6);
  expect(await page.$eval("#thread > article.user:nth-last-child(2) .point-chip", (e) => e.textContent)).toBe("Sub-agent: Explore");

  await send(page, "/review the last change");
  expect([last().command, last().text, last().agent]).toEqual(["review", "the last change", undefined]);
  await send(page, "/tmp is a folder");
  expect([last().command, last().text]).toEqual([undefined, "/tmp is a folder"]);
  await page.click("#model-btn");
  await page.waitForSelector("#model-pop:not([hidden]) .model-group");
  expect(await page.$$eval("#model-pop .model-group, #model-pop .model-label", (x) => x.map((e) => e.textContent)).then((all) => all.slice(0, 3))).toEqual(["OpenCode", "Build", "Plan"]);
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#model-pop .model-item")].find((i) => i.textContent === "Plan")!.click());
  await send(page, "plan the shelves");
  expect(last().agent).toBe("plan");

  const session = last().session, row = () => page.evaluate(() => [...document.querySelectorAll("#conversations a.conv")].map((a) => a.querySelector(".conv-title")!.textContent));
  expect([opencode.sessions.has(session), (await row()).includes("count the jars with a helper")]).toEqual([true, true]);
  await menu(page, "count the jars with a helper", "Delete");
  await page.waitForFunction(() => ![...document.querySelectorAll("#conversations a.conv .conv-title")].some((t) => t.textContent === "count the jars with a helper"));
  await page.reload({ waitUntil: "load" });
  await connect(page);
  expect([opencode.sessions.has(session), (await row()).includes("count the jars with a helper")]).toEqual([false, false]);

  const folders = () => page.$$eval("#conversations .cast-row .face.dir", (f) => f.length);
  expect(await folders()).toBeGreaterThan(0);
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#oc-pop .invite-card")].find((b) => b.textContent === "Disconnect")!.click());
  await page.waitForFunction(() => !document.querySelector("#conversations .cast-row .face.dir") && !document.querySelector("#opencode .dot.online"));
  await page.reload({ waitUntil: "load" });
  await page.click("#opencode");
  await page.waitForFunction(() => [...document.querySelectorAll("#oc-pop .invite-card")].some((b) => b.textContent === "Connect"));
  expect([await folders(), await page.$eval("#opencode .dot", (d) => d.classList.contains("online"))]).toEqual([0, false]);
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#oc-pop .invite-card")].find((b) => b.textContent === "Connect")!.click());
  await page.waitForSelector("#conversations .cast-row .face.dir");
  expect(await page.$eval("#opencode .dot", (d) => d.classList.contains("online"))).toBe(true);
  await page.close();
}, 90_000);

test("a character's chat and a group's are the same system: with OpenCode connected a chat with a character goes to OpenCode with her card and greeting, a second character makes it a group whose next speaker OpenCode is told, and another tab finds the chat under the same people with its greeting, clean messages and speakers; Chat only makes a new chat a plain one", async () => {
  const speaker = (p: Page) => p.evaluate(() => [...document.querySelectorAll("#thread > article.assistant")].at(-1)!.querySelector(".speaker")?.textContent);
  const said = (p: Page) => p.evaluate(() => [...document.querySelectorAll("#thread > article.assistant")].at(-1)!.querySelector<HTMLElement>(":scope > .markdown")!.innerText.trim());
  const page = await open(), last = () => opencode.log.prompts.at(-1)!, scene = () => (opencode.sessions.get(last().session)!.metadata as { chrysalis: object }).chrysalis;
  await Bun.write(`${dir}/Mira2.png`, await card("Mira", { description: "{{char}} keeps bees.", first_mes: "Welcome to the hive." }));
  await Bun.write(`${dir}/Otto.png`, await card("Otto", { description: "{{char}} mends boats." }));
  for (const f of ["Mira2", "Otto"]) await (await page.$("input#cast-file"))!.uploadFile(`${dir}/${f}.png`);
  await page.waitForFunction(() => ["Mira", "Otto"].every((n) => [...document.querySelectorAll("#conversations .cast-row .cast-name")].some((x) => x.textContent === n)));
  await connect(page);
  await page.keyboard.press("Escape");
  await inRow(page, "Otto", ".cast-new");
  await inRow(page, "Mira", ".cast-new");
  await page.waitForFunction(() => document.querySelector("main")!.hasAttribute("data-thread") && document.querySelector("#thread > article.assistant .markdown")?.textContent?.trim() === "Welcome to the hive.");
  const chats = streams.length;
  await send(page, "count my jars");
  expect([last().text, opencode.sessions.get(last().session)!.dir, opencode.sessions.get(last().session)!.title, scene(), streams.length - chats]).toEqual(["count my jars", home, "count my jars", { cast: ["Mira"], greeting: "Welcome to the hive." }, 0]);
  expect([/Write Mira's next reply/.test(last().persona!), /Mira keeps bees\./.test(last().persona!), /greeting:\nWelcome to the hive\./.test(last().persona!),
    await speaker(page), (await said(page)).startsWith("Mira here.")]).toEqual([true, true, true, "Mira", true]);

  await inRow(page, "Mira", ".cast-bring");
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#cast-invite .invite-row")].find((r) => r.querySelector(".invite-name")!.textContent === "Otto")!.click());
  await page.keyboard.press("Escape");
  await send(page, "@Otto what do you see");
  expect([last().text, last().meta, scene(), /mends boats/.test(last().persona!) && /keeps bees/.test(last().persona!), (await said(page)).startsWith("Otto\nOtto here.")])
    .toEqual(["@Otto what do you see\n\n[Otto next.]", { said: "@Otto what do you see", next: "Otto" }, { cast: ["Mira", "Otto"], greeting: "Welcome to the hive." }, true, true]);

  const other = await open();
  await other.waitForFunction(() => [...document.querySelectorAll("#conversations .cast-row .cast-name")].some((n) => n.textContent === "Mira and Otto"));
  await inRow(other, "Mira and Otto", ".cast-main");
  await other.waitForSelector("#conversations .folder-chats a.conv");
  await other.click("#conversations .folder-chats a.conv");
  await other.waitForFunction(() => document.querySelectorAll("#thread > article").length === 5);
  expect(await other.$$eval("#thread > article", (a) => a.map((x) => [x.querySelector(".speaker")?.textContent ?? "", (x.querySelector(".bubble") ?? x.querySelector(":scope > .markdown"))!.textContent!.trim().slice(0, 21)])))
    .toEqual([["", "Welcome to the hive."], ["User", "count my jars"], ["", "Mira here. The file s"], ["User", "@Otto what do you see"], ["", "Otto here. The file s"]]);
  await other.close();

  await inRow(page, "Otto", ".cast-new");
  await page.click("#model-btn");
  await page.waitForSelector("#model-pop:not([hidden]) .model-group");
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#model-pop .model-item")].find((i) => i.textContent === "Chat only")!.click());
  const asked = opencode.log.prompts.length, plain = streams.length;
  await send(page, "tell me of boats");
  expect([opencode.log.prompts.length - asked, streams.length - plain, await page.$eval("main", (m) => m.hasAttribute("data-thread")), /mends boats/.test(JSON.stringify(lastStream.messages[0]))]).toEqual([0, 1, false, true]);
  await page.evaluate(() => localStorage.setItem("tools", "on"));
  await page.close();
}, 90_000);
