import { expect, test } from "bun:test";
import { mkdirSync } from "node:fs";
import type { Page } from "puppeteer-core";
import { harness, card, castRows, dir, home, inRow, lastStream, open, opencode, png, send, spokenOf, streams, voicePage } from "./browser-harness";

harness();

const work = `${dir}/site`;
mkdirSync(work, { recursive: true });

async function connect(page: Page) {
  await page.click("#opencode");
  await page.waitForSelector("#oc-pop .invite-head");
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#oc-pop .invite-card")].find((b) => b.textContent === "Connect")?.click());
  await page.waitForFunction(() => /OpenCode 2/.test(document.querySelector("#oc-pop .invite-head")?.textContent ?? ""));
}
async function openFolder(page: Page, path: string) {
  await page.keyboard.press("Escape");
  await page.click("#add-cast");
  await page.click('#cast-add [data-add="folder"]');
  await page.waitForSelector("#oc-pop .oc-path");
  const at = () => page.$eval("#oc-pop .oc-path", (e) => e.textContent!);
  const into = async (name: string, want: string) => {
    await page.evaluate((name) => [...document.querySelectorAll<HTMLElement>("#oc-pop .invite-row")].find((r) => r.querySelector(".invite-name")!.textContent === name)!.click(), name);
    await page.waitForFunction((want) => document.querySelector("#oc-pop .oc-path")?.textContent === want, {}, want);
  };
  while ((await at()) !== "/") { const here = await at(); await into("..", here.replace(/\/[^/]+$/, "") || "/"); }
  let walked = "";
  for (const part of path.split("/").filter(Boolean)) await into(part, (walked += `/${part}`));
  await page.click("#oc-pop .oc-use");
  await page.waitForFunction(() => document.querySelector("main")!.hasAttribute("data-thread"));
}
const cards = (page: Page) => page.$$eval("#thread .tool-card", (c) => c.map((x) => [x.querySelector(".tool-name")!.textContent, x.querySelector(".tool-src")!.textContent, (x as HTMLElement).dataset.state]));
const lastText = (page: Page) => page.$eval("#thread > article.assistant:last-child > .markdown", (e) => (e as HTMLElement).innerText.trim());

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

test("OpenCode connects from its button, a folder opened there is a row, and a thread is a chat OpenCode answers: it gets the message and Chrysalis's own model (never one elsewhere unasked), its steps are cards, what it asks waits for an answer that goes back to it, and the thread comes back from OpenCode after a reload, and another tab shows its folder and thread in the sidebar without being asked", async () => {
  const page = await open(), prompts = opencode.log.prompts.length, chats = streams.length;
  await connect(page);
  expect(await page.$eval("#opencode .dot", (d) => d.classList.contains("online"))).toBe(true);
  await openFolder(page, work);
  expect([await castRows(page), await page.$eval("#input", (e) => (e as HTMLTextAreaElement).placeholder), await page.$eval("#model-btn .model-name", (e) => e.textContent)])
    .toEqual([["Chrysalis", "site"], "Message Chrysalis", "27B"]);
  await send(page, "what is in the note");
  const sent = opencode.log.prompts.at(-1)!;
  expect([opencode.log.prompts.length - prompts, sent.text, sent.model, /^You can make artifacts/.test(sent.persona ?? "") && !/no tools/.test(sent.persona!), streams.length - chats])
    .toEqual([1, "what is in the note", { providerID: "chrysalis", id: "qwen3.8-27b" }, true, 0]);
  expect([await cards(page), await page.$eval("#title", (e) => e.textContent)]).toEqual([[["Read", "OpenCode", "done"]], "what is in the note"]);
  expect(await page.$eval("#thread > article.assistant:last-child", (e) => (e as HTMLElement).innerText)).toContain("The file says hello and honey. You asked: what is in the note");

  await page.type("#input", "ask first, then read it");
  await page.keyboard.press("Enter");
  await page.waitForSelector(".tool-ask:not([hidden])");
  expect([await page.$$eval("#conversations .conv-status", (s) => s.map((x) => x.textContent)), opencode.log.replies.length]).toEqual([["Needs you"], 0]);
  await page.click('.tool-ask [data-decision="deny"]');
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send");
  expect([opencode.log.replies.at(-1), (await cards(page)).at(-1)]).toEqual(["reject", ["Read", "OpenCode", "declined"]]);
  expect(await page.$eval("#thread > article.assistant:last-child", (e) => (e as HTMLElement).innerText)).toContain("I did not read it.");

  await page.reload({ waitUntil: "load" });
  await page.waitForFunction(() => document.querySelectorAll("#thread > article").length === 4);
  expect((await cards(page)).map((c) => c[2])).toEqual(["done", "failed"]);
  const other = await open();
  await other.waitForSelector("#conversations .cast-row .face.dir");
  expect([await castRows(other), await other.$eval("#opencode .dot", (d) => d.classList.contains("online"))]).toEqual([["Chrysalis", "site"], true]);
  await inRow(other, "site", ".cast-main");
  await other.waitForSelector("#conversations .folder-chats a.conv");
  expect(await other.$$eval("#conversations .folder-chats .conv-title", (t) => t.map((x) => x.textContent))).toContain("what is in the note");
  await other.evaluate(() => [...document.querySelectorAll<HTMLElement>("#conversations .folder-chats a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "what is in the note")!.click());
  await other.waitForFunction(() => document.querySelectorAll("#thread > article").length === 4);
  expect(await other.$eval("#thread > article.assistant:last-child", (e) => (e as HTMLElement).innerText)).toContain("I did not read it.");
  const had = opencode.sessions.size;
  await send(other, "and again");
  expect([opencode.log.prompts.at(-1)!.session, opencode.sessions.size - had]).toEqual([sent.session, 0]);
  for (const p of [page, other]) await p.close();
}, 90_000);

test("a character works in a thread from the folder's invite: her card goes to OpenCode as that thread's instruction and her face heads the reply, taking her out makes it plain again; a model picked from OpenCode's list is the one sent; Stop interrupts OpenCode's turn", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mira.png`, await card("Mira", { description: "{{char}} keeps bees." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mira.png`);
  await page.waitForFunction(() => [...document.querySelectorAll("#conversations .cast-row .cast-name")].some((n) => n.textContent === "Mira"));
  await connect(page);
  await openFolder(page, work);
  const pick = (name: string) => page.evaluate((name) => [...document.querySelectorAll<HTMLElement>("#cast-invite .invite-row")].find((r) => r.querySelector(".invite-name")!.textContent === name)!.click(), name);
  await inRow(page, "site", ".cast-bring");
  expect(await page.$eval("#cast-invite .invite-head", (e) => e.textContent)).toBe("Who works in this thread");
  await pick("Mira");
  await page.keyboard.press("Escape");
  expect(await page.$eval("#input", (e) => (e as HTMLTextAreaElement).placeholder)).toBe("Message Mira");
  await send(page, "tidy the note");
  const first = opencode.log.prompts.at(-1)!;
  expect([/Write Mira's next reply in a fictional chat between Mira and User/.test(first.persona ?? ""), /Mira keeps bees\./.test(first.persona ?? ""), /can use the tools here/.test(first.persona ?? "")]).toEqual([true, true, true]);
  expect([await page.$eval("#thread > article.assistant:last-child .speaker", (e) => e.textContent),
    (await page.$eval("#thread > article.assistant:last-child", (e) => (e as HTMLElement).innerText)).includes("Mira here. The file says")]).toEqual(["Mira", true]);

  await inRow(page, "site", ".cast-bring");
  await pick("Mira");
  await page.keyboard.press("Escape");
  await page.click("#model-btn");
  await page.waitForSelector("#model-pop:not([hidden]) .model-group");
  expect(await page.$$eval("#model-pop .model-group, #model-pop .model-label", (x) => x.map((e) => e.textContent))).toEqual(["OpenCode", "Build", "Plan", "Chrysalis", "Qwen3.8 27B", "Elsewhere", "Cloud One"]);
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#model-pop .model-item")].find((i) => i.textContent!.includes("Cloud One"))!.click());
  await send(page, "and once more");
  const second = opencode.log.prompts.at(-1)!;
  expect([second.session, /Mira/.test(second.persona ?? ""), second.model]).toEqual([first.session, false, { providerID: "elsewhere", id: "cloud-1" }]);
  expect(await page.$eval("#thread > article.assistant:last-child .speaker", (e) => e.textContent)).toBe("Chrysalis");

  const stops = opencode.log.interrupts;
  await page.type("#input", "read it slowly");
  await page.keyboard.press("Enter");
  await page.waitForSelector('#thread .tool-card[data-state="running"]');
  await page.click("#send");
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send");
  expect(opencode.log.interrupts).toBe(stops + 1);
  await page.close();
}, 90_000);

test("a thread takes what a chat takes: attached text files and pictures go to OpenCode with the message and come back with the thread after a reload, a video does not; the thinking slider picks the model's own level and hides for a model with none; a question OpenCode asks is answered or skipped in its card", async () => {
  const page = await open(), last = () => opencode.log.prompts.at(-1)!;
  await Bun.write(`${dir}/jar.txt`, "clover honey, 12 jars");
  await Bun.write(`${dir}/jar.png`, png);
  await Bun.write(`${dir}/jar.mp4`, "not really a film");
  await connect(page);
  await openFolder(page, work);
  const file = (await page.$("input#attach-file"))!;
  await file.uploadFile(`${dir}/jar.txt`, `${dir}/jar.png`);
  await page.waitForFunction(() => document.querySelectorAll("#attachments .att").length === 2);
  await file.uploadFile(`${dir}/jar.mp4`);
  await page.waitForFunction(() => /OpenCode gets text files, PDFs and pictures/.test(document.querySelector("#toast span")?.textContent ?? ""));
  await page.click('#energy-stops [data-stop="0"]');
  await send(page, "read these");
  const [text, picture] = last().files!;
  expect([last().files!.length, text.name, Buffer.from(text.uri.slice(text.uri.indexOf(",") + 1), "base64").toString(), picture.name, picture.uri.startsWith("data:image/png;base64,"), last().model])
    .toEqual([2, "jar.txt", "clover honey, 12 jars", "jar.png", true, { providerID: "chrysalis", id: "qwen3.8-27b", variant: "none" }]);
  await page.click('#energy-stops [data-stop="2"]');
  await send(page, "and now");
  expect([last().files, last().model]).toEqual([undefined, { providerID: "chrysalis", id: "qwen3.8-27b", variant: "medium" }]);

  await page.reload({ waitUntil: "load" });
  await page.waitForFunction(() => document.querySelectorAll("#thread > article").length === 4);
  await page.waitForFunction(() => (document.querySelector<HTMLImageElement>("#thread > article.user .msg-files img")?.naturalWidth ?? 0) > 0);
  expect(await page.$eval("#thread > article.user .msg-files", (b) => [b.querySelector(".att-name")!.textContent, b.querySelector("img")!.title])).toEqual(["jar.txt", "jar.png"]);

  const asked = async (pick?: string) => {
    await page.type("#input", "which one should I read");
    await page.keyboard.press("Enter");
    await page.waitForSelector(".tool-ask:not([hidden]) .ask-form");
    const seen = [await page.$eval(".ask-form .ask-title", (e) => e.textContent), await page.$$eval(".ask-form .ask-option", (o) => o.map((x) => x.textContent)),
      await page.$$eval("#conversations .conv-status", (s) => s.map((x) => x.textContent))];
    if (pick) await page.evaluate((pick) => [...document.querySelectorAll<HTMLElement>(".ask-form .ask-option")].find((o) => o.textContent === pick)!.click(), pick);
    await page.click(pick ? '.ask-form button[type="submit"]' : '.ask-form [data-decision="deny"]');
    await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send");
    return seen;
  };
  expect(await asked("Heather")).toEqual(["Which jar", ["Clover", "Heather"], ["Needs you"]]);
  expect([opencode.log.answers.at(-1), (await lastText(page)).includes("You chose Heather.")]).toEqual([{ q0: "Heather" }, true]);
  await asked();
  expect([opencode.log.answers.at(-1), (await lastText(page)).includes("You did not choose.")]).toEqual(["skipped", true]);

  const slider = () => page.$eval(".think-ctl", (e) => getComputedStyle(e).display !== "none");
  expect(await slider()).toBe(true);
  await page.click("#model-btn");
  await page.waitForSelector("#model-pop:not([hidden]) .model-group");
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#model-pop .model-item")].find((i) => i.textContent!.includes("Cloud One"))!.click());
  expect(await slider()).toBe(false);
  await send(page, "plainly");
  expect(last().model).toEqual({ providerID: "elsewhere", id: "cloud-1" });
  await page.evaluate(() => localStorage.removeItem("thread-model"));
  await page.close();
}, 90_000);

test("a routine made by asking is listed in its thread, where it can be paused and deleted; when OpenCode wakes the thread by itself, the open page shows the routine's message and the reply as it is written, and another chat's sidebar shows the thread working", async () => {
  const page = await open();
  await connect(page);
  await openFolder(page, work);
  expect(await page.$("#routines .count")).toBe(null);
  await send(page, "count the bees every morning");
  const session = opencode.log.prompts.at(-1)!.session;
  await page.waitForSelector("#routines .count");
  await page.click("#routines");
  await page.waitForSelector("#routines-pop .routine");
  const rows = () => page.$$eval("#routines-pop .routine", (r) => r.map((x) => [x.querySelector(".invite-name")!.textContent, x.querySelector(".routine-when")!.textContent!.split(" · ")[0], x.querySelector("button")!.title]));
  expect(await rows()).toEqual([["morning bees", "cron 0 9 * * *", "Pause"]]);
  await page.click('#routines-pop .routine button[title="Pause"]');
  await page.waitForSelector('#routines-pop .routine button[title="Resume"]');
  expect([opencode.log.routines.at(-1), await page.$eval("#routines-pop .routine-when", (e) => e.textContent)]).toEqual(["pause morning bees", "cron 0 9 * * * · paused"]);
  await page.keyboard.press("Escape");

  const before = await page.$$eval("#thread > article", (a) => a.length), prompts = opencode.log.prompts.length;
  void opencode.wake(session, "morning bees", "Count the bees very slowly.");
  await page.waitForFunction((n) => document.querySelectorAll("#thread > article").length === n + 2, { timeout: 15_000 }, before);
  await page.waitForSelector('#thread .tool-card[data-state="running"]');
  expect(await page.$eval("#thread > article.user:nth-last-child(2)", (e) => [e.querySelector(".point-chip")!.textContent, e.querySelector(".bubble, .markdown")!.textContent!.trim()]))
    .toEqual(["Routine: morning bees", "Count the bees very slowly."]);
  expect(await page.$eval("#send", (e) => e.getAttribute("aria-label"))).toBe("Stop");
  const written = (text: string) => page.waitForFunction((text) => document.querySelector<HTMLElement>("#thread > article.assistant:last-child > .markdown")?.innerText.includes(text)
    && document.querySelector("#send")?.getAttribute("aria-label") === "Send", { timeout: 20_000 }, text);
  await written("You asked: Count the bees very slowly.");
  expect(opencode.log.prompts.length - prompts).toBe(0);

  await page.click("#new-chat");
  void opencode.wake(session, "morning bees", "Count them very slowly again.");
  await page.waitForFunction(() => [...document.querySelectorAll("#conversations a.conv")].some((a) => a.querySelector(".conv-title")!.textContent === "count the bees every morning" && /Working/.test(a.querySelector(".conv-status")?.textContent ?? "")),
    { timeout: 15_000 });
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#conversations a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "count the bees every morning")!.click());
  await page.waitForFunction((n) => document.querySelectorAll("#thread > article").length === n + 4, { timeout: 15_000 }, before);
  await written("You asked: Count them very slowly again.");

  const posts = opencode.log.routines.filter((r) => r.startsWith("POST")).length;
  await page.evaluate(() => document.querySelector("#thread > article.user")!.querySelector<HTMLElement>('[data-act="trash"]')!.click());
  await page.waitForFunction((n) => document.querySelectorAll("#thread > article").length === n + 3, {}, before);
  await page.waitForSelector("#routines .count");
  for (let k = 0; k < 100 && opencode.log.routines.at(-1) !== "pause morning bees"; k++) await Bun.sleep(50);
  await Bun.sleep(300);
  expect([opencode.log.routines.filter((r) => r.startsWith("POST")).slice(posts), await page.$eval("#routines .count", (c) => c.textContent)]).toEqual([['POST morning bees {"cron":"0 9 * * *"}'], "1"]);
  await page.click("#routines");
  await page.waitForSelector('#routines-pop .routine button[title="Resume"]');
  await page.evaluate(() => document.querySelector<HTMLElement>('#routines-pop .routine button[title="Delete this routine"]')!.click());
  await page.waitForFunction(() => !document.querySelector("#routines-pop .routine") && /None yet/.test(document.querySelector("#routines-pop .oc-note")?.textContent ?? ""));
  expect([opencode.log.routines.at(-1), await page.$("#routines .count")]).toEqual(["DELETE morning bees", null]);
  await page.close();
}, 90_000);

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
    .toEqual([["running", "running · N s · Shell"], '{ "shell": [ "ls jars" ] }', ["Needs you"]]);
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
  for (let k = 0; k < 2; k++) await page.evaluate(() => [...document.querySelectorAll("#conversations a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "count the jars with a helper")!.querySelector<HTMLElement>(".conv-del")!.click());
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

test("a thread is OpenCode's to change: a message sent while a turn runs goes to OpenCode and shows once the turn ends, Fork opens a new chat holding what came before a message, a reply edited in place and a message deleted are changed in OpenCode, an edited message sent again and Regenerate take the chat back and answer anew, and settling archives the thread in OpenCode so another tab shelves it too", async () => {
  const page = await open(), users = () => page.$$eval("#thread > article.user .bubble", (b) => b.map((x) => x.textContent!.trim()));
  const act = (text: string, name: string) => page.evaluate((text, name) => [...document.querySelectorAll("#thread > article.user")].find((a) => a.querySelector(".bubble")!.textContent!.trim() === text)!
    .querySelector<HTMLElement>(`[data-act="${name}"]`)!.click(), text, name);
  await connect(page);
  await openFolder(page, work);
  await send(page, "sort the first shelf");
  const session = opencode.log.prompts.at(-1)!.session, said = () => opencode.sessions.get(session)!.messages.filter((m) => m.type === "user").map((m) => m.text);
  await page.type("#input", "sort the second shelf slowly");
  await page.keyboard.press("Enter");
  await page.waitForSelector('#thread .tool-card[data-state="running"]');
  await page.type("#input", "and dust it");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /Sent: OpenCode reads it/.test(document.querySelector("#toast span")?.textContent ?? ""));
  expect([said(), await page.$eval("#send", (e) => e.getAttribute("aria-label")), await page.$eval("#input", (e) => (e as HTMLTextAreaElement).value)])
    .toEqual([["sort the first shelf", "sort the second shelf slowly", "and dust it"], "Stop", ""]);
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send" && document.querySelectorAll("#thread > article.user").length === 3, { timeout: 20_000 });
  expect(await users()).toEqual(["sort the first shelf", "sort the second shelf slowly", "and dust it"]);

  const before = opencode.sessions.size;
  await act("sort the second shelf slowly", "fork");
  await page.waitForFunction(() => document.querySelectorAll("#thread > article.user").length === 1 && /\(fork\)$/.test(document.querySelector("#title")?.textContent ?? ""));
  const forked = [...opencode.sessions.values()].at(-1)!;
  expect([opencode.sessions.size - before, await users(), forked.messages.filter((m) => m.type === "user").map((m) => m.text), said().length]).toEqual([1, ["sort the first shelf"], ["sort the first shelf"], 3]);

  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#conversations a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "sort the first shelf")!.click());
  await page.waitForFunction(() => document.querySelectorAll("#thread > article.user").length === 3);
  const replies = () => opencode.sessions.get(session)!.messages.filter((m) => m.type === "assistant").flatMap((m) => m.content.filter((p: { type: string }) => p.type === "text").map((p: { text: string }) => p.text));
  const editor = async (article: string, text: string, button: string) => {
    await page.evaluate((article) => document.querySelector(article)!.querySelector<HTMLElement>('[data-act="edit"]')!.click(), article);
    await page.waitForSelector("#thread .editor textarea");
    await page.$eval("#thread .editor textarea:last-of-type", (t, text) => { (t as HTMLTextAreaElement).value = text; t.dispatchEvent(new Event("input", { bubbles: true })); }, text);
    await page.evaluate((button) => [...document.querySelectorAll<HTMLElement>("#thread .editor-acts button")].find((b) => b.textContent === button)!.click(), button);
  };
  await editor("#thread > article.assistant", "The first shelf is sorted.", "Save");
  await page.waitForFunction(() => document.querySelector<HTMLElement>("#thread > article.assistant > .markdown")?.innerText.trim() === "The first shelf is sorted.");
  expect([replies()[0], said().length, await page.$$eval("#thread > article", (a) => a.length)]).toEqual(["The first shelf is sorted.", 3, 5]);

  await act("and dust it", "trash");
  await page.waitForFunction(() => document.querySelectorAll("#thread > article.user").length === 2);
  expect([said(), await users()]).toEqual([["sort the first shelf", "sort the second shelf slowly"], ["sort the first shelf", "sort the second shelf slowly"]]);

  const asked = opencode.log.prompts.length;
  await editor("#thread > article.user:nth-of-type(3)", "sort the third shelf", "Send");
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send" && /You asked: sort the third shelf/.test([...document.querySelectorAll<HTMLElement>("#thread > article.assistant")].at(-1)?.innerText ?? ""), { timeout: 20_000 });
  expect([said(), opencode.log.prompts.length - asked, await page.$$eval("#thread > article", (a) => a.length), replies()[0]]).toEqual([["sort the first shelf", "sort the third shelf"], 1, 4, "The first shelf is sorted."]);

  await page.evaluate(() => [...document.querySelectorAll("#thread > article.assistant")].at(-1)!.querySelector<HTMLElement>('[data-act="retry"]')!.click());
  for (let k = 0; k < 200 && opencode.log.prompts.length - asked < 2; k++) await Bun.sleep(50);
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send" && document.querySelectorAll("#thread > article").length === 4, { timeout: 20_000 });
  expect([said(), opencode.log.prompts.length - asked, opencode.log.prompts.at(-1)!.text]).toEqual([["sort the first shelf", "sort the third shelf"], 2, "sort the third shelf"]);

  await page.evaluate(() => [...document.querySelectorAll("#conversations .folder:not(.settled) a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "sort the first shelf")!.querySelector<HTMLElement>(".conv-settle")!.click());
  await page.waitForFunction(() => !!document.querySelector("#conversations .settled-head"));
  expect(!!opencode.sessions.get(session)!.archived).toBe(true);
  const other = await open();
  await other.waitForSelector("#conversations .settled-head");
  await other.click("#conversations .settled-head");
  expect(await other.$$eval("#conversations .folder.settled .conv-title", (t) => t.map((x) => x.textContent))).toContain("sort the first shelf");
  await other.evaluate(() => [...document.querySelectorAll("#conversations .folder.settled a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "sort the first shelf")!.querySelector<HTMLElement>(".conv-settle")!.click());
  await other.waitForFunction(() => ![...document.querySelectorAll("#conversations .folder.settled .conv-title")].some((t) => t.textContent === "sort the first shelf"));
  expect(!!opencode.sessions.get(session)!.archived).toBe(false);
  for (const p of [page, other]) await p.close();
}, 90_000);

test("while OpenCode is connected, a chat with Chrysalis is OpenCode at work in the home folder: a new chat's message goes to OpenCode there and its steps are cards, the chat is listed under Chrysalis and not as a folder, voice mode asks OpenCode and speaks its reply with the model picker and thinking slider still in a narrow message box, and once disconnected a new chat is a plain one again", async () => {
  const page = await open(), last = () => opencode.log.prompts.at(-1)!, chats = streams.length;
  await page.waitForSelector("#opencode .dot.online");
  await page.click("#new-chat");
  await send(page, "how much space is left on the computer");
  expect([last().text, opencode.sessions.get(last().session)!.dir, streams.length - chats, await cards(page)]).toEqual(["how much space is left on the computer", home, 0, [["Read", "OpenCode", "done"]]]);
  expect([(await castRows(page)).includes(home.split("/").pop()!), await page.$$eval("#conversations .folder.open .conv-title", (t) => t.map((x) => x.textContent))]).toEqual([false, ["how much space is left on the computer"]]);
  await page.close();

  const spoke = opencode.log.prompts.length, made = opencode.sessions.size, { page: talk, session } = await voicePage("pie");
  await talk.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 20_000 });
  const heard = opencode.log.prompts.slice(spoke);
  expect([heard.map((p) => p.text.split("\n\n").at(-1)), heard[0].text.startsWith("[The user is talking to you by voice"), heard[0].meta, opencode.sessions.size - made, opencode.sessions.get(last().session)!.dir,
    streams.slice(chats).filter((b) => b.stream).length]).toEqual([["lets do a snake game in Pygame"], true, { said: "lets do a snake game in Pygame" }, 1, home, 0]);
  for (const width of [760, 390]) {
    await talk.setViewport({ width, height: 800 });
    await Bun.sleep(500);
    expect(await talk.evaluate(() => [...["#model-btn", ".think-ctl", "#voice-pill"].map((q) => { const r = document.querySelector(q)!.getBoundingClientRect(); return r.width > 0 && r.left >= 0 && r.right <= innerWidth; }),
      document.querySelector(".composer-box")!.scrollWidth <= document.querySelector(".composer-box")!.clientWidth])).toEqual([true, true, true, true]);
  }
  await talk.reload({ waitUntil: "load" });
  await talk.waitForFunction(() => document.querySelectorAll("#thread > article").length === 2);
  expect(await talk.$eval("#thread > article.user .bubble", (b) => b.textContent)).toBe("lets do a snake game in Pygame");
  const { said, cancelled } = spokenOf(session());
  expect([...said].filter(([id]) => !cancelled.has(id)).map(([, t]) => t).join(" ")).toContain("You asked:");
  await talk.close();

  const plain = await open();
  await plain.click("#opencode");
  await plain.waitForFunction(() => [...document.querySelectorAll("#oc-pop .invite-card")].some((b) => b.textContent === "Disconnect"));
  await plain.evaluate(() => [...document.querySelectorAll<HTMLElement>("#oc-pop .invite-card")].find((b) => b.textContent === "Disconnect")!.click());
  await plain.waitForFunction(() => !document.querySelector("#opencode .dot.online"));
  const asked = opencode.log.prompts.length;
  await send(plain, "good evening");
  expect([opencode.log.prompts.length - asked, streams.length - chats > 0, await plain.$eval("main", (m) => m.hasAttribute("data-thread"))]).toEqual([0, true, false]);
  await plain.close();
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
