import { expect, test } from "bun:test";
import { mkdirSync } from "node:fs";
import type { Page } from "puppeteer-core";
import { harness, card, castRows, dir, inRow, lastStream, open, opencode, send, streams } from "./browser-harness";

harness();

const work = `${dir}/site`;
mkdirSync(work, { recursive: true });

async function connect(page: Page) {
  await page.click("#opencode");
  await page.waitForFunction(() => /OpenCode 2/.test(document.querySelector("#oc-pop .invite-head")?.textContent ?? ""));
}
async function openFolder(page: Page, path: string) {
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#oc-pop .invite-card")].at(-1)!.click());
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
const lastText = (page: Page) => page.$eval("#thread > article.assistant:last-child .markdown:last-of-type", (e) => (e as HTMLElement).innerText.trim());

test("OpenCode connects from its button, a folder opened there is a row, and a thread is a chat OpenCode answers: it gets the message and Chrysalis's own model (never one elsewhere unasked), its steps are cards, what it asks waits for an answer that goes back to it, and the thread comes back from OpenCode after a reload and in another tab", async () => {
  const page = await open(), prompts = opencode.log.prompts.length, chats = streams.length;
  await connect(page);
  expect(await page.$eval("#opencode .dot", (d) => d.classList.contains("online"))).toBe(true);
  await openFolder(page, work);
  expect([await castRows(page), await page.$eval("#input", (e) => (e as HTMLTextAreaElement).placeholder), await page.$eval("#model-btn .model-name", (e) => e.textContent)])
    .toEqual([["Chrysalis", "site"], "Message Chrysalis", "27B"]);
  await send(page, "what is in the note");
  const sent = opencode.log.prompts.at(-1)!;
  expect([opencode.log.prompts.length - prompts, sent.text, sent.model, sent.persona, streams.length - chats])
    .toEqual([1, "what is in the note", { providerID: "chrysalis", id: "qwen3.8-27b" }, undefined, 0]);
  expect([await cards(page), await page.$eval("#title", (e) => e.textContent)]).toEqual([[["Read", "OpenCode", "done"]], "About what is in"]);
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
  await connect(other);
  await page.evaluate(() => 0);
  await other.evaluate((work) => localStorage.setItem("folders", JSON.stringify([work])), work);
  await other.reload({ waitUntil: "load" });
  await other.waitForSelector("#conversations .cast-row .face.dir");
  await inRow(other, "site", ".cast-main");
  await other.waitForSelector("#conversations .folder-chats a.conv");
  expect(await other.$$eval("#conversations .folder-chats .conv-title", (t) => t.map((x) => x.textContent))).toEqual(["About what is in"]);
  await other.click("#conversations .folder-chats a.conv");
  await other.waitForFunction(() => document.querySelectorAll("#thread > article").length === 4);
  expect(await other.$eval("#thread > article.assistant:last-child", (e) => (e as HTMLElement).innerText)).toContain("I did not read it.");
  await send(other, "and again");
  expect([opencode.log.prompts.at(-1)!.session, opencode.sessions.size]).toEqual([sent.session, 1]);
  for (const p of [page, other]) await p.close();
}, 90_000);

test("a character works in a thread from the folder's invite: her card goes to OpenCode as that thread's instruction and her face heads the reply, taking her out makes it plain again; a model picked from OpenCode's list is the one sent; Stop interrupts OpenCode's turn", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mira.png`, await card("Mira", { description: "{{char}} keeps bees." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mira.png`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 2);
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
  expect([/You are Mira, working for User/.test(first.persona ?? ""), /Mira keeps bees\./.test(first.persona ?? "")]).toEqual([true, true]);
  expect([await page.$eval("#thread > article.assistant:last-child .speaker", (e) => e.textContent),
    (await page.$eval("#thread > article.assistant:last-child", (e) => (e as HTMLElement).innerText)).includes("Mira here. The file says")]).toEqual(["Mira", true]);

  await inRow(page, "site", ".cast-bring");
  await pick("Mira");
  await page.keyboard.press("Escape");
  await page.click("#model-btn");
  await page.waitForSelector("#model-pop:not([hidden]) .model-group");
  expect(await page.$$eval("#model-pop .model-group, #model-pop .model-label", (x) => x.map((e) => e.textContent))).toEqual(["Chrysalis", "Qwen3.8 27B", "Elsewhere", "Cloud One"]);
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#model-pop .model-item")].find((i) => i.textContent!.includes("Cloud One"))!.click());
  await send(page, "and once more");
  const second = opencode.log.prompts.at(-1)!;
  expect([second.session, second.persona, second.model]).toEqual([first.session, undefined, { providerID: "elsewhere", id: "cloud-1" }]);
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

test("settling puts a chat or a thread away on a shelf at the bottom, Undo and the shelf's own button bring it back, and it stays settled after a reload", async () => {
  const page = await open();
  await send(page, "good morning mirela");
  await connect(page);
  await openFolder(page, work);
  await send(page, "count the bees please");
  const titles = (sel: string) => page.$$eval(`#conversations ${sel} .conv-title`, (t) => t.map((x) => x.textContent));
  const settle = (title: string, sel = ".folder:not(.settled)") => page.evaluate((title, sel) => [...document.querySelectorAll(`#conversations ${sel} a.conv`)]
    .find((a) => a.querySelector(".conv-title")!.textContent === title)!.querySelector<HTMLElement>(".conv-settle")!.click(), title, sel);
  const mine = (await titles(".folder:not(.settled)")).filter((t) => t === "A title" || t === "About count the bees");
  expect(mine.sort()).toEqual(["A title", "About count the bees"]);
  await settle("A title");
  expect([await titles(".folder:not(.settled)").then((t) => t.includes("A title")), await page.$eval("#conversations .settled-head", (h) => h.textContent), await page.$eval("#toast span", (t) => t.textContent)])
    .toEqual([false, "Settled1", "A title is settled"]);
  await page.click("#toast button");
  expect([await titles(".folder:not(.settled)").then((t) => t.includes("A title")), await page.$("#conversations .settled-head")]).toEqual([true, null]);
  await settle("A title");
  await settle("About count the bees");
  await page.click("#conversations .settled-head");
  expect((await titles(".folder.settled")).sort()).toEqual(["A title", "About count the bees"]);
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#conversations .settled-head");
  await connect(page);
  await page.keyboard.press("Escape");
  await page.waitForFunction(() => document.querySelectorAll("#conversations .folder.settled a.conv").length === 2);
  await settle("About count the bees", ".folder.settled");
  await page.waitForFunction(() => document.querySelectorAll("#conversations .folder.settled a.conv").length === 1);
  expect([await titles(".folder.settled"), (await titles(".folder:not(.settled)")).includes("About count the bees")]).toEqual([["A title"], true]);
  await page.close();
}, 60_000);
