import { expect, test } from "bun:test";
import { existsSync, readdirSync, readlinkSync } from "node:fs";
import { harness, articles, dir, fresh, idle, kept, lastStream, memory, open, send, streams } from "./browser-harness";

harness();

test("an incognito chat, from the ghost button, is answered and leaves nothing: it is in no list and in nothing the browser keeps, its file is not kept either, the tab has no title for it, and the model server cannot write its prompt cache while its reply is written; the outline and its tab show only while it is the open chat and move nothing; it is gone on a new chat, on opening another chat, on the ghost button again and on a reload", async () => {
  const page = await open();
  await send(page, "an ordinary errand");
  const begun = streams.length, boxes = () => page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#app header *, #composer *")].filter((e) => e.offsetParent !== null).map((e) => { const r = e.getBoundingClientRect(); return [e.id || e.className, Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)].join(","); }).join("|"));
  const marked = () => page.evaluate(() => [document.documentElement.hasAttribute("data-incognito"), document.getElementById("incognito")!.getAttribute("aria-pressed"), getComputedStyle(document.body, "::before").content]);
  const pointed = () => readlinkSync(`${memory}/snapshots/qwen3.8-27b`);
  await fresh(page);
  const plain = await boxes();
  expect(await marked()).toEqual([false, "false", "none"]);

  await page.click("#incognito");
  expect([await marked(), await boxes() === plain]).toEqual([[true, "true", '"Incognito"'], true]);
  await Bun.write(`${dir}/ledger.txt`, "the hidden ledger");
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/ledger.txt`);
  await page.waitForFunction(() => document.querySelectorAll("#attachments .att").length === 1);
  await page.type("#input", "slowly, the secret plan");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /w3/.test(document.querySelector("#thread > article.assistant .markdown")?.textContent ?? ""), { timeout: 10_000 });
  expect([pointed(), existsSync(`${kept}/qwen3.8-27b`)]).toEqual([`${memory}/none/qwen3.8-27b`, true]);
  await page.click("#send");
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send");
  expect([streams.length - begun, JSON.stringify(lastStream.messages).includes("the hidden ledger"), lastStream.chrysalis_incognito]).toEqual([1, true, undefined]);
  expect(pointed()).toBe(`${kept}/qwen3.8-27b`);
  const kept_ = () => page.evaluate(async () => {
    const files = await new Promise<number>((done) => {
      const open = indexedDB.open("chrysalis", 1);
      open.onsuccess = () => { const all = open.result.transaction("files").objectStore("files").getAll(); all.onsuccess = () => done(all.result.length); };
    });
    const stored = [sessionStorage, localStorage].map((s) => Object.keys(s).map((k) => s.getItem(k)).join("\n")).join("\n");
    return [/secret plan|hidden ledger|ledger\.txt/.test(stored), files, [...document.querySelectorAll("#conversations .conv-title")].map((t) => t.textContent), document.title];
  });
  expect(await kept_()).toEqual([false, 0, ["A title"], "Chrysalis Inference"]);

  await fresh(page);
  expect([await articles(page), await marked(), (await kept_())[2]]).toEqual([[], [false, "false", "none"], ["A title"]]);
  await page.click("#incognito");
  await send(page, "a second secret");
  await page.click("#conversations a.conv");
  await page.waitForFunction(() => !document.documentElement.hasAttribute("data-incognito"));
  expect((await articles(page))[0]).toBe("an ordinary errand");
  await page.click("#incognito");
  await send(page, "a third secret");
  await page.click("#incognito");
  expect([await articles(page), (await marked())[0]]).toEqual([[], false]);
  await page.click("#incognito");
  await send(page, "a fourth secret");
  await idle(page, page.reload());
  expect([(await marked())[0], (await kept_()).slice(0, 3), readdirSync(memory)]).toEqual([false, [false, 0, ["A title"]], ["snapshots"]]);
  await page.close();
}, 60_000);
