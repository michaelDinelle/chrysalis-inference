import { expect, test } from "bun:test";
import { harness, card, fakes, fill, idle, lastStream, open, send, sent, shown, still, streams, sw, toolsOn } from "./browser-harness";

harness();

test("a chat past its context is compacted: a divider holds the summary, the next turn opens with it, a reload keeps it, and going back behind it brings the real history back", async () => {
  const page = await open();
  await send(page, "first question");
  await send(page, "overflow now");
  const wire = () => lastStream.messages.filter((m: any) => m.role !== "system").map((m: any) => [m.role, typeof m.content === "string" ? m.content : m.content.map((p: any) => p.text).join("")]);
  const divider = () => page.$$eval(".checkpoint", (d) => d.map((x) => [x.querySelector("summary")!.textContent, x.closest("article")!.dataset.i]));
  expect(await divider()).toEqual([["Context compacted", "3"]]);
  await page.click(".checkpoint summary");
  expect(await page.$eval(".checkpoint .markdown", (m) => (m as HTMLElement).innerText)).toContain("the word is lighthouse");
  expect(await page.$eval(".msg.assistant:last-child .error", () => true).catch(() => false)).toBe(false);

  await send(page, "next one");
  let sent = wire();
  expect(sent.map((m: string[]) => m[0])).toEqual(["user", "assistant", "user"]);
  expect(sent[0][1]).toStartWith("<conversation-checkpoint>");
  expect(sent[0][1]).toContain("the word is lighthouse");
  expect(sent[0][1]).toEndWith("overflow now");
  expect(sent[2][1]).toBe("next one");

  await send(page, "overflow again");
  const saved = await page.evaluate(() => (Object.values(JSON.parse(sessionStorage.getItem("chats")!)) as any[])[0].messages.flatMap((m: any) => m.checkpoints ?? []).map((k: any) => !!k.recent));
  expect(saved).toEqual([false, true]);
  await idle(page, page.reload());
  expect(await divider()).toEqual([["Context compacted", "3"], ["Context compacted", "7"]]);
  await send(page, "after the reload");
  sent = wire();
  expect(sent[0][1]).toStartWith("<conversation-checkpoint>");
  expect(sent.at(-1)[1]).toBe("after the reload");

  await page.evaluate(() => (document.querySelector('article[data-i="0"] [data-act="edit"]') as HTMLElement).click());
  await page.waitForSelector("textarea.edit-text, .editor textarea");
  await page.evaluate(() => { const t = document.querySelector(".editor textarea, textarea.edit-text") as HTMLTextAreaElement; t.value = "the first question, edited"; t.dispatchEvent(new Event("input", { bubbles: true })); });
  await page.evaluate(() => ([...document.querySelectorAll("button")].find((b) => b.textContent === "Save") as HTMLElement).click());
  expect(await divider()).toEqual([]);
  await send(page, "after the edit");
  sent = wire();
  expect(sent[0]).toEqual(["user", "the first question, edited"]);
  expect(sent.some((m: string[]) => m[1].includes("<conversation-checkpoint>"))).toBe(false);
  await page.close();
}, 60_000);

test("a sub-agent that fills its context says it is compacting, keeps going, and shows where in its log", async () => {
  const page = await open();
  await toolsOn(page);
  await page.click("#tools-btn");
  await page.waitForSelector(sw("agents"));
  await page.click(sw("agents"));
  await page.keyboard.press("Escape");
  await page.type("#input", "squeeze agent");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector(".agent-card .agent-now")?.textContent === "Checking its work", { timeout: 15_000 });
  await page.waitForFunction(() => document.querySelectorAll(".msg .stats").length === 1, { timeout: 20_000 });
  expect(await page.$eval(".agent-card", (c) => [c.getAttribute("data-state"), c.querySelector(".tool-name")!.textContent])).toEqual(["done", "Fill the context"]);
  await page.click(".agent-card .agent-head");
  await page.waitForSelector(".agent-log .checkpoint");
  expect(await page.$$eval(".agent-log > *", (l) => l.map((x) => x.className))).toEqual(["checkpoint", "round"]);
  await page.close();
}, 30_000);

test("a heavy run leaves the chat whole through a reload: the answers stay, and the sub-agent's log is what is let go", async () => {
  const page = await open();
  await toolsOn(page);
  await page.click("#tools-btn");
  await page.waitForSelector(sw("agents"));
  await page.click(sw("agents"));
  await page.keyboard.press("Escape");
  await page.type("#input", "flood agent");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelectorAll(".msg .stats").length > 0, { timeout: 60_000 });
  const cards = () => page.$$eval(".agent-card", (c) => c.map((x) => [x.getAttribute("data-state"), x.querySelector(".agent-meta")!.textContent, x.querySelector(".agent-now")!.textContent]));
  const end = await cards();
  expect(end.length).toBe(1);
  expect(end[0][1]).toStartWith("192 tools");
  const saved = await page.evaluate(() => sessionStorage.getItem("chats")!.length);
  expect(saved).toBeLessThan(200_000);
  expect(await page.evaluate(() => !!(Object.values(JSON.parse(sessionStorage.getItem("chats")!)) as any[])[0].messages.at(-1).stats)).toBe(true);
  await idle(page, page.reload());
  expect(await cards()).toEqual(end);
  await page.click(".agent-card .agent-head");
  expect(await page.$eval(".agent-card .agent-body > .markdown", (m) => (m as HTMLElement).innerText.trim())).toBe("Read it all.");
  expect(await page.$$eval(".agent-card .agent-log .round .tool-card", (c) => c.length)).toBe(160);
  expect(await page.$eval(".agent-card .agent-notes", (n) => n.textContent)).toBe("32 earlier tool calls are not kept through a reload");
  expect(await page.$$eval(".agent-card .agent-log .tool-result", (c) => c.length)).toBe(0);
  await send(page, "still here");
  expect(lastStream.messages.filter((m: any) => m.role === "tool").map((m: any) => m.content)).toEqual(["Read it all."]);
  await page.close();
}, 90_000);

test("a reload in the middle of a reply that compacted brings it back with one divider, and the words once", async () => {
  const page = await open();
  await send(page, "first question");
  await page.type("#input", "overflow now slowly");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /w3 /.test([...document.querySelectorAll(".msg.assistant")].at(-1)?.textContent ?? ""), { timeout: 15_000 });
  await page.waitForFunction(() => sessionStorage.getItem("chats")?.includes('"checkpoints"'), { timeout: 8000 });
  expect(await page.$$eval(".msg .stats", (s) => s.length)).toBe(1);
  await page.reload({ waitUntil: "load" });
  fakes.hurry = true;
  await page.waitForFunction(() => document.querySelectorAll(".msg .stats").length === 2, { timeout: 20_000 });
  expect(await page.$$eval(".checkpoint", (c) => c.length)).toBe(1);
  expect((await page.$eval(".msg.assistant:last-child", (m) => m.textContent!)).match(/w99 /g)?.length).toBe(1);
  await page.close();
}, 60_000);

test("a page cut off inside a round the server has since folded starts the reply over, and has every word once", async () => {
  const page = await open();
  await page.evaluate(() => {
    const reader = ReadableStreamDefaultReader.prototype as any, read = reader.read;
    reader.read = function () { return (window as any).cut ? Promise.reject(new Error("cut off")) : read.call(this); };
  });
  await toolsOn(page);
  await page.keyboard.press("Escape");
  const asked = streams.length;
  await page.type("#input", "tool slowly");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /w3 /.test([...document.querySelectorAll(".msg.assistant")].at(-1)?.textContent ?? ""), { timeout: 15_000 });
  await page.evaluate(() => { (window as any).cut = true; });
  fakes.hurry = true;
  while (streams.length < asked + 2) await Bun.sleep(50);
  await page.evaluate(() => { (window as any).cut = false; });
  await page.waitForFunction(() => document.querySelectorAll(".msg .stats").length === 1, { timeout: 20_000 });
  const text = await page.$eval(".msg.assistant:last-child", (m) => m.textContent!);
  expect(Array.from({ length: 100 }, (_, k) => text.match(new RegExp(`w${k} `, "g"))?.length)).toEqual(Array(100).fill(1));
  expect(text).toContain("The tool said: echo: once");
  await page.close();
}, 60_000);

test("a reply waiting for you says so in the list instead of Working, and tells you when you are in another chat", async () => {
  const page = await open(), status = () => page.$$eval(".conv-status", (s) => s.map((x) => x.textContent));
  await toolsOn(page);
  await page.keyboard.press("Escape");
  await page.type("#input", "tool note slowly");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /w3 /.test([...document.querySelectorAll(".msg.assistant")].at(-1)?.textContent ?? ""), { timeout: 15_000 });
  expect((await status())[0]).toStartWith("Working");
  fakes.hurry = true;
  await page.click("#new-chat");
  await page.waitForFunction(() => /a tool is waiting for you to allow it/.test(document.querySelector("#toast")?.textContent ?? ""), { timeout: 15_000 });
  expect(await status()).toEqual(["Needs you"]);
  await page.click(".conv:has(.conv-status)");
  await page.waitForSelector('.tool-card[data-state="approval"] [data-decision="once"]');
  await page.click('.tool-card[data-state="approval"] [data-decision="once"]');
  await page.waitForFunction(() => document.querySelectorAll(".msg .stats").length === 1, { timeout: 20_000 });
  expect(await status()).toEqual([]);
  await page.close();
}, 60_000);

test("a reply held up says why while it waits, then goes on by itself", async () => {
  const page = await open(), shown = (text: string) => page.waitForFunction((t) => (document.querySelector(".msg.assistant:last-child .waiting")?.textContent ?? "").includes(t), { timeout: 8000 }, text);
  fakes.memoryRefusals = 2;
  await page.type("#input", "needs memory");
  await page.keyboard.press("Enter");
  await shown("Waiting for memory: other replies are using it");
  await page.waitForFunction(() => document.querySelectorAll(".msg .stats").length === 1, { timeout: 10_000 });
  expect(await page.$eval(".msg.assistant:last-child", (m) => (m as HTMLElement).innerText)).toContain("You said: needs memory");
  fakes.naps = 1;
  await page.type("#input", "chrysalis naps");
  await page.keyboard.press("Enter");
  await shown("the machine is going to sleep; Chrysalis loads the model again when it wakes: trying again");
  await page.waitForFunction(() => document.querySelectorAll(".msg .stats").length === 2, { timeout: 10_000 });
  expect(await page.$$eval(".msg .error", (e) => e.length)).toBe(0);
  await page.close();
}, 30_000);

test("a tab with no room left says so instead of quietly losing the chat on the next reload", async () => {
  const page = await open();
  await page.evaluate(() => {
    for (let size = 1 << 22, n = 0; size >= 1; ) {
      try { sessionStorage.setItem(`pad${n++}`, "x".repeat(size)); } catch { size >>= 1; }
    }
  });
  await send(page, "first question");
  await page.waitForFunction(() => /no room left/.test(document.querySelector("#toast")?.textContent ?? ""), { timeout: 5000 });
  await page.close();
}, 30_000);
