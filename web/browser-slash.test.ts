import { expect, test } from "bun:test";
import { connect, harness, open, openFolder, opencode, send, work } from "./browser-harness";

harness();

test("a thread takes OpenCode's slash commands and shows its context: typing / lists OpenCode's own commands and the terminal's, the meter says how much of the context is used and where OpenCode compacts it, /compact compacts it and the thread shows the summary, live and after a reload, and /rename, /variants and /new do what they do in the terminal while an unknown one is sent as text", async () => {
  const page = await open(), meter = () => page.$eval("#meter", (m) => [...m.querySelectorAll(".ctx-pct, .ctx-left")].map((e) => e.textContent).join(" · "));
  const ask = async (text: string, said = text) => {
    await page.type("#input", text);
    await page.keyboard.press("Enter");
    await page.waitForFunction((said) => document.querySelector("#send")?.getAttribute("aria-label") === "Send"
      && (document.querySelector<HTMLElement>("#thread > article.assistant:last-child > .markdown")?.innerText ?? "").includes(`You asked: ${said}`), { timeout: 20_000 }, said);
  };
  await connect(page);
  await openFolder(page, work);
  expect(await meter()).toBe("");
  await send(page, "sort the first shelf");
  const session = opencode.log.prompts.at(-1)!.session;
  expect([await meter(), await page.$eval("#meter .ctx", (b) => (b as HTMLElement).title.split("\n"))])
    .toEqual(["0% · 170K left", ["Context: 160 tokens used", "OpenCode compacts it at 170,000 (169,840 to go)", "The model's window is 200,000"]]);
  await page.click("#meter .ctx");
  expect(await page.$$eval("#ctx-pop .ctx-row", (r) => r.map((x) => [...x.children].map((c) => c.textContent)))).toEqual([["Used", "160 tokens"], ["Compacts at", "170,000"], ["Until then", "169,840 tokens"], ["Model's window", "200,000"]]);
  await page.click("#meter .ctx");
  expect(await page.$eval("#ctx-pop", (e) => (e as HTMLElement).hidden)).toBe(true);

  await page.type("#input", "/");
  await page.waitForSelector("#slash-pop:not([hidden]) .slash-item");
  await page.waitForFunction(() => [...document.querySelectorAll("#slash-pop .slash-name")].some((n) => n.textContent === "/review"));
  const names = await page.$$eval("#slash-pop .slash-name", (n) => n.map((e) => e.textContent));
  expect([names[0], ["/new", "/compact", "/models", "/agents", "/variants", "/undo", "/fork", "/rename", "/sessions", "/open", "/cd", "/copy", "/export", "/skills", "/status", "/settings", "/help"].every((n) => names.includes(n))])
    .toEqual(["/review", true]);
  await page.type("#input", "comp");
  expect(await page.$$eval("#slash-pop .slash-name", (n) => n.map((e) => e.textContent))).toEqual(["/compact"]);
  const prompts = opencode.log.prompts.length;
  await page.keyboard.press("Enter");
  await page.waitForSelector("#thread details.checkpoint");
  expect([await page.$eval("#thread details.checkpoint summary", (e) => e.textContent), await page.$eval("#thread details.checkpoint .markdown", (e) => e.textContent!.trim()), await meter(),
    opencode.sessions.get(session)!.messages.filter((m) => m.type === "compaction").length, opencode.log.prompts.length - prompts, await page.$eval("#input", (i) => (i as HTMLTextAreaElement).value)])
    .toEqual(["Context compacted", "Summary: 1 requests so far.", "Compacted · 170K free", 1, 0, ""]);
  await page.reload();
  await page.waitForSelector("#thread details.checkpoint");
  expect(await meter()).toBe("Compacted · 170K free");
  await ask("and the second shelf");
  expect([await meter(), await page.$$eval("#thread details.checkpoint", (d) => d.length)]).toEqual(["0% · 170K left", 1]);
  await page.click("#meter .ctx");
  await page.click("#ctx-pop .btn");
  await page.waitForFunction(() => document.querySelectorAll("#thread details.checkpoint").length === 2);

  await page.type("#input", "/rename The shelves");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector("#title")?.textContent === "The shelves");
  expect(opencode.sessions.get(session)!.title).toBe("The shelves");
  await page.type("#input", "/thinking low");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector("#energy")?.getAttribute("aria-valuetext") === "Low");
  await ask("/review the shelf");
  expect([opencode.log.prompts.at(-1)!.command, opencode.log.prompts.at(-1)!.text]).toEqual(["review", "the shelf"]);
  const before = opencode.log.prompts.length;
  await ask("/zzz is not a command");
  expect([opencode.log.prompts.length - before, opencode.log.prompts.at(-1)!.text, opencode.log.prompts.at(-1)!.command]).toEqual([1, "/zzz is not a command", undefined]);
  await page.type("#input", "/new");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => !document.querySelector(".msg"));
  expect(await page.$eval("#folder-btn", (b) => b.getAttribute("title"))).toBe(`Working in ${work}`);
  await page.close();
}, 90_000);
