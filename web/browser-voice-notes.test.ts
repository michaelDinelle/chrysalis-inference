import { expect, test } from "bun:test";
import { ARTIFACTS_PROMPT } from "./artifacts";
import { harness, articles, card, closeSettings, fake, lastStream, noteShown, open, openSettings, send, sent, shown, streams, voicePage, fresh } from "./browser-harness";

harness();

test("voice mode's note goes before the first question said and not again, the typing note after it, and the system prompt, an old setting's kept as Chrysalis's description, never changes", async () => {
  const n0 = streams.length;
  const { page, session } = await voicePage("notes", { system: "Be brief.", first: (p) => send(p, "typed first") });
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="5"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  const note = await noteShown(page);
  expect(note).toMatch(/^The user is talking to you by voice/);
  await openSettings(page);
  expect(await page.$$eval("#settings textarea", (t) => t.map((x) => [x.id, (x as HTMLTextAreaElement).value]))).toEqual([["system-input", ""]]);
  await page.$eval("#knob-voice_speed", (el) => { (el as HTMLInputElement).value = "1.2"; el.dispatchEvent(new Event("change")); });
  await closeSettings(page);
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="7"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  await page.click("#send");
  await send(page, "typed again");
  const asked = streams.slice(n0).filter((b) => b.max_tokens !== 1);
  expect(asked.map((b) => sent(b).at(-1).content)).toEqual(["typed first", `[${note}]\n\nfirst question`, "second question",
    "third question", "[The user is typing again and reads your replies.]\n\ntyped again"]);
  expect(asked.every((b) => JSON.stringify(b.messages[0]) === JSON.stringify({ role: "system", content: "Be brief." }))).toBe(true);
  expect(session().some((e) => e.type === "config")).toBe(true);
  expect(await page.$$eval(".msg-note summary", (els) => els.map((e) => e.textContent))).toEqual(["Voice mode note", "Note to the model"]);
  await page.close();
}, 45_000);

test("a chat is sent one system prompt: the system prompt written in Settings goes with every chat, kept through Reset, after whoever the chat is with and before how to make artifacts, and nothing of it once it is emptied; Chrysalis's card is its own prompt, its description as written and its scenario after it", async () => {
  const page = await open();
  await page.setViewport({ width: 1600, height: 900 });
  await send(page, "first");
  expect(lastStream.messages[0]).toEqual({ role: "system", content: ARTIFACTS_PROMPT });
  await openSettings(page, "chat");
  await page.type("#system-input", "Answer {{MODEL}}'s way: plainly.");
  await page.click("#settings-reset");
  await closeSettings(page);
  await send(page, "second");
  expect(lastStream.messages.filter((m: any) => m.role === "system")).toEqual([{ role: "system", content: `Answer Qwen3.8-27B's way: plainly.\n\n${ARTIFACTS_PROMPT}` }]);
  expect(await page.evaluate(() => JSON.parse(sessionStorage.getItem("settings2")!).global)).toBe("Answer {{MODEL}}'s way: plainly.");
  await openSettings(page, "chat");
  expect(await page.$eval("#system-input", (t) => (t as HTMLTextAreaElement).value)).toBe("Answer {{MODEL}}'s way: plainly.");
  await page.click("#system-input", { count: 3 });
  await page.keyboard.press("Backspace");
  await closeSettings(page);
  await send(page, "third");
  expect(lastStream.messages[0]).toEqual({ role: "system", content: ARTIFACTS_PROMPT });
  await page.click("#thread > article.msg.assistant .speaker");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-field textarea");
  const typeIn = async (label: string, text: string) => {
    const at = await page.$$eval("#cast-sheet .card-field", (f, l) => f.findIndex((x) => x.querySelector(".card-label")!.textContent === l), label);
    await page.type(`#cast-sheet .card-field:nth-of-type(${at + 1}) textarea`, text);
  };
  await typeIn("Description", "Be terse, {{MODEL}}. {{char}} tends a greenhouse.");
  await typeIn("Scenario", "A rainy afternoon.");
  await page.keyboard.press("Escape");
  await send(page, "fourth");
  expect(lastStream.messages[0]).toEqual({ role: "system", content: `Be terse, Qwen3.8-27B. Chrysalis tends a greenhouse.\n\nScenario: A rainy afternoon.\n\n${ARTIFACTS_PROMPT}` });
  await page.click("#you");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-text");
  await page.type("#cast-sheet .card-text", "{{user}} keeps bees.");
  await page.keyboard.press("Escape");
  await send(page, "fifth");
  expect(lastStream.messages[0]).toEqual({ role: "system", content: `Be terse, Qwen3.8-27B. Chrysalis tends a greenhouse.\n\nScenario: A rainy afternoon.\n\nUser keeps bees.\n\n${ARTIFACTS_PROMPT}` });
  await page.click("#thread > article.msg.assistant .speaker");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-field textarea");
  await typeIn("Greeting", "Hello, {{user}}.");
  await page.keyboard.press("Escape");
  await fresh(page);
  expect(await articles(page)).toEqual(["Hello, User."]);
  await send(page, "sixth");
  expect(lastStream.messages.slice(1)).toEqual([{ role: "user", content: "[The scene begins.]" }, { role: "assistant", content: "Hello, User." }, { role: "user", content: "sixth" }]);
  await page.close();
}, 60_000);
