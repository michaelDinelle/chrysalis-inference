import { expect, test } from "bun:test";
import { harness, checks, chrysalisStopped, lastStream, noteShown, replyText, sent, voicePage } from "./browser-harness";

harness();

test("voice mode: talking over a reply silences it and stops Chrysalis's request; the reply stays whole with what was not heard grey, and the next message tells the model how far the user got", async () => {
  const stopped = chrysalisStopped, { page, session } = await voicePage("story");
  await page.waitForFunction(() => document.querySelector('#thread > article[data-i="0"] .bubble')?.textContent === "slow story", { timeout: 10_000 });
  for (const t = Date.now(); !session().some((e) => e.ev === "interrupt") && Date.now() - t < 10_000; ) await Bun.sleep(10);
  const talked = Date.now();
  for (; chrysalisStopped === stopped && Date.now() - talked < 3000; ) await Bun.sleep(10);
  expect(chrysalisStopped).toBe(stopped + 1);
  expect(Date.now() - talked).toBeLessThan(500);
  await page.waitForFunction(() => /interrupted/.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  const whole = await replyText(page, 1), chunks = session().filter((e) => e.ev === "chunk"), full = Array.from({ length: 40 }, (_, k) => `w${k}`).join(" ");
  await page.waitForFunction(() => !!(CSS as any).highlights.get("voice-unheard"));
  const grey = await page.evaluate(() => ((CSS as any).highlights.get("voice-unheard").values().next().value as Range).toString().trim()), kept = whole.slice(0, whole.length - grey.length).trim();
  expect([grey.length > 0, whole.endsWith(grey), full.startsWith(whole)]).toEqual([true, true, true]);
  expect(kept.length).toBeGreaterThan(0);
  expect(kept.length).toBeLessThan(whole.length);
  const before = chunks.map((c) => c.text).join("").trimEnd(), ends = chunks.map((c) => chunks.filter((x) => x.to <= c.to).map((x) => x.text).join("").trimEnd());
  expect([full[kept.length] === " ", kept.length <= before.length, ends.includes(kept)]).toEqual([true, true, false]);
  await page.waitForFunction(() => document.querySelector('#thread > article[data-i="2"] .bubble')?.textContent === "and then", { timeout: 10_000 });
  await page.waitForFunction(() => /voice/.test(document.querySelector('#thread > article[data-i="3"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  expect(sent(lastStream).filter((m: any) => m.role === "assistant").map((m: any) => m.content)).toEqual([whole]);
  expect(sent(lastStream).at(-1).content).toContain(`[The user talked over your last reply, which is read aloud. They heard it up to "${kept.split(" ").slice(-12).join(" ")}" and nothing after that.]\n\nand then`);
  expect(checks.findLast((b) => b.messages.at(-1).content.endsWith("Transcript: and then"))?.messages.at(-1).content)
    .toBe(`The user said before: slow story\nThe assistant said: ${whole}\nTranscript: and then`);
  expect(await replyText(page, 1)).toBe(whole);
  await page.close();
}, 30_000);

test("voice mode: a reply's words not heard yet are grey, and turn their own color as they are spoken", async () => {
  const { page } = await voicePage("long");
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Speaking", { timeout: 10_000 });
  const heard = () => page.evaluate(() => {
    const all = document.querySelector('#thread > article[data-i="1"] .markdown')!.textContent!.trim();
    const range = (CSS as any).highlights.get("voice-unspoken")?.values().next().value as Range | undefined;
    const grey = range?.toString().trim() ?? null, mid = range && getComputedStyle(range.startContainer.parentElement!).color;
    return { all, grey, said: grey === null ? null : all.slice(0, all.length - grey.length).trim(), mid };
  });
  await Bun.sleep(500);
  const a = await heard();
  await Bun.sleep(700);
  const b = await heard();
  expect(a.grey && b.grey && a.said && b.said).toBeTruthy();
  expect([b.all.startsWith(a.said!), b.all.startsWith(b.said!), b.said!.length > a.said!.length]).toEqual([true, true, true]);
  expect(/^(w\d+ )*w\d+$/.test(b.said!)).toBe(true);
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 15_000 });
  expect(await page.evaluate(() => (CSS as any).highlights.has("voice-unspoken"))).toBe(false);
  await page.close();
}, 30_000);

test("voice mode: a reply talked over before it made a sound is taken out, and the next words join its question", async () => {
  const { page } = await voicePage("merge");
  await page.waitForFunction(() => document.querySelector('#thread > article[data-i="0"] .bubble')?.textContent === "slow first second part", { timeout: 15_000 });
  await page.waitForFunction(() => /^w0 w1/.test(document.querySelector('#thread > article[data-i="1"] .markdown')?.textContent ?? ""), { timeout: 10_000 });
  expect(await page.$$eval("#thread > article.msg:not(.ghost)", (els) => els.length)).toBe(2);
  expect(sent(lastStream).filter((m: any) => m.role !== "system")).toEqual([{ role: "user", content: `[${await noteShown(page)}]\n\nslow first second part` }]);
  await page.close();
}, 30_000);
