import { expect, test } from "bun:test";
import { harness, base, chatContext, chrysalisStopped, closeSettings, idle, openSettings, replyText, send, sent, spokenOf, streamAt, streams, turnAt, voiceLog, voicePage } from "./browser-harness";

harness();

test("voice mode: a reply begins at a pause the turn waits out, unseen and unheard until the turn ends with its words, and then is heard at once", async () => {
  const n0 = streams.length, { page, session } = await voicePage("ahead");
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  const log = session(), turned = turnAt(log), mine = streams.slice(n0).map((b, i) => [b, streamAt[n0 + i]]).filter(([b]) => b.max_tokens !== 1);
  expect(mine.length).toBe(1);
  expect(mine[0][1]).toBeLessThan(turned);
  const played = log.findIndex((e) => e.type === "playback" && e.playing), turnIx = log.findIndex((e) => e.ev === "sent" && e.type === "turn");
  expect(played).toBeGreaterThan(turnIx);
  expect(log.slice(0, turnIx).some((e) => e.ev === "chunk")).toBe(true);
  expect(await replyText(page, 1)).toBe("Paris.");
  expect(await page.$$eval("#thread > article.msg", (els) => els.length)).toBe(2);
  await page.close();
}, 30_000);

test("voice mode: a reply begun at a pause is dropped unheard when the user goes on, and the whole question is answered", async () => {
  const n0 = streams.length, stopped = chrysalisStopped, { page, session } = await voicePage("onward");
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  const replies = streams.slice(n0).filter((b) => b.max_tokens !== 1).map((b) => sent(b).at(-1).content.split("\n\n").at(-1));
  expect(replies).toEqual(["slow what is the", "slow what is the capital of France"]);
  expect(chrysalisStopped).toBe(stopped + 1);
  expect(await page.$eval('#thread > article[data-i="0"] .bubble', (e) => e.textContent)).toBe("slow what is the capital of France");
  expect(await page.$$eval("#thread > article.msg", (els) => els.length)).toBe(2);
  const log = session(), { said, cancelled } = spokenOf(log);
  expect([...said.keys()].map((id) => cancelled.has(id))).toEqual([true, false]);
  expect(log.findIndex((e) => e.type === "cancel")).toBeLessThan(log.findIndex((e) => e.ev === "sent" && e.type === "turn"));
  await page.close();
}, 30_000);

test("voice mode: it says it is listening once the microphone gives audio, not when the service is ready", async () => {
  const page = await (await chatContext()).newPage();
  await page.evaluateOnNewDocument(() => {
    sessionStorage.setItem("settings2", JSON.stringify({ system: "", think: {}, fast: {}, voice: { name: "silent" }, artifacts: false }));
    const ask = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (c) => { await new Promise((r) => setTimeout(r, 1500)); return ask(c); };
  });
  await idle(page, page.goto(`${base}/`));
  const hellos = () => voiceLog().filter((e) => e.ev === "msg" && e.type === "hello").length, n0 = hellos();
  await page.click("#voice");
  for (const t = Date.now(); hellos() === n0 && Date.now() - t < 1000; ) await Bun.sleep(20);
  await Bun.sleep(300);
  expect([hellos() > n0, await page.$eval("#voice-status", (e) => e.textContent)]).toEqual([true, "Starting voice…"]);
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 5000 });
  await page.close();
}, 30_000);

test("voice mode: the thinking bar is the chat's, one setting for spoken and written replies", async () => {
  const { page } = await voicePage("silent");
  const bar = () => page.$eval("#energy", (e) => [e.getAttribute("aria-valuetext"), (e as HTMLElement).title]);
  expect(await bar()).toEqual(["Max", "Thinking: Max"]);
  await page.focus("#energy");
  await page.keyboard.press("ArrowLeft");
  expect(await bar()).toEqual(["Medium", "Thinking: Medium"]);
  await openSettings(page);
  expect(await page.$(".voice-think")).toBeNull();
  await closeSettings(page);
  await page.click("#send");
  expect(await bar()).toEqual(["Medium", "Thinking: Medium"]);
  await page.close();
}, 30_000);
