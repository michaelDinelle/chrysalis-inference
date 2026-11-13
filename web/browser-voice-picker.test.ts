import { expect, test } from "bun:test";
import { harness, checked, closeSettings, fake, idle, open, openSettings, send, settled, still, stored, voiceLog, voicePage, voicePages } from "./browser-harness";

harness();

test("voice picker: a voice model not downloaded yet is downloaded from the picker, and then offers its voices", async () => {
  const { page } = await voicePage("pick");
  await page.click("#voice-pick");
  await page.waitForSelector('#voice-menu:popover-open [data-get="big-tts"]');
  expect(await page.$$eval('#voice-menu [data-voice^="big-tts/"]', (els) => els.length)).toBe(0);
  await page.click('#voice-menu [data-get="big-tts"]');
  await page.waitForSelector('#voice-menu [data-voice="big-tts/zed"]', { timeout: 10_000 });
  expect(await page.$$eval("#voice-menu [data-get]", (els) => els.length)).toBe(0);
});

test("voice picker: the voice models' voices by language, each with its gender; a pick switches model and voice for the next sentence, and Settings shows the same choice", async () => {
  const { page, session } = await voicePage("pick");
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  expect(new Set(session().filter((e) => e.ev === "chunk").map((e) => `${e.tts}/${e.voice}`))).toEqual(new Set(["fake-tts/pick"]));
  expect(await page.$eval("#voice-pick .voice-pick-name", (el) => el.textContent)).toBe("Pick");
  await page.click("#voice-pick");
  await page.waitForSelector("#voice-menu:popover-open");
  const rows = await page.$$eval("#voice-menu > *", (els) => els.map((e) => [e.className, e.classList.contains("voice-chips")
    ? [...e.children].map((c) => (c as HTMLElement).dataset.voice?.split("/")[1] ?? (c as HTMLElement).dataset.get ?? "") : e.textContent]));
  const fake = ["plain", "pie", "story", "long", "merge", "join", "ahead", "onward", "checking", "silent", "pick", "notes", "point"];
  expect(rows.slice(1)).toEqual([["voice-chips voice-top", ["", "big-tts"]], ["voice-model", "Fake TTS"], ["voice-group", "American English"], ["voice-chips", fake],
    ["voice-group", "British English"], ["voice-chips", ["echo"]], ["voice-model", "Other TTS"], ["voice-group", "Spanish"], ["voice-chips", ["ana"]],
    ["voice-group", "American English"], ["voice-chips", ["bob"]]]);
  expect(await page.$$eval('#voice-menu .voice-chip[data-voice^="other-tts/"]', (els) => els.map((e) => (e as HTMLElement).title))).toEqual(["Spanish · Female", "American English · Male"]);
  expect(await page.$$eval('#voice-menu .voice-chip[aria-checked="true"]', (els) => els.map((e) => (e as HTMLElement).dataset.voice))).toEqual(["fake-tts/pick"]);
  await page.click('#voice-menu .voice-chip[data-voice="other-tts/bob"] > span');
  const configs = () => session().filter((e) => e.type === "config").map(({ stack, voice, speed }) => ({ stack, voice, speed }));
  expect(await settled(() => configs().length === 1)).toBe(true);
  expect(configs()).toEqual([{ stack: { tts: "other-tts" }, voice: "bob", speed: undefined }]);
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="3"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  const log = session(), after = log.slice(log.findIndex((e) => e.type === "config")).filter((e) => e.ev === "chunk");
  expect(after.length).toBeGreaterThan(0);
  expect(new Set(after.map((e) => `${e.tts}/${e.voice}`))).toEqual(new Set(["other-tts/bob"]));
  expect(await page.$eval("#voice-pick .voice-pick-name", (el) => el.textContent)).toBe("Bob");
  expect(await stored(page)).toEqual({ tts: "other-tts", name: "bob" });
  const chosen = async (id: string) => {
    await page.click("#voice-settings");
    await page.waitForFunction((id) => document.querySelector('#voice-menu:popover-open .voice-chip[aria-checked="true"]')?.getAttribute("data-voice") === id, {}, id);
  };
  await openSettings(page);
  await chosen("other-tts/bob");
  await page.click('#voice-menu .voice-chip[data-voice="other-tts/ana"] > span');
  await closeSettings(page);
  await page.waitForFunction(() => document.querySelector("#voice-pick .voice-pick-name")?.textContent === "Ana");
  expect(await settled(() => configs().length === 2)).toBe(true);
  expect(configs()[1]).toEqual({ stack: undefined, voice: "ana", speed: undefined });
  await page.click("#voice-pick");
  await page.click('#voice-menu .voice-chip[data-voice="fake-tts/echo"] > span');
  await page.waitForSelector("#toast:not([hidden]) .btn");
  await openSettings(page);
  await chosen("fake-tts/echo");
  await page.keyboard.press("Escape");
  await closeSettings(page);
  await page.click("#toast .btn");
  await page.waitForFunction(() => document.querySelector("#voice-pick .voice-pick-name")?.textContent === "Ana");
  await openSettings(page);
  await chosen("other-tts/ana");
  await page.keyboard.press("Escape");
  await closeSettings(page);
  expect(await settled(() => configs().length === 4)).toBe(true);
  expect(configs().slice(2)).toEqual([{ stack: { tts: "fake-tts" }, voice: "echo", speed: undefined }, { stack: { tts: "other-tts" }, voice: "ana", speed: undefined }]);
  expect(await stored(page)).toEqual({ tts: "other-tts", name: "ana" });
  await page.close();
}, 40_000);

test("Settings lists the voices in a menu from its voice button, over the dialog and moving nothing in it; a chip's button plays its sample and shows it playing until it has been heard, a chip picks the voice, and the menu goes with the dialog", async () => {
  const page = await open();
  await openSettings(page);
  const layout = () => page.$$eval("#settings, #settings *:not(#voice-menu, #voice-menu *)", (els) => els.flatMap((e) => e instanceof HTMLElement ? [[e.offsetLeft, e.offsetTop, e.offsetWidth, e.offsetHeight].join()] : []).join(";"));
  const still = await layout();
  await page.click("#voice-settings");
  await page.waitForSelector('#voice-menu:popover-open .voice-chip[data-voice="other-tts/bob"]', { timeout: 10_000 });
  expect(await layout()).toBe(still);
  const before = voiceLog().length;
  await page.click('#voice-menu .voice-chip[data-voice="other-tts/bob"] .voice-play');
  await page.waitForSelector('#voice-menu .voice-chip[data-voice="other-tts/bob"] .voice-play.playing');
  await page.waitForFunction(() => !document.querySelector("#voice-menu .voice-play.playing"), { timeout: 10_000 });
  const heard = voiceLog().slice(before);
  expect([heard.filter((e) => e.type === "preview").map((e) => [e.tts, e.voice]), heard.filter((e) => e.ev === "chunk").map((e) => e.id)])
    .toEqual([[["other-tts", "bob"]], ["preview"]]);
  expect(await page.$("#voice-menu:popover-open")).not.toBeNull();
  await page.click('#voice-menu .voice-chip[data-voice="other-tts/bob"] > span');
  await page.waitForFunction(() => !document.querySelector("#voice-menu:popover-open"));
  expect([await page.$eval("#voice-settings .voice-pick-name", (b) => b.textContent), await stored(page)]).toEqual(["Bob", { tts: "other-tts", name: "bob" }]);
  await page.click("#voice-settings");
  await page.waitForSelector("#voice-menu:popover-open");
  await page.keyboard.down("Control");
  await page.keyboard.press(",");
  await page.keyboard.up("Control");
  await page.waitForFunction(() => !(document.getElementById("settings") as HTMLDialogElement).open);
  await page.waitForFunction(() => !document.querySelector("#voice-menu:popover-open"), { timeout: 2000 });
  await page.close();
}, 30_000);

test("voice picker: a voice's play button plays its sample, apart from any reply or turn, and the voice's button closes the menu again", async () => {
  const { page, session } = await voicePage("silent");
  await page.click("#voice-pick");
  await page.waitForSelector('#voice-menu:popover-open .voice-chip[data-voice="other-tts/ana"]');
  await page.click('#voice-menu .voice-chip[data-voice="other-tts/ana"] .voice-play');
  expect(await settled(() => session().filter((e) => e.type === "playback").length === 2)).toBe(true);
  const log = session();
  expect(log.filter((e) => e.type === "preview").map((e) => e.voice)).toEqual(["ana"]);
  expect(log.filter((e) => e.ev === "chunk").map((e) => [e.id, e.voice, e.text])).toEqual([["preview", "ana", "This is Ana."]]);
  expect(log.filter((e) => e.type === "playback").map((e) => e.playing)).toEqual([true, false]);
  expect(log.some((e) => e.type === "say" || e.type === "config")).toBe(false);
  expect(await page.$$eval("#thread > article", (els) => els.length)).toBe(0);
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 3000 });
  expect(await page.$eval("#voice-pick .voice-pick-name", (el) => el.textContent)).toBe("Silent");
  expect(await page.$("#voice-menu:popover-open")).not.toBeNull();
  await page.click("#voice-pick");
  await Bun.sleep(300);
  expect(await page.$("#voice-menu:popover-open")).toBeNull();
  await page.close();
}, 30_000);

test("voice mode: mute stops the microphone's audio; ending it closes the socket, and the service stops once idle", async () => {
  for (const p of voicePages.splice(0)) if (!p.isClosed()) await p.close();
  const { page, session } = await voicePage("silent");
  const frames = () => session().filter((e) => e.ev === "frames").at(-1)?.n ?? 0;
  for (let t = Date.now(); frames() < 20 && Date.now() - t < 5000; ) await Bun.sleep(50);
  expect(frames()).toBeGreaterThanOrEqual(20);
  await page.click("#voice-mute");
  await Bun.sleep(300);
  const muted = frames();
  await Bun.sleep(700);
  expect(frames()).toBe(muted);
  expect(session().some((e) => e.type === "mute" && e.on === true)).toBe(true);
  await page.click("#voice-mute");
  for (let t = Date.now(); frames() < muted + 20 && Date.now() - t < 5000; ) await Bun.sleep(50);
  expect(frames()).toBeGreaterThanOrEqual(muted + 20);
  await page.click("#send");
  expect(await page.$eval("#voice-pill", (el) => (el as HTMLElement).hidden)).toBe(true);
  expect(await page.$eval(".composer-box", (el) => getComputedStyle(el).display)).not.toBe("none");
  for (let t = Date.now(); !session().some((e) => e.ev === "close") && Date.now() - t < 3000; ) await Bun.sleep(50);
  expect(session().some((e) => e.ev === "close")).toBe(true);
  const starts = voiceLog().filter((e) => e.ev === "start").length;
  for (let t = Date.now(); voiceLog().filter((e) => e.ev === "exit").length < starts && Date.now() - t < 6000; ) await Bun.sleep(100);
  expect(voiceLog().filter((e) => e.ev === "exit").length).toBe(starts);
  await page.close();
}, 30_000);

test("Settings picks the speech to text from the service's own models in the voice menu: one not downloaded yet is downloaded there, a pick is kept, and voice mode switches to it mid-session", async () => {
  const { page, session } = await voicePage("silent");
  await openSettings(page);
  await page.click("#listen-settings");
  await page.waitForSelector('#voice-menu:popover-open [data-get="big-stt"]');
  const chips = () => page.$$eval("#voice-menu .voice-chip", (els) => els.map((e) => [(e as HTMLElement).dataset.voice ?? `get:${(e as HTMLElement).dataset.get}`, e.textContent,
    e.getAttribute("aria-checked"), !!e.querySelector(".voice-play")]));
  expect(await chips()).toEqual([["", "Default", "true", false], ["fake-stt", "Fake STT", "false", false], ["other-stt", "Other STT", "false", false],
    ["get:big-stt", "Big STT · Download", null, false], ["get:huge-stt", "Huge STT · Download", null, false]]);
  await page.click('#voice-menu [data-get="big-stt"]');
  await page.waitForSelector('#voice-menu .voice-chip[data-voice="big-stt"]', { timeout: 10_000 });
  expect(voiceLog().filter((e) => e.type === "get").map(({ stt, tts }) => ({ stt, tts })).at(-1)).toEqual({ stt: "big-stt", tts: undefined });
  await page.click('#voice-menu .voice-chip[data-voice="other-stt"] > span');
  await page.waitForFunction(() => !document.querySelector("#voice-menu:popover-open"));
  expect([await page.$eval("#listen-settings .voice-pick-name", (b) => b.textContent), await stored(page)]).toEqual(["Other STT", { name: "silent", stt: "other-stt" }]);
  const configs = () => session().filter((e) => e.type === "config").map(({ stack }) => stack);
  expect(await settled(() => configs().length === 1)).toBe(true);
  expect(configs()).toEqual([{ stt: "other-stt" }]);
  await page.click("#listen-settings");
  await page.waitForFunction(() => document.querySelector('#voice-menu:popover-open .voice-chip[aria-checked="true"]')?.getAttribute("data-voice") === "other-stt");
  await page.click('#voice-menu .voice-chip[data-voice=""] > span');
  expect(await settled(() => configs().length === 2)).toBe(true);
  expect([configs()[1], await stored(page)]).toEqual([{ stt: "fake-stt" }, { name: "silent" }]);
  await closeSettings(page);
  await page.close();
}, 30_000);

test("the model menu picks speech to text and text to speech as it picks a model: the service's engines under the chat's models, the ones in use checked; a pick switches the voice session, and one not downloaded is downloaded, then used", async () => {
  const { page, session } = await voicePage("silent");
  const menu = () => page.$$eval("#model-pop > *", (els) => els.slice(els.findIndex((e) => e.textContent === "Speech to text")).map((e) =>
    e.classList.contains("model-group") ? e.textContent : [(e as HTMLElement).dataset.speech, e.querySelector(".model-label")?.textContent, e.getAttribute("aria-selected"), e.querySelector(".model-check")?.textContent]));
  await page.click("#model-btn");
  await page.waitForSelector('#model-pop:not([hidden]) [data-speech="tts/big-tts"]');
  expect(await menu()).toEqual(["Speech to text", ["stt/fake-stt", "Fake STT", "true", ""], ["stt/other-stt", "Other STT", "false", ""], ["stt/big-stt", "Big STT", "false", "Download"],
    ["stt/huge-stt", "Huge STT", "false", "Download"], "Text to speech", ["tts/fake-tts", "Fake TTS", "true", ""], ["tts/other-tts", "Other TTS", "false", ""], ["tts/big-tts", "Big TTS", "false", "Download"]]);
  const configs = () => session().filter((e) => e.type === "config").map(({ stack, voice }) => ({ stack, voice }));
  await page.click('[data-speech="tts/other-tts"]');
  await page.waitForFunction(() => document.getElementById("model-pop")!.hidden);
  expect(await settled(() => configs().length === 1)).toBe(true);
  expect([configs(), await stored(page)]).toEqual([[{ stack: { tts: "other-tts" }, voice: undefined }], { tts: "other-tts" }]);
  await page.click("#model-btn");
  await page.click('[data-speech="stt/huge-stt"]');
  await page.waitForSelector('[data-speech="stt/huge-stt"] .model-check');
  await page.waitForFunction(() => document.getElementById("model-pop")!.hidden, { timeout: 10_000 });
  expect(voiceLog().filter((e) => e.type === "get").map(({ stt }) => stt).at(-1)).toBe("huge-stt");
  expect(await settled(() => configs().length === 2)).toBe(true);
  expect([configs()[1], await stored(page)]).toEqual([{ stack: { stt: "huge-stt" }, voice: undefined }, { tts: "other-tts", stt: "huge-stt" }]);
  await page.click("#model-btn");
  await page.waitForSelector('#model-pop:not([hidden]) [data-speech="stt/huge-stt"][aria-selected="true"]');
  expect((await menu()).filter((r) => Array.isArray(r) && r[2] === "true").map((r) => r[0])).toEqual(["stt/huge-stt", "tts/other-tts"]);
  await page.close();
}, 30_000);
