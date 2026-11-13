import { expect, test } from "bun:test";
import { writeFileSync } from "node:fs";
import { cardPng } from "./cast";
import { harness, settled, untilIdle, card, checks, dir, fake, fill, inRow, lastStream, noteShown, open, openRow, pickVoice, png, replyText, send, sent, still, streams, voiceLog, voicePage } from "./browser-harness";

harness();

test("a character's voice is picked and heard from their card with voice mode never opened (the voice service lists its voices and speaks the sample, with no hello: no microphone, no speech recognition), Escape closing the voice menu before the card, and voice mode speaks with it", async () => {
  const page = await open();
  await Bun.write(`${dir}/Vox.png`, await cardPng({ spec: "chara_card_v2", spec_version: "2.0", data: { name: "Vox", description: "{{char}} sings.", first_mes: "La." } },
    new Blob([png], { type: "image/png" })));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Vox.png`);
  await page.waitForSelector("#conversations .cast-row");
  const n = voiceLog().length;
  await inRow(page, "Vox", ".cast-pic");
  await pickVoice(page, "#cast-sheet .card-voice .voice-name", "other-tts/bob");
  await page.click("#cast-sheet .card-voice .voice-play");
  const until = async (ok: () => boolean) => { for (let t = Date.now(); !ok(); await Bun.sleep(50)) if (Date.now() - t > 10_000) throw new Error("timed out"); };
  await until(() => voiceLog().slice(n).some((e) => e.type === "preview"));
  const s = voiceLog().slice(n).find((e) => e.type === "catalog").s, own = () => voiceLog().filter((e) => e.s === s);
  expect(own().filter((e) => e.ev === "msg").map((e) => [e.type, e.tts, e.voice])).toEqual([["catalog", undefined, undefined], ["preview", "other-tts", "bob"]]);
  await until(() => own().some((e) => e.ev === "chunk"));
  await page.click("#cast-sheet .card-voice .voice-name");
  await page.waitForSelector("#voice-menu:popover-open");
  await page.keyboard.press("Escape");
  await page.waitForFunction(() => !document.querySelector("#voice-menu:popover-open"));
  expect(await page.$eval("#cast-sheet", (s) => (s as HTMLElement).hidden)).toBe(false);
  await page.keyboard.press("Escape");
  await until(() => own().some((e) => e.ev === "close"));
  const m = voiceLog().length;
  await page.click("#voice");
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 10_000 });
  const asked = voiceLog().slice(m).filter((e) => e.ev === "msg" && (e.type === "hello" || e.type === "config"));
  expect(asked.some((e) => e.voice === "bob" && (e.stack?.tts === "other-tts" || asked.some((x) => x.stack?.tts === "other-tts")))).toBe(true);
  await page.click("#send");
  await page.close();
}, 30_000);

test("a reply is read aloud from its speaker button with voice mode never opened (no hello: no microphone, no speech recognition): in a group each reply in its own speaker's voice; the button shows it, and stops it", async () => {
  const page = await open();
  await Bun.write(`${dir}/Vox.png`, await card("Vox", { description: "{{char}} sings.", extensions: { chrysalis: { voice: { tts: "other-tts", name: "bob" } } } }));
  await Bun.write(`${dir}/Ada.png`, await card("Ada", { description: "{{char}} hums." }));
  writeFileSync(`${dir}/Duet.json`, JSON.stringify({ spec: "chrysalis_group", name: "Duet", members: ["Vox", "Ada"] }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Vox.png`, `${dir}/Ada.png`, `${dir}/Duet.json`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 4);
  await openRow(page, "Duet");
  await page.waitForSelector("#cast-next:not([hidden])");
  await send(page, "Vox, a duet");
  await page.click('#cast-next button[title="Ada answers"]');
  await untilIdle(page);
  expect(await page.$$eval("#thread > article.assistant .speaker", (s) => s.map((x) => x.textContent))).toEqual(["Vox", "Ada"]);
  const n = voiceLog().length, button = '#thread > article[data-i="1"] [data-act="speak"]', hers = '#thread > article[data-i="2"] [data-act="speak"]';
  for (const b of [button, hers]) {
    await page.click(b);
    await page.waitForSelector(`${b}.on`);
    await page.waitForFunction((b) => !document.querySelector(`${b}.on`), { timeout: 10_000 }, b);
  }
  const s = voiceLog().slice(n).find((e) => e.ev === "msg").s, mine = voiceLog().filter((e) => e.s === s);
  const says = mine.filter((e) => e.type === "say");
  expect(says.map((e) => e.text)).toEqual(["*She sings.* La la.", "Ada goes on."]);
  expect(says.map((x) => [...new Set(mine.filter((e) => e.ev === "chunk" && e.id === x.id).map((e) => `${e.tts}/${e.voice}`))]))
    .toEqual([["other-tts/bob"], ["fake-tts/plain"]]);
  await page.click(button);
  await page.waitForSelector(`${button}.on`);
  await page.click(button);
  expect(await page.$(`${button}.on`)).toBeNull();
  const until = async (ok: () => boolean) => { for (let t = Date.now(); !ok(); await Bun.sleep(50)) if (Date.now() - t > 10_000) throw new Error("timed out"); };
  await until(() => voiceLog().filter((e) => e.s === s).some((e) => e.type === "cancel"));
  await page.close();
}, 30_000);

test("voice mode: what is said shows as it is said, becomes the question, and the reply is spoken as it is written", async () => {
  const n0 = streams.length, c0 = checks.length;
  const { page, session } = await voicePage("plain");
  expect(await page.$eval(".composer-box", (el) => el.getBoundingClientRect().height > 0 && !!(el as HTMLElement).offsetParent)).toBe(true);
  expect(await page.$eval("#input", (el) => (el as HTMLTextAreaElement).offsetHeight > 0)).toBe(true);
  expect(await page.$eval("#send", (el) => el.getAttribute("aria-label"))).toBe("End voice mode");
  expect(await page.$eval("#voice-pill", (el) => (el as HTMLElement).title)).toMatch(/^Listening/);
  await page.waitForFunction(() => document.querySelector(".msg.ghost .bubble")?.textContent === "hello", { timeout: 10_000 });
  await page.waitForFunction(() => document.querySelector('#thread > article[data-i="0"] .bubble')?.textContent === "hello there", { timeout: 10_000 });
  expect(await page.$(".msg.ghost")).toBeNull();
  expect(await page.$('#thread > article[data-i="0"] .said')).not.toBeNull();
  await page.waitForFunction(() => /^voice \d\.\d\d s/.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 10_000 });
  const note = await noteShown(page);
  expect(sent(lastStream)).toEqual([{ role: "user", content: `[${note}]\n\nhello there` }]);
  expect(lastStream.continue_final_message).toBeUndefined();
  const check = checks.findLast((b) => b.messages.at(-1).content === "Transcript: hello there");
  expect([check?.temperature, check?.stream, check?.model, check?.messages.length]).toEqual([0, undefined, "qwen3.8-27b-fast", 2]);
  expect(checks.slice(c0).map((b) => b.messages.at(-1).content)).toEqual(["Transcript: hello there"]);
  expect(await page.$eval("#energy", (e) => e.getAttribute("aria-valuetext"))).toBe("Max");
  expect([lastStream.model, lastStream.reasoning_effort]).toEqual(["qwen3.8-27b", "xhigh"]);
  const mine = streams.slice(n0), warm = mine.findIndex((b) => b.max_tokens === 1);
  expect(mine[warm]?.messages).toEqual([{ role: "user", content: "Hi" }]);
  expect(mine[warm]?.model).toBe(lastStream.model);
  expect(warm).toBeLessThan(mine.findIndex((b) => sent(b).at(-1)?.content.endsWith("hello there")));
  expect(await replyText(page, 1)).toBe("You said: hello there");
  expect(await page.$eval('#thread > article[data-i="0"] .msg-note summary', (e) => e.textContent)).toBe("Voice mode note");
  expect(await page.$eval('#thread > article[data-i="0"] .msg-note-text', (e) => e.checkVisibility())).toBe(false);
  await page.click('#thread > article[data-i="0"] .msg-note summary');
  expect(await page.$eval('#thread > article[data-i="0"] .msg-note-text', (e) => [e.textContent, e.checkVisibility()])).toEqual([note, true]);
  const log = session(), says = log.filter((e) => e.type === "say"), id = says[0].id;
  expect(says.every((e) => e.id === id)).toBe(true);
  expect(says.map((e) => e.text).join("")).toBe("You said: hello there");
  expect(log.findIndex((e) => e.type === "say_end" && e.id === id)).toBeGreaterThan(log.findLastIndex((e) => e.type === "say"));
  await page.waitForFunction(() => document.querySelector("#voice-status")?.textContent === "Listening", { timeout: 10_000 });
  await settled(() => session().filter((e) => e.type === "playback").length >= 2);
  expect(session().filter((e) => e.type === "playback").map((e) => e.playing)).toEqual([true, false]);
  const told = session().filter((e) => e.type === "context").map((e) => e.text);
  expect(told[0]).toMatch(/^Qwen3\.8-27B\n/);
  expect(told.at(-1)).toMatch(/hello there\nYou said: hello there$/);
  await page.close();
}, 30_000);

test("voice mode: a question said with the pointer on a part of an artifact points at it (\"make this orange\")", async () => {
  const { page } = await voicePage("point", { keepPointer: true, first: async (page) => {
    await send(page, "draw the panda");
    await still(page);
    const at = await page.$eval(".svg-fig[data-art] .svg-paper", (p) => {
      const r = p.shadowRoot!.querySelector("#burger rect")!.getBoundingClientRect();
      return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
    });
    await page.mouse.move(at.x, at.y);
  } });
  await page.waitForFunction(() => document.querySelector('#thread > article[data-i="2"] .bubble')?.textContent === "make this orange", { timeout: 10_000 });
  await page.waitForSelector('#thread > article[data-i="3"] .stats', { timeout: 10_000 });
  expect(sent(lastStream).at(-1).content).toContain('The user is pointing at the part #burger of the artifact panda:\n<g id="burger">');
  expect(await page.$eval('#thread > article[data-i="2"] .point-chip', (e) => e.textContent)).toBe("panda · #burger");
  expect(await page.$eval(".art-canvas[data-v='2'] .svg-paper", (p) => p.shadowRoot!.querySelector("#burger rect")?.getAttribute("fill"))).toBe("#ff8c00");
}, 30_000);
