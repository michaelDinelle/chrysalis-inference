import { expect, test } from "bun:test";
import { harness, checked, checks, lastStream, replyText, sent, spokenOf, still, streams, voicePage } from "./browser-harness";

harness();

test("voice mode: a question shows grey as it was heard until its check is in, then white with the misheard words fixed (the heard ones on hover); the reply begun on the words as heard is dropped unheard, and the fixed question is answered", async () => {
  const n0 = streams.length, c0 = checks.length, { page, session } = await voicePage("pie");
  const q = '#thread > article[data-i="0"]';
  await page.waitForFunction((q) => document.querySelector(`${q} .bubble`)?.textContent === "lets do a snake game in pie game", { timeout: 10_000 }, q);
  expect(await page.$eval(q, (e) => [e.classList.contains("unsettled"), getComputedStyle(e.querySelector(".bubble")!).color]))
    .toEqual([true, await page.evaluate(() => { const d = document.createElement("div"); d.style.color = "var(--muted)"; document.body.append(d);
      const c = getComputedStyle(d).color; d.remove(); return c; })]);
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  expect(await page.$eval(`${q} .bubble`, (e) => e.textContent)).toBe("lets do a snake game in Pygame");
  expect(await page.$eval(`${q} .bubble .fixed`, (e) => [e.textContent, (e as HTMLElement).title])).toEqual(["Pygame", 'Speech recognition heard "pie game"']);
  expect(await page.$eval(q, (e) => e.classList.contains("unsettled"))).toBe(false);
  expect(await page.$$eval("#thread > article.msg", (els) => els.length)).toBe(2);
  expect(sent(lastStream).at(-1).content).toMatch(/\n\nlets do a snake game in Pygame$/);
  expect(JSON.stringify(lastStream)).not.toMatch(/=>|pie game/);
  expect(await replyText(page, 1)).toBe("You said: lets do a snake game in Pygame");
  const replies = streams.slice(n0).filter((b) => b.max_tokens !== 1).map((b) => sent(b).at(-1).content.split("\n\n").at(-1));
  expect(replies).toEqual(["lets do a snake game in pie game", "lets do a snake game in Pygame"]);
  expect(checks.slice(c0).map((b) => b.messages.at(-1).content)).toEqual(["Transcript: lets do a snake game in pie game"]);
  const { said, cancelled } = spokenOf(session());
  expect([...said].filter(([id]) => !cancelled.has(id)).map(([, t]) => t)).toEqual(["You said: lets do a snake game in Pygame"]);
  expect([...said].filter(([id]) => cancelled.has(id)).map(([, t]) => t)).toEqual(["You said: lets do a snake game in pie game"]);
  const log = session();
  expect(log.findIndex((e) => e.type === "playback" && e.playing)).toBeGreaterThan(log.findIndex((e) => e.type === "cancel"));
  await page.close();
}, 30_000);

test("voice mode: words said while a question is still being checked join it, and the two are answered once", async () => {
  const n0 = streams.length, { page, session } = await voicePage("join");
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 20_000 });
  expect(await page.$eval('#thread > article[data-i="0"] .bubble', (e) => e.textContent)).toBe("wait lets do a snake game in Pygame");
  expect(await page.$$eval("#thread > article.msg:not(.ghost)", (els) => els.length)).toBe(2);
  expect(await replyText(page, 1)).toBe("You said: wait lets do a snake game in Pygame");
  const replies = streams.slice(n0).filter((b) => b.max_tokens !== 1);
  expect(sent(replies.at(-1)).at(-1).content.split("\n\n").at(-1)).toBe("wait lets do a snake game in Pygame");
  const { said, cancelled } = spokenOf(session());
  expect([...said].filter(([id]) => !cancelled.has(id)).map(([, t]) => t)).toEqual(["You said: wait lets do a snake game in Pygame"]);
  await page.close();
}, 30_000);

test("voice mode: the words are checked while they are said: in a pause they turn their own color with the misheard ones fixed", async () => {
  const { page } = await voicePage("checking");
  await page.waitForFunction(() => document.querySelector(".msg.ghost .bubble .checked .fixed")?.textContent === "Pygame", { timeout: 10_000 });
  const ghost = await page.$eval(".msg.ghost .bubble", (b) => [b.textContent, getComputedStyle(b.querySelector(".checked")!).color, getComputedStyle(b).color]);
  const color = (v: string) => page.evaluate((v) => { const d = document.createElement("div"); d.style.color = `var(${v})`; document.body.append(d);
    const c = getComputedStyle(d).color; d.remove(); return c; }, v);
  expect(ghost).toEqual(["lets do a snake game in Pygame", await color("--text"), await color("--muted")]);
  await page.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 15_000 });
  expect(await replyText(page, 1)).toBe("You said: lets do a snake game in Pygame");
  await page.close();
}, 30_000);
