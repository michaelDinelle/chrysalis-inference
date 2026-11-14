import { expect, test } from "bun:test";
import { harness, fakes, CLEAN, SIZES, composerFaults, idle, look, open, resize, send, shown, still, voicePage } from "./browser-harness";

harness();

test("the context in the message box counts up while a reply is written (the prompt once read, then each token) and ends at the reply's own count", async () => {
  const page = await open();
  const used = async () => parseInt(/This chat: ([\d,]+) of 262,144/.exec(await page.$eval("#meter .ctx", (b) => (b as HTMLElement).title))![1].replace(/,/g, ""));
  await page.type("#input", "slow count");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /w3 /.test(document.querySelector(".msg.assistant")?.textContent ?? ""));
  const early = await used();
  await page.waitForFunction(() => /w20 /.test(document.querySelector(".msg.assistant")?.textContent ?? ""));
  const later = await used();
  expect([early > 10, later > early]).toEqual([true, true]);
  fakes.hurry = true;
  await page.waitForSelector(".msg.assistant .stats");
  expect(await used()).toBe(30);
  await page.close();
}, 30_000);

test("Chrysalis is dark whatever the system prefers or an old choice of Light left behind, and Settings has no theme to pick", async () => {
  const page = await open();
  const look = () => page.evaluate(() => {
    const [r, g, b] = getComputedStyle(document.body).backgroundColor.match(/[\d.]+/g)!.map(Number);
    return [0.2126 * r + 0.7152 * g + 0.0722 * b < 60, getComputedStyle(document.documentElement).colorScheme, document.querySelector("#theme-pick")];
  });
  const dark = [true, "dark", null];
  for (const value of ["dark", "light"]) {
    await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value }]);
    expect(await look()).toEqual(dark);
  }
  await page.evaluate(() => localStorage.setItem("theme", "light"));
  await idle(page, page.reload());
  expect(await look()).toEqual(dark);
  await page.close();
}, 30_000);

test("the message box holds every control inside it at each size from a big screen to a small phone, in a chat and in voice mode, and shows no blank button; voice mode keeps its pill, its voice and its stop", async () => {
  const page = await open();
  await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "reduce" }]);
  expect(await composerFaults(page)).toEqual(CLEAN);
  expect(await shown(page, "#voice-pick")).toBe(false);
  await send(page, "capital of France");
  for (const [width, height] of SIZES) {
    await resize(page, width, height);
    expect({ width, ...await composerFaults(page) }).toEqual({ width, ...CLEAN });
    expect(await shown(page, "#draft")).toBe(true);
  }
  await page.close();
  const { page: talking } = await voicePage("silent");
  await talking.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "reduce" }]);
  for (const [width, height] of SIZES) {
    await resize(talking, width, height);
    expect({ width, ...await composerFaults(talking) }).toEqual({ width, ...CLEAN });
    expect([await shown(talking, "#voice-pill"), await shown(talking, "#voice-pick"), await shown(talking, "#send")]).toEqual([true, true, true]);
  }
  await resize(talking, 1600, 900);
  await talking.click("#send");
  await talking.waitForFunction(() => !document.querySelector("#composer.voicing"));
  expect(await composerFaults(talking)).toEqual(CLEAN);
  expect(await shown(talking, "#voice-pick")).toBe(false);
}, 60_000);

test("the model list opens over its button and stays on the page at each size", async () => {
  const page = await open();
  await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "reduce" }]);
  for (const [width, height] of SIZES) {
    await resize(page, width, height);
    await page.click("#model-btn");
    await page.waitForSelector("#model-pop:not([hidden])");
    const at = await page.evaluate(() => {
      const pop = document.querySelector("#model-pop")!.getBoundingClientRect(), btn = document.querySelector("#model-btn")!.getBoundingClientRect();
      return { fits: pop.left >= 0 && pop.right <= innerWidth, over: pop.bottom <= btn.top && pop.left < btn.right && pop.right > btn.left };
    });
    expect([width, at]).toEqual([width, { fits: true, over: true }]);
    await page.keyboard.press("Escape");
    await page.waitForSelector("#model-pop[hidden]", { hidden: true });
  }
  await page.close();
}, 30_000);

test("a table fills its reply edge to edge with the columns' alignment kept, and a table with nothing around it shows no second box", async () => {
  const page = await open();
  await send(page, "a small table");
  const small = await page.$eval(".msg.assistant .table", (box) => {
    const bubble = box.closest(".markdown")!.getBoundingClientRect(), table = box.querySelector("table")!.getBoundingClientRect(), style = (s: string) => getComputedStyle(box.querySelector(s)!);
    return { edges: [Math.round(table.left - bubble.left), Math.round(bubble.right - table.right)], aligned: [style("th").textAlign, style("td:nth-child(2)").textAlign, style("td:nth-child(3)").textAlign].map((a) => a.replace("-webkit-", "")),
      column: Math.round(bubble.right - document.querySelector(".composer-box")!.getBoundingClientRect().right) };
  });
  expect(small).toEqual({ edges: [1, 1], aligned: ["left", "right", "center"], column: 0 });
  await send(page, "only a table");
  expect(await page.$$eval(".msg.assistant .table", (b) => {
    const box = b.at(-1)!, bubble = box.closest(".markdown")!, inner = box.getBoundingClientRect(), outer = bubble.getBoundingClientRect();
    return [Math.round(inner.top - outer.top), Math.round(outer.bottom - inner.bottom), getComputedStyle(box).borderTopWidth, getComputedStyle(box).borderBottomWidth];
  })).toEqual([1, 1, "0px", "0px"]);
  await page.close();
}, 30_000);

test("no table runs past its bubble at any size: it wraps at its words and tightens, and with no room left each row becomes a list of labelled values, a long link breaking to fit, and it is a table again when the room comes back", async () => {
  const page = await open();
  await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "reduce" }]);
  await send(page, "a wide table");
  await send(page, "a long link");
  const fit = () => page.$$eval(".msg.assistant .table", (boxes) => boxes.map((box) => {
    const words = (el: Element) => [...(el.firstChild as Text).data.matchAll(/\S+/g)].map((m) => {
      const r = document.createRange();
      r.setStart(el.firstChild!, m.index);
      r.setEnd(el.firstChild!, m.index + m[0].length);
      return r.getClientRects().length;
    });
    const cells = [...box.querySelectorAll("th, td")].filter((c) => c.getBoundingClientRect().width > 0), link = box.querySelector("td:nth-child(2)")!.textContent!.startsWith("https");
    return { stacked: box.classList.contains("stacked"), cut: box.scrollWidth > box.clientWidth, split: link ? 0 : cells.flatMap(words).filter((n) => n !== 1).length,
      label: box.classList.contains("stacked") ? getComputedStyle(box.querySelector("td:nth-child(2)")!, "::before").content : "",
      sides: (() => {
        const outer = box.getBoundingClientRect(), fields = [...box.querySelectorAll("tbody tr:first-child td")].map((c) => c.getBoundingClientRect());
        return [Math.round(Math.min(...fields.map((f) => f.left)) - outer.left), Math.round(outer.right - Math.max(...fields.map((f) => f.right)))];
      })(),
      inColumn: box.closest(".markdown")!.getBoundingClientRect().right <= document.querySelector(".composer-box")!.getBoundingClientRect().right + 0.5 };
  }));
  for (const [width, height] of SIZES) {
    await resize(page, width, height);
    const [wide, link] = await fit();
    expect({ width, cut: [wide.cut, link.cut], split: wide.split, inColumn: [wide.inColumn, link.inColumn] }).toEqual({ width, cut: [false, false], split: 0, inColumn: [true, true] });
    if (wide.stacked) expect([wide.label, wide.sides[0] - wide.sides[1]]).toEqual(['"Column number 1"', 0]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  }
  expect((await fit())[0].stacked).toBe(true);
  await resize(page, 1920, 1080);
  expect((await fit())[0].stacked).toBe(false);
  await page.close();
}, 40_000);

test("rows of pipes with the dashes left out, as the model sometimes writes them, are a table, after a line of text and while still being written; pipes in two lines, a list, uneven lines or code stay as they are", async () => {
  const page = await open();
  await page.evaluate(() => {
    (window as any).stray = 0;
    new MutationObserver(() => {
      const reply = [...document.querySelectorAll(".msg.assistant .markdown")].at(-1);
      if (reply?.querySelector("table") && [...reply.querySelectorAll(":scope > p")].some((p) => p.textContent!.includes("|"))) (window as any).stray++;
    }).observe(document.getElementById("thread")!, { childList: true, subtree: true, characterData: true });
  });
  await send(page, "steady a bare table");
  expect(await page.$eval(".msg.assistant .markdown", (m) => ({
    blocks: [...m.children].map((c) => c.tagName), head: [...m.querySelectorAll("th")].map((c) => c.textContent),
    rows: [...m.querySelectorAll("tbody tr")].map((r) => [...r.children].map((c) => c.textContent)), bold: m.querySelector("td strong")?.textContent,
  }))).toEqual({ blocks: ["P", "DIV", "P"], head: ["Fruit", "Color", "Typical Season"], bold: "Summer",
    rows: [["Apple", "Red/Green", "Fall"], ["Banana", "Yellow", "Summer"], ["Grape", "Purple", "Summer"]] });
  expect(await page.evaluate(() => (window as any).stray)).toBe(0);
  await send(page, "pipes but no table");
  expect(await page.$$eval(".msg.assistant .markdown", (m) => [m.at(-1)!.querySelectorAll("table").length, m.at(-1)!.querySelectorAll("li").length, m.at(-1)!.querySelector("pre")?.textContent]))
    .toEqual([0, 3, "q | r\ns | t\nu | v"]);
  await page.close();
}, 30_000);

test("the chat uses the room it has: on a big screen long messages run far past the old column and every message lines up with the message box; on a phone faces sit over the bubbles, which take the whole width, and the stats have a line of their own", async () => {
  const page = await open();
  await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "reduce" }]);
  await resize(page, 1920, 1080);
  await send(page, "a long reply please, ".repeat(30));
  const rect = (s: string) => page.$eval(s, (e) => { const r = e.getBoundingClientRect(); return { left: Math.round(r.left), right: Math.round(r.right), width: Math.round(r.width) }; });
  const big = { box: await rect(".composer-box"), mine: await rect(".msg.user .bubble"), myFace: await rect(".msg.user .speaker .face"), hers: await rect(".msg.assistant > .markdown"), herFace: await rect(".msg.assistant .speaker .face") };
  expect([big.box.width >= 1100, big.mine.width > 900, big.hers.width > 1000]).toEqual([true, true, true]);
  expect([big.herFace.left - big.box.left, big.hers.right - big.box.right, big.myFace.right - big.box.right]).toEqual([0, 0, 0]);
  await resize(page, 390, 844);
  const box = await rect(".composer-box"), hers = await rect(".msg.assistant > .markdown"), mine = await rect(".msg.user .bubble");
  expect([hers.left - box.left, hers.right - box.right, mine.right - box.right]).toEqual([0, 0, 0]);
  expect(await page.$eval(".msg.assistant", (m) => {
    const name = m.querySelector(".speaker > span:last-child")!.getBoundingClientRect(), face = m.querySelector(".speaker .face")!.getBoundingClientRect(), bubble = m.querySelector(":scope > .markdown")!.getBoundingClientRect();
    const stats = m.querySelector(".stats")!.getBoundingClientRect(), foot = m.querySelector(".msg-foot")!.getBoundingClientRect();
    return { faceAbove: face.bottom <= bubble.top, nameBeside: name.left >= face.right, statsWidth: Math.round(stats.width) === Math.round(foot.width) };
  })).toEqual({ faceAbove: true, nameBeside: true, statsWidth: true });
  await page.close();
}, 30_000);
