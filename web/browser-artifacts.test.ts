import { expect, test } from "bun:test";
import type { Page } from "puppeteer-core";
import { harness, fakes, card, fill, inPage, lastStream, look, open, pageFrame, pageTool, pickColor, send, sent, shown, still, streams } from "./browser-harness";

harness();

test("an artifact is changed in place: each reply that makes a version has a card; the panel shows the latest, the edit's changes (a whole element by its id, a search) painted into the picture as it was, steps back to the version before, opens the version a card names, and Compare shows both side by side", async () => {
  const page = await open();
  await send(page, "draw the panda");
  await send(page, "red coat");
  expect(await page.$$eval(".art-card", (c) => c.map((x) => [x.closest("article")!.dataset.i, x.querySelector(".art-name")!.textContent, x.querySelector(".art-sub")!.textContent])))
    .toEqual([["1", "panda", "SVG · v1 · now v2"], ["3", "panda", "SVG · v2 · 2 changes"]]);
  await still(page);
  expect(await page.$eval("#shelf .art-canvas .art-label", (l) => l.textContent)).toBe("panda · v2 · 2 changes");
  expect([await fill(page, "coat"), await fill(page, "background"), await fill(page, "burger")]).toEqual(["#d32f2f", "#fff3cd", null]);
  expect(await page.$eval(".art-canvas .svg-paper", (p) => p.shadowRoot!.querySelector("#burger rect")?.getAttribute("fill"))).toBe("#8b4513");
  await page.click(".art-canvas [data-art=prev]");
  expect([await page.$eval(".art-canvas .art-label", (l) => l.textContent), await fill(page, "coat")]).toEqual(["panda · v1", "#ffffff"]);
  await page.click(".art-canvas [data-art=next]");
  expect(await fill(page, "coat")).toBe("#d32f2f");
  await page.click('.art-card[data-v="1"]');
  expect([await page.$eval(".art-canvas .art-label", (l) => l.textContent), await page.$$eval(".art-card.on", (c) => c.map((x) => (x as HTMLElement).dataset.v))])
    .toEqual(["panda · v1", ["1"]]);
  await page.click('.art-card[data-v="2"]');
  await page.click("#shelf-compare");
  const sides = () => page.$$eval("#shelf-stage > .art-canvas", (s) => s.map((x) => [x.querySelector(".art-label")!.textContent,
    x.querySelector(".svg-paper")!.shadowRoot!.querySelector("#coat")?.getAttribute("fill")]));
  expect(await sides()).toEqual([["panda · v1", "#ffffff"], ["panda · v2 · 2 changes", "#d32f2f"]]);
  await page.close();
}, 30_000);

test("the canvas's tools change the picture by hand, with no model: the paint bucket fills the shape clicked, a selection dragged moves, a corner dragged resizes, Delete removes it and Undo brings it back; each change is the artifact's next version, and the next question carries it whole, which the model's edit then changes", async () => {
  const page = await open();
  await send(page, "draw the panda");
  await still(page);
  const n0 = streams.length;
  const inPaper = (sel: string, fy = 0.5) => page.$eval(".art-canvas .svg-paper", (p, sel, fy) => {
    const r = p.shadowRoot!.querySelector(sel as string)!.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height * (fy as number) };
  }, sel, fy);
  const tools = (sel: string) => page.click(`.art-canvas .art-tools ${sel}`);
  const caption = () => page.$eval(".art-canvas .art-label", (l) => l.textContent);
  const coat = await inPaper("#coat", 0.85);
  await page.mouse.click(coat.x, coat.y);
  await page.keyboard.press("Delete");
  expect([await page.$eval("#pointing", (e) => e.textContent), await fill(page, "coat"), await caption()]).toEqual(["Pointing at panda · #coat", "#ffffff", "panda · v1"]);
  await page.keyboard.press("Escape");
  await page.click(".art-canvas [data-art=edit]");
  await page.$eval(".art-canvas .art-tools input", (i) => {
    (i as HTMLInputElement).value = "#00aa00";
    i.dispatchEvent(new Event("input"));
  });
  const coatNow = await inPaper("#coat", 0.85);
  await page.mouse.click(coatNow.x, coatNow.y);
  expect([await fill(page, "coat"), await caption()]).toEqual(["#00aa00", "panda · v2 · your edit, not sent yet"]);
  await tools("[data-tool=select]");
  const burger = await inPaper("#burger rect");
  await page.mouse.click(burger.x, burger.y);
  expect(await page.$eval("#pointing", (e) => e.textContent)).toBe("Pointing at panda · #burger");
  const box = await inPaper(".sel");
  await page.mouse.move(box.x, box.y);
  await page.mouse.down();
  await page.mouse.move(box.x + 10, box.y, { steps: 3 });
  await page.mouse.move(box.x + 20, box.y, { steps: 3 });
  await page.mouse.up();
  const shifted = await fill(page, "burger", "transform");
  expect(shifted).toMatch(/^translate\(\d+(\.\d+)? 0\)$/);
  const se = await inPaper(".sel i[data-h=se]"), nw = await inPaper(".sel i[data-h=nw]");
  await page.mouse.move(se.x, se.y);
  await page.mouse.down();
  await page.mouse.move(nw.x + (se.x - nw.x) * 1.5, nw.y + (se.y - nw.y) * 1.5, { steps: 4 });
  await page.mouse.up();
  const moved = await fill(page, "burger", "transform");
  expect(moved).toMatch(/^translate\(-?\d+(\.\d+)? -?\d+(\.\d+)?\) scale\(1\.(4|5)\d*\)$/);
  await page.keyboard.press("Delete");
  expect(await page.$eval(".art-canvas .svg-paper", (p) => p.shadowRoot!.querySelector("#burger"))).toBeNull();
  await tools("[data-do=undo]");
  expect([await fill(page, "burger", "transform"), await caption()]).toEqual([moved, "panda · v2 · your edit, not sent yet"]);
  expect(streams.length).toBe(n0);
  await send(page, "red coat");
  const asked = lastStream.messages.at(-1).content as string;
  expect([asked.includes("The user changed panda by hand since your last reply. As it is now:"), asked.includes('fill="#00aa00"'),
    asked.includes(`transform="${moved}"`)]).toEqual([true, true, true]);
  expect([await fill(page, "coat"), await fill(page, "burger", "transform"), await caption()]).toEqual(["#d32f2f", moved, "panda · v3 · 2 changes"]);
  expect(await page.$$eval(".edit-chip", (c) => c.map((x) => x.textContent))).toEqual(["you changed panda"]);
  await page.close();
}, 30_000);

test("while a page is being written the pill in its header, the only thing saying so, shows the code so far, which grows as it is written, and the same pill, where it was, brings the page back", async () => {
  const page = await open();
  await page.type("#input", "steady counter page");
  await page.keyboard.press("Enter");
  await page.waitForSelector(".page-fig.drawing .page-writing", { timeout: 10_000 });
  const spot = () => page.$eval(".page-fig.drawing .page-writing", (b) => { const r = b.getBoundingClientRect(), f = b.closest(".page-fig")!.getBoundingClientRect(); return [Math.round(r.left - f.left), Math.round(r.top - f.top), b.querySelector(".page-peek")!.textContent] as const; });
  const [x, y] = await spot();
  expect(await page.$eval(".page-fig.drawing", (f) => [!!f.querySelector(".svg-bar > .page-writing"),
    [...f.querySelectorAll<HTMLElement>("*")].filter((e) => e.checkVisibility() && [...e.childNodes].some((n) => n.nodeType === 3 && n.textContent!.includes("Writing"))).length])).toEqual([true, 1]);
  await page.click(".page-fig.drawing .page-writing");
  expect(await spot()).toEqual([x, y, "Page"]);
  expect(await page.$eval(".page-fig.drawing", (f) => Math.round(f.querySelector(".svg-code")!.getBoundingClientRect().top - f.querySelector(".page-writing")!.getBoundingClientRect().bottom) < 40)).toBe(true);
  const code = () => page.$eval(".page-fig .svg-code code", (c) => c.textContent ?? "");
  expect(await page.$eval(".page-fig", (f) => f.classList.contains("show-code"))).toBe(true);
  await page.waitForFunction(() => (document.querySelector(".page-fig .svg-code code")?.textContent ?? "").length > 0, { timeout: 10_000 });
  const first = await code();
  await page.waitForFunction((n) => (document.querySelector(".page-fig .svg-code code")?.textContent ?? "").length > n, { timeout: 10_000 }, first.length);
  expect((await code()).startsWith(first)).toBe(true);
  await page.click(".page-fig.drawing .page-writing");
  expect([await page.$eval(".page-fig", (f) => f.classList.contains("show-code")), await spot()]).toEqual([false, [x, y, "Code"]]);
  fakes.hurry = true;
  await page.waitForSelector(".msg.assistant .stats", { timeout: 20_000 });
  await page.close();
}, 40_000);

test("a page artifact is drawn as it is written, its script waiting, and runs once whole; an edit to its look goes into the running page, which keeps what it holds, and one to its script runs it anew", async () => {
  const page = await open();
  const done = await page.$$eval(".msg .stats", (s) => s.length);
  await page.type("#input", "steady counter page");
  await page.keyboard.press("Enter");
  const frame = await pageFrame(page);
  await frame.waitForSelector("#title", { timeout: 10_000 });
  expect([await page.$$eval(".msg .stats", (s) => s.length), await frame.$eval("body", (b) => b.dataset.ran ?? "")]).toEqual([done, ""]);
  expect(await page.$eval(".art-canvas", (f) => f.classList.contains("drawing"))).toBe(true);
  fakes.hurry = true;
  await page.waitForFunction((n) => document.querySelectorAll(".msg .stats").length > n, { timeout: 20_000 }, done);
  await frame.waitForFunction(() => document.body.dataset.ran === "yes");
  await still(page);
  for (let k = 0; k < 3; k++) await frame.click("#inc");
  expect(await frame.$eval("#count", (c) => c.textContent)).toBe("3");
  await send(page, "blue button");
  await frame.waitForFunction(() => getComputedStyle(document.getElementById("inc")!).backgroundColor === "rgb(0, 0, 255)");
  expect([await frame.$eval("#count", (c) => [c.className, c.textContent]), await page.$$eval(".art-canvas iframe", (f) => f.length)]).toEqual([["big", "3"], 1]);
  expect(await page.$eval(".art-canvas .art-label", (l) => l.textContent)).toBe("counter · v2 · 2 changes");
  await send(page, "count by two");
  await page.waitForFunction(() => document.querySelectorAll(".art-canvas iframe").length === 1 && !document.querySelector(".art-canvas iframe.next"), { timeout: 10_000 });
  const anew = await pageFrame(page);
  await anew.waitForFunction(() => document.body.dataset.ran === "yes");
  expect(await anew.$eval("#count", (c) => c.textContent)).toBe("0");
  await anew.click("#inc");
  expect([await anew.$eval("#count", (c) => c.textContent), await anew.$eval("#inc", (b) => getComputedStyle(b).backgroundColor)]).toEqual(["2", "rgb(0, 0, 255)"]);
  expect(await page.$eval(".art-canvas .art-label", (l) => l.textContent)).toBe("counter · v3 · 1 change");
  await page.close();
}, 40_000);

test("pages labelled as models also label them, the id alone on the block's first line (its fence left open as the reply ends) or as an attribute, are artifacts by that id, and one with none is \"page\": each runs as a page with the tools, and an edit by its id changes it", async () => {
  const page = await open(), shown = (label: string) => page.waitForFunction((label) => document.querySelector("#shelf .art-canvas .art-label")?.textContent === label, { timeout: 10_000 }, label);
  await send(page, "a hall on its line");
  await shown("hall · v1");
  const sign = (text: string) => pageFrame(page).then((f) => f.waitForFunction((text) => document.querySelector("#sign")?.textContent === text, { timeout: 10_000 }, text));
  await sign("Welcome");
  await send(page, "rename the sign");
  await sign("Goodbye");
  await send(page, "a gate as an attribute");
  await shown("gate · v1");
  expect(await page.$$eval(".art-card", (c) => c.map((x) => `${x.querySelector(".art-name")!.textContent}: ${x.querySelector(".art-sub")!.textContent}`)))
    .toEqual(["hall: Page · v1 · now v2", "hall: Page · v2 · 1 change", "gate: Page · v1"]);
  expect(await page.$$eval(".shelf-tab", (t) => t.map((x) => x.textContent))).toEqual(["hall", "gate"]);
  await send(page, "a page with no name");
  await shown("page · v1");
  await page.click("#shelf .art-canvas [data-art=edit]");
  await page.waitForSelector("#shelf .art-canvas .art-tools.on");
  expect(await (await pageFrame(page)).$eval("#lamp", (e) => e.textContent)).toBe("Lamp");
  await page.click(".shelf-tab:first-child");
  await shown("hall · v2 · 1 change");
  await sign("Goodbye");
  await page.close();
}, 30_000);

test("a page's tools change it by hand, with no model, in the page as it runs: Select points at a part, Text retypes it, the paint bucket fills a background, Delete takes a part out and Undo brings it back; the next question carries the page whole and the part pointed at", async () => {
  const page = await open();
  await send(page, "counter page");
  const frame = await pageFrame(page);
  await frame.waitForFunction(() => document.body.dataset.ran === "yes");
  await still(page);
  const n0 = streams.length, caption = () => page.$eval(".art-canvas .art-label", (l) => l.textContent);
  await frame.click("#inc");
  await frame.click("#inc");
  await pageTool(page, "[data-tool=text]");
  let at = await inPage(page, "#title");
  await page.mouse.click(at.x, at.y);
  await page.keyboard.press("End");
  await page.keyboard.type(" so far");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /your edit/.test(document.querySelector(".art-canvas .art-label")?.textContent ?? ""));
  expect([await frame.$eval("#title", (t) => t.textContent), await caption()]).toEqual(["Clicks so far", "counter · v2 · your edit, not sent yet"]);
  await pickColor(page, "#00aa00");
  await (await pageFrame(page)).waitForFunction(() => (window as any).__chrysalis.mode === "bucket");
  at = await inPage(page, "#card");
  await page.mouse.click(at.x, at.y - 30);
  await frame.waitForFunction(() => getComputedStyle(document.getElementById("card")!).backgroundColor === "rgb(0, 170, 0)");
  await pageTool(page, "[data-tool=select]");
  at = await inPage(page, "#note");
  await page.mouse.click(at.x, at.y);
  await page.waitForFunction(() => document.getElementById("pointing")!.textContent === "Pointing at counter · #note");
  await page.keyboard.press("Delete");
  await frame.waitForFunction(() => !document.getElementById("note"));
  await page.click(".art-canvas .art-tools [data-do=undo]");
  await frame.waitForFunction(() => !!document.getElementById("note"));
  expect([await frame.$eval("#count", (c) => c.textContent), await frame.$eval("#title", (t) => t.textContent), streams.length]).toEqual(["2", "Clicks so far", n0]);
  at = await inPage(page, "#note");
  await page.mouse.click(at.x, at.y);
  await page.waitForFunction(() => document.getElementById("pointing")!.textContent === "Pointing at counter · #note");
  await send(page, "capital of France");
  const asked = lastStream.messages.at(-1).content as string;
  expect([asked.includes("The user changed counter by hand since your last reply. As it is now:"), asked.includes('<h1 id="title">Clicks so far</h1>'),
    asked.includes('<section class="card" id="card" style="background: #00aa00">'), asked.includes('<p id="note">Press the button.</p>'),
    asked.includes('The user is pointing at the part #note of the artifact counter:\n<p id="note">Press the button.</p>')]).toEqual([true, true, true, true, true]);
  await page.close();
}, 40_000);

test("an artifact being written can be left for another: a tab chosen in the panel while it is written stays chosen until the reply ends", async () => {
  const page = await open(), shown = () => page.$eval("#shelf .art-canvas", (c) => c.getAttribute("data-art"));
  await send(page, "draw the panda");
  const done = await page.$$eval(".msg .stats", (s) => s.length);
  await page.type("#input", "steady counter page");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector("#shelf .art-canvas")?.getAttribute("data-art") === "counter" && !!document.querySelector(".art-card.live"),
    { timeout: 10_000 });
  await page.click(".shelf-tab:first-child");
  await Bun.sleep(500);
  expect([await shown(), await page.$$eval(".art-card.live", (c) => c.length)]).toEqual(["panda", 1]);
  fakes.hurry = true;
  await page.waitForFunction((n) => document.querySelectorAll(".msg .stats").length > n, { timeout: 20_000 }, done);
  expect(await shown()).toBe("panda");
  await page.close();
}, 30_000);
