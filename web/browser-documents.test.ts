import { expect, test } from "bun:test";
import { harness, docRoot, fill, inDoc, lastStream, open, pageFrame, pageTool, pickColor, probes, send, sent, shown, still, streams } from "./browser-harness";

harness();

test("a LaTeX artifact reads as a document; pointing at a paragraph quotes its source, a double click retypes it there with no model, and the model's edit shows its changes as a diff", async () => {
  const page = await open();
  await send(page, "write the paper");
  await still(page);
  expect(await docRoot(page)).toBe(true);
  const shown = () => page.$eval(".art-canvas .doc-paper", (p) => p.shadowRoot!.querySelector(".doc-body")!.textContent!.replace(/\s+/g, " "));
  expect(await shown()).toMatch(/Rods.*1\s*Heat.*The rod cools as.*\(1\).*which is Fourier’s law\..*A second paragraph\./);
  let at = await inDoc(page, "A second paragraph");
  await page.mouse.click(at.x, at.y);
  expect(await page.$eval("#pointing", (e) => e.textContent)).toBe("Pointing at paper · “A second paragraph.”");
  await send(page, "capital of France");
  expect(lastStream.messages.at(-1).content).toBe("[The user is pointing at this part of the artifact paper:\nA second paragraph.]\n\ncapital of France");
  const n0 = streams.length;
  await page.click(".art-canvas [data-art=edit]");
  await still(page);
  at = await inDoc(page, "A second paragraph");
  await page.mouse.click(at.x, at.y, { count: 2 });
  await page.waitForFunction(() => !!document.querySelector(".art-canvas .doc-paper")!.shadowRoot!.querySelector("textarea.retype"));
  expect(await page.$eval(".art-canvas .doc-paper", (p) => p.shadowRoot!.querySelector("textarea")!.value)).toBe("A second paragraph.");
  await page.keyboard.press("End");
  await page.keyboard.type(" It has \\textbf{bold}.");
  await page.keyboard.down("Control");
  await page.keyboard.press("Enter");
  await page.keyboard.up("Control");
  await page.waitForFunction(() => /bold/.test(document.querySelector(".art-canvas .doc-paper")!.shadowRoot!.querySelector(".doc-body")!.textContent!));
  expect([await page.$eval(".art-canvas .doc-paper", (p) => p.shadowRoot!.querySelector("strong")?.textContent), streams.length,
    await page.$eval(".art-canvas .art-label", (l) => l.textContent)]).toEqual(["bold", n0, "paper · v2 · your edit, not sent yet"]);
  await send(page, "fix the sign");
  expect(await page.$eval(".art-canvas .art-label", (l) => l.textContent)).toBe("paper · v3 · 1 change");
  await page.click(".art-canvas [data-art=changes]");
  expect(await page.$$eval(".art-canvas .art-changes :is(.diff-add, .diff-del)", (s) => s.map((x) => x.textContent)))
    .toEqual(["-   u_t = -\\alpha u_{xx}", "+   u_t = \\alpha u_{xx}"]);
  expect(await page.$eval(".art-canvas .art-changes .diff-count", (c) => c.textContent)).toBe("+1 −1");
  await page.close();
}, 40_000);

test("a Markdown artifact reads as a document whose blocks the tools move and delete by hand: Delete takes the list out, Undo brings it back, Move up puts the last paragraph above it", async () => {
  const page = await open();
  await send(page, "write the notes");
  await still(page);
  const blocks = () => page.$eval(".art-canvas .doc-paper", (p) => [...p.shadowRoot!.querySelectorAll(".md > [data-at]")].map((b) => b.textContent!.replace(/\s+/g, " ").trim()));
  expect(await blocks()).toEqual(["Plan", "First we read.", "milk eggs", expect.stringMatching(/^Then we cook x\s*2/)]);
  const pick = async (text: string) => {
    const at = await inDoc(page, text);
    await page.mouse.click(at.x, at.y);
  };
  await page.click(".art-canvas [data-art=edit]");
  await still(page);
  await pick("milk");
  await page.keyboard.press("Delete");
  await page.waitForFunction(() => document.querySelector(".art-canvas .doc-paper")!.shadowRoot!.querySelectorAll(".md > [data-at]").length === 3);
  await page.keyboard.down("Control");
  await page.keyboard.press("z");
  await page.keyboard.up("Control");
  await page.waitForFunction(() => document.querySelector(".art-canvas .doc-paper")!.shadowRoot!.querySelectorAll(".md > [data-at]").length === 4);
  await pick("Then we cook");
  await page.click(".art-canvas .art-tools [data-do=up]");
  expect((await blocks()).slice(2).map((b) => b.slice(0, 12))).toEqual(["Then we cook", "milk eggs"]);
  await send(page, "capital of France");
  expect(lastStream.messages.at(-1).content).toContain("# Plan\n\nFirst we read.\n\nThen we cook $x^2$.\n\n- milk\n- eggs");
  await page.close();
}, 30_000);

test("code is changed by hand where it is shown: Edit, type, Done; the change is the artifact's next version, and Changes shows it", async () => {
  const page = await open();
  await send(page, "write the script");
  await page.click(".art-canvas [data-art=edit]");
  await page.waitForFunction(() => document.activeElement?.matches(".art-canvas code"));
  await page.keyboard.down("Control");
  await page.keyboard.press("End");
  await page.keyboard.up("Control");
  await page.keyboard.type("\nprint(area(3))");
  await page.click(".art-canvas [data-art=edit]");
  await page.waitForFunction(() => /your edit/.test(document.querySelector(".art-canvas .art-label")?.textContent ?? ""));
  expect(await page.$eval(".art-canvas pre code", (c) => c.textContent)).toBe("def area(r):\n    return 3.14 * r * r\n\nprint(area(2))\nprint(area(3))");
  await page.click(".art-canvas [data-art=changes]");
  expect(await page.$$eval(".art-canvas .art-changes :is(.diff-add, .diff-del)", (s) => s.map((x) => x.textContent)))
    .toEqual(["+ print(area(3))"]);
  await page.close();
}, 30_000);

test("3D pages load three.js and A-Frame from their usual addresses, which they cannot reach: three as a module with addons by an import map's names and by address, and the classic build older pages load; each fills the panel", async () => {
  const page = await open();
  for (const [ask, id, want] of [["three cube", "cube", "186 true"], ["steady old cube", "old", "186 true"]]) {
    await send(page, ask);
    await page.waitForFunction((id) => document.querySelector("#shelf .art-canvas")?.getAttribute("data-art") === id, { timeout: 10_000 }, id);
    const frame = await pageFrame(page);
    await frame.waitForFunction(() => !!document.body?.dataset.three, { timeout: 10_000 });
    expect(await frame.$eval("body", (b) => b.dataset.three)).toBe(want);
    expect(await page.$eval("#shelf .art-canvas iframe.page-frame:not(.next)", (f) => f.getBoundingClientRect().bottom > innerHeight - 40 && (f as HTMLElement).offsetHeight > 500)).toBe(true);
  }
  await page.click(".shelf-tab:first-child");
  await page.waitForFunction(() => document.querySelector("#shelf .art-canvas")?.getAttribute("data-art") === "cube");
  const frame = await pageFrame(page);
  await frame.waitForFunction(() => !!document.body?.dataset.three, { timeout: 10_000 });
  const lit = await frame.$eval("canvas", (c) => {
    const g = (c as HTMLCanvasElement).getContext("webgl2")!, px = new Uint8Array(4);
    g.readPixels(150, 100, 1, 1, g.RGBA, g.UNSIGNED_BYTE, px);
    return px[0] + px[1] + px[2];
  });
  expect(lit).toBeGreaterThan(0);
  expect(probes).toBe(0);
  await page.close();
}, 40_000);

test("an A-Frame scene is pointed into: the entity under the pointer is the one its camera sees there; the paint bucket colors it and the model's edit by id moves it, in the scene as it runs", async () => {
  const page = await open();
  await send(page, "a scene");
  let frame = await pageFrame(page);
  await frame.waitForFunction(() => (document.querySelector("a-scene") as any)?.renderStarted, { timeout: 15_000 });
  await still(page);
  const started = await frame.evaluate(() => (window as any).started);
  const boxAt = async () => {
    const box = (await (await page.$(".art-canvas iframe.page-frame:not(.next)"))!.boundingBox())!;
    const p = await frame.evaluate(() => {
      const scene = document.querySelector("a-scene") as any, T = (window as any).AFRAME.THREE, c = scene.canvas.getBoundingClientRect();
      const v = scene.querySelector("#box").object3D.getWorldPosition(new T.Vector3()).project(scene.camera);
      return { x: c.left + ((v.x + 1) / 2) * c.width, y: c.top + ((1 - v.y) / 2) * c.height };
    });
    return { x: box.x + 1 + p.x, y: box.y + 1 + p.y };
  };
  await pageTool(page, "[data-tool=select]");
  let at = await boxAt();
  await page.mouse.move(at.x, at.y);
  await page.mouse.click(at.x, at.y);
  await page.waitForFunction(() => document.getElementById("pointing")!.textContent === "Pointing at scene · #box");
  await page.mouse.click(at.x, at.y - 120);
  await page.waitForFunction(() => document.getElementById("pointing")!.textContent === "Pointing at scene · #sky");
  await pickColor(page, "#ff8800");
  await frame.waitForFunction(() => (window as any).__chrysalis.mode === "bucket");
  at = await boxAt();
  await page.mouse.click(at.x, at.y);
  await frame.waitForFunction(() => (document.querySelector("#box") as any).getAttribute("material").color === "#ff8800");
  await pageTool(page, "[data-do=done]");
  await send(page, "move the box");
  frame = await pageFrame(page);
  await frame.waitForFunction(() => (document.querySelector("#box") as any).object3D.position.x === 1);
  expect([await frame.evaluate(() => (window as any).started), await frame.evaluate(() => (document.querySelector("#box") as any).getAttribute("material").color)])
    .toEqual([started, "#4CC3D9"]);
  expect(probes).toBe(0);
  await page.close();
}, 60_000);

test("an edit that does not apply in full is asked for again, once, with the artifact as it is now, and its answer applies", async () => {
  const page = await open();
  await send(page, "draw the panda");
  await send(page, "broken edit");
  await page.waitForFunction(() => document.querySelectorAll(".msg.assistant .stats").length >= 3, { timeout: 20_000 });
  await Bun.sleep(300);
  const asked = streams.at(-1).messages.at(-1).content as string;
  expect([asked.includes("its SEARCH text is not in it"), asked.includes('```svg panda\n<svg viewBox="0 0 100 100"')]).toEqual([true, true]);
  expect(await page.$$eval(".msg.user.fix", (m) => m.length)).toBe(1);
  expect(await page.$$eval(".art-fail", (f) => f.map((x) => x.textContent).filter(Boolean))).toEqual(["Not applied: its SEARCH text is not in it"]);
  expect(await page.$eval(".art-canvas[data-v='3'] .svg-paper", (p) => p.shadowRoot!.querySelector("#burger rect")?.getAttribute("fill"))).toBe("#0000ff");
  await Bun.sleep(500);
  expect(await page.$$eval(".msg", (m) => m.length)).toBe(6);
  await page.close();
}, 30_000);
