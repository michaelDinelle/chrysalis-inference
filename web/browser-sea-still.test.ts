import { expect, test } from "bun:test";
import puppeteer from "puppeteer-core";
import { harness, CHAT_ARGS, base, browser, chatContext, checked, closeSettings, idle, look, openSettings, png, send, fresh } from "./browser-harness";

harness();

test("the sea drawn by the CPU, in a browser without WebGL, is the sea WebGL2 draws: the same water, the same butterfly, in a window and on a phone", async () => {
  const shots: Record<string, Awaited<ReturnType<typeof look>>> = {};
  for (const [name, flag] of [["gpu", "--enable-unsafe-swiftshader"], ["cpu", "--disable-3d-apis"]] as const) {
    const b = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: [...CHAT_ARGS, flag] });
    try {
      for (const [size, width, height] of [["window", 1280, 800], ["phone", 390, 844]] as const) {
        const page = await (await chatContext(b)).newPage();
        page.on("pageerror", (e) => { throw e; });
        await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "reduce" }]);
        await page.setViewport({ width, height });
        await page.goto(`${base}/`, { waitUntil: "load" });
        expect(await page.evaluate(() => !!document.createElement("canvas").getContext("webgl2"))).toBe(name === "gpu");
        const sea = await page.$eval("#sea", (e) => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
        for (let last = await look(page, sea), end = Date.now() + 10_000; Date.now() < end; ) {
          await Bun.sleep(100);
          const now = await look(page, sea, last.png);
          if (now.pink > (size === "window" ? 10_000 : 3_000) && now.changed === 0) break;
          last = now;
        }
        shots[`${name} ${size}`] = await look(page, sea, shots[`gpu ${size}`]?.png ?? "");
        await page.close();
      }
    } finally {
      await b.close();
    }
  }
  for (const size of ["window", "phone"]) {
    const gpu = shots[`gpu ${size}`], cpu = shots[`cpu ${size}`];
    expect(gpu.lit).toBeGreaterThan(size === "window" ? 20_000 : 5_000);
    expect(gpu.pink).toBeGreaterThan(size === "window" ? 10_000 : 3_000);
    expect(cpu.changed / cpu.lit).toBeLessThan(0.01);
    expect(Math.abs(cpu.pink - gpu.pink) / gpu.pink).toBeLessThan(0.02);
  }
}, 90_000);

test("with reduced motion the butterfly is gone from behind every chat with messages, one written in and one opened from the list, and a new chat raises it again", async () => {
  const gl = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: [...CHAT_ARGS, "--enable-unsafe-swiftshader"] });
  try {
    const page = await (await chatContext(gl)).newPage();
    page.on("pageerror", (e) => { throw e; });
    await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "reduce" }]);
    await page.setViewport({ width: 1920, height: 900 });
    await idle(page, page.goto(`${base}/`));
    await page.waitForSelector(".sea > canvas.ocean");
    const sea = (await (await page.$(".sea"))!.boundingBox())!, box = (await (await page.$(".composer-box"))!.boundingBox())!;
    const middle = { x: box.x + box.width / 2 - 150, y: (sea.y + box.y) / 2 - 100, width: 300, height: 200 };
    const settled = async (ok: (pink: number) => boolean) => {
      let pink = (await look(page, middle)).pink;
      for (const end = Date.now() + 5000; !ok(pink) && Date.now() < end; await Bun.sleep(100)) pink = (await look(page, middle)).pink;
      return pink;
    };
    expect(await settled((p) => p > 800)).toBeGreaterThan(800);
    await send(page, "capital of France");
    expect(await settled((p) => p < 50)).toBeLessThan(50);
    await fresh(page);
    expect(await settled((p) => p > 800)).toBeGreaterThan(800);
    await page.click(".conv");
    await page.waitForSelector("#thread .msg.assistant");
    expect(await settled((p) => p < 50)).toBeLessThan(50);
    await page.close();
  } finally {
    await gl.close();
  }
}, 60_000);

test("the sea stays behind a chat, the butterfly sunk into it and every click the chat's, and a new chat raises it again; Settings takes the sea away (remembered), from chats and from a new chat, which is then empty, and brings it back", async () => {
  const gl = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: [...CHAT_ARGS, "--enable-unsafe-swiftshader"] });
  try {
    const page = await (await chatContext(gl)).newPage();
    page.on("pageerror", (e) => { throw e; });
    await page.evaluateOnNewDocument(() => {
      const real = performance.now.bind(performance);
      let at = 0, from = 0, rate = 1;
      (window as any).slowMotion = (r: number) => { at = performance.now(); from = real(); rate = r; };
      performance.now = () => at + (real() - from) * rate;
    });
    await page.setViewport({ width: 1920, height: 900 });
    await idle(page, page.goto(`${base}/`));
    await page.waitForSelector(".sea > canvas.ocean");
    const rect = async (sel: string) => (await (await page.$(sel))!.boundingBox())!;
    const sea = await rect(".sea"), box = await rect(".composer-box");
    const middle = { x: box.x + box.width / 2 - 150, y: (sea.y + box.y) / 2 - 100, width: 300, height: 200 };
    const risen = async (least = 800) => {
      let pink = 0;
      for (const end = Date.now() + 5000; pink <= least && Date.now() < end; await Bun.sleep(100)) pink = (await look(page, middle)).pink;
      return pink;
    };
    const full = await risen();
    expect(full).toBeGreaterThan(800);

    await send(page, "capital of France");
    await Bun.sleep(1500);
    const column = await rect("#thread"), margin = { x: sea.x + 40, y: sea.y + 60, width: column.x - sea.x - 80, height: 300 };
    const sunk = await look(page, middle);
    expect(sunk.pink).toBeLessThan(50);
    expect(sunk.tint).toBeGreaterThan(300);
    expect((await look(page, margin)).lit).toBeGreaterThan(150);
    const hit = (x: number, y: number) => page.evaluate((x, y) => document.elementFromPoint(x, y)?.closest("#scroller, .sea")?.id, x, y);
    const reply = await rect(".msg.assistant .markdown");
    expect([await hit(reply.x + 5, reply.y + reply.height / 2), await hit(margin.x + 20, margin.y + 20)]).toEqual(["scroller", "scroller"]);

    await page.evaluate(() => (window as any).slowMotion(0.25));
    await fresh(page);
    const rising = (await look(page, middle)).pink;
    expect([rising < 0.7 * full, await risen(0.9 * full) > 0.9 * full]).toEqual([true, true]);
    await page.evaluate(() => (window as any).slowMotion(1));

    await page.click(".conv");
    await openSettings(page);
    await page.click("#sea-switch");
    await closeSettings(page);
    await page.waitForFunction((x) => document.getElementById("thread")!.getBoundingClientRect().x === x, {}, column.x);
    expect((await look(page, margin)).lit).toBe(0);
    await fresh(page);
    const plainly = () => page.evaluate(() => [document.querySelector("#thread .empty")!.childElementCount === 0, getComputedStyle(document.querySelector(".sea")!).display]);
    expect(await plainly()).toEqual([true, "none"]);
    expect((await look(page, margin)).lit).toBe(0);
    await idle(page, page.reload());
    expect([await page.$eval("#sea-switch", (b) => b.getAttribute("aria-checked")), await plainly()]).toEqual(["false", [true, "none"]]);
    await openSettings(page);
    await page.click("#sea-switch");
    await closeSettings(page);
    expect((await plainly())[1]).toBe("block");
    expect(await risen()).toBeGreaterThan(800);
    await page.close();
  } finally {
    await gl.close();
  }
}, 60_000);
