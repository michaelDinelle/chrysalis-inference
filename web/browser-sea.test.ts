import { expect, test } from "bun:test";
import puppeteer from "puppeteer-core";
import { harness, CHAT_ARGS, base, chatContext, idle, look, looks, open, png } from "./browser-harness";

harness();

test("a new chat is the butterfly in a sea under the message box too, never blanked by a side panel sliding shut; a click on the water makes ripples and keeps the message box focused, and with reduced motion draws nothing", async () => {
  const gl = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: [...CHAT_ARGS, "--enable-unsafe-swiftshader"] });
  try {
    for (const motion of ["no-preference", "reduce"] as const) {
      const page = await (await chatContext(gl)).newPage();
      page.on("pageerror", (e) => { throw e; });
      await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: motion }]);
      await page.setViewport({ width: 1600, height: 900 });
      await idle(page, page.goto(`${base}/`));
      await page.waitForSelector(".sea > canvas.ocean");
      const rect = async (sel: string) => (await (await page.$(sel))!.boundingBox())!;
      let sea = await rect(".sea"), box = await rect(".composer-box");
      const middle = { x: box.x + box.width / 2 - 150, y: (sea.y + box.y) / 2 - 100, width: 300, height: 200 };
      let pink = 0;
      for (const end = Date.now() + 5000; pink <= 800 && Date.now() < end; await Bun.sleep(100)) pink = (await look(page, middle)).pink;
      expect(pink).toBeGreaterThan(800);

      const open = sea.x, left = () => page.$eval(".sea", (e) => e.getBoundingClientRect().left);
      if (motion !== "reduce") await page.addStyleTag({ content: ":root { --pane-ease: 2s linear !important; }" });
      await page.click("#conversations-toggle");
      if (motion !== "reduce") {
        for (let k = 0; k < 2; k++) {
          await Bun.sleep(400);
          const from = await left(), { lit } = await look(page, { x: open + 20, y: sea.y + 40, width: 200, height: 200 }), to = await left();
          expect([from > 0, to < open, to < from]).toEqual([true, true, true]);
          expect(lit).toBeGreaterThan(500);
        }
      }
      await page.waitForFunction(() => document.querySelector(".sea")!.getBoundingClientRect().left === 0);
      sea = await rect(".sea"); box = await rect(".composer-box");
      const beside = { x: sea.x + 40, y: box.y, width: box.x - sea.x - 80, height: box.height };
      expect((await look(page, beside)).lit).toBeGreaterThan(150);
      expect(await page.evaluate((x, y) => document.elementFromPoint(x, y)?.className, beside.x + 20, box.y + box.height / 2)).toBe("ocean");

      await page.focus("#input");
      const at = { x: sea.x + 90, y: (sea.y + box.y) / 2 }, near = { x: at.x - 70, y: at.y - 80, width: 140, height: 160 };
      const far = { ...near, x: sea.x + sea.width - 90 - 70 };
      await page.mouse.move(at.x, at.y);
      await Bun.sleep(motion === "reduce" ? 300 : 1500);
      const was = await looks(page, [near, far]);
      await page.mouse.down();
      await page.mouse.up();
      await Bun.sleep(400);
      const [near1, far1] = (await looks(page, [near, far], was.png)).parts;
      expect(await page.evaluate(() => document.activeElement?.id)).toBe("input");
      if (motion === "reduce") expect([near1.changed, far1.changed]).toEqual([0, 0]);
      else {
        expect(near1.changed).toBeGreaterThan(2 * far1.changed + 500);
      }
      await page.close();
    }
  } finally {
    await gl.close();
  }
}, 90_000);
