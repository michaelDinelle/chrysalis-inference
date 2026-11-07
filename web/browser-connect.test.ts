import { expect, test } from "bun:test";
import type { Page } from "puppeteer-core";
import { harness, RUN, onNetwork, pastWarning, setUpShown, stageDrawn } from "./browser-harness";
import { mkdtempSync } from "node:fs";

harness();

test("past the chat's warning on the network, a device gets the connect page and none of the chat: it offers each kind of device the way its system trusts the authority, a browser that does not trust it stays there, one that cannot reach the trust check is told how to let it through, and nothing reaches the server", async () => {
  const lan = await onNetwork();
  try {
    const plain = await lan.launch(false, mkdtempSync(`${RUN}/untrusting-`));
    const UA = {
      iphone: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
      android: "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36",
      firefox: "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
      mac: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
      windows: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
    };
    const offered: Record<string, string | null> = {};
    const rechecked = (page: Page) => new Promise<void>((done) => {
      let seen = 0;
      const ended = (r: { url(): string }) => { if (r.url().includes("/check.gif") && ++seen === 1) done(); };
      page.on("requestfailed", ended);
      page.on("requestfinished", ended);
    });
    for (const [kind, ua] of Object.entries(UA)) {
      const page = await (await plain.createBrowserContext()).newPage();
      await page.setUserAgent(ua);
      await page.goto(lan.chat).catch(() => {});
      await pastWarning(page, false);
      await setUpShown(page);
      offered[kind] = await page.$eval(".go", (a) => a.getAttribute("href"));
      while (!(await page.$eval("#on", (e) => (e as HTMLElement).hidden))) await page.click("#on");
      expect(await page.$eval("#count", (e) => e.textContent)).toMatch(/ · Trust$/);
      if (kind === "android") {
        await rechecked(page);
        await page.evaluate(() => new Promise((done) => setTimeout(done)));
        expect([page.url(), await page.$eval("#setup", (e) => (e as HTMLElement).hidden)]).toEqual([lan.chat, false]);
        expect([await stageDrawn(page), await page.$eval("#stage", (e) => e.childElementCount)]).toEqual([true, 0]);
      }
      await page.close();
    }
    expect(offered).toEqual({ iphone: "/chrysalis.mobileconfig", android: "/chrysalis-ca.crt", firefox: "/chrysalis-ca-firefox.crt",
      mac: "/chrysalis.mobileconfig", windows: "/chrysalis-ca.crt" });

    const page = await (await plain.createBrowserContext()).newPage();
    await page.goto(lan.chat).catch(() => {});
    await pastWarning(page, false);
    await setUpShown(page);
    await page.setRequestInterception(true);
    page.on("request", (r) => { if (!r.url().includes("/check.gif")) void r.continue(); });
    await page.reload({ waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => /Cannot reach/.test(document.getElementById("status")!.textContent!), { timeout: 10_000 });
    expect([await page.$eval("#status", (e) => (e as HTMLElement).hidden), await page.$eval("#sub", (e) => e.textContent)])
      .toEqual([false, expect.stringContaining("chrysalis.sh --host 0.0.0.0")]);
    await page.close();
    expect(await lan.waiting()).toEqual([]);
  } finally {
    await lan.stop();
  }
}, 60_000);
