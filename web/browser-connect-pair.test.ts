import { expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import type { Page } from "puppeteer-core";
import { harness, RUN, chatContext, onNetwork, trusting } from "./browser-harness";

harness();

test("trusting the authority, Chromium and Firefox each show their own code on the connect page, which this computer's Connect a device allows, and the chat opens there with the line for other computers; a phone gets no Connect a device, and the chat stays in through a restart of Chrysalis", async () => {
  const lan = await onNetwork();
  try {
    const local = await (await chatContext()).newPage();
    await local.goto(lan.here, { waitUntil: "load" });
    await local.waitForSelector("#input", { timeout: 15_000 });
    await local.waitForSelector("#connect:not([hidden])");
    await local.click("#connect");
    await local.waitForSelector("#connect-view .dev-qr");
    expect(await local.$eval("#connect-view .dev-qr", (img) => (img as HTMLImageElement).naturalWidth > 0)).toBe(true);
    let chrome: Page | undefined;
    for (const [firefox, name] of [[false, "Linux · Chrome"], [true, "Linux · Firefox"]] as const) {
      const home = mkdtempSync(`${RUN}/trusts-`), profile = trusting(home, lan.tlsDir);
      const b = await lan.launch(firefox, firefox ? profile : home);
      const page = await b.newPage();
      await page.goto(lan.chat, { waitUntil: "load" });
      await page.waitForFunction(() => !(document.getElementById("code") as HTMLElement).hidden, { timeout: 10_000 });
      expect(await page.$eval("#setup", (e) => (e as HTMLElement).hidden)).toBe(true);
      const code = await page.$eval("#code", (e) => e.textContent!);
      expect(code).toMatch(/^\d{3} \d{3}$/);
      await local.waitForFunction((c) => [...document.querySelectorAll("#connect-view .dev-ask-code")].some((e) => e.textContent === c), { timeout: 5000 }, code);
      expect(await local.$$eval("#connect-view .dev-ask .dev-name", (n) => n.map((e) => e.textContent))).toEqual([name]);
      await local.evaluate((c) => [...document.querySelectorAll("#connect-view .dev-ask")]
        .find((e) => e.querySelector(".dev-ask-code")!.textContent === c)!.querySelector<HTMLButtonElement>(".btn.primary")!.click(), code);
      await page.waitForFunction((c) => location.href === c, { timeout: 15_000 }, lan.chat);
      await page.waitForSelector("#app:not([aria-busy])");
      expect([page.url(), await page.$eval("#connect", (e) => e.checkVisibility())]).toEqual([lan.chat, true]);
      await page.click("#connect");
      await page.waitForSelector("#connect-view .dev-url");
      const line = await page.$eval("#connect-view .dev-url", (e) => e.textContent!);
      expect(line).toMatch(/^curl -fsSk --pinnedpubkey sha256\/\/\S+ https:\/\/\S+\/opencode \| sh$/);
      expect(await page.evaluate(() => {
        let copied = "";
        const real = document.execCommand.bind(document);
        document.execCommand = (name: string) => { copied = getSelection()!.toString() || (document.activeElement as HTMLTextAreaElement).value; return real(name); };
        document.querySelector<HTMLButtonElement>("#connect-view .dev-list .btn")!.click();
        document.execCommand = real;
        return copied;
      })).toBe(line);
      expect(await page.$eval("#connect-view .dev-list .btn", (e) => e.textContent)).toBe("Copied");
      await page.click("#connect-close");
      const phone = await b.newPage();
      await phone.setUserAgent("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1");
      await phone.goto(lan.chat, { waitUntil: "load" });
      await phone.waitForSelector("#app:not([aria-busy])");
      expect(await phone.$eval("#connect", (e) => e.checkVisibility())).toBe(false);
      await phone.close();
      await local.waitForFunction((n) => (document.querySelector("#connect-view .dev-joined")?.textContent ?? "").includes(`${n} is connected`), { timeout: 5000 }, name);
      if (!firefox) chrome = page;
    }
    expect(await local.$$eval("#connect-view .dev-item .dev-name", (n) => n.map((e) => e.textContent)))
      .toEqual(["Linux · Chrome", "Linux · Firefox"]);

    await chrome!.evaluate(() => { (window as any).stayed = true; });
    await lan.restart();
    await chrome!.waitForResponse((r) => r.url().endsWith("/api/status") && r.status() === 200, { timeout: 10_000 });
    await chrome!.waitForSelector("#app:not([aria-busy])");
    expect(await chrome!.evaluate(async () => [(window as any).stayed, location.pathname, (await fetch("/api/status")).status])).toEqual([true, "/", 200]);
  } finally {
    await lan.stop();
  }
}, 60_000);
