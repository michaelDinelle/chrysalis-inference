import { expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { harness, RUN, onNetwork, pastWarning, setUpShown, stageDrawn } from "./browser-harness";

harness();

test("a browser that does not trust the authority gets the connect page with its own way to trust it, Chromium downloads it and is shown where its certificates are kept, Firefox gets its own file, and nothing reaches the server", async () => {
  const lan = await onNetwork();
  try {
    for (const firefox of [false, true]) {
      const home = firefox ? "" : mkdtempSync(`${RUN}/untrusting-`);
      const b = await lan.launch(firefox, home, ["--disable-3d-apis"]);
      const page = await b.newPage();
      await page.goto(lan.chat).catch(() => {});
      await pastWarning(page, firefox);
      await setUpShown(page);
      expect([page.url(), await page.$eval(".go", (a) => a.getAttribute("href")), await page.$("#input")])
        .toEqual([lan.chat, firefox ? "/chrysalis-ca-firefox.crt" : "/chrysalis-ca.crt", null]);
      if (!firefox) {
        expect([await stageDrawn(page), await page.$eval("#stage", (e) => e.childElementCount)]).toEqual([true, 0]);
        await (await b.target().createCDPSession()).send("Browser.setDownloadBehavior", { behavior: "default" });
        await page.click(".go");
        const kept = `${home}/Downloads/chrysalis-ca.crt`;
        for (let t = Date.now(); !existsSync(kept) && Date.now() - t < 5000; ) await Bun.sleep(100);
        expect(existsSync(kept) && readFileSync(kept, "utf8")).toBe(readFileSync(`${lan.tlsDir}/ca.pem`, "utf8"));
        await page.waitForFunction(() => document.getElementById("go")!.hidden, { timeout: 3000 });
        expect(await page.$eval("#now", (e) => [e.querySelector(".copyable code")?.textContent, !!e.querySelector(".copy")]))
          .toEqual(["chrome://certificate-manager/localcerts/usercerts", true]);
      }
      expect(await lan.waiting()).toEqual([]);
    }
  } finally {
    await lan.stop();
  }
}, 60_000);
