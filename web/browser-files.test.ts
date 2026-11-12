import { expect, test } from "bun:test";
import { writeFileSync } from "node:fs";
import type { Browser, Page } from "puppeteer-core";
import { ARTIFACTS_PROMPT } from "./artifacts";
import { harness, base, card, dir, fake, idle, lastStream, look, open, png, send, sent, streams, web } from "./browser-harness";

harness();

test("attached files go to Chrysalis as images, PDFs, videos and fenced text, and stay after a reload", async () => {
  const pdf = Buffer.from("%PDF-1.4\n%chrysalis\n"), mp4 = Buffer.from([0, 0, 0, 24, 102, 116, 121, 112, 109, 112, 52, 50]);
  writeFileSync(`${dir}/dot.png`, png);
  writeFileSync(`${dir}/fence.md`, "Has ``` inside.\n");
  writeFileSync(`${dir}/blob.bin`, new Uint8Array([0, 159, 146, 150, 255]));
  writeFileSync(`${dir}/doc.pdf`, pdf);
  writeFileSync(`${dir}/clip.mp4`, mp4);
  const page = await open();
  const cdp = await page.createCDPSession();
  await cdp.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: dir, browserContextId: page.browserContext().id });
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/dot.png`, `${dir}/fence.md`, `${dir}/blob.bin`, `${dir}/doc.pdf`, `${dir}/clip.mp4`);
  await page.waitForFunction(() => document.querySelectorAll("#attachments .att").length === 4);
  expect(await page.$eval("#toast", (t) => t.textContent)).toContain("blob.bin");
  await send(page, "look");
  const content = lastStream.messages.at(-1).content;
  const bytes = (url: string) => Buffer.from(url.split(",")[1], "base64");
  expect(content.map((p: any) => p.type)).toEqual(["image_url", "text", "file", "video_url", "text"]);
  expect(bytes(content[0].image_url.url).equals(png)).toBe(true);
  expect(content[1].text).toBe("fence.md\n````md\nHas ``` inside.\n````\n\n");
  expect(content[2].file.filename).toBe("doc.pdf");
  expect(bytes(content[2].file.file_data).equals(pdf)).toBe(true);
  expect(bytes(content[3].video_url.url).equals(mp4)).toBe(true);
  expect(content[4].text).toBe("look");
  await idle(page, page.reload());
  await page.waitForFunction(() => (document.querySelector(".msg-files img") as HTMLImageElement | null)?.naturalWidth === 2);
  await send(page, "again");
  expect(lastStream.messages.find((m: any) => m.role === "user").content).toEqual(content);
  await page.close();
}, 60_000);

