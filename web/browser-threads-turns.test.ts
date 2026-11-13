import { expect, test } from "bun:test";
import { cards, castRows, connect, dir, harness, home, lastText, open, openFolder, opencode, png, send, spokenOf, streams, voicePage, work, outside } from "./browser-harness";

harness();

test("a thread takes what a chat takes: attached text files and pictures go to OpenCode with the message and come back with the thread after a reload, a video does not; the thinking slider picks the model's own level and hides for a model with none; a question OpenCode asks is answered or skipped in its card", async () => {
  const page = await open(), last = () => opencode.log.prompts.at(-1)!;
  await outside(page, true);
  await Bun.write(`${dir}/jar.txt`, "clover honey, 12 jars");
  await Bun.write(`${dir}/jar.png`, png);
  await Bun.write(`${dir}/jar.mp4`, "not really a film");
  await connect(page);
  await openFolder(page, work);
  const file = (await page.$("input#attach-file"))!;
  await file.uploadFile(`${dir}/jar.txt`, `${dir}/jar.png`);
  await page.waitForFunction(() => document.querySelectorAll("#attachments .att").length === 2);
  await file.uploadFile(`${dir}/jar.mp4`);
  await page.waitForFunction(() => /OpenCode gets text files, PDFs and pictures/.test(document.querySelector("#toast span")?.textContent ?? ""));
  await page.click('#energy-stops [data-stop="0"]');
  await send(page, "read these");
  const [text, picture] = last().files!;
  expect([last().files!.length, text.name, Buffer.from(text.uri.slice(text.uri.indexOf(",") + 1), "base64").toString(), picture.name, picture.uri.startsWith("data:image/png;base64,"), last().model])
    .toEqual([2, "jar.txt", "clover honey, 12 jars", "jar.png", true, { providerID: "chrysalis", id: "qwen3.8-27b", variant: "none" }]);
  await page.click('#energy-stops [data-stop="2"]');
  await send(page, "and now");
  expect([last().files, last().model]).toEqual([undefined, { providerID: "chrysalis", id: "qwen3.8-27b", variant: "medium" }]);

  await page.reload({ waitUntil: "load" });
  await page.waitForFunction(() => document.querySelectorAll("#thread > article").length === 4);
  await page.waitForFunction(() => (document.querySelector<HTMLImageElement>("#thread > article.user .msg-files img")?.naturalWidth ?? 0) > 0);
  expect(await page.$eval("#thread > article.user .msg-files", (b) => [b.querySelector(".att-name")!.textContent, b.querySelector("img")!.title])).toEqual(["jar.txt", "jar.png"]);

  const asked = async (pick?: string) => {
    await page.type("#input", "which one should I read");
    await page.keyboard.press("Enter");
    await page.waitForSelector(".tool-ask:not([hidden]) .ask-form");
    const seen = [await page.$eval(".ask-form .ask-title", (e) => e.textContent), await page.$$eval(".ask-form .ask-option", (o) => o.map((x) => x.textContent)),
      await page.$$eval("#conversations .conv-status", (s) => s.map((x) => x.textContent))];
    if (pick) await page.evaluate((pick) => [...document.querySelectorAll<HTMLElement>(".ask-form .ask-option")].find((o) => o.textContent === pick)!.click(), pick);
    await page.click(pick ? '.ask-form button[type="submit"]' : '.ask-form [data-decision="deny"]');
    await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send");
    return seen;
  };
  expect(await asked("Heather")).toEqual(["Which jar", ["Clover", "Heather"], ["Input"]]);
  expect([opencode.log.answers.at(-1), (await lastText(page)).includes("You chose Heather.")]).toEqual([{ q0: "Heather" }, true]);
  await asked();
  expect([opencode.log.answers.at(-1), (await lastText(page)).includes("You did not choose.")]).toEqual(["skipped", true]);

  const slider = () => page.$eval(".think-ctl", (e) => getComputedStyle(e).display !== "none");
  expect(await slider()).toBe(true);
  await page.click("#model-btn");
  await page.waitForSelector("#model-pop:not([hidden]) .model-group");
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#model-pop .model-item")].find((i) => i.textContent!.includes("Cloud One"))!.click());
  expect(await slider()).toBe(false);
  await send(page, "plainly");
  expect(last().model).toEqual({ providerID: "elsewhere", id: "cloud-1" });
  await page.evaluate(() => localStorage.removeItem("thread-model"));
  await outside(page, false);
  await page.close();
}, 90_000);

test("a thread is OpenCode's to change: a message sent while a turn runs goes to OpenCode and shows once the turn ends, Fork opens a new chat holding what came before a message, a reply edited in place and a message deleted are changed in OpenCode, an edited message sent again and Regenerate take the chat back and answer anew, and settling archives the thread in OpenCode so another tab shelves it too", async () => {
  const page = await open(), users = () => page.$$eval("#thread > article.user .bubble", (b) => b.map((x) => x.textContent!.trim()));
  const act = (text: string, name: string) => page.evaluate((text, name) => [...document.querySelectorAll("#thread > article.user")].find((a) => a.querySelector(".bubble")!.textContent!.trim() === text)!
    .querySelector<HTMLElement>(`[data-act="${name}"]`)!.click(), text, name);
  await connect(page);
  await openFolder(page, work);
  await send(page, "sort the first shelf");
  const session = opencode.log.prompts.at(-1)!.session, said = () => opencode.sessions.get(session)!.messages.filter((m) => m.type === "user").map((m) => m.text);
  await page.type("#input", "sort the second shelf slowly");
  await page.keyboard.press("Enter");
  await page.waitForSelector('#thread .tool-card[data-state="running"]');
  await page.type("#input", "and dust it");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /Sent: OpenCode reads it/.test(document.querySelector("#toast span")?.textContent ?? ""));
  expect([said(), await page.$eval("#send", (e) => e.getAttribute("aria-label")), await page.$eval("#input", (e) => (e as HTMLTextAreaElement).value)])
    .toEqual([["sort the first shelf", "sort the second shelf slowly", "and dust it"], "Stop", ""]);
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send" && document.querySelectorAll("#thread > article.user").length === 3, { timeout: 20_000 });
  expect(await users()).toEqual(["sort the first shelf", "sort the second shelf slowly", "and dust it"]);

  const before = opencode.sessions.size;
  await act("sort the second shelf slowly", "fork");
  await page.waitForFunction(() => document.querySelectorAll("#thread > article.user").length === 1 && /\(fork\)$/.test(document.querySelector("#title")?.textContent ?? ""));
  const forked = [...opencode.sessions.values()].at(-1)!;
  expect([opencode.sessions.size - before, await users(), forked.messages.filter((m) => m.type === "user").map((m) => m.text), said().length]).toEqual([1, ["sort the first shelf"], ["sort the first shelf"], 3]);

  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#conversations a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "sort the first shelf")!.click());
  await page.waitForFunction(() => document.querySelectorAll("#thread > article.user").length === 3);
  const replies = () => opencode.sessions.get(session)!.messages.filter((m) => m.type === "assistant").flatMap((m) => m.content.filter((p: { type: string }) => p.type === "text").map((p: { text: string }) => p.text));
  const editor = async (article: string, text: string, button: string) => {
    await page.evaluate((article) => document.querySelector(article)!.querySelector<HTMLElement>('[data-act="edit"]')!.click(), article);
    await page.waitForSelector("#thread .editor textarea");
    await page.$eval("#thread .editor textarea:last-of-type", (t, text) => { (t as HTMLTextAreaElement).value = text; t.dispatchEvent(new Event("input", { bubbles: true })); }, text);
    await page.evaluate((button) => [...document.querySelectorAll<HTMLElement>("#thread .editor-acts button")].find((b) => b.textContent === button)!.click(), button);
  };
  await editor("#thread > article.assistant", "The first shelf is sorted.", "Save");
  await page.waitForFunction(() => document.querySelector<HTMLElement>("#thread > article.assistant > .markdown")?.innerText.trim() === "The first shelf is sorted.");
  expect([replies()[0], said().length, await page.$$eval("#thread > article", (a) => a.length)]).toEqual(["The first shelf is sorted.", 3, 5]);

  await act("and dust it", "trash");
  await page.waitForFunction(() => document.querySelectorAll("#thread > article.user").length === 2);
  expect([said(), await users()]).toEqual([["sort the first shelf", "sort the second shelf slowly"], ["sort the first shelf", "sort the second shelf slowly"]]);

  const asked = opencode.log.prompts.length;
  await editor("#thread > article.user:nth-of-type(3)", "sort the third shelf", "Send");
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send" && /You asked: sort the third shelf/.test([...document.querySelectorAll<HTMLElement>("#thread > article.assistant")].at(-1)?.innerText ?? ""), { timeout: 20_000 });
  expect([said(), opencode.log.prompts.length - asked, await page.$$eval("#thread > article", (a) => a.length), replies()[0]]).toEqual([["sort the first shelf", "sort the third shelf"], 1, 4, "The first shelf is sorted."]);

  await page.evaluate(() => [...document.querySelectorAll("#thread > article.assistant")].at(-1)!.querySelector<HTMLElement>('[data-act="retry"]')!.click());
  for (let k = 0; k < 200 && opencode.log.prompts.length - asked < 2; k++) await Bun.sleep(50);
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send" && document.querySelectorAll("#thread > article").length === 4, { timeout: 20_000 });
  expect([said(), opencode.log.prompts.length - asked, opencode.log.prompts.at(-1)!.text]).toEqual([["sort the first shelf", "sort the third shelf"], 2, "sort the third shelf"]);

  await page.evaluate(() => [...document.querySelectorAll("#conversations .folder:not(.settled) a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "sort the first shelf")!.querySelector<HTMLElement>(".conv-settle")!.click());
  await page.waitForFunction(() => !!document.querySelector("#conversations .settled-head"));
  expect(!!opencode.sessions.get(session)!.archived).toBe(true);
  const other = await open();
  await other.waitForSelector("#conversations .settled-head");
  await other.click("#conversations .settled-head");
  expect(await other.$$eval("#conversations .folder.settled .conv-title", (t) => t.map((x) => x.textContent))).toContain("sort the first shelf");
  await other.evaluate(() => [...document.querySelectorAll("#conversations .folder.settled a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "sort the first shelf")!.querySelector<HTMLElement>(".conv-settle")!.click());
  await other.waitForFunction(() => ![...document.querySelectorAll("#conversations .folder.settled .conv-title")].some((t) => t.textContent === "sort the first shelf"));
  expect(!!opencode.sessions.get(session)!.archived).toBe(false);
  for (const p of [page, other]) await p.close();
}, 90_000);

test("while OpenCode is connected, a chat with Chrysalis is OpenCode at work in the home folder: a new chat's message goes to OpenCode there and its steps are cards, the chat is listed under Chrysalis and not as a folder, voice mode asks OpenCode and speaks its reply with the model picker and thinking slider still in a narrow message box, and once disconnected a new chat is a plain one again", async () => {
  const page = await open(), last = () => opencode.log.prompts.at(-1)!, chats = streams.length;
  await page.waitForSelector("#opencode .dot.online");
  await page.click("#new-chat");
  await send(page, "how much space is left on the computer");
  expect([last().text, opencode.sessions.get(last().session)!.dir, streams.length - chats, await cards(page)]).toEqual(["how much space is left on the computer", home, 0, [["Read", "OpenCode", "done"]]]);
  expect([(await castRows(page)).includes(home.split("/").pop()!), await page.$$eval("#conversations .folder.open .conv-title", (t) => t.map((x) => x.textContent))]).toEqual([false, ["how much space is left on the computer"]]);
  await page.close();

  const spoke = opencode.log.prompts.length, made = opencode.sessions.size, { page: talk, session } = await voicePage("pie");
  await talk.waitForFunction(() => /^voice /.test(document.querySelector('#thread > article[data-i="1"] .stats')?.textContent ?? ""), { timeout: 20_000 });
  const heard = opencode.log.prompts.slice(spoke);
  expect([heard.map((p) => p.text.split("\n\n").at(-1)), heard[0].text.startsWith("[The user is talking to you by voice"), heard[0].meta, opencode.sessions.size - made, opencode.sessions.get(last().session)!.dir,
    streams.slice(chats).filter((b) => b.stream).length]).toEqual([["lets do a snake game in Pygame"], true, { said: "lets do a snake game in Pygame" }, 1, home, 0]);
  for (const width of [760, 390]) {
    await talk.setViewport({ width, height: 800 });
    await Bun.sleep(500);
    expect(await talk.evaluate(() => [...["#model-btn", ".think-ctl", "#voice-pill"].map((q) => { const r = document.querySelector(q)!.getBoundingClientRect(); return r.width > 0 && r.left >= 0 && r.right <= innerWidth; }),
      document.querySelector(".composer-box")!.scrollWidth <= document.querySelector(".composer-box")!.clientWidth])).toEqual([true, true, true, true]);
  }
  await talk.reload({ waitUntil: "load" });
  await talk.waitForFunction(() => document.querySelectorAll("#thread > article").length === 2);
  expect(await talk.$eval("#thread > article.user .bubble", (b) => b.textContent)).toBe("lets do a snake game in Pygame");
  const { said, cancelled } = spokenOf(session());
  expect([...said].filter(([id]) => !cancelled.has(id)).map(([, t]) => t).join(" ")).toContain("You asked:");
  await talk.close();

  const plain = await open();
  await plain.click("#opencode");
  await plain.waitForFunction(() => [...document.querySelectorAll("#oc-pop .invite-card")].some((b) => b.textContent === "Disconnect"));
  await plain.evaluate(() => [...document.querySelectorAll<HTMLElement>("#oc-pop .invite-card")].find((b) => b.textContent === "Disconnect")!.click());
  await plain.waitForFunction(() => !document.querySelector("#opencode .dot.online"));
  const asked = opencode.log.prompts.length;
  await send(plain, "good evening");
  expect([opencode.log.prompts.length - asked, streams.length - chats > 0, await plain.$eval("main", (m) => m.hasAttribute("data-thread"))]).toEqual([0, true, false]);
  await plain.close();
}, 90_000);
