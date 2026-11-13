import { expect, test } from "bun:test";
import { mkdirSync } from "node:fs";
import { until } from "./server-harness";
import { card, cards, castRows, connect, dir, harness, home, inRow, menu, open, openFolder, opencode, send, streams, work, outside, fresh } from "./browser-harness";

harness();

test("OpenCode connects from its button, a folder opened there is a row, and a thread is a chat OpenCode answers: it gets the message and Chrysalis's own model (never one elsewhere unasked), its steps are cards, what it asks waits for an answer that goes back to it, and the thread comes back from OpenCode after a reload, and another tab shows its folder and thread in the sidebar without being asked", async () => {
  const page = await open(), prompts = opencode.log.prompts.length, chats = streams.length;
  await connect(page);
  expect(await page.$eval("#opencode .dot", (d) => d.classList.contains("online"))).toBe(true);
  await openFolder(page, work);
  expect([await castRows(page), await page.$eval("#input", (e) => (e as HTMLTextAreaElement).placeholder), await page.$eval("#model-btn .model-name", (e) => e.textContent)])
    .toEqual([["Chrysalis", "site"], "Message Chrysalis", "27B"]);
  await send(page, "what is in the note");
  const sent = opencode.log.prompts.at(-1)!;
  expect([opencode.log.prompts.length - prompts, sent.text, sent.model, sent.persona, streams.length - chats])
    .toEqual([1, "what is in the note", { providerID: "chrysalis", id: "qwen3.8-27b" }, undefined, 0]);
  expect([await cards(page), await page.$eval("#title", (e) => e.textContent)]).toEqual([[["Read", "OpenCode", "done"]], "what is in the note"]);
  expect(await page.$eval("#thread > article.assistant:last-child", (e) => (e as HTMLElement).innerText)).toContain("The file says hello and honey. You asked: what is in the note");

  await page.type("#input", "ask first, then read it");
  await page.keyboard.press("Enter");
  await page.waitForSelector(".tool-ask:not([hidden])");
  expect([await page.$$eval("#conversations .conv-status", (s) => s.map((x) => x.textContent)), opencode.log.replies.length]).toEqual([["Approval"], 0]);
  await page.click('.tool-ask [data-decision="deny"]');
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send");
  expect([opencode.log.replies.at(-1), (await cards(page)).at(-1)]).toEqual(["reject", ["Read", "OpenCode", "declined"]]);
  expect(await page.$eval("#thread > article.assistant:last-child", (e) => (e as HTMLElement).innerText)).toContain("I did not read it.");

  await page.reload({ waitUntil: "load" });
  await page.waitForFunction(() => document.querySelectorAll("#thread > article").length === 4);
  expect((await cards(page)).map((c) => c[2])).toEqual(["done", "failed"]);
  const other = await open();
  await other.waitForSelector("#conversations .cast-row .face.dir");
  expect([await castRows(other), await other.$eval("#opencode .dot", (d) => d.classList.contains("online"))]).toEqual([["Chrysalis", "site"], true]);
  await inRow(other, "site", ".cast-main");
  await other.waitForSelector("#conversations .folder-chats a.conv");
  expect(await other.$$eval("#conversations .folder-chats .conv-title", (t) => t.map((x) => x.textContent))).toContain("what is in the note");
  await other.evaluate(() => [...document.querySelectorAll<HTMLElement>("#conversations .folder-chats a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "what is in the note")!.click());
  await other.waitForFunction(() => document.querySelectorAll("#thread > article").length === 4);
  expect(await other.$eval("#thread > article.assistant:last-child", (e) => (e as HTMLElement).innerText)).toContain("I did not read it.");
  const had = opencode.sessions.size;
  await send(other, "and again");
  expect([opencode.log.prompts.at(-1)!.session, opencode.sessions.size - had]).toEqual([sent.session, 0]);
  for (const p of [page, other]) await p.close();
}, 90_000);

test("the folder button's explorer goes by its path's crumbs, a filter (Enter goes into the first match), Backspace for up, a typed path and its recent folders, a folder is used with one click on its Use, and Close folder takes it and its threads out of the sidebar until Undo or until it is opened again", async () => {
  const patch = `${dir}/patch`;
  mkdirSync(patch, { recursive: true });
  const page = await open();
  await connect(page);
  await page.keyboard.press("Escape");
  await fresh(page);
  await page.waitForSelector("#folder-btn:not([hidden])");
  await page.click("#folder-btn");
  await page.waitForSelector("#oc-pop .oc-path");
  await page.$eval("#oc-pop .oc-path", (e, v) => { (e as HTMLInputElement).value = v; (e as HTMLInputElement).focus(); }, dir);
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => [...document.querySelectorAll("#oc-pop .oc-dir .invite-name")].some((x) => x.textContent === "patch"));
  const here = () => page.$eval("#oc-pop .oc-use", (b) => b.getAttribute("title"));
  const moved = async (want: string) => page.waitForFunction((want) => document.querySelector("#oc-pop .oc-use")?.getAttribute("title") === want, {}, want);
  await page.focus("#oc-pop .oc-path");
  await page.keyboard.type("pat");
  expect(await page.$$eval("#oc-pop .oc-dir .invite-name", (r) => r.map((x) => x.textContent))).toEqual(["patch"]);
  await page.keyboard.press("Enter");
  await moved(patch);
  await page.focus("#oc-pop .oc-path");
  await page.keyboard.press("Backspace");
  await moved(dir);
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#oc-pop .oc-crumb")].at(-2)!.click());
  await moved(dir.replace(/\/[^/]+$/, ""));
  expect(await here()).toBe(dir.replace(/\/[^/]+$/, ""));
  await page.$eval("#oc-pop .oc-path", (e, v) => { (e as HTMLInputElement).value = v; (e as HTMLInputElement).focus(); }, dir);
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => [...document.querySelectorAll("#oc-pop .oc-dir .invite-name")].some((x) => x.textContent === "patch"));
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#oc-pop .oc-dir")].find((r) => r.querySelector(".invite-name")!.textContent === "patch")!.querySelector<HTMLElement>(".oc-open")!.click());
  await page.waitForFunction(() => document.querySelector("#folder-btn")?.textContent === "patch");
  await send(page, "what is in the note");
  const sidebar = () => page.$$eval("#conversations .cast-row .cast-name, #conversations .conv-title", (r) => r.map((x) => x.textContent));
  expect([await page.$eval("#oc-pop", (e) => (e as HTMLElement).hidden), opencode.sessions.get(opencode.log.prompts.at(-1)!.session)!.dir, (await sidebar()).includes("patch")]).toEqual([true, patch, true]);
  const close = async () => {
    await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#conversations .cast-row")].find((r) => r.querySelector(".cast-name")!.textContent === "patch")!
      .dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true, clientX: 60, clientY: 200 })));
    await page.waitForSelector("#row-menu:not([hidden])");
    await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#row-menu button")].find((b) => b.textContent === "Close folder")!.click());
    await page.waitForFunction(() => ![...document.querySelectorAll("#conversations .cast-name")].some((x) => x.textContent === "patch"));
  };
  await close();
  expect([(await sidebar()).includes("what is in the note"), await page.$eval("#toast", (t) => t.textContent)]).toEqual([false, "patch is closedUndo"]);
  await page.click("#toast button");
  await page.waitForFunction(() => [...document.querySelectorAll("#conversations .cast-name")].some((x) => x.textContent === "patch"));
  await close();
  await page.reload();
  await page.waitForSelector("#conversations .cast-row");
  expect((await sidebar()).includes("patch")).toBe(false);
  await fresh(page);
  await page.waitForSelector("#folder-btn:not([hidden])");
  await page.click("#folder-btn");
  await page.waitForSelector("#oc-pop .oc-path");
  await page.$eval("#oc-pop .oc-path", (e, v) => { (e as HTMLInputElement).value = v; (e as HTMLInputElement).focus(); }, patch);
  await page.keyboard.press("Enter");
  await page.waitForFunction((want) => document.querySelector("#oc-pop .oc-use")?.getAttribute("title") === want, {}, patch);
  await page.$eval("#oc-pop .oc-path", (e) => { (e as HTMLInputElement).value = "/"; (e as HTMLInputElement).focus(); });
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector("#oc-pop .oc-use")?.getAttribute("title") === "/");
  const recent = await page.$$eval("#oc-pop .oc-dir .invite-row", (r) => r.map((x) => x.getAttribute("title")));
  expect(recent.includes(patch)).toBe(true);
  await page.evaluate((want) => [...document.querySelectorAll<HTMLElement>("#oc-pop .oc-dir .invite-row")].find((r) => r.getAttribute("title") === want)!.click(), patch);
  await page.waitForFunction((want) => document.querySelector("#oc-pop .oc-use")?.getAttribute("title") === want, {}, patch);
  await page.click("#oc-pop .oc-use");
  await page.waitForFunction(() => [...document.querySelectorAll("#conversations .cast-name")].some((x) => x.textContent === "patch"));
  await page.close();
}, 30_000);

test("a chat's folder is chosen from the folder button by the message box: before the first message it is where the chat starts, and in a thread it moves the thread to another folder in OpenCode", async () => {
  const there = `${home}/garden`, later = `${home}/shed`;
  mkdirSync(there, { recursive: true });
  mkdirSync(later, { recursive: true });
  const page = await open(), last = () => opencode.log.prompts.at(-1)!;
  if (!(await page.$("#opencode .dot.online"))) await connect(page);
  await page.keyboard.press("Escape");
  await fresh(page);
  await page.waitForSelector("#folder-btn:not([hidden])");
  const chip = () => page.$eval("#folder-btn", (b) => [b.textContent, b.getAttribute("title")]);
  expect(await chip()).toEqual([home.split("/").pop()!, `Working in ${home}`]);
  const openIn = async (name: string) => {
    await page.waitForFunction((name) => [...document.querySelectorAll("#oc-pop .oc-dir .invite-name")].some((x) => x.textContent === name), {}, name);
    await page.evaluate((name) => [...document.querySelectorAll<HTMLElement>("#oc-pop .oc-dir")].find((r) => r.querySelector(".invite-name")!.textContent === name)!.querySelector<HTMLElement>(".oc-open")!.click(), name);
  };
  await page.click("#folder-btn");
  await openIn("garden");
  await page.waitForFunction(() => document.querySelector("#folder-btn")?.textContent === "garden");
  await send(page, "what grows here");
  expect([last().text, opencode.sessions.get(last().session)!.dir]).toEqual(["what grows here", there]);
  const sid = last().session;
  await fresh(page);
  await send(page, "what is in the note");
  await page.click("#folder-btn");
  await openIn("shed");
  const moved = last().session;
  await page.waitForFunction((d) => document.querySelector("#folder-btn")?.getAttribute("title") === `Working in ${d}`, {}, later);
  expect([opencode.sessions.get(moved)!.dir, opencode.sessions.get(sid)!.dir, (await castRows(page)).includes("shed")]).toEqual([later, there, true]);
  await page.close();
}, 30_000);

test("a character works in a thread from the folder's invite: her card goes to OpenCode as that thread's instruction and her face heads the reply, taking her out makes it plain again; a model picked from OpenCode's list is the one sent; Stop interrupts OpenCode's turn", async () => {
  const page = await open();
  await outside(page, true);
  await Bun.write(`${dir}/Mira.png`, await card("Mira", { description: "{{char}} keeps bees." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mira.png`);
  await page.waitForFunction(() => [...document.querySelectorAll("#conversations .cast-row .cast-name")].some((n) => n.textContent === "Mira"));
  await connect(page);
  await openFolder(page, work);
  const pick = (name: string) => page.evaluate((name) => [...document.querySelectorAll<HTMLElement>("#cast-invite .invite-row")].find((r) => r.querySelector(".invite-name")!.textContent === name)!.click(), name);
  await inRow(page, "site", ".cast-bring");
  expect(await page.$eval("#cast-invite .invite-head", (e) => e.textContent)).toBe("Who works in this thread");
  await pick("Mira");
  await page.keyboard.press("Escape");
  expect(await page.$eval("#input", (e) => (e as HTMLTextAreaElement).placeholder)).toBe("Message Mira");
  await send(page, "tidy the note");
  const first = opencode.log.prompts.at(-1)!;
  expect([/Write Mira's next reply in a fictional chat between Mira and User/.test(first.persona ?? ""), /Mira keeps bees\./.test(first.persona ?? ""), /can use the tools here/.test(first.persona ?? "")]).toEqual([true, true, true]);
  expect([await page.$eval("#thread > article.assistant:last-child .speaker", (e) => e.textContent),
    (await page.$eval("#thread > article.assistant:last-child", (e) => (e as HTMLElement).innerText)).includes("Mira here. The file says")]).toEqual(["Mira", true]);

  await inRow(page, "site", ".cast-bring");
  await pick("Mira");
  await page.keyboard.press("Escape");
  await page.click("#model-btn");
  await page.waitForSelector("#model-pop:not([hidden]) .model-group");
  expect(await page.$$eval("#model-pop .model-group, #model-pop .model-label", (x) => x.map((e) => e.textContent))).toEqual(["OpenCode", "Build", "Plan", "Chrysalis", "Qwen3.8 27B", "Elsewhere: leaves this computer", "Cloud One"]);
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#model-pop .model-item")].find((i) => i.textContent!.includes("Cloud One"))!.click());
  await send(page, "and once more");
  const second = opencode.log.prompts.at(-1)!;
  expect([second.session, /Mira/.test(second.persona ?? ""), second.model]).toEqual([first.session, false, { providerID: "elsewhere", id: "cloud-1" }]);
  expect(await page.$eval("#thread > article.assistant:last-child .speaker", (e) => e.textContent)).toBe("Chrysalis");

  const stops = opencode.log.interrupts;
  await page.type("#input", "read it slowly");
  await page.keyboard.press("Enter");
  await page.waitForSelector('#thread .tool-card[data-state="running"]');
  await page.click("#send");
  await page.waitForFunction(() => document.querySelector("#send")?.getAttribute("aria-label") === "Send");
  expect(opencode.log.interrupts).toBe(stops + 1);
  const sid = opencode.log.prompts.at(-1)!.session, said = () => opencode.sessions.get(sid)!.messages.filter((m: any) => m.type === "user" || m.type === "assistant").map((m: any) => m.type);
  const shown = () => page.$$eval("#thread > article", (a) => a.map((x) => (x.classList.contains("user") ? "user" : "assistant")));
  const [before, was] = [said(), await shown()];
  await page.evaluate(() => document.querySelector("#thread > article.assistant:last-child")!.querySelector<HTMLElement>('[data-act="trash"]')!.click());
  await page.waitForFunction((n) => document.querySelectorAll("#thread > article").length === n - 1, {}, was.length);
  expect([before.at(-1), said(), await shown(), opencode.sessions.get(sid)!.dir]).toEqual(["assistant", before.slice(0, -1), was.slice(0, -1), work]);
  await outside(page, false);
  await page.close();
}, 90_000);

test("when OpenCode wakes a thread by itself, the open page shows the message that woke it and the reply as it is written, and another chat's sidebar shows the thread working", async () => {
  const page = await open();
  await connect(page);
  await openFolder(page, work);
  await send(page, "count the bees");
  const session = opencode.log.prompts.at(-1)!.session;

  const before = await page.$$eval("#thread > article", (a) => a.length), prompts = opencode.log.prompts.length;
  void opencode.wake(session, "Explore", "Count the bees until let go.");
  await page.waitForFunction((n) => document.querySelectorAll("#thread > article").length === n + 2, { timeout: 15_000 }, before);
  await page.waitForSelector('#thread .tool-card[data-state="running"]');
  expect(await page.$eval("#thread > article.user:nth-last-child(2)", (e) => [e.querySelector(".point-chip")!.textContent, e.querySelector(".bubble, .markdown")!.textContent!.trim()]))
    .toEqual(["Sub-agent: Explore", "Count the bees until let go."]);
  expect(await page.$eval("#send", (e) => e.getAttribute("aria-label"))).toBe("Stop");
  opencode.release();
  const written = (text: string) => page.waitForFunction((text) => document.querySelector<HTMLElement>("#thread > article.assistant:last-child > .markdown")?.innerText.includes(text)
    && document.querySelector("#send")?.getAttribute("aria-label") === "Send", { timeout: 20_000 }, text);
  await written("You asked: Count the bees until let go.");
  expect(opencode.log.prompts.length - prompts).toBe(0);

  await fresh(page);
  void opencode.wake(session, "Explore", "Count them again until let go.");
  await page.waitForFunction(() => [...document.querySelectorAll("#conversations a.conv")].some((a) => a.querySelector(".conv-title")!.textContent === "count the bees" && /Working/.test(a.querySelector(".conv-status")?.textContent ?? "")),
    { timeout: 15_000 });
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#conversations a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "count the bees")!.click());
  await page.waitForFunction((n) => document.querySelectorAll("#thread > article").length === n + 4, { timeout: 15_000 }, before);
  opencode.release();
  await written("You asked: Count them again until let go.");
  await page.close();
}, 90_000);

test("a chat's row has T3 Code's menu on a right click and on a long press: Pin moves it to a Pinned section at the top, Rename changes its title here and in OpenCode, Settle and Delete act as their buttons do, a long press opens the menu without opening the chat, a plain tap opens it, and a chat whose session is gone from OpenCode leaves the list", async () => {
  const page = await open();
  await connect(page);
  await page.keyboard.press("Escape");
  await fresh(page);
  await send(page, "weigh the first jar");
  const session = opencode.log.prompts.at(-1)!.session;
  await fresh(page);
  await send(page, "weigh the second jar");
  const titles = (sel: string) => page.$$eval(`#conversations ${sel} .conv-title`, (t) => t.map((x) => x.textContent));
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#conversations a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "weigh the first jar")!
    .dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true, clientX: 60, clientY: 300 })));
  await page.waitForSelector("#row-menu:not([hidden])");
  expect(await page.$$eval("#row-menu button", (b) => b.map((x) => x.textContent))).toEqual(["New chat", "Pin", "Settle", "Rename", "Fork", "Delete"]);
  await page.keyboard.press("Escape");
  expect(await page.$eval("#row-menu", (m) => (m as HTMLElement).hidden)).toBe(true);

  await menu(page, "weigh the first jar", "Pin");
  expect([await titles(".folder.pinned"), (await titles(".folder:not(.pinned):not(.settled)")).includes("weigh the first jar")]).toEqual([["weigh the first jar"], false]);
  await menu(page, "weigh the first jar", "Rename");
  await page.waitForSelector("#conversations .conv-rename");
  await page.$eval("#conversations .conv-rename", (i) => { (i as HTMLInputElement).value = "The first jar"; });
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => [...document.querySelectorAll("#conversations .folder.pinned .conv-title")].some((t) => t.textContent === "The first jar"));
  for (let k = 0; k < 100 && opencode.sessions.get(session)!.title !== "The first jar"; k++) await Bun.sleep(50);
  expect(opencode.sessions.get(session)!.title).toBe("The first jar");
  await page.reload({ waitUntil: "load" });
  await page.waitForFunction(() => [...document.querySelectorAll("#conversations .folder.pinned .conv-title")].some((t) => t.textContent === "The first jar"));

  const row = (await page.evaluateHandle(() => [...document.querySelectorAll("#conversations a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "The first jar")!)).asElement()!;
  const box = (await row.boundingBox())!, at = { x: box.x + box.width / 2, y: box.y + box.height / 2 }, was = await page.$eval("#title", (t) => t.textContent);
  const touch = await page.createCDPSession();
  await touch.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [at] });
  await page.waitForSelector("#row-menu:not([hidden])", { timeout: 3000 });
  await touch.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  await Bun.sleep(300);
  expect([await page.$eval("#title", (t) => t.textContent), await page.$eval("#row-menu", (m) => (m as HTMLElement).hidden)]).toEqual([was, false]);
  await page.keyboard.press("Escape");
  await touch.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [at] });
  await touch.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  await page.waitForFunction(() => document.querySelector("#title")?.textContent === "The first jar");
  expect(await page.$eval("#row-menu", (m) => (m as HTMLElement).hidden)).toBe(true);

  await menu(page, "The first jar", "Unpin");
  await menu(page, "The first jar", "Settle");
  await page.waitForSelector("#conversations .settled-head[aria-expanded]");
  expect([await page.$("#conversations .folder.pinned"), await until(() => !!opencode.sessions.get(session)!.archived)]).toEqual([null, true]);
  await menu(page, "weigh the second jar", "Delete");
  await page.waitForFunction(() => ![...document.querySelectorAll("#conversations .conv-title")].some((t) => t.textContent === "weigh the second jar"));
  await page.evaluate(() => document.querySelector<HTMLElement>("#conversations .settled-head[aria-expanded]")!.click());
  await page.waitForFunction(() => [...document.querySelectorAll("#conversations .conv-title")].some((t) => t.textContent === "The first jar"));
  opencode.sessions.delete(session);
  await page.waitForFunction(() => ![...document.querySelectorAll("#conversations .conv-title")].some((t) => t.textContent === "The first jar"), { timeout: 10_000 });
  await page.close();
}, 90_000);

test("a thread's turns carry the settings' samplers, max tokens and seed to OpenCode's session, the thinking ones at a thinking level and the fast ones with thinking off", async () => {
  const page = await open(), last = () => opencode.log.prompts.at(-1)!;
  await page.evaluate(() => {
    sessionStorage.setItem("settings2", JSON.stringify({ v: 5, system: "", think: { temperature: 0.31, top_k: 7 }, fast: { temperature: 0.12, presence_penalty: 0.4 }, max_tokens: 500, seed: 9 }));
  });
  await page.reload();
  await connect(page);
  await openFolder(page, work);
  const sampling = async () => {
    const r = await fetch(`http://127.0.0.1:${opencode.server.port}/api/session`, { headers: { authorization: `Basic ${Buffer.from("opencode:test-secret").toString("base64")}` } });
    return (await r.json()).data.find((s: { id: string }) => s.id === last().session)?.metadata?.["chrysalis-sampling"];
  };
  await send(page, "tuned at max");
  expect(await sampling()).toEqual({ temperature: 0.31, top_k: 7, max_tokens: 500, seed: 9 });
  await page.click('#energy-stops [data-stop="0"]');
  await send(page, "tuned with thinking off");
  expect(await sampling()).toEqual({ temperature: 0.12, presence_penalty: 0.4, max_tokens: 500, seed: 9 });
  expect(await page.$eval("#toast", (t) => t.classList.contains("show") && /straight to/.test(t.textContent ?? ""))).toBe(false);
  await page.close();
}, 60_000);

test("OpenCode here is kept to Chrysalis's models until the switch under its button is turned off past a warning: before that a model outside is not listed, a turn naming one is refused and a thread left on one answers from Chrysalis; after it the model is listed as leaving this computer and its thread is marked; turned back on, it is gone again", async () => {
  const page = await open(), last = () => opencode.log.prompts.at(-1)!;
  await connect(page);
  await openFolder(page, work);
  const listed = () => page.evaluate(() => fetch("/api/threads").then((r) => r.json()).then((j) => [j.outside, j.models.map((m: { id: string }) => m.id)]));
  const groups = async () => {
    await page.click("#model-btn");
    const out = await page.$$eval("#model-pop .model-group", (g) => g.map((x) => [x.textContent, x.classList.contains("outside")]));
    await page.keyboard.press("Escape");
    return out;
  };
  const refused = () => page.evaluate((dir) => fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stream: true, chrysalis_thread: { dir, text: "to the cloud", model: "elsewhere/cloud-1" } }) }).then((r) => r.text()), work);
  expect(await listed()).toEqual([false, ["chrysalis/qwen3.8-27b"]]);
  expect((await groups()).some(([, away]) => away)).toBe(false);
  const before = opencode.log.prompts.length;
  expect(await refused()).toMatch(/elsewhere\/cloud-1 is outside this computer/);
  expect(opencode.log.prompts.length).toBe(before);
  await page.evaluate(() => localStorage.setItem("thread-model", "elsewhere/cloud-1"));
  await send(page, "kept at home");
  expect(last().model).toMatchObject({ providerID: "chrysalis", id: "qwen3.8-27b" });
  expect(await page.$eval("#model-btn", (b) => b.classList.contains("outside"))).toBe(false);
  await page.click("#opencode");
  await page.click("#oc-only");
  await page.waitForSelector("#outside-view[open]");
  await page.click("#outside-keep");
  expect([await page.$eval("#outside-view", (d) => (d as HTMLDialogElement).open), (await listed())[0]]).toEqual([false, false]);
  await page.click("#oc-only");
  await page.click("#outside-allow");
  await page.waitForFunction(() => document.querySelector("#oc-only")?.getAttribute("aria-checked") === "false" && !!document.querySelector("#opencode .dot.outside"));
  expect(await listed()).toEqual([true, ["chrysalis/qwen3.8-27b", "elsewhere/cloud-1"]]);
  await page.keyboard.press("Escape");
  const toasts = () => page.$eval("#toast span", (t) => t.textContent ?? "");
  await fresh(page);
  await page.waitForFunction(() => !document.querySelector(".msg"));
  expect(await groups()).toContainEqual(["Elsewhere: leaves this computer", true]);
  expect(await page.$eval("#model-btn", (b) => b.classList.contains("outside"))).toBe(true);
  await page.evaluate(() => sessionStorage.setItem("settings2", JSON.stringify({ v: 5, system: "", think: { temperature: 0.31 }, fast: {}, seed: 9 })));
  await page.reload();
  await page.waitForSelector("#opencode .dot.outside");
  await openFolder(page, work);
  await send(page, "tuned somewhere else");
  expect(last().model).toMatchObject({ providerID: "elsewhere", id: "cloud-1" });
  expect(await toasts()).toBe("This OpenCode sends elsewhere/cloud-1 straight to cloud.example, so your sampler settings, max tokens and seed don't reach it");
  await page.click("#opencode");
  await page.click("#oc-only");
  await page.waitForFunction(() => document.querySelector("#oc-only")?.getAttribute("aria-checked") === "true" && !!document.querySelector("#opencode .dot.online"));
  expect(await listed()).toEqual([false, ["chrysalis/qwen3.8-27b"]]);
  expect(await refused()).toMatch(/elsewhere\/cloud-1 is outside this computer/);
  await page.close();
}, 90_000);
