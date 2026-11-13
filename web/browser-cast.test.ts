import { expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import type { Browser, Page } from "puppeteer-core";
import { harness, RUN, articles, card, fresh, castRows, dir, download, inRow, lastStream, look, open, openRow, pagerOf, pickVoice, png, pngText, send, sent, shown, streams, untilIdle, voiceLog } from "./browser-harness";

harness();

const rowMenu = async (page: Page, name: string, item = "") => {
  await page.evaluate((name) => [...document.querySelectorAll("#conversations .cast-row")].find((r) => r.querySelector(".cast-name")?.textContent === name)!
    .dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true, clientX: 60, clientY: 300 })), name);
  await page.waitForSelector("#row-menu:not([hidden])");
  const labels = await page.$$eval("#row-menu button", (b) => b.map((x) => x.textContent!));
  if (item) await page.evaluate((item) => [...document.querySelectorAll<HTMLElement>("#row-menu button")].find((b) => b.textContent === item)!.click(), item);
  else await page.keyboard.press("Escape");
  return labels;
};
const carry = async (page: Page, rows: string, name: string, to: () => Promise<number[]>) => {
  const from = await page.evaluate((rows, name) => { const r = [...document.querySelectorAll<HTMLElement>(rows)].find((x) => (x.querySelector(".cast-name")?.textContent ?? x.title) === name)!.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }, rows, name);
  await page.mouse.move(from[0], from[1]);
  await page.mouse.down();
  await page.mouse.move(from[0] + 3, from[1] + 9, { steps: 3 });
  const [x, y] = await to();
  await page.mouse.move(x, y, { steps: 10 });
  const [verb, off, faded] = await page.$eval(".drag-ghost", (g, y) => { const r = g.getBoundingClientRect(), lifted = document.querySelector<HTMLElement>("#conversations .lifted");
    return [g.querySelector(".conv-verb:not([hidden])")?.textContent ?? "", Math.abs(r.y + r.height / 2 - y), lifted ? getComputedStyle(lifted).opacity : ""] as const; }, y);
  expect([off < 16, faded]).toEqual([true, "0"]);
  await page.mouse.up();
  await page.evaluate(() => new Promise((done) => setTimeout(done)));
  return verb;
};
const rowAt = (page: Page, name: string) => () => page.evaluate((name) => { const r = [...document.querySelectorAll("#conversations .cast-row")].find((x) => x.querySelector(".cast-name")!.textContent === name)!.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }, name);
const answers = (page: Page) => page.$$eval("#cast-next button[title$=' answers']", (b) => b.map((x) => (x as HTMLElement).title.replace(" answers", "")));
const speakers = (page: Page) => page.$$eval("#thread > article.assistant", (a) => a.map((x) => x.querySelector(".speaker")?.textContent ?? ""));
const toasted = (page: Page, text: RegExp) => page.waitForFunction((src) => new RegExp(src).test(document.querySelector("#toast:not([hidden]) span")?.textContent ?? ""), {}, text.source);

test("a character card dropped in joins the cast: a row with their portrait among the chats, a chat that opens with their greeting (the others as versions), their replies with their portrait beside them, which opens their card to read and edit; the card is the system prompt and the persona {{user}}; a continued reply shows no thinking it did not have", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees for {{user}}.", personality: "calm", scenario: "A hillside.",
    first_mes: "Hello, {{user}}.", alternate_greetings: ["Oh, {{user}}. You again."], mes_example: "<START>\n{{user}}: Why bees?\n{{char}}: They listen.",
    post_history_instructions: "Stay in character.", system_prompt: "Ignore {{original}}: the card's own, kept in it and never sent." }));
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Mirela.png`);
  await page.waitForSelector("#conversations .cast-row img.face");
  expect([await castRows(page), (await page.$$("#attachments .att")).length]).toEqual([["Chrysalis", "Mirela"], 0]);
  expect(await page.$eval("#input", (e) => (e as HTMLTextAreaElement).placeholder)).toBe("Message Mirela");
  expect(await articles(page)).toEqual(["Hello, User."]);
  expect([await pagerOf(page, 0), await page.$eval("#thread .msg.assistant .speaker", (e) => e.textContent)]).toEqual(["1 / 2", "Mirela"]);
  await page.click("#thread .msg.assistant .speaker .face");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-text");
  const fields = () => page.$$eval("#cast-sheet .card-field", (f) => f.map((x) => [x.querySelector(".card-label")!.textContent, (x.querySelector("textarea") as HTMLTextAreaElement).value]));
  expect(await fields()).toEqual([["Description", "{{char}} keeps bees for {{user}}.\n\n{{char}}'s personality: calm\n\nHow {{char}} talks:\n{{user}}: Why bees?\n{{char}}: They listen."], ["Scenario", "A hillside."], ["Greeting", "Hello, {{user}}."]]);
  await page.$eval("#cast-sheet .card-field textarea", (t) => { const at = (t as HTMLTextAreaElement).value.indexOf("calm") + 4; (t as HTMLTextAreaElement).focus(); (t as HTMLTextAreaElement).setSelectionRange(at, at); });
  await page.keyboard.type(" and kind");
  await page.keyboard.press("Escape");

  expect([await page.$eval("#you .you-name", (e) => e.textContent), await page.$eval("#cast-sheet", (e) => (e as HTMLElement).hidden)]).toEqual(["User", true]);
  await page.click("#you");
  await page.waitForSelector("#cast-sheet:not([hidden]) input.card-name");
  expect(await page.$eval("#cast-sheet input.card-name", (e) => (e as HTMLInputElement).value)).toBe("User");
  await page.$eval("#cast-sheet input.card-name", (e) => { (e as HTMLInputElement).focus(); (e as HTMLInputElement).select(); });
  await page.keyboard.type("Robin");
  await page.keyboard.press("Escape");
  await send(page, "good morning mirela");
  expect([await page.$eval("#you .you-name", (e) => e.textContent), await page.$eval("#thread .msg.user .speaker", (e) => e.textContent)]).toEqual(["Robin", "Robin"]);
  const one = lastStream.messages;
  expect(one[0].content).toBe("Write Mirela's next reply in a fictional chat between Mirela and Robin.\n\nMirela keeps bees for Robin.\n\nMirela's personality: calm and kind" +
    "\n\nHow Mirela talks:\nRobin: Why bees?\nMirela: They listen.\n\nScenario: A hillside.");
  expect(one.slice(1)).toEqual([{ role: "user", content: "[The scene begins.]" }, { role: "assistant", content: "Hello, User." },
    { role: "user", content: "good morning mirela" }]);
  expect(await page.$$eval("#thread .msg.assistant .speaker", (s) => s.map((x) => x.textContent))).toEqual(["Mirela", "Mirela"]);
  await page.$eval('#thread > article[data-i="2"] [data-act="cont"]', (b) => (b as HTMLElement).click());
  await page.waitForFunction(() => /Good morning\..*Good morning\./.test(document.querySelector('#thread > article[data-i="2"] .markdown')?.textContent ?? ""));
  await untilIdle(page);
  expect([lastStream.continue_final_message, await page.$eval('#thread > article[data-i="2"] .thinking', (t) => (t as HTMLElement).hidden)]).toEqual([true, true]);
  await page.close();
}, 40_000);

test("the user is a character like any other: a portrait opens its card and closes it again (another's switches to theirs), in the chat, on their sidebar row and on a chat's row; the user's card has every field a character's has, sent as theirs are, their greeting the first message of a new chat (in the box), their voice the one their messages are read in; their messages and the replies alike in bubbles, each from its writer's side", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", first_mes: "Hello, {{user}}." }));
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Mirela.png`);
  await page.waitForFunction(() => document.querySelector("#thread .speaker")?.textContent === "Mirela");
  const shown = () => page.$eval("#cast-sheet", (s) => (s as HTMLElement).hidden ? "" : s.querySelector<HTMLInputElement>("input.card-name")!.value);
  const steps: string[] = [];
  for (const click of [() => page.click("#thread .msg.assistant .speaker .face"), () => page.click("#you"), () => page.click("#you"),
    () => inRow(page, "Mirela", ".cast-pic"), () => inRow(page, "Mirela", ".cast-pic")]) { await click(); steps.push(await shown()); }
  expect(steps).toEqual(["Mirela", "User", "", "Mirela", ""]);
  const parts = () => page.$eval("#cast-sheet", (s) => [...s.querySelectorAll(".card-pic, .card-voice, .card-label")].map((x) => x.classList.contains("card-pic") ? "picture" : x.classList.contains("card-voice") ? "voice" : x.textContent));
  await inRow(page, "Mirela", ".cast-pic");
  const hers = await parts();
  await page.click("#you");
  expect(await parts()).toEqual(hers);
  await pickVoice(page, "#cast-sheet .card-voice .voice-name", "other-tts/bob");
  for (const [label, text] of [["Description", "{{user}} is curious."], ["Greeting", "Hi, {{char}}."]]) {
    await page.evaluate((label) => {
      const f = [...document.querySelectorAll("#cast-sheet .card-field")].find((x) => x.querySelector(".card-label")!.textContent === label)!;
      f.querySelector("textarea")!.focus();
    }, label);
    await page.keyboard.type(text);
  }
  await page.keyboard.press("Escape");
  await inRow(page, "Mirela", ".cast-new");
  expect(await page.$eval("#input", (e) => (e as HTMLTextAreaElement).value)).toBe("Hi, Mirela.");
  const done = await page.$$eval(".msg .stats", (s) => s.length);
  await page.focus("#input");
  await page.keyboard.press("Enter");
  await page.waitForFunction((n) => document.querySelectorAll(".msg .stats").length > n, { timeout: 20_000 }, done);
  expect(lastStream.messages[0].content).toBe("Write Mirela's next reply in a fictional chat between Mirela and User.\n\nMirela keeps bees.\n\n" +
    "User is curious.");
  expect(lastStream.messages.at(-1)).toEqual({ role: "user", content: "Hi, Mirela." });
  await page.waitForSelector("#conversations a.conv.active .conv-face[data-card]");
  const at = await page.evaluate(() => location.hash);
  await page.click("#conversations a.conv.active .conv-face[data-card]");
  expect([await shown(), await page.evaluate(() => location.hash)]).toEqual(["Mirela", at]);
  await page.click("#conversations a.conv.active .conv-face[data-card]");
  expect(await shown()).toBe("");
  const look = await page.evaluate(() => {
    const at = (s: string) => document.querySelector(s)!.getBoundingClientRect(), mine = at(".msg.user .bubble"), hers = at(".msg.assistant > .markdown");
    const style = (s: string) => { const c = getComputedStyle(document.querySelector(s)!); return [c.borderTopWidth, c.backgroundColor, c.paddingLeft]; };
    return { sides: [hers.left - at(".msg.assistant .speaker .face").right, at(".msg.user .speaker .face").left - mine.right],
      bubbles: [style(".msg.user .bubble"), style(".msg.assistant > .markdown")] };
  });
  expect([look.sides[0] > 0 && look.sides[0] === look.sides[1], look.bubbles[0][0], look.bubbles[1]]).toEqual([true, "1px", look.bubbles[0]]);
  const n = voiceLog().length, button = '#thread > article[data-i="1"] [data-act="speak"]';
  await page.click(button);
  await page.waitForFunction((b) => !document.querySelector(`${b}.on`), { timeout: 10_000 }, button);
  const said = voiceLog().slice(n).filter((e) => e.ev === "chunk" || e.type === "say");
  expect([said.find((e) => e.type === "say")?.text, [...new Set(said.filter((e) => e.ev === "chunk").map((e) => `${e.tts}/${e.voice}`))]]).toEqual(["Hi, Mirela.", ["other-tts/bob"]]);
  await page.close();
}, 40_000);

test("a card is saved from its Download card button, named for them and as it is now, the PNG any app reads with the picture as given; a click on its picture saves nothing", async () => {
  const page = await open();
  const saved = mkdtempSync(`${RUN}/saved-`), cdp = await page.createCDPSession();
  await cdp.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: saved, browserContextId: page.browserContext().id });
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", first_mes: "Hello, {{user}}." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mirela.png`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 2);
  await inRow(page, "Mirela", ".cast-pic");
  await page.click("#cast-sheet .card-field textarea");
  await page.keyboard.press("End");
  await page.keyboard.type(" Twelve hives.");
  await Bun.sleep(600);
  await page.click("#cast-sheet .card-portrait");
  expect(await page.$eval("#cast-sheet", (s) => (s as HTMLElement).hidden)).toBe(false);
  await page.click("#cast-sheet .card-acts a.card-file");
  for (const t = Date.now(); !existsSync(`${saved}/Mirela.png`); await Bun.sleep(50)) if (Date.now() - t > 10_000) throw new Error(`not saved: ${readdirSync(saved)}`);
  await Bun.sleep(200);
  const got = pngText(readFileSync(`${saved}/Mirela.png`));
  const v3 = JSON.parse(Buffer.from(got.text.ccv3 ?? "", "base64").toString()), v2 = JSON.parse(Buffer.from(got.text.chara ?? "", "base64").toString());
  expect([readdirSync(saved), v3.data.name, v3.data.description, v2.data.description]).toEqual([["Mirela.png"], "Mirela", "{{char}} keeps bees. Twelve hives.", "{{char}} keeps bees. Twelve hives."]);
  expect(got.idat).toBe(pngText(png).idat);
  expect(await page.$eval("#cast-sheet a.card-file", (a) => [(a as HTMLAnchorElement).download, a.textContent])).toEqual(["Mirela.png", "Download card"]);
  await page.close();
}, 30_000);

test("the conversations are folders, Chrysalis's and each character's: a folder opened stays open while another's chat is, a closed one says how many chats it holds, and both stay as left after a reload", async () => {
  const page = await open();
  await send(page, "one for chrysalis");
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", first_mes: "Hello, {{user}}." }));
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Mirela.png`);
  await page.waitForFunction(() => document.querySelector("#thread .speaker")?.textContent === "Mirela");
  await send(page, "good morning mirela");
  const folders = () => page.$$eval("#conversations .folder", (f) => f.map((x) => [x.querySelector(".cast-name")!.textContent,
    x.classList.contains("open") ? `open: ${x.querySelectorAll(".conv").length}` : `closed: ${x.querySelector(".cast-count")?.textContent}`]));
  expect(await folders()).toEqual([["Mirela", "open: 1"], ["Chrysalis", "open: 1"]]);
  await openRow(page, "Chrysalis");
  expect(await folders()).toEqual([["Mirela", "open: 1"], ["Chrysalis", "closed: 1"]]);
  await openRow(page, "Chrysalis");
  await page.click("#conversations .folder:first-child .conv");
  expect(await folders()).toEqual([["Mirela", "open: 1"], ["Chrysalis", "open: 1"]]);
  await openRow(page, "Mirela");
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#conversations .folder");
  expect(await folders()).toEqual([["Mirela", "closed: 1"], ["Chrysalis", "open: 1"]]);
  await page.close();
}, 30_000);

test("a character deleted from their row's menu takes their chats with them (none become Chrysalis's) on the first click, Undo brings it all back, and a character deleted in another tab takes their chats here too", async () => {
  const page = await open();
  await send(page, "one for chrysalis");
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", first_mes: "Hello, {{user}}." }));
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Mirela.png`);
  await page.waitForFunction(() => document.querySelector("#thread .speaker")?.textContent === "Mirela");
  await send(page, "good morning mirela");
  const kept = () => page.evaluate(() => Object.values(JSON.parse(sessionStorage.getItem("chats")!)).map((c: any) => c.who ?? "chrysalis").sort());
  const folders = () => page.$$eval("#conversations .folder", (f) => f.map((x) => `${x.querySelector(".cast-name")!.textContent}: ${x.querySelectorAll(".conv").length || x.querySelector(".cast-count")?.textContent}`));
  const mirela = (await kept()).find((w) => w !== "chrysalis")!;
  await inRow(page, "Mirela", ".cast-pic");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-field");
  expect(await page.$$eval("#cast-sheet button", (b) => b.filter((x) => /delete/i.test(`${x.title}${x.textContent}`)).length)).toBe(0);
  await page.keyboard.press("Escape");
  await rowMenu(page, "Mirela", "Delete");
  await toasted(page, /^Mirela deleted, with 1 chat/);
  expect([await kept(), await folders(), await page.$$eval("#thread .speaker", (s) => s.length)]).toEqual([["chrysalis"], ["Chrysalis: 1"], 0]);
  await page.click("#toast .btn");
  await page.waitForFunction(() => document.querySelectorAll("#conversations .folder").length === 2);
  expect((await kept()).sort()).toEqual(["chrysalis", mirela].sort());
  await page.evaluate((id) => new Promise<void>((done) => {
    const open = indexedDB.open("chrysalis-cast", 1);
    open.onsuccess = () => { const t = open.result.transaction("people", "readwrite"); t.objectStore("people").delete(id); t.oncomplete = () => { open.result.close(); done(); }; };
  }), mirela);
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#conversations .folder");
  expect([await kept(), await folders()]).toEqual([["chrysalis"], ["Chrysalis: 1"]]);
  await page.close();
}, 30_000);

test("a group's file brings its members' cards with it: those with no card here come from the file, a name with no card anywhere is left out and said, the group is there after a reload, its card lists its members, each opening their own card, and its exported file brings it to a browser that has none of them", async () => {
  const page = await open();
  await Bun.write(`${dir}/Ada.png`, await card("Ada", { description: "{{char}} hums." }));
  writeFileSync(`${dir}/Duo.json`, JSON.stringify({ spec: "chrysalis_group", name: "Duo", members: ["Ada", "Bea", "Cy"], scenario: "A cellar.",
    cards: [{ spec: "chara_card_v2", spec_version: "2.0", data: { name: "Bea", description: "{{char}} drums." } }] }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Ada.png`, `${dir}/Duo.json`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .folder").length === 4);
  const tiles = () => page.evaluate(() => [...[...document.querySelectorAll("#conversations .cast-row")].find((r) => r.querySelector(".cast-name")!.textContent === "Duo")!
    .querySelectorAll(".face.mosaic > *")].map((f) => f.tagName === "IMG" ? "picture" : f.textContent));
  expect([await castRows(page), await tiles(), await page.$eval("#toast span", (t) => t.textContent)]).toEqual([["Ada", "Bea", "Chrysalis", "Duo"], ["picture", "B"], "Ada and Duo are here; no card here for Cy"]);
  await openRow(page, "Duo");
  await page.waitForSelector("#cast-next .talk");
  expect(await answers(page)).toEqual(["Ada", "Bea"]);
  await send(page, "Bea, a beat");
  expect([/Bea drums\./.test(lastStream.messages[0].content), /Scenario: A cellar\./.test(lastStream.messages[0].content), await speakers(page)]).toEqual([true, true, ["BBea"]]);
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#conversations .folder");
  expect([await castRows(page), await tiles()]).toEqual([["Ada", "Bea", "Chrysalis", "Duo"], ["picture", "B"]]);
  const into = mkdtempSync(`${dir}/duo-`), cdp = await page.createCDPSession();
  await cdp.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: into, browserContextId: page.browserContext().id });
  await rowMenu(page, "Duo", "Card");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-member");
  expect(await page.$$eval("#cast-sheet .card-member", (m) => m.map((x) => [x.querySelector(".card-member-open span:last-child")!.textContent, x.querySelector<HTMLButtonElement>(".card-member-out")!.disabled]))).toEqual([["Ada", true], ["Bea", true]]);
  expect(await page.$$eval("#conversations .folder", (f) => f.map((x) => [...x.querySelectorAll(".mosaic > [data-person]")].map((m) => (m as HTMLElement).title)).filter((m) => m.length))).toEqual([["Ada", "Bea"]]);
  await page.click("#cast-sheet .card-member:nth-child(2) .card-member-open");
  await page.waitForFunction(() => document.querySelector<HTMLInputElement>("#cast-sheet input.card-name")?.value === "Bea");
  await rowMenu(page, "Duo", "Card");
  await page.waitForSelector("#cast-sheet:not([hidden]) .strips");
  await page.click("#cast-sheet .strips > :first-child");
  await page.waitForFunction(() => document.querySelector<HTMLInputElement>("#cast-sheet input.card-name")?.value === "Ada");
  await page.click("#conversations .mosaic > [data-person]:nth-child(2)");
  await page.waitForFunction(() => document.querySelector<HTMLInputElement>("#cast-sheet input.card-name")?.value === "Duo" || !!document.querySelector("#cast-sheet .card-member"));
  await rowMenu(page, "Duo", "Card");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-acts button");
  await page.click("#cast-sheet .card-acts button");
  for (let k = 0; k < 50 && !existsSync(`${into}/Duo.json`); k++) await Bun.sleep(100);
  const file = JSON.parse(readFileSync(`${into}/Duo.json`, "utf8"));
  expect([file.spec, file.name, file.members, file.cards.map((c: any) => c.data.name), file.scenario]).toEqual(["chrysalis_group", "Duo", ["Ada", "Bea"], ["Ada", "Bea"], "A cellar."]);
  const other = await open();
  await (await other.$("input#cast-file"))!.uploadFile(`${into}/Duo.json`);
  await other.waitForFunction(() => document.querySelectorAll("#conversations .folder").length === 4);
  expect(await castRows(other)).toEqual(["Ada", "Bea", "Chrysalis", "Duo"]);
  await other.close();
  await page.close();
}, 30_000);

test("a group's replies are each one character's own message: whoever the message names answers, each of them in turn when it names several, and the next in turn when it names nobody; a face has that one answer, thought through first; let them talk takes turns; what was said before goes to the model with who said it; a card dropped on the chat joins the group where it is, and a character dragged onto another's chat makes a group of the two that the chat moves to; all of it there after a reload", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", scenario: "A hillside.", first_mes: "Hello, {{user}}." }));
  await Bun.write(`${dir}/Tomas.png`, await card("Tomas", { description: "{{char}} sells honey.", first_mes: "Honey?" }));
  await Bun.write(`${dir}/Ida.png`, await card("Ida", { description: "{{char}} paints hives." }));
  writeFileSync(`${dir}/Hive.json`, JSON.stringify({ spec: "chrysalis_group", name: "Hive", members: ["Mirela", "Tomas"], scenario: "Market day." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mirela.png`, `${dir}/Tomas.png`, `${dir}/Hive.json`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 4);
  await openRow(page, "Hive");
  await page.waitForSelector("#cast-next .talk");
  const told = (b: any) => b.messages.slice(1).map((m: any) => [m.role, m.content]);
  const settled = (n: number) => page.waitForFunction((n) => document.querySelectorAll("#thread > article.assistant").length === n && document.querySelectorAll("#thread > article.assistant .stats").length === n
    && document.querySelector("#send")?.getAttribute("aria-label") === "Send", { timeout: 20_000 }, n);
  await send(page, "a toast for the hive");
  expect([await speakers(page), told(lastStream), lastStream.stop, lastStream.messages[0].content.includes("Scenario: Market day."), lastStream.messages[0].content.includes("A hillside.")])
    .toEqual([["Mirela"], [["user", "a toast for the hive\n\n[Mirela next.]"]], ["\nUser:", "\n**User"], true, false]);
  expect(await page.$$eval("#thread > article.assistant > .markdown", (m) => m.map((x) => x.textContent!.trim()))).toEqual(["She raises a cup. To the bees."]);
  await send(page, "Tomas, and you?");
  expect([await speakers(page), told(lastStream)]).toEqual([["Mirela", "Tomas"],
    [["user", "a toast for the hive\n\n[Mirela next.]"], ["assistant", "*She raises a cup.* To the bees."], ["user", "Tomas, and you?\n\n[Tomas next.]"]]]);
  const n1 = streams.length;
  await page.type("#input", "Tomas, then Mirela: a song");
  await page.keyboard.press("Enter");
  await settled(4);
  expect([await speakers(page), streams.slice(n1).map((b) => b.messages.at(-1).content)]).toEqual([["Mirela", "Tomas", "Tomas", "Mirela"], ["Tomas, then Mirela: a song\n\n[Tomas next.]", "[Mirela next.]"]]);

  await page.click('#cast-next button[title="Tomas answers"]');
  await settled(5);
  expect([lastStream.continue_final_message, lastStream.reasoning_effort !== undefined, lastStream.messages.at(-1)])
    .toEqual([undefined, true, { role: "user", content: "[Tomas next.]" }]);
  expect([await speakers(page), await page.$eval('#thread > article.assistant:last-child .thought', (t) => t.textContent)]).toEqual([["Mirela", "Tomas", "Tomas", "Mirela", "Tomas"], "Tomas has the floor."]);
  const n0 = streams.length;
  await page.click("#cast-next .talk");
  await page.waitForFunction(() => !!document.querySelector("#cast-next .talk:not(.on)")
    && document.querySelector("#send")?.getAttribute("aria-label") === "Send", { timeout: 20_000 });
  expect(streams.slice(n0).map((b) => b.messages.at(-1))).toEqual(["Mirela", "Tomas", "Mirela", "Tomas"].map((n) => ({ role: "user", content: `[${n} next.]` })));

  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Ida.png`);
  await page.waitForFunction(() => document.querySelectorAll("#cast-next button[title$=' answers']").length === 3);
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Mirela.png`);
  await toasted(page, /Mirela is in this chat already/);
  expect([await castRows(page), (await page.$$("#attachments .att")).length]).toEqual([["Chrysalis", "Hive", "Ida", "Mirela", "Tomas"], 0]);
  await openRow(page, "Mirela");
  await page.waitForFunction(() => document.querySelector("#thread .speaker")?.textContent === "Mirela");
  expect(await carry(page, '#conversations .cast-row[data-person]:has(.cast-name)', "Tomas", () => page.$eval("#thread", (t) => { const r = t.getBoundingClientRect(); return [r.x + r.width / 2, r.y + 200]; }))).toBe("Join this chat");
  await page.waitForSelector("#cast-next .talk");
  expect([await castRows(page), await answers(page), await speakers(page)])
    .toEqual([["Chrysalis", "Hive", "Ida", "Mirela", "Mirela and Tomas", "Tomas"], ["Mirela", "Tomas"], ["Mirela"]]);

  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#conversations .cast-row");
  expect(await castRows(page)).toEqual(["Chrysalis", "Hive", "Ida", "Mirela", "Mirela and Tomas", "Tomas"]);
  await page.close();
}, 60_000);

test("a group is edited where it is: the invite on a character's own chat makes a group of them that the chat moves to, picking again takes one out and Undo brings them back, the last two stay; the invite on a group's row changes that group, whose row and chats stay; on a character's row it makes a new group; all sent as such and there after a reload", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", first_mes: "Hello, {{user}}." }));
  await Bun.write(`${dir}/Tomas.png`, await card("Tomas", { description: "{{char}} sells honey.", first_mes: "Honey?" }));
  await Bun.write(`${dir}/Ida.png`, await card("Ida", { description: "{{char}} paints hives." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mirela.png`, `${dir}/Tomas.png`, `${dir}/Ida.png`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 4);
  await page.waitForFunction(() => document.querySelector("#thread .speaker")?.textContent === "Mirela");
  const rows = () => page.$$eval("#cast-invite .invite-row", (r) => r.map((x) => [x.querySelector(".invite-name")!.textContent, x.getAttribute("aria-pressed") === "true", (x as HTMLButtonElement).disabled]).filter(([name]) => name !== "Chrysalis"));
  const pick = (name: string) => page.evaluate((name) => [...document.querySelectorAll<HTMLElement>("#cast-invite .invite-row")]
    .find((r) => r.querySelector(".invite-name")!.textContent === name)!.click(), name);
  expect(await page.$eval("#cast-next", (b) => (b as HTMLElement).hidden)).toBe(true);
  await inRow(page, "Mirela", ".cast-bring");
  await page.waitForSelector("#cast-invite:not([hidden])");
  expect(await rows()).toEqual([["Ida", false, false], ["Mirela", true, false], ["Tomas", false, false]]);
  await pick("Tomas");
  await pick("Ida");
  expect([await answers(page), await rows(), await castRows(page), await page.$eval("#input", (e) => (e as HTMLTextAreaElement).placeholder), await speakers(page)])
    .toEqual([["Mirela", "Tomas", "Ida"], [["Ida", true, false], ["Mirela", true, false], ["Tomas", true, false]],
      ["Chrysalis", "Ida", "Mirela", "Mirela, Tomas and Ida", "Tomas"], "Message Mirela, Tomas and Ida", ["Mirela"]]);
  await pick("Tomas");
  expect([await answers(page), await page.$eval("#toast span", (t) => t.textContent), await rows()])
    .toEqual([["Mirela", "Ida"], "Tomas left the group", [["Ida", true, true], ["Mirela", true, true], ["Tomas", false, false]]]);
  await page.click("#toast button");
  expect([await answers(page), await castRows(page)]).toEqual([["Mirela", "Tomas", "Ida"], ["Chrysalis", "Ida", "Mirela", "Mirela, Tomas and Ida", "Tomas"]]);
  await page.keyboard.press("Escape");
  expect(await page.$eval("#cast-invite", (e) => (e as HTMLElement).hidden)).toBe(true);

  await inRow(page, "Tomas", ".cast-new");
  await page.waitForFunction(() => document.querySelector("#thread .speaker")?.textContent === "Tomas");
  await inRow(page, "Tomas", ".cast-bring");
  await page.waitForSelector("#cast-invite:not([hidden])");
  expect([await page.$eval("#cast-invite .invite-head", (e) => e.textContent), await rows()]).toEqual(["Bring into this chat", [["Ida", false, false], ["Mirela", false, false], ["Tomas", true, false]]]);
  await pick("Ida");
  expect([await answers(page), await castRows(page)]).toEqual([["Tomas", "Ida"], ["Chrysalis", "Ida", "Mirela", "Mirela, Tomas and Ida", "Tomas", "Tomas and Ida"]]);
  await page.keyboard.press("Escape");
  await send(page, "a toast for the hive");
  const system = () => [/Tomas sells honey\./, /Ida paints hives\./, /Mirela keeps bees\./].map((r) => r.test(lastStream.messages[0].content));
  expect([system(), await speakers(page)]).toEqual([[true, true, false], ["Tomas", "Ida"]]);

  await inRow(page, "Tomas and Ida", ".cast-new");
  await page.waitForFunction(() => !document.querySelector("#thread > article"));
  expect(await answers(page)).toEqual(["Tomas", "Ida"]);
  await inRow(page, "Tomas and Ida", ".cast-bring");
  expect(await page.$eval("#cast-invite .invite-head", (e) => e.textContent)).toBe("Who is in Tomas and Ida");
  await pick("Mirela");
  await page.keyboard.press("Escape");
  await send(page, "a toast for the hive");
  const all = ["Chrysalis", "Ida", "Mirela", "Mirela, Tomas and Ida", "Tomas", "Tomas, Ida and Mirela"];
  expect([system(), await castRows(page), await page.evaluate(() => [...document.querySelectorAll("#conversations .folder")].find((f) => f.querySelector(".cast-name")!.textContent === "Tomas, Ida and Mirela")!.querySelectorAll("a.conv").length)]).toEqual([[true, true, true], all, 2]);
  await inRow(page, "Mirela", ".cast-bring");
  expect([await page.$eval("#cast-invite .invite-head", (e) => e.textContent), await rows(), await answers(page)])
    .toEqual(["A new group with Mirela and", [["Ida", false, false], ["Mirela", true, true], ["Tomas", false, false]], ["Tomas", "Ida", "Mirela"]]);
  await pick("Ida");
  const then = [...all, "Mirela and Ida"].sort();
  expect([await answers(page), await castRows(page), await page.$$eval("#thread > article", (a) => a.length)]).toEqual([["Mirela", "Ida"], then, 0]);
  await page.keyboard.press("Escape");
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#conversations .cast-row");
  expect(await castRows(page)).toEqual(then);
  await page.close();
}, 60_000);

test("groups saved by earlier versions are groups still: one kept in the cast store keeps its chats, its name and its scenario, a member since deleted left out, and a chat that held its own people becomes a group of them", async () => {
  const page = await open();
  await Bun.write(`${dir}/Ada.png`, await card("Ada", { description: "{{char}} hums." }));
  await Bun.write(`${dir}/Bea.png`, await card("Bea", { description: "{{char}} drums." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Ada.png`, `${dir}/Bea.png`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 3);
  await openRow(page, "Ada");
  await page.waitForFunction(() => (document.querySelector("#input") as HTMLTextAreaElement).placeholder === "Message Ada");
  await send(page, "good morning mirela");
  await page.evaluate(async () => {
    const db = await new Promise<IDBDatabase>((ok) => { const o = indexedDB.open("chrysalis-cast"); o.onsuccess = () => ok(o.result); });
    const people = await new Promise<{ id: string; kind: string; card: { data: { name: string } } }[]>((ok) => {
      const r = db.transaction("people").objectStore("people").getAll();
      r.onsuccess = () => ok(r.result);
    });
    const id = (n: string) => people.find((p) => p.kind === "character" && p.card.data.name === n)!.id;
    const t = db.transaction("groups", "readwrite");
    t.objectStore("groups").put({ id: "old-group", name: "The band", members: [id("Ada"), "gone", id("Bea")], scenario: "A cellar.", added: 1 });
    await new Promise((ok) => { t.oncomplete = ok; });
    const chats = JSON.parse(sessionStorage.getItem("chats")!), [first] = Object.values(chats) as any[], key = `cast:${[id("Ada"), id("Bea")].sort().join("+")}`;
    first.who = "old-group";
    chats.older = { ...structuredClone(first), id: "older", title: "An older chat", who: key, cast: [id("Bea"), id("Ada")], castName: "The pair", scenario: "A porch." };
    sessionStorage.setItem("chats", JSON.stringify(chats));
  });
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#cast-next .talk");
  expect([await castRows(page), await answers(page)]).toEqual([["Ada", "Bea", "Chrysalis", "The band", "The pair"], ["Ada", "Bea"]]);
  await send(page, "a duet");
  expect([/Scenario: A cellar\./, /Ada hums\./, /Bea drums\./].map((r) => r.test(lastStream.messages[0].content))).toEqual([true, true, true]);
  await openRow(page, "The pair");
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#conversations a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "An older chat")!.click());
  await page.waitForFunction(() => (document.querySelector("#input") as HTMLTextAreaElement).placeholder === "Message The pair");
  await send(page, "a duet");
  expect([/Scenario: A porch\./.test(lastStream.messages[0].content), await answers(page), await page.evaluate(() => Object.values(JSON.parse(sessionStorage.getItem("chats")!)).some((c: any) => "cast" in c))]).toEqual([true, ["Bea", "Ada"], false]);
  await page.close();
}, 30_000);

test("in a group, a message that names one of them, by @ or by any word of their name that is theirs alone, has that one answer; a message that names nobody has the next in turn answer", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela Bee", { description: "{{char}} keeps bees." }));
  await Bun.write(`${dir}/Tomas.png`, await card("Tomas Bee", { description: "{{char}} sells honey." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mirela.png`, `${dir}/Tomas.png`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 3);
  await page.waitForFunction(() => (document.querySelector("#input") as HTMLTextAreaElement).placeholder === "Message Mirela Bee");
  await inRow(page, "Mirela Bee", ".cast-bring");
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#cast-invite .invite-row")].find((r) => r.querySelector(".invite-name")!.textContent === "Tomas Bee")!.click());
  await page.keyboard.press("Escape");
  await page.waitForSelector("#cast-next .talk");
  const last = () => lastStream.messages.at(-1).content;
  await send(page, "@tomas how is the market?");
  expect(last()).toBe("@tomas how is the market?\n\n[Tomas Bee next.]");
  await send(page, "and the weather, anyone?");
  expect(last()).toBe("and the weather, anyone?\n\n[Mirela Bee next.]");
  await send(page, "what does tomas think, bee or no bee?");
  expect(last()).toBe("what does tomas think, bee or no bee?\n\n[Tomas Bee next.]");
  await send(page, "atomas is not a name");
  expect(last()).toBe("atomas is not a name\n\n[Mirela Bee next.]");
  await page.close();
}, 30_000);

test("every row has the same menu: a character, a group and a folder are pinned, settled, renamed, forked and deleted from it as a chat is; a pinned row goes above the list and a settled one below it, a renamed row keeps its place, a fork is a copy under a new name just below what it copies, a group deleted takes its chats and leaves its people, and the rows keep their order whichever chat was written in last", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", first_mes: "Hello, {{user}}." }));
  await Bun.write(`${dir}/Tomas.png`, await card("Tomas", { description: "{{char}} sells honey." }));
  writeFileSync(`${dir}/Hive.json`, JSON.stringify({ spec: "chrysalis_group", name: "Hive", members: ["Mirela", "Tomas"] }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mirela.png`, `${dir}/Tomas.png`, `${dir}/Hive.json`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 4);
  const zones = () => page.evaluate(() => ["#side-pins", "#side-list", "#side-shelf"].map((z) => [...document.querySelectorAll(`${z} .cast-row .cast-name`)].map((n) => n.textContent)));
  const order = (await zones())[1];
  expect([await rowMenu(page, "Mirela"), await rowMenu(page, "Hive"), await rowMenu(page, "Chrysalis")]).toEqual([
    ["New chat with Mirela", "Pin", "Settle", "Rename", "Fork", "Card", "Delete"], ["New chat with Hive", "Pin", "Settle", "Rename", "Fork", "Card", "Delete"], ["New chat", "Pin", "Settle", "Rename", "Fork", "Card", "Delete"]]);
  await openRow(page, "Tomas");
  await send(page, "good morning mirela");
  await openRow(page, "Hive");
  await send(page, "a toast for the hive");
  await page.evaluate(() => [...document.querySelectorAll("#conversations .folder")].find((f) => f.querySelector(".cast-name")!.textContent === "Hive")!.querySelector("a.conv")!
    .dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true, clientX: 60, clientY: 300 })));
  await page.waitForSelector("#row-menu:not([hidden])");
  expect(await page.$$eval("#row-menu button", (b) => b.map((x) => x.textContent))).toEqual(["New chat with Hive", "Pin", "Settle", "Rename", "Delete"]);
  await page.keyboard.press("Escape");
  expect((await zones())[1]).toEqual(order);

  await rowMenu(page, "Tomas", "Pin");
  expect(await zones()).toEqual([["Tomas"], order.filter((n) => n !== "Tomas"), []]);
  await rowMenu(page, "Hive", "Settle");
  await page.click("#side-shelf .settled-head");
  expect(await zones()).toEqual([["Tomas"], order.filter((n) => n !== "Tomas" && n !== "Hive"), ["Hive"]]);
  expect(await rowMenu(page, "Hive")).toEqual(["New chat with Hive", "Pin", "Un-settle", "Rename", "Fork", "Card", "Delete"]);
  await rowMenu(page, "Hive", "Un-settle");
  await rowMenu(page, "Tomas", "Unpin");
  expect(await zones()).toEqual([[], order, []]);

  await rowMenu(page, "Mirela", "Rename");
  await page.waitForSelector("#conversations .cast-row input.conv-rename");
  await page.keyboard.type("Mira");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => [...document.querySelectorAll("#conversations .cast-name")].some((n) => n.textContent === "Mira"));
  expect((await zones())[1]).toEqual(order.map((n) => (n === "Mirela" ? "Mira" : n)));
  await rowMenu(page, "Mira", "Fork");
  await toasted(page, /^Mira 2 is a copy of Mira$/);
  await rowMenu(page, "Hive", "Fork");
  await toasted(page, /^Hive 2 is a copy of Hive$/);
  expect((await zones())[1]).toEqual(order.flatMap((n) => (n === "Mirela" ? ["Mira", "Mira 2"] : n === "Hive" ? ["Hive", "Hive 2"] : [n])));
  expect(await castRows(page)).toEqual(["Chrysalis", "Hive", "Hive 2", "Mira", "Mira 2", "Tomas"]);
  await openRow(page, "Hive 2");
  await page.waitForSelector("#cast-next .talk");
  expect(await answers(page)).toEqual(["Mira", "Tomas"]);

  const kept = () => page.evaluate(() => Object.values(JSON.parse(sessionStorage.getItem("chats")!)).length);
  expect(await kept()).toBe(2);
  await rowMenu(page, "Hive", "Delete");
  await toasted(page, /^Hive deleted, with 1 chat$/);
  expect([await castRows(page), await kept()]).toEqual([["Chrysalis", "Hive 2", "Mira", "Mira 2", "Tomas"], 1]);
  await page.click("#toast .btn");
  await page.waitForFunction(() => [...document.querySelectorAll("#conversations .cast-name")].some((n) => n.textContent === "Hive"));
  expect(await kept()).toBe(2);
  await rowMenu(page, "Tomas", "Delete");
  await toasted(page, /^Tomas deleted, with 1 chat$/);
  expect([await castRows(page), await kept()]).toEqual([["Chrysalis", "Hive", "Hive 2", "Mira", "Mira 2"], 1]);
  await openRow(page, "Hive");
  await page.waitForSelector("#cast-next .talk");
  expect(await answers(page)).toEqual(["Mira"]);
  await page.close();
}, 60_000);

test("people are dragged like chats: a character onto another makes a group of the two, in the list where it was dropped, or opens the one they already have, onto a group joins it, onto the open chat joins that, or makes a plain chat theirs; a face dragged out of a group's picture leaves it, the last pair splits when the group has no chats and stays when it has; a chat dragged onto a group is that group's", async () => {
  const page = await open();
  for (const [n, d] of [["Mirela", "keeps bees"], ["Tomas", "sells honey"], ["Ida", "paints hives"]]) await Bun.write(`${dir}/${n}.png`, await card(n, { description: `{{char}} ${d}.` }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mirela.png`, `${dir}/Tomas.png`, `${dir}/Ida.png`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 4);
  const people = '#conversations .cast-row[data-person]', member = "#conversations .mosaic > [data-person]";
  const inside = () => page.$$eval("#conversations .folder", (f) => Object.fromEntries(f.filter((x) => x.querySelector(".mosaic")).map((x) => [x.querySelector(".cast-name")!.textContent, [...x.querySelectorAll(".mosaic > [data-person]")].map((m) => (m as HTMLElement).title)])));
  const onChat = () => page.$eval("#thread", (t) => { const r = t.getBoundingClientRect(); return [r.x + r.width / 2, r.y + 200]; });
  await fresh(page);
  expect(await carry(page, people, "Ida", onChat)).toBe("Ida's chat");
  await page.waitForFunction(() => (document.querySelector("#input") as HTMLTextAreaElement).placeholder === "Message Ida");
  expect(await carry(page, people, "Ida", onChat)).toBe("");
  const listed = () => page.$$eval("#side-list .cast-row .cast-name", (n) => n.map((x) => x.textContent));
  const before = await listed();
  expect(await carry(page, people, "Tomas", rowAt(page, "Mirela"))).toBe("Group with Mirela");
  await page.waitForFunction(() => !!document.querySelector("#conversations .cast-pic .mosaic"));
  expect(await listed()).toEqual(before.flatMap((n) => (n === "Mirela" ? ["Mirela and Tomas", n] : [n])));
  expect([await castRows(page), await inside()]).toEqual([["Chrysalis", "Ida", "Mirela", "Mirela and Tomas", "Tomas"], { "Mirela and Tomas": ["Mirela", "Tomas"] }]);
  expect(await carry(page, people, "Ida", rowAt(page, "Mirela and Tomas"))).toBe("Join Mirela and Tomas");
  await page.waitForFunction(() => document.querySelectorAll("#conversations .mosaic > [data-person]").length === 3);
  expect([await castRows(page), await inside(), await carry(page, people, "Ida", rowAt(page, "Mirela, Tomas and Ida"))]).toEqual([["Chrysalis", "Ida", "Mirela", "Mirela, Tomas and Ida", "Tomas"], { "Mirela, Tomas and Ida": ["Mirela", "Tomas", "Ida"] }, ""]);
  expect(await carry(page, member, "Tomas", rowAt(page, "Chrysalis"))).toBe("Take out");
  await page.waitForFunction(() => document.querySelectorAll("#conversations .mosaic > [data-person]").length === 2);
  expect(await inside()).toEqual({ "Mirela and Ida": ["Mirela", "Ida"] });

  await openRow(page, "Tomas");
  await page.waitForFunction(() => (document.querySelector("#input") as HTMLTextAreaElement).placeholder === "Message Tomas");
  await send(page, "good morning mirela");
  await page.waitForNetworkIdle({ idleTime: 200 });
  const chat = "#conversations a.conv";
  const from = await page.$eval(chat, (a) => { const r = a.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; });
  await page.mouse.move(from[0], from[1]);
  await page.mouse.down();
  await page.mouse.move(from[0] + 3, from[1] + 9, { steps: 3 });
  const [x, y] = await rowAt(page, "Mirela and Ida")();
  await page.mouse.move(x, y, { steps: 10 });
  expect(await page.$eval(".drag-ghost .conv-verb", (v) => [v.textContent, (v as HTMLElement).hidden])).toEqual(["Mirela and Ida", false]);
  await page.mouse.up();
  await page.evaluate(() => new Promise((done) => setTimeout(done)));
  await page.waitForSelector("#cast-next .talk");
  expect([await answers(page), await page.$eval("#conversations a.conv", (a) => a.closest(".folder")!.querySelector(".cast-name")!.textContent)]).toEqual([["Mirela", "Ida"], "Mirela and Ida"]);
  expect(await carry(page, member, "Ida", rowAt(page, "Chrysalis"))).toBe("");
  expect(await inside()).toEqual({ "Mirela and Ida": ["Mirela", "Ida"] });
  await rowMenu(page, "Mirela and Ida", "Fork");
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-pic .mosaic").length === 2);
  const copy = await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#conversations .folder")].find((f) => f.querySelector(".cast-name")!.textContent === "Mirela and Ida 2")!.dataset.who!);
  expect(await carry(page, `#conversations .mosaic > [data-group="${copy}"]`, "Ida", rowAt(page, "Chrysalis"))).toBe("Split up");
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-pic .mosaic").length === 1);
  expect(await castRows(page)).toEqual(["Chrysalis", "Ida", "Mirela", "Mirela and Ida", "Tomas"]);
  await openRow(page, "Tomas");
  expect(await carry(page, people, "Ida", rowAt(page, "Mirela"))).toBe("Open Mirela and Ida");
  await page.waitForFunction(() => (document.querySelector("#input") as HTMLTextAreaElement).placeholder === "Message Mirela and Ida");
  expect(await castRows(page)).toEqual(["Chrysalis", "Ida", "Mirela", "Mirela and Ida", "Tomas"]);
  await page.close();
}, 60_000);

test("Chrysalis is one of them: a character dragged onto it makes a group with it, where it answers when named, as itself and under its own face, its picture in the group's card the moving one its own card has; a chat with Chrysalis that has been written in becomes a group when someone is brought in; deleted from its menu, Chrysalis goes with its chats and out of its groups, Undo brings it all back, and with it gone the next plain chat written brings its row back", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mirela.png`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 2);
  const people = '#conversations .cast-row[data-person]';
  expect(await carry(page, people, "Mirela", rowAt(page, "Chrysalis"))).toBe("Group with Chrysalis");
  await page.waitForFunction(() => document.querySelectorAll("#conversations .mosaic > [data-person]").length === 2);
  await openRow(page, "Chrysalis and Mirela");
  await page.waitForSelector("#cast-next .talk");
  expect(await answers(page)).toEqual(["Chrysalis", "Mirela"]);
  await inRow(page, "Chrysalis", ".cast-pic");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-art > :first-child");
  const own = await page.$eval("#cast-sheet .card-art > :first-child", (e) => e.className);
  await rowMenu(page, "Chrysalis and Mirela", "Card");
  await page.waitForSelector("#cast-sheet:not([hidden]) .strips");
  expect([own, await page.$eval("#cast-sheet .strips > :first-child", (e) => e.className)]).toEqual(["card-sea", "card-sea"]);
  await page.keyboard.press("Escape");
  await send(page, "Mirela, a toast for the hive");
  await send(page, "and you, Chrysalis?");
  expect([await speakers(page), lastStream.messages.at(-1).content, /## Chrysalis/.test(lastStream.messages[0].content), /Mirela keeps bees\./.test(lastStream.messages[0].content)])
    .toEqual([["Mirela", "Chrysalis"], "and you, Chrysalis?\n\n[Chrysalis next.]", true, true]);

  await fresh(page);
  await send(page, "one for chrysalis");
  await inRow(page, "Chrysalis", ".cast-bring");
  await page.waitForSelector("#cast-invite:not([hidden])");
  expect(await page.$$eval("#cast-invite .invite-row", (r) => r.map((x) => [x.querySelector(".invite-name")!.textContent, x.getAttribute("aria-pressed") === "true", (x as HTMLButtonElement).disabled])))
    .toEqual([["Chrysalis", true, true], ["Mirela", false, false]]);
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#cast-invite .invite-row")].find((r) => r.querySelector(".invite-name")!.textContent === "Mirela")!.click());
  await page.keyboard.press("Escape");
  await page.waitForSelector("#cast-next .talk");
  expect([await answers(page), await speakers(page), await page.$eval("#conversations a.conv.active", (a) => a.closest(".folder")!.querySelector(".cast-name")!.textContent)]).toEqual([["Chrysalis", "Mirela"], ["Chrysalis"], "Chrysalis and Mirela"]);

  await fresh(page);
  await send(page, "another for chrysalis");
  const kept = () => page.evaluate(() => Object.values(JSON.parse(sessionStorage.getItem("chats")!)).length);
  expect([await castRows(page), await kept()]).toEqual([["Chrysalis", "Chrysalis and Mirela", "Mirela"], 3]);
  await rowMenu(page, "Chrysalis", "Delete");
  await toasted(page, /^Chrysalis deleted, with 1 chat$/);
  expect([await castRows(page), await kept(), await page.$$eval("#conversations .mosaic > [data-person]", (m) => m.map((x) => (x as HTMLElement).title)), await page.$$eval("#thread > article", (a) => a.length)])
    .toEqual([["Mirela", "Mirela"], 2, ["Mirela"], 0]);
  await page.click("#toast .btn");
  await page.waitForFunction(() => document.querySelectorAll("#conversations .mosaic > [data-person]").length === 2);
  expect([await castRows(page), await kept()]).toEqual([["Chrysalis", "Chrysalis and Mirela", "Mirela"], 3]);
  await rowMenu(page, "Chrysalis", "Delete");
  await toasted(page, /^Chrysalis deleted, with 1 chat$/);
  await send(page, "is anyone there");
  expect([await castRows(page), await speakers(page), lastStream.messages.at(-1).content]).toEqual([["Chrysalis", "Mirela", "Mirela"], ["Chrysalis"], "is anyone there"]);
  await page.close();
}, 60_000);

test("the sidebar's add makes a new character here, with no file: their card opens with the name to type and no row yet, the row is there once something is typed, in a colour that typing does not change and the card's colour picker does, the button again closes an empty card, what is typed is theirs in their chat and after a reload, and one left wholly empty leaves nothing; there is no menu behind the button", async () => {
  const page = await open();
  await page.click("#add-cast");
  await page.waitForSelector("#cast-sheet:not([hidden]) input.card-name");
  const state = async () => [await page.evaluate(() => document.activeElement === document.querySelector("#cast-sheet input.card-name")), await castRows(page), await page.$eval("#cast-sheet", (s) => (s as HTMLElement).hidden)];
  expect([await state(), await page.$("#cast-add")]).toEqual([[true, ["Chrysalis"], false], null]);
  await page.click("#add-cast");
  expect(await state()).toEqual([false, ["Chrysalis"], true]);
  await page.click("#add-cast");
  await page.waitForSelector("#cast-sheet:not([hidden]) input.card-name");
  await page.keyboard.type("W");
  expect(await state()).toEqual([true, ["Chrysalis", "W"], false]);
  await page.keyboard.press("Backspace");
  expect(await state()).toEqual([true, ["Chrysalis"], false]);
  await page.keyboard.press("Escape");
  expect(await state()).toEqual([false, ["Chrysalis"], true]);

  await page.click("#add-cast");
  await page.waitForSelector("#cast-sheet:not([hidden]) input.card-name");
  const tint = (at: string) => page.$eval(at, (f) => (f as HTMLElement).style.getPropertyValue("--tint"));
  await page.keyboard.type("W");
  const first = await tint("#cast-sheet .card-portrait");
  await page.keyboard.type("ren");
  expect([await page.$eval("#cast-sheet .card-portrait", (f) => f.textContent), /^#[0-9a-f]{6}$/.test(first), await tint("#cast-sheet .card-portrait"), await tint("#conversations .cast-pic .face"),
    await page.$eval("#cast-sheet input.card-tint", (b) => [(b as HTMLInputElement).type, (b as HTMLInputElement).value])]).toEqual(["W", true, first, first, ["color", first]]);
  const other = first === "#3366cc" ? "#cc6633" : "#3366cc";
  await page.$eval("#cast-sheet input.card-tint", (b, v) => { (b as HTMLInputElement).value = v; b.dispatchEvent(new Event("input", { bubbles: true })); b.dispatchEvent(new Event("change", { bubbles: true })); }, other);
  expect([await tint("#cast-sheet .card-portrait"), await tint("#conversations .cast-pic .face")]).toEqual([other, other]);
  for (const [label, text] of [["Description", "{{char}} mends clocks for {{user}}."], ["Greeting", "Tick, {{user}}."]]) {
    await page.evaluate((label) => [...document.querySelectorAll("#cast-sheet .card-field")].find((x) => x.querySelector(".card-label")!.textContent === label)!.querySelector("textarea")!.focus(), label);
    await page.keyboard.type(text);
  }
  await page.keyboard.press("Escape");
  expect(await castRows(page)).toEqual(["Chrysalis", "Wren"]);
  await inRow(page, "Wren", ".cast-new");
  await page.waitForFunction(() => document.querySelector("#thread > article.assistant .markdown")?.textContent?.trim() === "Tick, User.");
  await send(page, "what time is it");
  expect([lastStream.messages[0].content, await speakers(page)]).toEqual(["Write Wren's next reply in a fictional chat between Wren and User.\n\nWren mends clocks for User.", ["WWren", "WWren"]]);
  await page.reload({ waitUntil: "load" });
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 2);
  expect([await castRows(page), await tint("#conversations .cast-pic .face.initials")]).toEqual([["Chrysalis", "Wren"], other]);
  await page.close();
}, 30_000);
