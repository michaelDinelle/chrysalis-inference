import { expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import type { Browser } from "puppeteer-core";
import { harness, RUN, articles, card, castRows, dir, download, inRow, lastStream, look, open, openRow, pagerOf, pickVoice, png, pngText, send, sent, shown, streams, untilIdle, voiceLog } from "./browser-harness";

harness();

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
  expect((await fields()).slice(0, 4)).toEqual([["Description", "{{char}} keeps bees for {{user}}."], ["Personality", "calm"], ["Scenario", "A hillside."], ["Greeting", "Hello, {{user}}."]]);
  await page.click("#cast-sheet .card-field:nth-of-type(2) textarea");
  await page.keyboard.press("End");
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
    "\n\nScenario: A hillside.\n\nHow Mirela talks:\nRobin: Why bees?\nMirela: They listen.");
  expect(one.slice(1)).toEqual([{ role: "user", content: "[The scene begins.]" }, { role: "assistant", content: "Hello, User." },
    { role: "user", content: "good morning mirela\n\n[Stay in character.]" }]);
  expect(await page.$$eval("#thread .msg.assistant .speaker", (s) => s.map((x) => x.textContent))).toEqual(["Mirela", "Mirela"]);
  await page.$eval('#thread > article[data-i="2"] [data-act="cont"]', (b) => (b as HTMLElement).click());
  await page.waitForFunction(() => /Good morning\..*Good morning\./.test(document.querySelector('#thread > article[data-i="2"] .markdown')?.textContent ?? ""));
  await untilIdle(page);
  expect([lastStream.continue_final_message, await page.$eval('#thread > article[data-i="2"] .thinking', (t) => (t as HTMLElement).hidden)]).toEqual([true, true]);
  await page.close();
}, 40_000);

test("the user is a character like any other: a portrait opens its card and closes it again (another's switches to theirs); the user's card has every field a character's has, sent as theirs are, their greeting the first message of a new chat (in the box), their voice the one their messages are read in; their messages and the replies alike in bubbles, each from its writer's side", async () => {
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
  for (const [label, text] of [["Personality", "curious"], ["Greeting", "Hi, {{char}}."], ["How they talk", "{{user}}: Hm."], ["After the chat", "Keep it short."]]) {
    await page.evaluate((label) => {
      const f = [...document.querySelectorAll("#cast-sheet .card-field")].find((x) => x.querySelector(".card-label")!.textContent === label)!;
      f.closest("details")?.setAttribute("open", "");
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
    "User's personality: curious\n\nHow User talks:\nUser: Hm.");
  expect(lastStream.messages.at(-1)).toEqual({ role: "user", content: "Hi, Mirela.\n\n[Keep it short.]" });
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

test("a card's portrait is the card's file, named for them and as it is now: a click saves it, the PNG any app reads with the picture as given", async () => {
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
  await page.click("#cast-sheet a.card-file");
  for (const t = Date.now(); !existsSync(`${saved}/Mirela.png`); await Bun.sleep(50)) if (Date.now() - t > 10_000) throw new Error(`not saved: ${readdirSync(saved)}`);
  await Bun.sleep(200);
  const got = pngText(readFileSync(`${saved}/Mirela.png`));
  const v3 = JSON.parse(Buffer.from(got.text.ccv3 ?? "", "base64").toString()), v2 = JSON.parse(Buffer.from(got.text.chara ?? "", "base64").toString());
  expect([readdirSync(saved), v3.data.name, v3.data.description, v2.data.description]).toEqual([["Mirela.png"], "Mirela", "{{char}} keeps bees. Twelve hives.", "{{char}} keeps bees. Twelve hives."]);
  expect(got.idat).toBe(pngText(png).idat);
  expect(await page.$eval("#cast-sheet a.card-file", (a) => [(a as HTMLAnchorElement).download, a.querySelector("img")!.src === (a as HTMLAnchorElement).href]))
    .toEqual(["Mirela.png", true]);
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

test("a character deleted takes their chats with them (none become Chrysalis's), Undo brings it all back, and a character deleted in another tab takes their chats here too", async () => {
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
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-acts .icon");
  await page.click("#cast-sheet .card-acts .icon");
  expect(await page.$eval("#cast-sheet .card-acts .icon", (b) => b.textContent)).toBe("Delete, with 1 chat?");
  await page.click("#cast-sheet .card-acts .icon");
  await page.waitForFunction(() => /^Mirela deleted, with 1 chat/.test(document.querySelector("#toast")?.textContent ?? ""));
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

test("a group whose file names characters with no card here yet shows them all (their initials, in its face and on its card) and takes each in as their card comes", async () => {
  const page = await open();
  await Bun.write(`${dir}/Ada.png`, await card("Ada", { description: "{{char}} hums." }));
  await Bun.write(`${dir}/Bea.png`, await card("Bea", { description: "{{char}} drums." }));
  writeFileSync(`${dir}/Duo.json`, JSON.stringify({ spec: "chrysalis_group", name: "Duo", members: ["Ada", "Bea"] }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Ada.png`, `${dir}/Duo.json`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .folder").length === 3);
  const tiles = () => page.evaluate(() => [...[...document.querySelectorAll("#conversations .cast-row")].find((r) => r.querySelector(".cast-name")!.textContent === "Duo")!
    .querySelectorAll(".face.mosaic > *")].map((f) => f.tagName === "IMG" ? "picture" : f.textContent));
  expect(await tiles()).toEqual(["picture", "B"]);
  await inRow(page, "Duo", ".cast-pic");
  await page.waitForSelector("#cast-sheet:not([hidden]) .card-member.waiting");
  expect(await page.$$eval("#cast-sheet .strips > *", (s) => s.length)).toBe(2);
  await page.keyboard.press("Escape");
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Bea.png`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .folder").length === 4);
  expect(await tiles()).toEqual(["picture", "picture"]);
  await openRow(page, "Duo");
  await page.waitForSelector("#cast-next .talk");
  expect(await page.$$eval("#cast-next button[title$=' next']", (b) => b.map((x) => (x as HTMLElement).title))).toEqual(["Ada next", "Bea next"]);
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#conversations .folder");
  expect(await tiles()).toEqual(["picture", "picture"]);
  await page.close();
}, 30_000);

test("a group's replies are the scene, each paragraph its speaker's and one going on under the same face; a face has the next reply be theirs, thought through first; let them talk takes turns; cards dropped on a chat join it once (a chat with one character becomes a group, her replies hers); the cast stays after a reload", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", scenario: "A hillside.", first_mes: "Hello, {{user}}." }));
  await Bun.write(`${dir}/Tomas.png`, await card("Tomas", { description: "{{char}} sells honey.", first_mes: "Honey?" }));
  await Bun.write(`${dir}/Ida.png`, await card("Ida", { description: "{{char}} paints hives." }));
  writeFileSync(`${dir}/Hive.json`, JSON.stringify({ spec: "chrysalis_group", name: "Hive", members: ["Mirela", "Tomas"], scenario: "Market day." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mirela.png`, `${dir}/Tomas.png`, `${dir}/Hive.json`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 4);
  await openRow(page, "Hive");
  await page.waitForSelector("#cast-next .talk");
  await send(page, "a toast for the hive");
  const lines = (i: number) => page.$$eval(`#thread > article[data-i="${i}"] p.line`, (p) => p.map((x) => [x.querySelector(".who")?.textContent ?? "", x.classList.contains("same")]));
  expect(await lines(1)).toEqual([["Mirela", false], ["Tomas", false]]);
  expect(await page.$$eval("#thread p.line", (p) => p.map((x) => x.textContent))).toEqual(["MirelaShe raises a cup. To the bees.", "TomasTo the bees!"]);
  expect([lastStream.stop, lastStream.messages[0].content.includes("Scenario: Market day."), lastStream.messages[0].content.includes("A hillside.")])
    .toEqual([["\nUser:", "\n**User"], true, false]);
  await send(page, "the same speaker");
  expect(await lines(3)).toEqual([["Mirela", false], ["", true], ["", true], ["Tomas", false]]);

  await page.click('#cast-next button[title="Tomas next"]');
  await untilIdle(page);
  expect([lastStream.continue_final_message, lastStream.reasoning_effort !== undefined, lastStream.messages.at(-1)])
    .toEqual([undefined, true, { role: "user", content: "[Tomas next.]" }]);
  expect([await lines(4), await page.$eval('#thread > article[data-i="4"] .thought', (t) => t.textContent)]).toEqual([[["Tomas", false]], "Tomas has the floor."]);
  const n0 = streams.length;
  await page.click("#cast-next .talk");
  await page.waitForFunction(() => !!document.querySelector("#cast-next .talk:not(.on)")
    && document.querySelector("#send")?.getAttribute("aria-label") === "Send", { timeout: 20_000 });
  expect(streams.slice(n0).map((b) => b.messages.at(-1))).toEqual(["Mirela", "Tomas", "Mirela", "Tomas"].map((n) => ({ role: "user", content: `[${n} next.]` })));

  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Ida.png`);
  await page.waitForFunction(() => document.querySelectorAll("#cast-next button[title$=' next']").length === 3);
  await (await page.$("input#attach-file"))!.uploadFile(`${dir}/Mirela.png`);
  await page.waitForFunction(() => /Mirela is in this chat already/.test(document.querySelector("#toast")?.textContent ?? ""));
  expect([await castRows(page), (await page.$$("#attachments .att")).length]).toEqual([["Chrysalis", "Hive", "Ida", "Mirela", "Tomas"], 0]);
  await openRow(page, "Mirela");
  await page.waitForFunction(() => document.querySelector("#thread .speaker")?.textContent === "Mirela");
  await page.evaluate(() => {
    const row = [...document.querySelectorAll("#conversations .cast-row")].find((r) => r.querySelector(".cast-name")!.textContent === "Tomas")!;
    const dt = new DataTransfer(), at = document.getElementById("thread")!;
    row.dispatchEvent(new DragEvent("dragstart", { dataTransfer: dt, bubbles: true }));
    for (const type of ["dragenter", "dragover", "drop"]) at.dispatchEvent(new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true }));
  });
  await page.waitForSelector("#cast-next .talk");
  expect([await castRows(page), await page.$$eval("#cast-next button[title$=' next']", (b) => b.map((x) => (x as HTMLElement).title)), await lines(0)])
    .toEqual([["Chrysalis", "Hive", "Ida", "Mirela", "Mirela and Tomas", "Tomas"], ["Mirela next", "Tomas next"], [[`Mirela`, false]]]);

  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#conversations .cast-row");
  expect(await castRows(page)).toEqual(["Chrysalis", "Hive", "Ida", "Mirela", "Mirela and Tomas", "Tomas"]);
  await page.close();
}, 60_000);

test("a chat's own people: the invite on its row in the sidebar brings characters in and takes them out with Undo (the last one stays, the chat hers again), on another's row it starts a new chat with them, the sidebar's add starts one from nobody, a new chat on their row has the same people, and one chat's change leaves the other's as they were, sent as such and after a reload", async () => {
  const page = await open();
  await Bun.write(`${dir}/Mirela.png`, await card("Mirela", { description: "{{char}} keeps bees.", first_mes: "Hello, {{user}}." }));
  await Bun.write(`${dir}/Tomas.png`, await card("Tomas", { description: "{{char}} sells honey.", first_mes: "Honey?" }));
  await Bun.write(`${dir}/Ida.png`, await card("Ida", { description: "{{char}} paints hives." }));
  await (await page.$("input#cast-file"))!.uploadFile(`${dir}/Mirela.png`, `${dir}/Tomas.png`, `${dir}/Ida.png`);
  await page.waitForFunction(() => document.querySelectorAll("#conversations .cast-row").length === 4);
  await page.waitForFunction(() => document.querySelector("#thread .speaker")?.textContent === "Mirela");
  const next = () => page.$$eval("#cast-next button[title$=' next']", (b) => b.map((x) => (x as HTMLElement).title.replace(" next", "")));
  const rows = () => page.$$eval("#cast-invite .invite-row", (r) => r.map((x) => [x.querySelector(".invite-name")!.textContent, x.getAttribute("aria-pressed") === "true", (x as HTMLButtonElement).disabled]));
  const pick = (name: string) => page.evaluate((name) => [...document.querySelectorAll<HTMLElement>("#cast-invite .invite-row")]
    .find((r) => r.querySelector(".invite-name")!.textContent === name)!.click(), name);
  expect(await page.$eval("#cast-next", (b) => (b as HTMLElement).hidden)).toBe(true);
  await inRow(page, "Mirela", ".cast-bring");
  await page.waitForSelector("#cast-invite:not([hidden])");
  expect(await rows()).toEqual([["Ida", false, false], ["Mirela", true, true], ["Tomas", false, false]]);
  await pick("Tomas");
  await pick("Ida");
  expect([await next(), await rows(), await castRows(page), await page.$eval("#input", (e) => (e as HTMLTextAreaElement).placeholder)])
    .toEqual([["Mirela", "Tomas", "Ida"], [["Ida", true, false], ["Mirela", true, false], ["Tomas", true, false]],
      ["Chrysalis", "Ida", "Mirela", "Mirela, Tomas and Ida", "Tomas"], "Message Mirela, Tomas and Ida"]);
  await pick("Tomas");
  expect([await next(), await page.$eval("#toast span", (t) => t.textContent)]).toEqual([["Mirela", "Ida"], "Tomas left"]);
  await page.click("#toast button");
  expect(await next()).toEqual(["Mirela", "Tomas", "Ida"]);
  await pick("Tomas");
  await pick("Ida");
  expect([await next(), await rows(), await castRows(page), await page.$("#cast-next .talk"), await page.$eval("#thread .speaker", (s) => s.textContent)])
    .toEqual([[], [["Ida", false, false], ["Mirela", true, true], ["Tomas", false, false]], ["Chrysalis", "Ida", "Mirela", "Tomas"], null, "Mirela"]);
  await page.keyboard.press("Escape");
  expect(await page.$eval("#cast-invite", (e) => (e as HTMLElement).hidden)).toBe(true);

  await page.click("#add-cast");
  await page.click('#cast-add [data-add="group"]');
  await page.waitForSelector("#cast-invite:not([hidden])");
  expect([await rows(), await castRows(page)]).toEqual([[["Ida", false, false], ["Mirela", false, false], ["Tomas", false, false]], ["Chrysalis", "Ida", "Mirela", "Tomas"]]);
  await pick("Tomas");
  await page.waitForFunction(() => document.querySelector("#thread .speaker")?.textContent === "Tomas");
  await pick("Ida");
  expect([await next(), await castRows(page)]).toEqual([["Tomas", "Ida"], ["Chrysalis", "Ida", "Mirela", "Tomas", "Tomas and Ida"]]);
  await page.keyboard.press("Escape");
  await send(page, "a toast for the hive");
  const system = () => [/Tomas sells honey\./, /Ida paints hives\./, /Mirela keeps bees\./].map((r) => r.test(lastStream.messages[0].content));
  expect(system()).toEqual([true, true, false]);

  await inRow(page, "Tomas and Ida", ".cast-new");
  await page.waitForFunction(() => !document.querySelector("#thread > article"));
  expect(await next()).toEqual(["Tomas", "Ida"]);
  await inRow(page, "Tomas and Ida", ".cast-bring");
  await pick("Mirela");
  await page.keyboard.press("Escape");
  await send(page, "a toast for the hive");
  expect([system(), await castRows(page)]).toEqual([[true, true, true], ["Chrysalis", "Ida", "Mirela", "Tomas", "Tomas and Ida", "Tomas, Ida and Mirela"]]);
  await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#conversations .folder")]
    .find((f) => f.querySelector(".cast-name")!.textContent === "Tomas and Ida")!.querySelector<HTMLElement>("a.conv")!.click());
  await page.waitForFunction(() => document.querySelectorAll("#cast-next button[title$=' next']").length === 2);
  expect(await next()).toEqual(["Tomas", "Ida"]);
  await inRow(page, "Mirela", ".cast-bring");
  expect([await page.$eval("#cast-invite .invite-head", (e) => e.textContent), await rows(), await next()])
    .toEqual(["A new chat with Mirela and", [["Ida", false, false], ["Mirela", true, true], ["Tomas", false, false]], ["Tomas", "Ida"]]);
  await pick("Ida");
  const all = ["Chrysalis", "Ida", "Mirela", "Mirela and Ida", "Tomas", "Tomas and Ida", "Tomas, Ida and Mirela"];
  expect([await next(), await castRows(page), await page.$$eval("#thread > article", (a) => a.length)]).toEqual([["Mirela", "Ida"], all, 1]);
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#cast-next .talk");
  expect([await next(), await castRows(page)]).toEqual([["Mirela", "Ida"], all]);
  await page.close();
}, 60_000);

test("a group saved by an earlier version becomes its chats' own: each chat filed under it keeps its people, its name and its scenario, and a member since deleted is left out", async () => {
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
    const chats = JSON.parse(sessionStorage.getItem("chats")!);
    for (const c of Object.values(chats) as { who: string }[]) c.who = "old-group";
    sessionStorage.setItem("chats", JSON.stringify(chats));
  });
  await page.reload({ waitUntil: "load" });
  await page.waitForSelector("#cast-next .talk");
  expect([await castRows(page), await page.$$eval("#cast-next button[title$=' next']", (b) => b.map((x) => (x as HTMLElement).title))])
    .toEqual([["Ada", "Bea", "Chrysalis", "The band"], ["Ada next", "Bea next"]]);
  await send(page, "a duet");
  expect([/Scenario: A cellar\./, /Ada hums\./, /Bea drums\./].map((r) => r.test(lastStream.messages[0].content))).toEqual([true, true, true]);
  await page.close();
}, 30_000);
