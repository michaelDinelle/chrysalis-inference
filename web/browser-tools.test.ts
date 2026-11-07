import { expect, test } from "bun:test";
import { readFileSync, readdirSync, unlinkSync, writeFileSync } from "node:fs";
import type { Browser, Page } from "puppeteer-core";
import { ARTIFACTS_PROMPT } from "./artifacts";
import { harness, LOG, badge, base, card, checked, dir, fake, idle, lastStream, look, open, png, pop, send, sent, streams, subOf, sw, toolsOn, web } from "./browser-harness";

harness();

test("attached files go to Chrysalis as images, PDFs, videos and fenced text, stay after a reload, and travel in an export", async () => {
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
  await page.click("#export-all");
  const file = await (async () => { for (let k = 0; k < 50; k++) { const f = readdirSync(dir).find((x) => x.startsWith("chrysalis-chats-")); if (f) return f; await Bun.sleep(100); } })();
  const other = await open();
  await (await other.$("input#import-file"))!.uploadFile(`${dir}/${file}`);
  await other.waitForFunction(() => (document.querySelector(".msg-files img") as HTMLImageElement | null)?.naturalWidth === 2);
  await send(other, "and once more");
  expect(lastStream.messages.find((m: any) => m.role === "user").content).toEqual(content);
  expect(JSON.parse(readFileSync(`${dir}/${file}`, "utf8")).conversations[0].messages.length).toBe(4);
  unlinkSync(`${dir}/${file}`);
  await page.close();
  await other.close();
}, 60_000);

test("the Tools popover turns servers and single tools on and off for this chat, and each chat keeps its own", async () => {
  const page = await open();
  expect(await badge(page)).toEqual(["", "Tools: off in this chat"]);
  await page.click("#tools-btn");
  await page.waitForSelector(sw("mcp:fake"));
  const rows = await page.$$eval(`${pop} .tp-src`, (r) => r.map((x) => [x.querySelector(".tp-name")!.textContent, x.querySelector(".tp-sub")!.textContent]));
  expect(rows).toEqual([["Fetch URL", "Reads web pages at public addresses"], ["Sub-agents", "Hands tasks to agents that work side by side"],
    ["fake", "Not started"], ["broken", "Not started"]]);
  await page.click(sw("mcp:fake"));
  await page.waitForFunction(() => document.querySelector('#tools-pop [data-src="mcp:fake"]')?.closest(".tp-src")?.querySelector(".tp-sub")?.textContent === "5 of 5 tools on",
    { timeout: 20_000 });
  expect(await badge(page)).toEqual(["5", "Tools: 5 on in this chat"]);
  await page.click(`${pop} [data-do="more"][data-src="mcp:fake"]`);
  const tools = await page.$$eval(`${pop} .tp-tool`, (r) => r.map((x) => [x.querySelector(".tp-tname")!.textContent, x.querySelector(".tp-tag")!.textContent]));
  expect(tools).toEqual([["big", "read-only"], ["Echo", "read-only"], ["get.weather", "read-only"], ["slow", "read-only"], ["write_note", "asks first"]]);
  await page.click(sw("mcp:fake", "write_note"));
  expect(await checked(page, sw("mcp:fake", "write_note"))).toBe("false");
  expect(await subOf(page, "mcp:fake")).toBe("4 of 5 tools on");
  await page.click(sw("fetch"));
  expect(await badge(page)).toEqual(["5", "Tools: 5 on in this chat"]);
  const pageLength = () => page.$$eval(`${pop} .tp-src`, (r) => r.map((x) => x.querySelector(".knob label")?.textContent ?? "").filter(Boolean));
  expect(await pageLength()).toEqual(["Page length"]);
  await page.click(sw("fetch"));
  expect(await pageLength()).toEqual([]);
  await page.click(sw("fetch"));
  await page.click(sw("mcp:broken"));
  await page.waitForSelector(`${pop} .tp-err [data-do="retry"][data-src="mcp:broken"]`, { timeout: 20_000 });
  expect(await page.$eval(`${pop} .tp-err`, (e) => e.textContent)).toMatch(/ENOENT|nonexistent|not found/i);
  await page.click(sw("mcp:broken"));
  await page.keyboard.press("Escape");
  expect(await page.$eval(pop, (p) => (p as HTMLElement).hidden)).toBe(true);
  expect(await page.evaluate(() => document.activeElement?.id)).toBe("tools-btn");

  await send(page, "hi");
  expect(lastStream.tools.map((t: any) => t.function.name)).toEqual(["fake__big", "fake__echo", "fake__get_weather", "fake__slow", "fetch_url"]);
  expect(lastStream.messages[0].content).toBe("You are Qwen3.8-27B, running locally in Chrysalis Inference. The tools offered here are the only ones in this chat, " +
    `and there are no files or projects to explore. Use a tool when it helps; otherwise answer directly.\n\n${ARTIFACTS_PROMPT}`);
  const first = await page.evaluate(() => location.hash);

  await page.click("#new-chat");
  expect(await badge(page)).toEqual(["5", "Tools: 5 on in this chat"]);
  await page.click("#tools-btn");
  await page.waitForSelector(sw("fetch"));
  await page.click(sw("fetch"));
  await page.click(sw("mcp:fake"));
  await page.keyboard.press("Escape");
  expect(await badge(page)).toEqual(["", "Tools: off in this chat"]);
  await send(page, "no tools");
  expect(lastStream.tools).toBeUndefined();
  expect(lastStream.messages[0].content).toBe(`You are Qwen3.8-27B, running locally in Chrysalis Inference. There are no tools or files in this chat. Answer directly.\n\n${ARTIFACTS_PROMPT}`);

  await idle(page, page.goto(`${base}/${first}`));
  await page.waitForFunction(() => document.querySelector("#tools-btn .tools-n")?.textContent === "5", { timeout: 10_000 });
  await page.close();
}, 60_000);

test("a tool call is a card where it happened; it opens to its arguments and result, and the history goes back byte for byte", async () => {
  const page = await open();
  await toolsOn(page);
  await send(page, "tool echo");
  const shape = await page.$eval(".msg.assistant", (a) => ({
    order: [...a.querySelectorAll(".round > .markdown, .round .tool-card, :scope > .markdown")].map((x) => x.classList.contains("tool-card") ? "card" : x.textContent!.trim()),
    card: [".tool-name", ".tool-src", ".tool-args", ".tool-state"].map((s) => a.querySelector(`.tool-card ${s}`)!.textContent),
    open: (a.querySelector(".tool-body") as HTMLElement).hidden,
  }));
  expect(shape.order).toEqual(["Let me check.", "card", "The tool said: echo: hello"]);
  expect(shape.card.slice(0, 3)).toEqual(["Echo", "fake", "hello"]);
  expect(shape.card[3]).toMatch(/^done in \d/);
  expect(shape.open).toBe(true);
  await page.click(".tool-card .tool-head");
  const body = await page.$eval(".tool-card .tool-body", (b) => ({ hidden: (b as HTMLElement).hidden, text: [...b.children].map((x) => x.textContent!.trim()) }));
  expect(body.hidden).toBe(false);
  expect(body.text).toEqual(["Arguments", '{\n  "text": "hello"\n}', "Result", "echo: hello"]);
  expect(await page.$eval(".tool-card .tool-head", (b) => b.getAttribute("aria-expanded"))).toBe("true");

  const round1 = streams.at(-1);
  await send(page, "and then");
  const sent = lastStream.messages;
  expect(JSON.stringify(sent.slice(0, round1.messages.length))).toBe(JSON.stringify(round1.messages));
  expect(sent.slice(round1.messages.length)).toEqual([{ role: "assistant", content: "The tool said: echo: hello" }, { role: "user", content: "and then" }]);
  expect(round1.messages.at(-2).tool_calls[0].function.arguments).toBe('{"text":  "hello"}');
  await page.close();
}, 60_000);

test("a tool that asks first shows the call with Allow once, Always allow and Deny, and does as answered", async () => {
  writeFileSync(LOG, "");
  const page = await open();
  await toolsOn(page);
  const answer = async (button: string) => {
    const before = await page.$$eval(".msg .stats", (s) => s.length);
    await page.type("#input", "tool note");
    await page.keyboard.press("Enter");
    await page.waitForSelector('.tool-card[data-state="approval"] .tool-ask:not([hidden])', { timeout: 20_000 });
    const ask = await page.$eval('.tool-card[data-state="approval"]', (c) => ({ args: c.querySelector(".tool-ask pre")!.textContent,
      buttons: [...c.querySelectorAll(".tool-ask button")].map((b) => b.textContent), state: c.querySelector(".tool-state")!.textContent }));
    expect(ask).toEqual({ args: '{\n  "text": "n1"\n}', buttons: ["Deny", "Always allow", "Allow once"], state: "asks to run" });
    await page.click(`.tool-card[data-state="approval"] [data-decision="${button}"]`);
    await page.waitForFunction((n) => document.querySelectorAll(".msg .stats").length > n, { timeout: 20_000 }, before);
    return page.$$eval(".msg.assistant .tool-card", (c) => c.at(-1)!.getAttribute("data-state"));
  };
  expect(await answer("once")).toBe("done");
  expect(readFileSync(LOG, "utf8").match(/wrote n1/g)?.length).toBe(1);
  expect(await answer("deny")).toBe("declined");
  expect(streams.at(-1).messages.at(-1).content).toContain("The user declined this call");
  expect(readFileSync(LOG, "utf8").match(/wrote n1/g)?.length).toBe(1);
  expect(await answer("always")).toBe("done");
  await send(page, "tool note");
  expect(await page.$$eval(".msg.assistant .tool-card", (c) => c.at(-1)!.getAttribute("data-state"))).toBe("done");
  expect(readFileSync(LOG, "utf8").match(/wrote n1/g)?.length).toBe(3);
  await page.click("#tools-btn");
  await page.waitForSelector("#tools-pop .tp-allow .chip");
  expect(await page.$$eval("#tools-pop .tp-allow .chip", (c) => c.map((x) => x.textContent))).toEqual(["fake__write_note"]);
  await page.click("#tools-pop .tp-allow .chip");
  await page.waitForFunction(() => !document.querySelector("#tools-pop .tp-allow"));
  await page.keyboard.press("Escape");
  await page.type("#input", "tool note");
  await page.keyboard.press("Enter");
  await page.waitForSelector('.tool-card[data-state="approval"] .tool-ask:not([hidden])', { timeout: 20_000 });
  await page.close();
}, 90_000);

test("sub-agents: two tasks handed off at once get a card each that says what it is doing, a call that asks first asks in its card, one can be stopped alone, and only their answers go back to the reply", async () => {
  writeFileSync(LOG, "");
  const page = await open();
  await toolsOn(page);
  await page.click("#tools-btn");
  await page.waitForSelector(sw("agents"));
  await page.click(sw("agents"));
  expect(await checked(page, sw("agents"))).toBe("true");
  await page.keyboard.press("Escape");
  const cards = () => page.evaluate(() => [...[...document.querySelectorAll(".msg.assistant")].at(-1)!.querySelectorAll(".agent-card")].map((c) => ({
    label: c.querySelector(".tool-name")!.textContent, now: c.querySelector(".agent-now")!.textContent, meta: c.querySelector(".agent-meta")!.textContent,
    state: c.getAttribute("data-state") })));
  const ask = async () => {
    await page.type("#input", "agents");
    await page.keyboard.press("Enter");
    await page.waitForSelector('.agent-card .tool-card[data-state="approval"] .tool-ask:not([hidden])', { timeout: 20_000 });
  };
  const done = (n: number) => page.waitForFunction((k) => document.querySelectorAll(".msg .stats").length > k, { timeout: 20_000 }, n);
  const results = () => { const m = streams.at(-1).messages; return m.slice(m.findLastIndex((x: any) => x.role === "user")).filter((x: any) => x.role === "tool").map((x: any) => x.content); };

  await ask();
  const asking = await cards();
  expect(asking.map((c) => c.label)).toEqual(["Echo a word", "Write a note"]);
  expect(await page.evaluate(() => {
    const t = [...[...document.querySelectorAll(".msg.assistant")].at(-1)!.querySelectorAll("details.thinking")].find((d) => !d.closest(".round, .agent-card"))!;
    return [t.querySelector("summary")!.textContent, t.classList.contains("live")];
  })).toEqual(["Thoughts", false]);
  expect(asking[1]).toMatchObject({ state: "running" });
  expect(asking[1].now).toStartWith("Asks to run ");
  await page.click('.agent-card:has(.tool-card[data-state="approval"]) .agent-stop:not([hidden])');
  await done(0);
  expect((await cards()).map((c) => [c.state, c.now])).toEqual([["done", "The tool said: echo: hello"], ["stopped", "Stopped before it finished"]]);
  const [echoed, stopped] = results();
  expect(echoed).toBe("The tool said: echo: hello");
  expect(stopped).toStartWith("Stopped before it finished: the user stopped it.");
  expect(readFileSync(LOG, "utf8")).not.toContain("wrote n1");

  const before = await page.$$eval(".msg .stats", (s) => s.length);
  await ask();
  await page.click('.agent-card .tool-card[data-state="approval"] [data-decision="once"]');
  await done(before);
  const end = await cards();
  expect(end.map((c) => c.state)).toEqual(["done", "done"]);
  expect(end.map((c) => c.meta)).toEqual(["1 tool · 40 tokens", "1 tool · 40 tokens"]);
  expect(readFileSync(LOG, "utf8").match(/wrote n1/g)?.length).toBe(1);
  expect(results()).toEqual(end.map((c) => c.now));
  const subs = streams.filter((b) => /^You are a sub-agent/.test(b.messages[0].content)).slice(-4);
  expect(subs.map((b) => [b.messages[1].content, b.tools.map((t: any) => t.function.name).includes("subagent"), "logprobs" in b]).sort()).toEqual([
    ["tool echo", false, false], ["tool echo", false, false], ["tool note", false, false], ["tool note", false, false]]);
  const last = await page.evaluate(() => (([...document.querySelectorAll(".msg.assistant")].at(-1) as HTMLElement).querySelector(":scope > .markdown") as HTMLElement).innerText);
  expect(last).toBe(`The sub-agents said: ${results().join(" | ")}`);

  await page.evaluate(() => ([...[...document.querySelectorAll(".msg.assistant")].at(-1)!.querySelectorAll(".agent-card .agent-head")][0] as HTMLElement).click());
  const body = await page.evaluate(() => {
    const b = [...[...document.querySelectorAll(".msg.assistant")].at(-1)!.querySelectorAll(".agent-card")][0].querySelector(".agent-body") as HTMLElement;
    return { hidden: b.hidden, task: b.querySelector(".agent-task")!.textContent, tool: b.querySelector(".tool-card .tool-name")!.textContent,
      answer: (b.querySelector(":scope > .markdown") as HTMLElement).innerText.trim() };
  });
  expect(body).toEqual({ hidden: false, task: "tool echo", tool: "Echo", answer: "The tool said: echo: hello" });
  await idle(page, page.reload());
  expect(await cards()).toEqual(end);
  await page.close();
}, 90_000);
