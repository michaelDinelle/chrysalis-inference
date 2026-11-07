import { expect, test } from "bun:test";
import { harness, fakes, fill, lastStream, look, open, probes, send, shown, still } from "./browser-harness";

harness();

test("a chat whose reply is being written says Working, and for how long, where the list shows its age; once written it shows its age again", async () => {
  const page = await open();
  await page.type("#input", "slow count");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /^Working[1-9]\d*s$/.test(document.querySelector(".conv .conv-status")?.textContent ?? ""), { timeout: 10_000 });
  expect(await page.$(".conv .conv-ago")).toBeNull();
  fakes.hurry = true;
  await page.waitForSelector(".msg.assistant .stats", { timeout: 20_000 });
  await page.waitForFunction(() => !document.querySelector(".conv .conv-status") && document.querySelector(".conv .conv-ago")?.textContent === "now");
  await page.close();
}, 30_000);

test("pointing: the part under the pointer glows, a click keeps it in the message box, and the question tells the model that part as it is now", async () => {
  const page = await open();
  await send(page, "draw the panda");
  await still(page);
  const at = await page.$eval(".svg-fig[data-art] .svg-paper", (p) => {
    const r = p.shadowRoot!.querySelector("#burger rect")!.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  });
  const glow = () => page.$eval(".svg-fig[data-art] .svg-paper", (p) => p.shadowRoot!.querySelector("style.point")?.textContent ?? "");
  await page.mouse.move(at.x, at.y);
  expect(await glow()).toContain('[id="burger"] { filter');
  await page.mouse.click(at.x, at.y);
  expect(await page.$eval("#pointing", (e) => e.textContent)).toBe("Pointing at panda · #burger");
  await send(page, "make this orange");
  expect(lastStream.messages.at(-1).content).toBe('[The user is pointing at the part #burger of the artifact panda:\n<g id="burger">\n    ' +
    '<rect x="35" y="50" width="30" height="10" fill="#8b4513"/>\n  </g>]\n\nmake this orange');
  expect([await page.$eval("#pointing", (e) => (e as HTMLElement).hidden), await page.$eval(".point-chip", (e) => e.textContent)]).toEqual([true, "panda · #burger"]);
  expect(await page.$eval(".art-canvas[data-v='2'] .svg-paper", (p) => p.shadowRoot!.querySelector("#burger rect")?.getAttribute("fill"))).toBe("#ff8c00");
  await page.close();
}, 30_000);

test("TeX is typeset; prices, escaped dollars and code stay text", async () => {
  const page = await open();
  await send(page, "some math");
  const body = await page.$eval(".msg.assistant .markdown", (b) => ({
    math: [...b.querySelectorAll(".katex")].map((k) => k.querySelector("annotation")?.textContent ?? ""),
    display: b.querySelectorAll(".katex-display").length,
    text: [...b.querySelectorAll("p")].map((p) => p.textContent).join("\n"),
  }));
  expect(body.math).toEqual(["ax^2 + bx + c = 0", "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}", "e^{i\\pi} + 1 = 0", "\\int_0^1 x\\,dx"]);
  expect(body.display).toBe(2);
  expect(body.text).toContain("It costs $5 and $10. Escaped: $x$. Code: $HOME stays.");
  await page.close();
}, 30_000);

test("a chat reads as SillyTavern sets it: speech in quotes in its orange, the marks kept (one left open to its line's end), italics muted but a quote's inside one, a line break where one was written, code as written; the user's own text the same, its code and HTML as typed; a document's text as written", async () => {
  const page = await open();
  await send(page, '*knocks* "Is **the kettle** on?" and `ls "x"` <b>as typed</b>');
  const look = (sel: string) => page.$eval(sel, (b) => {
    const color = (e: Element | null) => e && getComputedStyle(e).color;
    return {
      quotes: [...b.querySelectorAll("q")].map((q) => q.textContent),
      marks: [...b.querySelectorAll("q")].map((q) => getComputedStyle(q, "::before").content + " " + getComputedStyle(q, "::after").content),
      breaks: b.querySelectorAll("br").length,
      text: color(b), quote: color(b.querySelector("q")), em: color(b.querySelector(":not(q) > em")), quoteEm: color(b.querySelector("q em")),
      html: b.innerHTML, said: b.textContent,
    };
  });
  const reply = await look(".msg.assistant .markdown"), mine = await look(".msg.user .bubble");
  expect(reply.quotes).toEqual(['"Good morning,"', '"Did you sleep well?"', '"It\'s still open', "«bonjour»"]);
  expect(reply.marks).toEqual(Array(4).fill("none none"));
  expect([reply.breaks, reply.quote, reply.quoteEm]).toEqual([3, "rgb(225, 138, 36)", "rgb(225, 138, 36)"]);
  expect([reply.em === reply.text, reply.em === reply.quote]).toEqual([false, false]);
  expect(reply.said).toContain('well?"\nThe kettle sings.\n"It');
  expect(mine.html).toBe('<em>knocks</em> <q>"Is <strong>the kettle</strong> on?"</q> and `ls "x"` &lt;b&gt;as typed&lt;/b&gt;');
  expect([mine.quote, mine.em, mine.marks]).toEqual([reply.quote, reply.em, ["none none"]]);
  await send(page, "write the letter");
  expect(await page.$eval(".art-canvas .doc-paper", (p) => {
    const md = p.shadowRoot!.querySelector(".md")!;
    return [md.querySelectorAll("q, br").length, md.textContent!.replace(/\s+/g, " ").trim()];
  })).toEqual([0, 'She wrote "come home" and signed it.']);
  await page.close();
}, 30_000);

test("speech and italics are set as they are written: a quote in its color from its opening mark, italics from their first word, and no asterisk shown on the way", async () => {
  const page = await open();
  await page.evaluate(() => {
    const seen: string[] = (window as any).seen = [];
    new MutationObserver(() => { const m = document.querySelector(".msg.assistant .markdown"); if (m) seen.push(m.innerHTML); })
      .observe(document.getElementById("thread")!, { childList: true, subtree: true, characterData: true });
  });
  await send(page, "steady speech");
  const seen: string[] = await page.evaluate(() => (window as any).seen);
  expect(seen.length).toBeGreaterThan(10);
  expect(seen.filter((h) => h.replace(/<[^>]*>/g, "").includes("*"))).toEqual([]);
  expect(seen.some((h) => /<q>"I kept[^"]*<\/q>/.test(h))).toBe(true);
  expect(seen.some((h) => /<em>she says(?!, and smiles at the window, where the snow is still falling\.<)[^<]*<\/em>/.test(h))).toBe(true);
  expect(seen.at(-1)).toContain('<q>"I kept the lamp lit for you, all night long."</q>');
  await page.close();
}, 30_000);

test("Mermaid blocks become diagrams; one that does not parse shows its source", async () => {
  const page = await open();
  await send(page, "a diagram");
  await page.waitForFunction(() => [...document.querySelectorAll(".svg-fig")].every((f) => f.querySelector(".svg-paper")!.shadowRoot!.querySelector("svg") || f.classList.contains("failed")), { timeout: 20_000 });
  const figs = await page.$$eval(".svg-fig", (fs) => fs.map((f) => ({
    state: f.querySelector(".svg-state")!.textContent,
    labels: [...(f.querySelector(".svg-paper")!.shadowRoot!.querySelectorAll("text, tspan"))].map((t) => t.textContent).join(" "),
    code: f.classList.contains("failed") ? f.querySelector(".svg-code code")!.textContent : "",
  })));
  expect(figs.map((f) => f.state)).toEqual(["Mermaid · flowchart", "Mermaid · sequence diagram", "Couldn't draw this diagram"]);
  for (const word of ["Ask", "Cached?", "Append", "Prefill"]) expect(figs[0].labels).toContain(word);
  expect(figs[1].labels).toContain("question");
  expect(figs[2].code).toContain("not mermaid ((");
  await page.close();
}, 30_000);

test("an HTML page runs in its sandbox and reaches nothing", async () => {
  const page = await open();
  await send(page, "a page");
  const frame = await (await page.waitForSelector("iframe.page-frame"))!.contentFrame();
  await frame.waitForFunction(() => document.getElementById("out")?.textContent !== "running", { timeout: 10_000 });
  expect(await frame.$eval("#out", (o) => o.textContent)).toBe("cookie blocked,fetch blocked,image blocked,parent blocked,storage blocked");
  expect(probes).toBe(0);
  await frame.click("#b");
  await frame.click("#b");
  expect(await frame.$eval("#b", (b) => b.textContent)).toBe("2");
  await page.close();
}, 30_000);
