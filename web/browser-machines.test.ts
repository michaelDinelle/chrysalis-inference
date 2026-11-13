import { expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { hostname } from "node:os";
import { RUN, base, connect, harness, lastText, open, opencode, send } from "./browser-harness";
import { fakeOpenCode } from "./fake-opencode";

harness();

test("another computer let in runs its own OpenCode from this page: it is under Run on once its line's helper connects, a new chat picked onto it works in its folders through its OpenCode on Chrysalis's models alone, the thread stays on that computer, and it is gone when the helper stops", async () => {
  const den = fakeOpenCode("den-secret"), away = mkdtempSync(`${RUN}/away-`), denHome = `${away}/home`, config = `${away}/config.json`, bin = `${away}/opencode`, keyFile = `${away}/key`;
  mkdirSync(`${denHome}/hive`, { recursive: true });
  writeFileSync(bin, `#!/bin/sh\nexec "${process.execPath}" "${import.meta.dir}/fake-away.ts" "$@"\n`);
  chmodSync(bin, 0o755);
  const page = await open(), json = { "Content-Type": "application/json" }, name = hostname().replace(/\.local$/, "");
  await connect(page);
  const machines = () => page.evaluate(() => fetch("/api/threads").then((r) => r.json()).then((j) => j.machines.map((m: { name: string; home: string }) => [m.name, m.home])));
  expect((await machines()).length).toBe(1);
  expect(await page.evaluate(() => fetch("/opencode/machine").then((r) => r.status))).toBe(401);

  const [secret, code] = (await (await fetch(`${base}/opencode/ask`, { method: "POST", headers: json, body: JSON.stringify({ name: "den" }) })).text()).split(" ");
  expect((await fetch(`${base}/api/pair/allow`, { method: "POST", headers: json, body: JSON.stringify({ code }) })).status).toBe(200);
  writeFileSync(keyFile, await (await fetch(`${base}/opencode/ask/${secret}`)).text());
  const helper = Bun.spawn([process.execPath, `${import.meta.dir}/machine.js`, base, keyFile], { stdout: "ignore", stderr: "inherit",
    env: { ...process.env, HOME: denHome, CHRYSALIS_OPENCODE: bin, FAKE_AWAY_URL: `http://127.0.0.1:${den.server.port}`, FAKE_AWAY_SECRET: "den-secret", FAKE_AWAY_CONFIG: config } });
  try {
    await page.waitForFunction(() => fetch("/api/threads").then((r) => r.json()).then((j) => j.machines.length === 2), { timeout: 20_000 });
    expect((await machines())[1]).toEqual([name, denHome]);
    const kept = JSON.parse(readFileSync(config, "utf8"));
    expect([kept.enabled_providers, kept.providers.chrysalis.settings]).toEqual([["chrysalis"], { baseURL: `${base}/v1`, apiKey: readFileSync(keyFile, "utf8") }]);

    await page.reload();
    await page.waitForSelector("#opencode .dot.online");
    await page.click("#new-chat");
    await page.waitForSelector("#machine-btn:not([hidden])");
    await page.click("#machine-btn");
    expect(await page.$$eval("#row-menu > *", (x) => x.map((e) => e.textContent))).toEqual(["Run on", name, name, "Add another computer"]);
    await page.evaluate(() => document.querySelectorAll<HTMLElement>("#row-menu button")[1].click());
    await page.waitForFunction((d) => document.querySelector("#folder-btn")?.getAttribute("title") === `Working in ${d}`, {}, denHome);
    await page.click("#folder-btn");
    await page.waitForFunction(() => [...document.querySelectorAll("#oc-pop .oc-dir .invite-name")].some((x) => x.textContent === "hive"));
    await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#oc-pop .oc-dir")].find((r) => r.querySelector(".invite-name")!.textContent === "hive")!.querySelector<HTMLElement>(".oc-open")!.click());
    await page.waitForFunction((d) => document.querySelector("#folder-btn")?.getAttribute("title") === `Working in ${d}/hive`, {}, denHome);

    const own = await fetch(`http://127.0.0.1:${den.server.port}/api/session`, { method: "POST", headers: { ...json, authorization: `Basic ${Buffer.from("opencode:den-secret").toString("base64")}` },
      body: JSON.stringify({ location: { directory: denHome }, title: "made in its own OpenCode" }) });
    expect(own.ok).toBe(true);
    const here = opencode.log.prompts.length;
    await send(page, "count the hives");
    const sent = den.log.prompts.at(-1)!;
    expect([opencode.log.prompts.length - here, sent.text, sent.model, den.sessions.get(sent.session)!.dir]).toEqual([0, "count the hives", { providerID: "chrysalis", id: "qwen3.8-27b" }, `${denHome}/hive`]);
    expect(await lastText(page)).toContain("You asked: count the hives");
    expect(await page.$eval("#machine-btn", (b) => [b.classList.contains("fixed"), b.getAttribute("title")])).toEqual([true, `Runs on ${name}`]);
    await page.click("#machine-btn");
    expect(await page.$eval("#row-menu", (m) => (m as HTMLElement).hidden)).toBe(true);
    const listed = () => page.evaluate(() => fetch("/api/threads").then((r) => r.json()).then((j) => j.threads.filter((t: { machine?: string }) => t.machine).map((t: { id: string; dir: string }) => [t.id, t.dir])));
    expect(await listed()).toEqual([[sent.session, `${denHome}/hive`]]);

    await page.reload();
    await page.waitForSelector("#opencode .dot.online");
    await page.waitForFunction(() => [...document.querySelectorAll("#conversations .conv-title")].some((t) => t.textContent === "count the hives"));
    await page.evaluate(() => [...document.querySelectorAll<HTMLElement>("#conversations a.conv")].find((a) => a.querySelector(".conv-title")!.textContent === "count the hives")!.click());
    await page.waitForFunction(() => document.querySelector("#machine-btn")?.classList.contains("fixed"));
    await send(page, "and the queens");
    expect([opencode.log.prompts.length - here, den.log.prompts.at(-1)!.text, den.log.prompts.at(-1)!.session]).toEqual([0, "and the queens", sent.session]);
  } finally {
    helper.kill();
    await helper.exited;
  }
  await page.waitForFunction(() => fetch("/api/threads").then((r) => r.json()).then((j) => j.machines.length === 1 && !j.threads.some((t: { machine?: string }) => t.machine)), { timeout: 10_000 });
  den.server.stop(true);
  await page.close();
}, 90_000);
