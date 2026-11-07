import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync } from "node:fs";
import { harness, CHUNKS, Ev, GAP_MS, LOG, NOTE, RUN, Round, all, ask, base, catalog, chrysalisAborted, collect, config, contents, decide, deltas, dir, events, fake, fetched, inner, json, mcpHttp, outcome, parsed, port, post, results, seen, site, sourceOf, toolEvents, until, voiceServer, unique, web } from "./server-harness";

harness();

test("a streamed reply is a job: Chrysalis's events in order, then the timings, then [DONE]", async () => {
  const res = await post();
  expect(res.headers.get("X-Chrysalis-Job")).toBeTruthy();
  const got = await all(res);
  expect(got.map(([i]) => i)).toEqual(got.map((_, i) => i));
  expect(deltas(got)).toEqual(Array.from({ length: CHUNKS }, (_, i) => `t${i}`));
  expect(JSON.parse(got.at(-2)![1]).timing.first).toBeGreaterThan(0);
  expect(got.at(-1)![1]).toBe("[DONE]");
});

test("a page that goes away leaves the reply running; following from a position gives the rest", async () => {
  const aborts = chrysalisAborted, ctrl = new AbortController();
  const res = await post(ctrl.signal), job = res.headers.get("X-Chrysalis-Job")!;
  const first: (readonly [number, string])[] = [];
  for await (const e of events(res)) if (first.push(e) === 3) break;
  ctrl.abort();
  await Bun.sleep(GAP_MS * (CHUNKS + 5));
  const rest = await all(await fetch(`${base}/api/chat/${job}?from=3`));
  expect(rest[0][0]).toBe(3);
  expect([...deltas(first), ...deltas(rest)]).toEqual(Array.from({ length: CHUNKS }, (_, i) => `t${i}`));
  expect(rest.at(-1)![1]).toBe("[DONE]");
  expect(chrysalisAborted).toBe(aborts);
});

test("Stop ends the reply early with a stopped event, and stops Chrysalis's request", async () => {
  const aborts = chrysalisAborted;
  const res = await post(), job = res.headers.get("X-Chrysalis-Job")!;
  const got: (readonly [number, string])[] = [];
  for await (const e of events(res)) {
    got.push(e);
    if (got.length === 2) expect((await fetch(`${base}/api/chat/${job}/stop`, { method: "POST" })).status).toBe(204);
  }
  expect(deltas(got).length).toBeLessThan(CHUNKS);
  expect(got.map(([, d]) => d)).toContain(JSON.stringify({ stopped: true }));
  expect(got.at(-1)![1]).toBe("[DONE]");
  await Bun.sleep(50);
  expect(chrysalisAborted).toBe(aborts + 1);
});

test("a request addressed to another site's name is refused; this machine's names and addresses are not", async () => {
  const status = (host: string) => fetch(`${base}/api/status`, { headers: { Host: host } });
  expect((await status("attacker.example")).status).toBe(403);
  expect((await status(`attacker.example:${port}`)).status).toBe(403);
  const ok = await status(`localhost:${port}`);
  expect(ok.status).toBe(200);
  expect(await ok.json()).toMatchObject({ state: "online", model: "fake-model" });
  expect((await status(`127.0.0.1:${port}`)).status).toBe(200);
});

test("a backend is kept only once it answers: an address where nothing does is said so, one off this computer's networks over http:// is refused (chats and the key would cross the internet in the clear), and replies still come from Chrysalis", async () => {
  const set = async (url: string) => {
    const r = await fetch(`${base}/api/backend`, { method: "POST", headers: json, body: JSON.stringify({ url }) });
    return [r.status, (await r.json()).error?.message];
  };
  expect(await set("127.0.0.1:9")).toEqual([400, "nothing answers at http://127.0.0.1:9: is the backend running?"]);
  expect(await set("http://1.1.1.1/v1")).toEqual([400, "1.1.1.1 is off this computer and its networks: it takes https://"]);
  expect((await (await fetch(`${base}/api/status`)).json()).backend).toMatchObject({ chrysalis: true, name: "Chrysalis" });
});

test("a backend is Chrysalis by what its model list says, wherever it is: one that says so gets Chrysalis's own fields as the page sent them, one that does not gets OpenAI's fields and llama.cpp's sampling fields alone, its thinking sent as reasoning reaching the page as reasoning_content", async () => {
  const got: Record<string, any> = {};
  const backend = (owner: string) => Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(req) {
    const path = new URL(req.url).pathname;
    if (path === "/v1/models") return Response.json({ object: "list", data: [{ id: "m", object: "model", owned_by: owner }] });
    if (path !== "/v1/chat/completions") return new Response("not found", { status: 404 });
    got[owner] = await req.json();
    return new Response(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { reasoning: "hmm", content: "ok" } }] })}\n\ndata: [DONE]\n\n`,
      { headers: { "Content-Type": "text/event-stream" } });
  } });
  const chrysalis = backend("chrysalis"), other = backend("llamacpp");
  const sent = { model: "m", stream: true, messages: [{ role: "user", content: "hi" }, { role: "assistant", content: "Hel" }], continue_final_message: true,
    chrysalis_original: { role: "assistant", content: "Hello" }, chrysalis_append_token: 7, samplers: ["temperature"], typical_p: 0.9, logprobs: true, top_logprobs: 5, temperature: 0.5 };
  const status = async () => (await (await fetch(`${base}/api/status`)).json()).backend;
  const ask = async () => [...(await (await fetch(`${base}/api/chat`, { method: "POST", headers: json, body: JSON.stringify(sent) })).text()).matchAll(/data: (\{.*\})/g)]
    .map((m) => JSON.parse(m[1]).choices?.[0]?.delta).filter(Boolean);
  try {
    for (const [s, owner] of [[chrysalis, "chrysalis"], [other, "llamacpp"]] as const) {
      expect((await fetch(`${base}/api/backend`, { method: "POST", headers: json, body: JSON.stringify({ url: `localhost:${s.port}` }) })).status).toBe(200);
      expect(await status()).toMatchObject({ url: `http://localhost:${s.port}/v1`, chrysalis: owner === "chrysalis", service: false });
      expect((await ask()).find((d) => d.content)).toEqual(owner === "chrysalis" ? { reasoning: "hmm", content: "ok" } : { reasoning_content: "hmm", content: "ok" });
    }
    const { stream, ...rest } = sent;
    expect(got.chrysalis).toEqual({ ...rest, stream });
    expect(got.llamacpp).toEqual({ model: "m", stream: true, messages: sent.messages, samplers: ["temperature"], typical_p: 0.9, logprobs: true, top_logprobs: 5,
      temperature: 0.5 });
  } finally {
    expect((await fetch(`${base}/api/backend`, { method: "POST", headers: json, body: JSON.stringify({ url: `127.0.0.1:${fake.port}/v1` }) })).status).toBe(200);
    expect(await status()).toMatchObject({ chrysalis: true, service: true });
    chrysalis.stop(true);
    other.stop(true);
  }
});

test("a backend that goes away is called what it is, not Chrysalis, once the reply has tried again", async () => {
  const gone = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: (req) => new URL(req.url).pathname === "/v1/models"
    ? Response.json({ object: "list", data: [{ id: "m", object: "model", owned_by: "llamacpp" }] }) : new Response("not found", { status: 404 }) });
  const at = `http://127.0.0.1:${gone.port}`;
  try {
    expect((await fetch(`${base}/api/backend`, { method: "POST", headers: json, body: JSON.stringify({ url: `${at}/v1` }) })).status).toBe(200);
    gone.stop(true);
    const text = await (await fetch(`${base}/api/chat`, { method: "POST", headers: json, body: JSON.stringify({ stream: true, model: "m", messages: [{ role: "user", content: "hello" }] }) })).text();
    const errors = [...text.matchAll(/data: (\{"error".*\})/g)].map((m) => JSON.parse(m[1]).error.message);
    expect(errors).toEqual([`nothing answers at ${at}: is the backend running?`]);
  } finally {
    expect((await fetch(`${base}/api/backend`, { method: "POST", headers: json, body: JSON.stringify({ url: `127.0.0.1:${fake.port}/v1` }) })).status).toBe(200);
  }
});

test("a job that is not there answers 404", async () => {
  expect((await fetch(`${base}/api/chat/${crypto.randomUUID()}?from=0`)).status).toBe(404);
});

test("a reply, and what starts or allows tools, must come as JSON: another site's form cannot send it", async () => {
  const form = { "Content-Type": "text/plain" };
  expect((await fetch(`${base}/api/chat`, { method: "POST", body: JSON.stringify({ stream: true, messages: [] }), headers: form })).status).toBe(415);
  expect((await fetch(`${base}/api/tools/reload`, { method: "POST", headers: form })).status).toBe(415);
  expect((await fetch(`${base}/api/tools/connect`, { method: "POST", body: '{"source":"mcp:fake"}', headers: form })).status).toBe(415);
  expect((await fetch(`${base}/api/chat/x/tool/y`, { method: "POST", body: '{"decision":"once"}', headers: form })).status).toBe(415);
});

test("an MCP server starts when a reply first needs it; its call and result go back to Chrysalis exactly as Chrysalis wrote them", async () => {
  expect((await sourceOf("mcp:fake")).state).toBe("idle");
  const args = '{"text":  "hi" }';
  const { said, res } = ask([{ calls: [{ name: "fake__echo", args }] }, { text: "Done." }], { "mcp:fake": {}, fetch: {} });
  const got = await collect(await res);
  const [first, second] = seen.get(said)!;
  expect(seen.get(said)!.length).toBe(2);
  expect(first.chrysalis_web).toBeUndefined();
  const names = first.tools.map((t: any) => t.function.name);
  expect(names).toEqual(["fake__big", "fake__echo", "fake__get_weather", "fake__slow", "fake__write_note", "fetch_url"]);
  expect(JSON.stringify(second.messages.slice(0, first.messages.length))).toBe(JSON.stringify(first.messages));
  expect(second.messages.slice(first.messages.length)).toEqual([
    { role: "assistant", content: "Round 0.", reasoning_content: "thinking 0", tool_calls: [{ id: "call_0_0", type: "function", function: { name: "fake__echo", arguments: args } }] },
    { role: "tool", tool_call_id: "call_0_0", content: "echo: hi" },
  ]);
  expect(JSON.stringify(second.tools)).toBe(JSON.stringify(first.tools));
  expect(toolEvents(got).map((t) => t.state)).toEqual(["running", "done"]);
  expect(toolEvents(got)[1]).toMatchObject({ id: "call_0_0", title: "Echo", source: "fake", readOnly: true, result: "echo: hi" });
  const order = got.map((e) => (e.tool ? `tool:${e.tool.state}` : e.round ? `round ${e.round}` : e.choices?.[0]?.delta?.content ?? (e.timing ? "timing" : e.done ? "DONE" : "")));
  expect(order.filter(Boolean)).toEqual(["Round 0.", "tool:running", "tool:done", "round 1", "Done.", "timing", "DONE"]);
  expect((await sourceOf("mcp:fake")).state).toBe("connected");
}, 30_000);

test("several calls in a round run side by side, and their results go back in the order they were called", async () => {
  const { said, res } = ask([{ calls: [{ name: "fake__slow", args: '{"ms": 400}' }, { name: "fake__echo", args: '{"text": "b"}' }] }, { text: "ok" }]);
  const t = Date.now(), got = await collect(await res);
  expect(results(seen.get(said)![1])).toEqual(["slept", "echo: b"]);
  const done = toolEvents(got).filter((e) => e.state === "done").map((e) => e.id);
  expect(done).toEqual(["call_0_1", "call_0_0"]);
  expect(Date.now() - t).toBeLessThan(2000);
}, 30_000);

test("a tool that is not read-only waits for the user: allow once runs it, deny does not, always stops asking", async () => {
  let { said, res } = ask([{ calls: [{ name: "fake__write_note", args: '{"text": "first"}' }] }, { text: "ok" }]);
  let r = await res, job = r.headers.get("X-Chrysalis-Job")!, got: Ev[] = [];
  for await (const e of parsed(r)) {
    got.push(e);
    if (e.tool?.state === "approval") {
      expect(e.tool).toMatchObject({ id: "call_0_0", title: "write_note", readOnly: false });
      await Bun.sleep(200);
      expect(seen.get(said)!.length).toBe(1);
      expect((await decide(job, "call_0_0", "once")).status).toBe(204);
    }
  }
  expect(toolEvents(got).map((t) => t.state)).toEqual(["approval", "running", "done"]);
  expect(results(seen.get(said)![1])).toEqual(["saved: first"]);
  expect((await decide(job, "call_0_0", "once")).status).toBe(409);

  ({ said, res } = ask([{ calls: [{ name: "fake__write_note", args: '{"text": "second"}' }] }, { text: "ok" }]));
  r = await res;
  job = r.headers.get("X-Chrysalis-Job")!;
  got = [];
  for await (const e of parsed(r)) { got.push(e); if (e.tool?.state === "approval") await decide(job, e.tool.id, "deny"); }
  expect(toolEvents(got).map((t) => t.state)).toEqual(["approval", "declined"]);
  expect(results(seen.get(said)![1])[0]).toContain("The user declined this call");
  expect(readFileSync(LOG, "utf8")).not.toContain("wrote second");

  const note = (t: string) => ({ calls: [{ name: "fake__write_note", args: JSON.stringify({ text: t }) }] });
  ({ said, res } = ask([note("third"), note("fourth"), { text: "ok" }]));
  r = await res;
  job = r.headers.get("X-Chrysalis-Job")!;
  got = [];
  for await (const e of parsed(r)) { got.push(e); if (e.tool?.state === "approval") await decide(job, e.tool.id, "always"); }
  expect(toolEvents(got).map((t) => t.state)).toEqual(["approval", "running", "done", "running", "done"]);
  expect(results(seen.get(said)![2])).toEqual(["saved: third", "saved: fourth"]);

  ({ said, res } = ask([note("fifth"), { text: "ok" }], { "mcp:fake": {} }, { allow: ["fake__write_note"] }));
  got = await collect(await res);
  expect(toolEvents(got).map((t) => t.state)).toEqual(["running", "done"]);
}, 30_000);

test("Stop during a tool call cancels the call on its server and ends the reply at once", async () => {
  const { said, res } = ask([{ calls: [{ name: "fake__slow", args: '{"ms": 2500}' }] }, { text: "never" }]);
  const r = await res, job = r.headers.get("X-Chrysalis-Job")!, got: Ev[] = [];
  let stoppedAt = 0;
  for await (const e of parsed(r)) {
    got.push(e);
    if (e.tool?.state === "running") { await Bun.sleep(100); stoppedAt = Date.now(); await fetch(`${base}/api/chat/${job}/stop`, { method: "POST" }); }
  }
  expect(Date.now() - stoppedAt).toBeLessThan(1000);
  expect(toolEvents(got).map((t) => t.state)).toEqual(["running", "stopped"]);
  expect(got).toContainEqual({ stopped: true });
  expect(got.at(-1)).toEqual({ done: true });
  expect(seen.get(said)!.length).toBe(1);
  await Bun.sleep(300);
  expect(readFileSync(LOG, "utf8")).toContain("cancelled slow");
}, 30_000);

test("Stop while a call waits for the user ends the reply, and the call never runs", async () => {
  const { said, res } = ask([{ calls: [{ name: "fake__write_note", args: '{"text": "unasked"}' }] }, { text: "never" }]);
  const r = await res, job = r.headers.get("X-Chrysalis-Job")!, got: Ev[] = [];
  for await (const e of parsed(r)) { got.push(e); if (e.tool?.state === "approval") await fetch(`${base}/api/chat/${job}/stop`, { method: "POST" }); }
  expect(toolEvents(got).map((t) => t.state)).toEqual(["approval", "stopped"]);
  expect(got).toContainEqual({ stopped: true });
  expect(seen.get(said)!.length).toBe(1);
  expect(readFileSync(LOG, "utf8")).not.toContain("wrote unasked");
}, 30_000);

test("a call to a tool the chat has not turned on gets an error result, and the reply goes on", async () => {
  const { said, res } = ask([{ calls: [{ name: "nope__x", args: "{}" }, { name: "fake__write_note", args: '{"text": "off"}' }] }, { text: "ok" }],
    { "mcp:fake": { off: ["write_note"] } });
  const got = await collect(await res);
  expect(seen.get(said)![0].tools.map((t: any) => t.function.name)).not.toContain("fake__write_note");
  expect(toolEvents(got).map((t) => t.state)).toEqual(["failed", "failed"]);
  const back = results(seen.get(said)![1]);
  expect(back[0]).toContain('no tool named "nope__x"');
  expect(back[1]).toContain('no tool named "fake__write_note"');
  expect(readFileSync(LOG, "utf8")).not.toContain("wrote off");
  expect(got.filter((e) => e.choices?.[0]?.delta?.content === "ok").length).toBe(1);
}, 30_000);

test("a page that follows a reply with tools again from any event gets all the rest of it", async () => {
  const slow = (n: number) => ({ calls: [{ name: "fake__echo", args: `{"text": "${n}"}` }], gap: 30 });
  const { res } = ask([slow(1), slow(2), { text: "end", gap: 30 }]);
  const r = await res, job = r.headers.get("X-Chrysalis-Job")!, early: (readonly [number, string])[] = [];
  for await (const e of events(r)) { early.push(e); if (early.length === 7) break; }
  await r.body!.cancel().catch(() => {});
  await all(await fetch(`${base}/api/chat/${job}?from=0`));
  const whole = await all(await fetch(`${base}/api/chat/${job}?from=0`));
  const state = (got: (readonly [number, string])[]) => {
    const evs = got.filter(([, d]) => d !== "[DONE]").map(([, d]) => JSON.parse(d));
    return { words: evs.flatMap((e) => e.choices?.[0]?.delta?.content ?? []).join(""), tools: evs.filter((e) => e.tool).map((e) => e.tool), rounds: evs.filter((e) => e.round).map((e) => e.round) };
  };
  let restarts = 0;
  for (let from = 1; from <= whole.at(-1)![0]; from++) {
    const rest = await all(await fetch(`${base}/api/chat/${job}?from=${from}`));
    if (rest[0][1] === '{"reset":true}') { restarts++; expect(rest.slice(1)).toEqual(whole); }
    else expect([...whole.filter(([i]) => i < from), ...rest]).toEqual(whole);
  }
  expect(restarts).toBeGreaterThan(0);
  const resumed = await all(await fetch(`${base}/api/chat/${job}?from=7`));
  expect(state(resumed[0][1] === '{"reset":true}' ? resumed.slice(1) : [...early, ...resumed])).toEqual(state(whole));
  expect(whole.filter(([, d]) => d.includes('"round"')).length).toBe(2);
  expect(whole.filter(([, d]) => d.includes('"state":"done"')).length).toBe(2);
}, 30_000);

test("a reply runs as many rounds of tool calls as it needs: no number of them ends it", async () => {
  const { said, res } = ask([...Array.from({ length: 40 }, () => ({ calls: [{ name: "fake__echo", args: '{"text": "again"}' }] })), { text: "That is all of it" }]);
  const got = await collect(await res);
  expect(seen.get(said)!.length).toBe(41);
  expect(toolEvents(got).filter((t) => t.state === "done").length).toBe(40);
  expect(contents(got).at(-1)).toBe("That is all of it");
  expect(got.filter((e) => e.error || e.note)).toEqual([]);
}, 60_000);

test("a tool's result is cut to the server's limit, and says so; a call past its time limit fails", async () => {
  const { said, res } = ask([{ calls: [{ name: "fake__big", args: '{"n": 50000}' }, { name: "fake__slow", args: '{"ms": 6000}' }] }, { text: "ok" }]);
  const t = Date.now(), got = await collect(await res);
  const [big, slow] = results(seen.get(said)![1]), whole = Array.from({ length: 50000 }, (_, i) => String.fromCharCode(97 + (i % 26))).join("");
  expect(big.startsWith(whole.slice(0, 40000))).toBe(true);
  expect(big.slice(40000)).toBe("\n\n[The result was cut here: 40,000 of 50,000 characters.]");
  expect(slow).toBe("Error: the tool did not answer within 3 s");
  expect(toolEvents(got).filter((e) => e.result).map((e) => e.state).sort()).toEqual(["done", "failed"]);
  expect(Date.now() - t).toBeLessThan(5000);
}, 30_000);

test("tool names are valid, unique function names, prefixed by their server; the page sees each source's state", async () => {
  await collect(await ask([{ text: "hi" }], { "mcp:a.b": {}, "mcp:a_b": {}, "mcp:web": {}, "mcp:narrow": {}, "mcp:plain": {}, "mcp:broken": {}, "mcp:off": {} }).res);
  const { sources } = await catalog(), by = Object.fromEntries(sources.map((s) => [s.id, s]));
  const fns = sources.flatMap((s) => s.tools.map((t: any) => t.fn));
  expect(new Set(fns).size).toBe(fns.length);
  for (const fn of fns) expect(fn).toMatch(/^[A-Za-z0-9_-]{1,64}$/);
  expect(by["mcp:a.b"].tools.map((t: any) => t.fn)).toContain("a_b__echo");
  expect(by["mcp:a_b"].tools.map((t: any) => t.fn)).toContain("a_b__echo_2");
  expect(by["mcp:fake"].tools.find((t: any) => t.name === "get.weather").fn).toBe("fake__get_weather");
  expect(by["mcp:web"]).toMatchObject({ state: "connected", detail: `http://127.0.0.1:${mcpHttp.port}/mcp` });
  expect(by["mcp:broken"].state).toBe("failed");
  expect(by["mcp:broken"].error).toMatch(/ENOENT|nonexistent|not found/i);
  expect(by["mcp:off"].state).toBe("disabled");
  expect(by["mcp:narrow"].tools.map((t: any) => t.name)).toEqual(["echo"]);
  expect(by["mcp:plain"]).toMatchObject({ state: "failed", error: '"url" must be https (plain http only to this machine)' });
  expect(by["mcp:fake"].detail).toBe(`${process.execPath} fake-mcp.ts`);
  expect(JSON.stringify(sources)).not.toContain(LOG);
  const { said, res } = ask([{ calls: [{ name: "web__echo", args: '{"text": "over http"}' }] }, { text: "ok" }], { "mcp:web": {} });
  await collect(await res);
  expect(results(seen.get(said)![1])).toEqual(["echo: over http"]);
}, 30_000);

test("Fetch URL refuses this machine and private networks, by address, by name and through a redirect, unless the config allows them", async () => {
  const sitePort = site.port;
  for (const url of [`http://127.0.0.1:${sitePort}/page`, `http://localhost:${sitePort}/page`, `http://[::1]:${sitePort}/page`, "http://10.1.2.3/", "http://169.254.169.254/latest"]) {
    const { result, state } = await fetched(url);
    expect(state).toBe("failed");
    expect(result).toContain("is not a public address");
  }
  expect((await fetched("file:///etc/passwd")).result).toContain("only http and https");

  config({ allowPrivate: ["127.0.0.1"] });
  await fetch(`${base}/api/tools/reload`, { method: "POST", headers: json });
  try {
    const page = await fetched(`http://127.0.0.1:${sitePort}/page`);
    expect(page.state).toBe("done");
    expect(page.result).toBe([`URL: http://127.0.0.1:${sitePort}/page`, "Status: 200 OK", "Content-Type: text/html", "Title: Chrysalis & tools", "",
      "# Fetched", "", `A [link](http://127.0.0.1:${sitePort}/docs) and \`code\`.`, "", "- one", "- two", "", "```", "let x = 1;", "```"].join("\n"));
    const hop = await fetched(`http://127.0.0.1:${sitePort}/hop`);
    expect(hop.state).toBe("failed");
    expect(hop.result).toContain("127.0.0.2 is not a public address");
    expect((await fetched(`http://127.0.0.2:${inner.port}/`)).result).toContain("127.0.0.2 is not a public address");

    config({ allowPrivate: true });
    await fetch(`${base}/api/tools/reload`, { method: "POST", headers: json });
    const through = await fetched(`http://127.0.0.1:${sitePort}/hop`);
    expect(through.result).toBe(`URL: http://127.0.0.2:${inner.port}/secret\nRedirected from: http://127.0.0.1:${sitePort}/hop\nStatus: 200 OK\nContent-Type: text/plain\n\nthe secret`);
    const missing = await fetched(`http://127.0.0.1:${sitePort}/nothing`);
    expect(missing.result).toContain("Status: 404 Not Found");
    const pic = await fetched(`http://127.0.0.1:${sitePort}/pic`);
    expect(pic.state).toBe("failed");
    expect(pic.result).toContain("is image/png (64 bytes): Fetch URL reads web pages and text only");
  } finally {
    config();
    await fetch(`${base}/api/tools/reload`, { method: "POST", headers: json });
  }
  expect((await fetched(`http://127.0.0.1:${sitePort}/page`)).result).toContain("is not a public address");
}, 60_000);

test("Fetch URL gives a long page in parts the model can ask for, within the chat's length", async () => {
  config({ allowPrivate: true });
  await fetch(`${base}/api/tools/reload`, { method: "POST", headers: json });
  try {
    const page = "0123456789".repeat(5000);
    const run = async (more: object, chars = 2000) => {
      const { said, res } = ask([{ calls: [{ name: "fetch_url", args: JSON.stringify({ url: `http://127.0.0.1:${site.port}/long`, ...more }) }] }, { text: "ok" }],
        { fetch: {} }, chars ? { fetchChars: chars } : {});
      await collect(await res);
      return results(seen.get(said)![1])[0] as string;
    };
    const head = await run({});
    expect(head.length).toBeLessThanOrEqual(2000);
    const body = head.split("\n\n")[1], end = body.length;
    expect(page.startsWith(body)).toBe(true);
    expect(head).toContain(`call fetch_url again with start=${end} to read on`);
    const next = (await run({ start: end })).split("\n\n")[1];
    expect(next).toBe(page.slice(end, end + next.length));
    const plain = await run({}, 0);
    expect(plain.length).toBeLessThanOrEqual(20_000);
    expect(plain).toContain(" of 50,000: call fetch_url again with start=");
  } finally {
    config();
    await fetch(`${base}/api/tools/reload`, { method: "POST", headers: json });
  }
}, 30_000);

test("voice: only this server's own pages get a socket; it starts the service, passes both sides' frames through as they are, and stops the service once idle", async () => {
  const log = `${mkdtempSync(`${RUN}/voice-`)}/voice.log`, events = () => {
    try { return readFileSync(log, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l)); } catch { return []; }
  };
  const v = await voiceServer({ CHRYSALIS_VOICE_CMD: JSON.stringify([process.execPath, `${import.meta.dir}/fake-voice.ts`]), FAKE_VOICE_LOG: log, CHRYSALIS_VOICE_IDLE_MS: "300" });
  try {
    expect(await outcome(v.socket("http://evil.example"))).toBe("closed");
    expect(await outcome(v.socket())).toBe("closed");
    expect(events()).toEqual([]);
    const s = v.socket(v.url), got: (string | ArrayBuffer)[] = [];
    s.binaryType = "arraybuffer";
    s.onmessage = (e) => got.push(e.data);
    expect(await outcome(s)).toBe("open");
    s.send(JSON.stringify({ type: "hello", voice: "silent" }));
    for (let k = 0; k < 10; k++) s.send(new Int16Array(320).buffer);
    s.send(JSON.stringify({ type: "say", id: "r1", text: "one two three four" }));
    s.send(JSON.stringify({ type: "say_end", id: "r1" }));
    expect(await until(() => got.some((m) => typeof m === "string" && JSON.parse(m).type === "said"))).toBe(true);
    expect(JSON.parse(got[0] as string)).toMatchObject({ type: "ready", voice: "silent", stack: { tts: "fake-tts" } });
    expect(events().filter((e) => e.ev === "frames").at(-1)?.n).toBe(10);
    const chunks = got.filter((m): m is ArrayBuffer => typeof m !== "string").map((b) => {
      const n = new DataView(b).getUint32(0, true);
      return { head: JSON.parse(new TextDecoder().decode(new Uint8Array(b, 4, n))), samples: (b.byteLength - 4 - n) / 2 };
    });
    expect(chunks.map((c) => c.head)).toEqual(events().filter((e) => e.ev === "chunk").map(({ id, text, from, to }, seq) => ({ id, seq, text, from, to, rate: 24000 })));
    expect(chunks.map((c) => [c.head.text, c.samples])).toEqual([["one two three ", 8640], ["four", 6000]]);
    s.close();
    expect(await until(() => events().some((e) => e.ev === "exit"))).toBe(true);
    expect(events().filter((e) => e.ev === "start").length).toBe(1);
  } finally {
    v.proc.kill();
  }
}, 20_000);

test("voice: a speech service that cannot start says so", async () => {
  const v = await voiceServer({ CHRYSALIS_VOICE_CMD: JSON.stringify(["/nonexistent/chrysalis-voice"]) });
  try {
    const s = v.socket(v.url), got: string[] = [];
    s.onmessage = (e) => got.push(e.data);
    expect(await outcome(s)).toBe("open");
    expect(await until(() => s.readyState === WebSocket.CLOSED)).toBe(true);
    expect(JSON.parse(got[0])).toMatchObject({ type: "error" });
    expect(JSON.parse(got[0]).message).toContain("The voice service did not start");
  } finally {
    v.proc.kill();
  }
});

test("a reply whose stream from Chrysalis keeps breaking off is tried again, then says so, rather than that Chrysalis is not running", async () => {
  const got = await collect(await ask([{ text: "Half of it", broken: true }]).res);
  expect(new Set(contents(got))).toEqual(new Set(["Half of it"]));
  expect(got.filter((e) => e.retry).map((e) => e.retry.attempt)).toEqual(Array.from({ length: 10 }, (_, i) => i + 1));
  expect(got.filter((e) => e.error).map((e) => e.error.message)).toEqual(["Chrysalis's reply broke off (Chrysalis's log has why)"]);
});

test("voice: a reply begun ahead runs no tool until the page says it stands (go), and one stopped before then runs none", async () => {
  const held = async (text: string) => {
    const said = `run ${JSON.stringify([{ calls: [{ name: "fake__echo", args: JSON.stringify({ text }) }] }, { text: "Done." }])}\n${unique()}`;
    const res = await fetch(`${base}/api/chat`, { method: "POST", headers: json, body: JSON.stringify({ stream: true, model: "m",
      chrysalis_ahead: true, messages: [{ role: "user", content: NOTE + said }], chrysalis_web: { tools: { "mcp:fake": {} } } }) });
    const got: Ev[] = [], done = (async () => { for await (const e of parsed(res)) got.push(e); })();
    for (const t = Date.now(); !got.some((e) => e.choices?.[0]?.finish_reason === "tool_calls") && Date.now() - t < 5000; ) await Bun.sleep(10);
    await Bun.sleep(300);
    expect([got.some((e) => e.choices?.[0]?.finish_reason === "tool_calls"), toolEvents(got)]).toEqual([true, []]);
    return { job: res.headers.get("X-Chrysalis-Job")!, got, done };
  };
  const go = await held("a");
  await fetch(`${base}/api/chat/${go.job}/go`, { method: "POST" });
  await go.done;
  expect([toolEvents(go.got).map((t) => t.state), contents(go.got).at(-1)]).toEqual([["running", "done"], "Done."]);
  const stop = await held("b");
  await fetch(`${base}/api/chat/${stop.job}/stop`, { method: "POST" });
  await stop.done;
  expect([toolEvents(stop.got), stop.got.some((e) => e.stopped)]).toEqual([[], true]);
}, 30_000);

test("the preview of a reply's request is what Chrysalis is then sent for it, tools and all", async () => {
  const said = `run ${JSON.stringify([{ text: "Hi." }])}\n${unique()}`;
  const body = { stream: true, model: "m", messages: [{ role: "system", content: "S" }, { role: "user", content: said }], chrysalis_web: { tools: { "mcp:fake": {}, fetch: {} } } };
  const preview = await (await fetch(`${base}/api/preview`, { method: "POST", headers: json, body: JSON.stringify(body) })).json();
  await collect(await fetch(`${base}/api/chat`, { method: "POST", headers: json, body: JSON.stringify(body) }));
  expect(preview.body).toEqual(seen.get(said)![0]);
  expect(preview.body.tools.map((t: any) => t.function.name)).toEqual(["fake__big", "fake__echo", "fake__get_weather", "fake__slow", "fake__write_note", "fetch_url"]);
  const ahead = await (await fetch(`${base}/api/preview`, { method: "POST", headers: json, body: JSON.stringify({ ...body, chrysalis_web: { tools: { agents: {}, fetch: {} } }, chrysalis_ahead: true }) })).json();
  expect([ahead.body.tools.map((t: any) => t.function.name), ahead.body.chrysalis_ahead]).toEqual([["fetch_url", "subagent"], undefined]);
  expect((await fetch(`${base}/api/preview`, { method: "POST", body: JSON.stringify(body) })).status).toBe(415);
}, 30_000);
