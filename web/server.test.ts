// The web server (server.ts) against a fake kiln: a streamed reply is a job that runs to its end whatever its page
// does, any page can follow it again from an event's position, only Stop ends it early (and stops kiln's request), and
// a request addressed to another site's name is refused (DNS rebinding). With tools, a reply is rounds: the fake kiln
// plays a script of tool calls, which server.ts runs on real MCP servers (fake-mcp.ts, over stdio and Streamable HTTP,
// through the SDK) and on the built-in Fetch URL against local sites, and sends back. Nothing here reaches the real
// kiln or systemd.
import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { fakeMcp } from "./fake-mcp";

const CHUNKS = 12, GAP_MS = 20;
let kilnAborted = 0;
// A tool script, as the user's message: "run ", the rounds as JSON (each the calls kiln answers with, or its text), and
// a line that makes the message unique. Each round is told by the assistant turns the request already holds; every
// request a script gets is kept.
interface Round { calls?: { name: string; args: string }[]; text?: string; gap?: number }
const seen = new Map<string, any[]>();
const enc = new TextEncoder();
const sse = (o: unknown) => enc.encode(`data: ${typeof o === "string" ? o : JSON.stringify(o)}\n\n`);
const fake = Bun.serve({
  hostname: "127.0.0.1",
  port: 0,
  async fetch(req) {
    const path = new URL(req.url).pathname;
    if (path === "/kiln/stats") return Response.json({ model: "fake-model" });
    if (path !== "/v1/chat/completions") return new Response("not found", { status: 404 });
    const body = await req.json();
    if (!body.stream) return Response.json({ choices: [{ message: { content: "a title" } }] });
    req.signal.addEventListener("abort", () => kilnAborted++);
    const said: string = body.messages.find((m: any) => m.role === "user")?.content ?? "";
    if (said.startsWith("run ")) {
      const script: Round[] = JSON.parse(said.slice(4).split("\n")[0]), k = body.messages.filter((m: any) => m.role === "assistant").length;
      seen.set(said, [...(seen.get(said) ?? []), body]);
      const r = script[Math.min(k, script.length - 1)];
      const chunks = [{ delta: { role: "assistant", content: "" } }, { delta: { reasoning_content: `thinking ${k}` } }, { delta: { content: r.text ?? `Round ${k}.` } },
        ...(r.calls ?? []).map((c, i) => ({ delta: { tool_calls: [{ index: i, id: `call_${k}_${i}`, type: "function", function: { name: c.name, arguments: c.args } }] } })),
        { delta: {}, finish_reason: r.calls?.length ? "tool_calls" : "stop" }];
      let i = 0;
      return new Response(new ReadableStream({
        async pull(ctl) {
          await Bun.sleep(r.gap ?? 2);
          if (i < chunks.length) ctl.enqueue(sse({ choices: [chunks[i++]] }));
          else {
            ctl.enqueue(sse({ choices: [], usage: { prompt_tokens: 100 + k, completion_tokens: 10 } }));
            ctl.enqueue(sse("[DONE]"));
            ctl.close();
          }
        },
      }), { headers: { "Content-Type": "text/event-stream" } });
    }
    let i = 0;
    return new Response(new ReadableStream({
      async pull(ctl) {
        await Bun.sleep(GAP_MS);
        if (i < CHUNKS) ctl.enqueue(sse({ choices: [{ delta: { content: `t${i++}` } }] }));
        else { ctl.enqueue(sse("[DONE]")); ctl.close(); }
      },
    }), { headers: { "Content-Type": "text/event-stream" } });
  },
});

// Sites for Fetch URL: one on 127.0.0.1, one on 127.0.0.2 (both this machine, both refused unless the config allows them).
const PAGE = `<!doctype html><html><head><title>Kiln &amp; tools</title><script>alert("no")</script><style>p{}</style></head><body>
<nav><a href="/">Home</a></nav><main><h1>Fetched</h1><p>A <a href="/docs">link</a> and <code>code</code>.</p><ul><li>one</li><li>two</li></ul>
<pre>let x = 1;</pre></main><footer>Copyright</footer></body></html>`;
const site = Bun.serve({
  hostname: "127.0.0.1", port: 0,
  fetch(req) {
    const u = new URL(req.url);
    if (u.pathname === "/page") return new Response(PAGE, { headers: { "Content-Type": "text/html; charset=utf-8" } });
    if (u.pathname === "/hop") return Response.redirect(`http://127.0.0.2:${inner.port}/secret`, 302);
    if (u.pathname === "/long") return new Response("0123456789".repeat(5000), { headers: { "Content-Type": "text/plain" } });
    if (u.pathname === "/pic") return new Response(new Uint8Array(64), { headers: { "Content-Type": "image/png" } });
    return new Response("missing", { status: 404, headers: { "Content-Type": "text/plain" } });
  },
});
const inner = Bun.serve({ hostname: "127.0.0.2", port: 0, fetch: () => new Response("the secret", { headers: { "Content-Type": "text/plain" } }) });
// An MCP server over Streamable HTTP (stateless: a server per request, as the SDK does it).
const mcpHttp = Bun.serve({
  hostname: "127.0.0.1", port: 0,
  async fetch(req) {
    const t = new WebStandardStreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
    await fakeMcp().connect(t);
    return t.handleRequest(req);
  },
});

const dir = mkdtempSync(`${tmpdir()}/kiln-tools-`), CONFIG = `${dir}/mcp.json`, LOG = `${dir}/mcp.log`;
const stdio = (timeout?: number) => ({ command: process.execPath, args: ["fake-mcp.ts"], cwd: import.meta.dir, env: { FAKE_MCP_LOG: LOG }, ...(timeout ? { timeout } : {}) });
const config = (fetch: object = {}) => writeFileSync(CONFIG, JSON.stringify({
  mcpServers: { fake: stdio(3), "a.b": stdio(), a_b: stdio(), web: { url: `http://127.0.0.1:${mcpHttp.port}/mcp` },
    narrow: { url: `http://localhost:${mcpHttp.port}/mcp`, tools: ["echo", "nonexistent"] }, plain: { url: "http://example.com/mcp" },
    broken: { command: "/nonexistent/kiln-test-mcp" }, off: { command: "true", disabled: true } },
  fetch,
}));
writeFileSync(LOG, "");
config();

let web: Bun.Subprocess<"ignore", "pipe", "inherit">, base = "", port = "";
beforeAll(async () => {
  web = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", KILN_URL: `http://127.0.0.1:${fake.port}`, KILN_HOST: "127.0.0.1", NODE_ENV: "production", KILN_MCP_CONFIG: CONFIG },
  });
  const reader = web.stdout.getReader(), dec = new TextDecoder();
  let out = "";
  while (!/kiln web: \S+/.test(out)) {
    const { value, done } = await reader.read();
    if (done) throw new Error(`server.ts exited: ${out}`);
    out += dec.decode(value);
  }
  base = out.match(/kiln web: (\S+)/)![1].replace(/\/$/, "");
  port = new URL(base).port;
});
afterAll(() => { web.kill(); fake.stop(true); site.stop(true); inner.stop(true); mcpHttp.stop(true); });

async function* events(res: Response) {   // a job's events as [position, data]
  const reader = res.body!.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += dec.decode(value, { stream: true });
    for (let end = buf.indexOf("\n\n"); end >= 0; end = buf.indexOf("\n\n")) {
      const block = buf.slice(0, end);
      buf = buf.slice(end + 2);
      yield [Number(block.match(/^id: (\d+)/m)![1]), block.match(/^data: (.*)$/m)![1]] as const;
    }
  }
}
const all = async (res: Response) => { const got: (readonly [number, string])[] = []; for await (const e of events(res)) got.push(e); return got; };
const json = { "Content-Type": "application/json" };
const post = (signal?: AbortSignal) =>
  fetch(`${base}/api/chat`, { method: "POST", body: JSON.stringify({ stream: true, messages: [] }), headers: json, signal });
const deltas = (got: (readonly [number, string])[]) => got.filter(([, d]) => d.includes('"delta"')).map(([, d]) => JSON.parse(d).choices[0].delta.content);

test("a streamed reply is a job: kiln's events in order, then the timings, then [DONE]", async () => {
  const res = await post();
  expect(res.headers.get("X-Kiln-Job")).toBeTruthy();
  const got = await all(res);
  expect(got.map(([i]) => i)).toEqual(got.map((_, i) => i));
  expect(deltas(got)).toEqual(Array.from({ length: CHUNKS }, (_, i) => `t${i}`));
  expect(JSON.parse(got.at(-2)![1]).timing.first).toBeGreaterThan(0);
  expect(got.at(-1)![1]).toBe("[DONE]");
});

test("a page that goes away leaves the reply running; following from a position gives the rest", async () => {
  const aborts = kilnAborted, ctrl = new AbortController();
  const res = await post(ctrl.signal), job = res.headers.get("X-Kiln-Job")!;
  let seen = 0;
  for await (const _ of events(res)) if (++seen === 3) break;
  ctrl.abort();
  await Bun.sleep(GAP_MS * (CHUNKS + 5));   // the reply ends with no page following it
  const rest = await all(await fetch(`${base}/api/chat/${job}?from=3`));
  expect(rest[0][0]).toBe(3);
  expect(deltas(rest)).toEqual(Array.from({ length: CHUNKS - 3 }, (_, i) => `t${i + 3}`));
  expect(rest.at(-1)![1]).toBe("[DONE]");
  expect(kilnAborted).toBe(aborts);
});

test("Stop ends the reply early with a stopped event, and stops kiln's request", async () => {
  const aborts = kilnAborted;
  const res = await post(), job = res.headers.get("X-Kiln-Job")!;
  const got: (readonly [number, string])[] = [];
  for await (const e of events(res)) {
    got.push(e);
    if (got.length === 2) expect((await fetch(`${base}/api/chat/${job}/stop`, { method: "POST" })).status).toBe(204);
  }
  expect(deltas(got).length).toBeLessThan(CHUNKS);
  expect(got.map(([, d]) => d)).toContain(JSON.stringify({ stopped: true }));
  expect(got.at(-1)![1]).toBe("[DONE]");
  await Bun.sleep(50);
  expect(kilnAborted).toBe(aborts + 1);
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

// A reply with tools: the script as the question, the chat's tool choice, the tools allowed without asking.
type Ev = Record<string, any>;
let asked = 0;
const ask = (script: Round[], tools: Record<string, { off?: string[] }> = { "mcp:fake": {} }, more: object = {}) => {
  const said = `run ${JSON.stringify(script)}\n${++asked}`;
  const res = fetch(`${base}/api/chat`, { method: "POST", headers: json, body: JSON.stringify({ stream: true, model: "m", messages: [{ role: "user", content: said }], kiln_web: { tools, ...more } }) });
  return { said, res };
};
async function* parsed(res: Response) { for await (const [, d] of events(res)) yield (d === "[DONE]" ? { done: true } : JSON.parse(d)) as Ev; }
const collect = async (res: Response) => { const got: Ev[] = []; for await (const e of parsed(res)) got.push(e); return got; };
const toolEvents = (got: Ev[]) => got.filter((e) => e.tool).map((e) => e.tool);
const results = (body: any) => body.messages.filter((m: any) => m.role === "tool").map((m: any) => m.content);
const catalog = async () => (await fetch(`${base}/api/tools`)).json() as Promise<{ sources: any[]; error?: string }>;
const sourceOf = async (id: string) => (await catalog()).sources.find((s) => s.id === id);
const decide = (job: string, call: string, decision: string) =>
  fetch(`${base}/api/chat/${job}/tool/${call}`, { method: "POST", headers: json, body: JSON.stringify({ decision }) });

test("an MCP server starts when a reply first needs it; its call and result go back to kiln exactly as kiln wrote them", async () => {
  expect((await sourceOf("mcp:fake")).state).toBe("idle");
  const args = '{"text":  "hi" }';                     // spaced as no serializer would: it must come back as it went
  const { said, res } = ask([{ calls: [{ name: "fake__echo", args }] }, { text: "Done." }], { "mcp:fake": {}, fetch: {} });
  const got = await collect(await res);
  const [first, second] = seen.get(said)!;
  expect(seen.get(said)!.length).toBe(2);
  expect(first.kiln_web).toBeUndefined();
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
  expect(done).toEqual(["call_0_1", "call_0_0"]);        // the quick one finished first
  expect(Date.now() - t).toBeLessThan(2000);
}, 30_000);

test("a tool that is not read-only waits for the user: allow once runs it, deny does not, always stops asking", async () => {
  // once
  let { said, res } = ask([{ calls: [{ name: "fake__write_note", args: '{"text": "first"}' }] }, { text: "ok" }]);
  let r = await res, job = r.headers.get("X-Kiln-Job")!, got: Ev[] = [];
  for await (const e of parsed(r)) {
    got.push(e);
    if (e.tool?.state === "approval") {
      expect(e.tool).toMatchObject({ id: "call_0_0", title: "write_note", readOnly: false });
      await Bun.sleep(200);
      expect(seen.get(said)!.length).toBe(1);             // nothing goes on while it waits
      expect((await decide(job, "call_0_0", "once")).status).toBe(204);
    }
  }
  expect(toolEvents(got).map((t) => t.state)).toEqual(["approval", "running", "done"]);
  expect(results(seen.get(said)![1])).toEqual(["saved: first"]);
  expect((await decide(job, "call_0_0", "once")).status).toBe(409);

  // deny
  ({ said, res } = ask([{ calls: [{ name: "fake__write_note", args: '{"text": "second"}' }] }, { text: "ok" }]));
  r = await res;
  job = r.headers.get("X-Kiln-Job")!;
  got = [];
  for await (const e of parsed(r)) { got.push(e); if (e.tool?.state === "approval") await decide(job, e.tool.id, "deny"); }
  expect(toolEvents(got).map((t) => t.state)).toEqual(["approval", "declined"]);
  expect(results(seen.get(said)![1])[0]).toContain("The user declined this call");
  expect(readFileSync(LOG, "utf8")).not.toContain("wrote second");

  // always: the reply's next call to it runs without asking
  const note = (t: string) => ({ calls: [{ name: "fake__write_note", args: JSON.stringify({ text: t }) }] });
  ({ said, res } = ask([note("third"), note("fourth"), { text: "ok" }]));
  r = await res;
  job = r.headers.get("X-Kiln-Job")!;
  got = [];
  for await (const e of parsed(r)) { got.push(e); if (e.tool?.state === "approval") await decide(job, e.tool.id, "always"); }
  expect(toolEvents(got).map((t) => t.state)).toEqual(["approval", "running", "done", "running", "done"]);
  expect(results(seen.get(said)![2])).toEqual(["saved: third", "saved: fourth"]);

  // allowed with the request (the page's "Always allow" setting)
  ({ said, res } = ask([note("fifth"), { text: "ok" }], { "mcp:fake": {} }, { allow: ["fake__write_note"] }));
  got = await collect(await res);
  expect(toolEvents(got).map((t) => t.state)).toEqual(["running", "done"]);
}, 30_000);

test("Stop during a tool call cancels the call on its server and ends the reply at once", async () => {
  const { said, res } = ask([{ calls: [{ name: "fake__slow", args: '{"ms": 2500}' }] }, { text: "never" }]);
  const r = await res, job = r.headers.get("X-Kiln-Job")!, got: Ev[] = [];
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
  const r = await res, job = r.headers.get("X-Kiln-Job")!, got: Ev[] = [];
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
  const r = await res, job = r.headers.get("X-Kiln-Job")!, early: (readonly [number, string])[] = [];
  for await (const e of events(r)) { early.push(e); if (early.length === 7) break; }   // gone in the middle of round 0
  await r.body!.cancel().catch(() => {});
  const rest = await all(await fetch(`${base}/api/chat/${job}?from=7`));
  const whole = await all(await fetch(`${base}/api/chat/${job}?from=0`));
  expect([...early, ...rest]).toEqual(whole);
  expect(whole.filter(([, d]) => d.includes('"round"')).length).toBe(2);
  expect(whole.filter(([, d]) => d.includes('"state":"done"')).length).toBe(2);
}, 30_000);

test("a reply stops after 24 rounds of tool calls, says so, and runs none of the last round's", async () => {
  const { said, res } = ask([{ calls: [{ name: "fake__echo", args: '{"text": "again"}' }] }]);
  const got = await collect(await res);
  expect(seen.get(said)!.length).toBe(24);
  const states = toolEvents(got).map((t) => t.state);
  expect(states.filter((s) => s === "done").length).toBe(23);
  expect(states.at(-1)).toBe("stopped");
  expect(got.some((e) => e.note?.includes("24 rounds"))).toBe(true);
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
  expect(by["mcp:narrow"].tools.map((t: any) => t.name)).toEqual(["echo"]);       // its config's "tools" only
  expect(by["mcp:plain"]).toMatchObject({ state: "failed", error: '"url" must be https (plain http only to this machine)' });
  expect(by["mcp:fake"].detail).toBe(`${process.execPath} fake-mcp.ts`);
  expect(JSON.stringify(sources)).not.toContain(LOG);                  // a server's environment stays here
  // a Streamable HTTP server's tools run like the others
  const { said, res } = ask([{ calls: [{ name: "web__echo", args: '{"text": "over http"}' }] }, { text: "ok" }], { "mcp:web": {} });
  await collect(await res);
  expect(results(seen.get(said)![1])).toEqual(["echo: over http"]);
}, 30_000);

const fetched = async (url: string, more: object = {}) => {
  const { said, res } = ask([{ calls: [{ name: "fetch_url", args: JSON.stringify({ url, ...more }) }] }, { text: "ok" }], { fetch: {} });
  const got = await collect(await res);
  return { result: results(seen.get(said)![1])[0] as string, state: toolEvents(got).at(-1)!.state };
};

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
    expect(page.result).toBe([`URL: http://127.0.0.1:${sitePort}/page`, "Status: 200 OK", "Content-Type: text/html", "Title: Kiln & tools", "",
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
    const plain = await run({}, 0);                                       // no length asked for: 20,000 at a time
    expect(plain.length).toBeLessThanOrEqual(20_000);
    expect(plain).toContain(" of 50,000: call fetch_url again with start=");
  } finally {
    config();
    await fetch(`${base}/api/tools/reload`, { method: "POST", headers: json });
  }
}, 30_000);
