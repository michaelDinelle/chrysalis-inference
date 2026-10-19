// The web server (server.ts) against a fake kiln: a streamed reply is a job that runs to its end whatever its page
// does, any page can follow it again from an event's position, only Stop ends it early (and stops kiln's request), and
// a request addressed to another site's name is refused (DNS rebinding). With tools, a reply is rounds: the fake kiln
// plays a script of tool calls, which server.ts runs on real MCP servers (fake-mcp.ts, over stdio and Streamable HTTP,
// through the SDK) and on the built-in Fetch URL against local sites, and sends back. Voice mode's socket goes to a
// speech service (fake-voice.ts) the server starts with the first socket and stops once idle. Nothing here reaches the
// real kiln or systemd.
import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { fakeMcp } from "./fake-mcp";
import { lanAddresses } from "./tls";

const CHUNKS = 12, GAP_MS = 20;
let kilnAborted = 0;
// A tool script, as the user's message: "run ", the rounds as JSON (each the calls kiln answers with, or its text), and
// a line that makes the message unique. Each round is told by the assistant turns the request already holds; every
// request a script gets is kept.
// A voice reply's rounds also say what kiln writes as its first step (step: after "<fixes>", up to the stop) and the
// calls it makes when held to the tools' grammar (forced).
interface Round { calls?: { name: string; args: string }[]; text?: string; gap?: number; step?: string; forced?: { name: string; args: string }[] }
const seen = new Map<string, any[]>(), steps = new Map<string, any[]>();
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
    // a continued reply's start is not one of its rounds; what was said, without a note before it
    const prefix = body.continue_final_message ? body.messages.at(-1) : undefined;
    const said: string = String(body.messages.find((m: any) => m.role === "user")?.content ?? "").replace(/^\[[\s\S]*?\]\n\n/, "");
    const script: Round[] | undefined = said.startsWith("run ") ? JSON.parse(said.slice(4).split("\n")[0]) : undefined;
    const k = body.messages.filter((m: any) => m.role === "assistant" && m !== prefix).length, r = script?.[Math.min(k, script.length - 1)];
    if (!body.stream && prefix?.content === "<fixes>" && r) {
      steps.set(said, [...(steps.get(said) ?? []), body]);
      return Response.json({ choices: [{ message: { content: r.step ?? "" }, finish_reason: "stop" }] });
    }
    if (!body.stream) return Response.json({ choices: [{ message: { content: "a title" } }] });
    req.signal.addEventListener("abort", () => kilnAborted++);
    if (r) {
      seen.set(said, [...(seen.get(said) ?? []), body]);
      const calls = body.tool_choice === "required" ? r.forced ?? [] : r.calls ?? [];
      const chunks = [...(prefix ? [{ delta: { kiln_prefix: { content: prefix.content } } }] : []),
        ...(body.tool_choice === "required" ? [] : [{ delta: { role: "assistant", content: "" } }, { delta: { reasoning_content: `thinking ${k}` } }, { delta: { content: r.text ?? `Round ${k}.` } }]),
        ...calls.map((c, i) => ({ delta: { tool_calls: [{ index: i, id: `call_${k}_${i}`, type: "function", function: { name: c.name, arguments: c.args } }] } })),
        { delta: {}, finish_reason: calls.length ? "tool_calls" : "stop" }];
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

// Bun's WebSocket takes request headers (an Origin, as a browser sends one); the DOM's type does not know them.
const Socket = WebSocket as unknown as new (url: string, o: { headers?: Record<string, string> }) => WebSocket;
// A server of its own, with voice mode's service set up as `env` says.
async function voiceServer(env: Record<string, string>) {
  const proc = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", KILN_URL: `http://127.0.0.1:${fake.port}`, KILN_HOST: "127.0.0.1", NODE_ENV: "production", KILN_MCP_CONFIG: CONFIG, ...env },
  });
  const reader = proc.stdout.getReader(), dec = new TextDecoder();
  let out = "";
  while (!/kiln web: \S+/.test(out)) {
    const { value, done } = await reader.read();
    if (done) throw new Error(`server.ts exited: ${out}`);
    out += dec.decode(value);
  }
  const url = out.match(/kiln web: (\S+)/)![1].replace(/\/$/, "");
  return { proc, url, socket: (origin?: string) => new Socket(`${url.replace(/^http/, "ws")}/api/voice`, origin ? { headers: { Origin: origin } } : {}) };
}
const outcome = (s: WebSocket) => new Promise<string>((res) => { s.onopen = () => res("open"); s.onclose = () => res("closed"); });
const until = async (ok: () => boolean, ms = 5000) => { for (const t = Date.now(); !ok() && Date.now() - t < ms; ) await Bun.sleep(25); return ok(); };

test("voice: only this server's own pages get a socket; it starts the service, passes both sides' frames through as they are, and stops the service once idle", async () => {
  const log = `${mkdtempSync(`${tmpdir()}/kiln-voice-`)}/voice.log`, events = () => {
    try { return readFileSync(log, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l)); } catch { return []; }
  };
  const v = await voiceServer({ KILN_VOICE_CMD: JSON.stringify([process.execPath, `${import.meta.dir}/fake-voice.ts`]), FAKE_VOICE_LOG: log, KILN_VOICE_IDLE_MS: "300" });
  try {
    expect(await outcome(v.socket("http://evil.example"))).toBe("closed");
    expect(await outcome(v.socket())).toBe("closed");
    expect(events()).toEqual([]);                    // a refused socket starts nothing
    const s = v.socket(v.url), got: (string | ArrayBuffer)[] = [];
    s.binaryType = "arraybuffer";
    s.onmessage = (e) => got.push(e.data);
    expect(await outcome(s)).toBe("open");
    s.send(JSON.stringify({ type: "hello", voice: "silent" }));   // before the service listens: held until it does
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

test("voice: a speech service that cannot start says how to install one", async () => {
  const v = await voiceServer({ KILN_VOICE_CMD: JSON.stringify(["/nonexistent/kiln-voice"]) });
  try {
    const s = v.socket(v.url), got: string[] = [];
    s.onmessage = (e) => got.push(e.data);
    expect(await outcome(s)).toBe("open");
    expect(await until(() => s.readyState === WebSocket.CLOSED)).toBe(true);
    expect(JSON.parse(got[0])).toMatchObject({ type: "error" });
    expect(JSON.parse(got[0]).message).toContain("make voice");
  } finally {
    v.proc.kill();
  }
});

test("on the network (KILN_HOST 0.0.0.0): this machine keeps plain HTTP; each address gets HTTPS whose certificate the server's own authority signs, and no plain HTTP; that authority vouches for nothing public", async () => {
  const dir = mkdtempSync(`${tmpdir()}/kiln-tls-`), ips = lanAddresses();
  expect(ips.length).toBeGreaterThan(0);
  const proc = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", KILN_URL: `http://127.0.0.1:${fake.port}`, KILN_HOST: "0.0.0.0", NODE_ENV: "production", KILN_MCP_CONFIG: CONFIG, KILN_TLS_DIR: dir },
  });
  try {
    const reader = proc.stdout.getReader(), dec = new TextDecoder();
    let out = "";
    while ((out.match(/kiln web: \S+/g) ?? []).length < 1 + ips.length) {
      const { value, done } = await reader.read();
      if (done) throw new Error(`server.ts exited: ${out}`);
      out += dec.decode(value);
    }
    const urls = [...out.matchAll(/kiln web: (\S+)/g)].map((m) => m[1].replace(/\/$/, "")), port = new URL(urls[0]).port;
    expect(urls[0]).toBe(`http://127.0.0.1:${port}`);
    expect(urls.slice(1)).toEqual(ips.map((ip) => `https://${ip}:${port}`));
    const ca = await (await fetch(`${urls[0]}/kiln-ca.crt`)).text();
    writeFileSync(`${dir}/ca-served.pem`, ca);
    // openssl checks the chain: Bun's BoringSSL has no IP name constraints ("unsupported name constraint type"), the
    // browsers' verifiers do
    const tlsCheck = (ip: string, ...with_: string[]) => Bun.spawnSync(["openssl", "s_client", "-connect", `${ip}:${port}`, "-verify_return_error",
      "-verify_ip", ip, ...with_], { stdin: new Uint8Array(), stdout: "pipe", stderr: "pipe" });
    for (const [k, u] of urls.slice(1).entries()) {
      const good = tlsCheck(ips[k], "-CAfile", `${dir}/ca-served.pem`);
      expect(good.stdout.toString()).toContain("Verify return code: 0 (ok)");
      expect(tlsCheck(ips[k]).exitCode).not.toBe(0);                       // nothing vouches for it without the authority
      expect((await fetch(`${u}/`, { tls: { rejectUnauthorized: false } })).status).toBe(200);
      expect(await fetch(`${u.replace("https:", "http:")}/`).then((r) => `plain ${r.status}`, () => "refused")).toBe("refused");
    }
    const sign = Bun.spawnSync(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-keyout", `${dir}/x.key`,
      "-out", `${dir}/x.pem`, "-days", "30", "-subj", "/CN=example.com", "-CA", `${dir}/ca.pem`, "-CAkey", `${dir}/ca.key`, "-addext", "subjectAltName=DNS:example.com"]);
    expect(sign.exitCode).toBe(0);
    const verify = Bun.spawnSync(["openssl", "verify", "-CAfile", `${dir}/ca-served.pem`, `${dir}/x.pem`], { stderr: "pipe" });
    expect(verify.exitCode).not.toBe(0);
    expect(verify.stdout.toString() + verify.stderr.toString()).toContain("permitted subtree violation");
  } finally {
    proc.kill();
  }
});

// Voice turns (kiln_voice): the reply's first step, at temperature 0, is what decides; the fake kiln plays a script's
// step and the calls it makes when held to the tools' grammar. A note before what was said speaks of searches itself.
const NOTE = "[The user is talking to you by voice. After the fixes, <tool>yes</tool> says whether answering needs a search.]\n\n";
const voiceAsk = (script: Round[], then = "") => {
  const said = `run ${JSON.stringify(script)}\n${++asked} ${then}`;
  const res = fetch(`${base}/api/chat`, { method: "POST", headers: json, body: JSON.stringify({ stream: true, model: "m", kiln_voice: true,
    messages: [{ role: "user", content: NOTE + said }], kiln_web: { tools: { "mcp:fake": {} }, route: true } }) });
  return { said, res };
};
const contents = (got: Ev[]) => got.map((e) => e.choices?.[0]?.delta?.content).filter(Boolean);

test("voice: the first step is the reply's own start at temperature 0; its fixes go to the page, and a \"no\" lets the reply go on from it", async () => {
  const { said, res } = voiceAsk([{ step: "tail => table</fixes>\n<tool>no", text: "Here it is." }]);
  const got = await collect(await res), [step] = steps.get(said)!, [reply] = seen.get(said)!;
  expect([step.temperature, step.stop, step.stream, step.continue_final_message, step.messages.at(-1)]).toEqual([0, ["</tool>"], false, true, { role: "assistant", content: "<fixes>" }]);
  expect(JSON.stringify(step.messages.slice(0, -1))).toBe(JSON.stringify(reply.messages.slice(0, -1)));     // the reply's own prompt
  expect(got.find((e) => e.heard)?.heard).toBe("<fixes>tail => table</fixes>\n<tool>no</tool>");
  expect(got.findIndex((e) => e.heard)).toBeLessThan(got.findIndex((e) => e.choices?.[0]?.delta?.content));
  expect([reply.continue_final_message, reply.messages.at(-1), reply.tool_choice]).toEqual([true, { role: "assistant", content: "<fixes>tail => table</fixes>\n<tool>no</tool>" }, undefined]);
  expect(seen.get(said)!.length).toBe(1);
  expect(toolEvents(got)).toEqual([]);
}, 30_000);

test("voice: a \"yes\" makes the step a turn of its own, then a call held to the tools' grammar that starts with the spoken line", async () => {
  const { said, res } = voiceAsk([{ step: "</fixes>\n<tool>yes" }, { forced: [{ name: "fake__echo", args: '{"text": "x"}' }] }, { text: "Done." }]);
  const got = await collect(await res), [forced, after] = seen.get(said)!;
  expect([forced.tool_choice, forced.logprobs, forced.continue_final_message]).toEqual(["required", false, undefined]);
  expect(forced.messages.at(-1)).toEqual({ role: "assistant", content: "<fixes></fixes>\n<tool>yes</tool>" });
  const [step, line, done] = contents(got);
  expect([step, done, contents(got).length]).toEqual(["<fixes></fixes>\n<tool>yes</tool>", "Done.", 3]);
  expect(line).toMatch(/^\S.*\.$/);                             // the spoken line the server writes before the call
  expect(toolEvents(got).map((t) => t.state)).toEqual(["running", "done"]);
  expect(after.messages.slice(-2)).toEqual([
    { role: "assistant", content: line, tool_calls: [{ id: "call_1_0", type: "function", function: { name: "fake__echo", arguments: '{"text": "x"}' } }] },
    { role: "tool", tool_call_id: "call_1_0", content: "echo: x" }]);
}, 30_000);

test("voice: a step that is not fix lines is left out, and the reply goes on from an empty one", async () => {
  const { said, res } = voiceAsk([{ step: "The user wants two games.\nIt could mean</fixes>\n<tool>no", text: "Here." }]);
  const got = await collect(await res), [reply] = seen.get(said)!;
  expect(got.find((e) => e.heard)?.heard).toBe("<fixes></fixes>");
  expect(reply.messages.at(-1)).toEqual({ role: "assistant", content: "<fixes></fixes>" });
}, 30_000);

test("voice: what was said decides an ask (\"search\"), not the note before it", async () => {
  const plain = voiceAsk([{ step: "</fixes>\n<tool>no", text: "Fine." }]);
  await collect(await plain.res);
  expect(seen.get(plain.said)!.map((b) => b.tool_choice)).toEqual([undefined]);
  const asked = voiceAsk([{ step: "</fixes>\n<tool>no" }, { forced: [{ name: "fake__echo", args: '{"text": "y"}' }] }, { text: "Found." }], "search for it");
  const got = await collect(await asked.res);
  expect(seen.get(asked.said)![0].tool_choice).toBe("required");
  expect(contents(got).slice(-1)).toEqual(["Found."]);
  expect(contents(got).length).toBe(3);                        // the step, the spoken line, the answer
}, 30_000);

test("the preview of a reply's request is what kiln is then sent for it, tools and all; a voice one says so", async () => {
  const said = `run ${JSON.stringify([{ text: "Hi." }])}\n${++asked}`;
  const body = { stream: true, model: "m", messages: [{ role: "system", content: "S" }, { role: "user", content: said }], kiln_web: { tools: { "mcp:fake": {}, fetch: {} } } };
  const preview = await (await fetch(`${base}/api/preview`, { method: "POST", headers: json, body: JSON.stringify(body) })).json();
  await collect(await fetch(`${base}/api/chat`, { method: "POST", headers: json, body: JSON.stringify(body) }));
  expect(preview.body).toEqual(seen.get(said)![0]);
  expect(preview.body.tools.map((t: any) => t.function.name)).toEqual(["fake__big", "fake__echo", "fake__get_weather", "fake__slow", "fake__write_note", "fetch_url"]);
  const voiced = await (await fetch(`${base}/api/preview`, { method: "POST", headers: json, body: JSON.stringify({ ...body, kiln_voice: true }) })).json();
  expect([voiced.voice, voiced.body.kiln_voice, preview.voice]).toEqual([true, undefined, false]);
  expect((await fetch(`${base}/api/preview`, { method: "POST", body: JSON.stringify(body) })).status).toBe(415);   // JSON only, as /api/chat
}, 30_000);
