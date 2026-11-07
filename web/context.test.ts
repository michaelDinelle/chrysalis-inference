import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { hasCheckpoint } from "./wire";

const RUN = mkdtempSync(`${tmpdir()}/chrysalis-context-`);
const enc = new TextEncoder();
const sse = (o: unknown) => enc.encode(`data: ${typeof o === "string" ? o : JSON.stringify(o)}\n\n`);
const count = (body: any) => Math.ceil(`${JSON.stringify(body.messages).replace(/data:[^"]+/g, "")}${JSON.stringify(body.tools ?? "")}`.length / (engine.chars ?? 3));

interface Engine { verdict?: (body: any) => string; chars?: number; window?: number; pool?: number; lanes?: number; listed?: number; think?: number; out?: (body: any, n: number) => Partial<Reply> | undefined; refuse?: (body: any, n: number) => Response | undefined; gap?: number }
interface Reply { text: string; reasoning: string; calls: { name: string; args: string }[]; finish: string; breaks: boolean; completion: number; streamed: number }
let engine: Engine = {};
let requests: any[] = [], summaries: any[] = [], reviews: any[] = [], rejected: string[] = [], busy = new Map<object, number>(), peak = 0, aborted = 0;

const said = (m: any) => typeof m.content === "string" ? m.content : (m.content ?? []).map((p: any) => p.text ?? "").join("");
const isSummary = (body: any) => /structured summary|Update the existing checkpoint/.test(body.messages.findLast((m: any) => m.role === "user") ? said(body.messages.findLast((m: any) => m.role === "user")) : "");
const isReview = (body: any) => /You are now the reviewer/.test(said(body.messages.at(-1) ?? {}));
const codes = (text: string) => [...new Set(text.match(/CODE-\d+-\d+/g) ?? [])];
const CONTEXT_FULL = (n: number, w: number) => Response.json({ error: { message: `the request (${n} tokens) exceeds the available context size (${w} tokens), try increasing it`, type: "invalid_request_error", code: "context_length_exceeded" } }, { status: 400 });
const KV_FULL = () => Response.json({ error: { message: "the request exceeds the available context size (KV cache full), try increasing it", type: "invalid_request_error", code: "context_length_exceeded" } }, { status: 400 });

function model(body: any): Partial<Reply> {
  const all = body.messages.map(said).join("\n");
  if (isReview(body)) return { text: engine.verdict?.(body) ?? '{"verdict":"done","reason":"it is all there"}' };
  if (isSummary(body)) return { text: `## Objective\n- finish the reading\n\n## Findings\n${codes(all).map((c) => `- ${c}`).join("\n")}\n\n## Next Move\n1. read on` };
  const task = /^You are a sub-agent|^direct/.test(said(body.messages[0])) ? /read (\w+) parts 1 to (\d+)/.exec(all) : null;
  if (!task) {
    const asked = /spawn: (.*)$/m.exec(all)?.[1];
    if (asked && !body.messages.some((m: any) => m.role === "tool")) return { calls: JSON.parse(asked).map((prompt: string, i: number) => ({ name: "subagent", args: JSON.stringify({ description: `Reader ${i}`, prompt }) })) };
    return { text: body.messages.some((m: any) => m.role === "tool") ? `Done: ${body.messages.filter((m: any) => m.role === "tool").map(said).join(" | ")}` : `You said: ${said(body.messages.at(-1))}` };
  }
  const [, book, n] = task, have = new Set(codes(all));
  for (let i = 1; i <= Number(n); i++) if (!have.has(code(book, i))) {
    return { text: `Reading part ${i}.`, calls: [{ name: "fake__read_part", args: JSON.stringify({ book, part: i }) }] };
  }
  return { text: `All of ${book}: ${Array.from({ length: Number(n) }, (_, i) => code(book, i + 1)).join(" ")}` };
}
const hash = (s: string) => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 9973, 7);
const code = (book: string, n: number) => `CODE-${hash(book)}-${n}`;

const fake = Bun.serve({
  hostname: "127.0.0.1", port: 0, idleTimeout: 0,
  async fetch(req) {
    const path = new URL(req.url).pathname;
    if (path === "/chrysalis/stats") return Response.json({ model: "fake", context: engine.window, kv_tokens: engine.pool, cache: engine.lanes ? { slots: Array(engine.lanes).fill(0) } : undefined });
    if (path === "/v1/models") return Response.json({ data: [{ id: "listed", owned_by: "someone", context_length: engine.listed }] });
    if (path !== "/v1/chat/completions") return new Response("not found", { status: 404 });
    const body = await req.json(), n = count(body), self = {};
    requests.push(body);
    if (isSummary(body)) summaries.push(body);
    if (isReview(body)) reviews.push(body);
    const refusal = engine.refuse?.(body, n);
    if (refusal) return refusal;
    if (engine.window && n >= engine.window) { rejected.push("window"); return CONTEXT_FULL(n, engine.window); }
    if (engine.pool && [...busy.values()].reduce((a, b) => a + b, 0) + n > engine.pool) { rejected.push("memory"); return KV_FULL(); }
    busy.set(self, n);
    peak = Math.max(peak, [...busy.values()].reduce((a, b) => a + b, 0));
    req.signal.addEventListener("abort", () => { aborted++; busy.delete(self); });
    const r = { text: "", reasoning: "", calls: [], finish: "", breaks: false, completion: 0, streamed: 0, ...model(body), ...engine.out?.(body, n) };
    r.finish ||= r.calls.length ? "tool_calls" : "stop";
    const room = engine.window ? engine.window - n - 1 : Infinity, words = Math.ceil((r.text.length + r.reasoning.length) / 3);
    const completion = r.completion || Math.min(words, room);
    return new Response(new ReadableStream({
      async start(ctl) {
        await Bun.sleep(engine.gap ?? 1);
        ctl.enqueue(sse({ choices: [{ delta: { role: "assistant", content: "" } }] }));
        ctl.enqueue(sse({ choices: [{ delta: {} }], chrysalis_prompt: { prompt_tokens: n, cached_tokens: 0 } }));
        if (r.reasoning) for (let k = 0; k < (r.streamed || 1); k++) ctl.enqueue(sse({ choices: [{ delta: { reasoning_content: r.streamed ? `${k} ` : r.reasoning } }] }));
        if (r.text) ctl.enqueue(sse({ choices: [{ delta: { content: r.text } }] }));
        if (r.breaks) {
          await Bun.sleep(20);
          busy.delete(self);
          return ctl.error(new Error("Chrysalis dropped the connection"));
        }
        r.calls.forEach((c: any, i: number) => ctl.enqueue(sse({ choices: [{ delta: { tool_calls: [{ index: i, id: `call_${requests.length}_${i}`, type: "function", function: { name: c.name, arguments: c.args } }] } }] })));
        await Bun.sleep(engine.gap ?? 1);
        busy.delete(self);
        ctl.enqueue(sse({ choices: [{ delta: {}, finish_reason: r.finish }] }));
        ctl.enqueue(sse({ choices: [], usage: { prompt_tokens: n, completion_tokens: completion, total_tokens: n + completion } }));
        ctl.enqueue(sse("[DONE]"));
        ctl.close();
      },
    }), { headers: { "Content-Type": "text/event-stream" } });
  },
});

const dir = mkdtempSync(`${RUN}/tools-`), CONFIG = `${dir}/mcp.json`;
writeFileSync(CONFIG, JSON.stringify({ mcpServers: { fake: { command: process.execPath, args: ["fake-mcp.ts"], cwd: import.meta.dir, env: { FAKE_MCP_PART_CHARS: "1500" } } } }));
let web: Bun.Subprocess<"ignore", "pipe", "inherit">, base = "";
beforeAll(async () => {
  web = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", CHRYSALIS_URL: `http://127.0.0.1:${fake.port}`, CHRYSALIS_HOST: "127.0.0.1", NODE_ENV: "production", CHRYSALIS_MCP_CONFIG: CONFIG,
      CHRYSALIS_TLS_DIR: `${RUN}/tls`, CHRYSALIS_CHECK_PORT: "0", CHRYSALIS_RETRY_MS: "5" },
  });
  const reader = web.stdout.getReader(), dec = new TextDecoder();
  let out = "";
  while (!/chrysalis web: \S+/.test(out)) {
    const { value, done } = await reader.read();
    if (done) throw new Error(`server.ts exited: ${out}`);
    out += dec.decode(value);
  }
  base = out.match(/chrysalis web: (\S+)/)![1].replace(/\/$/, "");
});
afterAll(() => { web.kill(); fake.stop(true); rmSync(RUN, { recursive: true, force: true }); });

type Ev = any;
const arrived = async (text: string) => { while (!requests.some((b) => said(b.messages.at(-1)) === text)) await Bun.sleep(5); };
let models = 0;
function setup(e: Engine) {
  engine = e;
  requests = []; summaries = []; reviews = []; rejected = []; busy = new Map(); peak = 0; aborted = 0;
  return `model-${++models}`;
}
async function reply(messages: object[], model: string, extra: object = {}, tools: object | null = { agents: {}, "mcp:fake": {} }): Promise<Ev[]> {
  const res = await fetch(`${base}/api/chat`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, stream: true, messages, ...(tools ? { chrysalis_web: { tools } } : {}), ...extra }) });
  return blocks(await res.text());
}
const blocks = (text: string): Ev[] => text.split("\n\n").flatMap((b) => { const d = /^data: (.*)$/m.exec(b)?.[1]; return d && d !== "[DONE]" ? [JSON.parse(d)] : []; });
const user = (content: string) => ({ role: "user", content });
const reading = (book: string, parts: number) => `read ${book} parts 1 to ${parts}`;
const spawn = (...books: [string, number][]) => user(`spawn: ${JSON.stringify(books.map(([b, n]) => reading(b, n)))}`);
const inner = (got: Ev[]) => got.filter((e) => e.agent).map((e) => e.agent.ev);
const outcome = (got: Ev[]) => got.filter((e) => e.tool?.result !== undefined && e.tool.title === "Sub-agent").map((e) => e.tool);
const all = (book: string, parts: number) => `All of ${book}: ${Array.from({ length: parts }, (_, i) => code(book, i + 1)).join(" ")}`;
const steps = () => requests.filter((b) => !isSummary(b) && !isReview(b));
const atWork = (b: any) => !isReview(b) && !isSummary(b) && /^You are a sub-agent/.test(b.messages[0].content);
const missing = (b: any) => { const have = new Set(codes(b.messages.map(said).join("\n"))), book = /read (\w+) parts 1 to (\d+)/.exec(b.messages.map(said).join("\n"))!; return Array.from({ length: Number(book[2]) }, (_, i) => code(book[1], i + 1)).some((c) => !have.has(c)); };
const judged = (b: any) => missing(b) ? '{"verdict":"continue","reason":"parts are unread","next_step":"read the next part"}' : '{"verdict":"done","reason":"every part is read"}';

test("a sub-agent that outgrows the context compacts as it goes and finishes with everything it found, and nothing is refused", async () => {
  const model = setup({ window: 8000 }), got = await reply([spawn(["alpha", 30])], model);
  const [end] = outcome(got);
  expect(end.state).toBe("done");
  expect(end.result).toBe(all("alpha", 30));
  expect(summaries.length).toBeGreaterThan(1);
  expect(inner(got).filter((e) => e.compacted).length).toBe(summaries.length);
  expect(rejected).toEqual([]);
}, 30_000);

test("the summary is asked for over the very conversation the model just read, so it fits and comes from the cache", async () => {
  const model = setup({ window: 8000 });
  await reply([spawn(["alpha", 20])], model);
  for (const s of summaries) {
    const sent = s.messages.slice(0, -1), before = steps().findLast((b) => requests.indexOf(b) < requests.indexOf(s) && /^You are a sub-agent/.test(b.messages[0].content))!;
    expect(sent.length).toBeGreaterThan(1);
    expect(before.messages.slice(0, sent.length)).toEqual(sent);
    expect(s.tools).toEqual(before.tools);
  }
}, 30_000);

test("after compacting, the task is still there word for word, and results the model has not read yet stay whole", async () => {
  const model = setup({ window: 8000 });
  await reply([spawn(["alpha", 20])], model);
  const after = steps().filter((b) => /^You are a sub-agent/.test(b.messages[0].content) && hasCheckpoint(said(b.messages[1])))[0];
  expect(after.messages[0].role).toBe("system");
  expect(said(after.messages[1])).toStartWith(`${reading("alpha", 20)}\n\n<conversation-checkpoint>`);
  expect(after.messages.at(-1).role).toBe("tool");
  expect(after.messages.at(-1).content).toContain("CODE-");
  expect(after.messages.at(-1).content.length).toBeGreaterThan(1500);
  expect(after.messages.at(-2).tool_calls[0].id).toBe(after.messages.at(-1).tool_call_id);
}, 30_000);

test("three at once share the memory: each is held near its share of it, so they finish side by side", async () => {
  const model = setup({ window: 12000, pool: 12000, lanes: 3 }), got = await reply([spawn(["alpha", 12], ["beta", 12], ["gamma", 12])], model);
  expect(outcome(got).map((t) => t.result).sort()).toEqual([all("alpha", 12), all("beta", 12), all("gamma", 12)]);
  expect(summaries.length).toBeGreaterThan(2);
  expect(Math.max(...steps().map(count))).toBeLessThan(7000);
  expect(peak).toBeLessThanOrEqual(12000);
}, 60_000);

test("one alone has the whole pool, and is not held to a share", async () => {
  const model = setup({ window: 12000, pool: 12000, lanes: 3 });
  const got = await reply([spawn(["alpha", 6])], model);
  expect(outcome(got)[0].result).toBe(all("alpha", 6));
  expect(summaries.length).toBe(0);
}, 30_000);

test("memory taken by something else is waited for: the reply says so, nothing is compacted, and it goes on when memory comes back", async () => {
  let refusals = 0;
  const model = setup({ refuse: (b) => /^You are a sub-agent/.test(b.messages[0].content) && refusals++ < 4 ? KV_FULL() : undefined });
  const got = await reply([spawn(["alpha", 2])], model);
  expect(outcome(got)[0].result).toBe(all("alpha", 2));
  expect(inner(got).filter((e) => e.waiting).length).toBe(4);
  expect(summaries).toEqual([]);
}, 30_000);

test("with no window known, the backend's own refusal starts the same compaction", async () => {
  const seen = { n: 0 };
  const model = setup({ refuse: (b, n) => { if (n >= 7000) { seen.n++; return CONTEXT_FULL(n, 7000); } } });
  const got = await reply([spawn(["alpha", 20])], model);
  expect(outcome(got)[0].result).toBe(all("alpha", 20));
  expect(seen.n).toBeGreaterThan(0);
  expect(inner(got).filter((e) => e.compacted).length).toBeGreaterThan(0);
}, 30_000);

test("a backend that cuts a long chat without a word is asked for its context, in its own way, so the chat is compacted first", async () => {
  const windows: Record<string, (w: number) => [string, object]> = {
    llamacpp: (w) => ["/props", { default_generation_settings: { n_ctx: w } }],
    koboldcpp: (w) => ["/api/extra/true_max_context_length", { value: w }],
    organization_owner: (w) => ["/api/v0/models", { data: [{ id: "elsewhere", loaded_context_length: 1 }, { id: "listed-organization_owner", loaded_context_length: w, max_context_length: 999999 }] }],
    library: (w) => ["/api/ps", { models: [{ name: "listed-library", context_length: w }] }],
  };
  for (const [owner, answer] of Object.entries(windows)) {
    const [path, body] = answer(8000);
    setup({ window: undefined });
    const other = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(req) {
      const at = new URL(req.url).pathname;
      if (at === "/v1/models") return Response.json({ data: [{ id: `listed-${owner}`, owned_by: owner }] });
      return at === path ? Response.json(body) : at === "/v1/chat/completions" ? fake.fetch(req) : new Response("not found", { status: 404 });
    } });
    try {
      expect((await fetch(`${base}/api/backend`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: `127.0.0.1:${other.port}/v1` }) })).status).toBe(200);
      const old = Array.from({ length: 24 }, (_, i) => [user(`Question ${i} ${"words ".repeat(120)}`), { role: "assistant", content: `Answer ${i} ${"words ".repeat(120)}` }]).flat();
      const got = await reply([...old, user("hello")], `listed-${owner}`, {}, null);
      expect([owner, got.filter((e) => e.error), got.filter((e) => e.compacted).length]).toEqual([owner, [], 1]);
    } finally {
      await fetch(`${base}/api/backend`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: `127.0.0.1:${fake.port}/v1` }) });
      other.stop(true);
    }
  }
}, 30_000);

test("another backend's refusal in its own words starts it too, and its listed context is used before any refusal", async () => {
  const words = ["This model's maximum context length is 7000 tokens. However, your messages resulted in 7100 tokens.", "the request exceeds the available context size"];
  for (const message of words) {
    const model = setup({ refuse: (b, n) => n >= 7000 ? Response.json({ error: { message } }, { status: 400 }) : undefined });
    const got = await reply([spawn(["alpha", 15])], model);
    expect(outcome(got)[0].result).toBe(all("alpha", 15));
  }
}, 30_000);

test("a reply the context cut off is compacted and carried on, and its text is not lost", async () => {
  let cut = true;
  const model = setup({ window: 9000, out: (b, n) => { if (n > 5000 && cut && !isSummary(b) && /^You are a sub-agent/.test(b.messages[0].content)) { cut = false; return { text: "Halfway through.", calls: [], finish: "length", completion: 9000 - n - 1 }; } } });
  const got = await reply([spawn(["alpha", 14])], model);
  expect(outcome(got)[0].result).toBe(all("alpha", 14));
  expect(summaries.length).toBeGreaterThan(0);
  expect(steps().some((b) => /^You are a sub-agent/.test(b.messages[0].content) && b.messages.some((m: any) => m.role === "assistant" && m.content === "Halfway through."))).toBe(true);
}, 30_000);

test("a reply cut off by the memory others hold goes on: that is not the reply's fault, and it does not count against it", async () => {
  let cuts = 0;
  const model = setup({ window: 100000, pool: 100000, lanes: 3, out: (b) => { if (cuts++ < 5 && /^You are a sub-agent/.test(b.messages[0].content)) return { text: "", reasoning: "thought", calls: [], finish: "length", completion: 10 }; } });
  const got = await reply([spawn(["alpha", 1])], model);
  expect(outcome(got)[0].result).toBe(all("alpha", 1));
}, 30_000);

test("a sub-agent that cannot finish within its reply limit is told so twice, then fails saying why", async () => {
  const model = setup({ out: (b) => /^You are a sub-agent/.test(b.messages[0].content) ? { text: "Still writing", calls: [], finish: "length", completion: 50 } : undefined });
  const got = await reply([spawn(["alpha", 1])], model, { max_tokens: 50 });
  const [end] = outcome(got);
  expect(end.state).toBe("failed");
  expect(end.result).toContain("Max tokens setting");
  expect(steps().filter((b) => /^You are a sub-agent/.test(b.messages[0].content)).length).toBe(3);
  expect(said(steps().filter((b) => /^You are a sub-agent/.test(b.messages[0].content)).at(-1).messages.at(-1))).toContain("output token limit");
}, 30_000);

test("the reply limit the user set in a chat is theirs: the reply stops there for them to continue", async () => {
  const model = setup({ out: () => ({ text: "Cut", calls: [], finish: "length", completion: 50 }) });
  const got = await reply([user("hello")], model, { max_tokens: 50 });
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(got.filter((e) => e.choices?.[0]?.finish_reason).map((e) => e.choices[0].finish_reason)).toEqual(["length"]);
  expect(requests.length).toBe(1);
}, 30_000);

test("a sub-agent that says it will go on and stops is sent back to work by the reviewer, and finishes", async () => {
  let narrated = false;
  const model = setup({ verdict: judged, out: (b) => { if (atWork(b) && !narrated && codes(b.messages.map(said).join("")).length === 3) { narrated = true; return { text: "Parts 1 to 3 done. Continuing with part 4.", calls: [] }; } } });
  const got = await reply([spawn(["alpha", 8])], model);
  expect(outcome(got)[0].result).toBe(all("alpha", 8));
  expect(narrated).toBe(true);
  const after = steps().find((b) => atWork(b) && b.messages.at(-2)?.content === "Parts 1 to 3 done. Continuing with part 4.");
  expect(said(after.messages.at(-1))).toContain("Next step: read the next part");
  expect(inner(got).filter((e) => e.reviewing).length).toBe(2);
}, 30_000);

test("a chat's reply with tools that stops halfway goes on the same way, and the page sees its text kept as a round", async () => {
  let narrated = false;
  const model = setup({ verdict: judged, out: (b) => { if (!isReview(b) && !narrated && codes(b.messages.map(said).join("")).length === 2) { narrated = true; return { text: "Continuing with part 3.", calls: [] }; } } });
  const got = await reply([{ role: "system", content: "direct" }, user(reading("alpha", 5))], model, {}, { "mcp:fake": {} });
  const text = got.flatMap((e) => e.choices?.[0]?.delta?.content ?? []).join("");
  expect(text).toEndWith(all("alpha", 5));
  expect(text).toContain("Continuing with part 3.");
  expect(got.filter((e) => e.round).length).toBe(6);
  expect(got.filter((e) => e.error || e.note)).toEqual([]);
}, 30_000);

test("a finished run is reviewed once and left alone, and the review itself never reaches the page as words", async () => {
  const model = setup({}), got = await reply([spawn(["alpha", 2])], model);
  expect(outcome(got)[0].result).toBe(all("alpha", 2));
  expect(reviews.filter((b) => /^You are a sub-agent/.test(b.messages[0].content)).length).toBe(1);
  expect(JSON.stringify(got)).not.toContain("it is all there");
  const [review] = reviews, before = steps().findLast((b) => atWork(b));
  expect(review.messages.slice(0, before.messages.length)).toEqual(before.messages);
  expect(review.tools).toEqual(before.tools);
}, 30_000);

test("a model that keeps promising and does nothing is stopped after three tries, and the reply says so", async () => {
  const model = setup({ verdict: () => '{"verdict":"continue","reason":"parts are unread"}', out: (b) => atWork(b) && b.messages.some((m: any) => m.role === "tool") ? { text: "I will go on now.", calls: [] } : undefined });
  const got = await reply([spawn(["alpha", 3])], model);
  const [end] = outcome(got);
  expect(end.state).toBe("done");
  expect(end.result).toContain("made no progress in three tries");
  expect(inner(got).filter((e) => e.note).length).toBe(1);
  expect(reviews.filter((b) => /^You are a sub-agent/.test(b.messages[0].content)).length).toBe(3);
}, 30_000);

test("promises with real work between them are not a stall: a model that keeps stopping but keeps reading gets to the end", async () => {
  const promised = new Set<number>();
  const model = setup({ verdict: judged, out: (b) => { const have = codes(b.messages.map(said).join("")).length; if (atWork(b) && have && have < 10 && have % 2 === 0 && !promised.has(have)) { promised.add(have); return { text: `That is ${have} parts. Going on.`, calls: [] }; } } });
  const got = await reply([spawn(["alpha", 10])], model);
  expect(outcome(got)[0].result).toBe(all("alpha", 10));
  expect(promised.size).toBe(4);
  expect(inner(got).filter((e) => e.note)).toEqual([]);
}, 30_000);

test("a review that fails leaves the reply as it is", async () => {
  const model = setup({ refuse: (b) => isReview(b) ? Response.json({ error: { message: "bad request" } }, { status: 400 }) : undefined, out: (b) => atWork(b) && b.messages.some((m: any) => m.role === "tool") ? { text: "This is as far as it goes.", calls: [] } : undefined });
  const got = await reply([spawn(["alpha", 3])], model);
  expect(outcome(got)[0]).toMatchObject({ state: "done", result: "This is as far as it goes." });
}, 30_000);

test("a review that says blocked, or nothing usable, leaves the reply as it is", async () => {
  for (const verdict of ['{"verdict":"blocked","reason":"it needs a password"}', "I cannot say."]) {
    const model = setup({ verdict: () => verdict, out: (b) => atWork(b) && b.messages.some((m: any) => m.role === "tool") ? { text: "This is as far as it goes.", calls: [] } : undefined });
    const got = await reply([spawn(["alpha", 3])], model);
    expect(outcome(got)[0].result).toBe("This is as far as it goes.");
  }
}, 30_000);

test("a reply with no tools and a reply under the user's own limit are not reviewed; one that promises a tool it never called is sent back", async () => {
  const none = setup({});
  await reply([user("hello")], none, {}, null);
  expect(reviews).toEqual([]);
  const limited = setup({});
  await reply([user("hello")], limited, { max_tokens: 500 }, { "mcp:fake": {} });
  expect(reviews).toEqual([]);
  let promised = false;
  const promising = setup({ verdict: (b) => b.messages.some((m: any) => m.role === "tool") ? '{"verdict":"done","reason":"it ran"}' : '{"verdict":"continue","reason":"it only said it would look"}',
    out: (b) => { if (isReview(b)) return; if (!promised) { promised = true; return { text: "Let me look that up.", calls: [] }; } if (!b.messages.some((m: any) => m.role === "tool")) return { text: "", calls: [{ name: "fake__echo", args: '{"text": "hi"}' }] }; return { text: "Found it.", calls: [] }; } });
  const got = await reply([user("look it up")], promising, {}, { "mcp:fake": {} });
  expect(got.filter((e) => e.tool?.state === "done").length).toBe(1);
  expect(got.flatMap((e) => e.choices?.[0]?.delta?.content ?? []).join("")).toBe("Let me look that up.Found it.");
}, 30_000);

test("Stop reaches a run that is being reviewed", async () => {
  const model = setup({ gap: 400 });
  const res = await fetch(`${base}/api/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model, stream: true, messages: [{ role: "system", content: "direct" }, user(reading("alpha", 1))], chrysalis_web: { tools: { "mcp:fake": {} } } }) });
  const job = res.headers.get("X-Chrysalis-Job")!;
  while (!reviews.length) await Bun.sleep(20);
  await fetch(`${base}/api/chat/${job}/stop`, { method: "POST" });
  expect(await res.text()).toContain('"stopped":true');
}, 30_000);

test("a broken stream is tried again: what was written stays, the model is told to go on, and the page sees the round end", async () => {
  let breaks = 1;
  const model = setup({ out: (b) => /^You are a sub-agent/.test(b.messages[0].content) && breaks-- > 0 ? { text: "Partly written", calls: [], breaks: true } : undefined });
  const got = await reply([spawn(["alpha", 2])], model);
  expect(outcome(got)[0].result).toBe(all("alpha", 2));
  const ev = inner(got);
  expect(ev.filter((e) => e.retry).map((e) => e.retry.attempt)).toEqual([1]);
  expect(ev.findIndex((e) => e.round)).toBeLessThan(ev.findIndex((e) => e.retry));
  const next = steps().filter((b) => /^You are a sub-agent/.test(b.messages[0].content))[1];
  expect(next.messages.at(-2).content).toBe("Partly written");
  expect(said(next.messages.at(-1))).toContain("interrupted");
}, 30_000);

test("a backend that is going to sleep or overloaded is tried again; one that refuses for good is not", async () => {
  let asleep = 2;
  const model = setup({ refuse: () => asleep-- > 0 ? Response.json({ error: { message: "the machine is going to sleep; Chrysalis loads the model again when it wakes", code: "unavailable" } }, { status: 503 }) : undefined });
  const got = await reply([user("hello")], model, {}, null);
  expect(got.filter((e) => e.retry).length).toBe(2);
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(got.flatMap((e) => e.choices?.[0]?.delta?.content ?? []).join("")).toBe("You said: hello");

  const refused = setup({ refuse: () => Response.json({ error: { message: "bad key" } }, { status: 401 }) });
  const end = await reply([user("hello")], refused, {}, null);
  expect(end.filter((e) => e.retry)).toEqual([]);
  expect(end.filter((e) => e.error).map((e) => e.error.message)).toEqual(["bad key"]);
  expect(requests.length).toBe(1);
}, 30_000);

test("a backend that stays down fails in the end, saying so", async () => {
  const model = setup({ refuse: () => Response.json({ error: { message: "overloaded" } }, { status: 500 }) });
  const got = await reply([user("hello")], model, {}, null);
  expect(got.filter((e) => e.retry).length).toBe(10);
  expect(got.filter((e) => e.error).map((e) => e.error.message)).toEqual(["overloaded"]);
}, 30_000);

test("a summary that does not follow the form gets one reminder, then the failure is said plainly: nothing is cut to fit", async () => {
  const model = setup({ window: 6000, out: (b) => isSummary(b) || /did not fill in the required summary/.test(said(b.messages.at(-1))) ? { text: "I will read on.", calls: [] } : undefined });
  const got = await reply([spawn(["alpha", 30])], model);
  const [end] = outcome(got);
  expect(end.state).toBe("failed");
  expect(end.result).toContain("did not write a summary");
  expect(summaries.length).toBe(2);
  expect(said(summaries[1].messages.at(-1))).toContain("did not fill in the required summary template");
}, 30_000);

test("a chat with no tools that reaches the ceiling compacts before it replies, and the new message goes whole, pictures and all", async () => {
  const model = setup({ window: 8000 });
  const old = Array.from({ length: 22 }, (_, i) => [user(`Question ${i} ${"words ".repeat(120)}`), { role: "assistant", content: `Answer ${i} ${"words ".repeat(120)}` }]).flat();
  const picture = { type: "image_url", image_url: { url: `data:image/png;base64,${"A".repeat(40000)}` } }, asked = { role: "user", content: [picture, { type: "text", text: "what is this?" }] };
  const got = await reply([{ role: "system", content: "Be brief." }, ...old, asked], model, {}, null);
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(got.filter((e) => e.compacted).length).toBe(1);
  expect(got.find((e) => e.compacted).compacted).toMatchObject({ first: true, kept: 0 });
  const last = requests.at(-1);
  expect(last.messages.map((m: any) => m.role)).toEqual(["system", "user"]);
  expect(last.messages[1].content[0].text).toStartWith("<conversation-checkpoint>");
  expect(last.messages[1].content.slice(1)).toEqual(asked.content);
  expect(said(last.messages[1])).toContain("[Assistant]: Answer 21 ");
}, 30_000);

test("a Continue that is cut off or breaks goes on from what it wrote, and is not sent as a Continue again", async () => {
  for (const out of [{ text: "More of it.", calls: [], finish: "length", completion: 50 }, { text: "More of it.", calls: [], breaks: true }]) {
    let once = true;
    const model = setup({ window: 100000, pool: 100000, lanes: 3, out: (b) => { if (b.continue_final_message && once) { once = false; return out; } } });
    const got = await reply([user("hello"), { role: "assistant", content: "So far" }], model, { continue_final_message: true });
    expect(got.filter((e) => e.error)).toEqual([]);
    expect(steps().length).toBe(2);
    expect(requests[1].continue_final_message).toBeUndefined();
    expect(requests[1].messages.map((m: any) => [m.role, m.content.slice(0, 12)])).toEqual([["user", "hello"], ["assistant", "More of it."], ["user", said(requests[1].messages.at(-1)).slice(0, 12)]]);
  }
}, 30_000);

test("a message too big to shrink is not summarized over and over: there is too little around it to be worth it", async () => {
  let rounds = 0;
  const model = setup({ window: 8000, chars: 5, out: (b) => isSummary(b) ? undefined : rounds++ < 4 ? { text: "", calls: [{ name: "fake__echo", args: '{"text": "hi"}' }] } : { text: "That is all", calls: [] } });
  const got = await reply([user("hi"), { role: "assistant", content: "hello" }, user("background ".repeat(2600))], model);
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(got.flatMap((e) => e.choices?.[0]?.delta?.content ?? []).join("")).toBe("That is all");
  expect(summaries).toEqual([]);
}, 30_000);

test("memory that never comes while nothing else runs does not hold a reply for ever: the reply says it cannot fit", async () => {
  const model = setup({ refuse: () => KV_FULL() }), got = await reply([user("hello")], model, {}, null);
  expect(got.filter((e) => e.waiting).length).toBe(10);
  expect(got.filter((e) => e.error).map((e) => e.error.message)).toEqual(["this message is larger than the whole context of the model, so there is nothing to summarize"]);
}, 30_000);

test("replies waiting for memory do not count as holding it: two that can never fit both end", async () => {
  const model = setup({ refuse: () => KV_FULL() });
  const both = await Promise.all([reply([user("one")], model, {}, null), reply([user("two")], model, {}, null)]);
  expect(both.map((got) => got.filter((e) => e.error).length)).toEqual([1, 1]);
}, 10_000);

test("a reply waits for memory for as long as another reply is really holding it", async () => {
  const model = setup({ gap: 700, refuse: (b) => said(b.messages.at(-1)) === "after you" ? (slow ? KV_FULL() : undefined) : undefined });
  let slow = true;
  const other = reply([user("first")], model, {}, null).then((r) => { slow = false; return r; });
  await arrived("first");
  const got = await reply([user("after you")], model, {}, null);
  await other;
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(got.filter((e) => e.waiting).length).toBeGreaterThan(10);
  expect(got.flatMap((e) => e.choices?.[0]?.delta?.content ?? []).join("")).toBe("You said: after you");
}, 30_000);

test("a chat larger than the whole window is summarized in parts, oldest first, and no part is skipped", async () => {
  const model = setup({ window: 6000 });
  const old = Array.from({ length: 30 }, (_, i) => [user(`Note ${code("old", i + 1)} ${"words ".repeat(100)}`), { role: "assistant", content: `Noted ${"words ".repeat(100)}` }]).flat();
  const got = await reply([...old, user("hello")], model, {}, null);
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(summaries.length).toBeGreaterThan(1);
  expect(codes(got.find((e) => e.compacted).compacted.summary).length).toBe(30);
  expect(rejected.length).toBeLessThanOrEqual(summaries.length);
}, 30_000);

test("a part accepted with no room left for its summary is made smaller, however the backend's window was learned", async () => {
  const model = setup({ refuse: (b, n) => n >= 3000 ? CONTEXT_FULL(n, 3000) : undefined, out: (b, n) => isSummary(b) && n >= 2200 ? { text: "## Objective\n- half of a", calls: [], finish: "length" } : undefined });
  const old = Array.from({ length: 12 }, (_, i) => [user(`Note ${code("old", i + 1)} ${"words ".repeat(60)}`), { role: "assistant", content: `Noted ${"words ".repeat(60)}` }]).flat();
  const got = await reply([...old, user("hello")], model, {}, null);
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(codes(got.find((e) => e.compacted).compacted.summary).length).toBe(12);
}, 30_000);

test("a message larger than the whole window cannot be summarized, and the reply says that", async () => {
  const model = setup({ window: 6000 });
  const got = await reply([user("hi"), { role: "assistant", content: "hello" }, user("words ".repeat(20000))], model, {}, null);
  expect(got.filter((e) => e.error).length).toBe(1);
  expect(got.find((e) => e.error).error.message).toBe("what this reply has to keep whole is larger than the context of the model");
}, 30_000);

test("many calls at once share the room left: their results are sized so the round fits", async () => {
  const model = setup({ window: 8000, out: (b) => /^You are a sub-agent/.test(b.messages[0].content) && !b.messages.some((m: any) => m.role === "tool")
    ? { text: "", calls: Array.from({ length: 8 }, (_, i) => ({ name: "fake__big", args: `{"n": ${40000 + i}}` })) } : /^You are a sub-agent/.test(b.messages[0].content) ? { text: "ok", calls: [] } : undefined });
  const got = await reply([spawn(["alpha", 1])], model);
  expect(outcome(got)[0].result).toBe("ok");
  const results = steps().findLast((b) => /^You are a sub-agent/.test(b.messages[0].content)).messages.filter((m: any) => m.role === "tool");
  expect(results.length).toBe(8);
  for (const r of results) expect(r.content.length).toBeLessThan(3000);
  expect(rejected).toEqual([]);
}, 30_000);

test("a sub-agent's long answer is cut to what the parent can take, and says so", async () => {
  const model = setup({ out: (b) => /^You are a sub-agent/.test(b.messages[0].content) ? { text: "x".repeat(60000), calls: [] } : undefined });
  const got = await reply([spawn(["alpha", 1])], model);
  expect(outcome(got)[0].result).toEndWith("[The result was cut here: 40,000 of 60,000 characters.]");
}, 30_000);

test("a long run is not cut at a number of rounds", async () => {
  const model = setup({ window: 400000 }), got = await reply([spawn(["alpha", 40])], model);
  expect(outcome(got)[0].result).toBe(all("alpha", 40));
  expect(inner(got).filter((e) => e.round).length).toBe(40);
}, 30_000);

test("a finished round is kept as its words, not its tokens: a replay says the same thing in far fewer events, and one who stopped inside it starts over", async () => {
  const model = setup({ out: (b) => /^You are a sub-agent/.test(b.messages[0].content) ? { reasoning: "thinking it over", streamed: 200 } : undefined });
  const body = { model, stream: true, messages: [spawn(["alpha", 3])], chrysalis_web: { tools: { agents: {}, "mcp:fake": {} } } };
  const res = await fetch(`${base}/api/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const job = res.headers.get("X-Chrysalis-Job")!, live = blocks(await res.text());
  const whole = blocks(await (await fetch(`${base}/api/chat/${job}?from=0`)).text());
  const text = (list: Ev[], field: string) => list.flatMap((e) => e.agent?.ev?.choices?.[0]?.delta?.[field] ?? []).join("");
  expect(live.length).toBeGreaterThan(whole.length + 300);
  expect(text(whole, "content")).toBe(text(live, "content"));
  expect(text(whole, "reasoning_content")).toBe(text(live, "reasoning_content"));
  expect(whole.filter((e) => e.agent?.ev?.tool).length).toBe(live.filter((e) => e.agent?.ev?.tool).length);
  const inside = live.findIndex((e) => e.agent?.ev?.choices?.[0]?.delta?.reasoning_content) + 2;
  const resumed = blocks(await (await fetch(`${base}/api/chat/${job}?from=${inside}`)).text());
  expect(resumed[0]).toEqual({ reset: true });
  expect(resumed.slice(1)).toEqual(whole);
  const past = await (await fetch(`${base}/api/chat/${job}?from=${live.length - 2}`)).text();
  expect(past).not.toContain('"reset"');
  expect(blocks(past).at(-1).timing).toBeDefined();
}, 30_000);

test("Stop reaches a reply that is waiting for memory, and one that is compacting", async () => {
  const waiting = setup({ gap: 700, refuse: (b) => said(b.messages.at(-1)) === "hello" ? KV_FULL() : undefined });
  const holder = fetch(`${base}/api/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: waiting, stream: true, messages: [user("holding it")] }) }).then((r) => r.text());
  await arrived("holding it");
  const res = await fetch(`${base}/api/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: waiting, stream: true, messages: [user("hello")] }) });
  const job = res.headers.get("X-Chrysalis-Job")!, reader = res.body!.getReader(), dec = new TextDecoder();
  let seen = "";
  while (!seen.includes('"waiting"')) seen += dec.decode((await reader.read()).value);
  await fetch(`${base}/api/chat/${job}/stop`, { method: "POST" });
  for (let r = await reader.read(); !r.done; r = await reader.read()) seen += dec.decode(r.value);
  expect(seen).toContain('"stopped":true');
  await holder;

  const compacting = setup({ window: 8000, gap: 300 });
  const body = { model: compacting, stream: true, messages: [...Array.from({ length: 30 }, (_, i) => [user(`Question ${i} ${"words ".repeat(120)}`), { role: "assistant", content: `Answer ${"words ".repeat(120)}` }]).flat(), user("go on")] };
  const summarized = summaries.length, again = await fetch(`${base}/api/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const id = again.headers.get("X-Chrysalis-Job")!;
  while (summaries.length === summarized) await Bun.sleep(5);
  await fetch(`${base}/api/chat/${id}/stop`, { method: "POST" });
  const text = await again.text();
  expect(text).toContain('"compacting":true');
  expect(text).toContain('"stopped":true');
  expect(text).not.toContain('"compacted"');
  for (const t = Date.now(); !aborted && Date.now() - t < 5000; ) await Bun.sleep(10);
  expect(aborted).toBeGreaterThan(0);
}, 30_000);
