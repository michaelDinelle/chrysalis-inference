import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { hasCheckpoint } from "./wire";

const RUN = mkdtempSync(`${tmpdir()}/chrysalis-context-`);
const enc = new TextEncoder();
const sse = (o: unknown) => enc.encode(`data: ${typeof o === "string" ? o : JSON.stringify(o)}\n\n`);
const count = (body: any) => Math.ceil(`${JSON.stringify(body.messages).replace(/data:[^"]+/g, "")}${JSON.stringify(body.tools ?? "")}`.length / (engine.chars ?? 3));

interface Engine { chars?: number; window?: number; pool?: number; lanes?: number; listed?: number; think?: number; out?: (body: any, n: number) => Partial<Reply> | undefined; refuse?: (body: any, n: number) => Response | undefined; gap?: number }
interface Reply { text: string; reasoning: string; calls: { name: string; args: string }[]; finish: string; breaks: boolean; completion: number; streamed: number }
let engine: Engine = {};
let requests: any[] = [], summaries: any[] = [], rejected: string[] = [], busy = new Map<object, number>(), peak = 0, aborted = 0;

const said = (m: any) => typeof m.content === "string" ? m.content : (m.content ?? []).map((p: any) => p.text ?? "").join("");
const isSummary = (body: any) => /structured summary|Update the existing checkpoint/.test(body.messages.findLast((m: any) => m.role === "user") ? said(body.messages.findLast((m: any) => m.role === "user")) : "");
const CONTEXT_FULL = (n: number, w: number) => Response.json({ error: { message: `the request (${n} tokens) exceeds the available context size (${w} tokens), try increasing it`, type: "invalid_request_error", code: "context_length_exceeded" } }, { status: 400 });
const KV_FULL = () => Response.json({ error: { message: "the request exceeds the available context size (KV cache full), try increasing it", type: "invalid_request_error", code: "context_length_exceeded" } }, { status: 400 });

const hash = (s: string) => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 9973, 7);
const code = (book: string, n: number) => `CODE-${hash(book)}-${n}`;
const codes = (text: string) => [...new Set(text.match(/CODE-\d+-\d+/g) ?? [])];
function model(body: any): Partial<Reply> {
  if (isSummary(body)) return { text: `## Objective\n- answer the user\n\n## Findings\n${codes(body.messages.map(said).join("\n")).map((c) => `- ${c}`).join("\n")}\n\n## Next Move\n1. answer` };
  return { text: `You said: ${said(body.messages.at(-1))}` };
}

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

let web: Bun.Subprocess<"ignore", "pipe", "inherit">, base = "";
beforeAll(async () => {
  web = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", CHRYSALIS_URL: `http://127.0.0.1:${fake.port}`, CHRYSALIS_HOST: "127.0.0.1", NODE_ENV: "production",
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
  requests = []; summaries = []; rejected = []; busy = new Map(); peak = 0; aborted = 0;
  return `model-${++models}`;
}
async function reply(messages: object[], model: string, extra: object = {}): Promise<Ev[]> {
  const res = await fetch(`${base}/api/chat`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, stream: true, messages, ...extra }) });
  return blocks(await res.text());
}
const blocks = (text: string): Ev[] => text.split("\n\n").flatMap((b) => { const d = /^data: (.*)$/m.exec(b)?.[1]; return d && d !== "[DONE]" ? [JSON.parse(d)] : []; });
const user = (content: string) => ({ role: "user", content });
const long = () => Array.from({ length: 22 }, (_, i) => [user(`Question ${i} ${"words ".repeat(120)}`), { role: "assistant", content: `Answer ${i} ${"words ".repeat(120)}` }]).flat();
const steps = () => requests.filter((b) => !isSummary(b));

test("memory taken by something else is waited for: the reply says so, nothing is compacted, and it goes on when memory comes back", async () => {
  let refusals = 0;
  const model = setup({ refuse: () => (refusals++ < 4 ? KV_FULL() : undefined) });
  const got = await reply([user("hello")], model);
  expect(got.flatMap((e) => e.choices?.[0]?.delta?.content ?? []).join("")).toBe("You said: hello");
  expect(got.filter((e) => e.waiting).length).toBe(4);
  expect(summaries).toEqual([]);
}, 30_000);

test("with no window known, the backend's own refusal starts the same compaction", async () => {
  const seen = { n: 0 };
  const model = setup({ refuse: (b, n) => { if (n >= 7000) { seen.n++; return CONTEXT_FULL(n, 7000); } } });
  const got = await reply([...long(), user("and now?")], model);
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(got.flatMap((e) => e.choices?.[0]?.delta?.content ?? []).join("")).toStartWith("You said: ");
  expect(seen.n).toBeGreaterThan(0);
  expect(got.filter((e) => e.compacted).length).toBeGreaterThan(0);
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
      const got = await reply([...old, user("hello")], `listed-${owner}`);
      expect([owner, got.filter((e) => e.error), got.filter((e) => e.compacted).length]).toEqual([owner, [], 1]);
    } finally {
      await fetch(`${base}/api/backend`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: `127.0.0.1:${fake.port}/v1` }) });
      other.stop(true);
    }
  }
}, 30_000);

test("another backend's refusal in its own words starts it too", async () => {
  const words = ["This model's maximum context length is 7000 tokens. However, your messages resulted in 7100 tokens.", "the request exceeds the available context size"];
  for (const message of words) {
    const model = setup({ refuse: (b, n) => n >= 7000 ? Response.json({ error: { message } }, { status: 400 }) : undefined });
    const got = await reply([...long(), user("and now?")], model);
    expect([got.filter((e) => e.error), got.filter((e) => e.compacted).length]).toEqual([[], 1]);
  }
}, 30_000);

test("the reply limit the user set in a chat is theirs: the reply stops there for them to continue", async () => {
  const model = setup({ out: () => ({ text: "Cut", calls: [], finish: "length", completion: 50 }) });
  const got = await reply([user("hello")], model, { max_tokens: 50 });
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(got.filter((e) => e.choices?.[0]?.finish_reason).map((e) => e.choices[0].finish_reason)).toEqual(["length"]);
  expect(requests.length).toBe(1);
}, 30_000);

test("a broken stream is tried again: what was written stays, the model is told to go on, and the page sees the round end", async () => {
  let breaks = 1;
  const model = setup({ out: () => (breaks-- > 0 ? { text: "Partly written", calls: [], breaks: true } : undefined) });
  const got = await reply([user("hello")], model);
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(got.filter((e) => e.retry).map((e) => e.retry.attempt)).toEqual([1]);
  expect(got.findIndex((e) => e.round)).toBeLessThan(got.findIndex((e) => e.retry));
  const next = steps()[1];
  expect(next.messages.at(-2).content).toBe("Partly written");
  expect(said(next.messages.at(-1))).toContain("interrupted");
}, 30_000);

test("a backend that is going to sleep or overloaded is tried again; one that refuses for good is not", async () => {
  let asleep = 2;
  const model = setup({ refuse: () => asleep-- > 0 ? Response.json({ error: { message: "the machine is going to sleep; Chrysalis loads the model again when it wakes", code: "unavailable" } }, { status: 503 }) : undefined });
  const got = await reply([user("hello")], model);
  expect(got.filter((e) => e.retry).length).toBe(2);
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(got.flatMap((e) => e.choices?.[0]?.delta?.content ?? []).join("")).toBe("You said: hello");

  const refused = setup({ refuse: () => Response.json({ error: { message: "bad key" } }, { status: 401 }) });
  const end = await reply([user("hello")], refused);
  expect(end.filter((e) => e.retry)).toEqual([]);
  expect(end.filter((e) => e.error).map((e) => e.error.message)).toEqual(["bad key"]);
  expect(requests.length).toBe(1);
}, 30_000);

test("a backend that stays down fails in the end, saying so", async () => {
  const model = setup({ refuse: () => Response.json({ error: { message: "overloaded" } }, { status: 500 }) });
  const got = await reply([user("hello")], model);
  expect(got.filter((e) => e.retry).length).toBe(10);
  expect(got.filter((e) => e.error).map((e) => e.error.message)).toEqual(["overloaded"]);
}, 30_000);

test("a summary that does not follow the form gets one reminder, then the failure is said plainly: nothing is cut to fit", async () => {
  const model = setup({ window: 6000, out: (b) => isSummary(b) || /did not fill in the required summary/.test(said(b.messages.at(-1))) ? { text: "I will read on.", calls: [] } : undefined });
  const got = await reply([...long(), user("and now?")], model);
  expect(got.filter((e) => e.error).map((e) => e.error.message).join()).toContain("did not write a summary");
  expect(summaries.map((b) => /did not fill in the required summary template/.test(said(b.messages.at(-1)))).slice(-2)).toEqual([false, true]);
  expect(summaries.filter((b) => /did not fill in the required summary template/.test(said(b.messages.at(-1)))).length).toBe(1);
}, 30_000);

test("a chat with no tools that reaches the ceiling compacts before it replies, and the new message goes whole, pictures and all", async () => {
  const model = setup({ window: 8000 });
  const old = Array.from({ length: 22 }, (_, i) => [user(`Question ${i} ${"words ".repeat(120)}`), { role: "assistant", content: `Answer ${i} ${"words ".repeat(120)}` }]).flat();
  const picture = { type: "image_url", image_url: { url: `data:image/png;base64,${"A".repeat(40000)}` } }, asked = { role: "user", content: [picture, { type: "text", text: "what is this?" }] };
  const got = await reply([{ role: "system", content: "Be brief." }, ...old, asked], model);
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(got.filter((e) => e.compacted).length).toBe(1);
  expect(got.find((e) => e.compacted).compacted).toMatchObject({ first: true, kept: 0 });
  const last = requests.at(-1);
  expect(last.messages.map((m: any) => m.role)).toEqual(["system", "user"]);
  expect(last.messages[1].content[0].text).toStartWith("<conversation-checkpoint>");
  expect(last.messages[1].content.slice(1)).toEqual(asked.content);
  expect(said(last.messages[1])).toContain("[Assistant]: Answer 21 ");
}, 30_000);

test("a Continue that breaks goes on from what it wrote, and is not sent as a Continue again", async () => {
  let once = true;
  const model = setup({ window: 100000, pool: 100000, lanes: 3, out: (b) => { if (b.continue_final_message && once) { once = false; return { text: "More of it.", calls: [], breaks: true }; } } });
  const got = await reply([user("hello"), { role: "assistant", content: "So far" }], model, { continue_final_message: true });
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(steps().length).toBe(2);
  expect(requests[1].continue_final_message).toBeUndefined();
  expect(requests[1].messages.map((m: any) => [m.role, m.content.slice(0, 12)])).toEqual([["user", "hello"], ["assistant", "More of it."], ["user", said(requests[1].messages.at(-1)).slice(0, 12)]]);
}, 30_000);

test("a message too big to shrink is not summarized: there is too little around it to be worth it", async () => {
  const model = setup({ window: 8000, chars: 5, out: (b) => (isSummary(b) ? undefined : { text: "That is all", calls: [] }) });
  const got = await reply([user("hi"), { role: "assistant", content: "hello" }, user("background ".repeat(2600))], model);
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(got.flatMap((e) => e.choices?.[0]?.delta?.content ?? []).join("")).toBe("That is all");
  expect(summaries).toEqual([]);
}, 30_000);

test("memory that never comes while nothing else runs does not hold a reply for ever: the reply says it cannot fit", async () => {
  const model = setup({ refuse: () => KV_FULL() }), got = await reply([user("hello")], model);
  expect(got.filter((e) => e.waiting).length).toBe(10);
  expect(got.filter((e) => e.error).map((e) => e.error.message)).toEqual(["this message is larger than the whole context of the model, so there is nothing to summarize"]);
}, 30_000);

test("replies waiting for memory do not count as holding it: two that can never fit both end", async () => {
  const model = setup({ refuse: () => KV_FULL() });
  const both = await Promise.all([reply([user("one")], model), reply([user("two")], model)]);
  expect(both.map((got) => got.filter((e) => e.error).length)).toEqual([1, 1]);
}, 10_000);

test("a reply waits for memory for as long as another reply is really holding it", async () => {
  const model = setup({ gap: 700, refuse: (b) => said(b.messages.at(-1)) === "after you" ? (slow ? KV_FULL() : undefined) : undefined });
  let slow = true;
  const other = reply([user("first")], model).then((r) => { slow = false; return r; });
  await arrived("first");
  const got = await reply([user("after you")], model);
  await other;
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(got.filter((e) => e.waiting).length).toBeGreaterThan(10);
  expect(got.flatMap((e) => e.choices?.[0]?.delta?.content ?? []).join("")).toBe("You said: after you");
}, 30_000);

test("a chat larger than the whole window is summarized in parts, oldest first, and no part is skipped", async () => {
  const model = setup({ window: 6000 });
  const old = Array.from({ length: 30 }, (_, i) => [user(`Note ${code("old", i + 1)} ${"words ".repeat(100)}`), { role: "assistant", content: `Noted ${"words ".repeat(100)}` }]).flat();
  const got = await reply([...old, user("hello")], model);
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(summaries.length).toBeGreaterThan(1);
  expect(codes(got.find((e) => e.compacted).compacted.summary).length).toBe(30);
  expect(rejected.length).toBeLessThanOrEqual(summaries.length);
}, 30_000);

test("a part accepted with no room left for its summary is made smaller, however the backend's window was learned", async () => {
  const model = setup({ refuse: (b, n) => n >= 3000 ? CONTEXT_FULL(n, 3000) : undefined, out: (b, n) => isSummary(b) && n >= 2200 ? { text: "## Objective\n- half of a", calls: [], finish: "length" } : undefined });
  const old = Array.from({ length: 12 }, (_, i) => [user(`Note ${code("old", i + 1)} ${"words ".repeat(60)}`), { role: "assistant", content: `Noted ${"words ".repeat(60)}` }]).flat();
  const got = await reply([...old, user("hello")], model);
  expect(got.filter((e) => e.error)).toEqual([]);
  expect(codes(got.find((e) => e.compacted).compacted.summary).length).toBe(12);
}, 30_000);

test("a message larger than the whole window cannot be summarized, and the reply says that", async () => {
  const model = setup({ window: 6000 });
  const got = await reply([user("hi"), { role: "assistant", content: "hello" }, user("words ".repeat(20000))], model);
  expect(got.filter((e) => e.error).length).toBe(1);
  expect(got.find((e) => e.error).error.message).toBe("what this reply has to keep whole is larger than the context of the model");
}, 30_000);

test("a finished round is kept as its words, not its tokens: a replay says the same thing in far fewer events, and one who stopped inside it starts over", async () => {
  let breaks = 1;
  const model = setup({ out: () => (breaks-- > 0 ? { reasoning: "thinking it over", streamed: 400, text: "Partly written", calls: [], breaks: true } : undefined) });
  const res = await fetch(`${base}/api/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model, stream: true, messages: [user("hello")] }) });
  const job = res.headers.get("X-Chrysalis-Job")!, live = blocks(await res.text());
  const whole = blocks(await (await fetch(`${base}/api/chat/${job}?from=0`)).text());
  const text = (list: Ev[], field: string) => list.flatMap((e) => e.choices?.[0]?.delta?.[field] ?? []).join("");
  expect(live.length).toBeGreaterThan(whole.length + 300);
  expect(text(whole, "content")).toBe(text(live, "content"));
  expect(text(whole, "reasoning_content")).toBe(text(live, "reasoning_content"));
  const inside = live.findIndex((e) => e.choices?.[0]?.delta?.reasoning_content) + 2;
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
