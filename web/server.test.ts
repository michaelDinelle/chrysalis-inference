import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync } from "node:fs";
import { harness, CHUNKS, Ev, GAP_MS, NOTE, RUN, Round, all, ask, base, chrysalisAborted, collect, contents, decide, deltas, events, fake, json, outcome, parsed, port, post, seen, until, voiceServer, unique, web } from "./server-harness";

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

test("a reply, and the answer to a thread's question, must come as JSON: another site's form cannot send it", async () => {
  const form = { "Content-Type": "text/plain" };
  expect((await fetch(`${base}/api/chat`, { method: "POST", body: JSON.stringify({ stream: true, messages: [] }), headers: form })).status).toBe(415);
  expect((await fetch(`${base}/api/chat/x/tool/y`, { method: "POST", body: '{"decision":"once"}', headers: form })).status).toBe(415);
});

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
    expect(chunks.map((c) => [c.head.text, c.samples])).toEqual([["one two three ", 3600], ["four", 2400]]);
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

test("the preview of a reply's request is what Chrysalis is then sent for it", async () => {
  const said = `run ${JSON.stringify([{ text: "Hi." }])}\n${unique()}`;
  const body = { stream: true, model: "m", messages: [{ role: "system", content: "S" }, { role: "user", content: said }] };
  const preview = await (await fetch(`${base}/api/preview`, { method: "POST", headers: json, body: JSON.stringify(body) })).json();
  await collect(await fetch(`${base}/api/chat`, { method: "POST", headers: json, body: JSON.stringify(body) }));
  expect(preview.body).toEqual(seen.get(said)![0]);
  expect((await fetch(`${base}/api/preview`, { method: "POST", body: JSON.stringify(body) })).status).toBe(415);
}, 30_000);
