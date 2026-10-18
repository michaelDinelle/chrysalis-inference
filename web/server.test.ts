// The web server (server.ts) against a fake kiln: a streamed reply is a job that runs to its end whatever its page
// does, any page can follow it again from an event's position, only Stop ends it early (and stops kiln's request), and
// a request addressed to another site's name is refused (DNS rebinding). Nothing here reaches the real kiln or systemd.
import { afterAll, beforeAll, expect, test } from "bun:test";

const CHUNKS = 12, GAP_MS = 20;
let kilnAborted = 0;
const fake = Bun.serve({
  hostname: "127.0.0.1",
  port: 0,
  async fetch(req) {
    const path = new URL(req.url).pathname;
    if (path === "/kiln/stats") return Response.json({ model: "fake-model" });
    if (path !== "/v1/chat/completions") return new Response("not found", { status: 404 });
    if (!(await req.json()).stream) return Response.json({ choices: [{ message: { content: "a title" } }] });
    req.signal.addEventListener("abort", () => kilnAborted++);
    let i = 0;
    const enc = new TextEncoder();
    return new Response(new ReadableStream({
      async pull(ctl) {
        await Bun.sleep(GAP_MS);
        if (i < CHUNKS) ctl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: `t${i++}` } }] })}\n\n`));
        else { ctl.enqueue(enc.encode("data: [DONE]\n\n")); ctl.close(); }
      },
    }), { headers: { "Content-Type": "text/event-stream" } });
  },
});

let web: Bun.Subprocess<"ignore", "pipe", "inherit">, base = "", port = "";
beforeAll(async () => {
  web = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", KILN_URL: `http://127.0.0.1:${fake.port}`, KILN_HOST: "127.0.0.1", NODE_ENV: "production" },
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
afterAll(() => { web.kill(); fake.stop(true); });

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
const post = (signal?: AbortSignal) =>
  fetch(`${base}/api/chat`, { method: "POST", body: JSON.stringify({ stream: true, messages: [] }), headers: { "Content-Type": "application/json" }, signal });
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
