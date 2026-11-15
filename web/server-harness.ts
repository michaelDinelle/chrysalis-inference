import { afterAll, beforeAll, expect } from "bun:test";
import { chmodSync, existsSync, mkdtempSync, rmSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";
import mdns from "multicast-dns";
import { advertise, discover } from "./link";
import { fingerprint, lanAddresses, netName, webTls } from "./tls";

export const RUN = mkdtempSync(`${tmpdir()}/chrysalis-test-`);
process.on("exit", () => rmSync(RUN, { recursive: true, force: true }));

process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";
process.env.CHRYSALIS_FIND_MS = "500";
process.env.CHRYSALIS_REFIND_MS = "0";
export const TLS = () => mkdtempSync(`${RUN}/tls-`);

export const CHUNKS = 12, GAP_MS = 20;
export let chrysalisAborted = 0;
export interface Round { calls?: { name: string; args: string }[]; text?: string; gap?: number; broken?: boolean; wait?: number }
export const seen = new Map<string, any[]>();
export const enc = new TextEncoder();
export const sse = (o: unknown) => enc.encode(`data: ${typeof o === "string" ? o : JSON.stringify(o)}\n\n`);
export let fake: ReturnType<typeof Bun.serve>;
const serveFake = () => Bun.serve({
  hostname: "127.0.0.1",
  port: 0,
  async fetch(req) {
    const path = new URL(req.url).pathname;
    if (path === "/chrysalis/stats") return Response.json({ model: "fake-model" });
    if (path !== "/v1/chat/completions") return new Response("not found", { status: 404 });
    const body = await req.json();
    if (/You are now the reviewer/.test(JSON.stringify(body.messages.at(-1)?.content ?? ""))) return new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: '{"verdict":"done","reason":"ok"}' }, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`, { headers: { "Content-Type": "text/event-stream" } });
    const prefix = body.continue_final_message ? body.messages.at(-1) : undefined;
    const said: string = String(body.messages.find((m: any) => m.role === "user")?.content ?? "").replace(/^\[[\s\S]*?\]\n\n/, "");
    const script: Round[] | undefined = said.startsWith("run ") ? JSON.parse(said.slice(4).split("\n")[0]) : undefined;
    const k = body.messages.filter((m: any) => m.role === "assistant" && m !== prefix).length, r = script?.[Math.min(k, script.length - 1)];
    if (!body.stream) return Response.json({ choices: [{ message: { content: "a title" } }] });
    req.signal.addEventListener("abort", () => chrysalisAborted++);
    if (r?.wait) await Bun.sleep(r.wait);
    if (r) {
      seen.set(said, [...(seen.get(said) ?? []), body]);
      const calls = r.calls ?? [];
      const chunks = [...(prefix ? [{ delta: { chrysalis_prefix: { content: prefix.content } } }] : []),
        ...(body.tool_choice === "required" ? [] : [{ delta: { role: "assistant", content: "" } }, { delta: { reasoning_content: `thinking ${k}` } }, { delta: { content: r.text ?? `Round ${k}.` } }]),
        ...calls.map((c, i) => ({ delta: { tool_calls: [{ index: i, id: `call_${k}_${i}`, type: "function", function: { name: c.name, arguments: c.args } }] } })),
        { delta: {}, finish_reason: calls.length ? "tool_calls" : "stop" }];
      let i = 0;
      return new Response(new ReadableStream({
        async pull(ctl) {
          await Bun.sleep(r.gap ?? 2);
          if (r.broken && i === chunks.length - 1) return ctl.error(new Error("Chrysalis dropped the connection"));
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

export let web: Bun.Subprocess<"ignore", "pipe", "inherit">, base = "", port = "";
async function start() {
  fake = serveFake();
  web = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", CHRYSALIS_URL: `http://127.0.0.1:${fake.port}`, CHRYSALIS_HOST: "127.0.0.1", NODE_ENV: "production",
      CHRYSALIS_TLS_DIR: TLS(), CHRYSALIS_CHECK_PORT: "0", CHRYSALIS_RETRY_MS: "5" },
  });
  const reader = web.stdout.getReader(), dec = new TextDecoder();
  let out = "";
  while (!/chrysalis web: \S+/.test(out)) {
    const { value, done } = await reader.read();
    if (done) throw new Error(`server.ts exited: ${out}`);
    out += dec.decode(value);
  }
  base = out.match(/chrysalis web: (\S+)/)![1].replace(/\/$/, "");
  port = new URL(base).port;
}
export function harness() {
  beforeAll(start);
  afterAll(async () => { web.kill(); await web.exited; fake.stop(true); });
}

export async function* events(res: Response) {
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
export const all = async (res: Response) => { const got: (readonly [number, string])[] = []; for await (const e of events(res)) got.push(e); return got; };
export const json = { "Content-Type": "application/json" };
export const post = (signal?: AbortSignal) =>
  fetch(`${base}/api/chat`, { method: "POST", body: JSON.stringify({ stream: true, messages: [] }), headers: json, signal });
export const deltas = (got: (readonly [number, string])[]) => got.filter(([, d]) => d.includes('"delta"')).map(([, d]) => JSON.parse(d).choices[0].delta.content);

export type Ev = Record<string, any>;

let asked = 0;
export const unique = () => ++asked;

export const ask = (script: Round[], more: object = {}) => {
  const said = `run ${JSON.stringify(script)}\n${++asked}`;
  const res = fetch(`${base}/api/chat`, { method: "POST", headers: json, body: JSON.stringify({ stream: true, model: "m", messages: [{ role: "user", content: said }], ...more }) });
  return { said, res };
};

export async function* parsed(res: Response) { for await (const [, d] of events(res)) yield (d === "[DONE]" ? { done: true } : JSON.parse(d)) as Ev; }

export const collect = async (res: Response) => { const got: Ev[] = []; for await (const e of parsed(res)) got.push(e); return got; };

export const decide = (job: string, call: string, decision: string) =>
  fetch(`${base}/api/chat/${job}/tool/${call}`, { method: "POST", headers: json, body: JSON.stringify({ decision }) });

export const Socket = WebSocket as unknown as new (url: string, o: { headers?: Record<string, string> }) => WebSocket;

export async function voiceServer(env: Record<string, string>) {
  const proc = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", CHRYSALIS_URL: `http://127.0.0.1:${fake.port}`, CHRYSALIS_HOST: "127.0.0.1", NODE_ENV: "production",
      CHRYSALIS_TLS_DIR: TLS(), CHRYSALIS_CHECK_PORT: "0", ...env },
  });
  const reader = proc.stdout.getReader(), dec = new TextDecoder();
  let out = "";
  while (!/chrysalis web: \S+/.test(out)) {
    const { value, done } = await reader.read();
    if (done) throw new Error(`server.ts exited: ${out}`);
    out += dec.decode(value);
  }
  const url = out.match(/chrysalis web: (\S+)/)![1].replace(/\/$/, "");
  return { proc, url, socket: (origin?: string) => new Socket(`${url.replace(/^http/, "ws")}/api/voice`, origin ? { headers: { Origin: origin } } : {}) };
}

export const outcome = (s: WebSocket) => new Promise<string>((res) => { s.onopen = () => res("open"); s.onclose = () => res("closed"); });

export const until = async (ok: () => boolean, ms = 5000) => { for (const t = Date.now(); !ok() && Date.now() - t < ms; ) await Bun.sleep(25); return ok(); };

export async function onNetwork(dir = TLS(), self = false) {
  const ips = lanAddresses();
  expect(ips.length).toBeGreaterThan(0);
  const proc = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PATH: mkdtempSync(`${RUN}/bin-`), PORT: "0", CHRYSALIS_URL: `http://127.0.0.1:${fake.port}`, CHRYSALIS_HOST: "0.0.0.0", NODE_ENV: "production",
      CHRYSALIS_SELF_AS_NETWORK: self ? "" : "1",
      CHRYSALIS_TLS_DIR: dir, CHRYSALIS_CHECK_PORT: "0", CHRYSALIS_LINK_PORT: "0", CHRYSALIS_VOICE_CMD: JSON.stringify([process.execPath, `${import.meta.dir}/fake-voice.ts`]),
      FAKE_VOICE_LOG: `${dir}/voice.log` },
  });
  const reader = proc.stdout.getReader(), dec = new TextDecoder();
  let out = "";
  while ((out.match(/chrysalis (web|check|link): \S+/g) ?? []).length < 1 + 3 * ips.length) {
    const { value, done } = await reader.read();
    if (done) throw new Error(`server.ts exited: ${out}`);
    out += dec.decode(value);
  }
  const strip = (m: RegExpExecArray) => m[1].replace(/\/$/, "");
  const checks = [...out.matchAll(/chrysalis check: (\S+)/g)].map(strip);
  return { proc, dir, ips, urls: [...out.matchAll(/chrysalis web: (\S+)/g)].map(strip), checks, links: [...out.matchAll(/chrysalis link: (\S+)/g)].map(strip) };
}

export function shown(ip: string, port: string, ca: string) {
  const out = Bun.spawnSync(["openssl", "s_client", "-connect", `${ip}:${port}`, "-verify_return_error", "-verify_ip", ip, "-CAfile", ca],
    { stdin: new Uint8Array(), stdout: "pipe", stderr: "pipe" }).stdout.toString();
  return { cert: out.match(/-----BEGIN CERTIFICATE-----[\s\S]*?-----END CERTIFICATE-----/)?.[0], ok: out.includes("Verify return code: 0 (ok)") };
}

export function announcement(fp: string): Promise<{ name: string; type: string; ttl?: number }[]> {
  return new Promise((done) => {
    const m = mdns(), got: { name: string; type: string; ttl?: number }[] = [];
    m.on("response", (r) => {
      const all = [...r.answers, ...(r.additionals ?? [])];
      if (all.some((x) => x.type === "TXT" && (x.data as Buffer[]).some((b) => String(b) === `fp=${fp}`))) got.push(...all);
    });
    m.query({ questions: [{ name: "_chrysalis._tcp.local", type: "PTR" }] });
    setTimeout(() => { m.destroy(); done(got); }, Number(process.env.CHRYSALIS_FIND_MS));
  });
}

export async function withoutEngine() {
  const free = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response() }), chrysalis = `http://127.0.0.1:${free.port}`, dir = TLS();
  free.stop(true);
  const proc = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", CHRYSALIS_URL: chrysalis, CHRYSALIS_HOST: "0.0.0.0", NODE_ENV: "production",
      CHRYSALIS_TLS_DIR: dir, CHRYSALIS_CHECK_PORT: "0", CHRYSALIS_LINK_PORT: "0" },
  });
  const reader = proc.stdout.getReader(), dec = new TextDecoder();
  let out = "";
  while (!/chrysalis link: \S+/.test(out)) {
    const { value, done } = await reader.read();
    if (done) throw new Error(`server.ts exited: ${out}`);
    out += dec.decode(value);
  }
  return { proc, dir, chrysalis, url: /chrysalis web: (\S+)/.exec(out)![1].replace(/\/$/, "") };
}

export const NOTE = "[The user is talking to you by voice. Only speak of a search once the call has returned.]\n\n";

export const contents = (got: Ev[]) => got.map((e) => e.choices?.[0]?.delta?.content).filter(Boolean);
