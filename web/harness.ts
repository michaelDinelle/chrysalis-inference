import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

export type Decision = "once" | "always" | "deny";
export interface Scene { cast: string[]; scenario?: string; name?: string; greeting?: string }
export interface Thread { id: string; title: string; dir: string; updated: number; idle: number; busy: boolean; archived: boolean; model?: string; scene?: Scene; machine?: string }
export interface Connector { name: string; state: typeof state; ensure: typeof ensure; stop: typeof stop; threads: typeof threads; models: typeof models; agents: typeof agents; transcript: typeof transcript;
  turn: typeof turn; say: typeof say; undo: typeof undo; fork: typeof fork; rewrite: typeof rewrite; change: typeof change; remove: typeof remove; routines: typeof routines; pause: typeof pause; cancel: typeof cancel }
export const name = "OpenCode";
export interface Model { id: string; name: string; provider: string; variants: string[]; straight?: string }
export interface Turn { session?: string; machine?: string; dir: string; text: string; model?: string; persona?: string; files?: { uri: string; name: string }[]; follow?: boolean; agent?: string;
  scene?: Scene; said?: string; next?: string; title?: string; sampling?: Record<string, unknown> }
export interface Routine { id: string; name: string; when: string; prompt: string; status: string; next?: number; runs: number }
export type Answer = Decision | "stop" | { answer: Record<string, unknown> };
export interface Step { content: string; reasoning?: string; calls: Call[] }
export interface Call { id: string; name: string; args: string; title: string; source: string; state: string; result?: string }
export interface Said { role: "user" | "assistant"; content: string; files?: { name: string; type: string; data: string }[]; sent?: string; oc?: string; ocs?: string[]; by?: string; reasoning?: string; rounds?: Step[]; calls?: Call[]; error?: string; stopped?: boolean }
type Listener = (type: string, data: any, from?: string) => void;

export const INSTALL = "curl -fsSL https://opencode.ai/v2/install | bash";
const SOURCE = name, PERSONA = "character", OURS = "chrysalis", FROM = "chrysalis-page";
const sceneOf = (s: any): Scene | undefined => (Array.isArray(s?.metadata?.[OURS]?.cast) ? s.metadata[OURS] : undefined);
const WITH_US = '"$@" & c=$!; trap "kill $c" TERM INT; while kill -0 "$CHRYSALIS_WEB_PID" 2>/dev/null && kill -0 $c 2>/dev/null; do sleep 1; done; kill $c 2>/dev/null';
const SKIPPED = "The user chose not to answer.", OLDEST = [2, 2, 4];
const THINKING = ["none", "low", "medium"].map((id) => ({ id, body: { reasoning_effort: id } }));

const binary = () => process.env.CHRYSALIS_OPENCODE ?? Bun.which("opencode") ?? [`${homedir()}/.opencode/bin/opencode`].find((p) => existsSync(p));
export type Send = (path: string, init: { method: string; headers: Record<string, string>; body?: string; signal?: AbortSignal }) => Promise<Response>;
export interface Machine { id: string; name: string; home: string; kind: string }
interface Host { id: string; url: string; auth: string; version: string; proc?: Bun.Subprocess; send?: Send; machine?: Machine }
let host: Host | undefined, starting: Promise<void> | undefined;
const away = new Map<string, Host>(), homes = new Map<string, string>(), plans = new Map<string, string>();
const gone = (): never => { throw new Error("That computer is not connected to Chrysalis"); };
const of = (machine?: string) => (machine ? away.get(machine) ?? gone() : host);
const on = (session?: string) => of(session ? homes.get(session) : undefined);
const live = (h: Host) => (h.id ? away.get(h.id) === h : host === h);
const listeners = new Map<string, Set<Listener>>(), running = new Map<string, Set<string>>(), early = new Map<string, [string, any, string?][]>(), roots = new Map<string, string>();
async function rootOf(id: string): Promise<string> {
  const known = roots.get(id);
  if (known) return known;
  const parent = await call("GET", `/api/session/${id}`, undefined, on(id)).then((r) => r.data?.parentID as string | undefined, () => undefined), root = parent ? await rootOf(parent) : id;
  roots.set(id, root);
  return root;
}
const TUNED = "chrysalis-sampling", tunings = new Map<string, Record<string, unknown> | undefined>();
let modelUrl: string | undefined;
let reached: (url: string) => boolean = () => false;
export const routeModel = (url: string, ours: (url: string) => boolean) => { modelUrl = url; reached = (u) => u.startsWith(`${url}/`) || ours(u); };
const via = (base: string) => `${modelUrl}/via/${Buffer.from(base).toString("base64url")}`;
const routes = new Map<string, string>();
export async function tuningOf(session: string): Promise<Record<string, unknown> | undefined> {
  const root = await rootOf(session);
  if (!tunings.has(root)) tunings.set(root, await call("GET", `/api/session/${root}`, undefined, on(root)).then((r) => r.data?.metadata?.[TUNED], () => undefined));
  return tunings.get(root);
}
let outside = false;
export const allowOutside = (on: boolean) => { outside = on; };
export const outsideAllowed = () => outside;
const local = (model: string) => model.startsWith("chrysalis/");
let woken: (session: string) => void = () => {};
export const onWake = (fn: (session: string) => void) => { woken = fn; };
process.on("exit", () => host?.proc?.kill());

export const state = () => (host ? { state: "on" as const, version: host.version } : { state: binary() || process.env.CHRYSALIS_OPENCODE_URL ? "off" as const : "absent" as const });

async function call(method: string, path: string, body: object | undefined, to: Host | undefined): Promise<any> {
  if (!to) throw new Error("OpenCode is not running");
  const init = { method, headers: { ...(to.send ? {} : { authorization: to.auth }), ...(body ? { "content-type": "application/json" } : {}) }, body: body && JSON.stringify(body) };
  const r = await (to.send ? to.send(path, init) : fetch(to.url + path, init));
  const text = await r.text();
  if (!r.ok) throw new Error(`OpenCode: ${(() => { try { const j = JSON.parse(text); return j.message ?? j.error?.message ?? text; } catch { return text; } })() || `HTTP ${r.status}`}`);
  return text ? JSON.parse(text) : undefined;
}

async function freePort() {
  const s = Bun.listen({ hostname: "127.0.0.1", port: 0, socket: { data() {} } }), port = s.port;
  s.stop(true);
  return port;
}

async function ourModels(chrysalis: string) {
  const listed = await fetch(`${chrysalis}/v1/models`, { signal: AbortSignal.timeout(3000) }).then((r) => r.json()).catch(() => ({ data: [] }));
  return Object.fromEntries((listed.data ?? []).filter((m: any) => typeof m?.id === "string" && !m.id.endsWith("-fast")).map((m: any) => [m.id, { variants: THINKING, capabilities: { input: ["text", "image", "pdf"] } }]));
}
export const awayConfig = async (chrysalis: string) => ({ experimental: { schedules: true }, enabled_providers: ["chrysalis"],
  providers: { chrysalis: { name: "Chrysalis", package: "@opencode/ai/providers/openai-compatible", settings: {}, models: await ourModels(chrysalis) } } });
export function ensure(chrysalis: string): Promise<void> {
  return (starting ??= (async () => {
    const outer = process.env.CHRYSALIS_OPENCODE_URL;
    if (outer) {
      host = await reach(outer, process.env.CHRYSALIS_OPENCODE_PASSWORD ?? "");
      return void watch(host);
    }
    const bin = binary();
    if (!bin) throw new Error("OpenCode is not installed on this computer");
    const models = await ourModels(chrysalis);
    for (;;) {
      const password = `${crypto.randomUUID()}${crypto.randomUUID()}`, port = await freePort(), { OPENCODE_SERVER_PASSWORD: _, BUN_BE_BUN: __, ...env } = process.env;
      const others: Record<string, any> = {};
      for (const [key, base] of outside ? routes : []) {
        const [provider, model] = key.split("\0");
        others[provider] ??= {};
        if (model) ((others[provider].models ??= {})[model] = { settings: { baseURL: via(base) } });
        else others[provider].settings = { baseURL: via(base) };
      }
      const proc = Bun.spawn(["sh", "-c", WITH_US, "sh", bin, "serve", "--hostname", "127.0.0.1", "--port", String(port)], { stdin: "ignore", stdout: "ignore", stderr: "ignore",
        env: { ...env, CHRYSALIS_WEB_PID: String(process.pid), OPENCODE_PASSWORD: password, OPENCODE_CONFIG_CONTENT: JSON.stringify({ experimental: { schedules: true }, ...(outside ? {} : { enabled_providers: ["chrysalis"] }), providers: { ...others, chrysalis: { name: "Chrysalis",
          package: "@opencode/ai/providers/openai-compatible", settings: { baseURL: `${modelUrl}/v1`, apiKey: "chrysalis" }, models } } }) } });
      const at = await reach(`http://127.0.0.1:${port}`, password, proc, Object.keys(models).length > 0);
      const [providers, all] = await Promise.all([call("GET", "/api/provider", undefined, at), call("GET", "/api/model", undefined, at)]);
      const bases = new Map<string, string>((providers.data ?? []).map((p: any) => [p.id, p.settings?.baseURL]));
      let more = false;
      for (const p of providers.data ?? []) if (p.id !== "chrysalis" && /^https?:/.test(p.settings?.baseURL ?? "") && !reached(p.settings.baseURL)) { routes.set(p.id, p.settings.baseURL); more = true; }
      for (const m of all.data ?? []) {
        const base = m.settings?.baseURL;
        if (m.providerID !== "chrysalis" && /^https?:/.test(base ?? "") && !reached(base) && base !== bases.get(m.providerID)) { routes.set(`${m.providerID}\0${m.id}`, base); more = true; }
      }
      if (!more) {
        void proc.exited.then(() => { if (host?.proc === proc) { host = undefined; starting = undefined; lost(at); } });
        host = at;
        return void watch(at);
      }
      proc.kill();
      await proc.exited;
    }
  })().catch((e) => { starting = undefined; throw e; }));
}

const tooOld = (version: string) => {
  const has = String(version).split(".").map(Number), short = OLDEST.findIndex((n, k) => has[k] !== n);
  return short >= 0 && !(has[short] > OLDEST[short]);
};
async function reach(url: string, password: string, proc?: Bun.Subprocess, local = false): Promise<Host> {
    const auth = `Basic ${Buffer.from(`opencode:${password}`, "utf8").toString("base64")}`;
    for (const deadline = Date.now() + 30_000; ; await Bun.sleep(150)) {
      const info = await fetch(`${url}/api/info`, { headers: { authorization: auth }, signal: AbortSignal.timeout(2000) }).then((r) => (r.ok ? r.json() : r.status === 401 ? 401 : null), () => null);
      if (info === 401) { proc?.kill(); throw new Error("OpenCode refused the password"); }
      if (info?.version) {
        if (tooOld(info.version)) { proc?.kill(); throw new Error(`OpenCode ${OLDEST.join(".")} or newer is needed and this is ${info.version}: run opencode upgrade on this computer`); }
        const listed = !proc || !local || await fetch(`${url}/api/model`, { headers: { authorization: auth } }).then((r) => r.json())
          .then((j) => (j.data ?? []).some((m: any) => m.providerID === "chrysalis"), () => false);
        if (listed) return { id: "", url, auth, version: info.version, proc };
      }
      if (Date.now() > deadline || (proc && proc.exitCode !== null)) { proc?.kill(); throw new Error("OpenCode did not start"); }
    }
}

export function stop() {
  const was = host;
  host = undefined;
  starting = undefined;
  was?.proc?.kill();
  if (was) lost(was);
}
function lost(h: Host) {
  const here = (id: string) => (homes.get(id) ?? "") === h.id;
  for (const id of [...running.keys()]) if (here(id)) running.delete(id);
  for (const [id, set] of [...listeners]) if (here(id)) for (const fn of [...set]) fn("lost", {});
}
export async function attach(machine: Machine, send: Send): Promise<void> {
  const info = await send("/api/info", { method: "GET", headers: {} }).then((r) => r.json());
  if (!info?.version || tooOld(info.version)) throw new Error(`OpenCode ${OLDEST.join(".")} or newer is needed on ${machine.name}: run opencode upgrade there`);
  detach(machine.id);
  const h: Host = { id: machine.id, url: "", auth: "", version: info.version, send, machine };
  away.set(machine.id, h);
  void watch(h);
  await threadsOf(h).catch(() => []);
}
export function detach(id: string) {
  const h = away.get(id);
  if (!h) return;
  away.delete(id);
  lost(h);
}
export const machines = () => [...away.values()].map((h) => ({ ...h.machine!, version: h.version }));
async function watch(h: Host) {
  while (live(h)) {
    try {
      const headers = { ...(h.send ? {} : { authorization: h.auth }), accept: "text/event-stream" };
      const r = await (h.send ? h.send("/api/event", { method: "GET", headers }) : fetch(`${h.url}/api/event`, { headers }));
      if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);
      const reader = r.body.getReader(), decoder = new TextDecoder();
      for (let buf = ""; ; ) {
        const { value, done } = await reader.read();
        if (!live(h)) return void reader.cancel().catch(() => {});
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        for (let end = buf.indexOf("\n\n"); end >= 0; end = buf.indexOf("\n\n")) {
          const data = buf.slice(0, end).split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n");
          buf = buf.slice(end + 2);
          if (!data) continue;
          let ev: any;
          try { ev = JSON.parse(data); } catch { continue; }
          const id = ev?.data?.sessionID ?? ev?.data?.form?.sessionID;
          if (typeof id !== "string") continue;
          if (h.id) homes.set(id, h.id);
          if (ev.type === "session.created") roots.set(id, ev.data.parentID ? await rootOf(ev.data.parentID) : id);
          const root = await rootOf(id), from = root === id ? undefined : id, to = listeners.get(root);
          if (ev.type === "session.execution.started") running.set(root, (running.get(root) ?? new Set()).add(id));
          if (/^session\.execution\.(succeeded|failed|interrupted)$/.test(ev.type)) { running.get(root)?.delete(id); if (!running.get(root)?.size) running.delete(root); }
          if (to) for (const fn of [...to]) fn(ev.type, ev.data, from);
          else if (ev.type === "session.execution.started" && !early.has(root)) { early.set(root, [[ev.type, ev.data, from]]); woken(root); }
          else early.get(root)?.push([ev.type, ev.data, from]);
        }
      }
    } catch {  }
    if (!live(h)) return;
    lost(h);
    await Bun.sleep(1000);
  }
}
function listen(id: string, fn: Listener) {
  const set = listeners.get(id) ?? new Set<Listener>();
  listeners.set(id, set.add(fn));
  for (const [type, data, from] of early.get(id) ?? []) fn(type, data, from);
  early.delete(id);
  return () => { set.delete(fn); if (!set.size) listeners.delete(id); };
}

const refOf = (model: string) => {
  const [name, variant] = model.split("#");
  return { providerID: name.slice(0, name.indexOf("/")), id: name.slice(name.indexOf("/") + 1), ...(variant ? { variant } : {}) };
};
const idOf = (m: any) => (m?.providerID && m?.id ? `${m.providerID}/${m.id}` : undefined);
const textOf = (content: any) => (Array.isArray(content) ? content.map((p: any) => p?.text ?? "").join("") : typeof content === "string" ? content : "");
const titled = (name: string) => name.replace(/[_-]+/g, " ").replace(/^./, (c) => c.toUpperCase());

export async function threads(): Promise<Thread[]> {
  return (await Promise.all([...(host ? [host] : []), ...away.values()].map((h) => threadsOf(h).catch(() => [])))).flat();
}
async function threadsOf(h: Host): Promise<Thread[]> {
  const out: Thread[] = [];
  for (let cursor: string | undefined, pages = 0; pages < 10; pages++) {
    const page = await call("GET", `/api/session?limit=100${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}`, undefined, h);
    for (const s of page.data ?? []) if (!s.parentID && s.location?.directory && (!h.id || s.metadata?.[FROM])) {
      if (h.id) homes.set(s.id, h.id);
      out.push({ id: s.id, title: s.title ?? "", dir: s.location.directory, updated: s.time?.updated ?? s.time?.created ?? 0, idle: s.time?.idle ?? 0, busy: running.has(s.id), archived: !!s.time?.archived, model: idOf(s.model),
        ...(sceneOf(s) ? { scene: sceneOf(s) } : {}), ...(h.id ? { machine: h.id } : {}) });
    }
    cursor = page.cursor?.next;
    if (!cursor || !page.data?.length) break;
  }
  return out;
}
export async function models(): Promise<Model[]> {
  const h = host ?? [...away.values()][0], [list, providers] = await Promise.all([call("GET", "/api/model", undefined, h), call("GET", "/api/provider", undefined, h)]);
  const names = new Map<string, string>((providers.data ?? []).map((p: any) => [p.id, p.name ?? p.id]));
  return (list.data ?? []).flatMap((m: any) => (idOf(m) && (outside || local(idOf(m)!)) ? [{ id: idOf(m)!, name: m.name ?? m.id, provider: names.get(m.providerID) ?? m.providerID,
    variants: (m.variants ?? []).map((v: any) => v.id), ...(/^https?:/.test(m.settings?.baseURL ?? "") && !reached(m.settings.baseURL) ? { straight: new URL(m.settings.baseURL).host } : {}) }] : []));
}
export async function agents(): Promise<{ id: string; name: string }[]> {
  return ((await call("GET", "/api/agent", undefined, host ?? [...away.values()][0])).data ?? []).filter((a: any) => a.mode !== "subagent" && !a.hidden).map((a: any) => ({ id: a.id, name: a.name ?? a.id }));
}
export const remove = (id: string) => call("DELETE", `/api/session/${id}`, undefined, on(id));
export const change = (id: string, to: { title?: string; archived?: boolean }) => call("PATCH", `/api/session/${id}`, to, on(id));
export const move = (id: string, directory: string) => call("POST", `/api/session/${id}/move`, { directory }, on(id));
export const say = (id: string, text: string) => call("POST", `/api/session/${id}/prompt`, { text }, on(id));
export const fork = async (id: string, before?: string) => {
  const made = (await call("POST", `/api/session/${id}/fork`, before ? { before } : {}, on(id))).data.id as string;
  if (homes.has(id)) {
    homes.set(made, homes.get(id)!);
    const was = (await call("GET", `/api/session/${made}`, undefined, on(id))).data?.metadata ?? {};
    await call("PATCH", `/api/session/${made}`, { metadata: { ...was, [FROM]: true } }, on(id));
  }
  return made;
};
export async function rewrite(id: string, to: { drop?: string[]; edit?: { id: string; text: string } }) {
  const go = (method: string, path: string, body?: object) => call(method, path, body, on(id));
  if (running.has(id)) throw new Error("OpenCode is still working in this chat");
  const was = (await go("GET", `/api/experimental/session/${id}/export`)).data, kept = (await go("GET", `/api/experimental/schedule?sessionID=${id}`).catch(() => ({ data: [] }))).data ?? [];
  const edited = (m: any) => (m.type === "user" ? { ...m, text: to.edit!.text, metadata: { ...m.metadata, [OURS]: { ...m.metadata?.[OURS], said: to.edit!.text } } }
    : { ...m, content: [...(m.content ?? []).filter((p: any) => p.type !== "text"), { type: "text", text: to.edit!.text }] });
  const messages = was.messages.filter((m: any) => !to.drop?.includes(m.id)).map((m: any) => (m.id === to.edit?.id ? edited(m) : m));
  await go("DELETE", `/api/session/${id}`);
  try { await go("POST", "/api/experimental/session/import", { info: was.info, messages, location: was.info.location }); }
  catch (e) { await go("POST", "/api/experimental/session/import", { ...was, location: was.info.location }); throw e; }
  const test = (c: any) => (c ? (c.type === "code" ? { code: c.code } : { run: c.run, ...(c.is !== undefined ? { is: c.is } : {}), ...(c.matches !== undefined ? { matches: c.matches } : {}), ...(c.changes ? { changes: true } : {}) }) : undefined);
  for (const r of kept.filter((r: any) => r.status !== "done")) {
    const tz = r.when.tz ? { tz: r.when.tz } : {}, when = r.when.type === "every" ? { every: r.when.every, ...tz } : r.when.type === "cron" ? { cron: r.when.cron, ...tz }
      : { at: new Date(r.nextDueAt ?? Date.now() + 60_000).toISOString() };
    const made = (await go("POST", "/api/experimental/schedule", { sessionID: id, ...(r.name ? { name: r.name } : {}), when, prompt: r.action.prompt, ...(r.if ? { if: test(r.if) } : {}), ...(r.until ? { until: test(r.until) } : {}) })).data;
    if (r.status === "paused") await go("POST", `/api/experimental/schedule/${made.id}/pause`);
  }
}
export async function undo(id: string, message: string) {
  const go = (method: string, path: string, body?: object) => call(method, path, body, on(id));
  await go("POST", `/api/session/${id}/revert/stage`, { messageID: message, files: true });
  await go("POST", `/api/session/${id}/revert/commit`);
}

const whenOf = (w: any) => (w.type === "every" ? `every ${w.every}` : w.type === "in" ? `once, ${w.in} after it was made` : w.type === "at" ? `once, at ${w.at}` : `cron ${w.cron}`);
export async function routines(session: string): Promise<Routine[]> {
  const list = (await call("GET", `/api/experimental/schedule?sessionID=${session}`, undefined, on(session))).data ?? [];
  for (const r of list) plans.set(r.id, homes.get(session) ?? "");
  return list.filter((r: any) => r.status !== "done").map((r: any) => ({ id: r.id, name: r.name ?? "", when: whenOf(r.when), prompt: r.action?.prompt ?? "", status: r.status, next: r.nextDueAt, runs: r.runCount ?? 0 }));
}
export const pause = (routine: string, paused: boolean) => call("POST", `/api/experimental/schedule/${routine}/${paused ? "pause" : "resume"}`, undefined, of(plans.get(routine)));
export const cancel = (routine: string) => call("DELETE", `/api/experimental/schedule/${routine}`, undefined, of(plans.get(routine)));

function callOf(p: any): Call {
  const s = p.state ?? {}, failed = s.status === "error", done = s.status === "completed";
  return { id: p.id, name: p.name, args: JSON.stringify(s.input ?? {}), title: titled(p.name), source: SOURCE, state: done ? "done" : failed ? "failed" : "stopped",
    ...(done || failed ? { result: typeof s.output === "string" ? s.output : textOf(s.content) || s.error?.message || (typeof s.error === "string" ? s.error : "") } : {}) };
}
export async function transcript(id: string): Promise<{ thread: Thread; messages: Said[] }> {
  const go = (method: string, path: string, body?: object) => call(method, path, body, on(id));
  const s = (await go("GET", `/api/session/${id}`)).data, all: any[] = [];
  for (let cursor: string | undefined; ; ) {
    const page = await go("GET", `/api/session/${id}/message?limit=200&${cursor ? `cursor=${encodeURIComponent(cursor)}` : "order=asc"}`);
    all.push(...(page.data ?? []));
    cursor = page.cursor?.next;
    if (!cursor || !page.data?.length) break;
  }
  const scene = sceneOf(s), messages: Said[] = scene?.greeting ? [{ role: "assistant", content: scene.greeting, ...(scene.cast.length === 1 ? { by: scene.cast[0] } : {}) }] : [];
  let steps: Step[] = [], ids: string[] = [], shown: string | undefined, by: string | undefined = scene?.cast.length === 1 ? scene.cast[0] : undefined;
  const close = (more: Partial<Said> = {}) => {
    if (!steps.length && !more.error) return;
    const last = steps.pop() ?? { content: "", calls: [] };
    messages.push({ role: "assistant", content: last.content, ...(shown ? { oc: shown } : {}), ocs: ids, ...(by ? { by } : {}), ...(last.reasoning ? { reasoning: last.reasoning } : {}), ...(last.calls.length ? { calls: last.calls } : {}),
      ...(steps.length ? { rounds: steps } : {}), ...more });
    steps = [];
    ids = [];
    shown = undefined;
  };
  for (const m of all) {
    if (m.type === "user") {
      close();
      const files = (m.files ?? []).filter((f: any) => typeof f.data === "string").map((f: any) => ({ name: f.name ?? "attachment", type: f.mime ?? "", data: f.data }));
      const ours = m.metadata?.[OURS];
      if (typeof ours?.next === "string") by = ours.next;
      messages.push({ role: "user", content: typeof ours?.said === "string" ? ours.said : m.text ?? "", oc: m.id, ocs: [m.id], ...(files.length ? { files } : {}), ...(m.metadata?.schedule ? { sent: `Routine${m.metadata.schedule.name ? `: ${m.metadata.schedule.name}` : ""}` } : m.metadata?.source === "subagent" ? { sent: `Sub-agent${m.metadata.agent ? `: ${m.metadata.agent}` : ""}` } : {}) });
    }
    else if (m.type === "assistant") {
      ids.push(m.id);
      shown = m.id;
      const parts: any[] = m.content ?? [], reasoning = parts.filter((p) => p.type === "reasoning").map((p) => p.text ?? "").join("");
      steps.push({ content: parts.filter((p) => p.type === "text").map((p) => p.text ?? "").join(""), ...(reasoning ? { reasoning } : {}), calls: parts.filter((p) => p.type === "tool").map(callOf) });
    } else if (m.type === "idle") { ids.push(m.id); close(m.outcome === "failed" ? { error: m.error?.message ?? "OpenCode could not finish this turn" } : m.outcome === "interrupted" ? { stopped: true } : {}); }
  }
  close();
  return { thread: { id: s.id, title: s.title ?? "", dir: s.location?.directory ?? "", updated: s.time?.updated ?? 0, idle: s.time?.idle ?? 0, busy: running.has(s.id), archived: !!s.time?.archived, model: idOf(s.model),
    ...(scene ? { scene } : {}), ...(homes.get(s.id) ? { machine: homes.get(s.id) } : {}) }, messages };
}

export async function folders(path: string, machine?: string) {
  if (machine) return call("GET", `/chrysalis/folders?path=${encodeURIComponent(path)}`, undefined, of(machine));
  const at = resolve(path || homedir());
  const dirs = readdirSync(at, { withFileTypes: true }).filter((e) => e.isDirectory() && !e.name.startsWith(".")).map((e) => e.name).sort((a, b) => a.localeCompare(b));
  return { path: at, up: dirname(at) === at ? undefined : dirname(at), home: homedir(), dirs: dirs.map((name) => ({ name, path: join(at, name) })) };
}

export async function turn(t: Turn, push: (d: object) => void, signal: AbortSignal, ask: (id: string) => Promise<Answer>, began: (session: string) => void) {
  const to = t.session ? on(t.session) : of(t.machine), go = (method: string, path: string, body?: object) => call(method, path, body, to);
  if (!t.follow && !to?.id && !existsSync(t.dir)) throw new Error(`There is no folder ${t.dir} on this computer`);
  const stored = t.model || !t.session || (outside && !to?.id) ? undefined : idOf((await go("GET", `/api/session/${t.session}`)).data?.model);
  for (const model of [t.model, stored]) if ((!outside || to?.id) && model && !local(model)) throw new Error(`${model.split("#")[0]} is outside this computer, and OpenCode here is kept to Chrysalis's models: allow models outside this computer from the OpenCode button, on the computer Chrysalis runs on`);
  const meta = t.scene || t.sampling || to?.id ? { metadata: { ...(to?.id ? { [FROM]: true } : {}), ...(t.scene ? { [OURS]: t.scene } : {}), ...(t.sampling ? { [TUNED]: t.sampling } : {}) } } : {};
  const id = t.session ?? (await go("POST", "/api/session", { location: { directory: t.dir }, ...(t.title?.trim() ? { title: t.title.trim().slice(0, 100) } : {}), ...meta })).data.id as string;
  if (to?.id) homes.set(id, to.id);
  if (t.sampling && !t.follow) tunings.set(id, t.sampling);
  began(id);
  if (t.session && (t.scene || t.sampling) && !t.follow) await go("PATCH", `/api/session/${id}`, meta);
  const entry = `/api/experimental/session/${id}/instructions/entries/${PERSONA}`;
  if (t.model && !t.follow) await go("POST", `/api/session/${id}/model`, { model: refOf(t.model) });
  if (t.agent && !t.follow) await go("POST", `/api/session/${id}/agent`, { agent: t.agent });
  if (t.persona && !t.follow) await go("PUT", entry, { value: t.persona });
  else if (t.session && !t.follow) await go("DELETE", entry).catch(() => {});
  const delta = (d: object) => push({ choices: [{ index: 0, delta: d }] });
  const index = new Map<string, number>(), names = new Map<string, string>(), since = new Map<string, number>(), declined = new Set<string>();
  let steps = 0;
  const known = (call: string, name = "tool", args?: string) => {
    if (index.has(call)) return;
    index.set(call, index.size);
    names.set(call, name);
    delta({ tool_calls: [{ index: index.get(call), id: call, type: "function", function: { name, arguments: args ?? "" } }] });
  };
  const tool = (call: string, state: string, more: object = {}) => push({ tool: { id: call, title: titled(names.get(call) ?? "tool"), source: SOURCE, state, ...more } });
  const ms = (call: string) => (since.has(call) ? Date.now() - since.get(call)! : undefined);
  const family = new Set<string>(), cards = new Map<string, string>();
  let stop = () => {};
  const done = new Promise<void>((resolve, reject) => {
    stop = listen(id, (type, d, from) => {
      const at = from ?? id;
      if (type === "lost") reject(new Error("The connection to OpenCode was lost: open the thread again to see where it is"));
      else if (type === "session.execution.started") family.add(at);
      else if (/^session\.execution\.(succeeded|failed|interrupted)$/.test(type)) {
        family.delete(at);
        if (!from && type.endsWith("failed")) reject(new Error(d.error?.message ?? "OpenCode could not finish this turn"));
        else if (!from && type.endsWith("interrupted") && !signal.aborted) reject(new Error(`OpenCode stopped this turn (${d.reason ?? "interrupted"})`));
        else if (!family.size) resolve();
      }
      else if (type === "permission.asked") {
        const callId = !from && d.source?.type === "tool" ? d.source.id as string : d.id as string, asks = JSON.stringify({ [d.action]: d.resources });
        known(callId, d.action, asks);
        tool(callId, "approval", { asks });
        void ask(callId).then((say) => {
          const allowed = say === "once" || say === "always";
          if (from) tool(callId, allowed ? "done" : "declined", { result: allowed ? "Allowed." : "Not run: it was not allowed." });
          else if (!allowed) { declined.add(callId); tool(callId, say === "deny" ? "declined" : "stopped", { result: "Not run: it was not allowed." }); }
          return go("POST", `/api/session/${at}/permission/${d.id}/reply`, { decision: allowed ? say : "reject" });
        }).catch(reject);
      }
      else if (type === "form.created") {
        const form = d.form, callId = from ? form.id as string : form.metadata?.tool?.id as string ?? form.id, path = `/api/session/${at}/form/${form.id}`;
        known(callId, "question", JSON.stringify({ question: form.title }));
        const put = async (error?: string): Promise<unknown> => {
          tool(callId, "asking", { form: { title: form.title, fields: form.fields, ...(error ? { error } : {}) } });
          const say = await ask(callId);
          if (typeof say !== "object") return go("DELETE", `${path}?message=${encodeURIComponent(SKIPPED)}`).finally(() => { if (from) tool(callId, "declined", { result: "Skipped." }); });
          return go("POST", `${path}/reply`, { answer: say.answer }).then(() => (from ? tool(callId, "done", { result: "Answered." }) : tool(callId, "running")), (e) => put(e instanceof Error ? e.message : String(e)));
        };
        void put().catch(() => {});
      }
      else if (type === "session.retry.scheduled") push({ retry: { message: d.error?.message ?? "The model did not answer" } });
      else if (from) {
        if (type === "session.tool.input.started") names.set(d.id, d.name);
        else if (type === "session.tool.called" && cards.has(from)) push({ tool: { id: cards.get(from), note: titled(names.get(d.id) ?? "tool") } });
      }
      else if (type === "session.step.started") { if (steps++) { push({ round: steps - 1 }); index.clear(); } }
      else if (type === "session.reasoning.delta") delta({ reasoning_content: d.delta });
      else if (type === "session.text.delta") delta({ content: d.delta });
      else if (type === "session.tool.input.started") known(d.id, d.name);
      else if (type === "session.tool.input.ended") delta({ tool_calls: [{ index: index.get(d.id), function: { arguments: d.text ?? "" } }] });
      else if (type === "session.tool.called") { known(d.id, undefined, JSON.stringify(d.input ?? {})); since.set(d.id, Date.now()); tool(d.id, "running"); }
      else if (type === "session.tool.progress") { if (typeof d.metadata?.sessionID === "string") cards.set(d.metadata.sessionID, d.id); }
      else if (type === "session.tool.success") tool(d.id, "done", { result: textOf(d.content), ms: ms(d.id) });
      else if (type === "session.tool.failed") { if (!declined.has(d.id)) tool(d.id, "failed", { result: `Error: ${d.error?.message ?? (textOf(d.content) || "the tool failed")}`, ms: ms(d.id) }); }
      else if (type === "session.step.ended" && d.tokens) push({ usage: { prompt_tokens: (d.tokens.input ?? 0) + (d.tokens.cache?.read ?? 0), completion_tokens: d.tokens.output ?? 0,
        prompt_tokens_details: { cached_tokens: d.tokens.cache?.read ?? 0 } } });
      else if (type === "session.compaction.started") push({ waiting: "Compacting the context" });
    });
  });
  const interrupt = () => { for (const one of new Set([id, ...family])) void go("POST", `/api/session/${one}/interrupt`).catch(() => {}); };
  signal.addEventListener("abort", interrupt, { once: true });
  try {
    const [, slash, rest] = /^\/(\S+)\s*([^]*)$/.exec(t.said ?? t.text) ?? [], files = t.files?.length ? { files: t.files } : {};
    const command = slash && ((await go("GET", `/api/command?directory=${encodeURIComponent(t.dir)}`)).data ?? []).some((c: any) => c.name === slash);
    const ours = t.said !== undefined || t.next ? { metadata: { [OURS]: { ...(t.said !== undefined ? { said: t.said } : {}), ...(t.next ? { next: t.next } : {}) } } } : {};
    if (command) await go("POST", `/api/session/${id}/command`, { name: slash, text: rest, ...files });
    else if (!t.follow) await go("POST", `/api/session/${id}/prompt`, { text: t.text, ...files, ...ours });
    if (signal.aborted) interrupt();
    await done;
  } finally {
    stop();
    signal.removeEventListener("abort", interrupt);
  }
  push({ idle: (await go("GET", `/api/session/${id}`).catch(() => null))?.data?.time?.idle ?? 0 });
  if (signal.aborted) throw new DOMException("stopped", "AbortError");
}
