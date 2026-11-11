import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

export type Decision = "once" | "always" | "deny";
export interface Thread { id: string; title: string; dir: string; updated: number; busy: boolean; model?: string }
export interface Model { id: string; name: string; provider: string }
export interface Turn { session?: string; dir: string; text: string; model?: string; persona?: string; files?: { data: string; mime: string; name?: string }[] }
export interface Step { content: string; reasoning?: string; calls: Call[] }
export interface Call { id: string; name: string; args: string; title: string; source: string; state: string; result?: string }
export interface Said { role: "user" | "assistant"; content: string; reasoning?: string; rounds?: Step[]; calls?: Call[]; error?: string; stopped?: boolean }
type Listener = (type: string, data: any) => void;

export const INSTALL = "curl -fsSL https://opencode.ai/v2/install | bash";
const SOURCE = "OpenCode", PERSONA = "character";
const WITH_US = '"$@" & c=$!; trap "kill $c" TERM INT; while kill -0 "$CHRYSALIS_WEB_PID" 2>/dev/null && kill -0 $c 2>/dev/null; do sleep 1; done; kill $c 2>/dev/null';
const NO_FORMS = "Chrysalis's page cannot show this form: ask in your reply instead, and the user will answer in their next message.";

const binary = () => process.env.CHRYSALIS_OPENCODE ?? Bun.which("opencode") ?? [`${homedir()}/.opencode/bin/opencode`].find((p) => existsSync(p));
let host: { url: string; auth: string; version: string; proc?: Bun.Subprocess } | undefined, starting: Promise<void> | undefined;
const listeners = new Map<string, Set<Listener>>(), running = new Set<string>();
process.on("exit", () => host?.proc?.kill());

export const state = () => (host ? { state: "on" as const, version: host.version } : { state: binary() || process.env.CHRYSALIS_OPENCODE_URL ? "off" as const : "absent" as const });

async function call(method: string, path: string, body?: object): Promise<any> {
  if (!host) throw new Error("OpenCode is not running");
  const r = await fetch(host.url + path, { method, headers: { authorization: host.auth, ...(body ? { "content-type": "application/json" } : {}) }, body: body && JSON.stringify(body) });
  const text = await r.text();
  if (!r.ok) throw new Error(`OpenCode: ${(() => { try { const j = JSON.parse(text); return j.message ?? j.error?.message ?? text; } catch { return text; } })() || `HTTP ${r.status}`}`);
  return text ? JSON.parse(text) : undefined;
}

async function freePort() {
  const s = Bun.listen({ hostname: "127.0.0.1", port: 0, socket: { data() {} } }), port = s.port;
  s.stop(true);
  return port;
}

export function ensure(chrysalis: string): Promise<void> {
  return (starting ??= (async () => {
    const outer = process.env.CHRYSALIS_OPENCODE_URL, password = outer ? process.env.CHRYSALIS_OPENCODE_PASSWORD ?? "" : `${crypto.randomUUID()}${crypto.randomUUID()}`;
    let url = outer, proc: Bun.Subprocess | undefined, local = 0;
    if (!url) {
      const bin = binary();
      if (!bin) throw new Error("OpenCode is not installed on this computer");
      const listed = await fetch(`${chrysalis}/v1/models`, { signal: AbortSignal.timeout(3000) }).then((r) => r.json(), () => ({ data: [] }));
      const models = Object.fromEntries((listed.data ?? []).filter((m: any) => typeof m?.id === "string" && !m.id.endsWith("-fast")).map((m: any) => [m.id, {}]));
      const { OPENCODE_SERVER_PASSWORD: _, ...env } = process.env, port = await freePort();
      local = Object.keys(models).length;
      proc = Bun.spawn(["sh", "-c", WITH_US, "sh", bin, "serve", "--hostname", "127.0.0.1", "--port", String(port)], { stdin: "ignore", stdout: "ignore", stderr: "ignore",
        env: { ...env, CHRYSALIS_WEB_PID: String(process.pid), OPENCODE_PASSWORD: password, OPENCODE_CONFIG_CONTENT: JSON.stringify({ providers: { chrysalis: { name: "Chrysalis",
          package: "@opencode/ai/providers/openai-compatible", settings: { baseURL: `${chrysalis}/v1`, apiKey: "chrysalis" }, models } } }) } });
      url = `http://127.0.0.1:${port}`;
      void proc.exited.then(() => { if (host?.proc === proc) { host = undefined; starting = undefined; lost(); } });
    }
    const auth = `Basic ${Buffer.from(`opencode:${password}`, "utf8").toString("base64")}`;
    for (const deadline = Date.now() + 30_000; ; await Bun.sleep(150)) {
      const info = await fetch(`${url}/api/info`, { headers: { authorization: auth }, signal: AbortSignal.timeout(2000) }).then((r) => (r.ok ? r.json() : r.status === 401 ? 401 : null), () => null);
      if (info === 401) { proc?.kill(); throw new Error("OpenCode refused the password"); }
      if (info?.version) {
        const listed = !proc || !local || await fetch(`${url}/api/model`, { headers: { authorization: auth } }).then((r) => r.json())
          .then((j) => (j.data ?? []).some((m: any) => m.providerID === "chrysalis"), () => false);
        if (listed) { host = { url, auth, version: info.version, proc }; break; }
      }
      if (Date.now() > deadline || (proc && proc.exitCode !== null)) { proc?.kill(); throw new Error("OpenCode did not start"); }
    }
    void watch(host);
  })().catch((e) => { starting = undefined; throw e; }));
}

function lost() {
  running.clear();
  for (const set of [...listeners.values()]) for (const fn of [...set]) fn("lost", {});
}
async function watch(h: NonNullable<typeof host>) {
  while (host === h) {
    try {
      const r = await fetch(`${h.url}/api/event`, { headers: { authorization: h.auth, accept: "text/event-stream" } });
      if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);
      const reader = r.body.getReader(), decoder = new TextDecoder();
      for (let buf = ""; ; ) {
        const { value, done } = await reader.read();
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
          if (ev.type === "session.execution.started") running.add(id);
          if (/^session\.execution\.(succeeded|failed|interrupted)$/.test(ev.type)) running.delete(id);
          for (const fn of [...(listeners.get(id) ?? [])]) fn(ev.type, ev.data);
        }
      }
    } catch {  }
    if (host !== h) return;
    lost();
    await Bun.sleep(1000);
  }
}
function listen(id: string, fn: Listener) {
  const set = listeners.get(id) ?? new Set<Listener>();
  listeners.set(id, set.add(fn));
  return () => { set.delete(fn); if (!set.size) listeners.delete(id); };
}

const refOf = (model: string) => ({ providerID: model.slice(0, model.indexOf("/")), id: model.slice(model.indexOf("/") + 1) });
const idOf = (m: any) => (m?.providerID && m?.id ? `${m.providerID}/${m.id}` : undefined);
const textOf = (content: any) => (Array.isArray(content) ? content.map((p: any) => p?.text ?? "").join("") : typeof content === "string" ? content : "");
const titled = (name: string) => name.replace(/[_-]+/g, " ").replace(/^./, (c) => c.toUpperCase());

export async function threads(): Promise<Thread[]> {
  const out: Thread[] = [];
  for (let cursor: string | undefined, pages = 0; pages < 10; pages++) {
    const page = await call("GET", `/api/session?limit=100${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}`);
    for (const s of page.data ?? []) if (!s.parentID && s.location?.directory)
      out.push({ id: s.id, title: s.title ?? "", dir: s.location.directory, updated: s.time?.updated ?? s.time?.created ?? 0, busy: running.has(s.id), model: idOf(s.model) });
    cursor = page.cursor?.next;
    if (!cursor || !page.data?.length) break;
  }
  return out;
}
export async function models(): Promise<Model[]> {
  const [list, providers] = await Promise.all([call("GET", "/api/model"), call("GET", "/api/provider")]);
  const names = new Map<string, string>((providers.data ?? []).map((p: any) => [p.id, p.name ?? p.id]));
  return (list.data ?? []).flatMap((m: any) => (idOf(m) ? [{ id: idOf(m)!, name: m.name ?? m.id, provider: names.get(m.providerID) ?? m.providerID }] : []));
}
export const remove = (id: string) => call("DELETE", `/api/session/${id}`);
export const rename = (id: string, title: string) => call("PATCH", `/api/session/${id}`, { title });

function callOf(p: any): Call {
  const s = p.state ?? {}, failed = s.status === "error", done = s.status === "completed";
  return { id: p.id, name: p.name, args: JSON.stringify(s.input ?? {}), title: titled(p.name), source: SOURCE, state: done ? "done" : failed ? "failed" : "stopped",
    ...(done || failed ? { result: typeof s.output === "string" ? s.output : textOf(s.content) || s.error?.message || (typeof s.error === "string" ? s.error : "") } : {}) };
}
export async function transcript(id: string): Promise<{ thread: Thread; messages: Said[] }> {
  const s = (await call("GET", `/api/session/${id}`)).data, all: any[] = [];
  for (let cursor: string | undefined; ; ) {
    const page = await call("GET", `/api/session/${id}/message?limit=200&${cursor ? `cursor=${encodeURIComponent(cursor)}` : "order=asc"}`);
    all.push(...(page.data ?? []));
    cursor = page.cursor?.next;
    if (!cursor || !page.data?.length) break;
  }
  const messages: Said[] = [];
  let steps: Step[] = [];
  const close = (more: Partial<Said> = {}) => {
    if (!steps.length && !more.error) return;
    const last = steps.pop() ?? { content: "", calls: [] };
    messages.push({ role: "assistant", content: last.content, ...(last.reasoning ? { reasoning: last.reasoning } : {}), ...(last.calls.length ? { calls: last.calls } : {}),
      ...(steps.length ? { rounds: steps } : {}), ...more });
    steps = [];
  };
  for (const m of all) {
    if (m.type === "user") { close(); messages.push({ role: "user", content: m.text ?? "" }); }
    else if (m.type === "assistant") {
      const parts: any[] = m.content ?? [], reasoning = parts.filter((p) => p.type === "reasoning").map((p) => p.text ?? "").join("");
      steps.push({ content: parts.filter((p) => p.type === "text").map((p) => p.text ?? "").join(""), ...(reasoning ? { reasoning } : {}), calls: parts.filter((p) => p.type === "tool").map(callOf) });
    } else if (m.type === "idle") close(m.outcome === "failed" ? { error: m.error?.message ?? "OpenCode could not finish this turn" } : m.outcome === "interrupted" ? { stopped: true } : {});
  }
  close();
  return { thread: { id: s.id, title: s.title ?? "", dir: s.location?.directory ?? "", updated: s.time?.updated ?? 0, busy: running.has(s.id), model: idOf(s.model) }, messages };
}

export function folders(path: string) {
  const at = resolve(path || homedir());
  const dirs = readdirSync(at, { withFileTypes: true }).filter((e) => e.isDirectory() && !e.name.startsWith(".")).map((e) => e.name).sort((a, b) => a.localeCompare(b));
  return { path: at, up: dirname(at) === at ? undefined : dirname(at), home: homedir(), dirs: dirs.map((name) => ({ name, path: join(at, name) })) };
}

export async function turn(t: Turn, push: (d: object) => void, signal: AbortSignal, ask: (id: string) => Promise<Decision | "stop">, began: (session: string) => void) {
  if (!existsSync(t.dir)) throw new Error(`There is no folder ${t.dir} on this computer`);
  const id = t.session ?? (await call("POST", "/api/session", { location: { directory: t.dir } })).data.id as string;
  began(id);
  if (t.model) await call("POST", `/api/session/${id}/model`, { model: refOf(t.model) });
  const entry = `/api/experimental/session/${id}/instructions/entries/${PERSONA}`;
  if (t.persona) await call("PUT", entry, { value: t.persona });
  else if (t.session) await call("DELETE", entry).catch(() => {});
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
  let stop = () => {};
  const done = new Promise<void>((resolve, reject) => {
    stop = listen(id, (type, d) => {
      if (type === "session.step.started") { if (steps++) { push({ round: steps - 1 }); index.clear(); } }
      else if (type === "session.reasoning.delta") delta({ reasoning_content: d.delta });
      else if (type === "session.text.delta") delta({ content: d.delta });
      else if (type === "session.tool.input.started") known(d.id, d.name);
      else if (type === "session.tool.input.ended") delta({ tool_calls: [{ index: index.get(d.id), function: { arguments: d.text ?? "" } }] });
      else if (type === "session.tool.called") { known(d.id, undefined, JSON.stringify(d.input ?? {})); since.set(d.id, Date.now()); tool(d.id, "running"); }
      else if (type === "session.tool.success") tool(d.id, "done", { result: textOf(d.content), ms: ms(d.id) });
      else if (type === "session.tool.failed") { if (!declined.has(d.id)) tool(d.id, "failed", { result: `Error: ${d.error?.message ?? (textOf(d.content) || "the tool failed")}`, ms: ms(d.id) }); }
      else if (type === "permission.asked") {
        const callId = d.source?.type === "tool" ? d.source.id as string : d.id as string;
        known(callId, d.action, JSON.stringify({ [d.action]: d.resources }));
        tool(callId, "approval");
        void ask(callId).then((say) => {
          if (say === "deny" || say === "stop") { declined.add(callId); tool(callId, say === "deny" ? "declined" : "stopped", { result: "Not run: it was not allowed." }); }
          return call("POST", `/api/session/${id}/permission/${d.id}/reply`, { decision: say === "once" || say === "always" ? say : "reject" });
        }).catch(reject);
      }
      else if (type === "form.created") void call("DELETE", `/api/session/${id}/form/${d.form.id}?message=${encodeURIComponent(NO_FORMS)}`).catch(() => {});
      else if (type === "session.step.ended" && d.tokens) push({ usage: { prompt_tokens: (d.tokens.input ?? 0) + (d.tokens.cache?.read ?? 0), completion_tokens: d.tokens.output ?? 0,
        prompt_tokens_details: { cached_tokens: d.tokens.cache?.read ?? 0 } } });
      else if (type === "session.retry.scheduled") push({ retry: { message: d.error?.message ?? "The model did not answer" } });
      else if (type === "session.compaction.started") push({ waiting: "Compacting the context" });
      else if (type === "session.execution.succeeded") resolve();
      else if (type === "session.execution.interrupted") (signal.aborted ? resolve() : reject(new Error(`OpenCode stopped this turn (${d.reason ?? "interrupted"})`)));
      else if (type === "session.execution.failed") reject(new Error(d.error?.message ?? "OpenCode could not finish this turn"));
      else if (type === "lost") reject(new Error("The connection to OpenCode was lost: open the thread again to see where it is"));
    });
  });
  const interrupt = () => void call("POST", `/api/session/${id}/interrupt`).catch(() => {});
  signal.addEventListener("abort", interrupt, { once: true });
  try {
    await call("POST", `/api/session/${id}/prompt`, { text: t.text, ...(t.files?.length ? { files: t.files.map((f) => ({ ...f, source: { type: "inline" } })) } : {}) });
    if (signal.aborted) interrupt();
    await done;
  } finally {
    stop();
    signal.removeEventListener("abort", interrupt);
  }
  if (signal.aborted) throw new DOMException("stopped", "AbortError");
}
