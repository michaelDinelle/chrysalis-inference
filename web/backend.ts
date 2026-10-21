// Where replies come from: any server with OpenAI's chat API, by its address (Chrysalis, Ollama, llama.cpp, LM Studio,
// vLLM, KoboldCpp, a hosted API). This server sends it every request, so no backend needs a browser let in (CORS) and
// tools, voice and replies that outlive their tab work with any of them. Chrysalis is one of them: it is told apart by
// what it says of itself (its model list's owned_by), which gets it its own fields; at this machine's Chrysalis
// service's address the page also loads its models and starts it. Only this computer's own screen changes the backend
// (server.ts): a device that could point this server at any address would read what only this machine can reach. Its
// key stays in backend.json (readable by this user only) and goes to the backend alone, never to a page.
import { lookup } from "node:dns/promises";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { isIP } from "node:net";
import { isPublic } from "./tool-fetch";

// chrysalis: its model list said it is Chrysalis, when it last answered
export interface Backend { url: string; key?: string; chrysalis?: boolean }
export interface Model { id: string; owned_by?: string }
type Listed = { ok: true; models: Model[]; owner?: string } | { ok: false; error: string };

const PROBE_MS = 5000, LIST_MS = 3000;
// The owned_by each backend gives its models, for its name on screen (any other is named by its address)
const NAMES: Record<string, string> = { chrysalis: "Chrysalis", llamacpp: "llama.cpp", koboldcpp: "KoboldCpp", vllm: "vLLM",
  organization_owner: "LM Studio", library: "Ollama" };

// What a backend other than Chrysalis is sent of a request: OpenAI's own fields, and the samplers llama.cpp and
// KoboldCpp take by the same names (top_k, min_p, DRY), which the page sends only when the user sets them. Chrysalis's
// own fields stay behind: a continued reply's tokens, and its sampler order, which llama.cpp would take as its whole
// chain (its DRY and XTC gone).
const STANDARD = new Set(["model", "messages", "tools", "tool_choice", "parallel_tool_calls", "stream", "stream_options", "max_tokens",
  "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "dry_multiplier", "dry_base", "dry_allowed_length",
  "seed", "stop", "reasoning_effort", "logprobs", "top_logprobs", "response_format"]);
export const standard = (req: Record<string, unknown>) => Object.fromEntries(Object.entries(req).filter(([k]) => STANDARD.has(k)));

// An address as people paste it, with or without http://, /v1 or the endpoint's own path, as its API's base (no slash at
// the end: a URL's path is never empty, so http://localhost:11434 has "/")
export function parseAddress(text: string): { url: URL; base: string } | string {
  let s = text.trim();
  if (!s) return "an address, like http://localhost:11434/v1";
  if (!/^[a-z][a-z\d+.-]*:\/\//i.test(s)) s = `http://${s}`;
  let u: URL;
  try { u = new URL(s); } catch { return `${text.trim()} is not an address`; }
  if (u.protocol !== "http:" && u.protocol !== "https:") return "only http:// and https:// addresses";
  if (u.username || u.password) return "no user name or password in the address: a key goes in the key field";
  if (u.search || u.hash) return "no ? or # in the address";
  return { url: u, base: u.origin + u.pathname.replace(/\/+$/, "").replace(/\/(chat\/completions|models)$/, "") };
}

// Off this machine and its networks: a reply's whole conversation leaves them, which the page says before it is set.
// A name counts as off them if any address it has is public.
export async function remote(u: URL): Promise<boolean> {
  const host = u.hostname.replace(/^\[|\]$/g, "");
  if (host === "localhost" || host.endsWith(".localhost")) return false;
  if (isIP(host)) return isPublic(host);
  try { return (await lookup(host, { all: true })).some((a) => isPublic(a.address)); } catch { return false; }
}

const why = (e: unknown, origin: string) => {
  const code = String((e as { code?: string })?.code ?? "");
  if ((e as Error)?.name === "TimeoutError") return `${origin} did not answer in ${PROBE_MS / 1000} s`;
  // Bun says UNABLE_TO_VERIFY_LEAF_SIGNATURE, "unable to verify the first certificate", for one signed by an unknown authority
  if (/CERT|SSL|TLS|VERIFY/i.test(code) || /certificate/i.test(String((e as Error)?.message))) return `${origin}'s certificate is not trusted (${code})`;
  return `nothing answers at ${origin}: is the backend running?`;
};

async function list(url: string, key: string | undefined, ms: number): Promise<Listed & { status?: number }> {
  const origin = new URL(url).origin;
  let r: Response;
  try {
    r = await fetch(`${url}/models`, { headers: key ? { Authorization: `Bearer ${key}` } : {}, signal: AbortSignal.timeout(ms) });
  } catch (e) {
    return { ok: false, error: why(e, origin) };
  }
  if (r.status === 401 || r.status === 403) return { ok: false, status: r.status, error: `${origin} ${key ? "refused the key" : "wants a key"}` };
  const body = await r.json().catch(() => null);
  if (!r.ok || !Array.isArray(body?.data))
    return { ok: false, status: r.status, error: `${url}/models answered ${r.status}${r.ok ? " without a model list" : ""}: not an OpenAI-style API there` };
  const models: Model[] = body.data.filter((m: any) => typeof m?.id === "string").map((m: any) => ({ id: m.id, owned_by: m.owned_by }));
  return { ok: true, models, owner: models.find((m) => m.owned_by)?.owned_by };
}

const saysChrysalis = (models: Model[]) => models.some((m) => m.owned_by === "chrysalis");

export class Backends {
  private now: Backend;
  private cache?: { at: number; listed: Promise<Listed> };
  private far?: Promise<boolean>;
  // service: this machine's Chrysalis service's API, the backend until another is set. It answers nothing while it is
  // stopped or loading a model (its model list waits for the load), so there it is Chrysalis by its address.
  constructor(private file: string, readonly service: string) {
    const saved = existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : null;
    this.now = typeof saved?.url === "string" ? saved : { url: service };
  }
  get url() { return this.now.url; }
  get atService() { return this.now.url === this.service; }
  get isChrysalis() { return this.atService || !!this.now.chrysalis; }
  headers(): Record<string, string> {
    return { "Content-Type": "application/json", ...(this.now.key ? { Authorization: `Bearer ${this.now.key}` } : {}) };
  }
  // its models (embedding models left out: they do not chat), asked at most every LIST_MS however many pages poll
  models(): Promise<Listed> {
    if (!this.cache || Date.now() - this.cache.at > LIST_MS) {
      const now = this.now;
      this.cache = { at: Date.now(), listed: list(now.url, now.key, PROBE_MS) };
      void this.cache.listed.then((l) => {
        if (l.ok && this.now === now && saysChrysalis(l.models) !== !!now.chrysalis) this.keep({ ...now, chrysalis: saysChrysalis(l.models) }, false);
      });
    }
    return this.cache.listed.then((l) => (l.ok ? { ...l, models: l.models.filter((m) => !/embed/i.test(m.id)) } : l));
  }
  // What the page shows of it; never its key, only whether it has one
  // service: at this machine's Chrysalis service, whose state server.ts reads from the service itself
  async about() {
    const l = this.atService ? undefined : await this.models();
    this.far ??= remote(new URL(this.now.url));
    return { url: this.now.url, chrysalis: this.isChrysalis, service: this.atService, key: !!this.now.key, remote: await this.far,
      name: this.atService ? "Chrysalis" : (l?.ok && l.owner && NAMES[l.owner]) || new URL(this.now.url).host, ...(l ? { listed: l } : {}) };
  }
  // A new backend: checked (its model list, with its key) before it is kept. key undefined keeps the key it has when the
  // address stays the same. Off this machine's networks it is kept only once the user has seen
  // where their chats would go (confirmRemote).
  async set(text: string, key: string | undefined, confirmRemote: boolean):
    Promise<{ ok: true } | { ok: false; error: string; remote?: string }> {
    let p = parseAddress(text);
    if (typeof p === "string") return { ok: false, error: p };
    // over http:// anyone on the way reads every chat and the key: off this machine's networks, https:// only
    const far = await remote(p.url);
    if (far && p.url.protocol === "http:") {
      if (/^http:\/\//i.test(text.trim())) return { ok: false, error: `${p.url.host} is off this computer and its networks: it takes https://` };
      p = parseAddress(`https://${text.trim()}`) as Exclude<typeof p, string>;
    }
    key = key === undefined ? (p.base === this.now.url ? this.now.key : undefined) : key.trim() || undefined;
    // Most serve their API under /v1, which a bare address tries first (llama.cpp answers at /models too); an address
    // with a path is tried as it is. Where nothing answers at all, nothing will at the other path either.
    const tries = p.base === p.url.origin ? [`${p.base}/v1`, p.base] : [p.base, ...(/\/v1$/.test(p.base) ? [] : [`${p.base}/v1`])];
    // this machine's Chrysalis service as it is, stopped or loading too (the page shows which, and loads it)
    if (tries.some((t) => t.replace("//localhost:", "//127.0.0.1:") === this.service)) return this.keep({ url: this.service });
    let url = tries[0], l = await list(url, key, PROBE_MS);
    for (const t of tries.slice(1)) {
      if (l.ok || !l.status || l.status === 401 || l.status === 403) break;
      const next = await list(t, key, PROBE_MS);
      if (next.ok || next.status === 401 || next.status === 403) [url, l] = [t, next];
    }
    if (!l.ok) return { ok: false, error: l.error };
    if (!confirmRemote && far) return { ok: false, error: "off this computer and its networks", remote: p.url.host };
    return this.keep({ url, ...(key ? { key } : {}), ...(saysChrysalis(l.models) ? { chrysalis: true } : {}) });
  }
  private keep(b: Backend, fresh = true): { ok: true } {
    this.now = b;
    if (fresh) this.cache = this.far = undefined;
    writeFileSync(this.file, JSON.stringify(b), { mode: 0o600 });
    return { ok: true };
  }
}
