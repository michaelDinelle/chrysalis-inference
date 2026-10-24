import { lookup } from "node:dns/promises";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { isIP } from "node:net";
import { isPublic } from "./tool-fetch";

export interface Backend { url: string; key?: string; chrysalis?: boolean }
export interface Model { id: string; owned_by?: string }
type Listed = { ok: true; models: Model[]; owner?: string } | { ok: false; error: string };

const PROBE_MS = 5000, LIST_MS = 3000;
const NAMES: Record<string, string> = { chrysalis: "Chrysalis", llamacpp: "llama.cpp", koboldcpp: "KoboldCpp", vllm: "vLLM",
  organization_owner: "LM Studio", library: "Ollama" };

const STANDARD = new Set(["model", "messages", "tools", "tool_choice", "parallel_tool_calls", "stream", "stream_options", "max_tokens",
  "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "dry_multiplier", "dry_base", "dry_allowed_length",
  "seed", "stop", "reasoning_effort", "logprobs", "top_logprobs", "response_format"]);
export const standard = (req: Record<string, unknown>) => Object.fromEntries(Object.entries(req).filter(([k]) => STANDARD.has(k)));

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

export async function remote(u: URL): Promise<boolean> {
  const host = u.hostname.replace(/^\[|\]$/g, "");
  if (host === "localhost" || host.endsWith(".localhost")) return false;
  if (isIP(host)) return isPublic(host);
  try { return (await lookup(host, { all: true })).some((a) => isPublic(a.address)); } catch { return false; }
}

const why = (e: unknown, origin: string) => {
  const code = String((e as { code?: string })?.code ?? "");
  if ((e as Error)?.name === "TimeoutError") return `${origin} did not answer in ${PROBE_MS / 1000} s`;
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
  async about() {
    const l = this.atService ? undefined : await this.models();
    this.far ??= remote(new URL(this.now.url));
    return { url: this.now.url, chrysalis: this.isChrysalis, service: this.atService, key: !!this.now.key, remote: await this.far,
      name: this.atService ? "Chrysalis" : (l?.ok && l.owner && NAMES[l.owner]) || new URL(this.now.url).host, ...(l ? { listed: l } : {}) };
  }
  async set(text: string, key: string | undefined, confirmRemote: boolean):
    Promise<{ ok: true } | { ok: false; error: string; remote?: string }> {
    let p = parseAddress(text);
    if (typeof p === "string") return { ok: false, error: p };
    const far = await remote(p.url);
    if (far && p.url.protocol === "http:") {
      if (/^http:\/\//i.test(text.trim())) return { ok: false, error: `${p.url.host} is off this computer and its networks: it takes https://` };
      p = parseAddress(`https://${text.trim()}`) as Exclude<typeof p, string>;
    }
    key = key === undefined ? (p.base === this.now.url ? this.now.key : undefined) : key.trim() || undefined;
    const tries = p.base === p.url.origin ? [`${p.base}/v1`, p.base] : [p.base, ...(/\/v1$/.test(p.base) ? [] : [`${p.base}/v1`])];
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
