import { lookup } from "node:dns/promises";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { isIP } from "node:net";

function toBig(ip: string): bigint | null {
  ip = ip.replace(/^\[|\]$/g, "").replace(/%.*$/, "");
  if (isIP(ip) === 4) return 0xffff00000000n | BigInt(ip.split(".").reduce((a, o) => a * 256 + Number(o), 0));
  if (isIP(ip) !== 6) return null;
  const dotted = ip.match(/(\d+\.\d+\.\d+\.\d+)$/);
  if (dotted) {
    const v = dotted[1].split(".").reduce((a, o) => a * 256 + Number(o), 0);
    ip = `${ip.slice(0, -dotted[1].length)}${(v >>> 16).toString(16)}:${(v & 0xffff).toString(16)}`;
  }
  const [head, tail] = ip.split("::"), h = head ? head.split(":") : [], t = tail === undefined ? null : tail ? tail.split(":") : [];
  const groups = t === null ? h : [...h, ...Array(8 - h.length - t.length).fill("0"), ...t];
  return groups.reduce((a, g) => (a << 16n) | BigInt(parseInt(g, 16)), 0n);
}
const block = (cidr: string): [bigint, number] | null => {
  const [ip, len] = cidr.split("/"), x = toBig(ip);
  if (x === null) return null;
  const v4 = isIP(ip.replace(/^\[|\]$/g, "")) === 4, n = len === undefined ? 128 : Number(len) + (v4 ? 96 : 0);
  return Number.isInteger(n) && n >= 0 && n <= 128 ? [x, n] : null;
};
const inBlock = (x: bigint, [base, len]: [bigint, number]) => len === 0 || x >> BigInt(128 - len) === base >> BigInt(128 - len);
const NON_PUBLIC = ["0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24",
  "192.0.2.0/24", "192.88.99.0/24", "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24", "224.0.0.0/4", "240.0.0.0/4",
  "::/96", "100::/64", "2001::/23", "2001:db8::/32", "fc00::/7", "fe80::/10", "fec0::/10", "ff00::/8"].map((c) => block(c)!);
const NAT64 = block("64:ff9b::/96")!, SIX_TO_FOUR = block("2002::/16")!;
const v4 = (n: bigint) => [24n, 16n, 8n, 0n].map((s) => String((n >> s) & 255n)).join(".");
function isPublic(ip: string): boolean {
  const x = toBig(ip);
  if (x === null) return false;
  if (inBlock(x, NAT64)) return isPublic(v4(x & 0xffffffffn));
  if (inBlock(x, SIX_TO_FOUR)) return isPublic(v4((x >> 80n) & 0xffffffffn));
  return !NON_PUBLIC.some((b) => inBlock(x, b));
}

export interface Backend { url: string; chrysalis?: boolean }
export interface Model { id: string; owned_by?: string; context?: number }
type Listed = { ok: true; models: Model[]; owner?: string } | { ok: false; error: string };

const PROBE_MS = 5000, LIST_MS = 3000;
const NAMES: Record<string, string> = { chrysalis: "Chrysalis", llamacpp: "llama.cpp", koboldcpp: "KoboldCpp", vllm: "vLLM",
  organization_owner: "LM Studio", library: "Ollama" };

const STANDARD = new Set(["model", "messages", "tools", "tool_choice", "parallel_tool_calls", "stream", "stream_options", "max_tokens",
  "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "dry_multiplier", "dry_base", "dry_allowed_length",
  "dynatemp_range", "dynatemp_exponent", "typical_p", "xtc_probability", "xtc_threshold", "top_n_sigma", "min_keep", "repeat_last_n", "repeat_penalty",
  "dry_penalty_last_n", "mirostat", "mirostat_tau", "mirostat_eta", "samplers",
  "seed", "stop", "reasoning_effort", "logprobs", "top_logprobs", "response_format"]);
export const standard = (req: Record<string, unknown>) => Object.fromEntries(Object.entries(req).filter(([k]) => STANDARD.has(k)));

export function parseAddress(text: string): { url: URL; base: string } | string {
  let s = text.trim();
  if (!s) return "an address, like http://localhost:11434/v1";
  if (!/^[a-z][a-z\d+.-]*:\/\//i.test(s)) s = `http://${s}`;
  let u: URL;
  try { u = new URL(s); } catch { return `${text.trim()} is not an address`; }
  if (u.protocol !== "http:" && u.protocol !== "https:") return "only http:// and https:// addresses";
  if (u.username || u.password) return "no user name or password in the address";
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

async function list(url: string, ms: number): Promise<Listed & { status?: number }> {
  const origin = new URL(url).origin;
  let r: Response;
  try {
    r = await fetch(`${url}/models`, { signal: AbortSignal.timeout(ms) });
  } catch (e) {
    return { ok: false, error: why(e, origin) };
  }
  if (r.status === 401 || r.status === 403) return { ok: false, status: r.status, error: `${origin} wants an API key, which Chrysalis does not send` };
  const body = await r.json().catch(() => null);
  if (!r.ok || !Array.isArray(body?.data))
    return { ok: false, status: r.status, error: `${url}/models answered ${r.status}${r.ok ? " without a model list" : ""}: not an OpenAI-style API there` };
  const models: Model[] = body.data.filter((m: any) => typeof m?.id === "string")
    .map((m: any) => ({ id: m.id, owned_by: m.owned_by, context: m.loaded_context_length ?? m.max_model_len ?? m.context_length ?? m.max_context_length }));
  return { ok: true, models, owner: models.find((m) => m.owned_by)?.owned_by };
}

const saysChrysalis = (models: Model[]) => models.some((m) => m.owned_by === "chrysalis");

const asked = (url: string) => fetch(url, { signal: AbortSignal.timeout(LIST_MS) }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
const WINDOWS: Record<string, (origin: string, model: string) => Promise<number | undefined>> = {
  llamacpp: async (o) => (await asked(`${o}/props`))?.default_generation_settings?.n_ctx,
  koboldcpp: async (o) => (await asked(`${o}/api/extra/true_max_context_length`))?.value,
  organization_owner: async (o, model) => {
    const m = (await asked(`${o}/api/v0/models`))?.data?.find((x: { id?: string }) => x.id === model);
    return m?.loaded_context_length ?? m?.max_context_length;
  },
  library: async (o, model) => (await asked(`${o}/api/ps`))?.models?.find((x: { name?: string; model?: string }) => x.name === model || x.model === model)?.context_length,
};

export class Backends {
  private now: Backend;
  private cache?: { at: number; listed: Promise<Listed> };
  private far?: Promise<boolean>;
  constructor(private file: string, readonly service: string) {
    const saved = existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : null;
    this.now = typeof saved?.url === "string" ? { url: saved.url, ...(saved.chrysalis ? { chrysalis: true } : {}) } : { url: service };
  }
  get url() { return this.now.url; }
  get atService() { return this.now.url === this.service; }
  get isChrysalis() { return this.atService || !!this.now.chrysalis; }
  headers(): Record<string, string> {
    return { "Content-Type": "application/json" };
  }
  models(): Promise<Listed> {
    if (!this.cache || Date.now() - this.cache.at > LIST_MS) {
      const now = this.now;
      this.cache = { at: Date.now(), listed: list(now.url, PROBE_MS) };
      void this.cache.listed.then((l) => {
        if (l.ok && this.now === now && saysChrysalis(l.models) !== !!now.chrysalis) this.keep({ ...now, chrysalis: saysChrysalis(l.models) }, false);
      });
    }
    return this.cache.listed.then((l) => (l.ok ? { ...l, models: l.models.filter((m) => !/embed/i.test(m.id)) } : l));
  }
  async windowOf(model: string): Promise<number | undefined> {
    const l = await this.models();
    if (!l.ok) return undefined;
    return l.models.find((m) => m.id === model)?.context ?? WINDOWS[l.owner ?? ""]?.(new URL(this.now.url).origin, model);
  }
  async about() {
    const l = this.atService ? undefined : await this.models();
    this.far ??= remote(new URL(this.now.url));
    return { url: this.now.url, chrysalis: this.isChrysalis, service: this.atService, remote: await this.far,
      name: this.atService ? "Chrysalis" : (l?.ok && l.owner && NAMES[l.owner]) || new URL(this.now.url).host, ...(l ? { listed: l } : {}) };
  }
  async set(text: string, confirmRemote: boolean):
    Promise<{ ok: true } | { ok: false; error: string; remote?: string }> {
    let p = parseAddress(text);
    if (typeof p === "string") return { ok: false, error: p };
    const far = await remote(p.url);
    if (far && p.url.protocol === "http:") {
      if (/^http:\/\//i.test(text.trim())) return { ok: false, error: `${p.url.host} is off this computer and its networks: it takes https://` };
      p = parseAddress(`https://${text.trim()}`) as Exclude<typeof p, string>;
    }
    const tries = p.base === p.url.origin ? [`${p.base}/v1`, p.base] : [p.base, ...(/\/v1$/.test(p.base) ? [] : [`${p.base}/v1`])];
    if (tries.some((t) => t.replace("//localhost:", "//127.0.0.1:") === this.service)) return this.keep({ url: this.service });
    let url = tries[0], l = await list(url, PROBE_MS);
    for (const t of tries.slice(1)) {
      if (l.ok || !l.status || l.status === 401 || l.status === 403) break;
      const next = await list(t, PROBE_MS);
      if (next.ok || next.status === 401 || next.status === 403) [url, l] = [t, next];
    }
    if (!l.ok) return { ok: false, error: l.error };
    if (!confirmRemote && far) return { ok: false, error: "off this computer and its networks", remote: p.url.host };
    return this.keep({ url, ...(saysChrysalis(l.models) ? { chrysalis: true } : {}) });
  }
  private keep(b: Backend, fresh = true): { ok: true } {
    this.now = b;
    if (fresh) this.cache = this.far = undefined;
    writeFileSync(this.file, JSON.stringify(b), { mode: 0o600 });
    return { ok: true };
  }
}
