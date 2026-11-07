import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { agentSource } from "./tool-agent";
import { fetchSource } from "./tool-fetch";
import { mcpSource } from "./tool-mcp";

export interface Tool {
  name: string;
  title: string;
  description: string;
  parameters: Record<string, unknown>;
  readOnly: boolean;
  run(args: Record<string, unknown>, ctx: { signal: AbortSignal; maxChars: number }): Promise<string>;
}
export type State = "idle" | "connecting" | "connected" | "failed" | "disabled";
export interface Source {
  id: string;
  label: string; kind: "fetch" | "agent" | "mcp";
  prefix: string;
  detail: string;
  spec: string;
  state: State; error?: string; tools: Tool[];
  timeoutMs: number; maxChars: number;
  chars?: number;
  connect(): Promise<void>;
  close(): Promise<void>;
  onChange?: () => void;
}
export interface Named { fn: string; tool: Tool; source: Source }

export const CONFIG = process.env.CHRYSALIS_MCP_CONFIG ?? `${homedir()}/.config/chrysalis/mcp.json`;
export const DEFAULT_TIMEOUT_S = 60, DEFAULT_MAX_CHARS = 40_000;

let sources: Source[] = [];
let configError: string | undefined;
let names = new Map<string, Named>();

export const clean = (s: string) => s.replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "tool";
function rename() {
  const next = new Map<string, Named>();
  for (const source of [...sources].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)))
    for (const tool of [...source.tools].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))) {
      const base = (source.prefix ? `${clean(source.prefix).slice(0, 30)}__${clean(tool.name)}` : clean(tool.name)).slice(0, 64);
      let fn = base;
      for (let k = 2; next.has(fn); k++) fn = `${base.slice(0, 64 - String(k).length - 1)}_${k}`;
      next.set(fn, { fn, tool, source });
    }
  names = next;
}

export async function reload() {
  let config: any = {};
  configError = undefined;
  try {
    config = JSON.parse(readFileSync(CONFIG, "utf8"));
    if (!config || typeof config !== "object" || Array.isArray(config)) throw new Error("not a JSON object");
  } catch (e: any) {
    if (e?.code !== "ENOENT") configError = `${CONFIG}: ${e?.message ?? e}`;
    config = {};
  }
  const servers = config.mcpServers && typeof config.mcpServers === "object" ? config.mcpServers : {};
  const wanted = [fetchSource(config.fetch ?? {}), agentSource(),
    ...Object.entries(servers).map(([name, spec]) => mcpSource(name, spec && typeof spec === "object" ? spec as Record<string, unknown> : {}))];
  const kept: Source[] = [];
  for (const w of wanted) {
    const old = sources.find((s) => s.id === w.id);
    if (old && old.spec === w.spec) kept.push(old);
    else { if (old) await old.close(); kept.push(w); }
  }
  for (const s of sources) if (!wanted.some((w) => w.id === s.id)) await s.close();
  sources = kept;
  for (const s of sources) s.onChange = rename;
  rename();
}

export async function closeAll() {
  await Promise.all(sources.map((s) => s.close()));
}

export function catalog() {
  const fnOf = new Map([...names.values()].map((n) => [n.tool, n.fn]));
  return {
    config: CONFIG, error: configError,
    sources: sources.map((s) => ({
      id: s.id, label: s.label, kind: s.kind, detail: s.detail, state: s.state, error: s.error,
      tools: [...names.values()].filter((n) => n.source === s)
        .map(({ tool: t }) => ({ fn: fnOf.get(t), name: t.name, title: t.title, description: t.description, readOnly: t.readOnly })),
    })),
  };
}

export const source = (id: string) => sources.find((s) => s.id === id);

export async function select(choice: Record<string, { off?: string[] }>): Promise<{ tools: Named[]; notes: string[] }> {
  const on = sources.filter((s) => choice[s.id] && s.state !== "disabled");
  await Promise.all(on.map((s) => s.connect()));
  const notes = on.filter((s) => s.state !== "connected").map((s) => `${s.label} is not available (${s.error ?? s.state}), so its tools are left out`);
  const tools = [...names.values()].filter((n) => n.source.state === "connected" && choice[n.source.id] && !choice[n.source.id].off?.includes(n.tool.name))
    .sort((a, b) => (a.fn < b.fn ? -1 : a.fn > b.fn ? 1 : 0));
  return { tools, notes };
}

export const openai = (n: Named, description?: string) =>
  ({ type: "function", function: { name: n.fn, description: description ?? n.tool.description, parameters: n.tool.parameters } });

export const cut = (out: string, limit: number) => out.length <= limit ? out
  : `${out.slice(0, limit)}\n\n[The result was cut here: ${limit.toLocaleString("en-US")} of ${out.length.toLocaleString("en-US")} characters.]`;

export async function run(n: Named, args: Record<string, unknown>, signal: AbortSignal, maxChars?: number, room = Infinity): Promise<string> {
  const limit = Math.min(maxChars ?? n.source.chars ?? n.source.maxChars, n.source.maxChars, room);
  const timeout = AbortSignal.timeout(n.source.timeoutMs);
  const both = AbortSignal.any([signal, timeout]);
  const out = await new Promise<string>((resolve, reject) => {
    const gone = () => reject(timeout.aborted && !signal.aborted ? new Error(`the tool did not answer within ${Math.round(n.source.timeoutMs / 1000)} s`)
      : new DOMException("stopped", "AbortError"));
    if (both.aborted) return gone();
    both.addEventListener("abort", gone, { once: true });
    n.tool.run(args, { signal: both, maxChars: limit }).then(resolve, reject);
  });
  return cut(out, limit);
}
