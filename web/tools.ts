// The tools a reply can call, whatever provides them: the built-in Fetch URL (tool-fetch.ts) and Sub-agents
// (tool-agent.ts), and the MCP servers in the config file (tool-mcp.ts). Each provider is a Source of Tools behind the interfaces below; server.ts only talks to
// this file. The config file is the only place a server's command, URL or environment comes from: the page (no logins,
// and on the LAN after `chrysalis.sh start --host 0.0.0.0`) can list sources, start, retry and reload them, never define one, or it
// could run any command on this machine.
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { agentSource } from "./tool-agent";
import { fetchSource } from "./tool-fetch";
import { mcpSource } from "./tool-mcp";

export interface Tool {
  name: string;                                     // the source's own name for it
  title: string;                                    // what the page calls it
  description: string;
  parameters: Record<string, unknown>;              // JSON schema of its arguments
  readOnly: boolean;                                // changes nothing anywhere: runs without asking
  run(args: Record<string, unknown>, ctx: { signal: AbortSignal; maxChars: number }): Promise<string>;
}
export type State = "idle" | "connecting" | "connected" | "failed" | "disabled";
export interface Source {
  id: string;                                       // "fetch", "agents", or "mcp:" and the server's name in the config
  label: string; kind: "fetch" | "agent" | "mcp";
  prefix: string;                                   // of its tools' function names ("" for the built-in one)
  detail: string;                                   // what it runs or reaches, for the page to show
  spec: string;                                     // its config, to tell a changed one on reload
  state: State; error?: string; tools: Tool[];
  timeoutMs: number; maxChars: number;
  chars?: number;                                   // what a call gives at a time when the chat asks for no other amount
  connect(): Promise<void>;                         // starts it if it is not running; settles once it runs or failed
  close(): Promise<void>;
  onChange?: () => void;                            // set by the registry: the tool list changed
}
// A tool as offered to the model, under a function name unique among every source's.
export interface Named { fn: string; tool: Tool; source: Source }

export const CONFIG = process.env.CHRYSALIS_MCP_CONFIG ?? `${homedir()}/.config/chrysalis/mcp.json`;
export const DEFAULT_TIMEOUT_S = 60, DEFAULT_MAX_CHARS = 40_000;

let sources: Source[] = [];
let configError: string | undefined;
let names = new Map<string, Named>();

// OpenAI's function names (and Qwen's tool format) take letters, digits, _ and -, up to 64.
export const clean = (s: string) => s.replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "tool";
// The function names, from scratch whenever a tool list changes: sources by id, tools by name, so the same tools get
// the same names, in the same order (the order is the prompt's tools block, which Chrysalis caches).
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

// What the page shows: every source with its state and tools (and their function names). Never a server's
// environment or headers, nor a URL's query: they hold tokens, and the page may be open on another device.
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

// The tools a chat has switched on, sorted by function name (so by source, then tool). Sources not running yet are
// started (and waited for); one that cannot start is left out, with a note saying so.
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

// A tool's run, held to its source's time limit and result size: a longer result is cut, and says so.
export async function run(n: Named, args: Record<string, unknown>, signal: AbortSignal, maxChars?: number): Promise<string> {
  const limit = Math.min(maxChars ?? n.source.chars ?? n.source.maxChars, n.source.maxChars);
  const timeout = AbortSignal.timeout(n.source.timeoutMs);
  const both = AbortSignal.any([signal, timeout]);
  const out = await new Promise<string>((resolve, reject) => {
    const gone = () => reject(timeout.aborted && !signal.aborted ? new Error(`the tool did not answer within ${Math.round(n.source.timeoutMs / 1000)} s`)
      : new DOMException("stopped", "AbortError"));
    if (both.aborted) return gone();
    both.addEventListener("abort", gone, { once: true });
    n.tool.run(args, { signal: both, maxChars: limit }).then(resolve, reject);
  });
  return out.length <= limit ? out
    : `${out.slice(0, limit)}\n\n[The result was cut here: ${limit.toLocaleString("en-US")} of ${out.length.toLocaleString("en-US")} characters.]`;
}
