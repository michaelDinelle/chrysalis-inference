// MCP servers as tool sources, through the official SDK: stdio ({command, args, env, cwd}) and Streamable HTTP
// ({url, headers}), as Claude Desktop's and Claude Code's config files write them. A server starts the first time a
// chat turns it on (or Retry, or a reply that needs it) and keeps running; when it fails or goes away it is started
// again after 1, 2, 4 ... up to 60 s. "tools": [names] keeps only those of a server's tools, whatever it lists (a
// remote server can change its list at any time, and its own read-only marks decide what runs without asking). A
// remote URL must be https, its headers often holding a key: plain http reaches only this machine.
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport, getDefaultEnvironment } from "@modelcontextprotocol/sdk/client/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { ToolListChangedNotificationSchema } from "@modelcontextprotocol/sdk/types.js";
import { DEFAULT_MAX_CHARS, DEFAULT_TIMEOUT_S, type Source, type Tool } from "./tools";

const CONNECT_MS = 30_000;                          // npx fetching a server's package the first time takes a while
const LOOPBACK = /^(localhost|127(\.\d{1,3}){3}|\[::1\])$/i;
const strings = (x: unknown) => (x && typeof x === "object" ? Object.fromEntries(Object.entries(x).filter(([, v]) => typeof v === "string")) as Record<string, string> : {});

// An MCP result as text: the model gets text parts as they are; pictures and audio are named, not shown (a chat's
// stored history keeps text only).
function text(r: any): string {
  const parts = (Array.isArray(r?.content) ? r.content : []).map((p: any) =>
    p.type === "text" ? p.text
      : p.type === "image" || p.type === "audio" ? `[${p.type === "image" ? "a picture" : "audio"} (${p.mimeType}) the chat cannot show]`
        : p.type === "resource" ? p.resource?.text ?? `[${p.resource?.uri}]`
          : p.type === "resource_link" ? `${p.name ?? "link"}: ${p.uri}` : "");
  const out = parts.filter(Boolean).join("\n\n") || (r?.structuredContent ? JSON.stringify(r.structuredContent) : "");
  if (r?.isError) throw new Error(out || "the tool failed");
  return out;
}

export function mcpSource(name: string, spec: Record<string, unknown>): Source {
  const url = typeof spec.url === "string" ? spec.url : undefined, command = typeof spec.command === "string" ? spec.command : undefined;
  const args = Array.isArray(spec.args) ? spec.args.map(String) : [];
  const only = Array.isArray(spec.tools) ? new Set(spec.tools.map(String)) : null;
  let detail = "", bad: string | undefined;
  if (spec.type === "sse") bad = "SSE servers are not supported: give the server's Streamable HTTP URL";
  else if (url) {
    try {
      const u = new URL(url);
      detail = u.origin + u.pathname;
      if (u.protocol !== "https:" && !(u.protocol === "http:" && LOOPBACK.test(u.hostname)))
        bad = `"url" must be https (plain http only to this machine)`;
    } catch { bad = `"url" is not a URL`; }
  } else if (command) detail = [command, ...args].join(" ");
  else bad = `needs a "command" (stdio) or a "url" (Streamable HTTP)`;

  let client: Client | undefined, pending: Promise<void> | undefined, closed = false, retry: Timer | undefined, failures = 0, stderr = "";
  const s: Source = {
    id: `mcp:${name}`, label: name, kind: "mcp", prefix: name, detail, spec: JSON.stringify(spec),
    state: spec.disabled === true ? "disabled" : bad ? "failed" : "idle", error: bad, tools: [],
    timeoutMs: (Number(spec.timeout) > 0 ? Number(spec.timeout) : DEFAULT_TIMEOUT_S) * 1000,
    maxChars: Number(spec.maxResultChars) > 0 ? Number(spec.maxResultChars) : DEFAULT_MAX_CHARS,
    connect() {
      if (closed || bad || s.state === "disabled" || s.state === "connected") return Promise.resolve();
      return (pending ??= attempt().finally(() => { pending = undefined; }));
    },
    async close() {
      closed = true;
      clearTimeout(retry);
      const c = client;
      client = undefined;
      await c?.close().catch(() => {});
    },
  };

  const fail = (why: string) => {
    client = undefined;
    s.tools = [];
    s.onChange?.();
    const said = stderr.trim().split("\n").at(-1)?.trim();
    s.state = "failed";
    s.error = said && !why.includes(said) ? `${why} (${said.slice(0, 300)})` : why;
    if (!closed) retry = setTimeout(() => void s.connect(), Math.min(60_000, 1000 * 2 ** failures++));
  };

  const listTools = async (c: Client) => {
    const all: any[] = [];
    let cursor: string | undefined;
    do {
      const r = await c.listTools(cursor ? { cursor } : undefined, { timeout: CONNECT_MS });
      all.push(...r.tools);
      cursor = r.nextCursor;
    } while (cursor);
    s.tools = all.filter((t) => !only || only.has(t.name)).map((t): Tool => ({
      name: t.name, title: t.title ?? t.annotations?.title ?? t.name, description: t.description ?? "",
      parameters: t.inputSchema ?? { type: "object", properties: {} },
      readOnly: t.annotations?.readOnlyHint === true,
      run: async (a, ctx) => text(await c.callTool({ name: t.name, arguments: a }, undefined, { signal: ctx.signal, timeout: s.timeoutMs })),
    }));
    s.onChange?.();
  };

  async function attempt() {
    clearTimeout(retry);
    s.state = "connecting";
    s.error = undefined;
    stderr = "";
    const c = new Client({ name: "kiln-web", version: "1.0.0" });
    const transport = url ? new StreamableHTTPClientTransport(new URL(url), { requestInit: { headers: strings(spec.headers) } })
      : new StdioClientTransport({ command: command!, args, env: { ...getDefaultEnvironment(), ...strings(spec.env) },
        cwd: typeof spec.cwd === "string" ? spec.cwd : undefined, stderr: "pipe" });
    if (transport instanceof StdioClientTransport) transport.stderr?.on("data", (d) => { stderr = (stderr + d).slice(-4000); });
    try {
      let timer: Timer | undefined;
      await Promise.race([c.connect(transport), new Promise((_, no) => { timer = setTimeout(() => no(new Error(`no answer within ${CONNECT_MS / 1000} s`)), CONNECT_MS); })])
        .finally(() => clearTimeout(timer));
      c.setNotificationHandler(ToolListChangedNotificationSchema, () => { if (client === c) void listTools(c).catch(() => {}); });
      await listTools(c);
      if (closed) return void c.close().catch(() => {});
      client = c;
      failures = 0;
      s.state = "connected";
      c.onclose = () => { if (client === c && !closed) fail("the server went away"); };
    } catch (e) {
      await c.close().catch(() => {});
      if (!closed) fail(e instanceof Error ? e.message : String(e));
    }
  }
  return s;
}
