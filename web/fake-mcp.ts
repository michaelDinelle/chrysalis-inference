import { appendFileSync } from "node:fs";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const log = (line: string) => { if (process.env.FAKE_MCP_LOG) appendFileSync(process.env.FAKE_MCP_LOG, `${line}\n`); };

export function fakeMcp(): McpServer {
  const s = new McpServer({ name: "fake", version: "1.0.0" });
  s.registerTool("echo", { title: "Echo", description: "Says the text back", inputSchema: { text: z.string() }, annotations: { readOnlyHint: true } },
    async ({ text }) => ({ content: [{ type: "text", text: `echo: ${text}` }] }));
  s.registerTool("write_note", { description: "Saves a note", inputSchema: { text: z.string() } },
    async ({ text }) => (log(`wrote ${text}`), { content: [{ type: "text", text: `saved: ${text}` }] }));
  s.registerTool("slow", { description: "Waits", inputSchema: { ms: z.number() }, annotations: { readOnlyHint: true } },
    ({ ms }, extra) => new Promise((resolve) => {
      const t = setTimeout(() => resolve({ content: [{ type: "text", text: "slept" }] }), ms);
      extra.signal.addEventListener("abort", () => { clearTimeout(t); log("cancelled slow"); resolve({ content: [{ type: "text", text: "cancelled" }] }); });
    }));
  s.registerTool("big", { description: "A long result", inputSchema: { n: z.number() }, annotations: { readOnlyHint: true } },
    async ({ n }) => ({ content: [{ type: "text", text: Array.from({ length: n }, (_, i) => String.fromCharCode(97 + (i % 26))).join("") }] }));
  s.registerTool("get.weather", { description: "A name that needs cleaning", annotations: { readOnlyHint: true } },
    async () => ({ content: [{ type: "text", text: "sunny" }] }));
  return s;
}

if (import.meta.main) await fakeMcp().connect(new StdioServerTransport());
