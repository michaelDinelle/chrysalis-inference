import { appendFileSync } from "node:fs";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const log = (line: string) => { if (process.env.FAKE_MCP_LOG) appendFileSync(process.env.FAKE_MCP_LOG, `${line}\n`); };

const WORDS = "river stone lantern harbor copper meadow signal winter orchard engine marble thread window garden shadow bridge candle forest silver anchor mirror compass ladder violet canyon feather tunnel summit breeze cobalt walnut ember glacier pepper saddle thimble quarry mosaic plume".split(" ");
const hash = (s: string) => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 9973, 7);
export const partCode = (book: string, part: number) => `CODE-${hash(book)}-${part}`;
const PART_CHARS = Number(process.env.FAKE_MCP_PART_CHARS);
function readPart(book: string, part: number, chars: number) {
  let seed = (hash(book) * 7919 + part) >>> 0, text = "";
  const next = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
  while (text.length < chars) text += `${Array.from({ length: 8 + Math.floor(next() * 10) }, () => WORDS[Math.floor(next() * WORDS.length)]).join(" ")}. `;
  const middle = text.indexOf(". ", Math.floor(chars / 2)) + 2;
  return `${text.slice(0, middle)}The code of part ${part} of ${book} is ${partCode(book, part)}. ${text.slice(middle)}`;
}

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
  if (PART_CHARS) s.registerTool("read_part", { title: "Read part", description: "Reads one numbered part of a book, which holds that part's code", inputSchema: { book: z.string(), part: z.number() }, annotations: { readOnlyHint: true } },
    async ({ book, part }) => ({ content: [{ type: "text", text: readPart(book.toLowerCase(), part, PART_CHARS) }] }));
  s.registerTool("get.weather", { description: "A name that needs cleaning", annotations: { readOnlyHint: true } },
    async () => ({ content: [{ type: "text", text: "sunny" }] }));
  return s;
}

if (import.meta.main) await fakeMcp().connect(new StdioServerTransport());
