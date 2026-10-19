// A reply's tool rounds as kiln is sent them, built here for both sides: server.ts sends each next round of a reply,
// the page resends the whole history with the next question. kiln's prompt cache makes a round, and the next question,
// an append only when the history comes back exactly as it was sent, so both build it with these and nothing else
// (the call ids and argument strings as kiln wrote them, the results as the tools returned them).
export interface WireCall { id: string; name: string; args: string }

export const assistantTurn = (content: string, reasoning?: string, calls: WireCall[] = []) => ({
  role: "assistant", content, ...(reasoning ? { reasoning_content: reasoning } : {}),
  ...(calls.length ? { tool_calls: calls.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: c.args } })) } : {}),
});

export const toolTurn = (id: string, result: string) => ({ role: "tool", tool_call_id: id, content: result });

// What the page asks server.ts for with a reply (kiln_web in the request, which kiln never sees): the tool sources this
// chat has on, each with the tools switched off in it (by the source's own names), and the tools that run without asking.
export interface ToolChoice { tools: Record<string, { off?: string[] }>; allow?: string[]; loop?: boolean; fetchChars?: number }
