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
// chat has on, each with the tools switched off in it (by the source's own names), the tools that run without asking,
// and route: a voice reply, whether it runs a tool is settled before its words go out (server.ts).
export interface ToolChoice { tools: Record<string, { off?: string[] }>; allow?: string[]; loop?: boolean; fetchChars?: number; route?: boolean }

// Descriptions kiln sends instead of a server's own, by the tool's name on its server. A description is where a model
// learns when to call a tool and when not (the vendors' guidance: what it does, when to use it and when not, its limits),
// and Exa's own says what the search does but not when.
export const DESCRIPTIONS: Record<string, string> = {
  web_search_exa: "Searches the web; returns titles, URLs, dates and excerpts. Use it for anything that may have changed since your " +
    "training (news, prices, versions, schedules, current roles), for niche facts you are unsure of, or when asked to look something up. " +
    "Don't use it for stable knowledge, math, writing, or text already in the chat. Start with one short query of distinctive keywords; " +
    "search again only if the results miss the question. Cite the pages you use by title and URL. If the results don't answer it, say " +
    "so; never describe results you did not receive.",
};

// A question's note (voice mode on, off: app.ts) as kiln is sent it, before the question's own text. What is said is
// checked for asks without it: the note itself speaks of searches.
export const noted = (note: string | undefined, text: string) => (note ? `[${note}]\n\n${text}` : text);
export const unnoted = (text: string) => text.replace(/^\[[\s\S]*?\]\n\n/, "");
