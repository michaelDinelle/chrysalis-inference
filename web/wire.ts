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
// the tools' descriptions as Settings has them (by function name), and voice mode's routing (route: whether the reply
// runs a tool is settled before its words go out, in server.ts) with its texts.
export interface Route { ask: string; preambles: string[]; asks: string[]; announces: string[] }
export interface ToolChoice {
  tools: Record<string, { off?: string[] }>; allow?: string[]; loop?: boolean; fetchChars?: number;
  descriptions?: Record<string, string>; route?: boolean | Partial<Route>;
}

// The defaults of the texts both sides use: the page shows and edits them (Settings), server.ts falls back to them
// for a request that brings none. The question is short because the guidance on when a tool helps is in the tools'
// own descriptions (DESCRIPTIONS), which the prompt already holds and kiln has cached.
export const ROUTING: Route = {
  ask: "Do you need one of your tools to reply to my last message? Answer yes or no.",
  preambles: ["Let me look that up.", "Let me check.", "One moment, I'll look."],
  asks: ["search", "google", "look it up", "look that up", "look this up", "look them up", "look up", "browse", "fetch", "online",
    "on the web", "on the internet", "open the link", "open this link", "open that link", "open the page", "open this page"],
  announces: ["search", "searching", "look that up", "look it up", "look this up", "look them up", "look those up", "look up",
    "looking up", "looking that up", "looking it up", "look for", "looking for", "look into", "looking into", "check on", "check for",
    "check online", "check the", "check that", "check it", "check this", "checking", "find out", "find that", "find the", "find it",
    "find a", "find some", "finding", "pull up", "pulling up", "browse", "browsing", "fetch", "fetching", "open the link",
    "open that link", "open the page", "opening"],
};

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

// A list of phrases as a pattern for whole words, whatever the spacing between them (null: an empty list).
export function phrasePattern(phrases: string[], before = ""): RegExp | null {
  const alts = phrases.map((p) => p.trim()).filter(Boolean).map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+"));
  return alts.length ? new RegExp(`\\b${before}(${alts.join("|")})\\b`, "i") : null;
}
