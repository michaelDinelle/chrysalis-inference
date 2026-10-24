export interface WireCall { id: string; name: string; args: string }

export const assistantTurn = (content: string, reasoning?: string, calls: WireCall[] = []) => ({
  role: "assistant", content, ...(reasoning ? { reasoning_content: reasoning } : {}),
  ...(calls.length ? { tool_calls: calls.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: c.args } })) } : {}),
});

export const toolTurn = (id: string, result: string) => ({ role: "tool", tool_call_id: id, content: result });

export interface ToolChoice { tools: Record<string, { off?: string[] }>; allow?: string[]; loop?: boolean; fetchChars?: number; route?: boolean }

export const DESCRIPTIONS: Record<string, string> = {
  web_search_exa: "Searches the web; returns titles, URLs, dates and excerpts. Use it for anything that may have changed since your " +
    "training (news, prices, versions, schedules, current roles), for niche facts you are unsure of, or when asked to look something up. " +
    "Don't use it for stable knowledge, math, writing, or text already in the chat. Start with one short query of distinctive keywords; " +
    "search again only if the results miss the question. Cite the pages you use by title and URL. If the results don't answer it, say " +
    "so; never describe results you did not receive.",
};

export const noted = (note: string | undefined, text: string) => (note ? `[${note}]\n\n${text}` : text);
export const unnoted = (text: string) => text.replace(/^\[[\s\S]*?\]\n\n/, "");
