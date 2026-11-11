import { hasCheckpoint } from "./wire";

export interface Turn { role: string; content?: unknown; reasoning_content?: string; tool_calls?: { function: { name: string; arguments: string } }[] }

const CHARS_PER_TOKEN = 4;
const RESERVE_SHARE = 0.15;
const RESERVE_MIN = 16_000;
const KEEP_TOKENS = 15_000, KEEP_SHARE = 0.2;
const TOOL_OUTPUT_CHARS = 1_250, USER_CHARS = 8_000, REQUEST_CHARS = 4_000;
const MEDIA_TOKENS: Record<string, number> = { image_url: 1_500, file: 2_000, video_url: 8_000 };

export const tokens = (text: string) => Math.ceil(text.length / CHARS_PER_TOKEN);

const textOf = (content: unknown) => typeof content === "string" ? content
  : Array.isArray(content) ? content.map((p: { text?: string }) => p?.text ?? "").join("\n") : "";

export function tokensIn(m: Turn) {
  const media = Array.isArray(m.content) ? m.content.reduce((n: number, p: { type?: string }) => n + (MEDIA_TOKENS[p?.type ?? ""] ?? 0), 0) : 0;
  return media + tokens(textOf(m.content)) + tokens(m.reasoning_content ?? "") + tokens((m.tool_calls ?? []).map((c) => c.function.name + c.function.arguments).join(""));
}

export const ceiling = (window: number) => window - Math.max(Math.floor(window * RESERVE_SHARE), window >= 2 * RESERVE_MIN ? RESERVE_MIN : 0);

export const summaryRoom = (window: number) => Math.max(ceiling(window), window - RESERVE_MIN);

export const covers = (m: Turn) => m.role === "user" && hasCheckpoint(textOf(m.content));

const clip = (text: string, most: number) => text.length <= most ? text
  : `${text.slice(0, most / 2)}\n[… ${(text.length - most).toLocaleString("en-US")} characters omitted …]\n${text.slice(-most / 2)}`;

const lines = (m: Turn) => {
  if (covers(m)) return [];
  if (m.role === "user") return [`[User]: ${clip(textOf(m.content), USER_CHARS)}`];
  if (m.role === "tool") return [`[Tool result]: ${clip(textOf(m.content), TOOL_OUTPUT_CHARS)}`];
  return [...(textOf(m.content) ? [`[Assistant]: ${textOf(m.content)}`] : []), ...(m.tool_calls ?? []).map((c) => `[Assistant tool call]: ${c.function.name}(${c.function.arguments})`)];
};

export function recentText(messages: Turn[], limit: number) {
  const most = Math.min(KEEP_TOKENS, Math.floor(limit * KEEP_SHARE)) * CHARS_PER_TOKEN, kept: string[] = [];
  let used = 0;
  for (const line of messages.flatMap(lines).reverse()) {
    const room = most - used;
    if (line.length > room) {
      if (!kept.length) kept.push(clip(line, room));
      break;
    }
    kept.push(line);
    used += line.length + 2;
  }
  return kept.reverse().join("\n\n");
}

const TEMPLATE = `You MUST use this format for your response (you may omit sections that aren't applicable). Do not include the <template> tags in your response.
<template>
## Objective
- [one or two brief sentences describing what the user is trying to accomplish]

## Requirements
- [constraints, preferences, requirements, and scope boundaries stated by the user, or "(none)"]

## Decisions
- [decisions already made and why, or "(none)"]

## Work State
Break the objective into smaller goals and report which are completed, which are being worked on, and which are blocked.
### Completed
- [goals that have been completed; otherwise "(none)"]

### Active
- [goals currently being worked on; otherwise "(none)"]

### Blocked
- [anything blocking progress, and why; otherwise "(none)"]

## Findings
- [every fact, figure, quote, code and result found so far that the answer needs, exactly as found; otherwise "(none)"]

## Sources
- [the titles and URLs those findings rest on; otherwise "(none)"]

## Next Move
1. [ordered list of next actions, or "(none)"]

## Important Context
- [facts the next agent cannot continue without and cannot easily find on its own; or "(none)"]
</template>`;

const RULES = `Rules:
- Keep each section concise. Use terse, single-line bullets, not prose paragraphs or nested lists.
- Keep every finding the work still needs: what was read and is not written here has to be read again.
- Preserve exact names, numbers, codes, commands, error strings, URLs, and identifiers.
- Carry forward only user questions or requests that remain unanswered or require further action. Do not repeat ones that newer history has answered or resolved. Preserve exact wording when carrying one forward.
- Do not mention the summary process or that context was compacted.`;

const HEADINGS = TEMPLATE.split("\n").filter((l) => l.startsWith("##")).map((l) => l.trim());

export const filled = (text: string) => text.split("\n").some((l) => HEADINGS.includes(l.trim()));

const SHARED = [
  "Summarize only what the user and the assistant said and did. Leave out instructions and setup the assistant was given rather than told by the user: the next agent receives current versions of these separately.",
  TEMPLATE, RULES, "Do not continue the task or call tools.",
  "Return only the structured summary in the requested format. Do not include a preamble, explanation, or other commentary.",
];

export const summaryPrompt = (update: boolean) => (update ? [
  "Update the existing checkpoint in the conversation above into one consolidated summary.",
  "Newer history always takes precedence over the existing checkpoint. Preserve previous information unless newer history clearly contradicts, supersedes, resolves, or makes it stale. If something is no longer relevant to continuing the work, you may remove it.",
  "Incorporate newer requirements, decisions, findings, progress, and context. Reconcile Work State and Next Move: move completed work out of Active, remove resolved blockers and answered questions, and preserve unresolved or pending work.",
  "Return only the updated Markdown sections. Do not reproduce the `<conversation-checkpoint>`, `<summary>`, or `<recent-context>` wrapper tags from the previous checkpoint.",
  ...SHARED,
] : ["You MUST summarize the conversation above into a structured summary that will be given to another agent to resume the work.", ...SHARED]).join("\n\n");

export const NUDGE = "The previous response did not fill in the required summary template. Do not call tools. Return the summary as text using the exact section headings from the template.";

export const CONTINUE_AFTER_BREAK = "The previous response was interrupted. Continue from where you left off without repeating completed content.";

const TOO_LONG = [/prompt is too long/i, /input is too long for (?:the )?(?:requested|selected) model/i, /too large for the model'?s configured input limit/i,
  /exceeded this model context window limit/i, /exceeds the context window/i, /exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))/i,
  /input token count.*exceeds the maximum/i, /number of input tokens exceeds maximum length/i, /tokens in request more than max tokens allowed/i, /maximum prompt length is \d+/i,
  /reduce the length of the messages/i, /requested input length \d+ exceeds maximum input length/i, /maximum context length is \d+ tokens/i,
  /exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?/i, /input(?: \(\d+ tokens\))? is longer than the model'?s context length/i, /exceeds the limit of \d+/i,
  /exceeds the available context size/i, /greater than the context length/i, /`inputs` tokens \+ `max_new_tokens` must be <= \d+/i, /context window exceeds limit/i,
  /context window exceeded/i, /exceeded model token limit/i, /context[_ ]length[_ ]exceeded/i, /context length is only \d+ tokens/i, /input length.*exceeds.*context length/i,
  /prompt (?:too long|exceeds max length)/i, /too large for model with \d+ maximum context length/i, /prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?/i,
  /model_context_window_exceeded/i, /range of input length should be/i, /too many tokens/i, /token limit exceeded/i, /request_too_large/i];
const NOT_TOO_LONG = [/^(throttling error|service unavailable):/i, /rate limit/i, /too many requests/i, /max[_ ]tokens must be less than/i, /too many tokens(?: per day)?, please wait/i];

export const saysTooLong = (message: string) => !NOT_TOO_LONG.some((p) => p.test(message)) && TOO_LONG.some((p) => p.test(message));
