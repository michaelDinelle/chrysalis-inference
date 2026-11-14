export type SessionContextBreakdownKey = "system" | "user" | "assistant" | "tool" | "other";

export type SessionContextBreakdownSegment = {
  key: SessionContextBreakdownKey;
  tokens: number;
  width: number;
  percent: number;
};

export type SessionTokens = { input?: number; output?: number; reasoning?: number; cache?: { read?: number; write?: number } };

type Part = {
  type: string;
  text?: string;
  state?: { status: string; input?: unknown; output?: unknown; content?: { text?: string }[]; error?: string | { message?: string } };
};

export type SessionMessage = {
  type: string;
  text?: string;
  content?: Part[];
  tokens?: SessionTokens;
};

const estimateTokens = (chars: number) => Math.ceil(chars / 4);
const toPercent = (tokens: number, input: number) => (tokens / input) * 100;
const toPercentLabel = (tokens: number, input: number) => Math.round(toPercent(tokens, input) * 10) / 10;

const charsFromAssistantPart = (part: Part) => {
  if (part.type === "text") return { assistant: part.text?.length ?? 0, tool: 0 };
  if (part.type === "reasoning") return { assistant: part.text?.length ?? 0, tool: 0 };
  if (part.type !== "tool" || !part.state) return { assistant: 0, tool: 0 };

  if (part.state.status === "streaming") return { assistant: 0, tool: typeof part.state.input === "string" ? part.state.input.length : 0 };
  const input = Object.keys((part.state.input as object | undefined) ?? {}).length * 16;
  const output = typeof part.state.output === "string" ? part.state.output.length : (part.state.content ?? []).reduce((sum, item) => sum + (item.text?.length ?? 0), 0);
  const error = typeof part.state.error === "string" ? part.state.error.length : (part.state.error?.message?.length ?? 0);
  if (part.state.status === "completed") return { assistant: 0, tool: input + output };
  if (part.state.status === "error") return { assistant: 0, tool: input + (output || error) };
  return { assistant: 0, tool: input };
};

const build = (
  tokens: { system: number; user: number; assistant: number; tool: number; other: number },
  input: number,
) => {
  return [
    {
      key: "system",
      tokens: tokens.system,
    },
    {
      key: "user",
      tokens: tokens.user,
    },
    {
      key: "assistant",
      tokens: tokens.assistant,
    },
    {
      key: "tool",
      tokens: tokens.tool,
    },
    {
      key: "other",
      tokens: tokens.other,
    },
  ]
    .filter((x) => x.tokens > 0)
    .map((x) => ({
      key: x.key,
      tokens: x.tokens,
      width: toPercent(x.tokens, input),
      percent: toPercentLabel(x.tokens, input),
    })) as SessionContextBreakdownSegment[];
};

export function estimateSessionContextBreakdown(args: { messages: readonly SessionMessage[]; input: number; systemPrompt?: string }) {
  if (!args.input) return [];

  const counts = args.messages.reduce(
    (acc, msg) => {
      if (msg.type === "user") return { ...acc, user: acc.user + (msg.text?.length ?? 0) };

      if (msg.type !== "assistant") return acc;
      const assistant = (msg.content ?? []).reduce(
        (sum, part) => {
          const next = charsFromAssistantPart(part);
          return {
            assistant: sum.assistant + next.assistant,
            tool: sum.tool + next.tool,
          };
        },
        { assistant: 0, tool: 0 },
      );
      return {
        ...acc,
        assistant: acc.assistant + assistant.assistant,
        tool: acc.tool + assistant.tool,
      };
    },
    {
      system: args.systemPrompt?.length ?? 0,
      user: 0,
      assistant: 0,
      tool: 0,
    },
  );

  const tokens = {
    system: estimateTokens(counts.system),
    user: estimateTokens(counts.user),
    assistant: estimateTokens(counts.assistant),
    tool: estimateTokens(counts.tool),
  };
  const estimated = tokens.system + tokens.user + tokens.assistant + tokens.tool;

  if (estimated <= args.input) {
    return build({ ...tokens, other: args.input - estimated }, args.input);
  }

  const scale = args.input / estimated;
  const scaled = {
    system: Math.floor(tokens.system * scale),
    user: Math.floor(tokens.user * scale),
    assistant: Math.floor(tokens.assistant * scale),
    tool: Math.floor(tokens.tool * scale),
  };
  const total = scaled.system + scaled.user + scaled.assistant + scaled.tool;
  return build({ ...scaled, other: Math.max(0, args.input - total) }, args.input);
}

export function sessionContextBreakdown(messages: readonly SessionMessage[]) {
  const tokens = messages.findLast((message) => message.type === "assistant" && message.tokens !== undefined)?.tokens;
  if (!tokens) return [];
  const total = (tokens.input ?? 0) + (tokens.output ?? 0) + (tokens.reasoning ?? 0) + (tokens.cache?.read ?? 0) + (tokens.cache?.write ?? 0);
  return estimateSessionContextBreakdown({ messages, input: total, systemPrompt: messages.findLast((message) => message.type === "system")?.text?.trim() });
}
