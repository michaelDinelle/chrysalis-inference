// Sub-agents, the built-in tool that hands a task to an agent of its own: a fresh conversation with the chat's other
// tools that works on it alone and gives back its final answer. server.ts runs it (subagent()): it needs the reply's
// request, its tool calls' approvals and its events, which a tool's run() is not given, so run() here is never called.
// The tool takes what OpenCode's subagent takes (a short label shown to the user and a self-contained task), not its
// background runs or sessions to resume.
import type { Source, Tool } from "./tools";

// The sub-agent's system prompt: it works for another model, which sees its final message only, and nobody can answer
// its questions.
export const SUBAGENT_SYSTEM = "You are a sub-agent, running locally in Chrysalis Inference. Another assistant, in a chat with a "
  + "user, handed you the task below and is waiting for your answer: it sees only your final message. Work on the task with "
  + "your tools until it is done, then reply with a complete answer that stands on its own: what you found, what it rests on, "
  + "and the sources you used (titles and URLs). If something could not be done or found, say so plainly. Nobody can answer "
  + "questions: where the task leaves something open, choose sensibly and say what you chose. Do what the task asks, nothing more.";

export function agentSource(): Source {
  const tool: Tool = {
    name: "subagent", title: "Sub-agent", readOnly: true,
    description: "Hands one task to a sub-agent: a fresh conversation with your other tools, which works on it alone and returns "
      + "its final answer as this call's result. Use it to research several independent things at once (calls in the same "
      + "message run side by side), or for work whose pages you need only a summary of: what a sub-agent reads stays out of "
      + "your context. It starts knowing nothing of this chat and cannot ask you anything, so put everything it needs in the "
      + "prompt and say what its answer should contain. Don't use it for one quick lookup you can do yourself.",
    parameters: { type: "object", properties: {
      description: { type: "string", description: "A short label for the task, 3 to 5 words, shown to the user" },
      prompt: { type: "string", description: "The task: everything the sub-agent needs to know, and what its answer should contain" },
    }, required: ["description", "prompt"] },
    run() { throw new Error("a sub-agent is run by the web server (server.ts)"); },
  };
  return {
    id: "agents", label: "Sub-agents", kind: "agent", prefix: "", detail: "built in", spec: "agents", state: "connected", tools: [tool],
    timeoutMs: Infinity, maxChars: Infinity,
    async connect() {},
    async close() {},
  };
}
