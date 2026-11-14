import { expect, test } from "bun:test";
import { sessionContextBreakdown, type SessionMessage } from "./breakdown";

const said = (n: number) => "x".repeat(n);
const shares = (messages: SessionMessage[]) => sessionContextBreakdown(messages).map((s) => [s.key, s.tokens, s.percent]);

test("a context is split among the system prompt, what the user said, what the model wrote and what its tools returned, four characters to a token, and the rest of the last reply's measured total is other", () => {
  const messages: SessionMessage[] = [
    { type: "user", text: said(400) },
    { type: "assistant", content: [{ type: "reasoning", text: said(200) }, { type: "text", text: said(200) }, { type: "tool", state: { status: "completed", input: { path: "a", limit: 2 }, content: [{ text: said(368) }] } }],
      tokens: { input: 100, output: 100, reasoning: 0, cache: { read: 800, write: 0 } } },
  ];
  expect(shares(messages)).toEqual([["user", 100, 10], ["assistant", 100, 10], ["tool", 100, 10], ["other", 700, 70]]);
  expect(shares([{ type: "system", text: said(800) }, ...messages])).toEqual([["system", 200, 20], ["user", 100, 10], ["assistant", 100, 10], ["tool", 100, 10], ["other", 500, 50]]);
  expect(shares(messages.slice(0, 1))).toEqual([]);
});

test("a tool's result counts whichever way OpenCode holds it, a failed one by its error, a running one by its input alone", () => {
  const one = (state: object): SessionMessage[] => [{ type: "assistant", content: [{ type: "tool", state: state as never }], tokens: { input: 1000, output: 0 } }];
  expect([shares(one({ status: "completed", input: { a: 1 }, output: said(384) })), shares(one({ status: "error", input: { a: 1 }, error: { message: said(384) } })), shares(one({ status: "error", input: { a: 1 }, error: said(384) })),
    shares(one({ status: "running", input: { a: 1, b: 2 } })), shares(one({ status: "streaming", input: said(400) }))].map((s) => s[0]))
    .toEqual([["tool", 100, 10], ["tool", 100, 10], ["tool", 100, 10], ["tool", 8, 0.8], ["tool", 100, 10]]);
});

test("an estimate larger than the measured total is scaled down to it, each part keeping its share", () => {
  expect(shares([{ type: "user", text: said(3000) }, { type: "assistant", content: [{ type: "text", text: said(1000) }], tokens: { input: 400, output: 100 } }])).toEqual([["user", 375, 75], ["assistant", 125, 25]]);
});
