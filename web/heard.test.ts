// A voice turn's fixes (heard.ts) on what Flash-Next wrote for real transcripts: the misheard words it found are swapped
// in when they sound like what was heard, the meanings it offered instead are not, and nothing else of the turn changes.
import { expect, test } from "bun:test";
import { applyFixes, readStep, soundsLike } from "./heard";

test("a fix is made when the words sound like what was heard, numbers and names as written", () => {
  for (const [heard, said] of [["quinn three point eight twenty seven B", "Qwen3.8-27B"], ["tail", "table"], ["pie game", "Pygame"],
    ["right", "write"], ["whether", "weather"], ["cocoa row", "Kokoro"], ["RS three", "RS3"], ["jason", "JSON"], ["pie torch", "PyTorch"]])
    expect([heard, said, soundsLike(heard, said)]).toEqual([heard, said, true]);
});

test("a meaning offered in place of the words is not", () => {
  for (const [heard, said] of [["RS three", "RuneScape 3"], ["tail", "list"], ["and", "in"], ["snake game", "Pong"]])
    expect([heard, said, soundsLike(heard, said)]).toEqual([heard, said, false]);
});

test("the first step's fixes and tool decision are read as written, and only the transcript's own words are swapped", () => {
  const step = "pie game => Pygame\ntail => list\nnot in it => anything\n</fixes>\n<tool>no";
  const { fixes, tool } = readStep(`<fixes>${step}`);
  expect(tool).toBe(false);
  expect(fixes).toEqual([{ heard: "pie game", said: "Pygame" }, { heard: "tail", said: "list" }, { heard: "not in it", said: "anything" }]);
  const said = "Let's do a snake game and pie game, and report it in a tail.";
  expect(applyFixes(said, fixes)).toEqual({ text: "Let's do a snake game and Pygame, and report it in a tail.", made: [fixes[0]] });
  expect(readStep("<fixes></fixes>\n<tool>yes")).toEqual({ fixes: [], tool: true });
  expect(readStep("<fixes>tail => table")).toEqual({ fixes: [], tool: undefined });       // not closed yet: nothing to act on
});
