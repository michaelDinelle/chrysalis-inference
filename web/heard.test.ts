import { expect, test } from "bun:test";
import { applyFixes, fixedParts, readHeard, soundsLike } from "./heard";

test("a fix is made when the words sound like what was heard, numbers and names as written", () => {
  for (const [heard, said] of [["quinn three point eight twenty seven B", "Qwen3.8-27B"], ["tail", "table"], ["pie game", "Pygame"],
    ["right", "write"], ["whether", "weather"], ["cocoa row", "Kokoro"], ["RS three", "RS3"], ["jason", "JSON"], ["pie torch", "PyTorch"]])
    expect([heard, said, soundsLike(heard, said)]).toEqual([heard, said, true]);
});

test("a meaning offered in place of the words is not", () => {
  for (const [heard, said] of [["RS three", "RuneScape 3"], ["tail", "list"], ["and", "in"], ["snake game", "Pong"]])
    expect([heard, said, soundsLike(heard, said)]).toEqual([heard, said, false]);
});

test("the check's fixes and tool decision are read as written, and only the transcript's own words are swapped", () => {
  const { fixes, tool } = readHeard("pie game => Pygame\ntail => list\nnot in it => anything\ntool: no");
  expect(tool).toBe(false);
  expect(fixes).toEqual([{ heard: "pie game", said: "Pygame" }, { heard: "tail", said: "list" }, { heard: "not in it", said: "anything" }]);
  const said = "Let's do a snake game and pie game, and report it in a tail.", made = applyFixes(said, fixes);
  expect(made).toEqual({ text: "Let's do a snake game and Pygame, and report it in a tail.", made: [fixes[0]] });
  expect(fixedParts(made.text, made.made)).toEqual(["Let's do a snake game and ", fixes[0], ", and report it in a tail."]);
  expect(readHeard("tool: yes")).toEqual({ fixes: [], tool: true });
  expect(readHeard("")).toEqual({ fixes: [], tool: false });
});

test("a remark in brackets after a fix is not part of what was said", () => {
  const { fixes } = readHeard("quinn three point eight twenty seven B => Qwen3.8-27B (likely; possibly Qwen3.5-27B)\nRS three => RS3 (RuneScape 3)");
  expect(fixes).toEqual([{ heard: "quinn three point eight twenty seven B", said: "Qwen3.8-27B" }, { heard: "RS three", said: "RS3" }]);
  expect(applyFixes("What is quinn three point eight twenty seven B and RS three?", fixes).text).toBe("What is Qwen3.8-27B and RS3?");
});
