import { expect, test } from "bun:test";
import { pointOf, shownWords, unitOf } from "./spoken";

const REPLIES = [
  "## Plan\n\nHere is **the plan**, step by step:\n\n1. Buy *milk* today.\n2. Call `mom` after lunch\n- a [nice link](https://example.com) here\n> quoted words, too",
  "Text before the code.\n\n```python\nprint('not read')\n```\n\nAfter the code.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nThe table holds two rows. ---\n\n***\n\nEnd - fine.",
  "Great 🎉 job! It's done — almost: 3 of 4 tests, 75% and $5.\n\n### Next\n\n+ one more ~~thing~~",
  "Only these words.\n\n* starred item\n    * nested item",
];

test("the words counted as a reply's are the words its voice reads, in order", () => {
  const py = Bun.spawnSync([`${import.meta.dir}/../data/venv/bin/python`, "-I", "-c",
    "import json, sys; sys.path.insert(0, sys.argv[1]); from chrysalis.voice.talk import speakable; print(json.dumps([speakable(t) for t in json.load(sys.stdin)]))",
    `${import.meta.dir}/..`], { stdin: Buffer.from(JSON.stringify(REPLIES)) });
  expect(py.exitCode).toBe(0);
  const read: string[] = JSON.parse(py.stdout.toString());
  const bare = (w: string) => w.replace(/\]\([^)]*\)/g, "").replace(/[^\p{L}\p{N}']/gu, "").toLowerCase();
  for (const [i, text] of REPLIES.entries()) {
    const shown = shownWords(text).map((w) => bare(text.slice(w.start, w.end))).filter(Boolean);
    expect([i, shown]).toEqual([i, read[i].split(/\s+/).map(bare).filter(Boolean)]);
  }
});

test("a place in the text counted in code points, as the service counts, and back", () => {
  const text = "a🎉b c";
  expect([unitOf(text, 2), text.slice(unitOf(text, 2)), pointOf(text, 3)]).toEqual([3, "b c", 2]);
});
