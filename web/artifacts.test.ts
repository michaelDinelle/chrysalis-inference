// The source-level helpers of artifacts (artifacts.ts) that have an outside reference: a version's changes as a line
// diff, against GNU diff's.
import { expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { lineDiff } from "./artifacts";

test("a version's changes take out and put in as few lines as GNU diff --minimal does, and read back to both versions", () => {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-diff-`);
  try {
    let seed = 7;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let t = 0; t < 200; t++) {
      const a = Array.from({ length: Math.floor(rnd() * 30) }, () => `l${Math.floor(rnd() * 8)}`);
      const b = a.flatMap((l) => (rnd() < 0.2 ? [] : rnd() < 0.2 ? [l, `n${Math.floor(rnd() * 5)}`] : [l]));
      if (rnd() < 0.3) b.unshift("head");
      const A = a.join("\n"), B = b.join("\n"), ours = lineDiff(A, B);
      writeFileSync(`${dir}/a`, `${A}\n`);
      writeFileSync(`${dir}/b`, `${B}\n`);
      const gnu = Bun.spawnSync(["diff", "--minimal", "-U0", `${dir}/a`, `${dir}/b`]).stdout.toString().split("\n").filter((l) => /^[-+](?![-+]{2} )/.test(l));
      expect([ours.filter((d) => d.op !== "+").map((d) => d.text).join("\n"), ours.filter((d) => d.op !== "-").map((d) => d.text).join("\n")]).toEqual([A, B]);
      expect(ours.filter((d) => d.op !== " ").length).toBe(gnu.length);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
