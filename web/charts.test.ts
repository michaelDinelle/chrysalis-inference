import { expect, test } from "bun:test";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const REPO = join(import.meta.dir, "..");
const check = (root: string) => {
  const run = Bun.spawnSync(["python3", `${root}/media/charts.py`, "--check"], { stdout: "pipe", stderr: "pipe" });
  return { code: run.exitCode, said: String(run.stderr) };
};
const copy = () => {
  const root = mkdtempSync(join(tmpdir(), "charts-"));
  mkdirSync(`${root}/media`);
  for (const file of ["README.md", "media/charts.py", "media/charts.json", "media/flash-next.png", "media/27b.png"]) copyFileSync(`${REPO}/${file}`, `${root}/${file}`);
  return root;
};
const edit = (path: string, from: string, to: string) => {
  const text = readFileSync(path, "utf8");
  expect(text).toContain(from);
  writeFileSync(path, text.replace(from, to));
};

test("the README's tables and both charts come from media/charts.json", () => {
  expect(check(REPO)).toEqual({ code: 0, said: "" });
  expect(check(copy())).toEqual({ code: 0, said: "" });
});

test("a number changed by hand in the README is caught", () => {
  const root = copy();
  edit(`${root}/README.md`, "| First token, short prompt | 0.24 s |", "| First token, short prompt | 0.20 s |");
  const { code, said } = check(root);
  expect(code).toBe(1);
  expect(said).toContain("0.20 s");
  expect(said).toContain("0.24 s");
});

test("a new number that reaches neither the README nor the charts is caught for all three", () => {
  const root = copy();
  const numbers = JSON.parse(readFileSync(`${root}/media/charts.json`, "utf8"));
  numbers.fn[0].ttft[0] = 0.2;
  writeFileSync(`${root}/media/charts.json`, JSON.stringify(numbers));
  const { code, said } = check(root);
  expect(code).toBe(1);
  for (const file of ["README.md has", "media/flash-next.png", "media/27b.png"]) expect(said).toContain(file);
});

test("a chart that was not drawn from the numbers is caught", () => {
  const root = copy();
  copyFileSync(`${REPO}/media/run.png`, `${root}/media/27b.png`);
  const { code, said } = check(root);
  expect(code).toBe(1);
  expect(said).toContain("media/27b.png");
  expect(said).not.toContain("flash-next");
});

test("a table the numbers do not generate is caught", () => {
  const root = copy();
  edit(`${root}/README.md`, "## Links\n", "| Writing, short prompt | 500 tok/s |\n\n## Links\n");
  const { code, said } = check(root);
  expect(code).toBe(1);
  expect(said).toContain("table row");
});

test("a table left out of the README is caught", () => {
  const root = copy();
  edit(`${root}/README.md`, "**Qwen3.8-27B, M4 Max**", "**Qwen3.8-27B on a Mac**");
  const { code, said } = check(root);
  expect(code).toBe(1);
  expect(said).toContain("Qwen3.8-27B, M4 Max");
});
