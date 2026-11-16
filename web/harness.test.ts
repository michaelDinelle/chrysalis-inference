import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { state } from "./harness";

test("OpenCode installed with npm on Windows is found where npm put it, even by a Chrysalis started before npm's folder was on its Path", () => {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-npm-`), was = { PATH: process.env.PATH, APPDATA: process.env.APPDATA, HOME: process.env.HOME };
  try {
    Object.assign(process.env, { PATH: "", APPDATA: dir, HOME: dir });
    expect(state().state).toBe("absent");
    mkdirSync(`${dir}/npm/node_modules/@opencode/cli/bin`, { recursive: true });
    writeFileSync(`${dir}/npm/node_modules/@opencode/cli/bin/opencode.exe`, "");
    expect(state().state).toBe("off");
  } finally {
    Object.assign(process.env, was);
    if (was.APPDATA === undefined) delete process.env.APPDATA;
    rmSync(dir, { recursive: true, force: true });
  }
});
