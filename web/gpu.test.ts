import { expect, test } from "bun:test";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { gpuEnv } from "./gpu";

function picker(script: string) {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-gpu-`), path = `${dir}/gpu`;
  writeFileSync(path, `#!/bin/sh\n${script}\n`);
  chmodSync(path, 0o755);
  return { path, done: () => rmSync(dir, { recursive: true }) };
}

test("the model server is given the GPU the engine's own picker names, and nothing when the picker finds none or is not there", () => {
  const found = picker("echo HIP_VISIBLE_DEVICES=1"), none = picker("echo 'chrysalis: none of this computer'\\''s GPUs can run it' >&2; exit 1");
  try {
    expect(gpuEnv({ PATH: process.env.PATH }, found.path)).toEqual({ HIP_VISIBLE_DEVICES: "1" });
    expect(gpuEnv({ PATH: process.env.PATH }, none.path)).toEqual({});
    expect(gpuEnv({ PATH: process.env.PATH }, `${found.path}-missing`)).toEqual({});
  } finally { found.done(); none.done(); }
});

test("a GPU already chosen in the environment is left alone, and the picker is not asked", () => {
  const loud = picker("echo HIP_VISIBLE_DEVICES=7");
  try {
    for (const key of ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]) expect(gpuEnv({ [key]: "0" }, loud.path)).toEqual({});
  } finally { loud.done(); }
});
