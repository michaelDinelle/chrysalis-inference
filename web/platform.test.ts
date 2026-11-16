import { expect, test } from "bun:test";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gpuFound, gpuHere } from "./gpu";
import { python, serverCommand, serverEnv, type Machine } from "./platform";

const root = join(tmpdir(), "chrysalis-root");
const machine = (platform: NodeJS.Platform, files: string[] = [], o: Partial<Machine> = {}): Machine => ({
  platform, arch: "x64", memory: 64 * 2 ** 30, env: {}, exists: (p) => files.includes(p), read: () => "", ...o,
});

test("the model server runs on the bundled Python wherever its packager put it: bin/python on Linux and macOS, Scripts\\python.exe or python.exe at the venv's top on Windows", () => {
  expect(serverCommand("qwen3.8-27b", machine("linux"), root)).toEqual([join(root, "data/venv/bin/python"), "-m", "chrysalis", "serve", "--host", "127.0.0.1", "--use", "qwen3.8-27b"]);
  expect(python(machine("darwin"), root)).toBe(join(root, "data/venv/bin/python"));
  expect(python(machine("win32", [join(root, "data/venv/Scripts/python.exe"), join(root, "data/venv/python.exe")]), root)).toBe(join(root, "data/venv/Scripts/python.exe"));
  expect(python(machine("win32", [join(root, "data/venv/python.exe")]), root)).toBe(join(root, "data/venv/python.exe"));
  expect(serverCommand(undefined, machine("win32"), root)).toEqual([join(root, "data/venv/Scripts/python.exe"), "-m", "chrysalis", "serve", "--host", "127.0.0.1"]);
});

test("on Windows the model server finds the engine's DLLs and the GPU runtime beside them first, through the one Path entry Windows keeps, whatever its case, and is told where its tuned matmul choices are kept before the engine loads", () => {
  const env = serverEnv(machine("win32", [], { env: { Path: "C:\\Windows\\System32", HOME: "x" } }), root);
  expect(Object.keys(env).filter((k) => k.toUpperCase() === "PATH")).toEqual(["Path"]);
  expect(env.Path.split(";")).toEqual([join(root, "build"), join(root, "build", "runtime"), "C:\\Windows\\System32"]);
  expect([env.PYTHONUTF8, "LD_LIBRARY_PATH" in env, env.CHRYSALIS_LT_CACHE]).toEqual(["1", false, join(root, "data", "cache", "hipblaslt.txt")]);
});

test("on Linux the model server loads the bundled ROCm and the ggml build this CPU runs fastest, ahead of what the system had", () => {
  const vnni = machine("linux", [], { env: { LD_LIBRARY_PATH: "/opt/x" }, read: (p) => (p === "/proc/cpuinfo" ? "flags\t: fpu avx2 avx512f avx512_vnni\n" : "") });
  expect(serverEnv(vnni, root).LD_LIBRARY_PATH.split(":")).toEqual([join(root, "data/rocm/lib"), join(root, "data/rocm/lib/rocm_sysdeps/lib"), join(root, "ggml/avx512"), "/opt/x"]);
  const plain = serverEnv(machine("linux", [], { read: () => "flags\t: fpu avx2 avx512f avx512vl\n" }), root);
  expect(plain.LD_LIBRARY_PATH.split(":").at(-1)).toBe(join(root, "ggml/avx2"));
  expect(Object.keys(serverEnv(machine("darwin"), root)).sort()).toEqual(["PYTHONIOENCODING", "PYTHONUTF8"]);
});

test("a GPU Chrysalis runs on is a Mac with Apple silicon, 30 GB and the Metal engine built, an NVIDIA driver with both CUDA engines built, or one the engine's own picker finds", () => {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-gpu-`), found = `${dir}/found`, none = `${dir}/none`;
  writeFileSync(found, "#!/bin/sh\necho HIP_VISIBLE_DEVICES=0\n");
  writeFileSync(none, "#!/bin/sh\nexit 1\n");
  chmodSync(found, 0o755);
  chmodSync(none, 0o755);
  try {
    const metal = join(root, "build/libchrysalis.dylib");
    expect([gpuHere(machine("darwin", [metal], { arch: "arm64", memory: 36 * 2 ** 30 }), root, none), gpuHere(machine("darwin", [metal], { arch: "arm64", memory: 16 * 2 ** 30 }), root, found),
      gpuHere(machine("darwin", [metal], { arch: "x64" }), root, found), gpuHere(machine("darwin", [], { arch: "arm64" }), root, found)]).toEqual([true, false, false, false]);
    const cuda = [join("C:\\Windows", "System32", "nvcuda.dll"), join(root, "build/libchrysalis.dll"), join(root, "build/libchrysalis_fn.dll")];
    expect([gpuHere(machine("win32", cuda, { env: { SystemRoot: "C:\\Windows" } }), root, none), gpuHere(machine("win32", cuda.slice(0, 2), { env: { SystemRoot: "C:\\Windows" } }), root, none)]).toEqual([true, false]);
    expect([gpuHere(machine("linux", ["/dev/nvidiactl", join(root, "build/libchrysalis.so"), join(root, "build/libchrysalis_fn.so")]), root, none),
      gpuHere(machine("linux"), root, found), gpuHere(machine("linux"), root, none), gpuHere(machine("linux"), root, `${dir}/missing`)]).toEqual([true, true, false, false]);
    const hip7 = join("C:\\Windows", "System32", "amdhip64_7.dll"), windows = { env: { SystemRoot: "C:\\Windows" } };
    expect([gpuFound(machine("win32", [hip7], windows), root, found).here, gpuFound(machine("win32", [], windows), root, found).here, gpuFound(machine("win32", [], windows), root, found).why?.includes("Windows has no amdhip64_7.dll")]).toEqual([true, false, true]);
    const says = `${dir}/says`;
    writeFileSync(says, "#!/bin/sh\necho 'chrysalis: this computer has no GPU that AMD'\\''s ROCm can use' >&2\nexit 1\n");
    chmodSync(says, 0o755);
    expect([gpuFound(machine("linux"), root, found).why, gpuFound(machine("linux"), root, says).why, gpuFound(machine("linux"), root, none).why]).toEqual([undefined, "this computer has no GPU that AMD's ROCm can use", `${none} stopped with code 1`]);
    expect(gpuFound(machine("linux"), root, `${dir}/missing`).why).toStartWith(`${dir}/missing could not be run: `);
    expect([gpuFound(machine("darwin", [metal], { arch: "arm64", memory: 36 * 2 ** 30 }), root, none).why, gpuFound(machine("darwin", [metal], { arch: "x64" }), root, found).why?.startsWith("a Mac runs it on Apple silicon")]).toEqual([undefined, true]);
  } finally { rmSync(dir, { recursive: true }); }
});
