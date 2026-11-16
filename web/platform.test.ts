import { expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gpuFound, gpuHere } from "./gpu";
import { dropDownload, python, ROOT, serverCommand, serverEnv, WATCHED, type Machine } from "./platform";

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
  expect(serverEnv(machine("darwin"), root)).toEqual({ PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1", HF_HOME: join(root, "data", "cache", "huggingface") });
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
    const hip6 = join("C:\\Windows", "System32", "amdhip64_6.dll"), windows = { env: { SystemRoot: "C:\\Windows" } };
    expect([gpuFound(machine("win32", [hip6], windows), root, found).here, gpuFound(machine("win32", [], windows), root, found).here, gpuFound(machine("win32", [], windows), root, found).why?.includes("Windows has no amdhip64_6.dll")]).toEqual([true, false, true]);
    const says = `${dir}/says`;
    writeFileSync(says, "#!/bin/sh\necho 'chrysalis: this computer has no GPU that AMD'\\''s ROCm can use' >&2\nexit 1\n");
    chmodSync(says, 0o755);
    expect([gpuFound(machine("linux"), root, found).why, gpuFound(machine("linux"), root, says).why, gpuFound(machine("linux"), root, none).why]).toEqual([undefined, "this computer has no GPU that AMD's ROCm can use", `${none} stopped with code 1`]);
    expect(gpuFound(machine("linux"), root, `${dir}/missing`).why).toStartWith(`${dir}/missing could not be run: `);
    expect([gpuFound(machine("darwin", [metal], { arch: "arm64", memory: 36 * 2 ** 30 }), root, none).why, gpuFound(machine("darwin", [metal], { arch: "x64" }), root, found).why?.startsWith("a Mac runs it on Apple silicon")]).toEqual([undefined, true]);
  } finally { rmSync(dir, { recursive: true }); }
});

test("cancelling a download removes what had arrived of that model, weights and partial files wherever its files go, and nothing else: the files the setup put there and the other model stay", () => {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-drop-`), m = `${dir}/data/models`;
  try {
    for (const d of ["qwen3.8-flash-next/src/Q8_0", "qwen3.8-flash-next/src/.cache/huggingface/download/Q8_0", ".cache/huggingface/download/qwen3.8-flash-next", ".cache/huggingface/download/qwen3.8-27b", "qwen3.8-27b"]) mkdirSync(`${m}/${d}`, { recursive: true });
    const made = ["qwen3.8-flash-next/mmproj.gguf", "qwen3.8-flash-next/preprocessor_config.json", "qwen3.8-flash-next/src/Q8_0/part.gguf", "qwen3.8-flash-next/src/.cache/huggingface/download/Q8_0/a.incomplete",
      ".cache/huggingface/download/qwen3.8-flash-next/b.incomplete", ".cache/huggingface/download/qwen3.8-27b/c.incomplete", "qwen3.8-27b/qwen3.8-27b.gguf", "tokenizer.json"];
    for (const f of made) writeFileSync(`${m}/${f}`, "x");
    expect(dropDownload("qwen3.8-flash-next", dir).length).toBe(4);
    expect(made.filter((f) => existsSync(`${m}/${f}`))).toEqual(["qwen3.8-flash-next/preprocessor_config.json", ".cache/huggingface/download/qwen3.8-27b/c.incomplete", "qwen3.8-27b/qwen3.8-27b.gguf", "tokenizer.json"]);
    expect(dropDownload("qwen3.8-flash-next", dir)).toEqual([]);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("the download the terminal watches counts each file once: the library reports a file being rebuilt and its bytes arriving as two bars of the same size, and the line read from the downloader never claims more than the files hold", () => {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-watched-`), py = python();
  try {
    mkdirSync(`${dir}/chrysalis`);
    for (const f of ["__init__.py", "get.py", "models.py"]) writeFileSync(`${dir}/chrysalis/${f}`, readFileSync(join(ROOT, "chrysalis", f)));
    writeFileSync(`${dir}/huggingface_hub.py`, [
      "import time",
      "class Info:",
      "    size = 1500000000",
      "class HfApi:",
      "    def get_paths_info(self, repo, paths, revision=None):",
      "        return [Info() for _ in paths]",
      "def hf_hub_download(repo, name, revision=None, local_dir=None, tqdm_class=None):",
      "    rebuilt = tqdm_class(total=Info.size, desc=name + ': reconstructing file', unit='B')",
      "    arrived = tqdm_class(total=Info.size, desc=name + ': downloading bytes', unit='B')",
      "    for _ in range(5):",
      "        arrived.update(Info.size // 5)",
      "        time.sleep(0.3)",
      "        rebuilt.update(Info.size // 5)",
      "",
    ].join("\n"));
    const said = (script: string) => {
      const p = Bun.spawnSync([py, "-c", script, "qwen3.8-27b"], { cwd: dir, stdout: "pipe", stderr: "pipe", env: { ...process.env, PYTHONPATH: dir, PYTHONUTF8: "1" } });
      return { code: p.exitCode, shown: [...p.stderr.toString().matchAll(/([\d.]+) \/ ([\d.]+) GB/g)].map((m) => [Number(m[1]), Number(m[2])]) };
    };
    const watched = said(WATCHED), counted = watched.shown.map(([done]) => done);
    expect([watched.code, watched.shown.length > 2, Math.max(...counted), new Set(watched.shown.map(([, total]) => total)).size]).toEqual([0, true, 3, 1]);
    expect(counted.some((done) => done > 0 && done < 3)).toBe(true);
    const twice = said("import sys\nclass Shown:\n    def __init__(self, out):\n        self.out = out\n    def isatty(self):\n        return True\n    def write(self, text):\n        return self.out.write(text)\n    def flush(self):\n        self.out.flush()\nsys.stderr = Shown(sys.stderr)\nfrom chrysalis import get\nget.fetch(sys.argv[1])");
    expect(Math.max(...twice.shown.map(([done]) => done))).toBeGreaterThan(3);
  } finally { rmSync(dir, { recursive: true, force: true }); }
}, 30_000);
