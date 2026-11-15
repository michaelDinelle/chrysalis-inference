import { existsSync, readFileSync } from "node:fs";
import { tmpdir, totalmem } from "node:os";
import { join } from "node:path";

export const ROOT = join(import.meta.dir, "..");

export interface Machine {
  platform: NodeJS.Platform;
  arch: string;
  memory: number;
  env: Record<string, string | undefined>;
  exists: (path: string) => boolean;
  read: (path: string) => string;
}

export const here = (): Machine => ({
  platform: process.platform, arch: process.arch, memory: totalmem(), env: process.env, exists: existsSync,
  read: (path) => { try { return readFileSync(path, "utf8"); } catch { return ""; } },
});

export const exe = (path: string, m: Machine = here()) => (m.platform === "win32" ? `${path}.exe` : path);
export const library = (name: string, m: Machine = here(), root = ROOT) =>
  join(root, "build", `${name}${m.platform === "darwin" ? ".dylib" : m.platform === "win32" ? ".dll" : ".so"}`);

export function python(m: Machine = here(), root = ROOT) {
  const tries = m.platform === "win32" ? ["data/venv/Scripts/python.exe", "data/venv/python.exe"] : ["data/venv/bin/python"];
  return join(root, tries.find((p) => m.exists(join(root, p))) ?? tries[0]);
}

const keyOf = (env: Record<string, string | undefined>, name: string) => Object.keys(env).find((k) => k.toUpperCase() === name) ?? name;
const before = (env: Record<string, string | undefined>, name: string, dirs: string[], sep: string) => {
  const key = keyOf(env, name), was = env[key];
  return { [key]: [...dirs, ...(was ? [was] : [])].join(sep) };
};

export function serverEnv(m: Machine = here(), root = ROOT): Record<string, string> {
  const utf8 = { PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" };
  if (m.platform === "win32") return { ...utf8, ...before(m.env, "PATH", [join(root, "build"), join(root, "build", "runtime")], ";") };
  if (m.platform === "darwin") return utf8;
  const ggml = join(root, "ggml", /\bavx512_vnni\b/.test(m.read("/proc/cpuinfo")) ? "avx512" : "avx2");
  return { ...utf8, ...before(m.env, "LD_LIBRARY_PATH", [join(root, "data/rocm/lib"), join(root, "data/rocm/lib/rocm_sysdeps/lib"), ggml], ":"),
    ...(m.exists("/dev/dxg") ? { HSA_ENABLE_DXG_DETECTION: "1" } : {}) };
}

export const serverCommand = (model?: string, m: Machine = here(), root = ROOT) =>
  [python(m, root), "-m", "chrysalis", "serve", "--host", "127.0.0.1", ...(model ? ["--use", model] : [])];
export const getCommand = (model: string, m: Machine = here(), root = ROOT) => [python(m, root), "-m", "chrysalis", "get", model];

export const ourScript = (file: string) => ({ cmd: [process.execPath, file], env: { BUN_BE_BUN: "1" } });

export const memoryRoot = (m: Machine = here()) => (m.platform === "linux" && m.exists("/dev/shm") ? "/dev/shm" : tmpdir());

export async function utf8Console(m: Machine = here()) {
  if (m.platform !== "win32") return;
  try {
    const { dlopen, FFIType } = await import("bun:ffi");
    const k = dlopen("kernel32.dll", { SetConsoleOutputCP: { args: [FFIType.u32], returns: FFIType.i32 }, SetConsoleCP: { args: [FFIType.u32], returns: FFIType.i32 } });
    k.symbols.SetConsoleOutputCP(65001);
    k.symbols.SetConsoleCP(65001);
  } catch {  }
}
