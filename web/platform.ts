import { existsSync, mkdirSync, readFileSync, realpathSync, rmdirSync } from "node:fs";
import { tmpdir, totalmem } from "node:os";
import { join } from "node:path";

export const ROOT = join(import.meta.dir, "..");
export const HOME = process.env.CHRYSALIS_HOME || ROOT;
export const MODELS = ["qwen3.8-27b", "qwen3.8-flash-next"];

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

export function python(m: Machine = here(), root = HOME) {
  const tries = m.platform === "win32" ? ["data/venv/Scripts/python.exe", "data/venv/python.exe"] : ["data/venv/bin/python"];
  return join(root, tries.find((p) => m.exists(join(root, p))) ?? tries[0]);
}

const keyOf = (env: Record<string, string | undefined>, name: string) => Object.keys(env).find((k) => k.toUpperCase() === name) ?? name;
const before = (env: Record<string, string | undefined>, name: string, dirs: string[], sep: string) => {
  const key = keyOf(env, name), was = env[key];
  return { [key]: [...dirs, ...(was ? [was] : [])].join(sep) };
};

export function serverEnv(m: Machine = here(), root = ROOT): Record<string, string> {
  const utf8 = { PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8", HF_HOME: join(root === ROOT ? HOME : root, "data", "cache", "huggingface") };
  if (m.platform === "win32")
    return { ...utf8, CHRYSALIS_LT_CACHE: join(root, "data", "cache", "hipblaslt.txt"), ...before(m.env, "PATH", [join(root, "build"), join(root, "build", "runtime")], ";") };
  if (m.platform === "darwin") return utf8;
  const ggml = join(root, "ggml", /\bavx512_vnni\b/.test(m.read("/proc/cpuinfo")) ? "avx512" : "avx2");
  return { ...utf8, ...before(m.env, "LD_LIBRARY_PATH", [join(root, "data/rocm/lib"), join(root, "data/rocm/lib/rocm_sysdeps/lib"), ggml], ":") };
}

export const serverCommand = (model?: string, m: Machine = here(), root = HOME) =>
  [python(m, root), "-m", "chrysalis", "serve", "--host", "127.0.0.1", ...(model ? ["--use", model] : [])];
export const getCommand = (model: string, m: Machine = here(), root = HOME) => [python(m, root), "-m", "chrysalis", "get", model];

export const ourScript = (file: string) => ({ cmd: [process.execPath, file], env: { BUN_BE_BUN: "1" } });

export function stopTree(proc: { pid: number; kill: () => void; exitCode: number | null }, m: Machine = here()) {
  if (proc.exitCode !== null) return;
  if (m.platform === "win32") {
    try { if (Bun.spawnSync(["taskkill", "/T", "/F", "/PID", String(proc.pid)], { stdout: "ignore", stderr: "ignore" }).exitCode === 0) return; } catch {  }
  }
  proc.kill();
}

const GUARD = join(import.meta.dir, "guard.ts");
export type Kept = Bun.Subprocess<"pipe", "ignore", "ignore">;
export function keep(cmd: string[], env: Record<string, string | undefined>): Kept {
  const guard = ourScript(GUARD);
  return Bun.spawn([...guard.cmd, ...cmd], { stdin: "pipe", stdout: "ignore", stderr: "ignore", env: { ...env, ...guard.env } });
}
export const release = (proc?: Kept) => { try { proc?.stdin.end(); } catch {  } };
export const noCores = (cmd: string[]) => (Bun.which("prlimit") ? ["prlimit", "--core=0", ...cmd] : cmd);

export const memoryRoot = (m: Machine = here()) => (m.platform === "linux" && m.exists("/dev/shm") ? "/dev/shm" : tmpdir());

const quiet = (cmd: string[]) => { const r = Bun.spawnSync(cmd, { stdin: "ignore", stdout: "pipe", stderr: "ignore" }); return r.exitCode === 0 ? r.stdout.toString() : undefined; };
const RAM = "chrysalis-ram-", RAM_MB = 1024;
let ram: { root: string; drop: () => void } | undefined;

function macRam() {
  const eject = (dev: string, at: string) => { quiet(["umount", "-f", at]); quiet(["hdiutil", "detach", "-force", dev]); try { rmdirSync(at); } catch {  } };
  for (const [, dev, at, pid] of (quiet(["mount"]) ?? "").matchAll(new RegExp(`^(/dev/disk\\d+) on (.*/${RAM}(\\d+)) \\(`, "gm"))) {
    try { process.kill(Number(pid), 0); } catch (e) { if ((e as { code?: string }).code !== "EPERM") eject(dev, at); }
  }
  const dev = quiet(["hdiutil", "attach", "-nomount", `ram://${RAM_MB * 2048}`])?.trim().split(/\s/)[0], at = join(realpathSync(tmpdir()), `${RAM}${process.pid}`);
  if (!dev) throw new Error("Incognito keeps its files in memory only, and macOS made no RAM disk for them");
  mkdirSync(at, { recursive: true });
  if (quiet(["newfs_hfs", "-M", "700", dev]) === undefined || quiet(["mount", "-t", "hfs", "-o", "nobrowse,noatime,nosuid", dev, at]) === undefined) {
    eject(dev, at);
    throw new Error("Incognito keeps its files in memory only, and macOS did not mount its RAM disk");
  }
  return { root: at, drop: () => eject(dev, at) };
}

function winRam() {
  const fsp = join(process.env["ProgramFiles(x86)"] ?? "C:\\Program Files (x86)", "WinFsp", "bin"), has = () => existsSync(join(fsp, "winfsp-x64.dll"));
  const ps = (script: string) => quiet(["powershell", "-NoProfile", "-NonInteractive", "-Command", script]);
  if (!has()) {
    console.log("Incognito keeps its files in memory only: Windows asks once to add the RAM drive they live on (WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos, https://github.com/winfsp/winfsp)");
    ps(`Start-Process msiexec -Verb RunAs -Wait -ArgumentList '/i', '"${join(ROOT, "build", "winfsp.msi")}"', '/qn', '/norestart'`);
    if (!has()) throw new Error("Incognito keeps its files in memory only, and Windows did not add WinFsp, the RAM drive they live on");
  }
  const mine = `Get-CimInstance Win32_Process -Filter "name='ramdrive.exe'" | ForEach-Object { "$($_.ProcessId) $($_.CommandLine)" }`;
  for (const [, drive, owner] of (ps(mine) ?? "").matchAll(/^(\d+) .*\\chrysalis\\ram(\d+)\s*$/gm)) {
    try { process.kill(Number(owner), 0); } catch { try { process.kill(Number(drive)); } catch {  } }
  }
  const share = `\\chrysalis\\ram${process.pid}`, at = `\\${share}`;
  const drive = keep([join(ROOT, "build", "ramdrive.exe"), "-i", "-F", "NTFS", "-n", "1048576", "-s", String(2 ** 31), "-u", share],
    { ...process.env, ...before(process.env, "PATH", [fsp], ";") });
  for (let tries = 0; !existsSync(at) && drive.exitCode === null && tries < 200; tries++) Bun.sleepSync(50);
  if (!existsSync(at)) { release(drive); throw new Error("Incognito keeps its files in memory only, and its RAM drive did not start"); }
  return { root: at, drop: () => release(drive) };
}

export function ramRoot(m: Machine = here()) {
  if (m.platform === "linux") {
    if (!m.exists("/dev/shm")) throw new Error("Incognito keeps its files in memory only, and this system has no /dev/shm");
    return "/dev/shm";
  }
  return (ram ??= m.platform === "darwin" ? macRam() : winRam()).root;
}
export function ramDrop() {
  const had = !!ram;
  ram?.drop();
  ram = undefined;
  return had;
}

export async function utf8Console(m: Machine = here()) {
  if (m.platform !== "win32") return;
  try {
    const { dlopen, FFIType } = await import("bun:ffi");
    const k = dlopen("kernel32.dll", { SetConsoleOutputCP: { args: [FFIType.u32], returns: FFIType.i32 }, SetConsoleCP: { args: [FFIType.u32], returns: FFIType.i32 } });
    k.symbols.SetConsoleOutputCP(65001);
    k.symbols.SetConsoleCP(65001);
  } catch {  }
}
