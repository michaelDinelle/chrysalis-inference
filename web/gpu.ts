import { join } from "node:path";
import { exe, here, library, ROOT, serverEnv, type Machine } from "./platform";

const PICK = exe(join(ROOT, "build", "gpu"));
const CHOSEN = ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"];

export function pick(env: Record<string, string | undefined> = process.env, picker = PICK): Record<string, string> | undefined {
  let out: string;
  try {
    const p = Bun.spawnSync([picker], { stdin: "ignore", stdout: "pipe", stderr: "inherit", env: env as Record<string, string> });
    if (p.exitCode !== 0) return undefined;
    out = p.stdout.toString();
  } catch { return undefined; }
  return Object.fromEntries([...out.matchAll(/^([A-Z_]+)=(.*?)\r?$/gm)].map((m) => [m[1], m[2]]));
}

export function gpuEnv(env: Record<string, string | undefined> = process.env, picker = PICK): Record<string, string> {
  if (CHOSEN.some((k) => env[k] !== undefined)) return {};
  return pick(env, picker) ?? {};
}

export function gpuFound(m: Machine = here(), root = ROOT, picker = exe(join(root, "build", "gpu"), m)): { here: boolean; env: Record<string, string> } {
  if (m.platform === "darwin") return { here: m.arch === "arm64" && m.memory >= 30 * 2 ** 30 && m.exists(library("libchrysalis", m, root)), env: {} };
  const nvidia = m.platform === "win32" ? join(m.env.SystemRoot ?? m.env.SYSTEMROOT ?? "C:\\Windows", "System32", "nvcuda.dll") : "/dev/nvidiactl";
  if (m.exists(nvidia) && m.exists(library("libchrysalis", m, root)) && m.exists(library("libchrysalis_fn", m, root))) return { here: true, env: {} };
  const found = pick({ ...m.env, ...serverEnv(m, root) }, picker);
  return { here: found !== undefined, env: CHOSEN.some((k) => m.env[k] !== undefined) ? {} : found ?? {} };
}
export const gpuHere = (m: Machine = here(), root = ROOT, picker = exe(join(root, "build", "gpu"), m)) => gpuFound(m, root, picker).here;

if (import.meta.main) for (const [k, v] of Object.entries(gpuEnv())) console.log(`${k}=${v}`);
