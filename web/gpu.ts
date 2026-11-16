import { dirname, join } from "node:path";
import { exe, here, library, ROOT, serverEnv, type Machine } from "./platform";

const PICK = exe(join(ROOT, "build", "gpu"));
const CHOSEN = ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"];

export function pick(env: Record<string, string | undefined> = process.env, picker = PICK): { env?: Record<string, string>; why?: string } {
  try {
    const p = Bun.spawnSync([picker], { stdin: "ignore", stdout: "pipe", stderr: "pipe", env: env as Record<string, string> });
    if (p.exitCode === 0) return { env: Object.fromEntries([...p.stdout.toString().matchAll(/^([A-Z_]+)=(.*?)\r?$/gm)].map((m) => [m[1], m[2]])) };
    const said = p.stderr.toString().trim().split(/\r?\n/).pop()!.replace(/^chrysalis: /, "");
    return { why: said || `${picker} stopped with code ${p.exitCode}` };
  } catch (e) { return { why: `${picker} could not be run: ${e instanceof Error ? e.message : String(e)}` }; }
}

export function gpuEnv(env: Record<string, string | undefined> = process.env, picker = PICK): Record<string, string> {
  if (CHOSEN.some((k) => env[k] !== undefined)) return {};
  const p = pick(env, picker);
  if (p.why) console.error(`chrysalis: ${p.why}`);
  return p.env ?? {};
}

export function gpuFound(m: Machine = here(), root = ROOT, picker = exe(join(root, "build", "gpu"), m)): { here: boolean; env: Record<string, string>; why?: string } {
  if (m.platform === "darwin") {
    const here = m.arch === "arm64" && m.memory >= 30 * 2 ** 30 && m.exists(library("libchrysalis", m, root));
    return { here, env: {}, ...(here ? {} : { why: "a Mac runs it on Apple silicon with 32 GB of memory or more, once make kernels has built its engine" }) };
  }
  const nvidia = m.platform === "win32" ? join(m.env.SystemRoot ?? m.env.SYSTEMROOT ?? "C:\\Windows", "System32", "nvcuda.dll") : "/dev/nvidiactl";
  if (m.exists(nvidia) && m.exists(library("libchrysalis", m, root)) && m.exists(library("libchrysalis_fn", m, root))) return { here: true, env: {} };
  if (m.platform === "win32" && !m.exists(join(dirname(nvidia), "amdhip64_7.dll")))
    return { here: false, env: {}, why: "AMD's driver is missing or older than Chrysalis needs (Windows has no amdhip64_7.dll): install the current AMD Software: Adrenalin Edition for a Radeon RX 7900" };
  const found = pick({ ...m.env, ...serverEnv(m, root) }, picker);
  return { here: !!found.env, env: CHOSEN.some((k) => m.env[k] !== undefined) ? {} : found.env ?? {}, ...(found.why ? { why: found.why } : {}) };
}
export const gpuHere = (m: Machine = here(), root = ROOT, picker = exe(join(root, "build", "gpu"), m)) => gpuFound(m, root, picker).here;

if (import.meta.main) for (const [k, v] of Object.entries(gpuEnv())) console.log(`${k}=${v}`);
