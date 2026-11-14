const PICK = `${import.meta.dir}/../build/gpu`;

export function gpuEnv(env: Record<string, string | undefined> = process.env, pick = PICK): Record<string, string> {
  if (env.HIP_VISIBLE_DEVICES !== undefined || env.ROCR_VISIBLE_DEVICES !== undefined || env.CUDA_VISIBLE_DEVICES !== undefined) return {};
  let out: string;
  try {
    const p = Bun.spawnSync([pick], { stdin: "ignore", stdout: "pipe", stderr: "inherit", env: env as Record<string, string> });
    if (p.exitCode !== 0) return {};
    out = p.stdout.toString();
  } catch { return {}; }
  return Object.fromEntries([...out.matchAll(/^([A-Z_]+)=(.*)$/gm)].map((m) => [m[1], m[2]]));
}

if (import.meta.main) for (const [k, v] of Object.entries(gpuEnv())) console.log(`${k}=${v}`);
