import { readdirSync, readFileSync } from "node:fs";

const read = (path: string) => { try { return readFileSync(path, "utf8"); } catch { return ""; } };

export function biggest(sys = "/sys"): string | undefined {
  const nodes = `${sys}/class/kfd/kfd/topology/nodes`;
  let names: string[];
  try { names = readdirSync(nodes); } catch { return; }
  let ordinal = 0, best = -1, most = -1;
  for (const node of names.map(Number).filter(Number.isInteger).sort((a, b) => a - b)) {
    if (!Number(read(`${nodes}/${node}/gpu_id`))) continue;
    const minor = /^drm_render_minor (\d+)$/m.exec(read(`${nodes}/${node}/properties`))?.[1];
    const vram = Number(read(`${sys}/class/drm/renderD${minor}/device/mem_info_vram_total`)) || 0;
    if (vram > most) { best = ordinal; most = vram; }
    ordinal++;
  }
  return best < 0 ? undefined : String(best);
}

export function gpuEnv(env: Record<string, string | undefined> = process.env): Record<string, string> {
  if (env.HIP_VISIBLE_DEVICES !== undefined || env.ROCR_VISIBLE_DEVICES !== undefined) return {};
  const pick = biggest();
  return pick === undefined ? {} : { HIP_VISIBLE_DEVICES: pick };
}

if (import.meta.main) console.log(gpuEnv().HIP_VISIBLE_DEVICES ?? process.env.HIP_VISIBLE_DEVICES ?? "");
