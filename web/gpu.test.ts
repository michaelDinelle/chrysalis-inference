import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { biggest, gpuEnv } from "./gpu";

function machine(nodes: { id: number; minor?: number; vram?: number }[]) {
  const sys = mkdtempSync(`${tmpdir()}/chrysalis-gpu-`);
  nodes.forEach((n, i) => {
    const node = `${sys}/class/kfd/kfd/topology/nodes/${i}`;
    mkdirSync(node, { recursive: true });
    writeFileSync(`${node}/gpu_id`, `${n.id}\n`);
    writeFileSync(`${node}/properties`, `cpu_cores_count 0\nsimd_count 4\ndrm_render_minor ${n.minor ?? 0}\nlocal_mem_size 0\n`);
    if (n.minor === undefined) return;
    mkdirSync(`${sys}/class/drm/renderD${n.minor}/device`, { recursive: true });
    writeFileSync(`${sys}/class/drm/renderD${n.minor}/device/mem_info_vram_total`, `${n.vram}\n`);
  });
  return sys;
}

const picked = (nodes: Parameters<typeof machine>[0]) => { const sys = machine(nodes); try { return biggest(sys); } finally { rmSync(sys, { recursive: true }); } };
const CPU = { id: 0 }, IGPU = { id: 62991, minor: 129, vram: 536870912 }, CARD = { id: 34772, minor: 128, vram: 25753026560 };

test("the card with the most memory is picked wherever the integrated graphics are listed, counting only the nodes that are GPUs", () => {
  expect(picked([CPU, IGPU, CARD])).toBe("1");
  expect(picked([CPU, CARD, IGPU])).toBe("0");
  expect(picked([CARD, CPU, IGPU])).toBe("0");
  expect(picked([IGPU, CPU, CPU, CARD])).toBe("1");
});

test("one GPU is picked, and a computer with none or without the compute driver picks nothing", () => {
  expect(picked([CPU, CARD])).toBe("0");
  expect(picked([CPU])).toBeUndefined();
  expect(biggest("/nowhere")).toBeUndefined();
});

test("a choice already made in the environment is left alone", () => {
  expect(gpuEnv({ HIP_VISIBLE_DEVICES: "0" })).toEqual({});
  expect(gpuEnv({ ROCR_VISIBLE_DEVICES: "" })).toEqual({});
});
