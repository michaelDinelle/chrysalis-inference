import { mkdirSync, renameSync, rmSync, symlinkSync } from "node:fs";

export const MODELS = ["qwen3.8-27b", "qwen3.8-flash-next"];

export function point(memory: string, kept: string, hidden: boolean) {
  for (const model of MODELS) {
    const link = `${memory}/snapshots/${model}`;
    mkdirSync(`${kept}/${model}`, { recursive: true });
    rmSync(`${link}.next`, { force: true });
    symlinkSync(hidden ? `${memory}/none/${model}` : `${kept}/${model}`, `${link}.next`);
    renameSync(`${link}.next`, link);
  }
}
