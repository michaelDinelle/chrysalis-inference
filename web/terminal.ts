import { readFileSync } from "node:fs";

const CSI = "\x1b[";
const wrap = (on: string, off: string) => (s: string) => `${CSI}${on}m${s}${CSI}${off}m`;
export const paint = {
  bold: wrap("1", "22"), dim: wrap("2", "22"), red: wrap("31", "39"), green: wrap("32", "39"), yellow: wrap("33", "39"),
  magenta: wrap("35", "39"), gray: wrap("90", "39"),
};
export const plain = (s: string) => s.replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g, "");
export const clean = (s: string) => s.replace(/[\x00-\x1f\x7f-\x9f]/g, "");
const width = (s: string) => Bun.stringWidth(s);

export function fold(s: string, room: number, indent = ""): string[] {
  const lines: string[] = [];
  let line = "";
  for (const word of s.split(" ")) {
    if (width(line) && width(line) + 1 + width(word) > room) { lines.push(line); line = indent + word; }
    else line = line || lines.length ? `${line} ${word}` : word;
  }
  return [...lines, line];
}

const RAMP = " .:-=+*#%@", FINE = 8;
let wings: { w: number; h: number; ink: Uint8Array } | undefined;
function drawing(cols: number, rows: number) {
  const w = cols * FINE, h = rows * FINE;
  if (wings?.w === w && wings.h === h) return wings;
  const svg = readFileSync(`${import.meta.dir}/logo.svg`, "utf8"), box = /viewBox="([^"]+)"/.exec(svg)![1].split(/\s+/).map(Number);
  const shapes = /\sd="([^"]+)"/.exec(svg)![1].split("M").filter(Boolean).map((part) => (part.match(/-?[\d.]+/g) ?? []).map(Number));
  const k = Math.min(w / box[2], (h * 2) / box[3]), ox = (w - k * box[2]) / 2 - k * box[0], oy = (h * 2 - k * box[3]) / 2 - k * box[1];
  const ink = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const at = (y + 0.5) * 2, cuts: number[] = [];
    for (const s of shapes) for (let i = 0; i < s.length; i += 2) {
      const j = (i + 2) % s.length, y0 = s[i + 1] * k + oy, y1 = s[j + 1] * k + oy;
      if ((y0 <= at) !== (y1 <= at)) cuts.push(s[i] * k + ox + ((at - y0) / (y1 - y0)) * (s[j] * k - s[i] * k));
    }
    cuts.sort((a, b) => a - b);
    for (let i = 0; i + 1 < cuts.length; i += 2) for (let x = Math.max(0, Math.ceil(cuts[i] - 0.5)); x < Math.min(w, cuts[i + 1] - 0.5); x++) ink[y * w + x] = 1;
  }
  return (wings = { w, h, ink });
}

export function butterfly(cols: number, rows: number, time: number, shown = 1): string[] {
  const { w, ink } = drawing(cols, rows), beat = 0.84 + 0.16 * Math.cos(time * 1.7), mid = w / 2;
  return Array.from({ length: rows }, (_, row) => {
    let line = "";
    for (let col = 0; col < cols; col++) {
      let sum = 0;
      for (let y = 0; y < FINE; y++) for (let x = 0; x < FINE; x++) {
        const from = Math.round(mid + (col * FINE + x + 0.5 - mid) / beat);
        if (from >= 0 && from < w) sum += ink[(row * FINE + y) * w + from];
      }
      const grain = Math.abs(Math.sin(col * 12.9898 + row * 78.233) * 43758.5453) % 1, glow = 0.8 + 0.2 * Math.sin(time * 2.3 - col * 0.35 - row * 0.5);
      line += grain < shown ? RAMP[Math.min(RAMP.length - 1, Math.round(Math.sqrt(sum / (FINE * FINE)) * glow * (RAMP.length - 1) * 1.25))] : " ";
    }
    return line.trimEnd();
  });
}

const BAR = /([\d.]+) \/ ([\d.]+) GB\s+(?:(\d+) MB\/s, (\d+):(\d\d) left)?/g;
export function progress(said: string): { done: number; total: number; rate: number } | undefined {
  const at = [...plain(said).matchAll(BAR)].pop();
  return at ? { done: Number(at[1]) * 1e9, total: Number(at[2]) * 1e9, rate: Number(at[3] ?? 0) * 1e6 } : undefined;
}
