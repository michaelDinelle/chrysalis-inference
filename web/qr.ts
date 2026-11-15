import { encode } from "uqr";

const MARGIN = 2;
const modules = (text: string) => encode(text, { ecc: "L", border: MARGIN }).data;

export function qrText(text: string): string {
  const rows = modules(text), light = (y: number, x: number) => y < rows.length && !rows[y][x];
  const lines: string[] = [];
  for (let y = 0; y < rows.length; y += 2)
    lines.push(`\x1b[40;37;1m${rows[y].map((_, x) => (light(y, x) ? (light(y + 1, x) ? "█" : "▀") : light(y + 1, x) ? "▄" : " ")).join("")}\x1b[0m`);
  return lines.join("\n");
}

export function qrSvg(text: string): string {
  const rows = modules(text), n = rows.length;
  const dark = rows.flatMap((row, y) => row.flatMap((on, x) => (on ? [`M${x} ${y}h1v1h-1z`] : []))).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${n} ${n}" width="${n * 8}" height="${n * 8}" shape-rendering="crispEdges">` +
    `<rect width="${n}" height="${n}" fill="#fff"/><path d="${dark}" fill="#000"/></svg>`;
}
