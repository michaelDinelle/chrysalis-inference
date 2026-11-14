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
