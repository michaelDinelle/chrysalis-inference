
export interface Word { start: number; end: number }

const WORDY = /[\p{L}\p{N}]/u;
const UNREAD = /```[\s\S]*?(?:```|$)|\$\$[\s\S]*?(?:\$\$|$)|\$[^$\n]+\$|!\[[^\]]*\]\([^)]*\)/g;
const TABLE_OR_RULE = /^\s*(?:\|.*\|.*|[|:\-\s]*-{3,}[|:\-\s]*|[*_=-]{3,})\s*$/;
const MARKS = /^\s*(?:(?:#{1,6}|>|[-*+•]|\d+[.)])\s+)+/;

export function shownWords(text: string): Word[] {
  const skip: [number, number][] = [];
  for (const m of text.matchAll(UNREAD)) skip.push([m.index!, m.index! + m[0].length]);
  const out: Word[] = [];
  let at = 0;
  for (const line of text.split("\n")) {
    const start = at;
    at += line.length + 1;
    if (TABLE_OR_RULE.test(line)) continue;
    const from = start + (line.match(MARKS)?.[0].length ?? 0);
    for (const m of line.slice(from - start).matchAll(/\S+/g)) {
      const s = from + m.index!, e = s + m[0].length;
      if (WORDY.test(m[0]) && !skip.some(([a, b]) => s < b && e > a)) out.push({ start: s, end: e });
    }
  }
  return out;
}

export function unitOf(text: string, k: number): number {
  let u = 0;
  for (let i = 0; i < k && u < text.length; i++) u += text.codePointAt(u)! > 0xffff ? 2 : 1;
  return u;
}
export const pointOf = (text: string, u: number) => [...text.slice(0, u)].length;
