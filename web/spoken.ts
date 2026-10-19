// Where a spoken reply is: the service sends each piece of speech with its range in the reply's text (code points) and
// the page shows the words said so far in their own color and the rest grey. A position in the text becomes a count of
// words, which the page finds again among the words it shows: the reply's markdown counted as it is shown, without
// what is not read (code blocks, tables, list and heading marks; voice/talk.py speakable) or not shown as words (math
// typeset, a picture, whose alt text is read).

export interface Word { start: number; end: number }   // UTF-16 offsets in the text

const WORDY = /[\p{L}\p{N}]/u;
// Spans of the text the page shows but does not read (code blocks, math, pictures), and lines it reads none of.
const UNREAD = /```[\s\S]*?(?:```|$)|\$\$[\s\S]*?(?:\$\$|$)|\$[^$\n]+\$|!\[[^\]]*\]\([^)]*\)/g;
const TABLE_OR_RULE = /^\s*(?:\|.*\|.*|[|:\-\s]*-{3,}[|:\-\s]*|[*_=-]{3,})\s*$/;
const MARKS = /^\s*(?:(?:#{1,6}|>|[-*+•]|\d+[.)])\s+)+/;

// The words of markdown as the page shows them, in order: runs of non-space with a letter or digit in them, outside
// code blocks, math, pictures and tables, past a line's heading, quote and list marks.
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

// The UTF-16 offset of code point k of text, and back.
export function unitOf(text: string, k: number): number {
  let u = 0;
  for (let i = 0; i < k && u < text.length; i++) u += text.codePointAt(u)! > 0xffff ? 2 : 1;
  return u;
}
export const pointOf = (text: string, u: number) => [...text.slice(0, u)].length;
