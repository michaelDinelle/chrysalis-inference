// What a voice turn's words were, by Qwen's own reading: the reply's first step lists what speech recognition misheard
// (<fixes>, a line each, "heard => said") and whether a tool is needed (<tool>yes|no</tool>), and the turn's text is
// the transcript with those fixes made. A fix is only a swap of the transcript's own words, so nothing else of what the
// user said can go missing (asked to repeat the whole message instead, Flash-Next dropped sentences: "I want to visit
// Quinn's bakery this weekend." came back as "Quinn's bakery"), and only one that sounds like what was heard is made:
// a model asked for misheard words also offers what it thinks was meant ("tail" => "list", "RS three" => "RuneScape 3").

const ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
  "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

// A number as it is said, so "27B" and "twenty seven B" sound alike (whole numbers up to 9999, a decimal point read out).
function spoken(n: string): string {
  if (n.includes(".")) return n.split(".").map((p, i) => (i ? [...p].map((d) => ONES[+d]).join(" ") : spoken(p))).join(" point ");
  const v = Number(n);
  if (!Number.isInteger(v) || v > 9999) return [...n].map((d) => ONES[+d] ?? "").join(" ");
  if (v < 20) return ONES[v];
  if (v < 100) return `${TENS[Math.floor(v / 10)]}${v % 10 ? ` ${ONES[v % 10]}` : ""}`;
  if (v < 1000) return `${ONES[Math.floor(v / 100)]} hundred${v % 100 ? ` ${spoken(String(v % 100))}` : ""}`;
  return `${spoken(String(Math.floor(v / 1000)))} thousand${v % 1000 ? ` ${spoken(String(v % 1000))}` : ""}`;
}

// The consonants of words as they sound (a rough metaphone): vowels and the letters that are mostly silent or vowel-like
// go, letters that sound the same become one, doubles one: "quinn", "Qwen" -> "kn"; "pie game", "Pygame" -> "pkm".
export function sound(text: string): string {
  const words = text.toLowerCase().replace(/(\d+(?:\.\d+)?)/g, " $1 ").replace(/[^a-z0-9. ]+/g, " ").split(/\s+/).filter(Boolean)
    .map((w) => (/^\d/.test(w) ? spoken(w.replace(/\.$/, "")) : w.replace(/\./g, ""))).join(" ");
  const s = words.replace(/ph/g, "f").replace(/gh/g, "").replace(/^kn|\skn/g, " n").replace(/wr/g, "r").replace(/ck/g, "k")
    .replace(/qu?/g, "k").replace(/c(?=[eiy])/g, "s").replace(/c/g, "k").replace(/x/g, "ks").replace(/z/g, "s").replace(/v/g, "f")
    .replace(/d/g, "t").replace(/b/g, "p").replace(/g/g, "k").replace(/[aeiouwhy\s]/g, "");
  return s.replace(/(.)\1+/g, "$1");
}

function distance(a: string, b: string): number {
  let row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const next = [i];
    for (let j = 1; j <= b.length; j++) next[j] = Math.min(row[j] + 1, next[j - 1] + 1, row[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    row = next;
  }
  return row[b.length];
}

// Alike when at most a third of the longer sound's consonants differ ("tail" / "table": "tl" / "tpl", one in three).
export const soundsLike = (heard: string, said: string) => {
  const a = sound(heard), b = sound(said);
  return a.length > 0 && b.length > 0 && distance(a, b) <= Math.floor(Math.max(a.length, b.length) / 3);
};

export interface Fix { heard: string; said: string }
// The first step as Qwen writes it: its fixes and its tool decision (undefined while either is not written yet).
export function readStep(step: string): { fixes: Fix[]; tool?: boolean } {
  const body = step.match(/<fixes>([\s\S]*?)<\/fixes>/)?.[1] ?? "";
  const fixes = body.split("\n").flatMap((line) => {
    const m = line.match(/^\s*(.+?)\s*=>\s*(.+?)\s*$/);
    return m ? [{ heard: m[1].replace(/^["']|["']$/g, ""), said: m[2].replace(/^["']|["']$/g, "") }] : [];
  });
  const tool = step.match(/<tool>\s*(yes|no)\b/i)?.[1].toLowerCase();
  return { fixes, tool: tool === undefined ? undefined : tool === "yes" };
}

// The transcript in pieces: its words as they were, and where a fix is made, the fix (a swap of the transcript's own
// words that sounds like them; a fix inside another's words is not made). The rest of the fixes are dropped.
export function heardParts(transcript: string, fixes: Fix[]): (string | Fix)[] {
  const parts: (string | Fix)[] = [transcript];
  for (const f of fixes) {
    const k = parts.findIndex((p) => typeof p === "string" && p.toLowerCase().includes(f.heard.toLowerCase()));
    if (k < 0 || !f.said || !soundsLike(f.heard, f.said)) continue;
    const p = parts[k] as string, at = p.toLowerCase().indexOf(f.heard.toLowerCase());
    parts.splice(k, 1, p.slice(0, at), { heard: p.slice(at, at + f.heard.length), said: f.said }, p.slice(at + f.heard.length));
  }
  return parts.filter((p) => p !== "");
}

export function applyFixes(transcript: string, fixes: Fix[]): { text: string; made: Fix[] } {
  const parts = heardParts(transcript, fixes), made = parts.filter((p): p is Fix => typeof p !== "string");
  return { text: parts.map((p) => (typeof p === "string" ? p : p.said)).join(""), made: made.map((f) => fixes.find((x) => x.heard.toLowerCase() === f.heard.toLowerCase())!) };
}

// A reply's first step as it opens the reply's text (the fixes, then with tools the decision), and the rest: what is shown
// and spoken of the reply is the rest.
const STEP = /^<fixes>[\s\S]*?<\/fixes>\s*(?:<tool>[\s\S]*?<\/tool>\s*)?/;
export function stepOf(text: string): { step: string; rest: string } {
  const step = text.match(STEP)?.[0] ?? (/^<fixes>/.test(text) && !text.includes("</fixes>") ? text : "");
  return { step, rest: text.slice(step.length) };
}
