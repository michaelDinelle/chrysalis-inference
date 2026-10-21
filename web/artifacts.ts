// Artifacts: what a reply makes (a picture, a page, a diagram, a document, code) in a fenced block whose info string
// names its kind and an id (```svg panda```), and changes in place later with an edit block (```edit panda```) that
// holds only what changes, so a change costs its own tokens, not the artifact's. It is plain text in the reply, so any
// backend's model can write it, and an edit applies piece by piece as it streams. Measured with Qwen3.8-Flash-Next
// (2026-10-20): a panda drawn in 1,987 tokens, its coat made red in 325, a copy under a starry sky in 956, a cat's eyes
// made green beside another picture in 272; every edit applied.
//
// An edit is any of:
// - a whole element with an id, which replaces the element with that id, or is added if there is none: on top (last in
//   the root), or, if it covers the whole picture (a new background: on top it hid the panda it was put behind), in
//   place of the rect that covers it now, else at the back: <g id="burger">…</g>
// - a search and replace, SEARCH copied from the artifact as it is (matched exactly, else line by line without the
//   lines' indentation):
//   <<<<<<< SEARCH / … / ======= / … / >>>>>>> REPLACE
// - an element without an id, which replaces the one of its kind whose geometry it repeats: models write the shapes
//   they recolor without the group that holds their id (a panda's coat: <ellipse cx="200" cy="300" … fill="#d32f2f"/>).
// ```edit copy from original``` makes the copy a new artifact from the original's latest version.

export const ARTIFACTS_PROMPT = `You can make artifacts: pictures (SVG), web pages (HTML), diagrams (Mermaid), documents (LaTeX, Markdown) and code. Write each one in a fenced block whose info string is its kind and then a short id you choose, like \`\`\`svg panda\`\`\`. Give every part its own id, the background too: <rect id="background">, <g id="burger">, <section id="intro">.

To change an artifact that already exists, never write it again: write an edit block for it, \`\`\`edit panda\`\`\`, with only what changes:
- to change a part, write its whole element, with its id, as it should now be; it replaces the element with that id (a new id adds the element on top):
<g id="burger">...the new burger...</g>
- for a small change, a search and replace, where SEARCH is text copied exactly from the artifact as it is now:
<<<<<<< SEARCH
fill="#ffffff"
=======
fill="#d32f2f"
>>>>>>> REPLACE
One edit block can hold several of these. To keep the original and change a copy, give the copy its own id and name the original: \`\`\`edit panda-red from panda\`\`\`.

The user can change an artifact themselves (paint it, move, resize or delete its parts); their next message then holds it as it is now, and your edits start from that.`;

export type Info = { kind: string; id: string } | { edit: string; from: string };
// A fence's info string as an artifact's: its kind and id, or an edit's target (and what it copies)
export function infoOf(lang: string | undefined): Info | null {
  const w = (lang ?? "").trim().split(/\s+/);
  if (w[0] === "edit") return w[1] ? { edit: w[1], from: w[2] === "from" && w[3] ? w[3] : w[1] } : null;
  return w.length === 2 && /^[a-z][\w+#.-]*$/i.test(w[0]) && /^[\w.-]+$/.test(w[1]) ? { kind: w[0].toLowerCase(), id: w[1] } : null;
}

export interface Change { kind: "element" | "search" | "shape"; text: string; search?: string; id?: string; partial?: boolean }
export interface Applied { change: Change; ok: boolean; how: string }

const HUNK = /<{7} SEARCH\n([\s\S]*?)\n?={7}\n([\s\S]*?)\n?>{7} REPLACE/g;
const OPEN = /<([A-Za-z][\w:.-]*)\b((?:[^>"']|"[^"]*"|'[^']*')*?)(\/?)>/g;
const idOf = (attrs: string) => attrs.match(/(?<![\w:-])id\s*=\s*["']([^"']+)["']/)?.[1];
// Kinds whose parts are elements: an edit to any other (code, LaTeX, Mermaid) is searches and replaces alone
const MARKUP = new Set(["svg", "html", "htm", "xml", "xhtml"]);
export const isMarkup = (kind: string) => MARKUP.has(kind);

// From the open tag at `at` in s: the end of its element (balanced by its own tag name), or -1 if it has not closed.
function elementEnd(s: string, at: number): number {
  OPEN.lastIndex = at;
  const m = OPEN.exec(s);
  if (!m || m.index !== at) return -1;
  if (m[3]) return OPEN.lastIndex;
  const name = m[1], tags = new RegExp(`<(/?)${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b((?:[^>"']|"[^"]*"|'[^']*')*?)(/?)>`, "g");
  tags.lastIndex = OPEN.lastIndex;
  for (let depth = 1, t = tags.exec(s); t; t = tags.exec(s)) {
    if (t[3]) continue;
    depth += t[1] ? -1 : 1;
    if (!depth) return tags.lastIndex;
  }
  return -1;
}

// What came of an element still arriving: up to its last complete tag, with what is open closed (an SVG's svgPrefix
// in app.ts is the same for a whole picture).
function closeOpen(s: string): string {
  const stack: string[] = [];
  let cut = 0;
  for (let i = 0; ;) {
    const lt = s.indexOf("<", i);
    if (lt < 0) break;
    if (s.startsWith("<!--", lt)) {
      const e = s.indexOf("-->", lt);
      if (e < 0) break;
      i = cut = e + 3;
      continue;
    }
    OPEN.lastIndex = lt;
    const m = OPEN.exec(s), close = /^<\/([A-Za-z][\w:.-]*)\s*>/.exec(s.slice(lt));
    if (close) {
      const at = stack.lastIndexOf(close[1]);
      if (at >= 0) stack.length = at;
      i = cut = lt + close[0].length;
    } else if (m && m.index === lt) {
      if (!m[3]) stack.push(m[1]);
      i = cut = OPEN.lastIndex;
    } else break;
    if (!stack.length) break;
  }
  return s.slice(0, cut) + stack.reverse().map((t) => `</${t}>`).join("");
}

// An edit block's changes, in order. live: the block is still arriving, so a search and replace counts once it is whole
// and the last element as far as it has come (partial), once its open tag (and so its id) is in. markup: elements count
// too.
export function changesOf(body: string, live = false, markup = true): Change[] {
  const out: Change[] = [];
  let rest = "", last = 0;
  for (const m of body.matchAll(HUNK)) {
    rest += body.slice(last, m.index);
    out.push({ kind: "search", search: m[1], text: m[2] });
    last = m.index + m[0].length;
  }
  const tail = body.slice(last), open = tail.indexOf("<<<<<<< SEARCH");
  rest += open >= 0 ? tail.slice(0, open) : tail;                     // a search and replace still arriving waits
  for (let i = 0; markup;) {
    OPEN.lastIndex = i;
    const m = OPEN.exec(rest);
    if (!m) break;
    const end = elementEnd(rest, m.index), id = idOf(m[2]);
    if (end < 0) {
      if (live && id) out.push({ kind: "element", id, text: closeOpen(rest.slice(m.index)), partial: true });
      break;
    }
    out.push({ kind: id ? "element" : "shape", id, text: rest.slice(m.index, end) });
    i = end;
  }
  return out;
}

// The span of the element with this id in src: [start, end), or null.
function byId(src: string, id: string): [number, number] | null {
  const re = new RegExp(`<[A-Za-z][\\w:.-]*\\b(?:[^>"']|"[^"]*"|'[^']*')*?(?<![\\w:-])id\\s*=\\s*["']${id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}["']`, "g");
  const m = re.exec(src);
  if (!m) return null;
  const end = elementEnd(src, m.index);
  return end < 0 ? null : [m.index, end];
}

// Whether an element's first rect covers the whole picture (its viewBox, or 100%): a background
function coversAll(text: string, src: string): boolean {
  const vb = src.match(/<svg\b[^>]*\bviewBox\s*=\s*["']\s*([-\d.]+)[\s,]+([-\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i), rect = text.match(/<rect\b[^>]*>/i);
  if (!rect) return false;
  const a = attrsOf(rect[0]), at = (k: string, v: string) => (a[k] ?? "0") === v || (a[k] ?? "") === "100%";
  return !!vb && Number(a.x ?? 0) <= Number(vb[1]) && Number(a.y ?? 0) <= Number(vb[2]) && (at("width", vb[3]) || a.width === "100%")
    && (at("height", vb[4]) || a.height === "100%");
}

const GEOMETRY = ["cx", "cy", "r", "rx", "ry", "x", "y", "x1", "y1", "x2", "y2", "width", "height", "d", "points"];
const attrsOf = (tag: string) => Object.fromEntries([...tag.matchAll(/([\w:-]+)\s*=\s*"([^"]*)"/g)].map((a) => [a[1], a[2]]));

// A search's lines found in src without their indentation: the span they take there
function looseFind(src: string, search: string): [number, number] | null {
  const want = search.split("\n").map((l) => l.trim()), lines = src.split("\n");
  while (want.length && !want.at(-1)) want.pop();
  if (!want.length) return null;
  let at = 0;
  for (let i = 0; i + want.length <= lines.length; at += lines[i].length + 1, i++) {
    if (want.every((w, k) => lines[i + k].trim() === w)) {
      const len = lines.slice(i, i + want.length).join("\n").length;
      return [at, at + len];
    }
  }
  return null;
}

// Each change in turn on src: what it became, and what came of each change.
export function applyChanges(src: string, changes: Change[]): { src: string; applied: Applied[] } {
  const applied: Applied[] = [];
  for (const c of changes) {
    if (c.kind === "search") {
      const exact = src.indexOf(c.search!), span = exact >= 0 ? [exact, exact + c.search!.length] as const : looseFind(src, c.search!);
      if (span) src = src.slice(0, span[0]) + c.text + src.slice(span[1]);
      applied.push({ change: c, ok: !!span, how: span ? (exact >= 0 ? "replaced" : "replaced, its indentation aside") : "its SEARCH text is not in it" });
    } else if (c.kind === "element") {
      const span = byId(src, c.id!), bg = !span && coversAll(c.text, src);
      const old = bg ? [...src.matchAll(/<rect\b(?:[^>"']|"[^"]*"|'[^']*')*>/gi)].find((m) => coversAll(m[0], src)) : undefined;
      const back = bg && !old ? src.match(/<svg\b(?:[^>"']|"[^"]*"|'[^']*')*>/i) : null;
      const root = span ? -1 : old ? -1 : back ? back.index! + back[0].length : src.search(/<\/(svg|body|html)\s*>\s*$/i);
      if (span) src = src.slice(0, span[0]) + c.text + src.slice(span[1]);
      else if (old) src = src.slice(0, old.index) + c.text + src.slice(elementEnd(src, old.index!));
      else if (root >= 0) src = src.slice(0, root) + (back ? "\n" : "") + c.text + (back ? "" : "\n") + src.slice(root);
      applied.push({ change: c, ok: !!span || !!old || root >= 0, how: span ? `#${c.id} replaced` : old ? `#${c.id} in place of the background`
        : root >= 0 ? `#${c.id} added ${back ? "at the back" : "on top"}` : `no #${c.id} in it, and no root to add it to` });
    } else {
      const tag = c.text.match(/^<([A-Za-z][\w:.-]*)\b[^>]*>/), want = tag ? attrsOf(tag[0]) : {};
      const key = GEOMETRY.filter((k) => k in want);
      const hits = tag && key.length ? [...src.matchAll(new RegExp(`<${tag[1]}\\b(?:[^>"']|"[^"]*"|'[^']*')*?/?>`, "g"))]
        .filter((m) => { const a = attrsOf(m[0]); return key.every((k) => a[k] === want[k]); }) : [];
      const end = hits.length === 1 ? elementEnd(src, hits[0].index!) : -1;
      if (end >= 0) src = src.slice(0, hits[0].index) + c.text + src.slice(end);
      applied.push({ change: c, ok: end >= 0, how: end >= 0 ? `<${tag![1]}> at the same place replaced`
        : hits.length > 1 ? `${hits.length} <${tag?.[1]}> share its geometry` : `no <${tag?.[1] ?? "?"}> with its geometry` });
    }
  }
  return { src, applied };
}

// The element with this id in src, as written there (to quote what the user points at)
export function elementById(src: string, id: string): string | null {
  const span = byId(src, id);
  return span ? src.slice(span[0], span[1]) : null;
}

// One artifact block of a chat, in order: where it is (whose message, which of its artifact blocks), its info, its text,
// and whether it is still arriving.
// by: "user", a version the user made by hand (its text the artifact whole, as they left it)
export interface Block { at: unknown; n: number; info: Info; text: string; live: boolean; label: string; by?: "user" }
export interface Version { id: string; kind: string; src: string; v: number; block: Block; from?: { id: string; v: number }; applied?: Applied[] }

// Every artifact's versions, from a chat's blocks in order. An edit applies to its target's latest version (or makes a
// copy of the one it names); one whose artifact does not exist yet makes nothing.
export function replay(blocks: Block[]): Map<string, Version[]> {
  const all = new Map<string, Version[]>();
  for (const b of blocks) {
    if ("kind" in b.info) {
      const list = all.get(b.info.id) ?? [];
      all.set(b.info.id, [...list, { id: b.info.id, kind: b.info.kind, src: b.text, v: list.length + 1, block: b }]);
      continue;
    }
    const base = all.get(b.info.from)?.at(-1);
    if (!base) continue;
    const changes = changesOf(b.text, b.live, isMarkup(base.kind)), { src, applied } = applyChanges(base.src, changes);
    // an edit that holds no change this can read (written some other way) changes nothing: said, as a change not applied
    if (!changes.length && !b.live && b.text.trim()) applied.push({ change: { kind: "search", text: b.text }, ok: false, how: "nothing in it reads as a change" });
    const list = all.get(b.info.edit) ?? [];
    all.set(b.info.edit, [...list, { id: b.info.edit, kind: base.kind, src, v: list.length + 1, block: b, applied,
      ...(b.info.from !== b.info.edit ? { from: { id: base.id, v: base.v } } : {}) }]);
  }
  return all;
}

// Edits by hand (the canvas's tools): changes to an artifact's source text itself, an element at a time, so the rest
// stays byte for byte as written. An element is found by its place among the elements in document order, as the
// picture drawn from the source has them.
export interface Span { tag: string; start: number; openEnd: number; end: number; parent: number }
const TAG = /<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<(\/?)([A-Za-z][\w:.-]*)\b((?:[^>"']|"[^"]*"|'[^']*')*?)(\/?)>/g;
// Every element of src in document order: its tag, where its open tag starts and ends, where it ends, its parent
export function elementSpans(src: string): Span[] {
  const out: Span[] = [], stack: number[] = [];
  for (const m of src.matchAll(TAG)) {
    if (!m[2]) continue;                                   // a comment or CDATA
    const end = m.index! + m[0].length;
    if (m[1]) {
      const at = stack.findLastIndex((k) => out[k].tag === m[2]);
      if (at >= 0) { out[stack[at]].end = end; stack.length = at; }
      continue;
    }
    out.push({ tag: m[2], start: m.index!, openEnd: end, end, parent: stack.at(-1) ?? -1 });
    if (!m[4]) stack.push(out.length - 1);
  }
  return out;
}

const attrRe = (name: string) => new RegExp(`(\\s${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*=\\s*)("[^"]*"|'[^']*')`);
// An open tag with an attribute set (null: taken off)
export function withAttr(open: string, name: string, value: string | null): string {
  const re = attrRe(name);
  if (value === null) return open.replace(new RegExp(`\\s${name}\\s*=\\s*("[^"]*"|'[^']*')`), "");
  return re.test(open) ? open.replace(re, `$1"${value}"`) : open.replace(/\s*(\/?)>$/, ` ${name}="${value}"$1>`);
}
// An open tag painted: the property set where it is set now (its style, or its attribute), else as an attribute, or in
// its style when a class may set it (a class's rule wins over an attribute, never over a style).
export function withPaint(open: string, prop: string, color: string): string {
  const style = open.match(attrRe("style"));
  const inStyle = style && new RegExp(`(^|;)\\s*${prop}\\s*:`).test(style[2].slice(1, -1));
  if (inStyle) return withAttr(open, "style", style![2].slice(1, -1).replace(new RegExp(`((?:^|;)\\s*${prop}\\s*:)[^;]*`), `$1 ${color}`));
  if (attrRe(prop).test(open) || !attrRe("class").test(open)) return withAttr(open, prop, color);
  return withAttr(open, "style", `${style ? style[2].slice(1, -1).replace(/;?\s*$/, "; ") : ""}${prop}: ${color}`);
}
export const setOpen = (src: string, s: Span, open: string) => src.slice(0, s.start) + open + src.slice(s.openEnd);
// The element taken out, with the line it stood on if it stood alone there
export function withoutElement(src: string, s: Span): string {
  const before = src.slice(0, s.start), after = src.slice(s.end), indent = before.match(/\n[ \t]*$/);
  return indent && /^[ \t]*\n/.test(after) ? before.slice(0, indent.index) + after.replace(/^[ \t]*/, "") : before + after;
}
// A copy of the element after it (a new id if it had one: ids must stay unique), moved a little so it shows
export function withCopy(src: string, s: Span, offset: number): string {
  let copy = src.slice(s.start, s.end);
  const id = copy.match(/^<[^>]*?\bid\s*=\s*["']([^"']+)["']/)?.[1];
  if (id) {
    let n = 2;
    while (new RegExp(`\\bid\\s*=\\s*["']${id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}-${n}["']`).test(src)) n++;
    copy = copy.replace(/^(<[^>]*?\bid\s*=\s*["'])([^"']+)/, `$1$2-${n}`);
  }
  const open = copy.match(/^<(?:[^>"']|"[^"]*"|'[^']*')*>/)![0], t = open.match(attrRe("transform"))?.[2].slice(1, -1);
  copy = withAttr(open, "transform", `translate(${offset} ${offset})${t ? ` ${t}` : ""}`) + copy.slice(open.length);
  const indent = src.slice(0, s.start).match(/\n([ \t]*)$/)?.[1] ?? "";
  return src.slice(0, s.end) + `\n${indent}` + copy + src.slice(s.end);
}
// The element moved one step forward (after the sibling drawn over it, so it is drawn over that) or backward (before the
// one under it). One step: a picture's first sibling is usually its background, and the back is behind it, out of sight.
export function withOrder(src: string, s: Span, spans: Span[], forward: boolean): string {
  const siblings = spans.filter((x) => x.parent === s.parent), k = siblings.indexOf(s), other = siblings[k + (forward ? 1 : -1)];
  if (!other) return src;
  const [a, b] = forward ? [s, other] : [other, s];      // a comes first in the text: swap the two, with what lies between
  return src.slice(0, a.start) + src.slice(b.start, b.end) + src.slice(a.end, b.start) + src.slice(a.start, a.end) + src.slice(b.end);
}
