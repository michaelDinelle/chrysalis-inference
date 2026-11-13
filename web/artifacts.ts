import { Tokenizer, TokenizerMode, type Token } from "parse5";

export const ARTIFACTS_PROMPT = `You can make artifacts: pictures (SVG), web pages (HTML), diagrams (Mermaid), documents (LaTeX, Markdown) and code. Write each one in a fenced block whose info string is its kind and then a short id you choose, like \`\`\`svg panda\`\`\`. Give every part its own id, the background too: <rect id="background">, <g id="burger">, <section id="intro">. A page can load three.js or A-Frame from their usual CDN addresses and nothing else; for a 3D scene prefer A-Frame, whose entities take ids like any element: <a-box id="table">.

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
export function infoOf(lang: string | undefined): Info | null {
  const w = (lang ?? "").trim().split(/\s+/).map((x, i) => (i ? x.replace(/^id\s*=\s*(["']?)([\w.-]+)\1$/, "$2") : x));
  if (w[0] === "edit") return w[1] ? { edit: w[1], from: w[2] === "from" && w[3] ? w[3] : w[1] } : null;
  return w.length === 2 && /^[a-z][\w+#.-]*$/i.test(w[0]) && /^[\w.-]+$/.test(w[1]) ? { kind: w[0].toLowerCase(), id: w[1] } : null;
}
export const PAGE_DOC = /^\s*(?:<!--[\s\S]*?-->\s*)*(?:<!doctype html|<html[\s>])/i;
export function artifactOf(lang: string | undefined, text: string): { info: Info; text: string } | null {
  const info = infoOf(lang);
  if (info) return { info, text };
  const kind = (lang ?? "").trim().toLowerCase(), line = text.match(/^([A-Za-z][\w.-]*)[ \t]*\n(?=\s*<)/);
  if (!/^[a-z][\w+#.-]*$/.test(kind)) return null;
  if (line) return { info: { kind, id: line[1] }, text: text.slice(line[0].length) };
  return /^html?$/.test(kind) && PAGE_DOC.test(text) ? { info: { kind, id: "page" }, text } : null;
}

export interface Change { kind: "element" | "search" | "shape"; text: string; search?: string; id?: string; partial?: boolean }
export interface Applied { change: Change; ok: boolean; how: string }

const HUNK = /<{7} SEARCH\n([\s\S]*?)\n?={7}\n([\s\S]*?)\n?>{7} REPLACE/g;
const MARKUP = new Set(["svg", "html", "htm", "xml", "xhtml"]);
export const isMarkup = (kind: string, src = "") => MARKUP.has(kind) || /^\s*<[!?A-Za-z]/.test(src);

interface Tag { kind: "open" | "close" | "other"; name: string; start: number; end: number; self: boolean; attrs: Record<string, string>; at: Record<string, [number, number]> }
const RAW: Record<string, (typeof TokenizerMode)[keyof typeof TokenizerMode]> = { script: TokenizerMode.SCRIPT_DATA, style: TokenizerMode.RAWTEXT, textarea: TokenizerMode.RCDATA, title: TokenizerMode.RCDATA };
const VOID = new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"]);
function tagsOf(src: string): Tag[] {
  const out: Tag[] = [];
  const tag = (kind: "open" | "close") => (t: Token.TagToken) => {
    const at = t.location!;
    out.push({ kind, name: t.tagName, start: at.startOffset, end: at.endOffset, self: t.selfClosing || (kind === "open" && VOID.has(t.tagName)),
      attrs: Object.fromEntries(t.attrs.map((a) => [a.name, a.value])), at: Object.fromEntries(Object.entries(at.attrs ?? {}).map(([k, v]) => [k, [v.startOffset, v.endOffset]])) });
    if (kind === "open" && !t.selfClosing && t.tagName in RAW) {
      tokenizer.state = RAW[t.tagName];
      tokenizer.lastStartTagName = t.tagName;
    }
  };
  const other = (t: Token.CommentToken | Token.DoctypeToken) => void out.push({ kind: "other", name: "", start: t.location!.startOffset, end: t.location!.endOffset, self: true, attrs: {}, at: {} });
  const tokenizer = new Tokenizer({ sourceCodeLocationInfo: true }, { onStartTag: tag("open"), onEndTag: tag("close"), onComment: other, onDoctype: other,
    onCharacter() {}, onNullCharacter() {}, onWhitespaceCharacter() {}, onEof() {} });
  tokenizer.write(src, true);
  return out;
}
const written = (src: string, t: { start: number; name: string }) => src.slice(t.start + 1, t.start + 1 + t.name.length);

function closeOpen(s: string): string {
  const stack: Tag[] = [];
  let cut = 0;
  for (const t of tagsOf(s)) {
    if (t.kind === "other") {
      if (!s.slice(t.start, t.end).endsWith(s.startsWith("<!--", t.start) ? "-->" : ">")) break;
      cut = t.end;
      continue;
    }
    if (t.kind === "close") {
      const at = stack.findLastIndex((x) => x.name === t.name);
      if (at >= 0) stack.length = at;
    } else if (!t.self) stack.push(t);
    cut = t.end;
    if (!stack.length) break;
  }
  return s.slice(0, cut) + stack.reverse().map((t) => `</${written(s, t)}>`).join("");
}

export function changesOf(body: string, live = false, markup = true): Change[] {
  const out: Change[] = [];
  let rest = "", last = 0;
  for (const m of body.matchAll(HUNK)) {
    rest += body.slice(last, m.index);
    out.push({ kind: "search", search: m[1], text: m[2] });
    last = m.index + m[0].length;
  }
  const tail = body.slice(last), open = tail.indexOf("<<<<<<< SEARCH");
  rest += open >= 0 ? tail.slice(0, open) : tail;
  for (const x of markup ? elementSpans(rest).filter((x) => x.parent < 0) : []) {
    const id = x.attrs.id;
    if (!x.closed) {
      if (live && id) out.push({ kind: "element", id, text: closeOpen(rest.slice(x.start)), partial: true });
      break;
    }
    out.push({ kind: id ? "element" : "shape", id, text: rest.slice(x.start, x.end) });
  }
  return out;
}

const byId = (spans: Span[], id: string) => { const x = spans.find((x) => x.attrs.id === id); return x?.closed ? x : null; };

const GEOMETRY = ["cx", "cy", "r", "rx", "ry", "x", "y", "x1", "y1", "x2", "y2", "width", "height", "d", "points"];
const viewBox = (spans: Span[]) => spans.find((x) => x.name === "svg")?.attrs.viewbox?.match(/^\s*([-\d.]+)[\s,]+([-\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/);
function coversAll(a: Record<string, string> | undefined, vb: RegExpMatchArray | null | undefined): boolean {
  if (!a || !vb) return false;
  const at = (k: string, v: string) => (a[k] ?? "0") === v || (a[k] ?? "") === "100%";
  return Number(a.x ?? 0) <= Number(vb[1]) && Number(a.y ?? 0) <= Number(vb[2]) && (at("width", vb[3]) || a.width === "100%") && (at("height", vb[4]) || a.height === "100%");
}
function rootEnd(src: string): number {
  const last = tagsOf(src).at(-1);
  return last?.kind === "close" && /^(svg|body|html)$/.test(last.name) && !src.slice(last.end).trim() ? last.start : -1;
}

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

export function applyChanges(src: string, changes: Change[]): { src: string; applied: Applied[] } {
  const applied: Applied[] = [];
  for (const c of changes) {
    if (c.kind === "search") {
      const exact = src.indexOf(c.search!), span = exact >= 0 ? [exact, exact + c.search!.length] as const : looseFind(src, c.search!);
      if (span) src = src.slice(0, span[0]) + c.text + src.slice(span[1]);
      applied.push({ change: c, ok: !!span, how: span ? (exact >= 0 ? "replaced" : "replaced, its indentation aside") : "its SEARCH text is not in it" });
    } else if (c.kind === "element") {
      const spans = elementSpans(src), span = byId(spans, c.id!), vb = viewBox(spans);
      const bg = !span && coversAll(elementSpans(c.text).find((x) => x.name === "rect")?.attrs, vb);
      const old = bg ? spans.find((x) => x.name === "rect" && coversAll(x.attrs, vb)) : undefined;
      const back = bg && !old ? spans.find((x) => x.name === "svg") : undefined;
      const root = span || old ? -1 : back ? back.openEnd : rootEnd(src);
      if (span) src = src.slice(0, span.start) + c.text + src.slice(span.end);
      else if (old) src = src.slice(0, old.start) + c.text + src.slice(old.end);
      else if (root >= 0) src = src.slice(0, root) + (back ? "\n" : "") + c.text + (back ? "" : "\n") + src.slice(root);
      applied.push({ change: c, ok: !!span || !!old || root >= 0, how: span ? `#${c.id} replaced` : old ? `#${c.id} in place of the background`
        : root >= 0 ? `#${c.id} added ${back ? "at the back" : "on top"}` : `no #${c.id} in it, and no root to add it to` });
    } else {
      const first = elementSpans(c.text)[0], want = first?.attrs ?? {}, key = GEOMETRY.filter((k) => k in want);
      const hits = first && key.length ? elementSpans(src).filter((x) => x.name === first.name && key.every((k) => x.attrs[k] === want[k])) : [];
      const hit = hits.length === 1 && hits[0].closed ? hits[0] : undefined;
      if (hit) src = src.slice(0, hit.start) + c.text + src.slice(hit.end);
      applied.push({ change: c, ok: !!hit, how: hit ? `<${first.tag}> at the same place replaced`
        : hits.length > 1 ? `${hits.length} <${first.tag}> share its geometry` : `no <${first?.tag ?? "?"}> with its geometry` });
    }
  }
  return { src, applied };
}

export function elementById(src: string, id: string): string | null {
  const span = byId(elementSpans(src), id);
  return span ? src.slice(span.start, span.end) : null;
}

export interface Block { at: unknown; n: number; info: Info; text: string; live: boolean; label: string; by?: "user" }
export interface Version { id: string; kind: string; src: string; v: number; block: Block; from?: { id: string; v: number }; applied?: Applied[] }

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
    const changes = changesOf(b.text, b.live, isMarkup(base.kind, base.src)), { src, applied } = applyChanges(base.src, changes);
    if (!changes.length && !b.live && b.text.trim()) applied.push({ change: { kind: "search", text: b.text }, ok: false, how: "nothing in it reads as a change" });
    const list = all.get(b.info.edit) ?? [];
    all.set(b.info.edit, [...list, { id: b.info.edit, kind: base.kind, src, v: list.length + 1, block: b, applied,
      ...(b.info.from !== b.info.edit ? { from: { id: base.id, v: base.v } } : {}) }]);
  }
  return all;
}

export interface Span { tag: string; name: string; attrs: Record<string, string>; start: number; openEnd: number; end: number; parent: number; closed: boolean }
export function elementSpans(src: string): Span[] {
  const out: Span[] = [], stack: number[] = [];
  for (const t of tagsOf(src)) {
    if (t.kind === "other") continue;
    if (t.kind === "close") {
      const at = stack.findLastIndex((k) => out[k].name === t.name);
      if (at >= 0) {
        Object.assign(out[stack[at]], { end: t.end, closed: true });
        stack.length = at;
      }
      continue;
    }
    out.push({ tag: written(src, t), name: t.name, attrs: t.attrs, start: t.start, openEnd: t.end, end: t.end, parent: stack.at(-1) ?? -1, closed: t.self });
    if (!t.self) stack.push(out.length - 1);
  }
  const open = out[stack.at(-1) ?? -1];
  if (open && open.name in RAW) open.end = src.length;
  return out;
}
export const contentEnd = (src: string, s: Span) => (s.end > s.openEnd ? src.lastIndexOf("<", s.end - 1) : s.openEnd);
export function markSource(src: string): string {
  const spans = elementSpans(src);
  let out = "", at = 0;
  spans.forEach((s, i) => {
    const close = src.slice(s.start, s.openEnd).endsWith("/>") ? s.openEnd - 2 : s.openEnd - 1;
    out += src.slice(at, close) + ` data-cx="${i}"`;
    at = close;
  });
  return out + src.slice(at);
}

const quoted = (value: string) => `"${value.replace(/&/g, "&amp;").replace(/"/g, "&quot;")}"`;
export function withAttr(open: string, name: string, value: string | null): string {
  const t = tagsOf(open)[0], at = t?.at[name.toLowerCase()];
  if (!t) return open;
  if (value === null) return at ? open.slice(0, at[0]).replace(/\s+$/, "") + open.slice(at[1]) : open;
  if (at) return `${open.slice(0, at[0])}${open.slice(at[0], at[0] + name.length)}=${quoted(value)}${open.slice(at[1])}`;
  const close = open.slice(0, t.end).endsWith("/>") ? 2 : 1;
  return `${open.slice(0, t.end - close).replace(/\s+$/, "")} ${name}=${quoted(value)}${open.slice(t.end - close)}`;
}
export function withPaint(open: string, prop: string, color: string): string {
  const a = tagsOf(open)[0]?.attrs ?? {}, style = a.style;
  const inStyle = style !== undefined && new RegExp(`(^|;)\\s*${prop}\\s*:`).test(style);
  if (inStyle) return withAttr(open, "style", style.replace(new RegExp(`((?:^|;)\\s*${prop}\\s*:)[^;]*`), `$1 ${color}`));
  if (prop in a || !("class" in a)) return withAttr(open, prop, color);
  return withAttr(open, "style", `${style !== undefined ? style.replace(/;?\s*$/, "; ") : ""}${prop}: ${color}`);
}
export function withBackground(open: string, color: string): string {
  const style = tagsOf(open)[0]?.attrs.style?.replace(/(?:^|;)\s*background(?:-color|-image)?\s*:[^;]*/g, "").replace(/^\s*;\s*|[;\s]+$/g, "");
  return withAttr(open, "style", `${style ? `${style}; ` : ""}background: ${color}`);
}
export function withContent(src: string, s: Span, content: string): string {
  return src.slice(0, s.openEnd) + content + src.slice(contentEnd(src, s));
}
export const setOpen = (src: string, s: Span, open: string) => src.slice(0, s.start) + open + src.slice(s.openEnd);
export function withoutElement(src: string, s: Span): string {
  const before = src.slice(0, s.start), after = src.slice(s.end), indent = before.match(/\n[ \t]*$/);
  return indent && /^[ \t]*\n/.test(after) ? before.slice(0, indent.index) + after.replace(/^[ \t]*/, "") : before + after;
}
export function withCopy(src: string, s: Span, offset?: number): string {
  let open = src.slice(s.start, s.openEnd);
  if (s.attrs.id) {
    const ids = new Set(elementSpans(src).map((x) => x.attrs.id));
    let n = 2;
    while (ids.has(`${s.attrs.id}-${n}`)) n++;
    open = withAttr(open, "id", `${s.attrs.id}-${n}`);
  }
  if (offset !== undefined) open = withAttr(open, "transform", `translate(${offset} ${offset})${s.attrs.transform ? ` ${s.attrs.transform}` : ""}`);
  const indent = src.slice(0, s.start).match(/\n([ \t]*)$/)?.[1] ?? "";
  return src.slice(0, s.end) + `\n${indent}` + open + src.slice(s.openEnd, s.end) + src.slice(s.end);
}
export function withOrder(src: string, s: Span, spans: Span[], forward: boolean): string {
  const siblings = spans.filter((x) => x.parent === s.parent), k = siblings.indexOf(s), other = siblings[k + (forward ? 1 : -1)];
  if (!other) return src;
  const [a, b] = forward ? [s, other] : [other, s];
  return src.slice(0, a.start) + src.slice(b.start, b.end) + src.slice(a.end, b.start) + src.slice(a.start, a.end) + src.slice(b.end);
}

export type DiffLine = { op: " " | "-" | "+"; text: string };
export function lineDiff(a: string, b: string): DiffLine[] {
  const x = a.split("\n"), y = b.split("\n"), n = x.length, m = y.length, max = n + m, v = new Int32Array(2 * max + 2), trace: Int32Array[] = [];
  let found = n === 0 && m === 0;
  for (let d = 0; d <= max && !found; d++) {
    trace.push(v.slice());
    for (let k = -d; k <= d; k += 2) {
      let i = k === -d || (k !== d && v[max + k - 1] < v[max + k + 1]) ? v[max + k + 1] : v[max + k - 1] + 1, j = i - k;
      while (i < n && j < m && x[i] === y[j]) { i++; j++; }
      v[max + k] = i;
      if (i >= n && j >= m) { found = true; break; }
    }
  }
  trace.push(v.slice());
  const out: DiffLine[] = [];
  let i = n, j = m;
  for (let d = trace.length - 2; d >= 0 && (i > 0 || j > 0); d--) {
    const t = trace[d], k = i - j;
    const down = k === -d || (k !== d && t[max + k - 1] < t[max + k + 1]), pk = down ? k + 1 : k - 1, pi = t[max + pk], pj = pi - pk;
    while (i > pi && j > pj) out.push({ op: " ", text: x[--i] }), j--;
    if (d > 0) { if (down) out.push({ op: "+", text: y[--j] }); else out.push({ op: "-", text: x[--i] }); }
  }
  while (i > 0 && j > 0) out.push({ op: " ", text: x[--i] }), j--;
  return out.reverse();
}
