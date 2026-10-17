// kiln web UI: chat with kiln's local models (Qwen3.8-27B, Qwen3.8-Flash-Next). Replies stream in with their reasoning and are rendered
// as markdown with highlighted code; conversations live in this tab's sessionStorage (a reload keeps them,
// closing the tab ends them). The settings (system prompt, sampling), kept the same way, apply to every conversation.
import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/common";
import { marked, type Links, type Token } from "marked";

type Effort = "xhigh" | "medium" | "low" | "off";
interface Stats { ttft: number; tps: number; tokens: number; prompt: number; cached: number; thought?: number; seed?: number; model?: string }
interface Message { role: "user" | "assistant"; content: string; reasoning?: string; stats?: Stats; error?: string; stopped?: boolean; job?: string }
interface Conversation { id: string; title: string; named?: boolean; updated: number; effort: Effort; messages: Message[] }
type SamplingKey = "temperature" | "top_p" | "top_k" | "min_p" | "presence_penalty" | "max_tokens" | "seed";
type Sampling = Partial<Record<SamplingKey, number>>;
interface Settings { system: string; sampling: Sampling }
interface Status { state: "online" | "loading" | "offline"; build?: string; model?: string; loading?: string; progress?: number; context?: number; kv_tokens?: number; cache?: { free_kv_tokens: number } }
interface View { el: HTMLElement; think: HTMLDetailsElement; summary: HTMLElement; thought: HTMLElement; body: HTMLElement; foot: HTMLElement; shown?: string; blocks?: Blocks }
interface Live { phase: "waiting" | "thinking" | "writing"; since: number; note: string; loading?: boolean }
// A reply being written. The web server owns it (a job, m.job its id): the page only follows it, so opening another
// chat, reloading, sleeping or a dropped connection leaves it running, and the page picks it up again from the last
// event it saw. Only Stop (or deleting the chat) ends it early.
interface Run {
  conv: Conversation; m: Message; live: Live;
  at: number;                                       // events of the job seen so far
  stopping: boolean; gone?: boolean;                // Stop pressed; the chat was deleted
  usage?: any; seed?: number; timing?: { start: number; first?: number; answer?: number; last?: number };
  frame: number; saved: number;
}

// kiln keeps one model on the GPU. The picker by Load / Unload chooses it (switching takes 5-20 s); every chat then
// talks to the model that is loaded.
const MODELS: [string, string, string][] = [
  ["qwen3.8-27b", "Qwen3.8-27B", "Quickest replies · 4 s to load"],
  ["qwen3.8-flash-next", "Qwen3.8-Flash-Next", "Fast at any length · 20 s to load"],
];
const modelName = (id?: string) => MODELS.find(([m]) => m === id)?.[1] ?? id ?? "";
// What kiln uses for a field left blank, with thinking on and off (DEFAULTS in kiln/server.py: Qwen's recommended
// values). Without a seed kiln draws a random one and reports it with the reply.
const SAMPLING_DEFAULTS: Record<"think" | "fast", Sampling> = {
  think: { temperature: 1, top_p: 0.95, top_k: 20, min_p: 0, presence_penalty: 0, max_tokens: 131072 },
  fast: { temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0, presence_penalty: 1.5, max_tokens: 131072 },
};
const EFFORTS: [Effort, string][] = [["xhigh", "Max"], ["medium", "Medium"], ["low", "Low"], ["off", "Off"]];
const SUGGESTIONS: [string, string][] = [
  ["Rate limiter", "Write a token-bucket rate limiter in Python that works with asyncio, with pytest tests."],
  ["SQLite schema", 'Design a SQLite schema for a todo app with projects, tags and due dates, and the query for a "due today" view.'],
  ["Borrow checker", "Why doesn't this compile, and what is the idiomatic fix?\n\nfn longest(a: &str, b: &str) -> &str {\n    if a.len() > b.len() { a } else { b }\n}"],
  ["Backup script", "Write a bash script that backs up ~/projects to a dated tarball and keeps only the 7 newest."],
];
const ICONS = {
  flame: '<path fill="currentColor" stroke="none" d="M12 1.5c.9 3.6-.9 5.9-2.6 7.9C7.8 11.2 6 13.1 6 15.9 6 19.3 8.7 22 12 22s6-2.6 6-6.2c0-2.7-1.4-4.7-2.6-6.1-.3 1.6-1.2 2.7-2.4 3.2 1-3.9-.2-8.1-1-11.4z"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  send: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" stroke="none"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
  retry: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
  edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  chevron: '<path d="M7 10l5 5 5-5"/>',
};
const TITLES = { copy: "Copy", retry: "Regenerate", edit: "Edit and resend" };
const icon = (name: keyof typeof ICONS) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const thread = $("thread"), scroller = $("scroller"), input = $<HTMLTextAreaElement>("input"), sendBtn = $<HTMLButtonElement>("send");
const settingsDialog = $<HTMLDialogElement>("settings-dialog"), systemInput = $<HTMLTextAreaElement>("system-input");
const samplingInputs = [...$("sampling").querySelectorAll("input")];

function h<K extends keyof HTMLElementTagNameMap>(tag: K, cls = "", text = ""): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

const escapeHtml = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

marked.use({
  gfm: true,
  renderer: {
    code({ text, lang }) {
      const language = (lang ?? "").trim().split(/\s/)[0].toLowerCase();
      const html = hljs.getLanguage(language) ? hljs.highlight(text, { language, ignoreIllegals: true }).value : escapeHtml(text);
      return `<div class="code"><div class="code-head"><span>${escapeHtml(language || "text")}</span>` +
        `<button type="button" class="copy">Copy</button></div><pre><code class="hljs">${html}</code></pre></div>`;
    },
  },
});
const markdown = (src: string) => DOMPurify.sanitize(marked.parse(src, { async: false }));

// ---- replies, block by block --------------------------------------------------------------------------------------
// A reply renders as its blocks (paragraphs, lists, code blocks, tables...). While it streams, only blocks that changed
// are rebuilt: finished ones keep their nodes (no re-highlighting; selections and toggles survive), so the work per
// update is the size of the growing tail, not of the reply. And the tail shows only settled text: whole words, whole
// code lines and table rows, no half-typed markdown, so nothing flashes and then changes shape.
interface Blocks { raws: string[]; nodes: Node[][] }

const FENCE = /^ {0,3}(?:`{3,}|~{3,})/gm;
function settle(src: string): string {
  const nl = src.lastIndexOf("\n"), head = src.slice(0, nl + 1);
  let line = src.slice(nl + 1);
  if ((src.match(FENCE)?.length ?? 0) % 2) return head;                    // in a code block: its whole lines
  if (/^\s*\|/.test(line)) return head;                                   // a table row once it is complete
  if (/^\s*(?:[#>*+-]|\d+[.)])*\s*$/.test(line)) return head;             // a bare "#", "-", "1." ... line
  if (!/\s$/.test(line)) line = line.replace(/[A-Za-z0-9À-ɏ'’_-]{1,24}$/, "");   // a word once it ends
  const text = head + line, p = text.lastIndexOf("\n\n") + 1;
  let para = text.slice(p)                                                // the paragraph being written:
    .replace(/!\[[^\]\n]*\]?(?:\([^)\n]*)?$/, "")                         // no half image,
    .replace(/\[([^\]\n]*)\]\([^)\n]*$/, "$1").replace(/\[([^\]\n]*)$/, "$1")   // a link is its text until it is whole,
    .replace(/<\/?[A-Za-z][^>\n]*$/, "")                                  // no half HTML tag,
    .replace(/(?:\*{1,3}|_{1,3}|~{1,2}|`+)$/, "");                        // markers with nothing after them wait;
  if ((para.replace(/```[\s\S]*?```/g, "").match(/`/g)?.length ?? 0) % 2) para += "`";   // open code and bold close
  if ((para.match(/\*\*/g)?.length ?? 0) % 2) para += "**";
  if ((para.match(/~~/g)?.length ?? 0) % 2) para += "~~";
  return text.slice(0, p) + para;
}

function htmlNodes(t: Token, links: Links): Node[] {
  const tpl = document.createElement("template");
  tpl.innerHTML = DOMPurify.sanitize(marked.parser(Object.assign([t], { links })));      // (options here would drop marked.use's renderer)
  return [...tpl.content.childNodes];
}

function renderBlocks(v: View, src: string, live: boolean) {
  if (!v.blocks) { v.body.replaceChildren(); v.blocks = { raws: [], nodes: [] }; }
  const b = v.blocks, tokens = marked.lexer(live ? settle(src) : src), blocks = tokens.filter((t) => t.type !== "space");
  let k = 0;
  while (k < blocks.length && k < b.raws.length && b.raws[k] === blocks[k].raw) k++;
  for (; k < blocks.length; k++) {
    const t = blocks[k], old = b.nodes[k] ?? [];
    const now = t.type === "code" && SVG_START.test(t.text) ? [svgFigure(old[0], t.text, live && k === blocks.length - 1)] : htmlNodes(t, tokens.links);
    b.raws[k] = t.raw;
    if (now.length === 1 && old.length === 1 && now[0] === old[0]) continue;   // a figure updated in place
    const at = b.nodes.slice(k).flat()[0] ?? null;
    for (const n of now) v.body.insertBefore(n, at);
    for (const n of old) n.parentNode?.removeChild(n);
    b.nodes[k] = now;
  }
  for (const n of b.nodes.slice(blocks.length).flat()) n.parentNode?.removeChild(n);
  b.nodes.length = b.raws.length = blocks.length;
  // a figure that was drawing at the tail finishes once something follows it (its block no longer changes)
  b.nodes.forEach((n, k) => {
    const f = n[0], t = blocks[k];
    if (f instanceof HTMLElement && f.classList.contains("drawing") && !(live && k === blocks.length - 1) && t.type === "code") svgFigure(f, t.text, false);
  });
}

// ---- SVG figures ----------------------------------------------------------------------------------------------------
// A code block holding an SVG document (models label it svg, xml or html) shows the picture instead, as a figure that
// draws itself while it streams: its size is set once the <svg> tag is in (the viewBox), then each element appears,
// with a short fade, once its tag is complete. The markup is sanitized (no scripts, handlers, foreignObject, external
// references) and lives in a shadow root, so its styles and ids stay inside it. The code is a click away.
const SVG_START = /^\s*(?:<\?xml[^>]*\?>\s*)?(?:<!--[\s\S]*?-->\s*|<!DOCTYPE[^>]*>\s*)*<svg[\s>]/i;
const SVG_SHADOW = ":host { display: block; color: #1f1b19; } svg { display: block; max-width: 100%; height: auto; } " +
  "svg[data-fit=fluid] { width: 640px; } :host(.zoom) svg { max-width: none; }";
const svgSource = new WeakMap<Element, string>();
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

// The document as a standalone file: from the root element on, with the namespaces models often leave out.
function svgFile(text: string) {
  let src = text.trim().replace(/^[\s\S]*?(?=<svg[\s>])/i, "");
  const open = src.match(/^<svg[^>]*>/i)?.[0] ?? "<svg>";
  let root = open;
  if (!/\sxmlns\s*=/.test(root)) root = root.replace(/^<svg/i, '<svg xmlns="http://www.w3.org/2000/svg"');
  if (/\bxlink:/.test(src) && !/\sxmlns:xlink\s*=/.test(root)) root = root.replace(/^<svg/i, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
  return root + src.slice(open.length);
}

// The part of an SVG that has arrived, as a document: up to the last complete tag (text waits for its closing tag),
// with the elements still open closed. Also mends a finished one that forgot to close something.
function svgPrefix(text: string): string | null {
  const start = text.search(/<svg[\s>]/i);
  if (start < 0) return null;
  const s = text.slice(start), stack: string[] = [];
  let i = 0, cut = 0;
  while (i < s.length) {
    const lt = s.indexOf("<", i);
    if (lt < 0) break;
    if (s.startsWith("<!--", lt) || s.startsWith("<![CDATA[", lt)) {
      const e = s.indexOf(s[lt + 2] === "-" ? "-->" : "]]>", lt);
      if (e < 0) break;
      i = cut = e + 3;
      continue;
    }
    let j = lt + 1, q = "";
    for (; j < s.length; j++) {
      const c = s[j];
      if (q) { if (c === q) q = ""; } else if (c === '"' || c === "'") q = c; else if (c === ">") break;
    }
    if (j >= s.length) break;                                            // a tag still arriving
    const m = s.slice(lt, j + 1).match(/^<(\/?)([A-Za-z][\w:.-]*)[^]*?(\/?)>$/);
    if (m && m[1]) {
      const at = stack.lastIndexOf(m[2]);
      if (at >= 0) stack.length = at;
    } else if (m && !m[3]) stack.push(m[2]);
    i = cut = j + 1;
    if (!stack.length) break;                                            // the root closed: the end
  }
  return cut ? s.slice(0, cut) + stack.reverse().map((t) => `</${t}>`).join("") : null;
}

// Only references inside the picture: a link or image may point at #id or a data: image, CSS at nothing outside.
let svgSanitizing = false;
const scrubCss = (css: string) => css.replace(/@import[^;]*;?/gi, "").replace(/url\(\s*(['"]?)(?!#|data:image\/)[^)]*\)/gi, "none");
DOMPurify.addHook("uponSanitizeAttribute", (_node, data) => {
  if (!svgSanitizing) return;
  if (/^(xlink:)?href$/i.test(data.attrName) && !/^\s*(#|data:image\/)/i.test(data.attrValue)) data.keepAttr = false;
  else if (/url\(|@import/i.test(data.attrValue)) data.attrValue = scrubCss(data.attrValue);
});
DOMPurify.addHook("uponSanitizeElement", (node, data) => {
  if (svgSanitizing && data.tagName === "style") node.textContent = scrubCss(node.textContent ?? "");
});

const sameEl = (a: Node, b: Node) => a.nodeType === b.nodeType && (a.nodeType !== 1 || ((a as Element).tagName === (b as Element).tagName
  && (a as Element).attributes.length === (b as Element).attributes.length
  && [...(b as Element).attributes].every((x) => (a as Element).getAttribute(x.name) === x.value)));
// Bring dst's subtree to src's, touching only what differs; new elements are collected (to fade them in).
function syncSvg(dst: Element, src: Element, added: Element[]) {
  const a = [...dst.childNodes], b = [...src.childNodes];
  b.forEach((y, i) => {
    const x = a[i];
    if (x && sameEl(x, y)) {
      if (y.nodeType === 1) syncSvg(x as Element, y as Element, added);
      else if (x.nodeValue !== y.nodeValue) x.nodeValue = y.nodeValue;
      return;
    }
    if (x) dst.replaceChild(y, x); else dst.appendChild(y);
    if (y.nodeType === 1) added.push(y as Element);
  });
  for (const x of a.slice(b.length)) x.remove();
}

function drawSvg(root: ShadowRoot, markup: string | null, fade: boolean): boolean {
  if (!markup) return !!root.querySelector("svg");
  svgSanitizing = true;
  let frag: DocumentFragment;
  try {
    frag = DOMPurify.sanitize(markup, { USE_PROFILES: { svg: true, svgFilters: true }, FORBID_TAGS: ["foreignObject"], RETURN_DOM_FRAGMENT: true });
  } finally { svgSanitizing = false; }
  const svg = frag.querySelector("svg");
  if (!svg) return false;
  const num = (n: string) => /^\s*[\d.]+\s*(px)?\s*$/.test(svg.getAttribute(n) ?? "") ? parseFloat(svg.getAttribute(n)!) : 0;
  const w = num("width"), hh = num("height");
  if (w && hh && !svg.hasAttribute("viewBox")) svg.setAttribute("viewBox", `0 0 ${w} ${hh}`);   // so it can scale down
  if (!w) svg.setAttribute("data-fit", "fluid");
  const cur = root.querySelector("svg"), added: Element[] = [];
  if (cur && sameEl(cur, svg)) syncSvg(cur, svg, added);
  else {
    cur ? cur.replaceWith(svg) : root.append(svg);
    added.push(...svg.children);
  }
  if (fade && !reducedMotion.matches)
    for (const el of added) if (el instanceof SVGGraphicsElement) el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 260, easing: "ease-out" });
  return true;
}

function svgFigure(old: Node | undefined, text: string, drawing: boolean): HTMLElement {
  let fig = old instanceof HTMLElement && old.classList.contains("svg-fig") ? old : null;
  if (!fig) {
    fig = h("figure", "svg-fig");
    fig.innerHTML = `<div class="svg-paper"></div><figcaption class="svg-bar"><span class="svg-state"></span><span class="svg-acts">` +
      ["code:Code", "copy:Copy", "save:Download", "zoom:Expand"].map((a) => `<button type="button" data-svg="${a.split(":")[0]}">${a.split(":")[1]}</button>`).join("") +
      `</span></figcaption><div class="code svg-code"><div class="code-head"><span>svg</span><button type="button" class="copy">Copy</button></div>` +
      `<pre><code class="hljs"></code></pre></div>`;
    fig.querySelector(".svg-paper")!.attachShadow({ mode: "open" }).innerHTML = `<style>${SVG_SHADOW}</style>`;
  }
  svgSource.set(fig, text);
  fig.classList.toggle("drawing", drawing);
  const t = performance.now();
  if (!drawing || t - Number(fig.dataset.drawn ?? 0) > 90) {             // at most ~11 redraws a second while it streams
    fig.dataset.drawn = String(t);
    const ok = drawSvg(fig.querySelector(".svg-paper")!.shadowRoot!, svgPrefix(text), drawing);
    fig.classList.toggle("failed", !drawing && !ok);
  }
  const svg = fig.querySelector(".svg-paper")!.shadowRoot!.querySelector("svg"), box = svg?.viewBox.baseVal;
  fig.querySelector(".svg-state")!.textContent = drawing ? "Drawing" : fig.classList.contains("failed") ? "Couldn't draw this SVG"
    : box && box.width ? `SVG · ${Math.round(box.width)} × ${Math.round(box.height)}` : "SVG";
  if (fig.classList.contains("show-code") || fig.classList.contains("failed")) svgCode(fig);
  return fig;
}

function svgCode(fig: Element) {
  const code = fig.querySelector(".svg-code code")!, text = svgSource.get(fig) ?? "";
  if (code.textContent !== text) code.innerHTML = hljs.highlight(text, { language: "xml", ignoreIllegals: true }).value;
}

function svgAction(btn: HTMLElement) {
  const fig = btn.closest(".svg-fig")!, text = svgSource.get(fig) ?? "";
  const act = btn.dataset.svg;
  if (act === "code") {
    btn.textContent = fig.classList.toggle("show-code") ? "Picture" : "Code";
    svgCode(fig);
  } else if (act === "copy") copy(text, btn);
  else if (act === "save") {
    const a = h("a");
    a.href = URL.createObjectURL(new Blob([svgFile(text)], { type: "image/svg+xml" }));
    a.download = "image.svg";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  } else if (act === "zoom") {
    const dlg = $<HTMLDialogElement>("svg-zoom"), host = h("div");
    host.className = "zoom";
    host.attachShadow({ mode: "open" }).innerHTML = `<style>${SVG_SHADOW}</style>`;
    drawSvg(host.shadowRoot!, svgPrefix(text), false);
    const svg = host.shadowRoot!.querySelector("svg"), vb = svg?.viewBox.baseVal;
    if (svg && vb && vb.width && vb.height) {                           // as large as the window allows, its shape kept
      const ar = vb.width / vb.height, w = Math.min(innerWidth * 0.94 - 40, (innerHeight * 0.92 - 40) * ar);
      svg.style.width = `${w}px`;
      svg.style.height = `${w / ar}px`;
    }
    dlg.querySelector(".svg-zoom-paper")!.replaceChildren(host);
    dlg.showModal();
  }
}
const fmtK = (n: number) => (n < 1000 ? String(n) : `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}K`);
const fmtS = (s: number) => `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function pref(key: string, fallback: string) {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}
function setPref(key: string, value: string) {
  try { localStorage.setItem(key, value); } catch { /* private window: keep the default */ }
}
function load<T>(key: string, fallback: T): T {
  try { return JSON.parse(sessionStorage.getItem(key) ?? "null") ?? fallback; } catch { return fallback; }
}
function store(key: string, value: unknown) {
  try { sessionStorage.setItem(key, JSON.stringify(value)); } catch { /* full or disabled: kept until a reload */ }
}

// Saved conversations by id, mirrored to sessionStorage. They are copied in and out, so unsaved changes to the
// open one (turns cut by "edit" and never resent) are not kept.
const chats: Record<string, Conversation> = load("chats", {});
let settings: Settings = load("settings", { system: "", sampling: {} });
let conv = blank();
const runs = new Map<string, Run>();               // conversation id -> its reply being written
const running = (c: Conversation = conv) => runs.get(c.id);
let status: Status = { state: "offline" };
let build = "";                                     // the server's code version when this page loaded
let follow = true;                                  // keep the newest text in view unless the user scrolled up
const views = new Map<Message, View>();

// Plain-HTTP pages (the web chat on the LAN) get no crypto.randomUUID; getRandomValues works on every page.
function randomId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}

function blank(): Conversation {
  return { id: randomId(), title: "New chat", updated: Date.now(), effort: pref("effort", "xhigh") as Effort, messages: [] };
}

// ---- rendering ---------------------------------------------------------------------------------------------

function render() {
  views.clear();
  thread.replaceChildren(...(conv.messages.length ? conv.messages.map(messageView) : [emptyState()]));
  $("title").textContent = conv.title;
  document.title = conv.messages.length ? `${conv.title} · kiln` : "kiln";
  renderSettingsButton();
  renderComposer();
  renderList();
}

function messageView(m: Message, i: number): HTMLElement {
  if (m.role === "assistant") {
    const v = assistantView(i);
    views.set(m, v);
    paint(v, m, [...runs.values()].find((r) => r.m === m)?.live);
    return v.el;
  }
  const el = h("article", "msg user"), foot = h("footer", "msg-foot");
  el.dataset.i = String(i);
  foot.innerHTML = actions(i === conv.messages.findLastIndex((x) => x.role === "user") ? ["copy", "edit"] : ["copy"]);
  el.append(h("div", "bubble", m.content), foot);
  return el;
}

function assistantView(i: number): View {
  const el = h("article", "msg assistant"), think = h("details", "thinking"), summary = h("summary"), thought = h("div", "thought");
  const body = h("div", "markdown"), foot = h("footer", "msg-foot");
  el.dataset.i = String(i);
  think.append(summary, thought);
  el.append(think, body, foot);
  return { el, think, summary, thought, body, foot };
}

const actions = (names: (keyof typeof TITLES)[]) =>
  names.map((a) => `<button type="button" class="act" data-act="${a}" title="${TITLES[a]}">${icon(a)}</button>`).join("");

// Fill an assistant message's view; `live` while it streams (cheap to call repeatedly).
function paint(v: View, m: Message, live?: Live) {
  const elapsed = live ? (performance.now() - live.since) / 1000 : 0;
  v.think.hidden = !m.reasoning;
  v.think.classList.toggle("live", live?.phase === "thinking");
  if (m.reasoning) {
    v.summary.textContent = live?.phase === "thinking" ? `Thinking · ${fmtS(elapsed)}`
      : m.stats?.thought ? `Thought for ${fmtS(m.stats.thought)}` : "Thoughts";
    if (v.thought.textContent !== m.reasoning) {
      v.thought.textContent = m.reasoning;
      if (live) v.thought.scrollTop = v.thought.scrollHeight;
    }
  }
  if (live?.phase === "waiting") {
    // while a model loads the sidebar's bar says so; here only the dot
    v.body.innerHTML = `<p class="waiting">${live.loading ? "" : `${live.note} · ${fmtS(elapsed)}`}</p>`;
    v.shown = v.blocks = undefined;
  } else if (v.shown !== m.content) {
    renderBlocks(v, m.content, !!live);
    v.shown = m.content;
  }
  if (live) return;
  if (m.error) v.body.append(h("div", "error", m.error));
  const s = m.stats;
  const facts = s ? [...(s.model ? [modelName(s.model)] : []), `${fmtS(s.ttft)} to first token`, `${Math.round(s.tps)} tok/s`, `${fmtK(s.tokens)} tokens`, `prompt ${fmtK(s.cached)}/${fmtK(s.prompt)} cached`]
    : m.stopped ? ["stopped"] : [];
  if (s?.seed !== undefined) facts.push(`seed ${s.seed}`);
  const last = conv.messages.at(-1) === m;
  v.foot.innerHTML = `<span class="stats">${facts.join(" · ")}</span>` + actions(last ? ["copy", "retry"] : ["copy"]);
}

function emptyState() {
  const e = h("div", "empty"), grid = h("div", "suggestions");
  e.innerHTML = `<svg class="flame" viewBox="0 0 24 24">${ICONS.flame}</svg><h2>What are we building?</h2>` +
    `<p>${modelName(current())} on your Radeon · 262K context · nothing leaves this machine</p>`;
  for (const [title, prompt] of SUGGESTIONS) {
    const b = h("button", "suggestion");
    b.type = "button";
    b.append(h("strong", "", title), h("span", "", prompt));
    b.onclick = () => submit(prompt);
    grid.append(b);
  }
  e.append(grid);
  return e;
}

function renderComposer() {
  const busy = !!running();
  sendBtn.innerHTML = icon(busy ? "stop" : "send");
  sendBtn.setAttribute("aria-label", busy ? "Stop" : "Send");
  sendBtn.classList.toggle("stop", busy);
  $("effort").replaceChildren(h("span", "label", "Think"), ...EFFORTS.map(([value, label]) => {
    const b = h("button", "", label);
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(conv.effort === value));
    b.onclick = () => { conv.effort = value; setPref("effort", value); renderComposer(); };
    return b;
  }));
  const s = conv.messages.findLast((m) => m.stats)?.stats;
  $("meter").textContent = s ? `${fmtK(s.prompt + s.tokens)} / ${fmtK(status.context ?? 262144)} tokens` : "";
}

// Highlighted while anything differs from the defaults.
function renderSettingsButton() {
  $("settings-btn").classList.toggle("on", !!settings.system.trim() || Object.keys(settings.sampling).length > 0);
}

function renderList() {
  const nav = $("conversations"), today = new Date().setHours(0, 0, 0, 0);
  const group = (t: number) => (t >= today ? "Today" : t >= today - 6 * 864e5 ? "Previous 7 days" : "Older");
  let current = "";
  nav.replaceChildren();
  for (const s of Object.values(chats).sort((a, b) => b.updated - a.updated)) {
    if (group(s.updated) !== current) nav.append(h("div", "group", (current = group(s.updated))));
    const a = h("a", `conv${s.id === conv.id ? " active" : ""}${runs.has(s.id) ? " running" : ""}`), del = h("button", "conv-del", "×");
    a.href = `#${s.id}`;
    del.type = "button";
    del.title = "Delete";
    del.onclick = (e) => {
      e.preventDefault();
      if (del.classList.contains("armed")) return void remove(s.id);
      del.classList.add("armed");
      del.textContent = "Delete";
      setTimeout(() => { del.classList.remove("armed"); del.textContent = "×"; }, 2500);
    };
    a.append(h("span", "conv-title", s.title), del);
    nav.append(a);
  }
}

// The model picker: what kiln has loaded (or is loading); while kiln is stopped, what Load will start.
let picked = pref("model", MODELS[0][0]);
const current = () => status.loading ?? (status.state === "online" ? status.model : undefined) ?? picked;

async function start() {
  await fetch("/api/kiln/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: picked }) });
}

async function pickModel(model: string) {
  picked = model;
  setPref("model", model);
  if (status.state !== "online" || status.model === model) return renderStatus();
  status = { ...status, loading: model, progress: undefined };   // shown at once; the page polls while kiln switches
  renderStatus();
  if (!conv.messages.length) render();
  try {
    const r = await fetch("/api/kiln/load", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model }) });
    if (!r.ok) console.warn("model switch failed:", await r.text());
  } finally {
    await refreshStatus();
  }
}

// The model menu opens above the status line: the models with a line each, a check on the one in use.
let menuOpen = false;
function setMenu(open: boolean) {
  menuOpen = open;
  const menu = $("model-menu"), btn = document.querySelector<HTMLElement>(".model-btn");
  btn?.setAttribute("aria-expanded", String(open));
  btn?.classList.toggle("open", open);
  if (!open) return void (menu.hidden = true);
  menu.replaceChildren(...MODELS.map(([id, label, about]) => {
    const item = h("button", id === current() ? "model-item current" : "model-item");
    item.type = "button";
    item.setAttribute("role", "menuitemradio");
    item.setAttribute("aria-checked", String(id === current()));
    const text = h("span", "model-text");
    text.append(h("span", "model-label", label), h("span", "model-about", about));
    const mark = h("span", "model-check");
    if (id === current()) mark.innerHTML = icon("check");
    item.append(text, mark);
    item.onclick = () => { setMenu(false); if (id !== current()) void pickModel(id); };
    return item;
  }));
  const sidebar = $("sidebar").getBoundingClientRect(), statusTop = $("status").getBoundingClientRect().top;
  menu.style.bottom = `${sidebar.bottom - statusTop + 6}px`;
  menu.hidden = false;
  (menu.querySelector(".current") as HTMLElement | null)?.focus();
}
document.addEventListener("click", (e) => {
  if (menuOpen && !(e.target as HTMLElement).closest("#model-menu, .model-btn")) setMenu(false);
});

let shownStatus = "";
function renderStatus() {
  const el = $("status");
  const loading = status.state === "loading" || !!status.loading;
  const pct = status.progress == null ? null : Math.round(status.progress * 100);
  // While a model loads only the bar moves: update it in place so its width animates between polls.
  if (loading && el.dataset.view === `loading:${current()}`) {
    const fill = el.querySelector<HTMLElement>(".bar.load > i"), num = el.querySelector(".pct");
    if (pct != null && fill && num) {
      el.querySelector(".bar.load")!.classList.remove("indeterminate");
      fill.style.width = `${pct}%`;
      num.textContent = `${pct}%`;
    }
    return;
  }
  // Polled every 4 s: redraw only what changed, so an idle page does not repaint.
  const key = JSON.stringify([status.state, status.model, status.loading, picked, status.context, status.kv_tokens, status.cache?.free_kv_tokens]);
  if (key === shownStatus) return;
  shownStatus = key;
  const line = h("div", "status-line");
  line.append(h("span", `dot ${loading ? "loading" : status.state}`));
  el.replaceChildren(line);
  el.dataset.view = loading ? `loading:${current()}` : status.state;
  if (loading) {
    if (menuOpen) setMenu(false);
    line.append(h("span", "loading-name", `Loading ${modelName(current())}`), h("span", "pct", pct == null ? "" : `${pct}%`));
    const bar = h("div", pct == null ? "bar load indeterminate" : "bar load"), fill = h("i");
    fill.style.width = `${pct ?? 0}%`;
    bar.append(fill);
    el.append(bar);
    return;
  }
  const button = (label: string, action: () => Promise<void>) => {
    const b = h("button", "link-btn", label);
    b.type = "button";
    b.onclick = async () => { await action(); await refreshStatus(); };
    return b;
  };
  const pick = h("button", "model-btn");
  pick.type = "button";
  pick.title = "Choose the model";
  pick.setAttribute("aria-haspopup", "menu");
  pick.setAttribute("aria-expanded", String(menuOpen));
  pick.append(h("span", "", modelName(current())));
  pick.insertAdjacentHTML("beforeend", icon("chevron"));
  pick.onclick = () => setMenu(!menuOpen);
  line.append(pick);
  if (status.state === "online") {
    const used = status.kv_tokens && status.cache ? 1 - status.cache.free_kv_tokens / status.kv_tokens : 0;
    const bar = h("div", "bar"), fill = h("i");
    fill.style.width = `${Math.round(used * 100)}%`;
    bar.append(fill);
    bar.title = `${fmtK(status.cache?.free_kv_tokens ?? 0)} of ${fmtK(status.kv_tokens ?? 0)} KV tokens free (cached conversations are dropped as needed)`;
    line.append(button("Unload", async () => { await fetch("/api/kiln/stop", { method: "POST" }); }));
    const kv = h("div", "status-line kv");
    kv.append(h("span", "", `${fmtK(status.context ?? 0)} context · cache ${Math.round(used * 100)}%`), bar);
    el.append(kv);
  } else {
    line.append(button("Load", start));
  }
  if (menuOpen) setMenu(true);                    // the line moved: keep the menu above it
}

// ---- talking to kiln ---------------------------------------------------------------------------------------

async function refreshStatus() {
  const was = current();
  try { status = await (await fetch("/api/status")).json(); } catch { status = { state: "offline" }; }
  build ||= status.build ?? "";
  if (status.state === "online" && status.model && !status.loading && status.model !== picked) {
    picked = status.model;                         // the picker follows kiln (another client may have switched it)
    setPref("model", picked);
  }
  if (current() !== was && !conv.messages.length) render();
  // The server has newer code: reload into it once nothing would be lost (chats are kept in sessionStorage).
  if (status.build && status.build !== build && !runs.size && !input.value && !settingsDialog.open) return location.reload();
  renderStatus();
}

function save(c: Conversation) {
  if (!c.messages.length) return;
  chats[c.id] = structuredClone(c);
  store("chats", chats);
  if (c === conv && location.hash !== `#${c.id}`) history.replaceState(null, "", `#${c.id}`);
  renderList();
}

// The request body. Assistant turns go back exactly as kiln sent them: kiln then replays its own tokens and
// every turn is a cache hit (only the new message is read).
function request(c: Conversation) {
  const messages: { role: string; content: string; reasoning_content?: string }[] = c.messages.slice(0, -1)
    .filter((m) => m.role === "user" || m.content || m.reasoning)
    .map((m) => (m.reasoning ? { role: m.role, content: m.content, reasoning_content: m.reasoning } : { role: m.role, content: m.content }));
  if (settings.system.trim()) messages.unshift({ role: "system", content: settings.system.trim() });
  return {
    model: c.effort === "off" ? `${current()}-fast` : current(), reasoning_effort: c.effort === "off" ? undefined : c.effort,
    ...settings.sampling, messages, stream: true, stream_options: { include_usage: true },
  };
}

// Server-sent events of a reply, as parsed JSON chunks, then DONE.
const DONE = Symbol("done");
async function* events(body: ReadableStream<Uint8Array>): AsyncGenerator<any> {
  const reader = body.getReader(), decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    for (let end = buf.indexOf("\n\n"); end >= 0; end = buf.indexOf("\n\n")) {
      const data = buf.slice(0, end).split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n");
      buf = buf.slice(end + 2);
      if (data === "[DONE]") return void (yield DONE);
      if (data) yield JSON.parse(data);
    }
  }
}

async function ensureOnline(run: Run) {
  if (status.state === "offline") await start();
  for (const deadline = Date.now() + 180_000; status.state !== "online" || status.loading; await refreshStatus()) {
    if (run.stopping) throw new DOMException("stopped", "AbortError");
    if (Date.now() > deadline) throw new Error("kiln did not start: see `kiln logs`");
    await sleep(100);
  }
}

const CUT_OFF = "The reply was cut off: the web server restarted while it was being written";

async function generate(c: Conversation = conv) {
  const m: Message = { role: "assistant", content: "", reasoning: "" };
  c.messages.push(m);
  const run: Run = { conv: c, m, live: { phase: "waiting", since: performance.now(), note: "Reading the prompt" }, at: 0, stopping: false, frame: 0, saved: 0 };
  runs.set(c.id, run);
  if (c === conv) {
    follow = true;
    render();
  }
  try {
    if (status.state !== "online" || status.loading) {
      run.live.loading = true;
      await ensureOnline(run);
      run.live.note = "Reading the prompt";
      run.live.loading = false;
      run.live.since = performance.now();
    }
    if (run.stopping) throw new DOMException("stopped", "AbortError");
    const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request(c)) });
    if (!res.ok || !res.body) throw new Error((await res.json().catch(() => null))?.error?.message ?? `HTTP ${res.status}`);
    m.job = res.headers.get("X-Kiln-Job") ?? undefined;
    save(c);                                       // a reload finds the reply and follows it again
    if (run.stopping) stopRun(run, true);
    await stream(run, res.body);
  } catch (e) {
    if (run.stopping) m.stopped = true;
    else m.error = e instanceof Error ? e.message : String(e);
  }
  finish(run);
}

// A reply that was being written when the page (re)loaded: followed again from its first event. The saved text stays
// until the server answers; if it no longer knows the reply, that text stays and says why it ends there.
async function resume(c: Conversation, m: Message) {
  const run: Run = { conv: c, m, live: { phase: "waiting", since: performance.now(), note: "Reading the prompt" }, at: 0, stopping: false, frame: 0, saved: 0 };
  runs.set(c.id, run);
  try {
    const res = await fetch(`/api/chat/${m.job}?from=0`);
    if (res.status === 404) throw new Error(CUT_OFF);
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    m.content = m.reasoning = "";
    await stream(run, res.body);
  } catch (e) {
    if (run.stopping) m.stopped = true;
    else m.error = e instanceof Error ? e.message : String(e);
  }
  finish(run);
}

// Follows a reply to its end. A dropped connection (sleep, network, the server busy) is picked up again from the last
// event seen, after 0.3, 0.6, 1.2 ... up to 8 s; only a server that no longer knows the reply ends it.
async function stream(run: Run, body: ReadableStream<Uint8Array> | null) {
  for (let attempt = 0; ; ) {
    if (body) {
      try {
        for await (const ev of events(body)) {
          run.at++;
          attempt = 0;
          if (ev === DONE) return;
          take(run, ev);
        }
      } catch { /* the connection dropped */ }
    }
    body = null;
    if (run.stopping && run.at === 0 && !run.m.job) throw new DOMException("stopped", "AbortError");
    await sleep(Math.min(8000, 300 * 2 ** attempt++));
    const res = await fetch(`/api/chat/${run.m.job}?from=${run.at}`).catch(() => null);
    if (res?.status === 404) throw new Error(CUT_OFF);
    if (res?.ok) body = res.body;
    else if (run.stopping) throw new DOMException("stopped", "AbortError");   // Stop while the server is unreachable
  }
}

// One event of a reply into its message, and onto the screen on the next frame.
function take(run: Run, ev: any) {
  const { m, live } = run, d = ev.choices?.[0]?.delta ?? {}, now = performance.now(), v = views.get(m);
  if (ev.error) m.error = ev.error.message;
  if (ev.stopped) m.stopped = true;
  if (ev.timing) run.timing = ev.timing;
  if (ev.usage) run.usage = ev.usage;
  if (ev.seed !== undefined) run.seed = ev.seed;
  if (d.reasoning_content) {
    if (live.phase !== "thinking") {
      live.phase = "thinking";
      live.since = now;
      if (v) v.think.open = true;
    }
    m.reasoning += d.reasoning_content;
  }
  if (d.content) {
    if (live.phase !== "writing" && v) v.think.open = false;
    live.phase = "writing";
    m.content += d.content;
  }
  if (now - run.saved > 2000) {                    // a reload shows the text so far even if the server forgets it
    run.saved = now;
    save(run.conv);
  }
  schedule(run);
}

function schedule(run: Run) {
  run.frame ||= requestAnimationFrame(() => {
    run.frame = 0;
    const v = views.get(run.m);
    if (!v) return;                                // its chat is not on screen
    paint(v, run.m, run.live);
    if (follow) scroller.scrollTop = scroller.scrollHeight;
  });
}
setInterval(() => { for (const r of runs.values()) schedule(r); }, 250);   // the elapsed-time labels

function finish(run: Run) {
  const { conv: c, m, usage, timing: t } = run;
  if (usage && t?.first && t.last) {
    m.stats = {
      ttft: (t.first - t.start) / 1000, tps: usage.completion_tokens / Math.max((t.last - t.first) / 1000, 1e-3),
      tokens: usage.completion_tokens, prompt: usage.prompt_tokens, cached: usage.prompt_tokens_details?.cached_tokens ?? 0,
      thought: m.reasoning && t.answer ? (t.answer - t.first) / 1000 : undefined, seed: run.seed, model: current(),
    };
  }
  delete m.job;
  cancelAnimationFrame(run.frame);
  if (runs.get(c.id) === run) runs.delete(c.id);
  if (run.gone) return;
  c.updated = Date.now();
  if (c === conv) render();
  else renderList();
  save(c);
  if (!c.named && !m.error) void name(c);
}

// Stop: kiln stops and the reply ends with what it has. (Without a job yet, generate() sends this once it has one.)
function stopRun(run: Run, now = false) {
  if (run.stopping && !now) return;
  run.stopping = true;
  if (run.m.job) void fetch(`/api/chat/${run.m.job}/stop`, { method: "POST" }).catch(() => {});
}

// Titles come from the -fast model with a title-generator prompt, which kiln runs as a low-priority request.
async function name(c: Conversation) {
  c.named = true;
  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: `${current()}-fast`, max_tokens: 24, messages: [
        { role: "system", content: "You are a title generator. Reply with a title of at most six words for the conversation that starts with the user's message below. Reply with the title only." },
        { role: "user", content: c.messages[0].content.slice(0, 2000) },
      ] }),
    });
    const title: string | undefined = (await res.json()).choices?.[0]?.message?.content?.split("\n")[0]?.replace(/^[\s#*"']+|[\s*"'.]+$/g, "");
    if (title) c.title = title.slice(0, 80);
  } catch { /* keep the first line of the question */ }
  save(c);
  if (c === conv) $("title").textContent = document.title = c.title;
}

// ---- user actions ------------------------------------------------------------------------------------------

function submit(text: string) {
  text = text.trim();
  if (!text || running()) return;
  input.value = "";
  grow();
  if (!conv.messages.length) conv.title = text.split("\n")[0].slice(0, 60);
  conv.messages.push({ role: "user", content: text });
  void generate();
}

function openConversation(id: string) {
  conv = runs.get(id)?.conv ?? (chats[id] ? structuredClone(chats[id]) : blank());   // a reply being written keeps going
  document.body.classList.remove("nav-open");
  render();
  scroller.scrollTop = scroller.scrollHeight;
}

function newChat() {
  conv = blank();
  history.replaceState(null, "", location.pathname);
  document.body.classList.remove("nav-open");
  render();
  input.focus();
}

function remove(id: string) {
  const r = runs.get(id);
  if (r) {
    r.gone = true;
    stopRun(r);
  }
  delete chats[id];
  store("chats", chats);
  if (id === conv.id) newChat();
  else renderList();
}

// The Clipboard API only exists on HTTPS and localhost pages, not on the web chat on the LAN (plain HTTP). The copy
// command works on every page: copy from a hidden textarea.
function copy(text: string, button: HTMLElement) {
  const area = h("textarea");
  area.value = text;
  area.readOnly = true;                                  // no on-screen keyboard on phones
  area.style.cssText = "position: fixed; top: 0; opacity: 0; font-size: 16px";   // 16px: iOS does not zoom
  document.body.append(area);
  area.select();
  area.setSelectionRange(0, text.length);                // iOS ignores select()
  document.execCommand("copy");
  area.remove();
  const label = button.textContent;
  button.classList.add("done");
  if (button.classList.contains("copy")) button.textContent = "Copied";
  setTimeout(() => { button.classList.remove("done"); if (button.classList.contains("copy")) button.textContent = label; }, 1200);
}

$("svg-zoom").addEventListener("click", (e) => (e.currentTarget as HTMLDialogElement).close());   // a click anywhere closes it

function grow() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 240)}px`;
}

thread.addEventListener("click", (e) => {
  const t = e.target as HTMLElement, code = t.closest<HTMLElement>(".copy"), act = t.closest<HTMLElement>("[data-act]");
  if (code) return void copy(code.closest(".code")?.querySelector("code")?.textContent ?? "", code);
  const svg = t.closest<HTMLElement>("[data-svg]");
  if (svg) return void svgAction(svg);
  if (!act) return;
  const i = Number(act.closest<HTMLElement>("article")?.dataset.i), m = conv.messages[i];
  if (act.dataset.act === "copy") return void copy(m.content, act);
  if (running()) return;
  if (act.dataset.act === "retry") {
    conv.messages.splice(i);
    void generate();
  } else if (act.dataset.act === "edit") {
    conv.messages.splice(i);
    render();
    input.value = m.content;
    grow();
    input.focus();
  }
});
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const r = running();
  if (r) stopRun(r);
  else submit(input.value);
});
input.addEventListener("input", grow);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    submit(input.value);
  }
});
// Esc stops this chat's reply, unless it is closing something: a menu, a dialog (Settings, a picture at full size)
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape" || e.defaultPrevented) return;
  if (menuOpen) return setMenu(false);
  if (document.querySelector("dialog[open]")) return;
  const r = running();
  if (r) stopRun(r);
});
scroller.addEventListener("scroll", () => { follow = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80; });
window.addEventListener("hashchange", () => {
  const id = location.hash.slice(1);
  if (id && id !== conv.id) openConversation(id);
});
$("new-chat").addEventListener("click", newChat);
$("menu").addEventListener("click", () => document.body.classList.toggle("nav-open"));
$("scrim").addEventListener("click", () => document.body.classList.remove("nav-open"));
$("settings-btn").addEventListener("click", () => {
  const defaults = SAMPLING_DEFAULTS[conv.effort === "off" ? "fast" : "think"];
  systemInput.value = settings.system;
  for (const f of samplingInputs) {
    f.value = String(settings.sampling[f.name as SamplingKey] ?? "");
    f.placeholder = String(defaults[f.name as SamplingKey] ?? "random");
  }
  settingsDialog.returnValue = "";                  // Esc keeps the previous value, which may be "save"
  settingsDialog.showModal();
});
$("settings-reset").addEventListener("click", () => {
  systemInput.value = "";
  for (const f of samplingInputs) f.value = "";
});
settingsDialog.addEventListener("close", () => {
  if (settingsDialog.returnValue !== "save") return;
  const sampling: Sampling = {};
  for (const f of samplingInputs) if (f.value !== "") sampling[f.name as SamplingKey] = f.valueAsNumber;
  settings = { system: systemInput.value, sampling };
  store("settings", settings);
  renderSettingsButton();
});
document.addEventListener("visibilitychange", () => { if (!document.hidden) void refreshStatus(); });

$("brand-mark").innerHTML = `<svg class="flame" viewBox="0 0 24 24">${ICONS.flame}</svg>`;
$("new-chat").innerHTML = `${icon("plus")}New chat`;
$("menu").innerHTML = icon("menu");
await refreshStatus();
for (const c of Object.values(chats)) {
  const m = c.messages.at(-1);
  if (m?.role === "assistant" && m.job) void resume(c, m);
}
if (location.hash.length > 1) openConversation(location.hash.slice(1));
else render();
// Every 4 s, or 4 times a second while a model loads (5-20 s).
(async () => {
  for (;;) {
    await sleep(status.state === "loading" || status.loading ? 250 : 4000);
    if (!document.hidden) await refreshStatus();
  }
})();
