// kiln web UI: chat with kiln's local models (Qwen3.8-27B, Qwen3.8-Flash-Next). Replies stream in with their reasoning and are rendered
// as markdown with highlighted code; conversations live in this tab's sessionStorage (a reload keeps them,
// closing the tab ends them). The settings (system prompt, sampling, reply length, seed), kept the same way, apply to
// every conversation.
import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/common";
import katex from "katex";
import "katex/dist/katex.min.css";
import { marked, type Links, type Token } from "marked";
import { type Attachment, KINDS, attach, dataUrl, fmtBytes, get as getFile, put as putFile, textBlock } from "./files";
import { type Cand, type Odds, type SamplerKey, type Stage, dryRuns, sample } from "./sampler";

type Effort = "xhigh" | "medium" | "low" | "off";
interface Stats { ttft: number; tps: number; tokens: number; prompt: number; cached: number; thought?: number; seed?: number; model?: string }
interface Message {
  id?: string; role: "user" | "assistant"; content: string; reasoning?: string; stats?: Stats; error?: string; stopped?: boolean; job?: string;
  params?: string;                                  // the sampling values that differed from the recommended ones
  effort?: Effort;                                  // the thinking setting it was written with (Continue keeps it)
  finish?: string; edited?: boolean;
  files?: Attachment[];                             // a user message's attached files (their contents: files.ts)
}
interface Conversation { id: string; title: string; named?: boolean; updated: number; effort: Effort; messages: Message[] }
type Sampler = Partial<Record<SamplerKey, number>> & { order?: Stage[] };
type Mode = "think" | "fast";                       // thinking on (Max, Medium, Low) or off: Qwen recommends other values
// Only what differs from the recommended values is kept, so whatever is left alone follows them.
interface Settings { system: string; think: Sampler; fast: Sampler; max_tokens?: number; seed?: number }
interface Status { state: "online" | "loading" | "offline"; build?: string; model?: string; loading?: string; progress?: number; context?: number; kv_tokens?: number; cache?: { free_kv_tokens: number } }
interface View {
  el: HTMLElement; think: HTMLDetailsElement; summary: HTMLElement; thought: HTMLElement; body: HTMLElement; foot: HTMLElement; shown?: string; blocks?: Blocks;
  tok?: { reasoning: number; content: number };     // tokens drawn so far, shown as tokens
}
// A token of a reply, with the model's probability for it and its likeliest alternatives (log probs from kiln).
interface Alt { t: string; p: number; id: number }
interface Tok { t: string; p: number; alts: Alt[] }
interface Live { phase: "waiting" | "thinking" | "writing"; since: number; note: string; loading?: boolean }
// A reply being written. The web server owns it (a job, m.job its id): the page only follows it, so opening another
// chat, reloading, sleeping or a dropped connection leaves it running, and the page picks it up again from the last
// event it saw. Only Stop (or deleting the chat) ends it early.
interface Run {
  conv: Conversation; m: Message; live: Live;
  at: number;                                       // events of the job seen so far
  stopping: boolean; gone?: boolean;                // Stop pressed; the chat was deleted
  usage?: any; seed?: number; timing?: { start: number; first?: number; answer?: number; last?: number };
  calls: string[];                                  // tools the model asked for (this chat offers none)
  pick?: Tok;                                       // the alternative a rewrite starts from
  frame: number; saved: number;
}

// kiln keeps one model on the GPU. The picker by Load / Unload chooses it (switching takes 5-20 s); every chat then
// talks to the model that is loaded.
const MODELS: [string, string, string][] = [
  ["qwen3.8-27b", "Qwen3.8-27B", "Quickest replies · 4 s to load"],
  ["qwen3.8-flash-next", "Qwen3.8-Flash-Next", "Fast at any length · 20 s to load"],
];
const modelName = (id?: string) => MODELS.find(([m]) => m === id)?.[1] ?? id ?? "";
// What kiln uses for a value it is not sent, with thinking on and off (DEFAULTS and MAX_OUTPUT in kiln/server.py:
// Qwen's recommended values). Without a seed kiln draws a random one and reports it with the reply.
const RECOMMENDED: Record<Mode, Record<SamplerKey, number>> = {
  think: { temperature: 1, top_p: 0.95, top_k: 20, min_p: 0, presence_penalty: 0, dry_multiplier: 0, dry_allowed_length: 2, dry_base: 1.75 },
  fast: { temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0, presence_penalty: 1.5, dry_multiplier: 0, dry_allowed_length: 2, dry_base: 1.75 },
};
const MAX_REPLY = 131072;
const STAGES: Stage[] = ["top_k", "temperature", "top_p", "min_p"];   // kiln's default order (engine/sampler.h)
// `slider` narrows the slider to the useful part of the range; typing still reaches all of it.
const SAMPLERS: Record<SamplerKey, { label: string; short: string; min: number; max: number; step: number; slider?: [number, number]; help: string }> = {
  temperature: { label: "Temperature", short: "temp", min: 0, max: 2, step: 0.01, help: "Temperature: under 1 sharpens the odds, over 1 flattens them; 0 always takes the likeliest." },
  top_p: { label: "Top P", short: "top-p", min: 0, max: 1, step: 0.01, help: "Top P: keeps the likeliest tokens until together they reach this share." },
  top_k: { label: "Top K", short: "top-k", min: 1, max: 64, step: 1, help: "Top K: keeps at most this many tokens (kiln looks at the 64 likeliest)." },
  min_p: { label: "Min P", short: "min-p", min: 0, max: 1, step: 0.01, slider: [0, 0.5], help: "Min P: drops tokens less likely than this share of the likeliest." },
  presence_penalty: { label: "Presence", short: "presence", min: -2, max: 2, step: 0.05, slider: [0, 2], help: "Presence penalty: lowers every token the reply has already used." },
  dry_multiplier: { label: "DRY", short: "dry", min: 0, max: 5, step: 0.05, slider: [0, 2], help: "DRY: lowers a token that would repeat a run of text already in the chat. Off at 0." },
  dry_allowed_length: { label: "Allowed run", short: "dry-run", min: 1, max: 20, step: 1, slider: [1, 8], help: "DRY's allowed run: repeats up to this many tokens long go unpenalized." },
  dry_base: { label: "Growth", short: "dry-base", min: 1, max: 4, step: 0.05, slider: [1, 3], help: "DRY's growth: how fast the penalty rises with each token a repeat goes on." },
};
// Real odds for the settings to act on: the first word of Qwen3.8-Flash-Next's answer to "Suggest a name for a cat.
// One word." with thinking off, drawn 2,000 times with the sampler wide open (names drawn twice or more).
const ODDS: [string, number][] = [["Luna", 703], ["Mochi", 470], ["Whiskers", 312], ["Milo", 182], ["Mittens", 78], ["Shadow", 39],
  ["Miso", 31], ["Whisper", 29], ["Moon", 14], ["Pumpkin", 14], ["Pixel", 13], ["Whisker", 11], ["Willow", 9], ["Muffin", 8], ["Echo", 7],
  ["Momo", 6], ["Jasper", 5], ["Nimbus", 5], ["Nebula", 4], ["Simba", 4], ["Noodle", 3], ["Misty", 3], ["Mocha", 3], ["Mimi", 2],
  ["Munchkin", 2], ["Zephyr", 2], ["Moss", 2]];
// A list that has started to repeat itself: the reply so far (token, id) and the next token's 64 candidates (token, log
// prob, id), as Qwen3.8-Flash-Next (thinking off) gave them for "List cat names, separated by commas."
const LOOP: { reply: [string, number][]; cands: [string, number, number][] } = {
  reply: [["Here",8160],[" are",513],[" some",1010],[" cat",7993],[" names",4874],[":",25],[" Luna",48130],[",",11],[" Milo",81533],[",",11],[" Shadow",23536],[",",11],[" Luna",48130],[",",11],[" Milo",81533],[",",11]],
  cands: [[" Shadow",-0.0896,23536],[" Bella",-3.5744,52670],[" Sim",-4.8584,4319],[" Wh",-4.8871,1190],[" Oliver",-5.0203,29260],[" Luna",-5.2131,48130],[" Cle",-5.7006,34809],[" Felix",-6.2065,54048],[" Charlie",-6.3093,24137],[" Max",-6.3492,7252],[" Leo",-6.3968,35530],[" N",-6.4557,443],[" Chloe",-6.4601,57365],[" Sm",-6.6385,4245],[" and",-6.8932,321],[" Snow",-7.1383,18331],[" Tiger",-7.1712,34560],[" shadow",-7.2331,12098],[" Willow",-7.2804,62177],[" Midnight",-7.311,61432],[" Ginger",-7.3367,65023],[" Bailey",-7.3892,40814],[" T",-7.5099,345],[" C",-7.6189,351],[" Fl",-7.664,2893],[" Belle",-7.6838,48808],[" Daisy",-7.8786,67789],[" Kitty",-7.9955,63947],[" Cookie",-8.0238,23588],[" Pepper",-8.026,50038],[" Jasper",-8.0536,68103],[" B",-8.1631,417],["Shadow",-8.196,22926],[" Stella",-8.2029,72059],[" Oscar",-8.2098,29695],[" Lily",-8.4152,45714],[" O",-8.4355,496],[" Simon",-8.4382,20660],[" Ch",-8.4431,882],[" S",-8.4961,326],[" Shadows",-8.5475,64206],[" Bear",-8.576,23237],[" M",-8.5788,380],[" Whisper",-8.5826,93630],[" Mitt",-8.6306,31539],[" Bell",-8.7311,17349],[" Mist",-8.7477,32969],[" Milo",-8.7748,81533],[" Barn",-8.7777,21722],[" Pumpkin",-8.7798,72693],[" Coco",-8.7924,80506],[" Princess",-8.8369,28330],[" Gus",-8.8442,78575],[" Lucy",-8.8952,42610],[" Olivia",-8.932,57283],[" Sophie",-8.9338,57306],[" Pop",-8.9441,9948],[" Jasmine",-8.9874,78548],[" Buddy",-9.0182,57515],[" Gar",-9.0378,11845],[" J",-9.0965,604],[" K",-9.1184,710],[" Cloud",-9.1404,14388],[" Call",-9.1473,6920]],
};
// The default system prompt, an ordinary one: shown in Settings, edited or deleted there. Without it Flash-Next takes
// some asks for a coding-agent task when it does not think ("design a cute SVG ..." -> "let me first explore the
// project structure" and calls to list_files / read_file: 16 replies out of 16, and the Q8_0 original as well); with
// it, 16 of 16 draw.
const DEFAULT_SYSTEM = "There are no tools or files in this chat. Answer directly.";
// Thinking, a slider in the message box: no thinking (the model's no-thinking mode), then Qwen's three efforts.
const EFFORTS: Effort[] = ["off", "low", "medium", "xhigh"];
const LEVEL: Record<Effort, [string]> = { off: ["Off"], low: ["Low"], medium: ["Medium"], xhigh: ["Max"] };
const HEAT = ["off", "low", "medium", "max"];      // each stop's look, cool to molten
const ICONS = {
  flame: '<path fill="currentColor" stroke="none" d="M12 1.5c.9 3.6-.9 5.9-2.6 7.9C7.8 11.2 6 13.1 6 15.9 6 19.3 8.7 22 12 22s6-2.6 6-6.2c0-2.7-1.4-4.7-2.6-6.1-.3 1.6-1.2 2.7-2.4 3.2 1-3.9-.2-8.1-1-11.4z"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  paneLeft: '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M9 3v18"/><path class="chev" d="M15.5 9 12.5 12l3 3"/>',
  paneRight: '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M15 3v18"/><path class="chev" d="M8.5 9l3 3-3 3"/>',
  send: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" stroke="none"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
  retry: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
  edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  chevron: '<path d="M7 10l5 5 5-5"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/>',
  cont: '<path d="M5 6l6 6-6 6M13 6l6 6-6 6"/>',
  tokens: '<path d="M4 19V11M9 19V5M14 19v-6M19 19V9"/>',
  wand: '<path d="M12 3l1.6 4.4L18 9l-4.4 1.6L12 15l-1.6-4.4L6 9l4.4-1.6z"/><path d="M18.5 14.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/>',
  down: '<path d="M12 5v14M5 12l7 7 7-7"/>',
  download: '<path d="M12 4v11M7.5 10.5 12 15l4.5-4.5M5 20h14"/>',
  clip: '<path d="M20 11.6 12.3 19.3a5.2 5.2 0 0 1-7.4-7.4l8.3-8.3a3.5 3.5 0 0 1 4.9 4.9l-8.3 8.3a1.7 1.7 0 0 1-2.5-2.5l7.4-7.4"/>',
  x: '<path d="M7 7l10 10M17 7 7 17"/>',
  grip: '<circle cx="9" cy="6" r="1.3" fill="currentColor" stroke="none"/><circle cx="15" cy="6" r="1.3" fill="currentColor" stroke="none"/><circle cx="9" cy="12" r="1.3" fill="currentColor" stroke="none"/><circle cx="15" cy="12" r="1.3" fill="currentColor" stroke="none"/><circle cx="9" cy="18" r="1.3" fill="currentColor" stroke="none"/><circle cx="15" cy="18" r="1.3" fill="currentColor" stroke="none"/>',
  power: '<path d="M12 3.5v8"/><path d="M6.6 6.8a7.5 7.5 0 1 0 10.8 0"/>',
  dice: '<rect x="4" y="4" width="16" height="16" rx="3.5"/><circle cx="9" cy="9" r="1.1" fill="currentColor"/><circle cx="15" cy="15" r="1.1" fill="currentColor"/><circle cx="15" cy="9" r="1.1" fill="currentColor"/><circle cx="9" cy="15" r="1.1" fill="currentColor"/>',
};
const TITLES = { copy: "Copy", edit: "Edit", trash: "Delete", tokens: "Token odds", retry: "Regenerate" };
const icon = (name: keyof typeof ICONS) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const thread = $("thread"), scroller = $("scroller"), input = $<HTMLTextAreaElement>("input"), sendBtn = $<HTMLButtonElement>("send");
const panel = $("settings"), systemInput = $<HTMLTextAreaElement>("system-input");
// The side panels (the conversations, Settings): one mechanism, in "the side panels" below. A panel's default width
// is style.css's (widthVar); `preferred` is the width it was given, `width` what the window allows of it.
interface Pane {
  name: string; el: HTMLElement; toggle: HTMLElement; key: string; beside: MediaQueryList; byDefault: boolean;
  widthVar: string; min: number; def: number; preferred: number; width: number; onOpen?: () => void;
}
const pane = (name: string, el: HTMLElement, key: string, besideFrom: number, widthVar: string, min: number, onOpen?: () => void): Pane => {
  const def = parseFloat(getComputedStyle(document.documentElement).getPropertyValue(widthVar));
  return { name, el, toggle: $(`${name}-toggle`), key, beside: matchMedia(`(min-width: ${besideFrom}px)`), byDefault: el.hasAttribute("data-open"),
    widthVar, min, def, preferred: def, width: def, onOpen };
};
const panes = {
  conversations: pane("conversations", $("sidebar"), "b", 761, "--side-open", 208),
  settings: pane("settings", panel, ",", 1460, "--panel-open", 340, () => { syncSettings(); growSystem(); }),
};
const PANES = Object.values(panes);

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
      const preview = PAGE_LANGS.has(language) ? `<button type="button" class="code-act" data-page="preview">Preview</button>` : "";
      return `<div class="code"><div class="code-head"><span>${escapeHtml(language || "text")}</span>` +
        `<span class="code-acts">${preview}<button type="button" class="copy">Copy</button></span></div><pre><code class="hljs">${html}</code></pre></div>`;
    },
  },
});
const markdown = (src: string) => DOMPurify.sanitize(marked.parse(src, { async: false }));

// TeX as Qwen writes it: $$ … $$ or \[ … \] on their own lines (display), $ … $ or \( … \) in a line. A $ opens math
// only before a non-space and closes it only after one, not before a digit, and math never crosses a backtick (a code
// span's $ is not a closing one), so prices stay text. The lexer leaves a
// placeholder; KaTeX typesets it once the block is sanitized: its output (with trust off) is safe, and DOMPurify would
// strip the MathML it carries for screen readers.
const INLINE_MATH = [/^\$\$([\s\S]+?)\$\$/, /^\\\(([\s\S]+?)\\\)/, /^\$(?![\s$])((?:\\[\s\S]|[^\\$\n`])+?)(?<![\s\\])\$(?!\d)/];
marked.use({
  extensions: [{
    name: "mathBlock", level: "block",
    start: (src) => src.match(/^ {0,3}(?:\$\$|\\\[)/m)?.index,
    tokenizer(src) {
      const m = src.match(/^ {0,3}\$\$([\s\S]+?)\$\$[ \t]*(?:\n|$)/) ?? src.match(/^ {0,3}\\\[([\s\S]+?)\\\][ \t]*(?:\n|$)/);
      if (m) return { type: "mathBlock", raw: m[0], text: m[1].trim() };
    },
    renderer: (t) => `<div class="math display" data-tex="${escapeHtml(t.text)}"></div>`,
  }, {
    name: "math", level: "inline",
    start: (src) => src.match(/(?<!\\)\$|\\\(/)?.index,
    tokenizer(src) {
      for (const [k, re] of INLINE_MATH.entries()) {
        const m = src.match(re);
        if (m) return { type: "math", raw: m[0], text: m[1].trim(), display: k === 0 };
      }
    },
    renderer: (t) => `<span class="math${t.display ? " display" : ""}" data-tex="${escapeHtml(t.text)}"></span>`,
  }],
});
function typeset(root: ParentNode) {
  for (const el of root.querySelectorAll<HTMLElement>(".math[data-tex]"))
    katex.render(el.dataset.tex!, el, { displayMode: el.classList.contains("display"), throwOnError: false, strict: "ignore", output: "htmlAndMathml" });
}

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
  const prose = src.replace(/^ {0,3}(`{3,}|~{3,})[^]*?^ {0,3}\1.*$/gm, "").replace(/`[^`\n]*`/g, "");
  if ((prose.match(/\$\$/g)?.length ?? 0) % 2) return src.slice(0, src.lastIndexOf("$$"));   // display math once it closes
  if (prose.lastIndexOf("\\[") > prose.lastIndexOf("\\]")) return src.slice(0, src.lastIndexOf("\\["));
  if (/^\s*\|/.test(line)) return head;                                   // a table row once it is complete
  if (/^\s*(?:[#>*+-]|\d+[.)])*\s*$/.test(line)) return head;             // a bare "#", "-", "1." ... line
  if (!/\s$/.test(line)) line = line.replace(/[A-Za-z0-9À-ɏ'’_-]{1,24}$/, "");   // a word once it ends
  const text = head + line, p = text.lastIndexOf("\n\n") + 1;
  let para = text.slice(p)                                                // the paragraph being written:
    .replace(/!\[[^\]\n]*\]?(?:\([^)\n]*)?$/, "")                         // no half image,
    .replace(/\[([^\]\n]*)\]\([^)\n]*$/, "$1").replace(/\[([^\]\n]*)$/, "$1")   // a link is its text until it is whole,
    .replace(/<\/?[A-Za-z][^>\n]*$/, "")                                  // no half HTML tag,
    .replace(/(?:\*{1,3}|_{1,3}|~{1,2}|`+)$/, "");                        // markers with nothing after them wait;
  const dollars = [...para.replace(/`[^`]*`/g, (c) => " ".repeat(c.length)).matchAll(/(?<![\\$])\$(?!\$)/g)];
  const lastDollar = dollars.at(-1)?.index ?? 0;                          // inline math once it closes ($5 is a price)
  if (dollars.length % 2 && !/^\$\d/.test(para.slice(lastDollar))) para = para.slice(0, lastDollar);
  if (para.lastIndexOf("\\(") > para.lastIndexOf("\\)")) para = para.slice(0, para.lastIndexOf("\\("));
  if ((para.replace(/```[\s\S]*?```/g, "").match(/`/g)?.length ?? 0) % 2) para += "`";   // open code and bold close
  if ((para.match(/\*\*/g)?.length ?? 0) % 2) para += "**";
  if ((para.match(/~~/g)?.length ?? 0) % 2) para += "~~";
  return text.slice(0, p) + para;
}

function htmlNodes(t: Token, links: Links): Node[] {
  const tpl = document.createElement("template");
  tpl.innerHTML = DOMPurify.sanitize(marked.parser(Object.assign([t], { links })));      // (options here would drop marked.use's renderer)
  typeset(tpl.content);
  return [...tpl.content.childNodes];
}

function renderBlocks(v: View, src: string, live: boolean) {
  if (!v.blocks) { v.body.replaceChildren(); v.blocks = { raws: [], nodes: [] }; }
  const b = v.blocks, tokens = marked.lexer(live ? settle(src) : src), blocks = tokens.filter((t) => t.type !== "space");
  let k = 0;
  while (k < blocks.length && k < b.raws.length && b.raws[k] === blocks[k].raw) k++;
  for (; k < blocks.length; k++) {
    const t = blocks[k], old = b.nodes[k] ?? [];
    const tail = live && k === blocks.length - 1;
    const page = t.type === "code" && !tail && PAGE_LANGS.has((t.lang ?? "").toLowerCase()) && (PAGE_DOC.test(t.text) || previewed.has(t.text));
    const now = t.type !== "code" ? htmlNodes(t, tokens.links) : SVG_START.test(t.text) ? [svgFigure(old[0], t.text, tail)]
      : /^mermaid$/i.test(t.lang ?? "") ? [mermaidFigure(old[0], t.text, tail)] : page ? [pageFigure(old[0], t.text)] : htmlNodes(t, tokens.links);
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
    if (f instanceof HTMLElement && f.classList.contains("drawing") && !(live && k === blocks.length - 1) && t.type === "code")
      (f.dataset.kind === "mermaid" ? mermaidFigure : svgFigure)(f, t.text, false);
  });
}

// A code block holding an SVG document (models label it svg, xml or html) shows the picture instead, as a figure that
// draws itself while it streams: its size is set once the <svg> tag is in (the viewBox), then each element appears,
// with a short fade, once its tag is complete. The markup is sanitized (no scripts, handlers, foreignObject, external
// references) and lives in a shadow root, so its styles and ids stay inside it. The code is a click away.
const SVG_START = /^\s*(?:<\?xml[^>]*\?>\s*)?(?:<!--[\s\S]*?-->\s*|<!DOCTYPE[^>]*>\s*)*<svg[\s>]/i;
const SVG_SHADOW = ":host { display: block; color: #1f1b19; } svg { display: block; width: var(--fig-w, 640px); max-width: 100%; height: auto; } " +
  ":host(.zoom) svg { max-width: none; }";
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

// The figure around a picture: the light page, its caption with the actions, and the code a click away.
function figureShell(lang: "svg" | "mermaid"): HTMLElement {
  const fig = h("figure", "svg-fig");
  fig.dataset.kind = lang;
  fig.innerHTML = `<div class="svg-paper"></div><figcaption class="svg-bar"><span class="svg-state"></span><span class="svg-acts">` +
    ["code:Code", "copy:Copy", "save:Download", "zoom:Expand"].map((a) => `<button type="button" data-svg="${a.split(":")[0]}">${a.split(":")[1]}</button>`).join("") +
    `</span></figcaption><div class="code svg-code"><div class="code-head"><span>${lang}</span><button type="button" class="copy">Copy</button></div>` +
    `<pre><code class="hljs"></code></pre></div>`;
  fig.querySelector(".svg-paper")!.attachShadow({ mode: "open" }).innerHTML = `<style>${SVG_SHADOW}</style>`;
  return fig;
}

function svgFigure(old: Node | undefined, text: string, drawing: boolean): HTMLElement {
  const fig = old instanceof HTMLElement && old.dataset.kind === "svg" ? old : figureShell("svg");
  svgSource.set(fig, text);
  fig.classList.toggle("drawing", drawing);
  const t = performance.now();
  if (!drawing || t - Number(fig.dataset.drawn ?? 0) > 90) {             // at most ~11 redraws a second while it streams
    fig.dataset.drawn = String(t);
    const ok = drawSvg(fig.querySelector(".svg-paper")!.shadowRoot!, svgPrefix(text), drawing);
    fig.classList.toggle("failed", !drawing && !ok);
  }
  fitFigure(fig);
  const svg = fig.querySelector(".svg-paper")!.shadowRoot!.querySelector("svg"), box = svg?.viewBox.baseVal;
  fig.querySelector(".svg-state")!.textContent = drawing ? "Drawing" : fig.classList.contains("failed") ? "Couldn't draw this SVG"
    : box && box.width ? `SVG · ${Math.round(box.width)} × ${Math.round(box.height)}` : "SVG";
  if (fig.classList.contains("show-code") || fig.classList.contains("failed")) svgCode(fig);
  return fig;
}

// A drawing as large as what can be seen at once: the column's width, or the scroll area's height less the caption,
// whichever runs out first, its shape kept. Set when its <svg> tag arrives, so it never changes size while it draws.
function fitFigure(fig: HTMLElement) {
  const paper = fig.querySelector<HTMLElement>(".svg-paper"), vb = paper?.shadowRoot!.querySelector("svg")?.viewBox.baseVal;
  if (!paper || !vb || !vb.width || !vb.height) return;               // (a page figure has no picture)
  const room = thread.clientWidth - 48 - 30;             // the column (the thread's padding), less the paper's padding and border
  const tall = Math.max(scroller.clientHeight - 74, 320); // the scroll area, less 8 + 6 px of gap, the paper's padding and the caption
  const natural = fig.dataset.kind === "mermaid" ? vb.width : Infinity;   // a diagram's text stays at its size
  const w = `${Math.floor(Math.min(room, tall * (vb.width / vb.height), natural))}px`;
  if (paper.style.getPropertyValue("--fig-w") !== w) paper.style.setProperty("--fig-w", w);
}
addEventListener("resize", () => thread.querySelectorAll<HTMLElement>(".svg-fig").forEach(fitFigure));

function svgCode(fig: HTMLElement) {
  const code = fig.querySelector(".svg-code code")!, text = svgSource.get(fig) ?? "";
  if (code.textContent !== text)
    code.innerHTML = fig.dataset.kind === "svg" ? hljs.highlight(text, { language: "xml", ignoreIllegals: true }).value : escapeHtml(text);
}

function svgAction(btn: HTMLElement) {
  const fig = btn.closest<HTMLElement>(".svg-fig")!, text = svgSource.get(fig) ?? "";
  const act = btn.dataset.svg, picture = fig.dataset.kind === "mermaid" ? pictures.get(fig) ?? "" : text;
  if (act === "code") {
    btn.textContent = fig.classList.toggle("show-code") ? "Picture" : "Code";
    svgCode(fig);
  } else if (act === "copy") copy(text, btn);
  else if (act === "save") download(fig.dataset.kind === "mermaid" ? "diagram.svg" : "image.svg", svgFile(picture), "image/svg+xml");
  else if (act === "zoom") {
    const dlg = $<HTMLDialogElement>("svg-zoom"), host = h("div");
    host.className = "zoom";
    host.attachShadow({ mode: "open" }).innerHTML = `<style>${SVG_SHADOW}</style>`;
    drawSvg(host.shadowRoot!, svgPrefix(picture), false);
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

// A ```mermaid block becomes a diagram in the SVG figure (Code shows its Mermaid source, Download saves the picture).
// Mermaid is most of the page's code, so it loads with the first diagram. While the block streams, each settled
// version that parses replaces the picture, a few times a second, so the diagram grows as it is written; a finished
// block that does not parse shows its source. Labels are SVG text (no HTML in foreignObject, which drawSvg removes).
const pictures = new WeakMap<Element, string>();      // a diagram's SVG as drawn
const DIAGRAMS: Record<string, string> = { graph: "flowchart", flowchart: "flowchart", sequenceDiagram: "sequence diagram",
  classDiagram: "class diagram", stateDiagram: "state diagram", "stateDiagram-v2": "state diagram", erDiagram: "ER diagram",
  gantt: "Gantt chart", pie: "pie chart", mindmap: "mind map", timeline: "timeline", journey: "user journey", gitGraph: "git graph",
  quadrantChart: "quadrant chart", xychart: "XY chart", "xychart-beta": "XY chart", sankey: "Sankey diagram", "sankey-beta": "Sankey diagram" };
// Mermaid's own ES module build, served by server.ts: imported by URL, so the page's bundle leaves it out, and it loads
// each kind of diagram's code when one is first drawn.
const MERMAID_URL = "/vendor/mermaid/mermaid.esm.min.mjs";
let mermaidLib: Promise<typeof import("mermaid").default> | undefined, mermaidSeq = 0;
const loadMermaid = () => mermaidLib ??= (import(MERMAID_URL) as Promise<typeof import("mermaid")>).then(({ default: m }) => {
  m.initialize({ startOnLoad: false, securityLevel: "strict", htmlLabels: false, flowchart: { htmlLabels: false }, theme: "base",
    fontFamily: "ui-sans-serif, system-ui, sans-serif", themeVariables: { primaryColor: "#fbe6d6", primaryBorderColor: "#d8621f",
      primaryTextColor: "#1f1b19", lineColor: "#6d635b", secondaryColor: "#f1ebe4", tertiaryColor: "#ffffff" } });
  return m;
});

function mermaidFigure(old: Node | undefined, text: string, drawing: boolean): HTMLElement {
  const fig = old instanceof HTMLElement && old.dataset.kind === "mermaid" ? old : figureShell("mermaid");
  svgSource.set(fig, text);
  fig.classList.toggle("drawing", drawing);
  const t = performance.now();
  if (!drawing || t - Number(fig.dataset.drawn ?? 0) > 300) {
    fig.dataset.drawn = String(t);
    void drawMermaid(fig, text, drawing);
  }
  captionMermaid(fig);
  return fig;
}

async function drawMermaid(fig: HTMLElement, text: string, drawing: boolean) {
  const seq = String(++mermaidSeq);
  fig.dataset.seq = seq;
  const m = await loadMermaid();
  let svg: string | undefined;
  if (await m.parse(text, { suppressErrors: true })) svg = await m.render(`kiln-mermaid-${seq}`, text).then((r) => r.svg, () => undefined);
  if (fig.dataset.seq !== seq) return;                   // a later version is on its way
  if (svg) {
    pictures.set(fig, svg);
    drawSvg(fig.querySelector(".svg-paper")!.shadowRoot!, svg, false);
    fitFigure(fig);
  }
  fig.classList.toggle("failed", !drawing && !svg);
  captionMermaid(fig);
  if (fig.classList.contains("show-code") || fig.classList.contains("failed")) svgCode(fig);
  keepUp();
}

function captionMermaid(fig: HTMLElement) {
  const kind = DIAGRAMS[(svgSource.get(fig) ?? "").replace(/^\s*(%%.*\n\s*)*/, "").split(/[\s;]/)[0]] ?? "diagram";
  fig.querySelector(".svg-state")!.textContent = fig.classList.contains("drawing") ? "Drawing"
    : fig.classList.contains("failed") ? "Couldn't draw this diagram" : `Mermaid · ${kind}`;
}


// A ```html block holding a whole document runs as a page once it is written; a fragment does when its Preview is
// clicked. The page runs in an iframe on /sandbox (server.ts): its own code runs, but it can reach neither the chat nor
// the network. It reports its height and the frame takes it, up to what can be seen at once (a taller page scrolls
// inside). A page sized by the viewport (100vh) reports the frame's own height, so the frame starts at a useful size.
const PAGE_LANGS = new Set(["html", "htm"]);
const PAGE_DOC = /^\s*(?:<!--[\s\S]*?-->\s*)*(?:<!doctype html|<html[\s>])/i;
const previewed = new Set<string>();                  // fragments whose Preview was clicked
const pageText = new WeakMap<HTMLIFrameElement, string>();
// The root element's box: its content's height (scrollHeight would never report less than the frame already is).
const REPORT = `<script>(() => { const html = document.documentElement, post = () => parent.postMessage({ kilnHeight: Math.ceil(html.getBoundingClientRect().height) }, "*");
  new ResizeObserver(post).observe(html); addEventListener("load", post); post(); })()</script>`;
const pageRoom = () => Math.max(scroller.clientHeight - 74, 240);

function pageFrame(text: string, fixed: boolean): HTMLIFrameElement {
  const frame = h("iframe", "page-frame");
  frame.setAttribute("sandbox", "allow-scripts allow-modals");
  frame.title = "Page from the reply";
  if (fixed) frame.dataset.fixed = "1";
  else frame.style.height = `${Math.min(480, pageRoom())}px`;
  pageText.set(frame, (PAGE_DOC.test(text) ? "" : "<!doctype html>") + text);
  frame.src = "/sandbox";
  return frame;
}

function pageFigure(old: Node | undefined, text: string): HTMLElement {
  if (old instanceof HTMLElement && old.dataset.kind === "page" && svgSource.get(old) === text) return old;
  const fig = h("figure", "svg-fig page-fig");
  fig.dataset.kind = "page";
  svgSource.set(fig, text);
  const title = text.match(/<title[^>]*>([^<]*)<\/title>/i)?.[1].trim();
  fig.innerHTML = `<figcaption class="svg-bar"><span class="svg-state"></span><span class="svg-acts">` +
    ["code:Code", "copy:Copy", "save:Download", "reload:Reload", "zoom:Expand"].map((a) => `<button type="button" data-page="${a.split(":")[0]}">${a.split(":")[1]}</button>`).join("") +
    `</span></figcaption><div class="code svg-code"><div class="code-head"><span>html</span><button type="button" class="copy">Copy</button></div>` +
    `<pre><code class="hljs"></code></pre></div>`;
  fig.querySelector(".svg-state")!.textContent = title ? `Page · ${title}` : "Page";
  fig.prepend(pageFrame(text, false));
  return fig;
}

addEventListener("message", (e) => {
  const frame = [...document.querySelectorAll<HTMLIFrameElement>("iframe.page-frame")].find((f) => f.contentWindow === e.source);
  if (!frame) return;
  if (e.data?.kilnReady) frame.contentWindow!.postMessage({ kilnPage: pageText.get(frame) + REPORT }, "*");
  else if (typeof e.data?.kilnHeight === "number" && !frame.dataset.fixed) {
    const px = `${Math.round(Math.min(Math.max(e.data.kilnHeight, 60), pageRoom()))}px`;
    if (frame.style.height !== px) frame.style.height = px;
  }
});

function pageAction(btn: HTMLElement) {
  const act = btn.dataset.page;
  if (act === "preview") {                               // a fragment's code block becomes its page
    const block = btn.closest<HTMLElement>(".code")!, text = block.querySelector("code")?.textContent ?? "";
    const m = conv.messages[Number(block.closest<HTMLElement>("article")?.dataset.i)], b = m && views.get(m)?.blocks;
    previewed.add(text);
    const fig = pageFigure(undefined, text), k = b ? b.nodes.findIndex((n) => n.includes(block)) : -1;
    block.replaceWith(fig);
    if (b && k >= 0) b.nodes[k] = [fig];
    return;
  }
  const fig = btn.closest<HTMLElement>(".page-fig")!, text = svgSource.get(fig) ?? "";
  if (act === "code") {
    btn.textContent = fig.classList.toggle("show-code") ? "Page" : "Code";
    const code = fig.querySelector(".svg-code code")!;
    if (code.textContent !== text) code.innerHTML = hljs.highlight(text, { language: "xml", ignoreIllegals: true }).value;
  } else if (act === "copy") copy(text, btn);
  else if (act === "save") download("page.html", text, "text/html");
  else if (act === "reload") fig.querySelector("iframe")!.replaceWith(pageFrame(text, false));
  else if (act === "zoom") {
    const dlg = $<HTMLDialogElement>("page-zoom");
    dlg.querySelector(".page-zoom-box")!.replaceChildren(pageFrame(text, true));
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
let settings = loadSettings();
let conv = blank();
const runs = new Map<string, Run>();               // conversation id -> its reply being written
const running = (c: Conversation = conv) => runs.get(c.id);
let status: Status = { state: "offline" };
let build = "";                                     // the server's code version when this page loaded
let follow = true;                                  // keep the newest text in view (see keepUp)
const views = new Map<Message, View>();
const tokens = new Map<string, { reasoning: Tok[]; content: Tok[] }>();   // replies' token odds by message id (this page only)
const tokenView = new Set<string>();                // replies shown as their tokens
let editing: Message | null = null;                 // the message being edited in place

// Plain-HTTP pages (the web chat on the LAN) get no crypto.randomUUID; getRandomValues works on every page.
function randomId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}

function blank(): Conversation {
  return { id: randomId(), title: "New chat", updated: Date.now(), effort: pref("effort", "xhigh") as Effort, messages: [] };
}

function render() {
  views.clear();
  thread.replaceChildren(...(conv.messages.length ? conv.messages.map(messageView) : [emptyState()]));
  $("title").textContent = conv.title;
  $("export").hidden = !conv.messages.length;
  document.title = conv.messages.length ? `${conv.title} · kiln` : "kiln";
  renderComposer();
  renderList();
}

function messageView(m: Message, i: number): HTMLElement {
  if (m === editing) return editorView(m, i);
  if (m.role === "assistant") {
    const v = assistantView(i);
    views.set(m, v);
    paint(v, m, [...runs.values()].find((r) => r.m === m)?.live);
    return v.el;
  }
  const el = h("article", "msg user"), foot = h("footer", "msg-foot");
  el.dataset.i = String(i);
  foot.innerHTML = slots([["copy", true], ["edit", !running()], ["trash", !running()]]);
  if (m.files?.length) el.append(filesView(m.files));
  if (m.content || !m.files?.length) el.append(h("div", "bubble", m.content));
  el.append(foot);
  return el;
}

// A message edited in place: its text, and a reply's thinking. Save keeps the rest of the chat; on the last user
// message, Save & regenerate also draws a new reply. Ctrl+Enter takes the main action, Esc cancels.
function editorView(m: Message, i: number): HTMLElement {
  const el = h("article", `msg ${m.role} editing`), box = h("div", "editor"), acts = h("div", "editor-acts");
  el.dataset.i = String(i);
  const area = (value: string, label: string) => {
    const t = h("textarea");
    t.value = value;
    t.rows = 1;
    t.setAttribute("aria-label", label);
    t.addEventListener("input", () => fit(t));
    return t;
  };
  const think = m.role === "assistant" && m.reasoning ? area(m.reasoning, "Thinking") : null;
  const text = area(m.content, m.role === "user" ? "Message" : "Reply");
  if (think) box.append(h("div", "ed-label", "Thinking"), think, h("div", "ed-label", "Reply"));
  box.append(text, acts);
  const again = m.role === "user" && i === conv.messages.findLastIndex((x) => x.role === "user");
  const button = (label: string, cls: string, fn: () => void) => {
    const b = h("button", cls, label);
    b.type = "button";
    b.onclick = fn;
    acts.append(b);
  };
  const done = (keep: boolean, regenerate = false) => {
    if (keep && (text.value !== m.content || (think && think.value !== m.reasoning))) {
      m.content = text.value;
      if (think) m.reasoning = think.value;
      m.edited = true;
      if (m.id) { tokens.delete(m.id); tokenView.delete(m.id); }
    }
    editing = null;
    if (regenerate) {
      conv.messages.splice(i + 1);
      void generate();
    } else render();
    save(conv);
  };
  acts.append(h("span", "editor-hint", again ? "Ctrl+Enter regenerates · Esc cancels" : "Ctrl+Enter saves · Esc cancels"));
  button("Cancel", "btn", () => done(false));
  button("Save", again ? "btn" : "btn primary", () => done(true));
  if (again) button("Save & regenerate", "btn primary", () => done(true, true));
  el.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); done(false); }
    else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); done(true, again); }
  });
  el.append(box);
  requestAnimationFrame(() => {
    for (const t of [think, text]) if (t) fit(t);
    text.focus();
    text.setSelectionRange(text.value.length, text.value.length);
  });
  return el;
}
const fit = (t: HTMLTextAreaElement) => { t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight + 2, innerHeight * 0.6)}px`; };

function deleteMessage(i: number) {
  const c = conv, [m] = c.messages.splice(i, 1);
  persist(c);
  render();
  toast(m.role === "user" ? "Message deleted" : "Reply deleted", () => {
    c.messages.splice(i, 0, m);
    persist(c);
    if (c === conv) render();
  });
}

function assistantView(i: number): View {
  const el = h("article", "msg assistant"), think = h("details", "thinking"), summary = h("summary"), thought = h("div", "thought");
  const body = h("div", "markdown"), foot = h("footer", "msg-foot");
  el.dataset.i = String(i);
  think.append(summary, thought);
  el.append(think, body, foot);
  // while it streams the thinking box keeps its newest line in view, until it is scrolled up by hand (back at its end,
  // it follows again)
  thought.addEventListener("wheel", (e) => { if (e.deltaY < 0) thought.dataset.held = "1"; }, { passive: true });
  thought.addEventListener("scroll", () => { if (thought.scrollHeight - thought.scrollTop - thought.clientHeight < 4) delete thought.dataset.held; });
  return { el, think, summary, thought, body, foot };
}

// Message actions in fixed places: from the right, delete, edit and copy on every message, then a reply's token odds,
// regenerate and continue. An action that is not available keeps its place empty, so none ever moves.
type Act = keyof typeof TITLES | "cont";
const slots = (list: [Act, boolean, string?][]) => list.map(([a, on, html]) => !on ? `<span class="act ghost" aria-hidden="true"></span>`
  : html ?? `<button type="button" class="act" data-act="${a}" title="${TITLES[a as keyof typeof TITLES]}">${icon(a)}</button>`).join("");

// Fill an assistant message's view; `live` while it streams (cheap to call repeatedly).
function paint(v: View, m: Message, live?: Live) {
  const elapsed = live ? (performance.now() - live.since) / 1000 : 0;
  v.think.hidden = !m.reasoning;
  v.think.classList.toggle("live", live?.phase === "thinking");
  if (m.reasoning) {
    v.summary.textContent = live?.phase === "thinking" ? `Thinking · ${fmtS(elapsed)}`
      : m.stats?.thought ? `Thought for ${fmtS(m.stats.thought)}` : "Thoughts";
  }
  const asTokens = !!m.id && tokenView.has(m.id) && tokens.has(m.id);
  v.el.classList.toggle("as-tokens", asTokens);
  if (asTokens) {
    if (!live) v.tok = undefined;                  // drawn afresh (and with the error, if any) once it is done
    paintTokens(v, m);
    if (live) {
      if (live.phase === "thinking" && !v.thought.dataset.held) v.thought.scrollTop = v.thought.scrollHeight;
      return;
    }
  } else {
    for (const box of [v.thought, v.body]) if (box.classList.contains("tokens")) { box.classList.remove("tokens"); box.replaceChildren(); }
    v.tok = undefined;
  }
  if (!asTokens && m.reasoning && v.thought.textContent !== m.reasoning) {
    v.thought.textContent = m.reasoning;
    if (live && !v.thought.dataset.held) v.thought.scrollTop = v.thought.scrollHeight;
  }
  if (asTokens) { /* drawn above */ } else if (live?.phase === "waiting") {
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
  if (m.params) facts.push(m.params);
  if (m.edited) facts.push("edited");
  if (s?.seed !== undefined) facts.push(`<button type="button" class="seed" data-act="seed" data-seed="${s.seed}" title="Use this seed for the next replies">seed ${s.seed}</button>`);
  const last = conv.messages.at(-1) === m, busy = !!running(), cut = m.finish === "length" || m.stopped;
  // Continue is spelled out when the reply was cut short (its length limit, or Stop)
  const cont = cut ? `<button type="button" class="btn" data-act="cont" title="Continue">${icon("cont")}Continue</button>`
    : `<button type="button" class="act" data-act="cont" title="Continue">${icon("cont")}</button>`;
  v.foot.innerHTML = `<span class="stats">${facts.join(" · ")}</span>` + (last ? slots([["cont", !busy, cont], ["retry", !busy]]) : "")
    + slots([["tokens", !!m.id && tokens.has(m.id)], ["copy", true], ["edit", !busy], ["trash", !busy]]);
  v.foot.querySelector("[data-act=tokens]")?.classList.toggle("on", asTokens);
}

// A new chat: the flame, nothing else.
function emptyState() {
  const e = h("div", "empty");
  e.innerHTML = `<svg class="flame" viewBox="0 0 24 24" aria-hidden="true">${ICONS.flame}</svg>`;
  return e;
}

// A reply as its tokens, tinted by how unsure the model was of each; a click on one shows what else it could have
// written there. Drawn incrementally while the reply streams.
function paintTokens(v: View, m: Message) {
  const td = tokens.get(m.id!)!;
  for (const [ch, box] of [["reasoning", v.thought], ["content", v.body]] as const) {
    const list = td[ch];
    if (!v.tok || v.tok[ch] > list.length || !box.classList.contains("tokens")) {
      box.replaceChildren();
      box.classList.add("tokens");
      v.tok = { reasoning: 0, content: 0, ...(v.tok ?? {}), [ch]: 0 };
    }
    const frag = document.createDocumentFragment();
    for (let i = v.tok[ch]; i < list.length; i++) {
      const t = h("span", "tok", list[i].t);
      t.dataset.k = ch === "reasoning" ? "r" : "c";
      t.dataset.i = String(i);
      if (list[i].p < 0.97) t.style.setProperty("--u", (1 - list[i].p).toFixed(2));
      frag.append(t);
    }
    box.append(frag);
    v.tok[ch] = list.length;
  }
  v.shown = v.blocks = undefined;                   // markdown is drawn afresh when the view goes back
}

const SPECIAL: Record<string, string> = { "<|im_end|>": "end of reply", "<|endoftext|>": "end of reply", "</think>": "end of thinking" };
const showTok = (t: string) => SPECIAL[t] ?? (t.replace(/\n/g, "↵").replace(/^ | $/g, "·") || "∅");

function closeTokPop() {
  $("tokpop").hidden = true;
  thread.querySelector(".tok.picked")?.classList.remove("picked");
}
// What else the model could have written at a token; picking one rewrites the reply from there.
function openTokPop(span: HTMLElement) {
  closeTokPop();
  const m = conv.messages[Number(span.closest<HTMLElement>("article")?.dataset.i)];
  const ch = span.dataset.k === "r" ? "reasoning" : "content", i = Number(span.dataset.i), t = m?.id && tokens.get(m.id)?.[ch][i];
  if (!m || !t) return;
  const pop = $("tokpop"), busy = !!running();
  pop.replaceChildren(h("div", "tokpop-head", busy ? "Alternatives (after the reply is done, pick one to rewrite from there)" : "Rewrite from here with"));
  const alts = t.alts.some((a) => a.t === t.t) ? t.alts : [{ t: t.t, p: t.p, id: -1 }, ...t.alts];
  for (const a of alts) {
    const b = h("button", "tokalt"), code = h("code", SPECIAL[a.t] ? "special" : "", showTok(a.t)), bar = h("i");
    b.type = "button";
    bar.style.width = `${Math.max(2, a.p * 100)}%`;
    b.append(code, h("span", "tokalt-bar"), h("span", "tokalt-p", pct(a.p) || "<1%"));
    b.querySelector(".tokalt-bar")!.append(bar);
    if (a.t === t.t) { b.classList.add("chosen"); b.title = "What it wrote"; }
    b.disabled = a.t === t.t || busy;
    b.onclick = () => { closeTokPop(); pickToken(m, ch, i, a); };
    pop.append(b);
  }
  pop.hidden = false;
  span.classList.add("picked");
  const r = span.getBoundingClientRect(), w = pop.offsetWidth, ph = pop.offsetHeight;
  pop.style.left = `${Math.max(8, Math.min(innerWidth - w - 8, r.left))}px`;
  pop.style.top = `${r.bottom + 6 + ph > innerHeight - 8 ? r.top - ph - 6 : r.bottom + 6}px`;
}

function pickToken(m: Message, ch: "reasoning" | "content", i: number, a: Alt) {
  const td = tokens.get(m.id!)!, before = td[ch].slice(0, i).map((x) => x.t).join("");
  if (SPECIAL[a.t] === "end of reply") {            // nothing to write: the reply just ends there
    const was = { ...m }, wasTok = structuredClone(td);
    td[ch] = td[ch].slice(0, i);
    m[ch] = before;
    if (ch === "reasoning") { td.content = []; m.content = ""; }
    m.finish = "stop";
    save(conv);
    render();
    toast("The reply ends there now", () => { Object.assign(m, was); tokens.set(m.id!, wasTok); save(conv); render(); });
    return;
  }
  continueReply(m, { ch, i, alt: a, before });
}


function renderComposer() {
  const busy = !!running();
  sendBtn.innerHTML = icon(busy ? "stop" : "send");
  sendBtn.setAttribute("aria-label", busy ? "Stop" : "Send");
  sendBtn.classList.toggle("stop", busy);
  const draftBtn = $<HTMLButtonElement>("draft");
  draftBtn.innerHTML = icon(drafting ? "stop" : "wand");
  draftBtn.title = drafting ? "Stop the draft" : input.value.trim() ? "Finish my message (draft)" : "Draft my next message";
  draftBtn.setAttribute("aria-label", draftBtn.title);
  draftBtn.classList.toggle("on", !!drafting);
  draftBtn.hidden = busy || !conv.messages.length;
  $("composer").classList.toggle("drafting", !!drafting);
  renderControls();
  if (modeOf(conv) !== chatMode) {                  // Settings shows the values of the thinking slider's mode
    chatMode = modeOf(conv);
    syncSettings();                                 // (it does nothing while Settings is closed)
  }
}

// Anything differs from the recommended values (what Reset all settings would change).
const customized = () => settings.system !== DEFAULT_SYSTEM || Object.keys(settings.think).length + Object.keys(settings.fast).length > 0
  || !!settings.max_tokens || settings.seed !== undefined;

function renderList() {
  const nav = $("conversations"), today = new Date().setHours(0, 0, 0, 0);
  const group = (t: number) => (t >= today ? "Today" : t >= today - 6 * 864e5 ? "Previous 7 days" : "Older");
  let current = "";
  nav.replaceChildren();
  for (const s of Object.values(chats).sort((a, b) => b.updated - a.updated)) {
    if (group(s.updated) !== current) nav.append(h("div", "group", (current = group(s.updated))));
    const a = h("a", `conv${s.id === conv.id ? " active" : ""}${runs.has(s.id) ? " running" : ""}`), del = h("button", "conv-del", "×");
    a.href = `#${s.id}`;
    a.onclick = (e) => { if (!e.defaultPrevented) closeOver(panes.conversations); };   // the open chat too (no hash change)
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

// The engine's state lives with the model in the message box (renderControls): nothing of it in the sidebar.
function renderStatus() {
  renderControls();
}

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
  if (status.build && status.build !== build && !runs.size && !input.value && !panel.contains(document.activeElement)
    && !editing && !drafting) return location.reload();
  renderStatus();
}

// Saved, or forgotten once it has no messages left (deleting its last one).
function persist(c: Conversation) {
  if (c.messages.length) return save(c);
  delete chats[c.id];
  store("chats", chats);
  renderList();
}

function save(c: Conversation) {
  if (!c.messages.length) return;
  chats[c.id] = structuredClone(c);
  store("chats", chats);
  if (c === conv && location.hash !== `#${c.id}`) history.replaceState(null, "", `#${c.id}`);
  renderList();
}

// The request body for a reply after the messages `upto`. Assistant turns go back exactly as kiln sent them: kiln then
// replays its own tokens and every turn is a cache hit (only the new message is read). The reply's tokens come with
// their odds (log probs, five alternatives each) for the token view. effort: the thinking setting (the chat's by
// default); sampling: whose values (by default the effort's mode).
function request(c: Conversation, upto: Message[], o: { effort?: Effort; sampling?: Mode; final?: object; extra?: Record<string, unknown> } = {}) {
  const effort = o.effort ?? c.effort, mode: Mode = effort === "off" ? "fast" : "think";
  const messages: object[] = upto
    .filter((m) => m.role === "user" || m.content || m.reasoning)
    .map((m) => (m.reasoning ? { role: m.role, content: m.content, reasoning_content: m.reasoning } : { role: m.role, content: m.role === "user" ? userContent(m) : m.content }));
  if (settings.system.trim()) messages.unshift({ role: "system", content: settings.system.trim() });
  if (o.final) messages.push(o.final);
  return {
    model: effort === "off" ? `${current()}-fast` : current(), reasoning_effort: effort === "off" ? undefined : effort,
    ...samplerFields(o.sampling ?? mode), max_tokens: settings.max_tokens, seed: settings.seed, logprobs: true, top_logprobs: 5,
    messages, stream: true, stream_options: { include_usage: true }, ...o.extra,
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
  const m: Message = { id: randomId(), role: "assistant", content: "", reasoning: "", params: changed(modeOf(c)) || undefined, effort: c.effort };
  const upto = c.messages.slice();
  c.messages.push(m);
  await startRun(c, m, () => request(c, upto));
}

// Writes on from the end of the last reply, or (pick) from one of its tokens replaced by an alternative. kiln is
// given the reply as it generated it (kiln_original), finds the exact tokens to keep and serves them from its cache;
// it restates the reply as it now stands before the new text.
function continueReply(m: Message, pick?: { ch: "reasoning" | "content"; i: number; alt: Alt; before: string }) {
  const c = conv, at = c.messages.indexOf(m);
  if (at < 0 || running(c)) return;
  const original = { role: "assistant", content: m.content, reasoning_content: m.reasoning ?? "" };
  const final = !pick ? original : pick.ch === "content" ? { ...original, content: pick.before } : { role: "assistant", content: "", reasoning_content: pick.before };
  const effort = m.effort ?? c.effort;
  c.messages.splice(at + 1);
  m.error = m.finish = undefined;
  m.stopped = false;
  m.params = changed(effort === "off" ? "fast" : "think") || undefined;
  const extra = pick ? { continue_final_message: true, kiln_original: original, kiln_append_token: pick.alt.id } : { continue_final_message: true };
  const upto = c.messages.slice(0, at);
  void startRun(c, m, () => request(c, upto, { effort, final, extra }),
    pick && { t: pick.alt.t, p: pick.alt.p, alts: tokens.get(m.id!)?.[pick.ch][pick.i]?.alts ?? [] });
}

// body: the request, made once the conversation's files are read (a file that is gone ends the reply with an error)
async function startRun(c: Conversation, m: Message, body: () => object, pick?: Tok) {
  const phase = m.content ? "writing" : m.reasoning ? "thinking" : "waiting";
  const run: Run = { conv: c, m, live: { phase, since: performance.now(), note: "Reading the prompt" }, at: 0, stopping: false, frame: 0, saved: 0, calls: [], pick };
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
    await filesReady(c.messages);
    const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body()) });
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

async function resume(c: Conversation, m: Message) {
  const run: Run = { conv: c, m, live: { phase: "waiting", since: performance.now(), note: "Reading the prompt" }, at: 0, stopping: false, frame: 0, saved: 0, calls: [] };
  runs.set(c.id, run);
  try {
    const res = await fetch(`/api/chat/${m.job}?from=0`);
    if (res.status === 404) throw new Error(CUT_OFF);
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    m.content = m.reasoning = "";
    if (m.id) tokens.delete(m.id);                 // the job's events bring them back
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
  const { m, live } = run, choice = ev.choices?.[0], d = choice?.delta ?? {}, now = performance.now(), v = views.get(m);
  if (ev.error) m.error = ev.error.message;
  if (choice?.finish_reason) m.finish = choice.finish_reason;
  if (d.kiln_prefix) {                             // a continued reply as it now stands
    m.reasoning = d.kiln_prefix.reasoning_content ?? "";
    m.content = d.kiln_prefix.content ?? "";
    keepTokens(m, run.pick);
    if (v) v.tok = undefined;
  }
  if (choice?.logprobs && m.id) {
    const td = tokens.get(m.id) ?? { reasoning: [], content: [] };
    tokens.set(m.id, td);
    const tok = (e: any): Tok => ({ t: e.token, p: Math.exp(e.logprob), alts: (e.top_logprobs ?? []).map((a: any) => ({ t: a.token, p: Math.exp(a.logprob), id: a.id })) });
    for (const e of choice.logprobs.reasoning_content ?? []) td.reasoning.push(tok(e));
    for (const e of choice.logprobs.content ?? []) td.content.push(tok(e));
  }
  if (ev.stopped) m.stopped = true;
  if (ev.timing) run.timing = ev.timing;
  if (ev.usage) run.usage = ev.usage;
  if (ev.seed !== undefined) run.seed = ev.seed;
  for (const c of d.tool_calls ?? []) if (c.function?.name) run.calls.push(c.function.name);
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

// After kiln restates a continued reply: keep the tokens it still starts with; the rest (a picked alternative) stands
// in as one token.
function keepTokens(m: Message, pick?: Tok) {
  const td = m.id && tokens.get(m.id);
  if (!td) return;
  for (const ch of ["reasoning", "content"] as const) {
    const text = m[ch] ?? "", list = td[ch];
    let n = 0, len = 0;
    while (n < list.length && text.startsWith(list[n].t, len)) len += list[n++].t.length;
    list.length = n;
    if (len < text.length) list.push({ t: text.slice(len), p: pick?.p ?? 1, alts: pick?.alts ?? [] });
  }
}

function schedule(run: Run) {
  run.frame ||= requestAnimationFrame(() => {
    run.frame = 0;
    const v = views.get(run.m);
    if (!v) return;                                // its chat is not on screen
    paint(v, run.m, run.live);
    keepUp();
  });
}

// Following a reply: the newest text stays in view until the user scrolls up (wheel, touch, keys or the scrollbar; a
// scroll the page makes itself never counts), and again once they are back at the end. Following only scrolls down,
// so it never takes the page from under someone reading. A figure drawing at the end is shown from its top and the
// page holds still while it draws (its size is set when it starts).
function followTarget() {
  const end = scroller.scrollHeight - scroller.clientHeight;
  const fig = thread.querySelector<HTMLElement>(".msg:last-child .svg-fig.drawing");
  if (!fig) return end;
  return Math.min(end, fig.getBoundingClientRect().top - scroller.getBoundingClientRect().top + scroller.scrollTop - 8);
}
function keepUp() {
  if (follow) {
    const t = followTarget();
    if (scroller.scrollTop < t - 1) scroller.scrollTop = t;
  }
  $("to-latest").hidden = follow || scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
}
function stopFollowing() {
  follow = false;
  keepUp();
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
  if (run.calls.length && !m.error)                // shown, not dropped: the reply would otherwise just stop
    m.error = `The model asked to run ${[...new Set(run.calls)].join(", ")}, but this chat has no tools, so the reply ends here.`;
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
        { role: "user", content: (c.messages[0].content || (c.messages[0].files ?? []).map((f) => f.name).join(", ")).slice(0, 2000) },
      ] }),
    });
    const title: string | undefined = (await res.json()).choices?.[0]?.message?.content?.split("\n")[0]?.replace(/^[\s#*"']+|[\s*"'.]+$/g, "");
    if (title) c.title = title.slice(0, 80);
  } catch { /* keep the first line of the question */ }
  save(c);
  if (c === conv) $("title").textContent = document.title = c.title;
}

function submit(text: string) {
  text = text.trim();
  if ((!text && !pending.length) || running() || drafting) return;
  input.value = "";
  grow();
  const files = pending;
  pending = [];
  renderPending();
  if (!conv.messages.length) conv.title = (text || files.map((f) => f.name).join(", ")).split("\n")[0].slice(0, 60);
  conv.messages.push({ role: "user", content: text, ...(files.length ? { files } : {}) });
  void generate();
}

// Picked with the clip, dropped anywhere on the page or pasted: shown as chips in the message box until it is sent,
// then above the message (pictures as pictures). kiln is sent each file before the text that was typed.
let pending: Attachment[] = [];
const bodies = new Map<string, string>();               // a file's text, or a picture's data URL, ready for requests
const thumbs = new Map<string, Promise<string>>();      // a picture's object URL, for showing it

const thumb = (a: Attachment) => {
  let u = thumbs.get(a.id);
  if (!u) thumbs.set(a.id, (u = getFile(a.id).then((b) => (b instanceof Blob ? URL.createObjectURL(b) : ""))));
  return u;
};

function fileChip(a: Attachment, remove?: () => void): HTMLElement {
  const el = h("span", "att"), text = h("span", "att-text");
  if (a.kind === "image") {
    const img = h("img");
    img.alt = "";
    void thumb(a).then((u) => { img.src = u; });
    el.append(img);
  } else el.append(h("span", "att-icon", (a.name.includes(".") ? a.name.split(".").pop()! : "txt").slice(0, 4)));
  text.append(h("span", "att-name", a.name), h("span", "att-size", fmtBytes(a.size)));
  el.title = a.name;
  el.append(text);
  if (remove) {
    const x = h("button", "act att-x");
    x.type = "button";
    x.innerHTML = icon("x");
    x.title = x.ariaLabel = `Remove ${a.name}`;
    x.onclick = remove;
    el.append(x);
  }
  return el;
}

function filesView(files: Attachment[]) {
  const box = h("div", "msg-files");
  for (const a of files) {
    if (a.kind !== "image") { box.append(fileChip(a)); continue; }
    const img = h("img");
    img.alt = img.title = a.name;
    img.dataset.zoom = a.id;
    void thumb(a).then((u) => { img.src = u; });
    box.append(img);
  }
  return box;
}

function renderPending() {
  const box = $("attachments");
  box.hidden = !pending.length;
  box.replaceChildren(...pending.map((a) => fileChip(a, () => { pending = pending.filter((x) => x !== a); renderPending(); input.focus(); })));
}

async function addFiles(list: Iterable<File>) {
  for (const f of list) {
    const r = await attach(f);
    if (typeof r === "string") toast(r);
    else pending.push(r);
  }
  renderPending();
  input.focus();
}

async function filesReady(msgs: Message[]) {
  for (const a of msgs.flatMap((m) => m.files ?? [])) {
    if (bodies.has(a.id)) continue;
    const data = await getFile(a.id);
    if (data === undefined) throw new Error(`${a.name} is no longer in this tab (closed tabs' files are deleted): attach it again`);
    bodies.set(a.id, typeof data === "string" ? data : await dataUrl(data));
  }
}

// A user message as kiln gets it: its files (pictures as images, PDFs as files, videos as videos, texts as fenced
// blocks), then the text typed.
function userContent(m: Message): string | object[] {
  if (!m.files?.length) return m.content;
  const parts: object[] = m.files.map((a) => {
    const body = bodies.get(a.id) ?? "";
    if (a.kind === "image") return { type: "image_url", image_url: { url: body } };
    if (a.kind === "pdf") return { type: "file", file: { file_data: body, filename: a.name } };
    if (a.kind === "video") return { type: "video_url", video_url: { url: body } };
    return { type: "text", text: `${textBlock(a.name, body)}\n\n` };
  });
  if (m.content) parts.push({ type: "text", text: m.content });
  return parts;
}

// Chats live and die with the tab; a file keeps one. Export writes the open chat (or all of them) as JSON, Import adds
// chats from such files under new ids, so nothing already here is replaced. Token odds stay behind (this page only).
interface ChatFile { kiln: "chats"; version: 1; exported: string; conversations: Conversation[]; files?: Record<string, string> }

function download(name: string, data: string, type: string) {
  const a = h("a");
  a.href = URL.createObjectURL(new Blob([data], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// Attached files go along: texts as they are, pictures as data URLs.
async function exportChats(list: Conversation[], name: string) {
  const files: Record<string, string> = {};
  for (const a of list.flatMap((c) => c.messages.flatMap((m) => m.files ?? []))) {
    const data = await getFile(a.id);
    if (data !== undefined) files[a.id] = typeof data === "string" ? data : await dataUrl(data);
  }
  const file: ChatFile = { kiln: "chats", version: 1, exported: new Date().toISOString(),
    conversations: list.map((c) => ({ ...structuredClone(c), messages: c.messages.map(({ job, ...m }) => m) })), files };
  download(name, JSON.stringify(file, null, 1), "application/json");
}

const blobOf = (url: string) => {                       // a data URL's bytes (fetch would need connect-src data:)
  const [head, b64] = url.split(",", 2), bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));
  return new Blob([bytes], { type: head.slice(5).split(";")[0] });
};
const fileName = (title: string) => `${title.replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "chat"}.kiln.json`;

// One chat from a file, or null if it is not one: roles and text only where kiln expects them.
function importedChat(x: any): Conversation | null {
  if (!x || typeof x !== "object" || !Array.isArray(x.messages)) return null;
  const messages: Message[] = [];
  for (const m of x.messages) {
    if (!m || (m.role !== "user" && m.role !== "assistant") || typeof m.content !== "string") return null;
    const { job, ...rest } = m;
    if (rest.files !== undefined && !(Array.isArray(rest.files) && rest.files.every((a: any) => typeof a?.id === "string"
      && typeof a.name === "string" && KINDS.includes(a.kind)))) return null;
    messages.push(rest);
  }
  if (!messages.length) return null;
  return { id: randomId(), title: typeof x.title === "string" && x.title ? x.title.slice(0, 80) : "Imported chat", named: true,
    updated: Date.now(), effort: EFFORTS.includes(x.effort) ? x.effort : "xhigh", messages };
}

async function importFiles(files: FileList) {
  let added: Conversation[] = [], bad = 0;
  for (const f of files) {
    try {
      const data = JSON.parse(await f.text());
      for (const [id, body] of Object.entries(data?.files ?? {}))
        if (typeof body === "string") await putFile(id, body.startsWith("data:image/") ? blobOf(body) : body);
      const list = (data?.kiln === "chats" && Array.isArray(data.conversations) ? data.conversations : [data]).map(importedChat);
      bad += list.filter((c: Conversation | null) => !c).length;
      added = added.concat(list.filter(Boolean));
    } catch { bad++; }
  }
  for (const c of added) save(c);
  if (added.length) openConversation(added.at(-1)!.id);
  toast(added.length ? `Imported ${added.length} chat${added.length > 1 ? "s" : ""}${bad ? ` (${bad} not a kiln chat)` : ""}`
    : "Nothing to import: not a chat exported from kiln");
}

// kiln writes the message the user would likely send next, into the composer to edit and send. It is asked as the
// assistant (Qwen will not write a user turn: at one it just ends it), without thinking, from the conversation
// kiln has cached, which it leaves as it was (kiln_aside). Text already in the composer is how the draft begins.
const DRAFT_ASK = "Write the next message I would send in this conversation, as me: same voice, same goals, natural and brief. Reply with only that message.";
let drafting: { job?: string; stop: boolean } | null = null;

async function draft() {
  if (drafting) {
    drafting.stop = true;
    if (drafting.job) void fetch(`/api/chat/${drafting.job}/stop`, { method: "POST" }).catch(() => {});
    return;
  }
  const c = conv, begin = input.value;
  if (running(c) || !c.messages.length || status.state !== "online") return;
  const d: { job?: string; stop: boolean } = (drafting = { stop: false });
  try { await filesReady(c.messages); } catch (e) { drafting = null; return toast(`The draft failed: ${e instanceof Error ? e.message : e}`); }
  const body = request(c, c.messages, {
    sampling: "fast", final: begin.trim() ? { role: "assistant", content: begin } : undefined,
    extra: { kiln_aside: true, max_tokens: 400, logprobs: false, seed: undefined, ...(begin.trim() ? { continue_final_message: true } : {}) },
  });
  (body.messages as object[]).splice(begin.trim() ? -1 : body.messages.length, 0, { role: "user", content: DRAFT_ASK });
  input.readOnly = true;
  renderComposer();
  try {
    const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok || !res.body) throw new Error((await res.json().catch(() => null))?.error?.message ?? `HTTP ${res.status}`);
    d.job = res.headers.get("X-Kiln-Job") ?? undefined;
    if (d.stop && d.job) void fetch(`/api/chat/${d.job}/stop`, { method: "POST" }).catch(() => {});
    let text = begin;
    for await (const ev of events(res.body)) {
      if (ev === DONE) break;
      if (ev.error) throw new Error(ev.error.message);
      const delta = ev.choices?.[0]?.delta ?? {};
      if (delta.kiln_prefix) text = delta.kiln_prefix.content ?? text;
      if (delta.content) text += delta.content;
      input.value = text;
      grow();
    }
  } catch (e) {
    if (!d.stop) toast(`The draft failed: ${e instanceof Error ? e.message : e}`);
  } finally {
    drafting = null;
    input.readOnly = false;
    renderComposer();
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }
}

function openConversation(id: string) {
  conv = runs.get(id)?.conv ?? (chats[id] ? structuredClone(chats[id]) : blank());   // a reply being written keeps going
  closeOver(panes.conversations);
  render();
  scroller.scrollTop = scroller.scrollHeight;
  follow = true;
  keepUp();
}

function newChat() {
  conv = blank();
  history.replaceState(null, "", location.pathname);
  closeOver(panes.conversations);
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
// a page at full size takes clicks: only one beside it closes it (or Esc)
$("page-zoom").addEventListener("click", (e) => { if (e.target === e.currentTarget) (e.currentTarget as HTMLDialogElement).close(); });
$("page-zoom").addEventListener("close", () => $("page-zoom").querySelector(".page-zoom-box")!.replaceChildren());

function grow() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 240)}px`;
}

thread.addEventListener("click", (e) => {
  const t = e.target as HTMLElement, code = t.closest<HTMLElement>(".copy"), act = t.closest<HTMLElement>("[data-act]");
  const tok = t.closest<HTMLElement>(".tok");
  if (tok) return void openTokPop(tok);
  if (code) return void copy(code.closest(".code")?.querySelector("code")?.textContent ?? "", code);
  const svg = t.closest<HTMLElement>("[data-svg]");
  if (svg) return void svgAction(svg);
  const page = t.closest<HTMLElement>("[data-page]");
  if (page) return void pageAction(page);
  if (t instanceof HTMLImageElement && t.dataset.zoom) {
    const big = h("img");
    big.src = t.src;
    big.alt = t.alt;
    $("svg-zoom").querySelector(".svg-zoom-paper")!.replaceChildren(big);
    return void $<HTMLDialogElement>("svg-zoom").showModal();
  }
  if (!act) return;
  const i = Number(act.closest<HTMLElement>("article")?.dataset.i), m = conv.messages[i];
  if (act.dataset.act === "copy") return void copy(m.content, act);
  if (act.dataset.act === "seed") return void useSeed(Number(act.dataset.seed));
  if (act.dataset.act === "tokens" && m.id) {
    if (!tokenView.delete(m.id)) tokenView.add(m.id);
    return render();
  }
  if (running()) return;
  if (act.dataset.act === "retry") {
    conv.messages.splice(i);
    void generate();
  } else if (act.dataset.act === "cont") continueReply(m);
  else if (act.dataset.act === "trash") deleteMessage(i);
  else if (act.dataset.act === "edit") {
    editing = m;
    render();
  }
});
document.addEventListener("pointerdown", (e) => {
  if (!$("tokpop").hidden && !(e.target as Element).closest("#tokpop, .tok")) closeTokPop();
});
scroller.addEventListener("scroll", () => { if (!$("tokpop").hidden) closeTokPop(); }, { passive: true });
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const r = running();
  if (r) stopRun(r);
  else submit(input.value);
});
input.addEventListener("input", () => { grow(); if (!drafting) $("draft").title = input.value.trim() ? "Finish my message (draft)" : "Draft my next message"; });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    submit(input.value);
  }
});
// Ctrl+B and Ctrl+, toggle the side panels. Esc stops this chat's reply, unless it is closing something: a menu, a
// picture at full size, a panel over the chat.
document.addEventListener("keydown", (e) => {
  const keyed = (e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && PANES.find((p) => e.key.toLowerCase() === p.key);
  if (keyed) {
    e.preventDefault();
    return setPane(keyed, !isOpen(keyed));
  }
  if (e.key !== "Escape" || e.defaultPrevented) return;
  if (!$("tokpop").hidden) return closeTokPop();
  if (modelsOpen) return setModelsOpen(false, true);
  if (document.querySelector("dialog[open]")) return;
  const over = PANES.find((p) => isOpen(p) && !p.beside.matches);
  if (over) return setPane(over, false);
  const r = running();
  if (r) stopRun(r);
});
// Back at the end by scrolling down to it: following again. Only a downward scroll counts: the first pixels of an
// upward one (a trackpad's small steps, a wheel's smooth scroll) are still at the end, and taking those as a return
// snapped the page back under the user's hand.
let lastTop = 0;
scroller.addEventListener("scroll", () => {
  const down = scroller.scrollTop > lastTop + 0.5;
  lastTop = scroller.scrollTop;
  if (!follow && down && scroller.scrollTop >= followTarget() - 2) follow = true;
  keepUp();
});
scroller.addEventListener("wheel", (e) => { if (e.deltaY < 0) stopFollowing(); }, { passive: true });
let touchY = 0;
scroller.addEventListener("touchstart", (e) => { touchY = e.touches[0].clientY; }, { passive: true });
scroller.addEventListener("touchmove", (e) => {
  if (e.touches[0].clientY > touchY + 2) stopFollowing();                  // the finger moves down: the page up
  touchY = e.touches[0].clientY;
}, { passive: true });
scroller.addEventListener("pointerdown", (e) => { if (e.target === scroller && e.offsetX >= scroller.clientWidth) stopFollowing(); });   // its scrollbar
document.addEventListener("keydown", (e) => {
  if (["PageUp", "ArrowUp", "Home"].includes(e.key) && !(e.target as Element).closest?.("input, textarea, select, [contenteditable]")) stopFollowing();
});
$("to-latest").addEventListener("click", () => {
  follow = true;
  scroller.scrollTo({ top: followTarget(), behavior: reducedMotion.matches ? "auto" : "smooth" });
});
window.addEventListener("hashchange", () => {
  const id = location.hash.slice(1);
  if (id && id !== conv.id) openConversation(id);
});
$("new-chat").addEventListener("click", newChat);
$("export").addEventListener("click", () => void exportChats([conv], fileName(conv.title)));
$("export-all").addEventListener("click", () => {
  const all = Object.values(chats).sort((a, b) => b.updated - a.updated);
  if (all.length) void exportChats(all, `kiln-chats-${new Date().toISOString().slice(0, 10)}.json`);
  else toast("No chats to export yet");
});
$("attach").addEventListener("click", () => $("attach-file").click());
$("attach-file").addEventListener("change", (e) => {
  const picker = e.currentTarget as HTMLInputElement;
  if (picker.files?.length) void addFiles(picker.files).finally(() => { picker.value = ""; });
});
input.addEventListener("paste", (e) => {
  const files = [...(e.clipboardData?.files ?? [])];
  if (!files.length) return;
  e.preventDefault();
  void addFiles(files);
});
// Files dragged over the page light up the message box; dropped anywhere, they are attached.
let dragDepth = 0;
const dragsFiles = (e: DragEvent) => !!e.dataTransfer?.types.includes("Files");
document.addEventListener("dragenter", (e) => { if (dragsFiles(e) && dragDepth++ === 0) document.querySelector(".composer-box")!.classList.add("dropping"); });
document.addEventListener("dragleave", (e) => { if (dragsFiles(e) && --dragDepth === 0) document.querySelector(".composer-box")!.classList.remove("dropping"); });
document.addEventListener("dragover", (e) => { if (dragsFiles(e)) e.preventDefault(); });
document.addEventListener("drop", (e) => {
  if (!dragsFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  document.querySelector(".composer-box")!.classList.remove("dropping");
  void addFiles(e.dataTransfer!.files);
});
$("import").addEventListener("click", () => $("import-file").click());
$("import-file").addEventListener("change", (e) => {
  const input = e.currentTarget as HTMLInputElement;
  if (input.files?.length) void importFiles(input.files).finally(() => { input.value = ""; });
});

// The conversations (left) and Settings (right) work alike: a toggle fixed in the panel's top corner (it never moves)
// and a key. Where the window has room a panel sits beside the chat (the chat glides to its new place), its inner
// edge resizes it, and it stays as it was left, width and all; with too little room it slides over the chat, starts
// closed, and a click outside it or Esc closes it. One panel over the chat at a time.
const MAIN_MIN = 640;                                    // the chat beside the panels keeps this much (its message box fits)
const isOpen = (p: Pane) => p.el.hasAttribute("data-open");
const besideOpen = (p: Pane) => isOpen(p) && p.beside.matches;
// Changes to the layout while `still` > 0 (a drag, a window being resized, a state restored) take effect at once
// rather than easing.
let still = 0;
function hush() {
  still++;
  document.documentElement.classList.add("no-anim");
}
function unhush() {
  requestAnimationFrame(() => requestAnimationFrame(() => { if (--still === 0) document.documentElement.classList.remove("no-anim"); }));
}
// Each panel's width: the one it was given, as far as the window allows. The chat keeps MAIN_MIN, and Settings gives
// way to the conversations (down to its minimum) before they give way to it.
function fitPanes() {
  const { conversations: c, settings: s } = panes, room = innerWidth - MAIN_MIN;
  c.width = Math.round(Math.max(c.min, Math.min(c.preferred, room - (besideOpen(s) ? s.min : 0))));
  s.width = Math.round(Math.max(s.min, Math.min(s.preferred, room - (besideOpen(c) ? c.width : 0))));
  for (const p of PANES) document.documentElement.style.setProperty(p.widthVar, `${p.width}px`);
}
function setPane(p: Pane, open: boolean, { focus = true, animate = true } = {}) {
  if (!animate) hush();
  p.el.toggleAttribute("data-open", open);
  p.el.toggleAttribute("data-over", !p.beside.matches);
  p.el.inert = !open;
  p.toggle.setAttribute("aria-expanded", String(open));
  if (p.beside.matches) setPref(`pane-${p.name}`, open ? "open" : "closed");
  if (open && !p.beside.matches) for (const q of PANES) if (q !== p) closeOver(q);
  fitPanes();                                            // the other panel may have more room now, or less
  if (open) {
    p.onOpen?.();
    if (focus && !p.beside.matches) p.el.focus({ preventScroll: true });
  } else if (p.el.contains(document.activeElement)) p.toggle.focus();
  if (!animate) unhush();
}
function closeOver(p: Pane) {
  if (isOpen(p) && !p.beside.matches) setPane(p, false, { focus: false });
}
// as it was left (beside the chat) or closed (over it), at the width it was given, without sliding there
function restorePane(p: Pane) {
  const saved = pref(`pane-${p.name}`, "");
  p.preferred = Number(pref(`pane-${p.name}-width`, "")) || p.def;
  setPane(p, p.beside.matches && (saved ? saved === "open" : p.byDefault), { focus: false, animate: false });
}
// Dragging the inner edge resizes the panel: it follows the pointer (no easing), within the window's limits, and the
// width it ends at is kept. A press that does not move (2 px) changes nothing; a double click gives the panel its
// default width back, eased.
function resizeWith(p: Pane, rail: HTMLElement) {
  const dir = p.el.dataset.side === "left" ? 1 : -1;
  rail.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || !besideOpen(p)) return;
    e.preventDefault();
    rail.setPointerCapture(e.pointerId);
    const x0 = e.clientX, w0 = p.width;
    let x = x0, frame = 0, moved = false;
    const apply = () => {
      frame = 0;
      if (!moved) return;
      p.preferred = w0 + (x - x0) * dir;
      fitPanes();
    };
    const move = (ev: PointerEvent) => {
      x = ev.clientX;
      if (!moved && Math.abs(x - x0) > 2) {
        moved = true;
        hush();
      }
      frame ||= requestAnimationFrame(apply);
    };
    const up = (ev: PointerEvent) => { x = ev.clientX; };
    const end = () => {                                  // capture ends on pointerup and on pointercancel
      cancelAnimationFrame(frame);
      apply();
      rail.classList.remove("dragging");
      document.documentElement.classList.remove("pane-resizing");
      rail.removeEventListener("pointermove", move);
      rail.removeEventListener("pointerup", up);
      rail.removeEventListener("lostpointercapture", end);
      if (!moved) return;
      p.preferred = p.width;
      setPref(`pane-${p.name}-width`, String(p.width));
      unhush();
    };
    rail.classList.add("dragging");
    document.documentElement.classList.add("pane-resizing");
    rail.addEventListener("pointermove", move);
    rail.addEventListener("pointerup", up);
    rail.addEventListener("lostpointercapture", end);
  });
  rail.addEventListener("dblclick", () => {
    p.preferred = p.def;
    setPref(`pane-${p.name}-width`, "");
    fitPanes();
  });
}
for (const p of PANES) {
  const label = p.toggle.getAttribute("aria-label")!;
  p.toggle.innerHTML = icon(p.el.dataset.side === "left" ? "paneLeft" : "paneRight");
  p.toggle.title = `${label} (Ctrl+${p.key.toUpperCase()})`;
  p.toggle.addEventListener("click", () => setPane(p, !isOpen(p)));
  const rail = p.el.querySelector<HTMLElement>(".pane-rail")!;
  rail.title = "Drag to resize · double-click for the default width";
  rail.setAttribute("aria-label", `Resize ${label.toLowerCase()}`);
  resizeWith(p, rail);
  p.beside.addEventListener("change", () => restorePane(p));
}
addEventListener("resize", () => { hush(); fitPanes(); unhush(); });
// Over the chat, a click outside the panel closes it (and goes on to do what it was for), except on a toast (its Undo).
document.addEventListener("pointerdown", (e) => {
  const t = e.target as Node;
  for (const p of PANES) if (!p.el.contains(t) && !p.toggle.contains(t) && !$("toast").contains(t)) closeOver(p);
});
// Thinking is an energy bar of four stops, Off to Max, always in view: it runs hotter to the right (Medium flows; Max
// is molten: it flows faster, a shine sweeps it, sparks rise from it). The model's name beside the send button opens
// the models, a flat list; it is the only place a model is chosen (the sidebar shows the engine's state).
let modelsOpen = false;

function buildControls() {
  const energy = $("energy"), seeded = (i: number) => ((Math.sin(i * 12.9898) * 43758.5453) % 1 + 1) % 1;
  const sparks = Array.from({ length: 12 }, (_, i) =>
    `<b class="spark" style="--x:${(4 + seeded(i) * 92).toFixed(1)}%;--d:${(1.1 + seeded(i + 40) * 1.2).toFixed(2)}s;--delay:-${(seeded(i + 80) * 2.4).toFixed(2)}s;--dx:${((seeded(i + 120) - 0.5) * 8).toFixed(1)}px;--w:${seeded(i + 160) > 0.7 ? 2.5 : 1.5}px"></b>`).join("");
  energy.innerHTML = `<span class="energy-track"><span class="energy-fill"><i class="energy-shine"></i>${sparks}</span>` +
    EFFORTS.map((_, k) => `<i class="energy-dot" style="--k:${k}"></i>`).join("") + `</span><span class="energy-thumb"></span>`;
  $("energy-stops").innerHTML = EFFORTS.map((e, k) => `<button type="button" tabindex="-1" data-stop="${k}" style="--k:${k}">${LEVEL[e][0]}</button>`).join("");
  $("energy-stops").addEventListener("click", (e) => {
    const b = (e.target as Element).closest<HTMLElement>("[data-stop]");
    if (b) setEffort(Number(b.dataset.stop));
  });
  // A third of the way to a stop takes it (the thumb leads the hand), and it snaps there in under 0.1 s.
  const stopAt = (x: number, dir: number) => {
    const r = energy.getBoundingClientRect(), t = energy.querySelector<HTMLElement>(".energy-thumb")!.offsetWidth;
    return Math.min(3, Math.max(0, Math.round(((x - r.left - t / 2) / (r.width - t)) * 3 + 0.17 * dir)));
  };
  let lastX = 0;
  energy.addEventListener("pointerdown", (e) => {
    energy.setPointerCapture(e.pointerId);
    energy.classList.add("dragging");
    lastX = e.clientX;
    setEffort(stopAt(e.clientX, 0));
  });
  energy.addEventListener("pointermove", (e) => {
    if (!energy.hasPointerCapture(e.pointerId)) return;
    const dir = Math.sign(e.clientX - lastX);
    lastX = e.clientX;
    if (dir) setEffort(stopAt(e.clientX, dir));
  });
  for (const end of ["pointerup", "pointercancel"]) energy.addEventListener(end, () => energy.classList.remove("dragging"));
  energy.addEventListener("keydown", (e) => {
    const at = EFFORTS.indexOf(conv.effort), step = ({ ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 } as Record<string, number>)[e.key];
    if (step) setEffort(at + step);
    else if (e.key === "Home" || e.key === "End") setEffort(e.key === "Home" ? 0 : 3);
    else return;
    e.preventDefault();
  });
  $("model-btn").addEventListener("click", () => setModelsOpen(!modelsOpen));
  $("power").addEventListener("click", async () => {
    if (status.state === "online") await fetch("/api/kiln/stop", { method: "POST" });
    else await start();
    await refreshStatus();
  });
  document.addEventListener("pointerdown", (e) => {
    if (modelsOpen && !(e.target as Element).closest("#model-pop, #model-btn")) setModelsOpen(false);
  });
}

function setEffort(i: number) {
  const e = EFFORTS[Math.min(3, Math.max(0, i))];
  if (e === conv.effort) return;
  conv.effort = e;
  setPref("effort", e);
  renderComposer();
}

function setModelsOpen(open: boolean, refocus = false) {
  modelsOpen = open;
  $("model-pop").hidden = !open;
  delete $("model-pop").dataset.key;
  renderControls();
  if (open) $("model-pop").querySelector<HTMLElement>(".current")?.focus();
  else if (refocus) $("model-btn").focus();
}

const shortName = (id: string) => modelName(id).replace(/^Qwen3\.8-/, "");
function renderControls() {
  const at = EFFORTS.indexOf(conv.effort), energy = $("energy");
  energy.dataset.heat = HEAT[at];
  energy.style.setProperty("--v", String(at));
  energy.setAttribute("aria-valuenow", String(at));
  energy.setAttribute("aria-valuetext", LEVEL[conv.effort][0]);
  energy.title = `Thinking: ${LEVEL[conv.effort][0]}`;
  energy.querySelectorAll(".energy-dot").forEach((d, k) => d.classList.toggle("lit", at > 0 && k <= at));   // the stops inside the fill
  $("energy-stops").dataset.heat = HEAT[at];
  $("energy-stops").querySelectorAll("button").forEach((b, k) => b.classList.toggle("on", k === at));
  // the model, and the engine's state: a dot (ready, loading, stopped), the load's progress while it switches
  const state = status.loading || status.state === "loading" ? "loading" : status.state;
  const pct = status.progress == null ? "" : `${Math.round(status.progress * 100)}%`, name = modelName(current());
  const btn = $("model-btn");
  btn.innerHTML = `<span class="dot ${state}"></span>${shortName(current())}${state === "loading" && pct ? `<span class="model-pct">${pct}</span>` : ""}${icon("chevron")}`;
  btn.title = state === "online" ? `${name}: ready` : state === "loading" ? `Loading ${name}` : `${name}: stopped`;
  btn.setAttribute("aria-expanded", String(modelsOpen));
  btn.classList.toggle("open", modelsOpen);
  // usage, in view: this chat's length against the context, and how full kiln's cache of conversations is
  const used = status.kv_tokens && status.cache ? 1 - status.cache.free_kv_tokens / status.kv_tokens : 0, ctx = fmtK(status.context ?? 262144);
  const s = conv.messages.findLast((m) => m.stats)?.stats, meter = $("meter");
  meter.textContent = [s ? `${fmtK(s.prompt + s.tokens)} / ${ctx} context` : state === "online" ? `${ctx} context` : "",
    state === "online" ? `cache ${Math.round(used * 100)}%` : ""].filter(Boolean).join(" · ");
  meter.title = state === "online" ? `${fmtK(status.cache?.free_kv_tokens ?? 0)} of ${fmtK(status.kv_tokens ?? 0)} KV tokens free (cached conversations are dropped as needed)` : "";
  // power: unload the model, or load it when it is stopped
  const power = $<HTMLButtonElement>("power");
  power.innerHTML = icon("power");
  power.title = power.ariaLabel = state === "online" ? `Unload ${name}` : state === "loading" ? `Loading ${name}` : `Load ${name}`;
  power.disabled = state === "loading";
  if (!modelsOpen) return;
  // the list (redrawn only when something in it changed, so a poll never takes the focus from it)
  const pop = $("model-pop");
  const key = JSON.stringify([current(), state, pct]);
  if (pop.dataset.key === key) return;
  pop.dataset.key = key;
  const rows = MODELS.map(([id, label]) => {
    const on = id === current(), item = h("button", on ? "model-item current" : "model-item");
    item.type = "button";
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", String(on));
    const mark = h("span", "model-check");
    if (on && state === "loading") mark.textContent = pct || "…";
    else if (on) mark.innerHTML = icon("check");
    item.append(h("span", "model-label", label), mark);
    item.onclick = () => {
      if (!on || state === "offline") { picked = id; setPref("model", id); void (state === "offline" ? start().then(refreshStatus) : pickModel(id)); }
      setModelsOpen(false, true);
    };
    return item;
  });
  pop.replaceChildren(...rows);
}

// A panel beside the chat (it opens without moving the chat when there is room). Every change applies to the next
// reply and can be undone; each value shows what kiln is sent, marked where it differs from the recommended one.

const modeOf = (c: Conversation): Mode => (c.effort === "off" ? "fast" : "think");
let chatMode: Mode | undefined;                     // the Think setting Settings last showed
const shownMode = (): Mode => modeOf(conv);             // the values Settings shows: those of the chat's Think setting
let systemEdited = false;

function loadSettings(): Settings {
  const s = load<Settings | null>("settings2", null);
  if (s) return s;
  // the earlier shape: one set of values for both modes, and a system prompt added after a built-in line
  const old = load<{ system?: string; sampling?: Record<string, number> } | null>("settings", null);
  const { max_tokens, seed, ...rest } = old?.sampling ?? {};
  return { system: [DEFAULT_SYSTEM, old?.system?.trim()].filter(Boolean).join("\n\n"), think: { ...rest }, fast: { ...rest }, max_tokens, seed };
}
function saveSettings() {
  store("settings2", settings);
  syncSettings();
}
const value = (mode: Mode, key: SamplerKey) => settings[mode][key] ?? RECOMMENDED[mode][key];
const orderOf = (mode: Mode) => settings[mode].order ?? STAGES;
const fmtValue = (key: SamplerKey, v: number) => SAMPLERS[key].step >= 1 ? String(v) : v.toFixed(2);
// The sampling fields of a request for a mode: only what differs from the recommended values.
function samplerFields(mode: Mode) {
  const { order, ...rest } = settings[mode];
  return { ...rest, ...(order ? { samplers: order } : {}) };
}
// What a reply was drawn with, where it differs from the recommended values (its footer shows it).
function changed(mode: Mode) {
  const out = (Object.keys(SAMPLERS) as SamplerKey[]).filter((k) => settings[mode][k] !== undefined)
    .map((k) => `${SAMPLERS[k].short} ${fmtValue(k, settings[mode][k]!)}`);
  if (settings[mode].order) out.push(settings[mode].order!.map((s) => SAMPLERS[s].short).join("›"));
  if (settings.max_tokens) out.push(`max ${fmtK(settings.max_tokens)}`);
  return out.join(" · ");
}

let toastTimer = 0;
function toast(text: string, undo?: () => void) {
  const t = $("toast"), b = h("button", "btn", "Undo");
  b.type = "button";
  b.onclick = () => { t.hidden = true; undo?.(); };
  t.replaceChildren(h("span", "", text), ...(undo ? [b] : []));
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { t.hidden = true; }, 6000);
}

function useSeed(seed: number) {
  const before = settings.seed;
  settings.seed = seed;
  saveSettings();
  toast(`Seed ${seed} is fixed for the next replies`, () => { settings.seed = before; saveSettings(); });
}

// Two real moments of Qwen3.8-Flash-Next run through kiln's sampler (make_dist in engine/sampler.h) with the settings
// as they stand: the first name of a reply (what the stages do) and the next name of a list that has started to repeat
// itself (what presence and DRY do). Each sits above the values it shows.

const FIRST: Cand[] = ODDS.map(([word, c], i) => ({ word, logit: Math.log(c), id: -1 - i }));
const NEXT: Cand[] = LOOP.cands.map(([t, lp, id]) => ({ word: t.trim(), logit: lp, id }));

const pct = (p: number) => (p >= 0.995 ? "100%" : p >= 0.01 ? `${Math.round(p * 100)}%` : p > 0 ? "<1%" : "");
const CUT_BY: Record<Stage, string> = { top_k: "top-k", top_p: "top-p", min_p: "min-p", temperature: "temp 0" };

// A chart: a row per named candidate (the rest share one): the name, a bar for its odds now with a tick where the
// model had them, and its share, or what cut it (struck through). Min P is a line across the bars. Pointing at a row
// tells it in the line below; nothing changes size.
function chart(fig: HTMLElement, shown: number[], tagOf: (r: Odds) => string) {
  const box = h("div", "odd-rows"), min = h("div", "odd-min"), caption = h("figcaption", "odd-caption");
  const els = [...shown, -1].map(() => {
    const el = h("div", "odd"), word = h("span", "odd-word"), track = h("span", "odd-track"), p = h("span", "odd-p");
    const base = h("i", "odd-base"), bar = h("i", "odd-bar"), tick = h("b", "odd-tick"), tag = h("em", "odd-tag");
    track.append(base, bar, tick, tag);
    el.append(word, track, p);
    box.append(el);
    return { el, word, base, bar, tick, tag, p };
  });
  min.append(h("span", "", "min-p"));
  box.append(min);
  fig.append(box, caption);
  let at = -1, last: (() => void) | null = null;
  box.addEventListener("pointerover", (e) => {
    const i = els.findIndex((x) => x.el.contains(e.target as Node));
    if (i !== at) { at = i; last?.(); }
  });
  box.addEventListener("pointerleave", () => { at = -1; last?.(); });
  return (rows: Odds[], minLine: { at: number; tempered: boolean } | undefined, explain: (r: Odds | null, rest?: Odds[]) => string) => {
    last = () => {
      const rest = rows.filter((_, i) => !shown.includes(i)), cuts = rest.filter((r) => r.cut).length;
      const agg: Odds = { word: `${rest.length} more`, raw: rest.reduce((a, r) => a + r.raw, 0), p: rest.reduce((a, r) => a + r.p, 0),
        cut: cuts === rest.length ? rest[0].cut : "", pres: 0, dry: 0 };
      [...shown.map((i) => rows[i]), agg].forEach((r, k) => {
        const o = els[k], isRest = k === shown.length;
        o.el.classList.toggle("cut", !!r.cut);
        o.el.classList.toggle("rest", isRest);
        o.el.classList.toggle("at", k === at);
        o.word.textContent = r.word;
        o.base.style.width = `${r.raw * 100}%`;
        o.bar.style.width = `${r.cut ? 0 : r.p * 100}%`;
        o.tick.style.left = `${r.raw * 100}%`;
        o.tag.style.setProperty("--w", `${(r.cut ? r.raw : r.p) * 100}%`);
        o.tag.classList.toggle("inside", !r.cut && r.p > 0.62);   // a long bar carries its tag at its end
        o.tag.textContent = isRest ? (cuts && !r.cut ? `${cuts} cut` : "") : tagOf(r);
        o.p.textContent = r.cut ? CUT_BY[r.cut] : pct(r.p);
      });
      const kept = rows.filter((r) => !r.cut), topNow = kept.reduce((a, r) => (r.p > a.p ? r : a), kept[0]);
      min.classList.toggle("on", !!minLine);
      min.classList.toggle("untempered", !!minLine && !minLine.tempered);
      if (minLine) min.style.setProperty("--at", String(Math.min(1, minLine.at * topNow.p)));
      caption.innerHTML = at < 0 ? explain(null) : at === shown.length ? explain(null, rest) : explain(rows[shown[at]]);
    };
    last();
  };
}

let drawFirst: ReturnType<typeof chart> | null = null, drawNext: ReturnType<typeof chart> | null = null;
const LOOP_FROM = LOOP.reply.findIndex(([t]) => t === ":") + 1;   // the reply line shows the list, not its lead-in
function drawCharts() {
  const mode = shownMode();
  // the first name of a reply: what the stages keep
  drawFirst ??= chart($("odds-first"), [0, 1, 2, 3, 4, 5], () => "");
  const first = sample(FIRST, [], (k) => value(mode, k), orderOf(mode));
  drawFirst(first.rows, first.minLine, (r, rest) => {
    if (rest) {
      const by = STAGES.map((s) => [s, rest.filter((x) => x.cut === s).length] as const).filter(([, c]) => c);
      return `<b>${rest.length} more names</b> ${pct(rest.reduce((a, x) => a + x.p, 0)) || "0%"} together` +
        (by.length ? `<span> · cut: ${by.map(([s, c]) => `${c} by ${CUT_BY[s]}`).join(", ")}</span>` : "");
    }
    if (r) return `<b>${r.word}</b> ${r.cut ? `cut by ${CUT_BY[r.cut]}` : pct(r.p)} <span>· ${pct(r.raw)} before</span>`;
    const kept = first.rows.filter((x) => !x.cut);
    return kept.length === 1 ? `Always <b>${kept[0].word}</b>` : `<b>${kept.length}</b> of ${first.rows.length} names can be drawn <span>· Luna ${pct(first.rows[0].p)}</span>`;
  });
  // the next name of a list that repeats itself: what presence and DRY push down
  drawNext ??= chart($("odds-next"), [0, 1, 4, 5], (r) => [r.pres > 0 ? "used" : r.pres < 0 ? "used, boosted" : "", r.dry ? "repeats" : ""].filter(Boolean).join(" · "));
  const next = sample(NEXT, LOOP.reply, (k) => value(mode, k), orderOf(mode)), { best } = dryRuns(LOOP.reply), allowed = value(mode, "dry_allowed_length");
  const dryOn = value(mode, "dry_multiplier") > 0 && !!best && best.run >= allowed, presOn = value(mode, "presence_penalty") !== 0;
  const runText = best ? LOOP.reply.slice(best.at - best.run + 1, best.at + 2).map(([t]) => t).join("").trim() : "";
  drawNext(next.rows, next.minLine, (r, rest) => {
    if (rest) return `<b>${rest.length} more</b> ${pct(rest.reduce((a, x) => a + x.p, 0)) || "0%"} together`;
    if (r) {
      const why = [r.pres ? `already used (presence ${r.pres > 0 ? "−" : "+"}${Math.abs(r.pres).toFixed(2)})` : "",
        r.dry ? `would repeat “${runText}” (DRY −${r.dry.toFixed(2)})` : ""].filter(Boolean);
      return `<b>${r.word}</b> ${pct(r.raw)} → ${r.cut ? `cut by ${CUT_BY[r.cut]}` : pct(r.p)}${why.length ? `<span> · ${why.join(", ")}</span>` : ""}`;
    }
    const kept = next.rows.filter((x) => !x.cut), topNow = kept.reduce((a, x) => (x.p > a.p ? x : a), kept[0]);
    return `<b>${kept.length}</b> of ${next.rows.length} can be drawn <span>· ${topNow.word} ${pct(topNow.p)}${topNow.word === "Shadow" ? ", the loop goes on" : ""}</span>`;
  });
  // the reply so far, the run DRY sees and the names presence counts as used marked in it
  // Each highlighted stretch (the run, both times; what followed it) is one span, and spaces stay outside the marks,
  // so highlights and underlines cover the words only.
  const line = $("odds-line");
  line.replaceChildren(h("span", "faint", "…"));
  let group: HTMLElement | null = null, groupCls = "";
  LOOP.reply.slice(LOOP_FROM).forEach(([t], j) => {
    const i = j + LOOP_FROM, [, space, word] = /^(\s*)([\s\S]*)$/.exec(t)!;
    const cls = dryOn && best && (i > best.at - best.run && i <= best.at || i >= LOOP.reply.length - best.run) ? "run" : dryOn && best && i === best.at + 1 ? "echo" : "";
    if (cls !== groupCls) {
      if (space) line.append(space);
      group = cls ? h("span", cls) : null;
      groupCls = cls;
      if (group) line.append(group);
    } else if (space) (group ?? line).append(space);
    (group ?? line).append(h("span", presOn && /\w/.test(word) ? "used" : "", word));
  });
  line.append(h("i", "caret"));
}

// One value on one line: label, slider (its fill shows the value, a notch the recommended one), the number, ↺.
interface KnobSpec {
  id: string; label: string; help: string; min: number; max: number; step: number;
  get(): number; rec(): number; custom(): boolean; set(v: number | undefined): void; fmt(v: number): string;
  pos(v: number): number; val(pos: number): number;    // value <-> slider position, 0..1
}
const knobs: (() => void)[] = [];
const roundTo = (v: number, step: number) => Math.round(v / step) * step;
function knob(k: KnobSpec) {
  const el = h("div", "knob"), label = h("label", "", k.label), reset = h("button", "act knob-reset");
  const num = h("input", "knob-num"), track = h("div", "knob-track"), range = h("input"), rec = h("i", "knob-rec");
  el.title = k.help;
  num.id = `knob-${k.id}`;
  label.htmlFor = num.id;
  num.type = "text";
  num.inputMode = "decimal";
  num.spellcheck = false;
  num.autocomplete = "off";
  range.type = "range";
  range.className = "slider";
  range.min = "0";
  range.max = "1";
  range.step = "any";
  range.tabIndex = -1;                             // the number takes the keyboard (arrows step it)
  range.setAttribute("aria-hidden", "true");
  reset.type = "button";
  reset.innerHTML = icon("retry");
  track.append(rec, range);
  el.append(label, track, num, reset);
  const commit = (v: number) => {
    if (!Number.isFinite(v)) return sync(true);
    v = Math.min(k.max, Math.max(k.min, roundTo(v, k.step)));
    k.set(Math.abs(v - k.rec()) < k.step / 2 ? undefined : +v.toFixed(6));
    saveSettings();
  };
  const typed = () => Number(num.value.replace(/[,_\s]/g, "").replace(/^off$/i, "0"));
  range.addEventListener("input", () => commit(k.val(range.valueAsNumber)));
  range.addEventListener("dblclick", () => { k.set(undefined); saveSettings(); });
  reset.addEventListener("click", () => { k.set(undefined); saveSettings(); });
  num.addEventListener("change", () => commit(typed()));
  num.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit(typed());
    else if (e.key === "Escape") { e.preventDefault(); sync(true); num.blur(); }
    else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault();
      const v = k.get(), up = e.key === "ArrowUp";
      commit(k.id === "max_tokens" ? (up ? v * 2 : v / 2) : v + (up ? 1 : -1) * k.step * (e.shiftKey ? 10 : 1));
      sync(true);
    }
  });
  function sync(force = false) {
    const v = k.get(), p = k.pos(v), custom = k.custom();
    el.classList.toggle("custom", custom);
    if (force || document.activeElement !== num) num.value = k.fmt(v);
    range.value = String(p);
    range.style.setProperty("--fill", `${p * 100}%`);
    rec.style.setProperty("--at", String(k.pos(k.rec())));
    rec.title = reset.title = `Recommended: ${k.fmt(k.rec())}`;
  }
  knobs.push(sync);
  return el;
}
function samplerKnob(key: SamplerKey) {
  const d = SAMPLERS[key], [lo, hi] = d.slider ?? [d.min, d.max];
  return knob({
    id: key, label: d.label, help: d.help, min: d.min, max: d.max, step: d.step,
    fmt: (v) => (key === "dry_multiplier" && v === 0 ? "off" : fmtValue(key, v)),
    get: () => value(shownMode(), key), rec: () => RECOMMENDED[shownMode()][key], custom: () => settings[shownMode()][key] !== undefined,
    set: (v) => { if (v === undefined) delete settings[shownMode()][key]; else settings[shownMode()][key] = v; },
    pos: (v) => Math.min(1, Math.max(0, (v - lo) / (hi - lo))), val: (p) => lo + p * (hi - lo),
  });
}

function buildSettings() {
  // the four stages, one row each, in the order the sampler applies them: drag a row by its grip (or arrows on it)
  for (const st of STAGES) {
    const row = samplerKnob(st), grip = h("button", "grip");
    row.dataset.stage = st;
    grip.type = "button";
    grip.title = "Drag to reorder: the sampler applies these top to bottom";
    grip.setAttribute("aria-label", `Move ${SAMPLERS[st].label}`);
    grip.innerHTML = icon("grip");
    grip.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      grip.setPointerCapture(e.pointerId);
      row.classList.add("dragging");
      const move = (ev: PointerEvent) => {
        const rows = [...$("samplers").children] as HTMLElement[], i = rows.indexOf(row), mid = (el: HTMLElement) => el.getBoundingClientRect().top + el.offsetHeight / 2;
        if (i > 0 && ev.clientY < mid(rows[i - 1])) moveStage(st, i - 1);
        else if (i < rows.length - 1 && ev.clientY > mid(rows[i + 1])) moveStage(st, i + 1);
      };
      const up = () => {
        row.classList.remove("dragging");
        grip.removeEventListener("pointermove", move);
        grip.removeEventListener("pointerup", up);
        grip.removeEventListener("pointercancel", up);
      };
      grip.addEventListener("pointermove", move);
      grip.addEventListener("pointerup", up);
      grip.addEventListener("pointercancel", up);
    });
    grip.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
      e.preventDefault();
      moveStage(st, orderOf(shownMode()).indexOf(st) + (e.key === "ArrowUp" ? -1 : 1));
      grip.focus();
    });
    row.prepend(grip);
    stageRows.set(st, row);
  }
  $("order-reset").onclick = () => { delete settings[shownMode()].order; saveSettings(); };
  const dryMore = h("div", "knob-more");
  dryMore.append(samplerKnob("dry_allowed_length"), samplerKnob("dry_base"));
  knobs.push(() => { dryMore.classList.toggle("off", value(shownMode(), "dry_multiplier") <= 0); });
  $("repetition").replaceChildren(samplerKnob("presence_penalty"), samplerKnob("dry_multiplier"), dryMore);

  const LO = 256, HI = () => status.context ?? 262144;   // reply length: powers of two on the slider
  const maxLen = knob({
    id: "max_tokens", label: "Max length", help: "Max length: the reply stops here; thinking counts toward it.", min: 1, max: 262144, step: 1,
    fmt: (v) => v.toLocaleString("en-US"), get: () => settings.max_tokens ?? MAX_REPLY, rec: () => MAX_REPLY,
    custom: () => settings.max_tokens !== undefined, set: (v) => { settings.max_tokens = v; },
    pos: (v) => Math.min(1, Math.max(0, Math.log(v / LO) / Math.log(HI() / LO))),
    val: (p) => 2 ** Math.round(Math.log2(LO) + p * Math.log2(HI() / LO)),
  });
  // Seed: random for every reply, or fixed. A reply's footer shows its seed; clicking it fixes that one here.
  const seed = h("div", "knob seed-knob"), modes = h("div", "segmented"), num = h("input", "knob-num"), dice = h("button", "act");
  seed.title = "Seed: the same seed, chat and settings draw the same reply. Click a reply's seed to fix it here.";
  const label = h("label", "", "Seed");
  label.htmlFor = num.id = "knob-seed";
  num.type = "text";
  num.inputMode = "numeric";
  num.spellcheck = false;
  num.autocomplete = "off";
  dice.type = "button";
  dice.innerHTML = icon("dice");
  dice.title = "Another seed";
  seed.append(label, modes, num, dice);
  const random = () => crypto.getRandomValues(new Uint32Array(1))[0];
  for (const [text, fixed] of [["Random", false], ["Fixed", true]] as const) {
    const b = h("button", "", text);
    b.type = "button";
    b.setAttribute("role", "radio");
    b.onclick = () => { settings.seed = fixed ? settings.seed ?? random() : undefined; saveSettings(); if (fixed) num.focus(); };
    modes.append(b);
  }
  const setSeed = () => {
    const v = Number(num.value.replace(/[,_\s]/g, ""));
    if (num.value.trim() && Number.isInteger(v) && v >= 0 && v < 2 ** 32) settings.seed = v;
    saveSettings();
  };
  num.addEventListener("change", setSeed);
  num.addEventListener("keydown", (e) => {
    if (e.key === "Enter") setSeed();
    else if (e.key === "Escape") { e.preventDefault(); num.value = String(settings.seed ?? ""); num.blur(); }
  });
  dice.onclick = () => { settings.seed = random(); saveSettings(); };
  knobs.push(() => {
    const fixed = settings.seed !== undefined;
    seed.classList.toggle("custom", fixed);
    seed.classList.toggle("fixed", fixed);
    modes.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-checked", String(i === Number(fixed))));
    if (document.activeElement !== num) num.value = String(settings.seed ?? "");
  });
  $("reply-knobs").replaceChildren(maxLen, seed);


  systemInput.addEventListener("input", () => {
    settings.system = systemInput.value;
    systemEdited = true;
    growSystem();
    saveSettings();
  });
  $("system-reset").onclick = () => {
    const before = settings.system;
    settings.system = DEFAULT_SYSTEM;
    saveSettings();
    growSystem();
    toast("The system prompt is back to the default", () => { settings.system = before; saveSettings(); growSystem(); });
  };
  $("settings-reset").onclick = () => {
    if (!customized()) return toast("Everything is already at the recommended values");
    const before = structuredClone(settings);
    settings = { system: DEFAULT_SYSTEM, think: {}, fast: {} };
    saveSettings();
    growSystem();
    toast("Everything is back to the recommended values", () => { settings = before; saveSettings(); growSystem(); });
  };
  measureScrollbar();
  addEventListener("resize", measureScrollbar);
}

// The chat stays put when Settings opens beside it: its column keeps the place it has with Settings closed, as far
// as the room left allows (style.css), which needs the scrollbar's width.
function measureScrollbar() {
  document.documentElement.style.setProperty("--sb", `${scroller.offsetWidth - scroller.clientWidth}px`);
}

function growSystem() {
  systemInput.style.height = "auto";
  systemInput.style.height = `${Math.min(systemInput.scrollHeight + 2, innerHeight * 0.4)}px`;
}

const stageRows = new Map<Stage, HTMLElement>();
function moveStage(stage: Stage, to: number) {
  const order = orderOf(shownMode()).filter((x) => x !== stage);
  order.splice(Math.max(0, Math.min(order.length, to)), 0, stage);
  if (order.join() === STAGES.join()) delete settings[shownMode()].order;
  else settings[shownMode()].order = order;
  saveSettings();
}
// The stage rows in the current order; the ones that move glide to their new place.
function placeStages() {
  const box = $("samplers"), order = orderOf(shownMode()), rows = [...box.children] as HTMLElement[];
  if (rows.map((x) => x.dataset.stage).join() === order.join()) return;
  const before = new Map(rows.map((el) => [el, el.getBoundingClientRect().top]));
  box.append(...order.map((st) => stageRows.get(st)!));
  if (reducedMotion.matches) return;
  for (const [el, top] of before) {
    const dy = top - el.getBoundingClientRect().top;
    if (dy && top) el.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], { duration: 160, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)" });
  }
}

function syncSettings() {
  if (!isOpen(panes.settings)) return;
  if (document.activeElement !== systemInput && systemInput.value !== settings.system) systemInput.value = settings.system;
  $("system-reset").classList.toggle("on", settings.system !== DEFAULT_SYSTEM);

  $("mode-note").textContent = shownMode() === "think" ? "with thinking" : "thinking off";
  $("order-reset").classList.toggle("on", !!settings[shownMode()].order);
  placeStages();
  for (const sync of knobs) sync();
  drawCharts();
}

document.addEventListener("visibilitychange", () => { if (!document.hidden) void refreshStatus(); });

$("brand-mark").innerHTML = `<svg class="flame" viewBox="0 0 24 24">${ICONS.flame}</svg>`;
$("new-chat").innerHTML = `${icon("plus")}New chat`;
$("export").innerHTML = icon("download");
$("attach").innerHTML = icon("clip");
$("draft").addEventListener("click", () => void draft());
buildControls();
$("to-latest").innerHTML = icon("down");
$("settings-reset").insertAdjacentHTML("afterbegin", icon("retry"));
for (const id of ["system-reset", "order-reset"]) $(id).innerHTML = icon("retry");
buildSettings();
for (const p of PANES) restorePane(p);
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
