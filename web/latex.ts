import katex from "katex";

export interface TexOptions { today?: string }

const MATH = new Map<string, string>();
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const ACCENTS: Record<string, string> = { "'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303", "=": "\u0304", ".": "\u0307",
  u: "\u0306", v: "\u030c", H: "\u030b", c: "\u0327", k: "\u0328", r: "\u030a", d: "\u0323", b: "\u0331" };
const SYMBOLS: Record<string, string> = { ldots: "…", dots: "…", textellipsis: "…", textbackslash: "\\", textasciitilde: "~", textasciicircum: "^",
  textbar: "|", textless: "<", textgreater: ">", textendash: "–", textemdash: "—", textbullet: "•", textdegree: "°", copyright: "©", textcopyright: "©",
  textregistered: "®", texttrademark: "™", S: "§", P: "¶", dag: "†", ddag: "‡", pounds: "£", euro: "€", ss: "ß", ae: "æ", AE: "Æ", oe: "œ", OE: "Œ",
  o: "ø", O: "Ø", aa: "å", AA: "Å", l: "ł", L: "Ł", i: "ı", j: "ȷ", quad: "\u2003", qquad: "\u2003\u2003", enspace: "\u2002", thinspace: "\u2009",
  ",": "\u2009", ";": "\u2005", ":": "\u2005", " ": " ", "!": "", "/": "", "-": "", "&": "&", "%": "%", $: "$", "#": "#", _: "_", "{": "{", "}": "}",
  textunderscore: "_", textdollar: "$", textpercent: "%", checkmark: "✓", textquoteleft: "‘", textquoteright: "’", textquotedblleft: "“", textquotedblright: "”" };
const STYLE: Record<string, [string, string]> = { textbf: ["strong", ""], textit: ["em", ""], emph: ["em", ""], textsl: ["em", ""], underline: ["u", ""],
  texttt: ["code", ""], textsc: ["span", "sc"], textsf: ["span", "sf"], textrm: ["span", ""], textup: ["span", ""], textnormal: ["span", ""],
  textmd: ["span", ""], mbox: ["span", ""], text: ["span", ""], hbox: ["span", ""], uline: ["u", ""], sout: ["s", ""], st: ["s", ""],
  textsuperscript: ["sup", ""], textsubscript: ["sub", ""], tiny: ["span", "tiny"], small: ["span", "small"], large: ["span", "large"] };
const SWITCH: Record<string, [string, string]> = { bf: ["strong", ""], bfseries: ["strong", ""], it: ["em", ""], itshape: ["em", ""], em: ["em", ""],
  sl: ["em", ""], slshape: ["em", ""], tt: ["code", ""], ttfamily: ["code", ""], sc: ["span", "sc"], scshape: ["span", "sc"], sffamily: ["span", "sf"],
  rmfamily: ["span", ""], normalfont: ["span", ""], tiny: ["span", "tiny"], scriptsize: ["span", "tiny"], footnotesize: ["span", "small"],
  small: ["span", "small"], normalsize: ["span", ""], large: ["span", "large"], Large: ["span", "larger"], LARGE: ["span", "larger"],
  huge: ["span", "huge"], Huge: ["span", "huge"] };
const QUIET = new Set(["centering", "raggedright", "raggedleft", "noindent", "indent", "newpage", "clearpage", "cleardoublepage", "pagebreak", "linebreak",
  "nopagebreak", "smallskip", "medskip", "bigskip", "vfill", "hfill", "par", "protect", "relax", "maketitle", "tableofcontents", "listoffigures",
  "listoftables", "appendix", "frontmatter", "mainmatter", "backmatter", "onecolumn", "twocolumn", "sloppy", "fussy", "nocite", "pagestyle",
  "thispagestyle", "pagenumbering", "setlength", "addtolength", "setcounter", "addtocounter", "vspace", "hspace", "label", "bibliographystyle",
  "usepackage", "documentclass", "geometry", "hypersetup", "graphicspath", "newtheorem", "newcommand", "renewcommand", "providecommand",
  "DeclareMathOperator", "theoremstyle", "title", "author", "date", "thanks", "and", "selectfont", "fontsize", "lstset", "definecolor",
  "captionsetup", "allowdisplaybreaks", "numberwithin", "makeatletter", "makeatother", "hline", "toprule", "midrule", "bottomrule", "cline",
  "cmidrule", "addlinespace", "rowcolor", "arraystretch", "renewenvironment", "newenvironment", "newgeometry", "restoregeometry", "include", "input"]);
const QUIET_ARGS: Record<string, number> = { vspace: 1, hspace: 1, label: 1, bibliographystyle: 1, usepackage: 1, documentclass: 1, geometry: 1,
  hypersetup: 1, graphicspath: 1, newtheorem: 2, newcommand: 2, renewcommand: 2, providecommand: 2, DeclareMathOperator: 2, theoremstyle: 1,
  title: 1, author: 1, date: 1, thanks: 1, pagestyle: 1, thispagestyle: 1, pagenumbering: 1, setlength: 2, addtolength: 2, setcounter: 2,
  addtocounter: 2, nocite: 1, fontsize: 2, lstset: 1, definecolor: 3, captionsetup: 1, numberwithin: 2, cline: 1, cmidrule: 1, rowcolor: 1,
  renewenvironment: 3, newenvironment: 3, newgeometry: 1, include: 1, input: 1 };
const MATH_ENVS = new Set(["equation", "align", "gather", "multline", "eqnarray", "flalign", "alignat", "displaymath", "math"]);
const LISTS: Record<string, string> = { itemize: "ul", enumerate: "ol", description: "dl" };
const THEOREMS: Record<string, string> = { theorem: "Theorem", lemma: "Lemma", proposition: "Proposition", corollary: "Corollary",
  definition: "Definition", remark: "Remark", example: "Example", claim: "Claim", conjecture: "Conjecture", note: "Note", exercise: "Exercise",
  problem: "Problem", solution: "Solution", observation: "Observation", fact: "Fact", assumption: "Assumption", hypothesis: "Hypothesis" };
const SECTIONS = ["part", "chapter", "section", "subsection", "subsubsection", "paragraph", "subparagraph"];
const BLOCKS = new Set([...SECTIONS, "[", "begin", "maketitle", "tableofcontents", "appendix", "bibliography", "printbibliography"]);
const VERBATIM = new Set(["verbatim", "Verbatim", "lstlisting", "minted", "verbatim*", "alltt", "comment"]);

function groupEnd(s: string, i: number): number {
  const open = s[i], close = open === "{" ? "}" : "]";
  let depth = 0;
  for (let k = i; k < s.length; k++) {
    const c = s[k];
    if (c === "\\") { k++; continue; }
    if (c === "%") { const nl = s.indexOf("\n", k); if (nl < 0) return -1; k = nl; continue; }
    if (c === "{") depth++;
    else if (c === "}") depth--;
    if (open === "[" && c === "]" && depth === 0) return k;
    if (open === "{" && depth === 0) return k;
    if (depth < 0) return -1;
  }
  return -1;
}
const skipSpace = (s: string, i: number) => { while (i < s.length && /[ \t\r\n]/.test(s[i]) && !(s[i] === "\n" && /^\n[ \t]*\n/.test(s.slice(i)))) i++; return i; };
function args(s: string, i: number, n: number, opt = true): { opts: string[]; groups: string[]; end: number } {
  const opts: string[] = [], groups: string[] = [];
  let k = i;
  for (;;) {
    const j = skipSpace(s, k);
    if (opt && s[j] === "[" && (groups.length < n || (n === 0 && !opts.length))) {
      const e = groupEnd(s, j);
      if (e < 0) break;
      opts.push(s.slice(j + 1, e));
      k = e + 1;
      continue;
    }
    if (groups.length >= n) break;
    if (s[j] === "{") {
      const e = groupEnd(s, j);
      if (e < 0) { groups.push(s.slice(j + 1)); k = s.length; break; }
      groups.push(s.slice(j + 1, e));
      k = e + 1;
    } else if (j < s.length && s[j] !== "}" && s[j] !== "\\") {
      groups.push(s[j]);
      k = j + 1;
    } else break;
  }
  return { opts, groups, end: k };
}
function envEnd(s: string, i: number, env: string): [number, number] {
  const re = new RegExp(`\\\\(begin|end)\\s*\\{${env.replace(/[*]/g, "\\*")}\\}`, "g");
  re.lastIndex = i;
  let depth = 1;
  for (let m = re.exec(s); m; m = re.exec(s)) {
    depth += m[1] === "begin" ? 1 : -1;
    if (!depth) return [m.index, m.index + m[0].length];
  }
  return [s.length, s.length];
}
function splitTop(s: string, sep: "\\\\" | "&" | "\\item"): { text: string; at: number }[] {
  const out: { text: string; at: number }[] = [];
  let depth = 0, env = 0, from = 0;
  for (let k = 0; k < s.length; k++) {
    const c = s[k];
    if (c === "%" ) { const nl = s.indexOf("\n", k); k = nl < 0 ? s.length : nl; continue; }
    if (c === "{") depth++;
    else if (c === "}") depth--;
    else if (c === "\\") {
      if (s.startsWith("\\begin", k)) env++;
      else if (s.startsWith("\\end", k)) env--;
      const top = depth === 0 && env === 0;
      if (top && sep === "\\\\" && s[k + 1] === "\\") { out.push({ text: s.slice(from, k), at: from }); k++; from = k + 1; continue; }
      if (top && sep === "\\item" && s.startsWith("\\item", k) && !/[A-Za-z]/.test(s[k + 5] ?? "")) { out.push({ text: s.slice(from, k), at: from }); from = k + 5; k += 4; continue; }
      k++;
      continue;
    } else if (c === "&" && sep === "&" && depth === 0 && env === 0) { out.push({ text: s.slice(from, k), at: from }); from = k + 1; }
  }
  out.push({ text: s.slice(from), at: from });
  return out;
}

class Doc {
  title = ""; author = ""; date = "";
  macros: Record<string, string> = {};
  theorems: Record<string, string> = { ...THEOREMS };
  counts: Record<string, number> = {};
  sec = [0, 0, 0, 0, 0, 0, 0];
  appendix = false;
  labels = new Map<string, string>();
  cites = new Map<string, number>();
  notes: string[] = [];
  toc: { level: number; num: string; html: string; id: string }[] = [];
  lastNum = "";
  equation = 0;
  constructor(public src: string, public opts: TexOptions) {}

  count(k: string) { return (this.counts[k] = (this.counts[k] ?? 0) + 1); }
  label(key: string, num = this.lastNum) { if (!this.labels.has(key)) this.labels.set(key, num); }

  math(tex: string, display: boolean): string {
    const key = `${display ? "D" : "i"}${JSON.stringify(this.macros)}\u0000${tex}`;
    let html = MATH.get(key);
    if (html === undefined) {
      try {
        html = katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore", output: "htmlAndMathml", macros: { ...this.macros } });
      } catch { html = `<code class="tex-err">${esc(tex)}</code>`; }
      if (MATH.size > 2000) MATH.delete(MATH.keys().next().value!);
      MATH.set(key, html);
    }
    return html;
  }

  displayMath(env: string, body: string): string {
    const star = env.endsWith("*") || env === "displaymath" || env === "math";
    let name = env.replace(/\*$/, "");
    let inner = body;
    if (name === "eqnarray") { name = "align"; inner = inner.replace(/&\s*([=<>]|\\[a-z]+)\s*&/g, "&$1 "); }
    if (name === "multline") name = "gather";
    if (name === "alignat") inner = inner.replace(/^\s*\{\d+\}/, "");
    if (name === "flalign" || name === "alignat") name = "align";
    if (name === "displaymath" || name === "math") name = "equation";
    const many = name === "align" || name === "gather";
    const rows = many ? splitTop(inner, "\\\\").map((r) => r.text.replace(/^\s*\[[^\]]*\]/, "")) : [inner];
    const tagged = rows.map((row, k) => {
      let r = row;
      const labels = [...r.matchAll(/\\label\s*\{([^}]*)\}/g)].map((m) => m[1]);
      r = r.replace(/\\label\s*\{[^}]*\}/g, "");
      const off = star || /\\(nonumber|notag)\b/.test(r) || (many && !r.trim() && k === rows.length - 1);
      r = r.replace(/\\(nonumber|notag)\b/g, "");
      if (!off && !/\\tag\b/.test(r)) {
        const n = String(++this.equation);
        this.lastNum = n;
        for (const l of labels) this.label(l, n);
        r += ` \\tag{${n}}`;
      } else for (const l of labels) this.label(l, r.match(/\\tag\s*\{([^}]*)\}/)?.[1] ?? "");
      return r;
    });
    if (many && !tagged.at(-1)!.trim()) tagged.pop();
    return this.math(many ? `\\begin{${name}*}${tagged.join("\\\\")}\\end{${name}*}` : `\\begin{equation*}${tagged[0]}\\end{equation*}`, true);
  }

  inline(s: string, depth = 0): string {
    let out = "", k = 0;
    const text = (t: string) => esc(t.replace(/---/g, "—").replace(/--/g, "–").replace(/``/g, "“").replace(/''/g, "”").replace(/`/g, "‘")
      .replace(/'/g, "’").replace(/~/g, "\u00a0").replace(/\s+/g, " "));
    while (k < s.length) {
      const c = s[k];
      if (c === "%") { const nl = s.indexOf("\n", k); k = nl < 0 ? s.length : nl + 1; continue; }
      if (c === "$") {
        const dbl = s[k + 1] === "$", close = dbl ? s.indexOf("$$", k + 2) : this.mathClose(s, k + 1);
        if (close < 0) { out += text(s.slice(k)); break; }
        out += dbl ? `<span class="tex-display">${this.math(s.slice(k + 2, close), true)}</span>` : this.math(s.slice(k + 1, close), false);
        k = close + (dbl ? 2 : 1);
        continue;
      }
      if (c === "{") {
        const e = groupEnd(s, k), inner = s.slice(k + 1, e < 0 ? s.length : e);
        k = e < 0 ? s.length : e + 1;
        const sw = inner.match(/^\s*\\([A-Za-z]+)\b\s*/);
        const st = sw && SWITCH[sw[1]];
        out += st ? wrap(st, this.inline(inner.slice(sw![0].length), depth + 1)) : this.inline(inner, depth + 1);
        continue;
      }
      if (c === "}") { k++; continue; }
      if (c === "\\") {
        const m = s.slice(k + 1).match(/^([A-Za-z]+\*?|.)/);
        if (!m) { k++; continue; }
        const name = m[1], after = k + 1 + name.length;
        const r = this.command(s, name, after, depth);
        out += r.html;
        k = r.end;
        continue;
      }
      let e = k;
      while (e < s.length && !"$\\{}%".includes(s[e])) e++;
      out += text(s.slice(k, e));
      k = e;
    }
    return out;
  }
  mathClose(s: string, i: number): number {
    for (let k = i; k < s.length; k++) { if (s[k] === "\\") { k++; continue; } if (s[k] === "$") return k; }
    return -1;
  }

  command(s: string, name: string, at: number, depth: number): { html: string; end: number } {
    const base = name.replace(/\*$/, "");
    const a = (n: number, opt = true) => args(s, at, n, opt);
    if (name === "(") { const e = s.indexOf("\\)", at); return e < 0 ? { html: esc(s.slice(at)), end: s.length } : { html: this.math(s.slice(at, e), false), end: e + 2 }; }
    if (name === "[") { const e = s.indexOf("\\]", at); return e < 0 ? { html: esc(s.slice(at)), end: s.length } : { html: `<span class="tex-display">${this.displayMath("equation", s.slice(at, e))}</span>`, end: e + 2 }; }
    if (name === "\\") return { html: "<br>", end: args(s, s[at] === "*" ? at + 1 : at, 0).end };
    if (base === "newline" || base === "linebreak") return { html: "<br>", end: at };
    if (name in SYMBOLS && !/[A-Za-z]/.test(name)) return { html: esc(SYMBOLS[name]), end: at };
    if (name in ACCENTS) {
      const r = a(1, false), ch = (r.groups[0] ?? "").replace(/^\\([ij])$/, (_m, x) => (x === "i" ? "ı" : "ȷ"));
      return { html: esc((ch + ACCENTS[name]).normalize("NFC")), end: r.end };
    }
    if (name in SYMBOLS) { const sp = s[at] === "{" && s[at + 1] === "}" ? 2 : 0; return { html: esc(SYMBOLS[name]), end: at + sp }; }
    if (base === "today") return { html: esc(this.opts.today ?? new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })), end: at };
    if (base === "LaTeX") return { html: 'L<span class="tex-a">a</span>T<span class="tex-e">e</span>X', end: at };
    if (base === "TeX") return { html: 'T<span class="tex-e">e</span>X', end: at };
    if (name in STYLE) { const r = a(1, false); return { html: wrap(STYLE[name], this.inline(r.groups[0] ?? "", depth + 1)), end: r.end }; }
    if (name in SWITCH) return { html: "", end: at };
    if (base === "textcolor" || base === "color" && s[skipSpace(s, at)] === "{") {
      const r = a(base === "color" ? 1 : 2), color = cssColor(r.groups[0] ?? "");
      return base === "color" ? { html: "", end: r.end } : { html: color ? `<span style="color:${color}">${this.inline(r.groups[1] ?? "", depth + 1)}</span>` : this.inline(r.groups[1] ?? "", depth + 1), end: r.end };
    }
    if (base === "colorbox" || base === "fcolorbox" || base === "fbox" || base === "framebox") {
      const n = base === "fcolorbox" ? 3 : base === "colorbox" ? 2 : 1, r = a(n), color = base === "colorbox" ? cssColor(r.groups[0] ?? "") : "";
      return { html: `<span class="tex-box"${color ? ` style="background:${color}"` : ""}>${this.inline(r.groups.at(-1) ?? "", depth + 1)}</span>`, end: r.end };
    }
    if (base === "href") {
      const r = a(2, false), url = (r.groups[0] ?? "").replace(/\\([#%&_~])/g, "$1");
      return { html: safeUrl(url) ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${this.inline(r.groups[1] ?? "", depth + 1)}</a>` : this.inline(r.groups[1] ?? "", depth + 1), end: r.end };
    }
    if (base === "url" || base === "nolinkurl") {
      const r = a(1, false), url = (r.groups[0] ?? "").replace(/\\([#%&_~])/g, "$1");
      return { html: safeUrl(url) && base === "url" ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer"><code>${esc(url)}</code></a>` : `<code>${esc(url)}</code>`, end: r.end };
    }
    if (base === "footnote") {
      const r = a(1), n = this.notes.push(this.inline(r.groups[0] ?? "", depth + 1));
      return { html: `<sup class="tex-fn"><a href="#tex-fn-${n}">${n}</a></sup>`, end: r.end };
    }
    if (base === "cite" || base === "citep" || base === "citet" || base === "parencite" || base === "textcite" || base === "autocite") {
      const r = a(1), keys = (r.groups[0] ?? "").split(",").map((x) => x.trim()).filter(Boolean);
      const nums = keys.map((key) => { if (!this.cites.has(key)) this.cites.set(key, this.cites.size + 1); return `\u0000cite:${key}\u0000`; });
      const note = r.opts.at(-1) ? `, ${this.inline(r.opts.at(-1)!, depth + 1)}` : "";
      return { html: `<span class="tex-cite">[${nums.join(", ")}${note}]</span>`, end: r.end };
    }
    if (base === "ref" || base === "eqref" || base === "pageref" || base === "autoref" || base === "cref" || base === "Cref" || base === "nameref") {
      const r = a(1, false), key = (r.groups[0] ?? "").trim(), ref = `<a class="tex-ref" href="#tex-l-${esc(key)}">\u0000ref:${key}\u0000</a>`;
      return { html: base === "eqref" ? `(${ref})` : ref, end: r.end };
    }
    if (base === "label") { const r = a(1, false); this.label((r.groups[0] ?? "").trim()); return { html: `<span id="tex-l-${esc((r.groups[0] ?? "").trim())}"></span>`, end: r.end }; }
    if (base === "includegraphics") { const r = a(1); return { html: `<span class="tex-graphic">${esc(r.groups[0] ?? "picture")}</span>`, end: r.end }; }
    if (base === "verb") {
      const d = s[at], e = d ? s.indexOf(d, at + 1) : -1;
      return e < 0 ? { html: "", end: s.length } : { html: `<code>${esc(s.slice(at + 1, e))}</code>`, end: e + 1 };
    }
    if (base === "item") { const r = a(0); return { html: r.opts[0] ? `<strong>${this.inline(r.opts[0], depth + 1)}</strong> ` : "", end: r.end }; }
    const mine = this.macros[`\\${base}`];
    if (mine !== undefined && depth < 20) {
      const n = Math.max(0, ...[...mine.matchAll(/#(\d)/g)].map((x) => Number(x[1])));
      const r = a(n, false);
      return { html: this.inline(mine.replace(/#(\d)/g, (_x, d) => r.groups[Number(d) - 1] ?? ""), depth + 1), end: r.end };
    }
    if (QUIET.has(base)) return { html: "", end: a(QUIET_ARGS[base] ?? 0).end };
    const r = a(1);
    return { html: r.groups.length ? this.inline(r.groups[0], depth + 1) : "", end: r.end };
  }

  blocks(s: string, at: number): string {
    let out = "", k = 0, para = -1;
    const flush = (end: number) => {
      if (para < 0) return;
      const t = s.slice(para, end);
      if (t.replace(/%[^\n]*/g, "").trim()) out += `<p data-at="${at + para + lead(t)}" data-end="${at + end - trail(t)}">${this.inline(t)}</p>`;
      para = -1;
    };
    while (k < s.length) {
      if (s[k] === "%") { const nl = s.indexOf("\n", k); k = nl < 0 ? s.length : nl + 1; continue; }
      if (s[k] === "\n" && /^\n[ \t]*\n/.test(s.slice(k))) { flush(k); k = s.slice(k).match(/^\n[ \t]*\n\s*/)![0].length + k; continue; }
      if (s[k] === "\\") {
        const m = s.slice(k).match(/^\\([A-Za-z]+\*?|\[)/), name = m?.[1] ?? "";
        if (BLOCKS.has(name.replace(/\*$/, ""))) flush(k);
        const block = BLOCKS.has(name.replace(/\*$/, "")) ? this.block(s, k, name, at) : null;
        if (block) { out += block.html; k = block.end; continue; }
        if (name) { if (para < 0) para = k; k += 1 + name.length; continue; }
        if (para < 0) para = k;
        k += 2;
        continue;
      }
      if (s[k] === "$" && s[k + 1] === "$" && (para < 0 || /\n\s*$/.test(s.slice(para, k)))) {
        const e = s.indexOf("$$", k + 2), end = e < 0 ? s.length : e + 2;
        flush(k);
        out += `<div class="tex-eq" data-at="${at + k}" data-end="${at + end}">${this.displayMath("equation*", s.slice(k + 2, e < 0 ? s.length : e))}</div>`;
        k = end;
        continue;
      }
      if (para < 0 && !/\s/.test(s[k])) para = k;
      k++;
    }
    flush(s.length);
    return out;
  }

  block(s: string, k: number, name: string, at: number): { html: string; end: number } | null {
    const base = name.replace(/\*$/, ""), pos = (a: number, b: number) => `data-at="${at + a}" data-end="${at + b}"`;
    if (name === "[") {
      const e = s.indexOf("\\]", k + 2), end = e < 0 ? s.length : e + 2;
      return { html: `<div class="tex-eq" ${pos(k, end)}>${this.displayMath("equation", s.slice(k + 2, e < 0 ? s.length : e))}</div>`, end };
    }
    const level = SECTIONS.indexOf(base);
    if (level >= 0) {
      const r = args(s, k + 1 + name.length, 1), star = name.endsWith("*");
      let num = "";
      if (!star && base !== "paragraph" && base !== "subparagraph") {
        this.sec[level]++;
        for (let l = level + 1; l < this.sec.length; l++) this.sec[l] = 0;
        const top = SECTIONS.indexOf(this.sec[1] ? "chapter" : "section");
        const parts = this.sec.slice(top, level + 1).map(String);
        if (this.appendix && parts.length) parts[0] = String.fromCharCode(64 + Number(parts[0]));
        num = level === 0 ? romanOf(this.sec[0]) : parts.join(".");
        this.lastNum = num;
      }
      const tag = `h${Math.min(6, Math.max(2, level))}`, html = this.inline(r.groups[0] ?? ""), id = `tex-s-${this.toc.length + 1}`;
      if (!star && level <= 4) this.toc.push({ level, num, html, id });
      return { html: `<${tag} id="${id}" ${pos(k, r.end)}>${num ? `<span class="tex-num">${esc(num)}</span>` : ""}${html}</${tag}>`, end: r.end };
    }
    const past = k + 1 + name.length;
    if (base === "maketitle") return { html: this.titleBlock(k, past, at), end: past };
    if (base === "tableofcontents") return { html: `<nav class="tex-toc" ${pos(k, past)}>\u0000toc\u0000</nav>`, end: past };
    if (base === "appendix") { this.appendix = true; this.sec = this.sec.map(() => 0); return { html: "", end: past }; }
    if (base === "bibliography" || base === "printbibliography") { const r = args(s, k + 1 + name.length, base === "bibliography" ? 1 : 0); return { html: "", end: r.end }; }
    if (base === "caption") return null;
    if (base !== "begin") return null;
    const head = s.slice(k).match(/^\\begin\s*\{([^}]*)\}/);
    if (!head) return null;
    const env = head[1], start = k + head[0].length, [cEnd, end] = envEnd(s, start, env), body = s.slice(start, cEnd), e0 = env.replace(/\*$/, "");
    const block = (cls: string, inner: string, tag = "div") => ({ html: `<${tag} class="${cls}" ${pos(k, end)}>${inner}</${tag}>`, end });
    if (e0 === "document") return { html: this.blocks(body, at + start), end };
    if (MATH_ENVS.has(e0)) return block("tex-eq", this.displayMath(env, body));
    if (e0 in LISTS) return this.list(env, body, start, k, end, at);
    if (VERBATIM.has(env)) {
      if (env === "comment") return { html: "", end };
      const code = body.replace(/^\s*\[[^\]\n]*\]/, "").replace(/^\s*\{[^}\n]*\}/, env === "minted" ? "" : "$&").replace(/^\n/, "").replace(/\n\s*$/, "");
      return block("tex-verb", `<pre><code>${esc(code)}</code></pre>`);
    }
    if (e0 === "abstract") return block("tex-abstract", `<div class="tex-abstract-head">Abstract</div>${this.blocks(body, at + start)}`);
    if (e0 === "quote" || e0 === "quotation" || e0 === "verse") return block("tex-quote", this.blocks(body, at + start), "blockquote");
    if (e0 === "center" || e0 === "flushleft" || e0 === "flushright" || e0 === "minipage") {
      const inner = e0 === "minipage" ? body.replace(/^\s*(\[[^\]]*\])*\s*\{[^}]*\}/, "") : body;
      return block(`tex-${e0}`, this.blocks(inner, at + start + (body.length - inner.length)));
    }
    if (e0 === "figure" || e0 === "table" || e0 === "wrapfigure" || e0 === "sidewaystable") return this.float(e0 === "table" || e0 === "sidewaystable" ? "table" : "figure", body, start, k, end, at);
    if (e0 === "tabular" || e0 === "tabularx" || e0 === "longtable" || e0 === "array") return block("tex-table", this.tabular(body, e0 === "tabularx"));
    if (e0 === "thebibliography") return block("tex-bib", this.bibliography(body));
    if (e0 === "proof") {
      const r = args(body, 0, 0);
      return block("tex-proof", `<span class="tex-thm-head">${r.opts[0] ? this.inline(r.opts[0]) : "Proof"}.</span> ${this.blocks(body.slice(r.end), at + start + r.end)}<span class="tex-qed">∎</span>`);
    }
    if (e0 in this.theorems) {
      const r = args(body, 0, 0), n = env.endsWith("*") ? "" : String(this.count(`thm:${this.theorems[e0]}`));
      this.lastNum = n;
      const title = `${this.theorems[e0]}${n ? ` ${n}` : ""}${r.opts[0] ? ` (${this.inline(r.opts[0])})` : ""}.`;
      return block(`tex-thm tex-${e0}`, `<span class="tex-thm-head">${title}</span> ${this.blocks(body.slice(r.end), at + start + r.end)}`);
    }
    if (e0 === "tikzpicture" || e0 === "pgfpicture" || e0 === "picture" || e0 === "forest" || e0 === "circuitikz")
      return block("tex-graphic", `${esc(env)} picture (not drawn here)`);
    return block(`tex-env tex-${esc(e0)}`, this.blocks(body.replace(/^\s*(\[[^\]]*\]|\{[^}]*\})*/, ""), at + start));
  }

  list(env: string, body: string, start: number, k: number, end: number, at: number): { html: string; end: number } {
    const kind = env.replace(/\*$/, ""), tag = LISTS[kind], items = splitTop(body, "\\item").slice(1);
    const html = items.map((it) => {
      const r = args(it.text, 0, 0), label = r.opts[0], rest = it.text.slice(r.end), from = at + start + it.at - 5;
      const to = at + start + it.at + it.text.length - trail(it.text);
      const inner = this.blocks(rest, at + start + it.at + r.end);
      if (tag === "dl") return `<dt data-at="${from}" data-end="${to}">${label ? this.inline(label) : ""}</dt><dd>${inner}</dd>`;
      return `<li data-at="${from}" data-end="${to}"${label ? ` class="tex-labeled" data-label="${esc(label.replace(/[{}\\]/g, ""))}"` : ""}>${inner}</li>`;
    }).join("");
    return { html: `<${tag} class="tex-list" data-at="${at + k}" data-end="${at + end}">${html}</${tag}>`, end };
  }

  float(kind: "figure" | "table", body: string, start: number, k: number, end: number, at: number): { html: string; end: number } {
    const inner = body.replace(/^\s*\[[^\]]*\]/, "");
    let caption = "";
    const cap = inner.search(/\\caption\b/), name = kind === "figure" ? "Figure" : "Table";
    let rest = inner;
    if (cap >= 0) {
      const r = args(inner, cap + 8, 1), n = this.count(kind);
      this.lastNum = String(n);
      caption = `<figcaption><span class="tex-cap">${name} ${n}:</span> ${this.inline(r.groups[0] ?? "")}</figcaption>`;
      rest = inner.slice(0, cap) + " ".repeat(r.end - cap) + inner.slice(r.end);
    }
    const labels = [...rest.matchAll(/\\label\s*\{([^}]*)\}/g)].map((m) => m[1]);
    for (const l of labels) this.label(l);
    const content = this.blocks(rest, at + start + (body.length - inner.length));
    const top = kind === "table";
    return { html: `<figure class="tex-float tex-${kind}" data-at="${at + k}" data-end="${at + end}">${top ? caption : ""}${content}${top ? "" : caption}</figure>`, end };
  }

  tabular(body: string, x: boolean): string {
    const r = args(body, 0, x ? 2 : 1), spec = (r.groups[x ? 1 : 0] ?? "").replace(/\{[^}]*\}/g, "").replace(/[|@!<>]/g, "").replace(/\*\{(\d+)\}\{([^}]*)\}/g, (_m, n, c) => c.repeat(Number(n)));
    const align = [...spec].filter((c) => /[lcrpmbXS]/.test(c)).map((c) => (c === "c" ? "center" : c === "r" ? "right" : "left"));
    const rows = splitTop(body.slice(r.end), "\\\\");
    let html = "", rule = false;
    for (const row of rows) {
      let t = row.text.replace(/^\s*\[[^\]]*\]/, "");
      const ruled = /\\(hline|toprule|midrule|bottomrule|cmidrule|cline)\b/.test(t);
      t = t.replace(/\\(hline|toprule|midrule|bottomrule|addlinespace|endhead|endfirsthead|endfoot|endlastfoot)\b/g, "").replace(/\\(cline|cmidrule)(\([^)]*\))?\s*\{[^}]*\}/g, "");
      if (ruled && !t.trim() && html) rule = true;
      if (!t.trim()) { if (ruled) rule = true; continue; }
      const cells = splitTop(t, "&");
      let col = 0;
      html += `<tr${rule ? ' class="tex-rule"' : ""}>` + cells.map((cell) => {
        const mc = cell.text.trim().match(/^\\multicolumn\s*\{(\d+)\}\s*\{([^}]*)\}\s*\{([\s\S]*)\}$/);
        const span = mc ? Number(mc[1]) : 1, a = mc ? (mc[2].includes("c") ? "center" : mc[2].includes("r") ? "right" : "left") : align[col] ?? "left";
        col += span;
        return `<td${span > 1 ? ` colspan="${span}"` : ""} style="text-align:${a}">${this.inline(mc ? mc[3] : cell.text)}</td>`;
      }).join("") + "</tr>";
      rule = false;
    }
    return `<table>${html}</table>`;
  }

  bibliography(body: string): string {
    const items = body.split(/\\bibitem\b/).slice(1);
    return `<div class="tex-bib-head">References</div><ol>${items.map((it) => {
      const r = args(it, 0, 1), key = (r.groups[0] ?? "").trim();
      if (!this.cites.has(key)) this.cites.set(key, this.cites.size + 1);
      return `<li id="tex-cite-${esc(key)}" value="\u0000citeno:${key}\u0000">${this.inline(it.slice(r.end))}</li>`;
    }).join("")}</ol>`;
  }

  titleBlock(k: number, end: number, at: number): string {
    const authors = this.author.split(/\\and\b/).map((a) => `<div class="tex-author">${this.inline(a.replace(/\\thanks\s*\{[^}]*\}/g, ""))}</div>`).join("");
    return `<header class="tex-title" data-at="${at + k}" data-end="${at + end}">${this.title ? `<h1>${this.inline(this.title)}</h1>` : ""}` +
      `${authors ? `<div class="tex-authors">${authors}</div>` : ""}${this.date !== "\u0000none" ? `<div class="tex-date">${this.date ? this.inline(this.date) : this.inline("\\today")}</div>` : ""}</header>`;
  }
}

const wrap = ([tag, cls]: [string, string], html: string) => `<${tag}${cls ? ` class="tex-${cls}"` : ""}>${html}</${tag}>`;
const lead = (t: string) => t.length - t.trimStart().length;
const trail = (t: string) => t.length - t.trimEnd().length;
const safeUrl = (u: string) => /^(https?:|mailto:|#)/i.test(u.trim());
const romanOf = (n: number) => ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][n] ?? String(n);
const NAMED = new Set(["red", "green", "blue", "cyan", "magenta", "yellow", "black", "white", "gray", "grey", "orange", "purple", "brown", "pink",
  "violet", "teal", "olive", "lime", "darkgray", "lightgray"]);
function cssColor(c: string): string {
  const t = c.trim();
  if (NAMED.has(t)) return t;
  const hex = t.match(/^(?:\[HTML\]\s*)?\{?#?([0-9a-fA-F]{6})\}?$/);
  return hex ? `#${hex[1]}` : "";
}

function preamble(doc: Doc, pre: string) {
  const one = (cmd: string) => { const i = pre.search(new RegExp(`\\\\${cmd}\\s*[\\[{]`)); return i < 0 ? null : args(pre, i + cmd.length + 1, 1).groups[0] ?? ""; };
  doc.title = one("title") ?? "";
  doc.author = one("author") ?? "";
  const date = one("date");
  doc.date = date === null ? "" : date.trim() ? date : "\u0000none";
  for (const m of pre.matchAll(/\\(?:re)?newcommand\*?\s*\{?\\([A-Za-z]+)\}?\s*(?:\[(\d)\])?(?:\[[^\]]*\])?\s*\{/g)) {
    const open = m.index! + m[0].length - 1, e = groupEnd(pre, open);
    if (e > 0) doc.macros[`\\${m[1]}`] = pre.slice(open + 1, e);
  }
  for (const m of pre.matchAll(/\\DeclareMathOperator\*?\s*\{\\([A-Za-z]+)\}\s*\{([^}]*)\}/g)) doc.macros[`\\${m[1]}`] = `\\operatorname{${m[2]}}`;
  for (const m of pre.matchAll(/\\newtheorem\*?\s*\{([^}]+)\}\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}/g)) doc.theorems[m[1]] = m[2];
}

export function latexToHtml(src: string, opts: TexOptions = {}): string {
  const doc = new Doc(src, opts), begin = src.search(/\\begin\s*\{document\}/);
  preamble(doc, src);
  for (const m of src.matchAll(/\\bibitem\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}/g)) if (!doc.cites.has(m[1].trim())) doc.cites.set(m[1].trim(), doc.cites.size + 1);
  let html: string;
  if (begin >= 0) {
    const start = src.indexOf("}", begin) + 1, [cEnd] = envEnd(src, start, "document");
    html = doc.blocks(src.slice(start, cEnd), start);
  } else html = doc.blocks(src.replace(/\\(documentclass|usepackage)(\[[^\]]*\])?\{[^}]*\}/g, (m) => " ".repeat(m.length)), 0);
  if (doc.notes.length) html += `<section class="tex-notes">${doc.notes.map((n, k) => `<p id="tex-fn-${k + 1}"><sup>${k + 1}</sup> ${n}</p>`).join("")}</section>`;
  const toc = doc.toc.map((t) => `<a class="tex-toc-${t.level}" href="#${t.id}">${t.num ? `<span class="tex-num">${esc(t.num)}</span>` : ""}${t.html}</a>`).join("");
  return `<article class="tex">${html.replace(/\u0000(ref|cite|citeno|toc):?([^\u0000]*)\u0000/g, (_m, kind, key) =>
    kind === "toc" ? toc : kind === "ref" ? esc(doc.labels.get(key) || "??") : String(doc.cites.get(key) ?? "?"))}</article>`;
}

export const LATEX_CSS = `:host { display: block; }
.tex { max-width: 44em; margin: 0 auto; padding: 28px 34px; color: #1d1b1a; background: #fffefb; font: 16px/1.55 "Latin Modern Roman", "CMU Serif", "Computer Modern", Georgia, "Times New Roman", serif; text-align: justify; hyphens: auto; }
.tex h1 { margin: 0.2em 0 0.4em; font-size: 1.7em; font-weight: normal; text-align: center; line-height: 1.25; }
.tex-title { margin-bottom: 1.6em; text-align: center; }
.tex-authors { display: flex; flex-wrap: wrap; justify-content: center; gap: 0.3em 2.2em; font-size: 1.1em; }
.tex-date { margin-top: 0.5em; }
.tex h2, .tex h3, .tex h4, .tex h5, .tex h6 { margin: 1.3em 0 0.5em; font-weight: bold; line-height: 1.3; text-align: left; }
.tex h2 { font-size: 1.3em; } .tex h3 { font-size: 1.12em; } .tex h4, .tex h5, .tex h6 { font-size: 1em; }
.tex-num { margin-right: 0.9em; }
.tex p { margin: 0 0 0.75em; }
.tex p + p { text-indent: 1.5em; }
.tex-abstract { margin: 0 3em 1.6em; font-size: 0.93em; }
.tex-abstract-head { margin-bottom: 0.4em; font-weight: bold; text-align: center; }
.tex-eq { margin: 0.6em 0; overflow-x: auto; overflow-y: hidden; }
.tex-display { display: block; margin: 0.6em 0; overflow-x: auto; }
.tex-list { margin: 0.4em 0 0.8em; padding-left: 2em; }
.tex-list li { margin: 0.2em 0; }
.tex-list li > p, .tex-list dd > p { margin: 0; text-indent: 0; }
.tex-labeled { list-style: none; } .tex-labeled::before { content: attr(data-label) " "; margin-left: -1.6em; font-weight: bold; }
dl.tex-list dt { font-weight: bold; } dl.tex-list dd { margin-left: 1.5em; }
.tex-quote { margin: 0.6em 2.5em; }
.tex-center { text-align: center; } .tex-flushright { text-align: right; } .tex-flushleft { text-align: left; }
.tex-float { margin: 1.2em 0; text-align: center; }
.tex-float figcaption { margin: 0.5em 2em; font-size: 0.93em; text-align: center; }
.tex-cap { font-weight: bold; }
.tex-table, .tex-float .tex-table { display: flex; justify-content: center; overflow-x: auto; }
.tex table { border-collapse: collapse; border-top: 1.5px solid #222; border-bottom: 1.5px solid #222; }
.tex td { padding: 0.25em 0.7em; text-align: left; vertical-align: top; }
.tex tr.tex-rule td { border-top: 0.8px solid #222; }
.tex-graphic { display: inline-flex; align-items: center; justify-content: center; min-width: 12em; min-height: 6em; padding: 0.5em 1em; border: 1px dashed #b9b3ad; border-radius: 6px; color: #8a827c; font: 13px ui-monospace, monospace; }
.tex-thm, .tex-proof { margin: 0.8em 0; }
.tex-thm > p, .tex-proof > p { display: inline; margin: 0; }
.tex-thm { font-style: italic; } .tex-definition, .tex-remark, .tex-example, .tex-note, .tex-solution { font-style: normal; }
.tex-thm-head { font-style: normal; font-weight: bold; }
.tex-proof .tex-thm-head { font-style: italic; font-weight: normal; }
.tex-qed { float: right; }
.tex-verb pre { margin: 0.6em 0; padding: 0.7em 0.9em; border-radius: 6px; background: #f4f1ec; font: 13px/1.45 ui-monospace, "JetBrains Mono", monospace; text-align: left; white-space: pre-wrap; }
.tex code { font: 0.9em ui-monospace, "JetBrains Mono", monospace; }
.tex a { color: #8b1d4f; text-decoration: none; } .tex a:hover { text-decoration: underline; }
.tex-sc { font-variant: small-caps; } .tex-sf { font-family: system-ui, sans-serif; }
.tex-tiny { font-size: 0.7em; } .tex-small { font-size: 0.88em; } .tex-large { font-size: 1.15em; } .tex-larger { font-size: 1.35em; } .tex-huge { font-size: 1.7em; }
.tex-box { padding: 0 0.25em; border: 0.8px solid currentColor; }
.tex-a { margin-left: -0.32em; margin-right: -0.12em; font-size: 0.75em; vertical-align: 0.25em; } .tex-e { margin-left: -0.15em; margin-right: -0.1em; vertical-align: -0.22em; }
.tex-toc { display: flex; flex-direction: column; gap: 0.15em; margin: 0 0 1.4em; }
.tex-toc a { color: inherit; } .tex-toc-3 { padding-left: 1.5em; } .tex-toc-4 { padding-left: 3em; }
.tex-bib-head { margin: 1.3em 0 0.5em; font-size: 1.3em; font-weight: bold; }
.tex-bib ol { padding-left: 2.2em; } .tex-bib li { margin: 0.25em 0; }
.tex-notes { margin-top: 2em; padding-top: 0.6em; border-top: 0.8px solid #999; font-size: 0.88em; }
.tex-notes p { margin: 0.2em 0; text-indent: 0; }
.tex-err { color: #b00020; }`;
