// Fetch URL, the built-in tool: GETs an http(s) address and returns it as text the model reads well (a page as compact
// markdown: its title, headings, paragraphs, lists, links, tables and code, without scripts, styles and navigation),
// with the final URL and the HTTP status. The web server runs on the user's machine and may be reachable from the LAN,
// so a model (or a page it read) must not reach what only this machine can: an address that resolves to a loopback,
// private, link-local or otherwise non-public IP is refused, at every redirect, unless the config file allows it
// ("fetch": {"allowPrivate": true}, or a list of addresses and CIDR blocks). The check runs in the connection's own
// DNS lookup, on the addresses it then connects to, so a name that answers differently the second time gets nowhere.
import { lookup as dnsLookup } from "node:dns";
import http from "node:http";
import https from "node:https";
import { isIP } from "node:net";
import type { Readable } from "node:stream";
import { createBrotliDecompress, createGunzip, createInflate } from "node:zlib";
import type { Source, Tool } from "./tools";

const MAX_REDIRECTS = 5, MAX_BYTES = 4 << 20, DEFAULT_TIMEOUT_S = 20, DEFAULT_CHARS = 20_000;

// Addresses as 128-bit numbers, IPv4 as IPv4-mapped IPv6 (::ffff:a.b.c.d).
function toBig(ip: string): bigint | null {
  ip = ip.replace(/^\[|\]$/g, "").replace(/%.*$/, "");
  if (isIP(ip) === 4) return 0xffff00000000n | BigInt(ip.split(".").reduce((a, o) => a * 256 + Number(o), 0));
  if (isIP(ip) !== 6) return null;
  const dotted = ip.match(/(\d+\.\d+\.\d+\.\d+)$/);
  if (dotted) {
    const v = dotted[1].split(".").reduce((a, o) => a * 256 + Number(o), 0);
    ip = `${ip.slice(0, -dotted[1].length)}${(v >>> 16).toString(16)}:${(v & 0xffff).toString(16)}`;
  }
  const [head, tail] = ip.split("::"), h = head ? head.split(":") : [], t = tail === undefined ? null : tail ? tail.split(":") : [];
  const groups = t === null ? h : [...h, ...Array(8 - h.length - t.length).fill("0"), ...t];
  return groups.reduce((a, g) => (a << 16n) | BigInt(parseInt(g, 16)), 0n);
}
const block = (cidr: string): [bigint, number] | null => {
  const [ip, len] = cidr.split("/"), x = toBig(ip);
  if (x === null) return null;
  const v4 = isIP(ip.replace(/^\[|\]$/g, "")) === 4, n = len === undefined ? 128 : Number(len) + (v4 ? 96 : 0);
  return Number.isInteger(n) && n >= 0 && n <= 128 ? [x, n] : null;
};
const inBlock = (x: bigint, [base, len]: [bigint, number]) => len === 0 || x >> BigInt(128 - len) === base >> BigInt(128 - len);
// Not the public internet (IANA's special-purpose registries): this host, private networks, shared and link-local
// space, documentation and benchmarking ranges, multicast, reserved. NAT64 and 6to4 addresses carry an IPv4 address,
// which is judged instead.
const NON_PUBLIC = ["0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24",
  "192.0.2.0/24", "192.88.99.0/24", "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24", "224.0.0.0/4", "240.0.0.0/4",
  "::/96", "100::/64", "2001::/23", "2001:db8::/32", "fc00::/7", "fe80::/10", "fec0::/10", "ff00::/8"].map((c) => block(c)!);
const NAT64 = block("64:ff9b::/96")!, SIX_TO_FOUR = block("2002::/16")!;
const v4 = (n: bigint) => [24n, 16n, 8n, 0n].map((s) => String((n >> s) & 255n)).join(".");
export function isPublic(ip: string): boolean {
  const x = toBig(ip);
  if (x === null) return false;
  if (inBlock(x, NAT64)) return isPublic(v4(x & 0xffffffffn));
  if (inBlock(x, SIX_TO_FOUR)) return isPublic(v4((x >> 80n) & 0xffffffffn));
  return !NON_PUBLIC.some((b) => inBlock(x, b));
}

type Policy = true | [bigint, number][];            // every address, or public ones and these blocks
const allowed = (ip: string, policy: Policy) => policy === true || isPublic(ip) || (toBig(ip) !== null && policy.some((b) => inBlock(toBig(ip)!, b)));
const refused = (host: string, ip: string) => Object.assign(new Error(
  `${host === ip ? ip : `${host} (${ip})`} is not a public address: Fetch URL reaches public addresses only (the config file's "fetch": {"allowPrivate": ...} can allow others)`),
  { code: "KILN_PRIVATE" });

interface Got { status: number; statusText: string; headers: http.IncomingHttpHeaders; body: Readable }
function get(url: URL, policy: Policy, signal: AbortSignal): Promise<Got> {
  const host = url.hostname.replace(/^\[|\]$/g, "");
  if (isIP(host) && !allowed(host, policy)) return Promise.reject(refused(host, host));
  return new Promise((resolve, reject) => {
    const req = (url.protocol === "https:" ? https : http).request(url, {
      method: "GET", signal,
      headers: { "User-Agent": "kiln-web (Fetch URL tool)", Accept: "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.5",
        "Accept-Encoding": "gzip, deflate, br" },
      lookup: (name: string, opts: any, cb: any) => dnsLookup(name, { all: true }, (err, addrs) => {
        if (err) return cb(err);
        const bad = addrs.find((a) => !allowed(a.address, policy));
        if (bad) return cb(refused(name, bad.address));
        if (opts?.all) cb(null, addrs);
        else cb(null, addrs[0].address, addrs[0].family);
      }),
    } as https.RequestOptions, (res) => {
      const enc = String(res.headers["content-encoding"] ?? "").toLowerCase();
      const body = enc === "gzip" ? res.pipe(createGunzip()) : enc === "deflate" ? res.pipe(createInflate()) : enc === "br" ? res.pipe(createBrotliDecompress()) : res;
      resolve({ status: res.statusCode ?? 0, statusText: res.statusMessage ?? "", headers: res.headers, body });
    });
    req.on("error", reject);
    req.end();
  });
}

async function read(body: Readable, max: number): Promise<{ bytes: Buffer; cut: boolean }> {
  const chunks: Buffer[] = [];
  let n = 0;
  for await (const c of body) {
    chunks.push(c);
    n += c.length;
    if (n > max) { body.destroy(); return { bytes: Buffer.concat(chunks).subarray(0, max), cut: true }; }
  }
  return { bytes: Buffer.concat(chunks), cut: false };
}

const ENTITIES: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", ndash: "–", mdash: "—", hellip: "…",
  lsquo: "‘", rsquo: "’", ldquo: "“", rdquo: "”", laquo: "«", raquo: "»", copy: "©", reg: "®", trade: "™", middot: "·", bull: "•",
  times: "×", divide: "÷", deg: "°", euro: "€", pound: "£", yen: "¥", cent: "¢", sect: "§", para: "¶", shy: "", zwj: "", zwnj: "",
  larr: "←", rarr: "→", uarr: "↑", darr: "↓", minus: "−", plusmn: "±", frac12: "½", eacute: "é", egrave: "è", agrave: "à", ouml: "ö", uuml: "ü", auml: "ä" };
const decode = (s: string) => s.replace(/&(#x[\da-f]+|#\d+|[a-z][a-z\d]*);/gi, (m, e: string) => {
  if (e[0] === "#") {
    const n = e[1] === "x" || e[1] === "X" ? parseInt(e.slice(2), 16) : Number(e.slice(1));
    return n > 0 && n < 0x110000 ? String.fromCodePoint(n) : m;
  }
  return ENTITIES[e.toLowerCase()] ?? m;
});

const VOID = new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr", "param"]);
const RAW = new Set(["script", "style", "textarea", "title", "xmp", "noscript", "template", "svg", "math", "iframe", "object", "canvas"]);
const SKIP = new Set(["nav", "aside", "footer", "form", "button", "select", "dialog", "head", "menu"]);
const BLOCK = new Set(["p", "div", "section", "article", "main", "header", "figure", "figcaption", "table", "ul", "ol", "dl", "blockquote",
  "pre", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "details", "summary", "address", "dt", "dd"]);
const AUTO = new Set(["p", "li", "tr", "td", "th", "dt", "dd", "option"]);
const TAG = /<!--[\s\S]*?-->|<![^>]*>|<\?[\s\S]*?\?>|<(\/?)([a-zA-Z][\w:-]*)((?:[^>"']|"[^"]*"|'[^']*')*)>|[^<]+|</g;
const attr = (attrs: string, name: string) => attrs.match(new RegExp(`(?:^|\\s)${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s>]+))`, "i"))?.slice(1).find((x) => x !== undefined);

// A page as compact markdown. Inside <main> (or <article>s) only, when the page has one: the rest is the site's
// header, menus and footer.
export function htmlText(html: string, base: string): { title: string; text: string } {
  const lower = html.toLowerCase(), title = decode(html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1] ?? "").replace(/\s+/g, " ").trim();
  const scoped = /<main[\s>]|role\s*=\s*["']?main/.test(lower) ? "main" : /<article[\s>]/.test(lower) ? "article" : "";
  // at: where its text starts in `out` (a link or quote is rewritten once it closes); n: an <ol>'s items so far
  const stack: { tag: string; skip: boolean; scope: boolean; at: number; href?: string; n: number }[] = [];
  let out = "", skip = 0, inScope = scoped ? 0 : 1, pre = 0;
  const brk = (s: string) => { if (!skip && inScope) out = out.replace(/[ \t]+$/, "") + s; };
  const emit = (s: string) => {
    if (skip || !inScope || !s) return;
    if (pre) { out += s; return; }
    s = s.replace(/\s+/g, " ");
    if (!out || /\s$/.test(out)) s = s.replace(/^ /, "");
    out += s;
  };
  const open = (tag: string, attrs: string) => {
    const hidden = /(?:^|\s)hidden(?:\s|=|$)/i.test(attrs) || /aria-hidden\s*=\s*["']?true/i.test(attrs) || /role\s*=\s*["']?(navigation|banner|contentinfo)/i.test(attrs);
    const isScope = scoped && (tag === scoped || (scoped === "main" && /role\s*=\s*["']?main/i.test(attrs)));
    const sk = SKIP.has(tag) || hidden || (tag === "header" && !stack.some((e) => e.tag === "article" || e.tag === "main"));
    const e = { tag, skip: sk, scope: !!isScope, at: out.length, href: undefined as string | undefined, n: 0 };
    if (sk) skip++;
    if (isScope) inScope++;
    stack.push(e);
    if (/^h[1-6]$/.test(tag)) { brk("\n\n"); emit(`${"#".repeat(Number(tag[1]))} `); }
    else if (tag === "li") {
      const list = [...stack].reverse().find((x) => x.tag === "ul" || x.tag === "ol");
      const depth = stack.filter((x) => x.tag === "ul" || x.tag === "ol").length;
      brk("\n" + "  ".repeat(Math.max(0, depth - 1)));
      emit(list?.tag === "ol" ? `${++list.n}. ` : "- ");
    } else if (tag === "pre") { brk("\n\n```\n"); pre++; }
    else if (tag === "br") { if (!skip && inScope) out += "\n"; }
    else if (tag === "hr") brk("\n\n---\n\n");
    else if (tag === "tr") brk("\n|");
    else if (tag === "td" || tag === "th") emit(" ");
    else if (tag === "code" && !pre) emit("`");
    else if (tag === "strong" || tag === "b") emit("**");
    else if (tag === "a") e.href = attr(attrs, "href");
    else if (tag === "img") { const alt = attr(attrs, "alt"); if (alt?.trim()) emit(`[image: ${decode(alt).trim()}]`); }
    else if (BLOCK.has(tag)) brk("\n\n");
    if (VOID.has(tag)) close(tag);
  };
  const close = (tag: string) => {
    const k = stack.map((e) => e.tag).lastIndexOf(tag);
    if (k < 0) return;
    while (stack.length > k) {
      const e = stack.pop()!;
      if (/^h[1-6]$/.test(e.tag) || e.tag === "p" || BLOCK.has(e.tag)) brk("\n\n");
      if (e.tag === "pre") { pre--; if (!skip && inScope) out = out.replace(/\n*$/, "\n```\n\n"); }
      else if (e.tag === "td" || e.tag === "th") emit(" |");
      else if (e.tag === "code" && !pre) emit("`");
      else if (e.tag === "strong" || e.tag === "b") emit("**");
      else if (e.tag === "blockquote" && !skip && inScope) out = out.slice(0, e.at) + out.slice(e.at).replace(/^(?=.)/gm, "> ");
      else if (e.tag === "a" && !skip && inScope && e.href && !/^\s*(javascript:|#)/i.test(e.href)) {
        const label = out.slice(e.at).trim();
        let href = decode(e.href);
        try { href = new URL(href, base).href; } catch { /* left as written */ }
        if (label) out = `${out.slice(0, e.at)}[${label}](${href})`;
      }
      if (e.skip) skip--;
      if (e.scope) inScope--;
    }
  };
  for (const m of html.matchAll(TAG)) {
    const [whole, slash, rawTag, attrs] = m;
    if (!rawTag) { if (whole[0] !== "<" || whole === "<") emit(decode(whole)); continue; }   // (not comments, doctypes)
    const tag = rawTag.toLowerCase();
    if (slash) { close(tag); continue; }
    if (AUTO.has(tag) || BLOCK.has(tag)) {
      const top = stack.at(-1)?.tag;
      if (top === "p" || (AUTO.has(tag) && top === tag)) close(top!);
    }
    if (RAW.has(tag)) {                               // its content is not markup (or not text): jump past its end
      const end = lower.indexOf(`</${tag}`, m.index! + whole.length);
      TAG.lastIndex = end < 0 ? html.length : lower.indexOf(">", end) + 1 || html.length;
      continue;
    }
    open(tag, attrs ?? "");
  }
  while (stack.length) close(stack.at(-1)!.tag);
  const text = out.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").replace(/^\|$/gm, "").trim();
  return { title, text };
}

const TEXTUAL = /^(text\/|application\/(json|xml|javascript|x-ndjson|yaml|x-yaml|toml|csv|ld\+json)|[^;]*\+(json|xml))/;
const charset = (type: string, head: Buffer) => type.match(/charset=["']?([\w-]+)/i)?.[1]
  ?? head.toString("latin1").match(/<meta[^>]+charset=["']?([\w-]+)/i)?.[1] ?? "utf-8";
const decodeAs = (bytes: Buffer, cs: string) => { try { return new TextDecoder(cs).decode(bytes); } catch { return new TextDecoder().decode(bytes); } };

export function fetchSource(cfg: Record<string, unknown>): Source {
  const allow = cfg.allowPrivate;
  const policy: Policy = allow === true ? true : Array.isArray(allow) ? allow.map((c) => block(String(c))).filter((b): b is [bigint, number] => !!b) : [];
  const tool: Tool = {
    name: "fetch_url", title: "Fetch URL", readOnly: true,
    description: "Reads one web page or text file by its URL (http or https, public addresses) and returns it as readable text: the "
      + "page's title, headings, paragraphs, lists, links, tables and code, with the final URL and the HTTP status. A long page "
      + "comes in parts: call again with `start` to read on. Use it when the user gives or names a page, or to read a search "
      + "result in full. Only fetch URLs that appeared in the conversation or in tool results.",
    parameters: { type: "object", properties: {
      url: { type: "string", description: "The http or https URL to fetch" },
      start: { type: "integer", description: "Where to start reading, in characters (for the rest of a long page)", minimum: 0 },
    }, required: ["url"] },
    async run(args, { signal, maxChars }) {
      let url: URL;
      try { url = new URL(String(args.url ?? "")); } catch { throw new Error(`not a URL: ${String(args.url ?? "")}`); }
      const first = url.href;
      let got: Got;
      for (let hop = 0; ; hop++) {
        if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error(`only http and https URLs can be fetched, not ${url.protocol}`);
        got = await get(url, policy, signal);
        const loc = got.headers.location;
        if (got.status < 300 || got.status >= 400 || !loc) break;
        got.body.resume();
        if (hop >= MAX_REDIRECTS) throw new Error(`more than ${MAX_REDIRECTS} redirects (the last to ${loc})`);
        url = new URL(loc, url);
      }
      const type = String(got.headers["content-type"] ?? "").toLowerCase(), mime = type.split(";")[0].trim();
      const html = mime === "text/html" || mime === "application/xhtml+xml";
      if (mime && !html && !TEXTUAL.test(mime)) {
        got.body.destroy();
        throw new Error(`${url.href} is ${mime}${got.headers["content-length"] ? ` (${got.headers["content-length"]} bytes)` : ""}: Fetch URL reads web pages and text only`);
      }
      const { bytes, cut: bytesCut } = await read(got.body, MAX_BYTES);
      const raw = decodeAs(bytes, charset(type, bytes.subarray(0, 2048)));
      if (!mime && raw.includes("\0")) throw new Error(`${url.href} is not text: Fetch URL reads web pages and text only`);
      const page = html ? htmlText(raw, url.href) : { title: "", text: raw };
      const start = Math.max(0, Math.floor(Number(args.start) || 0)), room = Math.max(500, maxChars - 600);
      const part = page.text.slice(start, start + room), end = start + part.length;
      const head = [`URL: ${url.href}`, ...(url.href !== first ? [`Redirected from: ${first}`] : []), `Status: ${got.status} ${got.statusText}`.trim(),
        ...(mime ? [`Content-Type: ${mime}`] : []), ...(page.title ? [`Title: ${page.title}`] : [])];
      const more = end < page.text.length ? `\n\n[Characters ${start.toLocaleString("en-US")}-${end.toLocaleString("en-US")} of ${page.text.length.toLocaleString("en-US")}: `
        + `call fetch_url again with start=${end} to read on.]` : bytesCut ? `\n\n[The page was longer than ${MAX_BYTES >> 20} MB: only its start was read.]` : "";
      return `${head.join("\n")}\n\n${start >= page.text.length && page.text ? `[Nothing at start=${start}: the text is ${page.text.length} characters.]` : part || "[No text on this page.]"}${more}`;
    },
  };
  return {
    id: "fetch", label: "Fetch URL", kind: "fetch", prefix: "", spec: JSON.stringify(cfg ?? {}),
    detail: policy === true ? "any address (allowPrivate: true)" : policy.length ? `public addresses and ${(allow as unknown[]).join(", ")}` : "public addresses only",
    state: "connected", tools: [tool],
    timeoutMs: (Number(cfg.timeout) > 0 ? Number(cfg.timeout) : DEFAULT_TIMEOUT_S) * 1000,
    maxChars: Number(cfg.maxResultChars) > 0 ? Number(cfg.maxResultChars) : 100_000, chars: DEFAULT_CHARS,
    connect: () => Promise.resolve(), close: () => Promise.resolve(),
  };
}
