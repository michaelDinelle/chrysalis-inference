// kiln web UI: chat with the local Qwen3.8-27B server. Replies stream in with their reasoning and are rendered
// as markdown with highlighted code; conversations live in this tab's sessionStorage (a reload keeps them,
// closing the tab ends them). The settings (system prompt, sampling), kept the same way, apply to every conversation.
import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/common";
import { marked } from "marked";

type Effort = "xhigh" | "medium" | "low" | "off";
interface Stats { ttft: number; tps: number; tokens: number; prompt: number; cached: number; thought?: number; seed?: number }
interface Message { role: "user" | "assistant"; content: string; reasoning?: string; stats?: Stats; error?: string; stopped?: boolean }
interface Conversation { id: string; title: string; named?: boolean; updated: number; effort: Effort; messages: Message[] }
type SamplingKey = "temperature" | "top_p" | "top_k" | "min_p" | "presence_penalty" | "max_tokens" | "seed";
type Sampling = Partial<Record<SamplingKey, number>>;
interface Settings { system: string; sampling: Sampling }
interface Status { state: "online" | "loading" | "offline"; build?: string; context?: number; kv_tokens?: number; cache?: { free_kv_tokens: number } }
interface View { el: HTMLElement; think: HTMLDetailsElement; summary: HTMLElement; thought: HTMLElement; body: HTMLElement; foot: HTMLElement; shown?: string }
interface Live { phase: "waiting" | "thinking" | "writing"; since: number; note: string }

const MODEL = "qwen3.8-27b";
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
let busy: AbortController | null = null;
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
    paint(v, m);
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
    v.body.innerHTML = `<p class="waiting">${live.note} · ${fmtS(elapsed)}</p>`;
    v.shown = undefined;
  } else if (v.shown !== m.content) {
    v.body.innerHTML = markdown(m.content);
    v.shown = m.content;
  }
  if (live) return;
  if (m.error) v.body.append(h("div", "error", m.error));
  const s = m.stats;
  const facts = s ? [`${fmtS(s.ttft)} to first token`, `${Math.round(s.tps)} tok/s`, `${fmtK(s.tokens)} tokens`, `prompt ${fmtK(s.cached)}/${fmtK(s.prompt)} cached`]
    : m.stopped ? ["stopped"] : [];
  if (s?.seed !== undefined) facts.push(`seed ${s.seed}`);
  const last = conv.messages.at(-1) === m;
  v.foot.innerHTML = `<span class="stats">${facts.join(" · ")}</span>` + actions(last ? ["copy", "retry"] : ["copy"]);
}

function emptyState() {
  const e = h("div", "empty"), grid = h("div", "suggestions");
  e.innerHTML = `<svg class="flame" viewBox="0 0 24 24">${ICONS.flame}</svg><h2>What are we building?</h2>` +
    "<p>Qwen3.8-27B on your Radeon · 262K context · nothing leaves this machine</p>";
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
  sendBtn.innerHTML = icon(busy ? "stop" : "send");
  sendBtn.setAttribute("aria-label", busy ? "Stop" : "Send");
  sendBtn.classList.toggle("stop", !!busy);
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
    const a = h("a", s.id === conv.id ? "conv active" : "conv"), del = h("button", "conv-del", "×");
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

function renderStatus() {
  const el = $("status"), line = h("div", "status-line");
  line.append(h("span", `dot ${status.state}`));
  el.replaceChildren(line);
  const button = (label: string, action: string) => {
    const b = h("button", "link-btn", label);
    b.type = "button";
    b.onclick = async () => { await fetch(`/api/kiln/${action}`, { method: "POST" }); await refreshStatus(); };
    return b;
  };
  if (status.state === "online") {
    const used = status.kv_tokens && status.cache ? 1 - status.cache.free_kv_tokens / status.kv_tokens : 0;
    const bar = h("div", "bar"), fill = h("i");
    fill.style.width = `${Math.round(used * 100)}%`;
    bar.append(fill);
    bar.title = `${fmtK(status.cache?.free_kv_tokens ?? 0)} of ${fmtK(status.kv_tokens ?? 0)} KV tokens free (cached conversations are dropped as needed)`;
    line.append(h("span", "", `Qwen3.8-27B · ${fmtK(status.context ?? 0)}`), button("Unload", "stop"));
    const kv = h("div", "status-line kv");
    kv.append(h("span", "", `Prompt cache ${Math.round(used * 100)}%`), bar);
    el.append(kv);
  } else if (status.state === "loading") {
    line.append(h("span", "", "Loading the model…"));
  } else {
    line.append(h("span", "", "kiln is stopped"), button("Start", "start"));
  }
}

// ---- talking to kiln ---------------------------------------------------------------------------------------

async function refreshStatus() {
  try { status = await (await fetch("/api/status")).json(); } catch { status = { state: "offline" }; }
  build ||= status.build ?? "";
  // The server has newer code: reload into it once nothing would be lost (chats are kept in sessionStorage).
  if (status.build && status.build !== build && !busy && !input.value && !settingsDialog.open) return location.reload();
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
    model: c.effort === "off" ? `${MODEL}-fast` : MODEL, reasoning_effort: c.effort === "off" ? undefined : c.effort,
    ...settings.sampling, messages, stream: true, stream_options: { include_usage: true },
  };
}

// Server-sent events of an OpenAI streaming response, as parsed JSON chunks.
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
      if (data === "[DONE]") return;
      if (data) yield JSON.parse(data);
    }
  }
}

async function ensureOnline(signal: AbortSignal) {
  if (status.state === "offline") await fetch("/api/kiln/start", { method: "POST" });
  for (const deadline = Date.now() + 180_000; status.state !== "online"; await refreshStatus()) {
    if (signal.aborted) throw new DOMException("stopped", "AbortError");
    if (Date.now() > deadline) throw new Error("kiln did not start: see `kiln logs`");
    await sleep(100);
  }
}

async function generate() {
  const c = conv, m: Message = { role: "assistant", content: "", reasoning: "" }, ctrl = new AbortController();
  c.messages.push(m);
  busy = ctrl;
  follow = true;
  render();
  const v = views.get(m)!, live: Live = { phase: "waiting", since: performance.now(), note: "Reading the prompt" };
  const timer = setInterval(() => { paint(v, m, live); if (follow) scroller.scrollTop = scroller.scrollHeight; }, 60);
  let t0 = performance.now(), tFirst = 0, tAnswer = 0, tLast = 0, usage: any, seed: number | undefined;
  try {
    if (status.state !== "online") {
      live.note = status.state === "loading" ? "Loading the model" : "Starting kiln";
      await ensureOnline(ctrl.signal);
      live.note = "Reading the prompt";
      live.since = t0 = performance.now();
    }
    const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request(c)), signal: ctrl.signal });
    if (!res.ok || !res.body) throw new Error((await res.json().catch(() => null))?.error?.message ?? `HTTP ${res.status}`);
    for await (const ev of events(res.body)) {
      const d = ev.choices?.[0]?.delta ?? {}, now = performance.now();
      if (d.reasoning_content) {
        if (live.phase !== "thinking") {
          live.phase = "thinking";
          live.since = now;
          v.think.open = true;
        }
        m.reasoning += d.reasoning_content;
      }
      if (d.content) {
        if (!tAnswer) {
          tAnswer = now;
          v.think.open = false;
        }
        live.phase = "writing";
        m.content += d.content;
      }
      if (d.reasoning_content || d.content) {
        tFirst ||= now;
        tLast = now;
      }
      if (ev.usage) usage = ev.usage;
      if (ev.seed !== undefined) seed = ev.seed;
    }
    if (usage && tFirst) {
      m.stats = {
        ttft: (tFirst - t0) / 1000, tps: usage.completion_tokens / Math.max((tLast - tFirst) / 1000, 1e-3),
        tokens: usage.completion_tokens, prompt: usage.prompt_tokens, cached: usage.prompt_tokens_details?.cached_tokens ?? 0,
        thought: m.reasoning && tAnswer ? (tAnswer - tFirst) / 1000 : undefined, seed,
      };
    }
  } catch (e) {
    if (ctrl.signal.aborted) m.stopped = true;
    else m.error = e instanceof Error ? e.message : String(e);
  } finally {
    clearInterval(timer);
    if (busy === ctrl) busy = null;              // a newer reply may have started after this one was stopped
  }
  c.updated = Date.now();
  if (c === conv) render();
  save(c);
  if (!c.named && !m.error) await name(c);
}

// Titles come from the -fast model with a title-generator prompt, which kiln runs as a low-priority request.
async function name(c: Conversation) {
  c.named = true;
  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: `${MODEL}-fast`, max_tokens: 24, messages: [
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
  if (!text || busy) return;
  input.value = "";
  grow();
  if (!conv.messages.length) conv.title = text.split("\n")[0].slice(0, 60);
  conv.messages.push({ role: "user", content: text });
  void generate();
}

function openConversation(id: string) {
  busy?.abort();
  conv = chats[id] ? structuredClone(chats[id]) : blank();
  document.body.classList.remove("nav-open");
  render();
  scroller.scrollTop = scroller.scrollHeight;
}

function newChat() {
  busy?.abort();
  conv = blank();
  history.replaceState(null, "", location.pathname);
  document.body.classList.remove("nav-open");
  render();
  input.focus();
}

function remove(id: string) {
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

function grow() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 240)}px`;
}

thread.addEventListener("click", (e) => {
  const t = e.target as HTMLElement, code = t.closest<HTMLElement>(".copy"), act = t.closest<HTMLElement>("[data-act]");
  if (code) return void copy(code.closest(".code")?.querySelector("code")?.textContent ?? "", code);
  if (!act) return;
  const i = Number(act.closest<HTMLElement>("article")?.dataset.i), m = conv.messages[i];
  if (act.dataset.act === "copy") return void copy(m.content, act);
  if (busy) return;
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
  if (busy) busy.abort();
  else submit(input.value);
});
input.addEventListener("input", grow);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    submit(input.value);
  }
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && busy) busy.abort(); });
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
if (location.hash.length > 1) openConversation(location.hash.slice(1));
else render();
// Every 4 s, or 4 times a second while the model loads (a load takes ~1.5 s).
(async () => {
  for (;;) {
    await sleep(status.state === "loading" ? 250 : 4000);
    if (!document.hidden) await refreshStatus();
  }
})();
