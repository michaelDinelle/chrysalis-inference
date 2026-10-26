import { type Theme, choose, chosen } from "./theme";
import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/common";
import katex from "katex";
import "katex/dist/katex.min.css";
import { marked, type Links, type Token, type Tokens } from "marked";
import { type Attachment, KINDS, attach, dataUrl, fmtBytes, get as getFile, put as putFile, textBlock } from "./files";
import { type Cand, type Odds, type SamplerKey, type Stage, DEFAULT_STAGES, sample } from "./sampler";
import { type Part, type VoiceAsk, type VoiceInfo, type VoiceReady, Speaker, VoiceClient } from "./voice";
import { type Fix, applyFixes, fixedParts, readHeard } from "./heard";
import { pointOf, shownWords, unitOf } from "./spoken";
import { DESCRIPTIONS, type ToolChoice, type WireCall, assistantTurn, noted, toolTurn } from "./wire";
import { type Block, type Span, type Version, ARTIFACTS_PROMPT, PAGE_DOC, elementById, elementSpans, idOf, artifactOf, infoOf, type Info, markSource, replay, setOpen, withAttr, withBackground,
  withContent, withCopy, withOrder, withPaint, withoutElement, lineDiff } from "./artifacts";
import { LATEX_CSS, latexToHtml } from "./latex";
import logo from "./logo.svg";
import { type CardData, type Group, type Person, type Voice, cardPng, castPrompt, dropGroup, dropPerson, fill as fillCard, loadCast, readCast, saveGroup, savePerson, speakerOf,
  voiceOf } from "./cast";
import { Ocean } from "./ocean";

type Effort = "xhigh" | "medium" | "low" | "off";
interface Stats { ttft: number; tps: number; tokens: number; prompt: number; cached?: number; thought?: number; seed?: number; model?: string; ctx?: number }
type CallState = "approval" | "running" | "done" | "failed" | "declined" | "stopped";
interface Call extends WireCall { title?: string; source?: string; readOnly?: boolean; state?: CallState; result?: string; ms?: number; agent?: AgentRun }
interface AgentRun { content: string; reasoning: string; calls?: Call[]; rounds?: Round[]; notes?: string[]; read?: boolean; tokens?: number; error?: string }
interface Round { content: string; reasoning?: string; calls: Call[] }
interface Versions { shown: number; others: Message[][] }
interface Message {
  id?: string; role: "user" | "assistant"; content: string; reasoning?: string; stats?: Stats; error?: string; stopped?: boolean; job?: string;
  params?: string;
  effort?: Effort;
  as?: string;
  finish?: string; edited?: boolean;
  files?: Attachment[];
  rounds?: Round[];
  calls?: Call[];
  base?: number;
  starting?: boolean;
  notes?: string[];
  versions?: Versions;
  voice?: boolean;
  voiceMs?: number;
  interrupted?: boolean;
  note?: string;
  heard?: Fix[];
  point?: string; pointAt?: string;
  fix?: boolean;
  edits?: { id: string; kind: string; src: string }[];
}
interface Conversation { id: string; title: string; named?: boolean; updated: number; effort: Effort; messages: Message[]; tools?: ToolChoice["tools"];
  pending?: Record<string, { kind: string; srcs: string[] }>;
  who?: string }
type Sampler = Partial<Record<SamplerKey, number>> & { order?: Stage[] };
type Mode = "think" | "fast";
interface Settings {
  v?: 5;
  system: string; think: Sampler; fast: Sampler; max_tokens?: number; seed?: number; allow?: string[]; fetchChars?: number; voice?: VoiceSettings;
  artifacts?: false;
}
interface VoiceSettings { tts?: string; name?: string; speed?: number; prompt?: string }
interface ToolInfo { fn: string; name: string; title: string; description: string; readOnly: boolean }
interface SourceInfo { id: string; label: string; kind: "fetch" | "agent" | "mcp"; detail: string; state: "idle" | "connecting" | "connected" | "failed" | "disabled"; error?: string; tools: ToolInfo[] }
interface Catalog { config: string; error?: string; sources: SourceInfo[] }
interface Status { state: "online" | "loading" | "offline"; build?: string; model?: string; loading?: string; progress?: number; context?: number; kv_tokens?: number; cache?: { free_kv_tokens: number };
  check?: boolean; here?: boolean; waiting?: number;
  backend?: Backend; link?: Linking }
interface Linking { linked?: { name: string }; found: { name: string; fp: string }[]; refused?: string;
  pairing?: { name: string; code: string; state: "waiting" | "allowed" | "gone" | "error"; error?: string; theirs?: boolean; confirmed?: boolean } }
interface Backend { url: string; chrysalis: boolean; service: boolean; key: boolean; remote: boolean; name: string;
  listed?: { ok: true; models: { id: string }[] } | { ok: false; error: string } }
interface View {
  el: HTMLElement; think: HTMLDetailsElement; summary: HTMLElement; thought: HTMLElement; body: HTMLElement; foot: HTMLElement; shown?: string; blocks?: Blocks;
  tok?: { reasoning: number; content: number };
  rounds: HTMLElement; calls: HTMLElement; notes: HTMLElement; roundsShown?: number;
}
interface Alt { t: string; p: number; id: number }
interface Tok { t: string; p: number; alts: Alt[] }
interface Live { phase: "waiting" | "thinking" | "writing"; since: number; note: string; loading?: boolean }
interface Run {
  conv: Conversation; m: Message; live: Live;
  at: number;
  stopping: boolean; gone?: boolean;
  usage?: { prompt_tokens: number; completion_tokens: number; cached?: number }; last?: any;
  read?: number; written?: number;
  seed?: number; timing?: { start: number; first?: number; answer?: number; last?: number; gen?: number };
  pick?: Tok;
  frame: number; saved: number;
}

const MODELS: [string, string, string][] = [
  ["qwen3.8-27b", "Qwen3.8-27B", "Quickest replies · 4 s to load"],
  ["qwen3.8-flash-next", "Qwen3.8-Flash-Next", "Fast at any length · 20 s to load"],
];
const modelName = (id?: string) => MODELS.find(([m]) => m === id)?.[1] ?? id?.split("/").pop()?.replace(/\.gguf$/i, "") ?? "";
const NEUTRAL = { dynatemp_range: 0, dynatemp_exponent: 1, typical_p: 1, xtc_probability: 0, xtc_threshold: 0.1, top_n_sigma: -1, min_keep: 0,
  repeat_last_n: 64, repeat_penalty: 1, frequency_penalty: 0, dry_multiplier: 0, dry_base: 1.75, dry_allowed_length: 2, dry_penalty_last_n: -1,
  mirostat: 0, mirostat_tau: 5, mirostat_eta: 0.1 };
const RECOMMENDED: Record<Mode, Record<SamplerKey, number>> = {
  think: { ...NEUTRAL, temperature: 1, top_p: 0.95, top_k: 20, min_p: 0, presence_penalty: 0 },
  fast: { ...NEUTRAL, temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0, presence_penalty: 1.5 },
};
const MAX_REPLY = 131072;
const STAGE_ALIASES: Record<string, Stage> = { penalties: "penalties", dry: "dry", top_n_sigma: "top_n_sigma", top_k: "top_k", typ_p: "typ_p",
  typical_p: "typ_p", typical: "typ_p", top_p: "top_p", nucleus: "top_p", min_p: "min_p", xtc: "xtc", temperature: "temperature", temp: "temperature" };
interface SamplerSpec { label: string; short: string; min: number; max: number; step: number; slider?: [number, number]; help: string; off?: number; all?: number }
const SAMPLERS: Record<SamplerKey, SamplerSpec> = {
  temperature: { label: "Temperature", short: "temp", min: 0, max: 5, step: 0.01, slider: [0, 2],
    help: "Controls the randomness of the generated text by affecting the probability distribution of the output tokens. Higher = more random, lower = more focused." },
  dynatemp_range: { label: "Dynamic temperature range", short: "dynatemp", min: 0, max: 5, step: 0.01, slider: [0, 2], off: 0,
    help: "Addon for the temperature sampler. The added value to the range of dynamic temperature, which adjusts probabilities by entropy of tokens." },
  dynatemp_exponent: { label: "Dynamic temperature exponent", short: "dynatemp-exp", min: 0, max: 10, step: 0.01, slider: [0, 3],
    help: "Addon for the temperature sampler. Smoothes out the probability redistribution based on the most probable token." },
  top_k: { label: "Top K", short: "top-k", min: 0, max: 64, step: 1, off: 0, help: "Keeps only k top tokens (Chrysalis samples from the 64 likeliest)." },
  top_p: { label: "Top P", short: "top-p", min: 0, max: 1, step: 0.01, help: "Limits tokens to those that together have a cumulative probability of at least p." },
  min_p: { label: "Min P", short: "min-p", min: 0, max: 1, step: 0.01, slider: [0, 0.5],
    help: "Limits tokens based on the minimum probability for a token to be considered, relative to the probability of the most likely token." },
  xtc_probability: { label: "XTC probability", short: "xtc", min: 0, max: 1, step: 0.01, off: 0,
    help: "XTC sampler cuts out top tokens; this parameter controls the chance of cutting tokens at all. 0 disables XTC." },
  xtc_threshold: { label: "XTC threshold", short: "xtc-t", min: 0, max: 1, step: 0.01, slider: [0, 0.5],
    help: "XTC sampler cuts out top tokens; this parameter controls the token probability that is required to cut that token." },
  typical_p: { label: "Typical P", short: "typ-p", min: 0, max: 1, step: 0.01, help: "Sorts and limits tokens based on the difference between log-probability and entropy." },
  top_n_sigma: { label: "Top n sigma", short: "top-nσ", min: -1, max: 10, step: 0.01, slider: [0, 4], off: -1,
    help: "Keeps tokens whose logit is within n standard deviations of the top logit. Off at -1." },
  min_keep: { label: "Min keep", short: "min-keep", min: 0, max: 64, step: 1, slider: [0, 16], help: "The samplers that cut tokens keep at least this many." },
  repeat_last_n: { label: "Repeat last N", short: "last-n", min: -1, max: 262144, step: 1, slider: [0, 2048], off: 0, all: -1,
    help: "Last n tokens to consider for penalizing repetition. 0 disables the penalties, -1 is the whole context." },
  repeat_penalty: { label: "Repeat penalty", short: "repeat", min: 0, max: 5, step: 0.01, slider: [1, 2], help: "Controls the repetition of token sequences in the generated text." },
  presence_penalty: { label: "Presence penalty", short: "presence", min: -2, max: 2, step: 0.05, slider: [0, 2],
    help: "Limits tokens based on whether they appear in the output or not." },
  frequency_penalty: { label: "Frequency penalty", short: "frequency", min: -2, max: 2, step: 0.05, slider: [0, 2],
    help: "Limits tokens based on how often they appear in the output." },
  dry_multiplier: { label: "DRY multiplier", short: "dry", min: 0, max: 5, step: 0.05, slider: [0, 2], off: 0,
    help: "DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling multiplier." },
  dry_base: { label: "DRY base", short: "dry-base", min: 1, max: 4, step: 0.05, slider: [1, 3],
    help: "DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling base value." },
  dry_allowed_length: { label: "DRY allowed length", short: "dry-len", min: 1, max: 20, step: 1, slider: [1, 8],
    help: "DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the allowed length for DRY sampling." },
  dry_penalty_last_n: { label: "DRY penalty last N", short: "dry-last-n", min: -1, max: 262144, step: 1, slider: [0, 8192], off: 0, all: -1,
    help: "DRY sampling reduces repetition in generated text even across long contexts. This parameter sets DRY penalty for the last n tokens. -1 is the whole context." },
  mirostat: { label: "Mirostat", short: "mirostat", min: 0, max: 2, step: 1, off: 0,
    help: "Mirostat replaces the other samplers with one that keeps the text's surprise near tau: 1 is Mirostat, 2 Mirostat 2.0." },
  mirostat_tau: { label: "Mirostat tau", short: "tau", min: 0, max: 20, step: 0.1, slider: [0, 10], help: "Mirostat's target entropy." },
  mirostat_eta: { label: "Mirostat eta", short: "eta", min: 0, max: 1, step: 0.01, help: "Mirostat's learning rate." },
};
const ODDS: [string, number][] = [["Luna", 703], ["Mochi", 470], ["Whiskers", 312], ["Milo", 182], ["Mittens", 78], ["Shadow", 39],
  ["Miso", 31], ["Whisper", 29], ["Moon", 14], ["Pumpkin", 14], ["Pixel", 13], ["Whisker", 11], ["Willow", 9], ["Muffin", 8], ["Echo", 7],
  ["Momo", 6], ["Jasper", 5], ["Nimbus", 5], ["Nebula", 4], ["Simba", 4], ["Noodle", 3], ["Misty", 3], ["Mocha", 3], ["Mimi", 2],
  ["Munchkin", 2], ["Zephyr", 2], ["Moss", 2]];
const DEFAULT_SYSTEM = "You are {{MODEL}}, running locally in Chrysalis Inference. There are no tools or files in this chat. Answer directly.";
const DEFAULT_SYSTEM_TOOLS = "You are {{MODEL}}, running locally in Chrysalis Inference. The tools offered here are the only ones in this chat, and there are no files or projects to explore. Use a tool when it helps; otherwise answer directly.";
const PLAIN_SYSTEM = "There are no tools or files in this chat. Answer directly.";
const DEFAULT_VOICE_PROMPT = "The user is talking to you by voice; a word may still be misheard, so take what they most likely meant, by " +
  "how it sounds and by the conversation, without remarking on it. Your reply is read aloud, except code blocks and tables, which are " +
  "shown and not read: speak briefly in plain sentences (but what they ask you to write, such as an essay or a story, in full), put code " +
  "and tables in markdown blocks and say what you put there, and do what is asked rather than asking whether to. When a question needs " +
  "counting or logic, work it out loud in a few short steps, then answer.";
const VOICE_TOOLS = "Only say you searched or did something once the tool call for it has returned, and never describe results you " +
  "have not received from a tool. When results come back, give the key points briefly, and say the links are on screen rather than reading them.";
const CHECK = "You fix speech recognition errors in what a user says to {{MODEL}}, a voice assistant running locally in Chrysalis Inference{{TOOLS}}. " +
  "For each transcript, write one line per misheard word or phrase: the transcript's exact words => the words that sound like them " +
  "which the user meant (names spelled right, numbers in digits). Read the transcript against the conversation: words that do not fit " +
  "what the user is asking were misheard, names of libraries, products and models as everyday words (\"pie torch\" for \"PyTorch\") " +
  "and everyday words as others that sound alike (\"read it\" for \"write it\", \"by\" for \"buy\"). Fix only what was misheard: " +
  "never rephrase, censor, explain or expand, and keep words that fit and names you have no reason to doubt. Write \"-\" when nothing " +
  "was misheard.";
const CHECK_TOOLS = " Then a last line, \"tool: yes\" if answering needs a web search or a page (they ask to search, look up or check " +
  "something, or the answer depends on current facts such as news, weather, prices or schedules), else \"tool: no\".";
const VOICE_OFF = "The user is typing again and reads your replies.";
const TITLE_PROMPT = "You are a title generator. Reply with a title of at most six words for the conversation that starts with the user's message below. Reply with the title only.";
const DRAFT_PROMPT = "Write the next message I would send in this conversation, as me: same voice, same goals, natural and brief. Reply with only that message.";
const fill = (text: string) => text.replaceAll("{{MODEL}}", modelName(current()) || "a Qwen model");
const EFFORTS: Effort[] = ["off", "low", "medium", "xhigh"];
const LEVEL: Record<Effort, [string]> = { off: ["Off"], low: ["Low"], medium: ["Medium"], xhigh: ["Max"] };
const HEAT = ["off", "low", "medium", "max"];
const ICONS = {
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
  prev: '<path d="M14.5 6.5 9 12l5.5 5.5"/>',
  next: '<path d="M9.5 6.5 15 12l-5.5 5.5"/>',
  wrench: '<path d="M14.7 6.3a4 4 0 0 0 5 5L21 12.6a6 6 0 0 1-7.4 1.6L7 20.8a2.1 2.1 0 0 1-3-3l6.6-6.6A6 6 0 0 1 12.2 3.8l1.3 1.3a4 4 0 0 0 1.2 1.2z"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.6 3.8 5.6 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.6-3.8-9S9.5 5.6 12 3z"/>',
  agent: '<circle cx="12" cy="5.5" r="2.2"/><circle cx="6" cy="18.5" r="2.2"/><circle cx="18" cy="18.5" r="2.2"/><path d="M12 7.7v3.8M6 16.3v-1.8a3 3 0 0 1 3-3h6a3 3 0 0 1 3 3v1.8"/>',
  wave: '<path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2"/>',
  play: '<path fill="currentColor" stroke="none" d="M8 5.8v12.4a.8.8 0 0 0 1.2.7l10-6.2a.8.8 0 0 0 0-1.4l-10-6.2a.8.8 0 0 0-1.2.7z"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/>',
  phone: '<rect x="6.5" y="2.5" width="11" height="19" rx="2.5"/><path d="M10.5 18.5h3"/>',
  shapes: '<rect x="3.5" y="10.5" width="10" height="10" rx="2"/><circle cx="15.5" cy="8.5" r="5"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
  page: '<rect x="3" y="4" width="18" height="16" rx="2.5"/><path d="M3 9h18M6.5 6.5h.01M9.5 6.5h.01"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2.5"/><circle cx="9" cy="10" r="1.8"/><path d="m21 16-5-5-9 9"/>',
  doc: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>',
  diagram: '<rect x="3" y="3" width="7" height="6" rx="1.5"/><rect x="14" y="15" width="7" height="6" rx="1.5"/><path d="M6.5 9v4.5a1.5 1.5 0 0 0 1.5 1.5h6"/>',
  code: '<path d="m8 8-4 4 4 4M16 8l4 4-4 4M13.5 5l-3 14"/>',
  speak: '<path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z"/><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/>',
  person: '<circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/>',
  personAdd: '<circle cx="10" cy="8" r="4"/><path d="M2.5 21c1.4-4 4.2-6 7.5-6 1.6 0 3 .5 4.3 1.3M19 14v6M16 17h6"/>',
  upload: '<path d="M12 15V4M7.5 8.5 12 4l4.5 4.5M5 20h14"/>',
  micOff: '<path d="M15 9.4V6a3 3 0 0 0-5.7-1.3M9 9v2a3 3 0 0 0 4.9 2.3M18.5 11a6.5 6.5 0 0 1-1 3.4M5.5 11a6.5 6.5 0 0 0 10.4 5.2M12 17.5V21M4 4l16 16"/>',
};
const TITLES = { copy: "Copy", edit: "Edit", trash: "Delete", tokens: "Token odds", retry: "Regenerate (a new version)", speak: "Read aloud" };
const icon = (name: keyof typeof ICONS) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const thread = $("thread"), scroller = $("scroller"), input = $<HTMLTextAreaElement>("input"), sendBtn = $<HTMLButtonElement>("send");
const panel = $<HTMLDialogElement>("settings"), systemInput = $<HTMLTextAreaElement>("system-input");
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
  artifacts: pane("artifacts", $("shelf"), ".", 1100, "--panel-open", 340, () => renderShelf()),
};
const PANES = Object.values(panes);

function h<K extends keyof HTMLElementTagNameMap>(tag: K, cls = "", text = ""): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

const people = new Map<string, Person>(), groups = new Map<string, Group>(), faces = new Map<string, string>();
const ME = "me", CHRYSALIS = "chrysalis";
people.set(CHRYSALIS, { id: CHRYSALIS, kind: "assistant", card: { spec: "chara_card_v2", spec_version: "2.0", data: { name: "Chrysalis" } }, added: 0 });
people.set(ME, { id: ME, kind: "persona", card: { spec: "chara_card_v2", spec_version: "2.0", data: { name: "User", description: "" } }, added: 0 });
const CAST_DRAG = "application/x-chrysalis-cast";
let talking = 0;
const persona = () => people.get(ME)!;
const userName = () => nameOf(ME);
const userCard = () => ({ ...persona().card.data, name: userName() });
const folderOf = (c: Conversation) => (c.who && c.who !== CHRYSALIS && (people.has(c.who) || groups.has(c.who)) ? c.who : "");
const whoOf = (c: Conversation) => (c.who && c.who !== CHRYSALIS ? c.who : CHRYSALIS);
function castOf(c: Conversation): { members: Person[]; group?: Group } {
  const id = folderOf(c), g = id ? groups.get(id) : undefined;
  if (!id) return { members: [] };
  return g ? { members: g.members.map((m) => people.get(m)).filter((p): p is Person => !!p), group: g } : { members: [people.get(id)!] };
}
const chatsOf = (id: string) => Object.values(chats).filter((s) => whoOf(s) === id).sort((a, b) => b.updated - a.updated);
const hue = (s: string) => [...s].reduce((a, ch) => (a * 31 + ch.charCodeAt(0)) % 360, 7);
function face(id: string): HTMLElement {
  const g = groups.get(id);
  if (g) {
    const ms = inGroup(g).slice(0, 4), s = h("span", `face mosaic m${Math.max(ms.length, 1)}`);
    s.append(...(ms.length ? ms : [initials("?")]));
    return s;
  }
  const url = faces.get(id), p = people.get(id), name = p?.card.data.name.trim() || "?";
  if (url) {
    const img = h("img", "face");
    img.src = url;
    img.alt = "";
    img.draggable = false;
    return img;
  }
  if (id === ME) {
    const s = h("span", "face self");
    s.innerHTML = icon("person");
    return s;
  }
  if (id === CHRYSALIS) {
    const s = h("span", "face chrysalis"), img = h("img");
    img.src = logo;
    img.alt = "";
    img.draggable = false;
    s.append(img);
    return s;
  }
  return initials(name);
}
function initials(name: string) {
  const s = h("span", "face initials", name.slice(0, 1).toUpperCase());
  s.style.setProperty("--hue", String(hue(name)));
  return s;
}
const inGroup = (g: Group) => [...g.members.map((m) => face(m)), ...(g.missing ?? []).map((n) => { const f = initials(n); f.title = `${n}: no card yet`; return f; })];
function setFace(p: Person) {
  const was = faces.get(p.id);
  if (was) URL.revokeObjectURL(was);
  if (p.image) faces.set(p.id, URL.createObjectURL(p.image));
  else faces.delete(p.id);
}
const nameOf = (id: string) => groups.get(id)?.name ?? (people.get(id)?.card.data.name.trim() || (id === CHRYSALIS ? "Chrysalis" : id === ME ? "User" : ""));
const names = (ids: string[]) => { const n = ids.map(nameOf); return n.length < 2 ? n.join("") : `${n.slice(0, -1).join(", ")} and ${n.at(-1)}`; };

function renderCast() {
  $("you").replaceChildren(face(ME), h("span", "you-name", nameOf(ME)));
  $("you").dataset.card = ME;
  renderList();
  renderCastNext();
}
const unfolded = new Set<string>((() => { try { return JSON.parse(pref("unfolded", `["${CHRYSALIS}"]`)) as string[]; } catch { return [CHRYSALIS]; } })());
function unfold(id: string, open = true) {
  if (open === unfolded.has(id)) return;
  if (open) unfolded.add(id); else unfolded.delete(id);
  setPref("unfolded", JSON.stringify([...unfolded]));
}
function castRow(id: string, list: Conversation[], open: boolean): HTMLElement {
  const g = groups.get(id), name = nameOf(id);
  const row = h("div", `cast-row${list.some((s) => runs.has(s.id)) ? " running" : ""}`);
  const pic = h("button", "cast-pic"), main = h("button", "cast-main"), add = h("button", "act cast-new");
  pic.type = main.type = add.type = "button";
  pic.title = pic.ariaLabel = g ? `${name}: who is in it` : `${name}'s card`;
  pic.append(face(id));
  pic.dataset.card = id;
  main.append(h("span", "cast-name", name || "Unnamed"));
  if (list.length) {
    main.insertAdjacentHTML("beforeend", icon("chevron"));
    main.setAttribute("aria-expanded", String(open));
    if (!open) main.append(h("span", "cast-count", String(list.length)));
  }
  main.onclick = () => {
    if (!list.length) return newChat(id);
    unfold(id, !open);
    renderList();
  };
  add.title = add.ariaLabel = `New chat with ${name}`;
  add.innerHTML = icon("plus");
  add.onclick = () => newChat(id);
  row.append(pic, main, add);
  if (people.get(id)?.kind === "character") {
    row.draggable = true;
    row.ondragstart = (e) => { e.dataTransfer!.setData(CAST_DRAG, id); e.dataTransfer!.effectAllowed = "copy"; };
  }
  return row;
}
function pickFolder(id: string) {
  const last = chatsOf(id)[0];
  unfold(id);
  if (last && last.id !== conv.id) location.hash = last.id;
  else if (!last && whoOf(conv) !== id) newChat(id);
  else closeOver(panes.conversations);
}
function writerOf(c: Conversation): Person | undefined {
  const { members, group } = castOf(c);
  return group ? undefined : members[0] ?? (c.who ? undefined : people.get(CHRYSALIS));
}
function greet(c: Conversation) {
  const p = writerOf(c);
  if (!p) return;
  const d = p.card.data, say = (t: string): Message => ({ id: randomId(), role: "assistant", content: fillCard(t, d.name, userName()), reasoning: "" });
  const all = [d.first_mes ?? "", ...(d.alternate_greetings ?? [])].filter((t) => t.trim());
  if (!all.length) return;
  const first = say(all[0]);
  if (all.length > 1) first.versions = { shown: 0, others: all.slice(1).map((t) => [say(t)]) };
  c.messages.push(first);
}
function castSystem(c: Conversation) {
  const { members, group } = castOf(c);
  const base = members.length > 1 ? "Write the next part of a fictional scene between {{user}} and these characters."
    : "Write {{char}}'s next reply in a fictional chat between {{char}} and {{user}}.";
  return castPrompt(base, members.map((p) => p.card.data), userCard(), group?.scenario);
}

async function adopt(f: File): Promise<string | null> {
  const got = await readCast(f).catch(() => null);
  if (!got) return null;
  if ("group" in got) {
    const ids = got.group.members.map((n) => [...people.values()].find((p) => p.kind === "character" && p.card.data.name === n)?.id);
    const missing = got.group.members.filter((_, i) => !ids[i]);
    const g: Group = { id: randomId(), name: got.group.name, members: ids.filter((x): x is string => !!x), ...(missing.length ? { missing } : {}),
      scenario: got.group.scenario, added: Date.now() };
    groups.set(g.id, g);
    await saveGroup(g);
    if (missing.length) toast(`${g.name}: no card yet for ${missing.join(", ")}`);
    return g.id;
  }
  const d = got.card.data, had = [...people.values()].find((p) => p.kind === "character" && p.card.data.name === d.name
    && (p.card.data.description ?? "") === (d.description ?? ""));
  if (had) return had.id;
  const p: Person = { id: randomId(), kind: "character", card: got.card, image: got.image, added: Date.now() };
  people.set(p.id, p);
  setFace(p);
  await savePerson(p);
  for (const g of groups.values()) if (g.missing?.includes(d.name)) {
    g.members = [...g.members, p.id];
    g.missing = g.missing.filter((n) => n !== d.name);
    if (!g.missing.length) delete g.missing;
    await saveGroup(g);
  }
  return p.id;
}
async function welcome(files: Iterable<File>, into: boolean): Promise<File[]> {
  const had = new Set([...people.keys(), ...groups.keys()]), ids: string[] = [], rest: File[] = [];
  for (const f of files) {
    const id = await adopt(f);
    if (id) ids.push(id); else rest.push(f);
  }
  const them = ids.filter((id) => people.has(id)), fresh = ids.filter((id) => !had.has(id));
  if (into && them.length && folderOf(conv)) join(them);
  else if (ids.length) {
    pickFolder(ids[0]);
    renderCast();
    if (fresh.length) toast(`${names(fresh)} ${fresh.length > 1 ? "are" : "is"} here`, () => { for (const id of fresh) void forget(id); });
  }
  return rest;
}
function join(ids: string[]) {
  const c = conv, { members, group } = castOf(c), fresh = [...new Set(ids)].filter((id) => !members.some((m) => m.id === id));
  if (!fresh.length) return toast(`${names(ids)} ${ids.length > 1 ? "are" : "is"} in this chat already`);
  if (group) {
    const before = { members: group.members, name: group.name };
    group.members = [...before.members, ...fresh];
    if (before.name === names(before.members)) group.name = names(group.members);
    void saveGroup(group);
    toast(`${names(fresh)} joined ${before.name}`, () => { Object.assign(group, before); void saveGroup(group); renderCast(); render(); });
  } else {
    const one = members[0].id, g: Group = { id: randomId(), name: names([one, ...fresh]), members: [one, ...fresh], added: Date.now() };
    for (const m of c.messages) if (m.role === "assistant") m.as ??= one;
    groups.set(g.id, g);
    void saveGroup(g);
    c.who = g.id;
    save(c);
    toast(`${names(fresh)} joined: a group now`, () => {
      c.who = one;
      save(c);
      groups.delete(g.id);
      void dropGroup(g.id);
      renderCast();
      render();
    });
  }
  renderCast();
  render();
}
async function forget(id: string) {
  const p = people.get(id), g = groups.get(id), name = nameOf(id), gone = Object.values(chats).filter((c) => c.who === id);
  const was = [...groups.values()].filter((x) => x.members.includes(id)).map((x) => ({ x, members: x.members }));
  if (g) { groups.delete(id); await dropGroup(id); }
  if (p) { people.delete(id); await dropPerson(id); }
  for (const { x } of was) { x.members = x.members.filter((m) => m !== id); await saveGroup(x); }
  dropChats(gone.map((c) => c.id));
  renderCast();
  return { name, chats: gone.length, undo: async () => {
    if (p) { people.set(id, p); setFace(p); await savePerson(p); }
    if (g) { groups.set(id, g); await saveGroup(g); }
    for (const { x, members } of was) { x.members = members; await saveGroup(x); }
    for (const c of gone) chats[c.id] = c;
    store("chats", chats);
    renderCast();
    render();
  } };
}
function dropChats(ids: string[]) {
  for (const id of ids) {
    const r = runs.get(id);
    if (r) { r.gone = true; stopRun(r); }
    delete chats[id];
  }
  store("chats", chats);
  if (ids.includes(conv.id) || (conv.who && !folderOf(conv))) newChat();
  else render();
}
const castless = (c: Conversation) => !!c.who && c.who !== CHRYSALIS && !people.has(c.who) && !groups.has(c.who);

function openAdd(at: HTMLElement) {
  const menu = $("cast-add"), r = at.getBoundingClientRect();
  menu.hidden = false;
  menu.style.left = `${r.left}px`;
  menu.style.top = `${r.bottom + 6}px`;
}
for (const [add, svg] of [["card", icon("personAdd")], ["group", icon("agent")]]) $("cast-add").querySelector(`[data-add="${add}"]`)!.insertAdjacentHTML("afterbegin", svg);
$("cast-add").addEventListener("click", async (e) => {
  const b = (e.target as HTMLElement).closest<HTMLElement>("[data-add]");
  if (!b) return;
  $("cast-add").hidden = true;
  if (b.dataset.add === "card") return $("cast-file").click();
  const g: Group = { id: randomId(), name: "New group", members: [], added: Date.now() };
  groups.set(g.id, g);
  await saveGroup(g);
  pickFolder(g.id);
  renderCast();
  openSheet(g.id);
});
$("cast-file").addEventListener("change", async (e) => {
  const input = e.currentTarget as HTMLInputElement;
  for (const f of await welcome(input.files ?? [], false)) toast(`${f.name} holds no character card`);
  input.value = "";
});

let sheetOf = "", sampler: Speaker | undefined, sheetFile: { id: string; url: string } | undefined;
function openSheet(id: string) {
  const sheet = $("cast-sheet"), g = groups.get(id);
  const person = people.get(id);
  if (!person && !g) return;
  const again = sheetOf === id, top = sheet.querySelector(".card-scroll")?.scrollTop ?? 0;
  sheetOf = id;
  let saving = 0;
  const keep = () => {
    clearTimeout(saving);
    saving = window.setTimeout(() => { void (person ? savePerson(person) : saveGroup(g!)); if (person) void cardFile(person); }, 300);
  };
  const btn = (cls: string, title: string, html: string, onclick: () => void) => {
    const b = h("button", cls);
    b.type = "button";
    b.title = b.ariaLabel = title;
    b.innerHTML = html;
    b.onclick = onclick;
    return b;
  };
  const picture = (p: Person) => {
    const f = h("input");
    f.type = "file";
    f.accept = "image/*";
    f.onchange = () => {
      const file = f.files?.[0];
      if (!file) return;
      p.image = file;
      keep();
      setFace(p);
      if (sheetFile) URL.revokeObjectURL(sheetFile.url);
      sheetFile = undefined;
      renderCast();
      render();
      openSheet(id);
    };
    f.click();
  };
  const grow = (area: HTMLTextAreaElement) => { area.style.height = "auto"; area.style.height = `${area.scrollHeight + 2}px`; };
  const text = (value: string, set: (v: string) => void, cls: string, placeholder = "") => {
    const area = h("textarea", cls);
    area.value = value;
    area.placeholder = placeholder;
    area.rows = 1;
    area.spellcheck = false;
    area.oninput = () => { set(area.value); grow(area); keep(); };
    requestAnimationFrame(() => grow(area));
    return area;
  };
  const n = Object.values(chats).filter((c) => c.who === id).length, theirs = n ? `, with ${n} chat${n > 1 ? "s" : ""}` : "";
  const del = () => {
    const b = btn("icon", `Delete ${nameOf(id) || "them"}${n ? " and their chats" : ""}`, icon("trash"), async () => {
      if (!b.classList.contains("armed")) { b.classList.add("armed"); b.innerHTML = `${icon("trash")}Delete${theirs}?`; return; }
      closeSheet();
      const f = await forget(id);
      toast(`${f.name} deleted${theirs}`, () => void f.undo());
    });
    return b;
  };

  const art = h("div", "card-art"), title = h("div", "card-title"), body = h("div", "card-body"), scroll = h("div", "card-scroll");
  if (g) {
    const strips = h("div", "strips");
    const all = inGroup(g).slice(0, 5);
    strips.append(...(all.length ? all : [face(id)]));
    art.append(strips);
  } else if (person!.kind === "assistant" && !person!.image && seaPortrait()) {
    art.append(cardSea!.layer);
    scroll.addEventListener("scroll", () => cardSea!.ocean.place(), { passive: true });
  } else {
    const file = h("a", "card-file"), pic = person!.image ? h("img") : face(id);
    file.download = file.title = cardName(id);
    file.draggable = false;
    if (sheetFile?.id === id) file.href = sheetFile.url;
    if (pic instanceof HTMLImageElement) {
      pic.src = sheetFile?.id === id ? sheetFile.url : faces.get(id)!;
      pic.alt = "";
      pic.draggable = false;
    }
    file.append(pic);
    art.append(file);
    if (sheetFile?.id !== id) void cardFile(person!);
  }
  art.append(h("div", "card-scrim"), title);
  if (person) art.append(btn("card-float card-pic", person.image ? "Another picture" : "A picture", icon("edit"), () => picture(person)));
  const name = h("input", "card-name");
  name.value = person ? person.card.data.name : g!.name;
  name.placeholder = id === ME ? "User" : id === CHRYSALIS ? "Chrysalis" : g ? "The group's name" : "Their name";
  name.ariaLabel = "Name";
  name.spellcheck = false;
  name.oninput = () => {
    if (person) person.card.data.name = name.value; else g!.name = name.value;
    const file = art.querySelector<HTMLAnchorElement>("a.card-file");
    if (file) file.download = file.title = cardName(id);
    keep();
    renderCast();
    if (castOf(conv).members.some((m) => m.id === id) || folderOf(conv) === id) render();
  };
  const d = person?.card.data;
  title.append(name);
  if (d) title.append(h("div", "card-tags", [d.creator ? `by ${d.creator}` : "", ...(d.tags ?? []).slice(0, 3)].filter(Boolean).join(" · ")));
  scroll.append(art, body);

  if (person) {
    const field = (label: string, k: keyof CardData, placeholder: string) => {
      const f = h("label", "card-field");
      f.append(h("span", "card-label", label), text(String(d![k] ?? ""), (v) => { d![k] = v; }, "card-text", placeholder));
      return f;
    };
    const more = h("details", "card-more");
    more.append(h("summary", "", "More"), field("How they talk", "mes_example", "Example lines: {{user}}: … {{char}}: …"),
      field("After the chat", "post_history_instructions", "A note after the last message"));
    body.append(voicePick(person, keep), field("Description", "description", person.kind === "assistant" ? "Who they are, after the system prompt" : "Who they are: looks, past, manner"),
      field("Personality", "personality", "A few words"), field("Scenario", "scenario", "Where and when the chat begins"),
      field("Greeting", "first_mes", "Their first message"), more);
    if (person.kind === "character") {
      const acts = h("div", "card-acts");
      acts.append(del());
      body.append(acts);
    }
  } else {
    const members = h("div", "card-members");
    members.append(...[...people.values()].filter((x) => x.kind === "character").map((x) => {
      const inside = g!.members.includes(x.id), b = h("button", `card-member ${inside ? "in" : "out"}`);
      b.type = "button";
      b.title = `${x.card.data.name}: ${inside ? "in the group (click to take out)" : "click to add"}`;
      b.append(face(x.id));
      b.onclick = () => {
        g!.members = inside ? g!.members.filter((m) => m !== x.id) : [...g!.members, x.id];
        keep();
        renderCast();
        render();
        openSheet(id);
      };
      return b;
    }), ...(g!.missing ?? []).map((n) => {
      const m = h("span", "card-member waiting");
      m.title = `${n}: no card yet (theirs joins the group once it is added)`;
      m.append(initials(n));
      return m;
    }));
    const who = h("div", "card-section"), where = h("label", "card-field"), acts = h("div", "card-acts");
    who.append(h("span", "card-label", "Who is in it"), members);
    where.append(h("span", "card-label", "Scenario"), text(g!.scenario ?? "", (v) => { g!.scenario = v || undefined; }, "card-text",
      "Where they are and what is going on (in place of their own)"));
    acts.append(btn("", "The group as a file (with its members' cards beside it, it imports anywhere this chat runs)", `${icon("download")}Export`,
      () => download(`${g!.name || "group"}.json`, JSON.stringify({ spec: "chrysalis_group", name: g!.name,
        members: [...g!.members.map((m) => people.get(m)?.card.data.name).filter(Boolean), ...(g!.missing ?? [])], ...(g!.scenario ? { scenario: g!.scenario } : {}) }, null, 2), "application/json")),
      del());
    body.append(who, where, acts);
  }
  sheet.replaceChildren(btn("card-float card-x", "Close", icon("x"), closeSheet), scroll);
  sheet.hidden = false;
  if (again) scroll.scrollTop = top;
  const side = $("sidebar").getBoundingClientRect();
  sheet.style.left = `${Math.max(12, Math.min(side.right + 10, innerWidth - sheet.offsetWidth - 12))}px`;
  sheet.style.top = `${Math.max(12, Math.min(64, innerHeight - sheet.offsetHeight - 12))}px`;
}
let cardSea: { layer: HTMLElement; ocean: Ocean; ok: boolean } | undefined;
function seaPortrait() {
  if (!seaReady) return false;
  if (!cardSea) {
    const layer = h("div", "card-sea"), ocean = new Ocean(layer, { box: "", hint: "", room: "#cast-sheet .card-sea-room" });
    layer.append(h("div", "card-sea-room"));
    cardSea = { layer, ocean, ok: ocean.start() };
    ocean.butterfly(layer);
  }
  return cardSea.ok;
}
const samples = () => (sampler ??= new Speaker((m) => { if (sampler?.previewed) toast(m); }));
async function cardFile(p: Person) {
  const url = URL.createObjectURL(await cardPng(p.card, p.image));
  if (sheetOf !== p.id) return URL.revokeObjectURL(url);
  if (sheetFile) URL.revokeObjectURL(sheetFile.url);
  sheetFile = { id: p.id, url };
  const file = $("cast-sheet").querySelector<HTMLAnchorElement>("a.card-file");
  if (!file) return;
  file.href = url;
  file.querySelector("img")?.setAttribute("src", url);
}
const cardName = (id: string) => `${nameOf(id) || "character"}.png`;
function closeSheet() {
  $("cast-sheet").hidden = true;
  if (sheetFile) URL.revokeObjectURL(sheetFile.url);
  sheetFile = undefined;
  sheetOf = "";
  sampler?.close();
  sampler = undefined;
}
function voicePick(p: Person, keep: () => void) {
  const box = h("div", "card-voice"), sel = h("select"), play = h("button", "act");
  const assistant = p.kind === "assistant";
  const hear = () => {
    const [tts, name] = sel.value ? sel.value.split("\n") : assistant ? [] : [settings.voice?.tts, settings.voice?.name];
    samples().preview(tts, name);
  };
  const fill = (none_ = "Voices…") => {
    const mine = assistant ? { tts: settings.voice?.tts, name: settings.voice?.name } : voiceOf(p.card.data);
    const models = (voiceCatalog.tts ?? []).filter((m) => m.installed && m.voices?.length);
    const none = h("option", "", models.length ? (assistant ? "The default voice" : `${nameOf(CHRYSALIS)}'s voice`) : none_);
    none.value = "";
    sel.replaceChildren(none, ...models.map((m) => {
      const group = h("optgroup");
      group.label = m.label;
      for (const x of m.voices!) {
        const o = h("option", "", `${x.label} · ${voiceGroup(x)}`);
        o.value = `${m.name}\n${x.id}`;
        o.selected = mine?.tts === m.name && mine?.name === x.id;
        group.append(o);
      }
      return group;
    }));
    play.disabled = !models.length;
  };
  box.title = "The voice they speak with in voice mode";
  sel.ariaLabel = "Voice";
  play.type = "button";
  play.title = play.ariaLabel = "Hear it";
  play.innerHTML = icon("play");
  play.onclick = hear;
  sel.onchange = () => {
    const [tts, name] = sel.value.split("\n");
    if (assistant) {
      setVoice({ tts: tts || undefined, name: name || undefined });
      saveSettings();
    } else {
      const ext = (p.card.data.extensions ??= {}), own = (ext.chrysalis ??= {}) as { voice?: Voice };
      if (sel.value) own.voice = { tts, name }; else delete own.voice;
      keep();
    }
    if (voice && castOf(voice.c).members.some((m) => m.id === p.id)) pushVoice();
    hear();
  };
  fill();
  if (!voiceCatalog.tts?.length) {
    void samples().catalog().then((c) => {
      if (c?.tts?.length) {
        voiceCatalog = c;
        setPref("voice-catalog", JSON.stringify(c));
      }
      if (box.isConnected) fill("No voices");
    });
  }
  box.insertAdjacentHTML("afterbegin", icon("wave"));
  box.append(sel, play);
  return box;
}
document.addEventListener("pointerdown", (e) => {
  const t = e.target as HTMLElement;
  if (!$("cast-add").hidden && !t.closest("#cast-add, #add-cast")) $("cast-add").hidden = true;
  if (!$("cast-sheet").hidden && !t.closest("#cast-sheet, [data-card]")) closeSheet();
});
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("cast-add").hidden) { $("cast-add").hidden = true; e.preventDefault(); }
  else if (!$("cast-sheet").hidden) { closeSheet(); e.preventDefault(); }
}, true);
document.addEventListener("click", (e) => {
  const id = (e.target as HTMLElement).closest<HTMLElement>("[data-card]")?.dataset.card;
  if (!id) return;
  if (sheetOf === id) closeSheet(); else openSheet(id);
});

function sceneText(src: string, c: Conversation) {
  const { members, group } = castOf(c);
  if (!group || !members.length) return src;
  const names = members.map((m) => m.card.data.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  return src.replace(new RegExp(`(?<!\\n)\\n(?=\\s*(?:\\*\\*)?(?:${names})(?:\\*\\*)?\\s*:)`, "g"), "\n\n");
}
function speakLine(n: Node | undefined, c: Conversation, before: Node | undefined, as?: string) {
  const { members, group } = castOf(c);
  if (!group || !(n instanceof HTMLParagraphElement)) return;
  const said = speakerOf(n.textContent ?? "", members.map((m) => m.card.data.name));
  const m = said ? members.find((x) => x.card.data.name === said.name)
    : people.get((before ? before instanceof HTMLElement && before.dataset.who : as) || "");
  if (!m) return;
  if (said) {
    let drop = said.rest;
    const walker = document.createTreeWalker(n, NodeFilter.SHOW_TEXT);
    for (let t = walker.nextNode(); t && drop > 0; t = walker.nextNode()) {
      const k = Math.min(drop, t.textContent!.length);
      t.textContent = t.textContent!.slice(k);
      drop -= k;
    }
    for (const el of [...n.querySelectorAll("strong, b")]) if (!el.textContent) el.remove();
  }
  n.dataset.who = m.id;
  n.classList.add("line");
  if (before instanceof HTMLElement && before.dataset.who === m.id) return void n.classList.add("same");
  const tag = h("span", "who"), name = m.card.data.name;
  tag.dataset.card = m.id;
  tag.title = `${name}'s card`;
  tag.append(face(m.id), h("b", "", name));
  n.prepend(tag);
}
function speakerHead(c: Conversation) {
  const w = writerOf(c);
  return w ? head(w.id) : null;
}
function head(id: string) {
  const el = h("div", "speaker"), name = nameOf(id);
  el.dataset.card = id;
  el.title = `${name}'s card`;
  el.append(face(id), h("span", "", name));
  return el;
}

function renderCastNext() {
  const box = $("cast-next"), { members, group } = castOf(conv);
  input.placeholder = members.length ? `Message ${group ? group.name : members[0].card.data.name}` : "Message Chrysalis";
  box.hidden = !group || !members.length;
  if (box.hidden) return;
  const talk = h("button", `talk${talking ? " on" : ""}`);
  talk.type = "button";
  talk.title = talking ? "Stop them" : "They go on by themselves for a few turns (typing stops them)";
  talk.innerHTML = talking ? `${icon("stop")}Stop them` : `${icon("play")}Let them talk`;
  talk.onclick = () => {
    if (talking) { talking = 0; return renderCastNext(); }
    talking = 4;
    renderCastNext();
    talkOn();
  };
  box.replaceChildren(h("span", "", "Next"), ...members.map((m) => {
    const b = h("button");
    b.type = "button";
    b.title = b.ariaLabel = `${m.card.data.name} next`;
    b.append(face(m.id));
    b.onclick = () => replyAs(m);
    return b;
  }), talk);
}
function replyAs(p: Person, auto = false) {
  if (running()) return;
  if (!auto) talking = 0;
  const text = input.value.trim();
  if (text) submit(text, false);
  void generate(conv, conv.messages.length, p.id);
  renderCastNext();
}
let reader: Speaker | undefined, reading: Message | undefined, readerIdle = 0;
function readAloud(m: Message) {
  const again = reading === m;
  stopReading();
  if (!again) {
    const c = conv, { members, group } = castOf(c), s = settings.voice ?? {};
    const base: VoiceAsk = { ...(s.tts ? { stack: { tts: s.tts } } : {}), ...(s.name ? { voice: s.name } : {}), speed: s.speed ?? 1 };
    const parts: { text: string; who?: Person }[] = [];
    if (m.role === "user") parts.push({ text: m.content, who: persona() });
    else if (!group) parts.push({ text: m.content, who: writerOf(c) });
    else {
      const names = members.map((x) => x.card.data.name);
      let who = m.as ? people.get(m.as) : undefined;
      for (const para of sceneText(m.content, c).split(/\n{2,}/)) {
        const said = speakerOf(para, names), last = parts.at(-1);
        if (said) who = members.find((x) => x.card.data.name === said.name);
        const text = said ? para.slice(said.rest) : para;
        if (last && last.who === who) last.text += `\n\n${text}`;
        else parts.push({ text, who });
      }
    }
    reading = m;
    clearTimeout(readerIdle);
    void (reader ??= new Speaker((e) => toast(e))).read(parts.map((x) => ({ text: x.text, voice: x.who && voiceOf(x.who.card.data) })), base, () => {
      reading = undefined;
      paintReading();
    });
  }
  paintReading();
}
function stopReading() {
  reader?.stop();
  reading = undefined;
  paintReading();
}
function paintReading() {
  for (const b of thread.querySelectorAll<HTMLElement>("[data-act=speak]")) {
    const on = !!reading && conv.messages[Number(b.closest<HTMLElement>("article")?.dataset.i)] === reading;
    b.classList.toggle("on", on);
    b.title = on ? "Stop reading" : TITLES.speak;
    b.innerHTML = icon(on ? "stop" : "speak");
  }
  clearTimeout(readerIdle);
  if (!reading && reader) readerIdle = window.setTimeout(() => { reader?.close(); reader = undefined; }, 60_000);
}

function talkOn() {
  const { members, group } = castOf(conv);
  if (!talking || running() || !group || !members.length) return;
  talking--;
  const said = conv.messages.filter((m) => m.role === "assistant").map((m) => sceneText(m.content, conv)).join("\n\n").split(/\n\n/);
  const lastSpoke = [...said].reverse().map((t) => speakerOf(t, members.map((m) => m.card.data.name))?.name).find(Boolean);
  const next = members.length > 1 ? members.filter((m) => m.card.data.name !== lastSpoke) : members;
  replyAs(next[Math.floor(Math.random() * next.length)], true);
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
    br: () => "<br>\n",
  },
});
const markdown = (src: string) => DOMPurify.sanitize(marked.parse(src, { async: false }));

const QUOTES: Record<string, string> = { '"': '"', "“": "”", "«": "»", "「": "」", "『": "』", "＂": "＂" };
const chatTokens = (src: string) => marked.lexer(src, { ...marked.defaults, breaks: true });
marked.use({
  extensions: [{
    name: "quote", level: "inline",
    start: (src) => src.match(/["“«「『＂]/)?.index,
    tokenizer(src) {
      const close = QUOTES[src[0]];
      if (!close || !this.lexer.options.breaks) return;
      const nl = src.indexOf("\n"), line = nl < 0 ? src : src.slice(0, nl), end = line.indexOf(close, 1);
      const text = end < 0 ? line.slice(1) : line.slice(1, end);
      return { type: "quote", raw: end < 0 ? line : line.slice(0, end + 1), open: src[0], close: end < 0 ? "" : close, tokens: this.lexer.inlineTokens(text) };
    },
    renderer(t) { return `<q>${escapeHtml(t.open)}${this.parser.parseInline(t.tokens!)}${escapeHtml(t.close)}</q>`; },
  }],
});

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

interface Blocks { raws: string[]; nodes: Node[][] }

const FENCE = /^ {0,3}(?:`{3,}|~{3,})/gm;
function settle(src: string): string {
  const nl = src.lastIndexOf("\n"), head = src.slice(0, nl + 1);
  let line = src.slice(nl + 1);
  if ((src.match(FENCE)?.length ?? 0) % 2) return head;
  const prose = src.replace(/^ {0,3}(`{3,}|~{3,})[^]*?^ {0,3}\1.*$/gm, "").replace(/`[^`\n]*`/g, "");
  if ((prose.match(/\$\$/g)?.length ?? 0) % 2) return src.slice(0, src.lastIndexOf("$$"));
  if (prose.lastIndexOf("\\[") > prose.lastIndexOf("\\]")) return src.slice(0, src.lastIndexOf("\\["));
  if (/^\s*\|/.test(line)) return head;
  if (/^\s*(?:[#>*+-]|\d+[.)])*\s*$/.test(line)) return head;
  if (!/\s$/.test(line)) line = line.replace(/[A-Za-z0-9À-ɏ'’_-]{1,24}$/, "");
  const text = head + line, p = text.lastIndexOf("\n\n") + 1;
  let para = text.slice(p)
    .replace(/!\[[^\]\n]*\]?(?:\([^)\n]*)?$/, "")
    .replace(/\[([^\]\n]*)\]\([^)\n]*$/, "$1").replace(/\[([^\]\n]*)$/, "$1")
    .replace(/<\/?[A-Za-z][^>\n]*$/, "")
    .replace(/(?:\*{1,3}|_{1,3}|~{1,2}|`+)$/, "");
  const dollars = [...para.replace(/`[^`]*`/g, (c) => " ".repeat(c.length)).matchAll(/(?<![\\$])\$(?!\$)/g)];
  const lastDollar = dollars.at(-1)?.index ?? 0;
  if (dollars.length % 2 && !/^\$\d/.test(para.slice(lastDollar))) para = para.slice(0, lastDollar);
  if (para.lastIndexOf("\\(") > para.lastIndexOf("\\)")) para = para.slice(0, para.lastIndexOf("\\("));
  if ((para.replace(/```[\s\S]*?```/g, "").match(/`/g)?.length ?? 0) % 2) para += "`";
  if ((para.match(/\*\*/g)?.length ?? 0) % 2) para = para.trimEnd() + "**";
  if ((para.replace(/^ {0,3}[*+-] +|`[^`]*`/gm, "").replace(/\*\*|\s\*\s/g, "").match(/\*/g)?.length ?? 0) % 2) para = para.trimEnd() + "*";
  if ((para.match(/~~/g)?.length ?? 0) % 2) para = para.trimEnd() + "~~";
  return text.slice(0, p) + para;
}

function htmlNodes(t: Token, links: Links): Node[] {
  const tpl = document.createElement("template");
  tpl.innerHTML = DOMPurify.sanitize(marked.parser(Object.assign([t], { links })));
  typeset(tpl.content);
  return [...tpl.content.childNodes];
}

const place = (into: Node, n: Node, at: Node | null) => {
  if (n.isConnected && into.isConnected && "moveBefore" in into) (into as Element).moveBefore(n, at);
  else into.insertBefore(n, at);
};

function renderBlocks(v: { body: HTMLElement; blocks?: Blocks }, src: string, live: boolean, art?: { c: Conversation; holder: object }) {
  if (!v.blocks) { v.body.replaceChildren(); v.blocks = { raws: [], nodes: [] }; }
  const text = art ? sceneText(src, art.c) : src;
  const b = v.blocks, tokens = chatTokens(live ? settle(text) : text), blocks = tokens.filter((t) => t.type !== "space");
  let seen = 0, all: Map<string, Version[]> | undefined;
  const nth = art ? blocks.map((t) => (t.type === "code" && artifactOf(t.lang, t.text) ? seen++ : -1)) : [];
  const artNodes = (k: number, old: Node | undefined): Node[] => {
    const ver = versionAt(all ??= artifactsOf(art!.c), art!.holder, nth[k]);
    if (!ver) return unknownEdit(blocks[k] as Tokens.Code, tokens.links);
    return [artCard(old, ver, all.get(ver.id)!.length)];
  };
  let k = 0;
  while (k < blocks.length && k < b.raws.length && b.raws[k] === blocks[k].raw) k++;
  for (; k < blocks.length; k++) {
    const t = blocks[k], old = b.nodes[k] ?? [];
    const tail = live && k === blocks.length - 1;
    const page = t.type === "code" && !tail && PAGE_LANGS.has((t.lang ?? "").toLowerCase()) && (PAGE_DOC.test(t.text) || previewed.has(t.text));
    const now = t.type !== "code" ? htmlNodes(t, tokens.links) : nth[k] >= 0 ? artNodes(k, old[0])
      : SVG_START.test(t.text) ? [svgFigure(old[0], t.text, tail)]
      : /^mermaid$/i.test(t.lang ?? "") ? [mermaidFigure(old[0], t.text, tail)] : page ? [pageFigure(old[0], t.text)] : htmlNodes(t, tokens.links);
    if (art && t.type === "paragraph") speakLine(now[0], art.c, k ? b.nodes[k - 1]?.[0] : undefined, (art.holder as Message).as);
    b.raws[k] = t.raw;
    if (now.length === 1 && old.length === 1 && now[0] === old[0]) continue;
    const at = b.nodes.slice(k).flat()[0] ?? null;
    for (const n of now) place(v.body, n, at);
    for (const n of old) n.parentNode?.removeChild(n);
    b.nodes[k] = now;
  }
  for (const n of b.nodes.slice(blocks.length).flat()) n.parentNode?.removeChild(n);
  b.nodes.length = b.raws.length = blocks.length;
  b.nodes.forEach((n, k) => {
    const f = n[0], t = blocks[k];
    if (!(f instanceof HTMLElement && f.classList.contains(nth[k] < 0 ? "drawing" : "live") && !(live && k === blocks.length - 1) && t.type === "code")) return;
    if (nth[k] < 0) return void (f.dataset.kind === "mermaid" ? mermaidFigure : svgFigure)(f, t.text, false);
    const now = artNodes(k, f);
    if (now[0] !== f) f.replaceWith(...now);
    b.nodes[k] = now;
  });
}

const SVG_START = /^\s*(?:<\?xml[^>]*\?>\s*)?(?:<!--[\s\S]*?-->\s*|<!DOCTYPE[^>]*>\s*)*<svg[\s>]/i;
const SVG_SHADOW = ":host { display: block; color: #1f1b1c; } svg { display: block; width: var(--fig-w, 640px); max-width: 100%; height: auto; } " +
  ":host(.zoom) svg { max-width: none; } :host(.fill) { height: 100%; } :host(.fill) svg { width: 100%; height: 100%; max-width: none; }";
const svgSource = new WeakMap<Element, string>();
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

function svgFile(text: string) {
  let src = text.trim().replace(/^[\s\S]*?(?=<svg[\s>])/i, "");
  const open = src.match(/^<svg[^>]*>/i)?.[0] ?? "<svg>";
  let root = open;
  if (!/\sxmlns\s*=/.test(root)) root = root.replace(/^<svg/i, '<svg xmlns="http://www.w3.org/2000/svg"');
  if (/\bxlink:/.test(src) && !/\sxmlns:xlink\s*=/.test(root)) root = root.replace(/^<svg/i, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
  return root + src.slice(open.length);
}

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
    if (j >= s.length) break;
    const m = s.slice(lt, j + 1).match(/^<(\/?)([A-Za-z][\w:.-]*)[^]*?(\/?)>$/);
    if (m && m[1]) {
      const at = stack.lastIndexOf(m[2]);
      if (at >= 0) stack.length = at;
    } else if (m && !m[3]) stack.push(m[2]);
    i = cut = j + 1;
    if (!stack.length) break;
  }
  return cut ? s.slice(0, cut) + stack.reverse().map((t) => `</${t}>`).join("") : null;
}

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

const pictureOf = (root: ShadowRoot | null | undefined) => [...(root?.children ?? [])].find((e): e is SVGSVGElement => e instanceof SVGSVGElement);
function drawSvg(root: ShadowRoot, markup: string | null, fade: boolean): boolean {
  if (!markup) return !!pictureOf(root);
  svgSanitizing = true;
  let frag: DocumentFragment;
  try {
    frag = DOMPurify.sanitize(markup, { USE_PROFILES: { svg: true, svgFilters: true }, FORBID_TAGS: ["foreignObject"], RETURN_DOM_FRAGMENT: true });
  } finally { svgSanitizing = false; }
  const svg = frag.querySelector("svg");
  if (!svg) return false;
  const num = (n: string) => /^\s*[\d.]+\s*(px)?\s*$/.test(svg.getAttribute(n) ?? "") ? parseFloat(svg.getAttribute(n)!) : 0;
  const w = num("width"), hh = num("height");
  if (w && hh && !svg.hasAttribute("viewBox")) svg.setAttribute("viewBox", `0 0 ${w} ${hh}`);
  const cur = pictureOf(root), added: Element[] = [];
  if (cur && sameEl(cur, svg)) syncSvg(cur, svg, added);
  else {
    cur ? cur.replaceWith(svg) : root.querySelector("style")!.after(svg);
    added.push(...svg.children);
  }
  if (fade && !reducedMotion.matches)
    for (const el of added) if (el instanceof SVGGraphicsElement) el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 260, easing: "ease-out" });
  return true;
}

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

function svgFigure(old: Node | undefined, text: string, drawing: boolean, fade = drawing): HTMLElement {
  const fig = old instanceof HTMLElement && old.dataset.kind === "svg" ? old : figureShell("svg");
  svgSource.set(fig, text);
  fig.classList.toggle("drawing", drawing);
  const t = performance.now();
  if (!drawing || t - Number(fig.dataset.drawn ?? 0) > 90) {
    fig.dataset.drawn = String(t);
    const ok = drawSvg(fig.querySelector(".svg-paper")!.shadowRoot!, svgPrefix(text), fade);
    fig.classList.toggle("failed", !drawing && !ok);
  }
  fitFigure(fig);
  const svg = pictureOf(fig.querySelector(".svg-paper")!.shadowRoot), box = svg?.viewBox.baseVal;
  fig.querySelector(".svg-state")!.textContent = drawing ? "Drawing" : fig.classList.contains("failed") ? "Couldn't draw this SVG"
    : box && box.width ? `SVG · ${Math.round(box.width)} × ${Math.round(box.height)}` : "SVG";
  if (fig.classList.contains("show-code") || fig.classList.contains("failed")) svgCode(fig);
  return fig;
}

function fitFigure(fig: HTMLElement) {
  const paper = fig.querySelector<HTMLElement>(".svg-paper"), vb = pictureOf(paper?.shadowRoot)?.viewBox.baseVal;
  if (!paper || !vb || !vb.width || !vb.height) return;
  const stage = fig.closest<HTMLElement>(".shelf-stage");
  if (stage && !fig.isConnected) return;
  const room = stage ? paper.clientWidth - 30 : thread.clientWidth - 48 - 30;
  const tall = stage ? Math.max(paper.clientHeight - 30, 120) : Math.max(scroller.clientHeight - 74, 320);
  const natural = fig.dataset.kind === "mermaid" ? vb.width : Infinity;
  const w = `${Math.floor(Math.min(room, tall * (vb.width / vb.height), natural))}px`;
  if (paper.style.getPropertyValue("--fig-w") !== w) paper.style.setProperty("--fig-w", w);
}
addEventListener("resize", () => thread.querySelectorAll<HTMLElement>(".svg-fig").forEach(fitFigure));
new ResizeObserver(() => $("shelf-stage").querySelectorAll<HTMLElement>(".svg-fig").forEach(fitFigure)).observe($("shelf-stage"));

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
    const svg = pictureOf(host.shadowRoot), vb = svg?.viewBox.baseVal;
    if (svg && vb && vb.width && vb.height) {
      const ar = vb.width / vb.height, w = Math.min(innerWidth * 0.94 - 40, (innerHeight * 0.92 - 40) * ar);
      svg.style.width = `${w}px`;
      svg.style.height = `${w / ar}px`;
    }
    dlg.querySelector(".svg-zoom-paper")!.replaceChildren(host);
    dlg.showModal();
  }
}

const pictures = new WeakMap<Element, string>();
const DIAGRAMS: Record<string, string> = { graph: "flowchart", flowchart: "flowchart", sequenceDiagram: "sequence diagram",
  classDiagram: "class diagram", stateDiagram: "state diagram", "stateDiagram-v2": "state diagram", erDiagram: "ER diagram",
  gantt: "Gantt chart", pie: "pie chart", mindmap: "mind map", timeline: "timeline", journey: "user journey", gitGraph: "git graph",
  quadrantChart: "quadrant chart", xychart: "XY chart", "xychart-beta": "XY chart", sankey: "Sankey diagram", "sankey-beta": "Sankey diagram" };
const MERMAID_URL = "/vendor/mermaid/mermaid.esm.min.mjs";
let mermaidLib: Promise<typeof import("mermaid").default> | undefined, mermaidSeq = 0;
const loadMermaid = () => mermaidLib ??= (import(MERMAID_URL) as Promise<typeof import("mermaid")>).then(({ default: m }) => {
  m.initialize({ startOnLoad: false, securityLevel: "strict", htmlLabels: false, flowchart: { htmlLabels: false }, theme: "base",
    fontFamily: "ui-sans-serif, system-ui, sans-serif", themeVariables: { primaryColor: "#fde1e8", primaryBorderColor: "#e24d85",
      primaryTextColor: "#1f1b1c", lineColor: "#6e6164", secondaryColor: "#f3e9eb", tertiaryColor: "#ffffff" } });
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
  if (await m.parse(text, { suppressErrors: true })) svg = await m.render(`chrysalis-mermaid-${seq}`, text).then((r) => r.svg, () => undefined);
  if (fig.dataset.seq !== seq) return;
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


const PAGE_LANGS = new Set(["html", "htm"]);
const previewed = new Set<string>();
const isPage = (kind: string, src: string) => PAGE_LANGS.has(kind) || kind === "aframe" || PAGE_DOC.test(src);
const pageRoom = () => Math.max(scroller.clientHeight - 74, 240);
const pageStart = () => Math.min(480, pageRoom());
const pageFill = (frame: HTMLElement) => Math.round(Math.min(pageRoom(), Math.max(240, frame.clientWidth * 0.62)));
addEventListener("resize", () => document.querySelectorAll<HTMLIFrameElement>("iframe.page-frame.fills:not([data-fixed])").forEach((f) => { f.style.height = `${pageFill(f)}px`; }));

interface PageFrame { ready: boolean; want?: { lead: string; text: string; live: boolean }; sent?: string; ran?: string; loaded?: boolean; mode?: string; pin?: number | null }

const LIBS: [RegExp, (m: RegExpMatchArray) => string][] = [
  [/^(?:https?:)?\/\/[^"'\s]*?\/three(?:@[\w.^~-]+)?\/build\/three\.module(?:\.min)?\.js(?:\?[^"'\s]*)?$|^https?:\/\/(?:esm\.sh|cdn\.skypack\.dev)\/three(?:@[\w.^~-]+)?$|^three$/, () => "three.module.js"],
  [/^(?:https?:)?\/\/[^"'\s]*?\/three(?:\.js)?(?:@[\w.^~-]+|\/r?[\d.]+)?\/(?:build\/)?three(?:\.min)?\.js$/, () => "three.min.js"],
  [/^(?:(?:https?:)?\/\/[^"'\s]*?\/three(?:@[\w.^~-]+)?\/examples\/jsm\/|three\/(?:addons|examples\/jsm)\/)([\w-]+(?:\/[\w-]+)*\.js)(?:\?[^"'\s]*)?$/, (m) => `addons/${m[1]}`],
  [/^(?:https?:)?\/\/[^"'\s]*?\/three(?:@[\w.^~-]+)?\/examples\/js\/([\w-]+(?:\/[\w-]+)*\.js)$/, (m) => `classic/${m[1]}`],
  [/^(?:https?:)?\/\/[^"'\s]*?\/aframe(?:@[\w.^~-]+)?\/dist\/aframe[\w.-]*\.js$|^(?:https?:)?\/\/aframe\.io\/releases\/[\w.]+\/aframe(?:\.min)?\.js$/, () => "aframe"],
];
const vendorOf = (spec: string) => {
  for (const [re, path] of LIBS) { const m = spec.match(re); if (m) return path(m); }
  return null;
};
function libsOf(src: string): Map<string, string> {
  const out = new Map<string, string>(), specs = new Set<string>();
  for (const m of src.matchAll(/<script\b[^>]*?\ssrc\s*=\s*["']([^"']+)["']/gi)) specs.add(m[1]);
  for (const m of src.matchAll(/\b(?:from|import)\s*\(?\s*["']([^"']+)["']/g)) specs.add(m[1]);
  for (const map of src.matchAll(/<script\b[^>]*type\s*=\s*["']importmap["'][^>]*>([\s\S]*?)<\/script>/gi))
    for (const m of map[1].matchAll(/"([^"]+)"\s*:\s*"([^"]+)"/g)) { specs.add(m[2]); if (!m[1].endsWith("/")) specs.add(m[1]); }
  for (const spec of specs) { const v = vendorOf(spec); if (v) out.set(spec, v); }
  if ([...out.values()].some((v) => v.startsWith("addons/"))) out.set("three", "three.module.js");
  return out;
}
const vendorCode = new Map<string, Promise<string | null>>();
const fetchVendor = (file: string) => {
  let got = vendorCode.get(file);
  if (!got) vendorCode.set(file, got = fetch(file === "aframe" ? "/vendor/aframe/aframe.min.js" : `/vendor/three/${file}`).then((r) => (r.ok ? r.text() : null), () => null));
  return got;
};
const vendorDone = new Map<string, string | null>();
const frames = new WeakMap<HTMLIFrameElement, PageFrame>();
const scriptsOf = (src: string) => elementSpans(src).filter((s) => s.tag.toLowerCase() === "script").map((s) => src.slice(s.start, s.end)).join("\0");

function pageFrame(fixed: boolean): HTMLIFrameElement {
  const frame = h("iframe", "page-frame");
  frame.setAttribute("sandbox", "allow-scripts allow-modals");
  frame.title = "Page from the reply";
  if (fixed) frame.dataset.fixed = "1";
  else frame.style.height = `${pageStart()}px`;
  frames.set(frame, { ready: false });
  frame.src = "/sandbox";
  return frame;
}
function pagePaper(fixed: boolean): HTMLElement {
  const paper = h("div", "page-paper");
  paper.append(pageFrame(fixed));
  return paper;
}
const frameOf = (paper: Element) => [...paper.querySelectorAll<HTMLIFrameElement>(":scope > iframe.page-frame")].at(-1);
function showPage(paper: HTMLElement, text: string, live: boolean) {
  const frame = frameOf(paper)!, lead = (PAGE_DOC.test(text) ? "" : "<!doctype html>") +
    (/<a-scene[\s>]/.test(text) && !/<script\b[^>]*aframe/i.test(text) ? '<script src="https://aframe.io/releases/1.8.0/aframe.min.js"></script>' : "");
  frames.get(frame)!.want = { lead, text, live };
  flushPage(frame);
}
function flushPage(frame: HTMLIFrameElement) {
  const st = frames.get(frame), paper = frame.parentElement as HTMLElement | null;
  if (!st?.ready || !paper || !frame.contentWindow) return;
  const msg: { mode?: string; pin?: number | null; page?: string; write?: boolean; libs?: Record<string, string> } = {}, mode = pageModeOf(paper), pin = pagePin(paper);
  if (mode !== st.mode) msg.mode = st.mode = mode;
  if (pin !== st.pin) msg.pin = st.pin = pin;
  const want = st.want, key = want && `${want.live}:${want.lead}${want.text}`;
  if (want && key !== st.sent) {
    const scripts = scriptsOf(want.lead + want.text), write = !want.live && scripts !== (st.ran ?? "");
    if (write && st.ran !== undefined) {
      const next = pageFrame(!!frame.dataset.fixed);
      next.classList.add("next");
      if (!frame.dataset.fixed) next.style.height = frame.style.height;
      frames.get(next)!.want = want;
      paper.append(next);
      setTimeout(() => swapIn(next), 3000);
    } else {
      const libs = write ? libsOf(want.lead + want.text) : new Map<string, string>(), missing = [...new Set(libs.values())].filter((f) => !vendorDone.has(f));
      if (missing.length) {
        void Promise.all(missing.map((f) => fetchVendor(f).then((code) => vendorDone.set(f, code)))).then(() => flushPage(frame));
        if (Object.keys(msg).length) frame.contentWindow.postMessage({ chrysalis: msg }, "*");
        return;
      }
      st.sent = key;
      msg.page = want.lead + markSource(want.text);
      if (write) {
        msg.write = true;
        st.ran = scripts;
        const code: Record<string, string> = {};
        for (const [spec, file] of libs) if (vendorDone.get(file)) code[spec] = vendorDone.get(file)!;
        if (Object.keys(code).length) msg.libs = code;
      }
    }
  }
  if (Object.keys(msg).length) frame.contentWindow.postMessage({ chrysalis: msg }, "*");
}
function swapIn(next: HTMLIFrameElement) {
  if (!next.classList.contains("next") || !next.isConnected) return;
  for (const f of next.parentElement!.querySelectorAll(":scope > iframe.page-frame")) if (f !== next) f.remove();
  next.classList.remove("next");
}

function pageFigure(old: Node | undefined, text: string, live = false, fixed = false): HTMLElement {
  let fig = old instanceof HTMLElement && old.dataset.kind === "page" ? old : undefined;
  if (!fig) {
    fig = h("figure", "svg-fig page-fig");
    fig.dataset.kind = "page";
    fig.innerHTML = `<figcaption class="svg-bar"><span class="svg-state"></span><span class="svg-acts">` +
      ["code:Code", "copy:Copy", "save:Download", "reload:Reload", "zoom:Expand"].map((a) => `<button type="button" data-page="${a.split(":")[0]}">${a.split(":")[1]}</button>`).join("") +
      `</span></figcaption><div class="code svg-code"><div class="code-head"><span>html</span><button type="button" class="copy">Copy</button></div>` +
      `<pre><code class="hljs"></code></pre></div>`;
    fig.prepend(pagePaper(fixed));
  }
  if (svgSource.get(fig) !== text) {
    svgSource.set(fig, text);
    if (fig.classList.contains("show-code")) fig.querySelector(".svg-code code")!.innerHTML = hljs.highlight(text, { language: "xml", ignoreIllegals: true }).value;
  }
  const title = text.match(/<title[^>]*>([^<]*)<\/title>/i)?.[1].trim();
  fig.querySelector(".svg-state")!.textContent = live ? "Writing the page" : title ? `Page · ${title}` : "Page";
  fig.classList.toggle("drawing", live);
  showPage(fig.querySelector<HTMLElement>(".page-paper")!, text, live);
  return fig;
}

addEventListener("message", (e) => {
  const frame = [...document.querySelectorAll<HTMLIFrameElement>("iframe.page-frame")].find((f) => f.contentWindow === e.source);
  const st = frame && frames.get(frame), d = e.data, paper = frame?.parentElement;
  if (!frame || !st || !paper || typeof d !== "object" || !d) return;
  if (d.chrysalisReady) {
    st.ready = true;
    return flushPage(frame);
  }
  if (typeof d.chrysalisHeight === "number") {
    if (d.loaded) st.loaded = true;
    if (!frame.dataset.fixed) {
      const fill = d.fills && !(d.chrysalisHeight > (d.view ?? 0) * 1.25);
      frame.classList.toggle("fills", fill);
      const want = fill ? pageFill(frame) : st.loaded ? d.chrysalisHeight : Math.max(pageFill(frame), d.chrysalisHeight);
      const px = `${Math.round(Math.min(Math.max(want, 60), pageRoom()))}px`;
      if (frame.style.height !== px) frame.style.height = px;
    }
    if (d.loaded) swapIn(frame);
    return;
  }
  if (frame !== frameOf(paper) || pageModeOf(paper) === "use") return;
  if ("chrysalisHover" in d) hovered = pagePoint(paper, d.chrysalisHover);
  else if ("chrysalisPick" in d) {
    pinned = pagePoint(paper, d.chrysalisPick);
    st.pin = pagePin(paper);
    frame.focus();
    glow();
    renderPointing();
  } else if (typeof d.chrysalisPaint === "number") paintPage(paper, d.chrysalisPaint);
  else if (typeof d.chrysalisText?.cx === "number" && typeof d.chrysalisText.html === "string") retypePage(paper, d.chrysalisText.cx, d.chrysalisText.html);
  else if (typeof d.chrysalisKey === "string") editPage(paper, d.chrysalisKey);
  else if (d.chrysalisNote) toast("That text is made by the page's own code: ask for it to be changed");
});

function pageAction(btn: HTMLElement) {
  const act = btn.dataset.page;
  if (act === "preview") {
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
  else if (act === "reload") {
    const paper = fig.querySelector<HTMLElement>(".page-paper")!;
    paper.querySelectorAll(":scope > iframe").forEach((f) => f.remove());
    paper.append(pageFrame(false));
    showPage(paper, text, false);
  } else if (act === "zoom") {
    const dlg = $<HTMLDialogElement>("page-zoom"), paper = pagePaper(true);
    dlg.querySelector(".page-zoom-box")!.replaceChildren(paper);
    showPage(paper, text, false);
    dlg.showModal();
  }
}

interface ArtRead { text: string; blocks: { n: number; info: Info; text: string; closed: boolean }[] }
const artRead = new WeakMap<object, ArtRead>();
function artBlocksOf(holder: object, text: string): ArtRead["blocks"] {
  let r = artRead.get(holder);
  if (!r || r.text !== text) {
    let n = 0;
    const blocks = marked.lexer(text).flatMap((t) => {
      const a = t.type === "code" ? artifactOf(t.lang, t.text) : null;
      return a ? [{ n: n++, info: a.info, text: a.text, closed: /\n {0,3}(?:`{3,}|~{3,})[^\n]*\n?$/.test(t.raw) }] : [];
    });
    artRead.set(holder, r = { text, blocks });
  }
  return r.blocks;
}
function artifactsOf(c: Conversation): Map<string, Version[]> {
  const blocks: Block[] = [], writing = runs.get(c.id)?.m;
  let label = "";
  for (const m of c.messages) {
    if (m.role === "user") {
      label = m.content;
      for (const [n, e] of (m.edits ?? []).entries())
        blocks.push({ at: m, n, info: { kind: e.kind, id: e.id }, text: e.src, live: false, label: "Your edit", by: "user" });
      continue;
    }
    const parts: [object, string][] = [...(m.rounds ?? []).map((r): [object, string] => [r, r.content]), [m, m === writing ? settle(m.content) : m.content]];
    for (const [holder, text] of parts)
      for (const b of artBlocksOf(holder, text))
        blocks.push({ at: holder, n: b.n, info: b.info, text: b.text, live: m === writing && holder === m && !b.closed, label });
  }
  Object.entries(c.pending ?? {}).forEach(([id, p], n) =>
    blocks.push({ at: c, n, info: { kind: p.kind, id }, text: p.srcs.at(-1)!, live: false, label: "Your edit, not sent yet", by: "user" }));
  return replay(blocks);
}
const versionAt = (all: Map<string, Version[]>, holder: object, n: number) =>
  [...all.values()].flat().find((v) => v.block.at === holder && v.block.n === n);

const captionOf = (ver: Version) => {
  const done = (ver.applied ?? []).filter((a) => a.ok).length;
  const changes = ver.block.by === "user" ? ` · your edit${ver.block.at === conv ? ", not sent yet" : ""}` : !ver.applied ? ""
    : ver.block.live ? ` · changing${done ? `, ${done} so far` : ""}` : ` · ${done} change${done === 1 ? "" : "s"}`;
  return `${ver.id} · v${ver.v}${ver.from ? ` · from ${ver.from.id} v${ver.from.v}` : ""}${changes}`;
};
const beforeOf = (ver: Version): Version | undefined => {
  const all = artifactsOf(conv);
  return ver.from ? all.get(ver.from.id)?.[ver.from.v - 1] : ver.v > 1 ? all.get(ver.id)?.[ver.v - 2] : undefined;
};
const failedOf = (ver: Version) => (ver.applied ?? []).filter((a) => !a.ok && !a.change.partial);
const actButton = (act: string, name: string, label?: string) => {
  const b = h("button", "", name);
  b.type = "button";
  b.dataset.art = act;
  if (label) b.title = b.ariaLabel = label;
  return b;
};

function unknownEdit(t: Tokens.Code, links: Links): Node[] {
  const info = infoOf(t.lang)!, nodes = htmlNodes(t, links);
  nodes.unshift(h("p", "art-fail", `No artifact “${"edit" in info ? info.from : ""}” comes before this edit, so it changes nothing.`));
  return nodes;
}

function drawVersion(prev: HTMLElement | undefined, ver: Version, drawing: boolean, count: number, fade = drawing): HTMLElement {
  const src = ver.src;
  let el: HTMLElement;
  if (ver.kind === "svg" || (["xml", "html"].includes(ver.kind) && SVG_START.test(src))) el = svgFigure(prev?.dataset.kind === "svg" ? prev : undefined, src, drawing, fade);
  else if (ver.kind === "mermaid") el = mermaidFigure(prev?.dataset.kind === "mermaid" ? prev : undefined, src, drawing);
  else if (isPage(ver.kind, src)) el = pageFigure(prev?.dataset.kind === "page" ? prev : undefined, src, drawing, true);
  else if (DOC_KINDS.has(ver.kind)) el = docFigure(prev?.dataset.kind === "doc" ? prev : undefined, ver.kind, src, drawing);
  else el = htmlNodes({ type: "code", raw: "", lang: ver.kind, text: src } as Tokens.Code, {} as Links)[0] as HTMLElement;
  el.classList.add("art-canvas");
  el.dataset.art = ver.id;
  el.dataset.v = String(ver.v);
  const head = el.querySelector(".svg-bar") ?? el.querySelector(".code-head");
  if (head && !head.querySelector(".art-label")) {
    const step = h("span", "art-step");
    step.append(actButton("prev", "‹", "Earlier version"), h("span", "art-label"), actButton("next", "›", "Later version"));
    head.prepend(step);
    head.querySelector(".svg-acts, .code-acts")?.prepend(actButton("use", "Use this version", "Make this version the latest, as your own edit"),
      actButton("changes", "Changes"), ...(el.dataset.kind === "mermaid" ? []
        : [actButton("edit", "Edit", el.classList.contains("code") ? "Change the code here, by hand" : "Change it by hand: select, retype, fill, move")]));
  }
  if (head) {
    head.querySelector(".art-label")!.textContent = captionOf(ver);
    head.querySelector<HTMLButtonElement>("[data-art=prev]")!.disabled = ver.v <= 1;
    head.querySelector<HTMLButtonElement>("[data-art=next]")!.disabled = ver.v >= count;
    head.querySelector(".art-step")!.classList.toggle("single", count < 2);
    head.querySelector<HTMLElement>("[data-art=changes]")!.hidden = !beforeOf(ver);
    head.querySelector<HTMLElement>("[data-art=use]")!.hidden = ver.v >= count;
  }
  const paper = el.querySelector<HTMLElement>(el.dataset.kind === "svg" ? ".svg-paper" : el.dataset.kind === "page" ? ".page-paper" : el.dataset.kind === "doc" ? ".doc-paper" : ":scope > :not(*)");
  if (paper && !paper.dataset.pointable) {
    paper.dataset.pointable = "1";
    (el.dataset.kind === "page" ? pagePointable : el.dataset.kind === "doc" ? docPointable : pointable)(paper);
  }
  paper?.classList.toggle("busy", drawing);
  if (paper) glow();
  let fail = el.querySelector<HTMLElement>(":scope > .art-fail");
  if (failedOf(ver).length && !fail) el.append(fail = h("p", "art-fail"));
  if (fail) fail.textContent = failedOf(ver).length ? `Not applied: ${failedOf(ver).map((a) => a.how).join("; ")}` : "";
  return el;
}

const KINDS_SHOWN: Record<string, [keyof typeof ICONS, string]> = { svg: ["image", "SVG"], mermaid: ["diagram", "Diagram"], page: ["page", "Page"],
  doc: ["doc", "Document"], code: ["code", "Code"] };
const canvasKind = (ver: Version) => ver.kind === "svg" || (["xml", "html"].includes(ver.kind) && SVG_START.test(ver.src)) ? "svg"
  : ver.kind === "mermaid" ? "mermaid" : isPage(ver.kind, ver.src) ? "page" : DOC_KINDS.has(ver.kind) ? "doc" : "code";
function cardText(ver: Version, count: number) {
  const done = (ver.applied ?? []).filter((a) => a.ok).length, [, kind] = KINDS_SHOWN[canvasKind(ver)];
  const what = ver.block.live ? (ver.applied ? `Changing${done ? `, ${done} so far` : ""}` : "Writing") : ver.block.by === "user"
    ? `your edit${ver.block.at === conv ? ", not sent yet" : ""}` : ver.applied ? `${done} change${done === 1 ? "" : "s"}` : "";
  return [kind, `v${ver.v}`, ver.from ? `from ${ver.from.id} v${ver.from.v}` : "", what, ver.v < count && !ver.block.live ? `now v${count}` : ""]
    .filter(Boolean).join(" · ");
}
function artCard(old: Node | undefined, ver: Version, count: number): HTMLElement {
  let el = old instanceof HTMLElement && old.classList.contains("art-card") ? old : undefined;
  if (!el) {
    el = h("div", "art-card");
    el.tabIndex = 0;
    el.setAttribute("role", "button");
    const text = h("span", "art-text");
    text.append(h("b", "art-name"), h("span", "art-sub"));
    el.append(h("span", "art-icon"), text, h("span", "art-go", "Open"));
  }
  paintCard(el, ver, count);
  return el;
}
function paintCard(el: HTMLElement, ver: Version, count: number) {
  const kind = canvasKind(ver);
  if (el.dataset.shape !== kind) {
    el.dataset.shape = kind;
    el.querySelector(".art-icon")!.innerHTML = icon(KINDS_SHOWN[kind][0]);
  }
  el.dataset.art = ver.id;
  el.dataset.v = String(ver.v);
  el.querySelector(".art-name")!.textContent = ver.id;
  el.querySelector(".art-sub")!.textContent = cardText(ver, count);
  el.classList.toggle("live", ver.block.live);
  el.title = `Open ${ver.id} v${ver.v}`;
  const failed = failedOf(ver);
  let fail = el.querySelector<HTMLElement>(":scope > .art-fail");
  if (failed.length && !fail) el.append(fail = h("span", "art-fail"));
  if (fail) fail.textContent = failed.length ? `Not applied: ${failed.map((a) => a.how).join("; ")}` : "";
}

function artAction(btn: HTMLElement) {
  const el = btn.closest<HTMLElement>("[data-v]")!, all = artifactsOf(conv), list = all.get(el.dataset.art!) ?? [];
  const ver = list[Number(el.dataset.v) - 1];
  if (!ver) return;
  const act = btn.dataset.art, which = el.dataset.which === "other" ? "other" : "main";
  if (act === "use") {
    shelf.main = { id: ver.id };
    shelf.other = undefined;
    return editByHand(ver.id, list.at(-1)!, ver.src);
  }
  if (act === "prev" || act === "next") {
    const to = list[ver.v - 1 + (act === "prev" ? -1 : 1)];
    if (!to) return;
    shelf[which] = { id: ver.id, v: which === "main" && to.v === list.length ? undefined : to.v };
    renderShelf();
    $("shelf-stage").querySelector<HTMLElement>(`:scope > [data-which=${which}] [data-art=${act}]`)?.focus();
    return;
  }
  if (act === "edit") {
    const paper = el.querySelector<HTMLElement>(":scope > [data-edit]");
    return el.classList.contains("code") ? editCode(el, btn, ver) : paper ? setEditing(paper, paper.dataset.editing !== "1") : undefined;
  }
  let box = el.querySelector<HTMLElement>(":scope > .art-changes");
  if (!box) {
    const was = beforeOf(ver);
    if (!was) return;
    el.append(box = diffView(was, ver));
    box.hidden = true;
  }
  box.hidden = !box.hidden;
  btn.textContent = box.hidden ? "Changes" : "Hide changes";
}

interface Side { id: string; v?: number }
const shelf: { open: boolean; main?: Side; other?: Side; compare: boolean; shut?: string; known?: Set<string>; of?: string } = { open: false, compare: false };
const pickVersion = (all: Map<string, Version[]>, s: Side) => {
  const list = all.get(s.id)!;
  return (s.v && list.find((x) => x.v === s.v)) || list.at(-1)!;
};
const liveKey = (ver: Version) => `${conv.id}/${ver.id}/${ver.v}`;
function openShelf(id?: string, v?: number) {
  shelf.open = true;
  shelf.main = id ? { id, v: v === artifactsOf(conv).get(id)?.length ? undefined : v } : undefined;
  shelf.other = undefined;
  renderShelf();
  if (!panes.artifacts.beside.matches) $("shelf").focus({ preventScroll: true });
}
function closeShelf() {
  const live = [...artifactsOf(conv).values()].flat().find((x) => x.block.live);
  shelf.open = false;
  shelf.shut = live && liveKey(live);
  renderShelf();
}
function besideOf(ver: Version): Side | undefined {
  return ver.from ? { id: ver.from.id, v: ver.from.v } : ver.v > 1 ? { id: ver.id, v: ver.v - 1 } : undefined;
}
let shelfAt = 0, shelfLater: number | undefined;
function shelfSoon() {
  if (shelfLater) return;
  const wait = Math.max(0, 60 - (performance.now() - shelfAt));
  shelfLater = window.setTimeout(() => { shelfLater = undefined; renderShelf(); }, wait);
}
function renderShelf() {
  shelfAt = performance.now();
  const all = artifactsOf(conv), ids = [...all.keys()], btn = $<HTMLButtonElement>("artifacts-toggle"), el = $("shelf"), p = panes.artifacts;
  const live = [...all.values()].flat().find((x) => x.block.live), made = [...all.values()].flat().filter((x) => x.block.by !== "user");
  if (shelf.of !== conv.id) {
    shelf.of = conv.id;
    shelf.known = new Set(made.filter((x) => !x.block.live).map(liveKey));
    shelf.main = shelf.other = undefined;
  }
  const fresh = made.filter((x) => !shelf.known!.has(liveKey(x))).at(-1);
  for (const x of made) if (!x.block.live) shelf.known!.add(liveKey(x));
  if (fresh && p.beside.matches && shelf.shut !== liveKey(fresh) && (!shelf.open || !shelf.main?.v)) {
    shelf.open = true;
    shelf.main = { id: fresh.id };
    shelf.other = undefined;
  }
  const want = shelf.open && ids.length > 0;
  btn.hidden = !ids.length;
  btn.innerHTML = icon("shapes") + `<span class="art-count">${ids.length}</span>`;
  btn.title = btn.ariaLabel = `Artifacts in this chat: ${ids.length}`;
  btn.classList.toggle("on", want);
  btn.setAttribute("aria-expanded", String(want));
  if (isOpen(p) !== want || (want && el.hasAttribute("data-over") === p.beside.matches)) setPane(p, want, { focus: false });
  const stage = $("shelf-stage");
  if (!want) {
    stage.replaceChildren();
    return markCards(all);
  }
  if (!shelf.main || !all.has(shelf.main.id)) shelf.main = { id: live?.id ?? ids.at(-1)! };
  else if (live && !shelf.main.v && shelf.main.id !== live.id) shelf.main = { id: live.id };
  const main = pickVersion(all, shelf.main);
  if (shelf.other && !all.has(shelf.other.id)) shelf.other = undefined;
  const other = shelf.compare ? (shelf.other ? pickVersion(all, shelf.other) : (() => { const b = besideOf(main); return b ? pickVersion(all, b) : undefined; })()) : undefined;
  const tabs = $("shelf-tabs");
  if (tabs.dataset.key !== JSON.stringify([ids, main.id])) {
    tabs.dataset.key = JSON.stringify([ids, main.id]);
    tabs.replaceChildren(...ids.map((id) => {
      const list = all.get(id)!, b = h("button", id === main.id ? "shelf-tab on" : "shelf-tab");
      b.type = "button";
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", String(id === main.id));
      b.innerHTML = icon(KINDS_SHOWN[canvasKind(list.at(-1)!)][0]);
      b.append(id);
      b.onclick = () => { shelf.main = { id }; shelf.other = undefined; renderShelf(); };
      return b;
    }));
  }
  $("shelf-compare").setAttribute("aria-pressed", String(shelf.compare));
  $("shelf-compare").classList.toggle("on", shelf.compare);
  const show = (which: "main" | "other", ver: Version | undefined) => {
    const was = stage.querySelector<HTMLElement>(`:scope > [data-which=${which}]`) ?? undefined;
    if (!ver) return void was?.remove();
    const count = all.get(ver.id)!.length, drawing = ver.block.live && ver.v === count;
    const key = JSON.stringify([ver.id, ver.v, count, ver.src.length, ver.src.slice(-64), drawing, cardText(ver, count)]);
    if (was?.dataset.key === key) return;
    const now = drawVersion(was?.dataset.art === ver.id ? was : undefined, ver, drawing, count);
    now.dataset.which = which;
    now.dataset.key = key;
    if (now !== was) {
      if (was) was.replaceWith(now);
      else if (which === "other") stage.prepend(now);
      else stage.append(now);
    }
    if (now.dataset.kind === "svg" || now.dataset.kind === "mermaid") fitFigure(now);
  };
  show("other", other);
  show("main", main);
  stage.classList.toggle("two", !!other);
  if (shelf.compare && !other) stage.dataset.note = "Only one version so far: nothing to compare it with";
  else delete stage.dataset.note;
  markCards(all, main);
}
function markCards(all: Map<string, Version[]>, main?: Version) {
  for (const card of thread.querySelectorAll<HTMLElement>(".art-card")) {
    const list = all.get(card.dataset.art!), ver = list?.[Number(card.dataset.v) - 1];
    if (ver) paintCard(card, ver, list!.length);
    card.classList.toggle("on", !!main && card.dataset.art === main.id && card.dataset.v === String(main.v));
  }
}

interface Point { art: string; sel: string; partId?: string; markup?: string; tag?: string; paper?: HTMLElement; cx?: number; at?: number; label?: string }
let hovered: Point | undefined, pinned: Point | undefined;
let paintColor = pref("paint-color", "#e24d85");
type EditKind = "svg" | "page" | "doc";
const editMode = (paper: HTMLElement) => paper.dataset.editing === "1" && !paper.classList.contains("busy");
const toolOf = (paper: HTMLElement) => (editMode(paper) ? paper.dataset.tool ?? "select" : "view");
const samePoint = (a?: Point, b?: Point) => !!a && !!b && a.art === b.art && a.sel === b.sel;
function pointOn(paper: HTMLElement, e: Event, leaf = toolOf(paper) === "fill"): Point | undefined {
  const host = paper.closest<HTMLElement>("[data-art]"), svg = pictureOf(paper.shadowRoot), t = e.composedPath()[0];
  if (!host || !svg || !(t instanceof Element) || t === svg || !svg.contains(t)) return undefined;
  const near = t.closest("[id]"), part = !leaf && near && near !== svg && svg.contains(near) ? near : t;
  let sel = part.id ? `[id="${CSS.escape(part.id)}"]` : "";
  if (!part.id) for (let x: Element | null = part; x && x !== svg; x = x.parentElement)
    sel = `> :nth-child(${[...x.parentElement!.children].indexOf(x) + 1}) ${sel}`;
  return { art: host.dataset.art!, sel: part.id ? sel : `:scope ${sel}`.trim(), partId: part.id || undefined, markup: part.outerHTML, tag: part.tagName, paper };
}
const partIn = (paper: HTMLElement, sel: string) => pictureOf(paper.shadowRoot)?.querySelector<SVGGraphicsElement>(sel) ?? null;
const ruleFor = (sel: string) => (sel.startsWith(":scope") ? sel.replace(":scope", ":host > svg") : `:host > svg ${sel}`);

const EDITOR_CSS = `:host { position: relative; outline: none; }
:host(.busy) .sel { display: none; }
.sel { position: absolute; z-index: 1; box-sizing: border-box; border: 1.5px dashed #e24d85; border-radius: 2px; cursor: move; touch-action: none; }
.sel[hidden] { display: none; }
.sel i { position: absolute; width: 10px; height: 10px; box-sizing: border-box; border: 1.5px solid #e24d85; border-radius: 2px; background: #fff; }
.sel i[data-h=nw] { left: -6px; top: -6px; cursor: nwse-resize; } .sel i[data-h=ne] { right: -6px; top: -6px; cursor: nesw-resize; }
.sel i[data-h=se] { right: -6px; bottom: -6px; cursor: nwse-resize; } .sel i[data-h=sw] { left: -6px; bottom: -6px; cursor: nesw-resize; }`;
const TOOL_ICONS = {
  select: '<path d="M6 3.5l12 7.2-5.3 1.5-2.6 5.3z"/>',
  bucket: '<path d="M5 11.5 11.5 5l7 7-6.5 6.5a2 2 0 0 1-2.8 0L5 14.3a2 2 0 0 1 0-2.8z"/><path d="M4.5 12.5h13.5"/><path d="M20 15.5s1.5 2 1.5 3a1.5 1.5 0 0 1-3 0c0-1 1.5-3 1.5-3z"/>',
  undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
  forward: '<path d="M12 19V6M6.5 11.5 12 6l5.5 5.5"/>',
  backward: '<path d="M12 5v13M6.5 12.5 12 18l5.5-5.5"/>',
  duplicate: ICONS.copy,
  delete: ICONS.trash,
  text: '<path d="M5 7V5h14v2"/><path d="M12 5v14"/><path d="M9 19h6"/>',
};
const toolIcon = (k: keyof typeof TOOL_ICONS) => `<svg viewBox="0 0 24 24" aria-hidden="true">${TOOL_ICONS[k]}</svg>`;

const TOOLS: Record<EditKind, [string, string, keyof typeof TOOL_ICONS][]> = {
  svg: [["select", "Select", "select"], ["fill", "Fill", "bucket"]],
  page: [["select", "Select", "select"], ["text", "Text", "text"], ["fill", "Fill", "bucket"]],
  doc: [["select", "Select", "select"], ["text", "Text", "text"]],
};
const HINTS: Record<EditKind, Record<string, string>> = {
  svg: { select: "Click a part to select it, then drag it to move it, a corner to resize it", fill: "Click a shape to fill it with the color" },
  page: { select: "Click a part of the page to select it", text: "Click text and type: Enter keeps it, Esc puts it back", fill: "Click a part to fill its background with the color" },
  doc: { select: "Click a block to select it", text: "Click a block to change its source: Ctrl+Enter keeps it, Esc puts it back" },
};
const MOVES: [string, string, keyof typeof TOOL_ICONS][] = [["up", "Up", "forward"], ["down", "Down", "backward"], ["duplicate", "Copy", "duplicate"], ["delete", "Delete", "delete"]];
const PICKED: Record<EditKind, [string, string, keyof typeof TOOL_ICONS][]> = {
  svg: [["forward", "Forward", "forward"], ["backward", "Back", "backward"], ["duplicate", "Copy", "duplicate"], ["delete", "Delete", "delete"]],
  page: MOVES,
  doc: MOVES,
};
const toolbars = new WeakMap<HTMLElement, HTMLElement>();
function addToolbar(paper: HTMLElement, kind: EditKind) {
  paper.dataset.edit = kind;
  paper.dataset.tool ??= "select";
  const bar = h("div", "art-tools"), button = ([k, name, i]: [string, string, keyof typeof TOOL_ICONS], attr: string) =>
    `<button type="button" data-${attr}="${k}" title="${name}">${toolIcon(i)}<span>${name}</span></button>`;
  bar.setAttribute("role", "toolbar");
  bar.ariaLabel = "Change it by hand";
  bar.innerHTML = `<span class="seg">${TOOLS[kind].map((t) => button(t, "tool")).join("")}</span>` +
    (kind !== "doc" ? `<label class="swatch" title="The color Fill paints with"><input type="color"></label>` : "") +
    `<span class="hint"></span><span class="picked" hidden><b></b>${PICKED[kind].map((t) => button(t, "do")).join("")}</span>` +
    `<span class="end">${button(["undo", "Undo", "undo"], "do")}<button type="button" class="done" data-do="done">Done</button></span>`;
  bar.addEventListener("click", (e) => {
    const b = (e.target as Element).closest<HTMLButtonElement>("button");
    if (!b) return;
    if (b.dataset.tool) {
      paper.dataset.tool = b.dataset.tool;
      hovered = undefined;
      glow();
    } else if (b.dataset.do === "done") setEditing(paper, false);
    else (kind === "svg" ? editSelected : kind === "page" ? editPage : editDoc)(paper, b.dataset.do!);
  });
  bar.querySelector("input")?.addEventListener("input", (e) => {
    paintColor = (e.target as HTMLInputElement).value;
    setPref("paint-color", paintColor);
    paper.dataset.tool = "fill";
    glow();
  });
  paper.parentElement?.querySelector(":scope > .art-tools")?.remove();
  paper.before(bar);
  toolbars.set(paper, bar);
}
function setEditing(paper: HTMLElement, on: boolean) {
  for (const other of document.querySelectorAll<HTMLElement>("[data-editing]")) {
    delete other.dataset.editing;
    other.classList.remove("editing");
  }
  if (on) paper.dataset.editing = "1";
  paper.classList.toggle("editing", on);
  hovered = undefined;
  glow();
  if (on && paper.tabIndex >= 0) paper.focus({ preventScroll: true });
}
function syncToolbars() {
  for (const paper of document.querySelectorAll<HTMLElement>("[data-edit]")) {
    const bar = toolbars.get(paper), on = editMode(paper), kind = paper.dataset.edit as EditKind, tool = toolOf(paper);
    const art = paper.closest<HTMLElement>("[data-art]")?.dataset.art;
    paper.closest(".art-canvas")?.querySelector("[data-art=edit]")?.setAttribute("aria-pressed", String(on));
    if (!bar) continue;
    bar.classList.toggle("on", on);
    if (!on) continue;
    bar.querySelectorAll<HTMLElement>("[data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === tool));
    const swatch = bar.querySelector<HTMLElement>(".swatch");
    if (swatch) { swatch.style.background = paintColor; swatch.querySelector("input")!.value = paintColor; }
    bar.querySelector<HTMLButtonElement>("[data-do=undo]")!.disabled = !art || !conv.pending?.[art];
    const picked = tool === "select" && pinned?.paper === paper ? pinned : undefined;
    bar.querySelector<HTMLElement>(".hint")!.hidden = !!picked;
    bar.querySelector<HTMLElement>(".hint")!.textContent = HINTS[kind][tool] ?? "";
    bar.querySelector<HTMLElement>(".picked")!.hidden = !picked;
    bar.querySelector(".picked b")!.textContent = picked ? pointName(picked).split(" · ").slice(1).join(" · ") || "Selected" : "";
  }
  for (const fig of $("shelf-stage").querySelectorAll<HTMLElement>(":scope > .svg-fig")) {
    fitFigure(fig);
    const paper = fig.querySelector<HTMLElement>(".svg-paper");
    if (paper) placeSelection(paper);
  }
}

function pointable(paper: HTMLElement) {
  const root = paper.shadowRoot!;
  paper.tabIndex = 0;
  const style = document.createElement("style"), box = h("div", "sel");
  style.textContent = EDITOR_CSS;
  box.innerHTML = `<i data-h="nw"></i><i data-h="ne"></i><i data-h="se"></i><i data-h="sw"></i>`;
  box.hidden = true;
  root.append(style, box);
  addToolbar(paper, "svg");
  const inTools = (e: Event) => e.composedPath().includes(box);
  paper.addEventListener("mousedown", (e) => {
    if (inTools(e)) return;
    e.preventDefault();
    paper.focus({ preventScroll: true });
  });
  paper.addEventListener("pointermove", (e) => {
    const p = inTools(e) ? undefined : pointOn(paper, e);
    if (!samePoint(p, hovered) || (p && hovered?.paper !== paper)) { hovered = p; glow(); }
  });
  paper.addEventListener("pointerleave", () => { if (hovered) { hovered = undefined; glow(); } });
  paper.addEventListener("click", (e) => {
    if (inTools(e)) return;
    const p = pointOn(paper, e);
    if (toolOf(paper) === "fill") return void (p && paintShape(paper, p));
    pinned = p && !samePoint(p, pinned) ? p : undefined;
    glow();
    renderPointing();
  });
  box.addEventListener("pointerdown", (e) => dragSelected(paper, e));
  paper.addEventListener("keydown", (e) => {
    const mod = e.ctrlKey || e.metaKey, k = e.key, on = editMode(paper);
    const nudge = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[k];
    if (k === "Escape" && (pinned || on)) escape(paper);
    else if (!on) return;
    else if (mod && k.toLowerCase() === "z") editSelected(paper, "undo");
    else if (mod && k.toLowerCase() === "d") editSelected(paper, "duplicate");
    else if (k === "Delete" || k === "Backspace") editSelected(paper, "delete");
    else if (nudge && pinned?.paper === paper) moveSelected(paper, nudge[0] * (e.shiftKey ? 10 : 1), nudge[1] * (e.shiftKey ? 10 : 1));
    else return;
    e.preventDefault();
    e.stopPropagation();
  });
}
function escape(paper: HTMLElement) {
  if (pinned) {
    pinned = undefined;
    glow();
    renderPointing();
  } else setEditing(paper, false);
}

function shownVersion(paper: HTMLElement): Version | undefined {
  const host = paper.closest<HTMLElement>("[data-art][data-v]");
  return host ? artifactsOf(conv).get(host.dataset.art!)?.[Number(host.dataset.v) - 1] : undefined;
}
function spanFor(paper: HTMLElement, el: Element, src: string): Span | undefined {
  const svg = pictureOf(paper.shadowRoot)!, drawn = [svg, ...svg.querySelectorAll("*")], k = drawn.indexOf(el);
  const spans = elementSpans(src), from = spans.findIndex((x) => x.tag.toLowerCase() === "svg"), tag = el.tagName.toLowerCase();
  const at = k >= 0 && from >= 0 ? spans[from + k] : undefined;
  if (at && at.tag.toLowerCase() === tag) return at;
  const open = (x: Span) => src.slice(x.start, x.openEnd);
  if (el.id) return spans.find((x) => new RegExp(`(?<![\\w:-])id\\s*=\\s*["']${el.id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}["']`).test(open(x)));
  const geo = ["cx", "cy", "r", "rx", "ry", "x", "y", "width", "height", "d", "points", "x1", "y1", "x2", "y2"].filter((a) => el.hasAttribute(a));
  const hits = geo.length ? spans.filter((x) => x.tag.toLowerCase() === tag && geo.every((a) => open(x).match(new RegExp(`\\s${a}\\s*=\\s*"([^"]*)"`))?.[1] === el.getAttribute(a))) : [];
  return hits.length === 1 ? hits[0] : undefined;
}
function editByHand(id: string, base: Version, src: string) {
  if (src === base.src) return;
  ((conv.pending ??= {})[id] ??= { kind: base.kind, srcs: [] }).srcs.push(src);
  save(conv);
  repaintArtifact(id);
}
function undoHand(id: string) {
  const p = conv.pending?.[id];
  if (!p) return;
  p.srcs.pop();
  if (!p.srcs.length) delete conv.pending![id];
  if (conv.pending && !Object.keys(conv.pending).length) delete conv.pending;
  save(conv);
  repaintArtifact(id);
}
function repaintArtifact(id: string) {
  if (shelf.main?.id === id) shelf.main = { id };
  renderShelf();
  glow();
}
const fmtTransform = (m: DOMMatrix): string | null => {
  const r = (x: number) => +x.toFixed(3);
  if (Math.abs(m.b) > 1e-6 || Math.abs(m.c) > 1e-6 || Math.abs(m.a - m.d) > 1e-6) return `matrix(${[m.a, m.b, m.c, m.d, m.e, m.f].map(r).join(" ")})`;
  return [r(m.e) || r(m.f) ? `translate(${r(m.e)} ${r(m.f)})` : "", r(m.a) !== 1 ? `scale(${r(m.a)})` : ""].filter(Boolean).join(" ") || null;
};
const matrixOf = (el: SVGGraphicsElement) => [...el.transform.baseVal].reduce((m, t) => m.multiply(DOMMatrix.fromMatrix(t.matrix)), new DOMMatrix());
function setTransform(paper: HTMLElement, el: Element, t: string | null) {
  const ver = shownVersion(paper), span = ver && spanFor(paper, el, ver.src);
  if (!ver || !span) return void toast("That part could not be found in the artifact's source");
  editByHand(ver.id, ver, setOpen(ver.src, span, withAttr(ver.src.slice(span.start, span.openEnd), "transform", t)));
}
function moveSelected(paper: HTMLElement, dx: number, dy: number) {
  const el = pinned?.paper === paper ? partIn(paper, pinned.sel) : null;
  if (el) setTransform(paper, el, fmtTransform(new DOMMatrix().translate(dx, dy).multiply(matrixOf(el))));
}
function dragSelected(paper: HTMLElement, e: PointerEvent) {
  const el = pinned?.paper === paper ? partIn(paper, pinned.sel) : null, box = e.currentTarget as HTMLElement;
  if (!el || e.button !== 0) return;
  e.preventDefault();
  e.stopPropagation();
  box.setPointerCapture(e.pointerId);
  const corner = (e.target as HTMLElement).dataset.h, r = el.getBoundingClientRect(), cur = matrixOf(el), was = el.getAttribute("transform");
  const toUser = (x: number, y: number) => new DOMPoint(x, y).matrixTransform((el.parentNode as SVGGraphicsElement).getScreenCTM()!.inverse());
  const pivot = corner ? { x: corner.includes("w") ? r.right : r.left, y: corner.includes("n") ? r.bottom : r.top } : null;
  const reach = pivot ? Math.hypot(e.clientX - pivot.x, e.clientY - pivot.y) || 1 : 1, from = toUser(e.clientX, e.clientY);
  let pre = new DOMMatrix();
  const move = (ev: PointerEvent) => {
    if (pivot) {
      const k = Math.max(0.05, Math.hypot(ev.clientX - pivot.x, ev.clientY - pivot.y) / reach), p = toUser(pivot.x, pivot.y);
      pre = new DOMMatrix().translate(p.x, p.y).scale(k).translate(-p.x, -p.y);
    } else {
      const to = toUser(ev.clientX, ev.clientY);
      pre = new DOMMatrix().translate(to.x - from.x, to.y - from.y);
    }
    const t = fmtTransform(pre.multiply(cur));
    if (t) el.setAttribute("transform", t); else el.removeAttribute("transform");
    placeSelection(paper);
  };
  const up = () => {
    box.removeEventListener("pointermove", move);
    if (pre.isIdentity) return;
    if (was === null) el.removeAttribute("transform"); else el.setAttribute("transform", was);
    setTransform(paper, el, fmtTransform(pre.multiply(cur)));
  };
  box.addEventListener("pointermove", move);
  box.addEventListener("pointerup", up, { once: true });
  box.addEventListener("pointercancel", up, { once: true });
}
function paintShape(paper: HTMLElement, p: Point) {
  const el = partIn(paper, p.sel), ver = shownVersion(paper), span = el && ver && spanFor(paper, el, ver.src);
  if (!ver || !span) return void toast("That shape could not be found in the artifact's source");
  editByHand(ver.id, ver, setOpen(ver.src, span, withPaint(ver.src.slice(span.start, span.openEnd), "fill", paintColor)));
}
function editSelected(paper: HTMLElement, act: string) {
  const ver = shownVersion(paper);
  if (!ver) return;
  if (act === "undo") return undoHand(ver.id);
  const el = pinned?.paper === paper ? partIn(paper, pinned.sel) : null, span = el && spanFor(paper, el, ver.src);
  if (!span) return;
  const spans = elementSpans(ver.src);
  if (act === "delete") {
    pinned = undefined;
    renderPointing();
    editByHand(ver.id, ver, withoutElement(ver.src, span));
  } else if (act === "duplicate") editByHand(ver.id, ver, withCopy(ver.src, span, 12));
  else {
    if (!pinned!.partId) pinned = undefined;
    renderPointing();
    editByHand(ver.id, ver, withOrder(ver.src, spans[spans.indexOf(span)], spans, act === "forward"));
  }
}
const PAGE_MODES: Record<string, string> = { select: "select", text: "text", fill: "bucket" };
const pageModeOf = (paper: HTMLElement) => PAGE_MODES[toolOf(paper)] ?? "use";
const pagePin = (paper: HTMLElement) => (pinned?.paper === paper && pinned.cx !== undefined ? pinned.cx : null);
function pagePoint(paper: HTMLElement, cx: unknown): Point | undefined {
  const host = paper.closest<HTMLElement>("[data-art]"), ver = shownVersion(paper);
  const span = typeof cx === "number" && ver ? elementSpans(ver.src)[cx] : undefined;
  if (!host || !ver || !span) return undefined;
  return { art: host.dataset.art!, sel: `cx:${cx}`, cx: cx as number, partId: idOf(ver.src.slice(span.start, span.openEnd)), markup: ver.src.slice(span.start, span.end), tag: span.tag, paper };
}
const pagePointable = (paper: HTMLElement) => addToolbar(paper, "page");
function editPage(paper: HTMLElement, act: string) {
  const ver = shownVersion(paper);
  if (!ver) return;
  if (act === "undo") return undoHand(ver.id);
  const was = pinned?.paper === paper ? pinned : undefined, spans = elementSpans(ver.src), span = was?.cx === undefined ? undefined : spans[was.cx];
  if (act === "unpin" && !was) return setEditing(paper, false);
  if (act === "unpin" || !was || !span) {
    pinned = undefined;
    glow();
    return renderPointing();
  }
  const src = act === "delete" ? withoutElement(ver.src, span) : act === "duplicate" ? withCopy(ver.src, span)
    : act === "up" || act === "down" ? withOrder(ver.src, span, spans, act === "down") : ver.src;
  const at = was.partId ? elementSpans(src).findIndex((x) => idOf(src.slice(x.start, x.openEnd)) === was.partId)
    : act === "duplicate" ? was.cx! : -1;
  pinned = undefined;
  editByHand(ver.id, ver, src);
  pinned = act !== "delete" && at >= 0 ? pagePoint(paper, at) : undefined;
  glow();
  renderPointing();
}
function paintPage(paper: HTMLElement, cx: number) {
  const ver = shownVersion(paper), spans = ver ? elementSpans(ver.src) : [], span = spans[cx];
  if (!ver || !span) return;
  let inPicture = false;
  for (let p = span.parent; p >= 0 && !inPicture; p = spans[p].parent) inPicture = spans[p].tag.toLowerCase() === "svg";
  const open = ver.src.slice(span.start, span.openEnd), mat = open.match(/\smaterial\s*=\s*"([^"]*)"/)?.[1];
  const entity = /^a-/i.test(span.tag) && (mat === undefined ? withAttr(open, "color", paintColor)
    : withAttr(open, "material", /(^|;)\s*color\s*:/.test(mat) ? mat.replace(/((?:^|;)\s*color\s*:)[^;]*/, `$1 ${paintColor}`) : `${mat.trim() ? `${mat.replace(/;?\s*$/, "")}; ` : ""}color: ${paintColor}`));
  editByHand(ver.id, ver, setOpen(ver.src, span, entity || (inPicture ? withPaint(open, "fill", paintColor) : withBackground(open, paintColor))));
}
function retypePage(paper: HTMLElement, cx: number, html: string) {
  const ver = shownVersion(paper), span = ver && elementSpans(ver.src)[cx];
  if (!ver || !span || span.end <= span.openEnd) return void toast("That text could not be found in the page's source");
  editByHand(ver.id, ver, withContent(ver.src, span, html));
}

function diffView(was: Version, ver: Version): HTMLElement {
  const lines = lineDiff(was.src, ver.src), box = h("div", "code art-changes");
  const add = lines.filter((d) => d.op === "+").length, del = lines.filter((d) => d.op === "-").length;
  const near = lines.map((d, k) => lines.slice(Math.max(0, k - 3), k + 4).some((x) => x.op !== " "));
  let html = "", gap = 0;
  lines.forEach((d, k) => {
    if (!near[k]) return void gap++;
    if (gap) html += `<span class="diff-gap">⋯ ${gap} line${gap > 1 ? "s" : ""} the same</span>`;
    gap = 0;
    html += `<span class="diff-${d.op === "+" ? "add" : d.op === "-" ? "del" : "same"}">${d.op === " " ? " " : d.op} ${escapeHtml(d.text) || " "}</span>`;
  });
  if (gap) html += `<span class="diff-gap">⋯ ${gap} line${gap > 1 ? "s" : ""} the same</span>`;
  box.innerHTML = `<div class="code-head"><span>${escapeHtml(`${was.id} v${was.v} → ${ver.id} v${ver.v}`)}</span><span class="diff-count">` +
    `<span class="diff-plus">+${add}</span> <span class="diff-minus">−${del}</span></span></div><pre class="diff"><code>${html || '<span class="diff-gap">No change</span>'}</code></pre>`;
  return box;
}

function editCode(el: HTMLElement, btn: HTMLElement, ver: Version) {
  const code = el.querySelector<HTMLElement>("pre code")!;
  if (code.isContentEditable) return code.blur();
  const before = code.innerHTML;
  code.textContent = ver.src;
  code.contentEditable = "plaintext-only";
  code.spellcheck = false;
  btn.textContent = "Done";
  el.classList.add("editing");
  code.focus();
  let dropped = false;
  const keys = (e: KeyboardEvent) => {
    if (e.key === "Escape") { dropped = true; code.blur(); }
    else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) code.blur();
    else return;
    e.preventDefault();
  };
  code.addEventListener("keydown", keys);
  code.addEventListener("blur", () => {
    code.removeEventListener("keydown", keys);
    code.contentEditable = "false";
    el.classList.remove("editing");
    btn.textContent = "Edit";
    const text = (code.textContent ?? "").replace(ver.src.endsWith("\n") ? /(?!)/ : /\n$/, "");
    code.innerHTML = before;
    if (!dropped && text !== ver.src) editByHand(ver.id, artifactsOf(conv).get(ver.id)!.at(-1)!, text);
  }, { once: true });
}

const DOC_KINDS = new Set(["latex", "tex", "markdown", "md"]);
const isTex = (kind: string) => kind === "latex" || kind === "tex";
const MD_CSS = `:host { display: block; }
.md { max-width: 46em; padding: 18px 26px; color: var(--text); font: 15px/1.6 var(--sans); }
.md > [data-at] > :first-child { margin-top: 0; } .md > [data-at] { margin: 0 0 0.85em; }
.md h1, .md h2, .md h3, .md h4 { margin: 1.1em 0 0.4em; line-height: 1.3; } .md h1 { font-size: 1.6em; } .md h2 { font-size: 1.3em; } .md h3 { font-size: 1.1em; }
.md p { margin: 0 0 0.7em; } .md ul, .md ol { margin: 0 0 0.7em; padding-left: 1.6em; }
.md blockquote { margin: 0; padding-left: 1em; border-left: 3px solid var(--line); color: var(--muted); }
.md table { border-collapse: collapse; } .md th, .md td { padding: 4px 10px; border: 1px solid var(--line); }
.md code { font: 0.9em var(--mono); } .md a { color: var(--accent); }
.md .code { margin: 0; overflow: hidden; border: 1px solid var(--line); border-radius: 10px; background: var(--code); }
.md .code-head { display: flex; justify-content: space-between; padding: 3px 8px 3px 12px; border-bottom: 1px solid var(--line); color: var(--faint); font: 12px var(--mono); }
.md .code-head button { display: none; } .md pre { margin: 0; padding: 10px 14px; overflow-x: auto; }
.md img { max-width: 100%; }`;
const DOC_EDIT_CSS = `.doc-body { position: relative; }
.doc-body [data-at] { border-radius: 4px; }
textarea.retype { display: block; box-sizing: border-box; width: 100%; margin: 0.4em 0; padding: 8px 10px; border: 1.5px solid #e24d85; border-radius: 6px;
  background: rgb(255 255 255 / 0.92); color: #1f1b1c; font: 13px/1.5 ui-monospace, "JetBrains Mono", monospace; resize: vertical; outline: none; }
:host(.busy) .tools { display: none; }`;
let docSheet: CSSStyleSheet | undefined;
function docRules(): CSSStyleSheet {
  if (docSheet) return docSheet;
  const rules: string[] = [];
  for (const sheet of document.styleSheets) {
    let list: CSSRuleList;
    try { list = sheet.cssRules; } catch { continue; }
    for (const r of list) if (r instanceof CSSStyleRule && /\.(katex|hljs)/.test(r.selectorText)) rules.push(r.cssText);
  }
  docSheet = new CSSStyleSheet();
  docSheet.replaceSync(rules.join("\n"));
  return docSheet;
}
function docHtml(kind: string, src: string, live: boolean): string {
  if (isTex(kind)) return latexToHtml(live ? texSoFar(src) : src);
  const tokens = marked.lexer(src);
  let at = 0, html = "";
  const tpl = document.createElement("template");
  for (const t of tokens) {
    const start = src.startsWith(t.raw, at) ? at : Math.max(at, src.indexOf(t.raw, at));
    at = start + t.raw.length;
    if (t.type === "space" || t.type === "def") continue;
    tpl.innerHTML = "";
    tpl.content.append(...htmlNodes(t, tokens.links));
    html += `<div data-at="${start}" data-end="${start + t.raw.trimEnd().length}">${tpl.innerHTML}</div>`;
  }
  return `<article class="md">${html}</article>`;
}
function texSoFar(src: string): string {
  const re = /\\(begin|end)\s*\{(equation|align|gather|multline|eqnarray|flalign|alignat|tabular|array)\*?\}|\\\[|\\\]/g;
  let open = -1;
  for (const m of src.matchAll(re)) {
    if (m[1] === "begin" || m[0] === "\\[") { if (open < 0) open = m.index!; }
    else open = -1;
  }
  return open < 0 ? src : src.slice(0, open);
}
function syncChildren(live: Element, next: Element) {
  const have = [...live.childNodes], want = [...next.childNodes];
  want.forEach((w, k) => {
    const h = have[k];
    if (h && h.isEqualNode(w)) return;
    if (h && h.nodeType === 1 && w.nodeType === 1 && (h as Element).tagName === (w as Element).tagName && !(h as Element).hasAttribute("data-at") && (h as Element).childElementCount)
      return syncChildren(h as Element, w as Element);
    if (h) h.replaceWith(w); else live.append(w);
  });
  for (const h of have.slice(want.length)) h.remove();
}
function docPaper(kind: string): HTMLElement {
  const paper = h("div", "doc-paper");
  paper.dataset.doc = isTex(kind) ? "tex" : "md";
  const root = paper.attachShadow({ mode: "open" });
  root.adoptedStyleSheets = [docRules()];
  root.innerHTML = `<style>${isTex(kind) ? LATEX_CSS : MD_CSS}\n${DOC_EDIT_CSS}</style><style class="point"></style><div class="doc-body"></div>`;
  return paper;
}
function showDoc(paper: HTMLElement, kind: string, src: string, live: boolean) {
  const body = paper.shadowRoot!.querySelector(".doc-body")!, key = `${live}:${src}`;
  if (paper.dataset.key === key || body.querySelector("textarea.retype")) return;
  paper.dataset.key = key;
  const tpl = document.createElement("template");
  tpl.innerHTML = docHtml(kind, src, live);
  const next = tpl.content.firstElementChild!;
  const cur = body.firstElementChild;
  if (cur && cur.tagName === next.tagName && cur.className === next.className) syncChildren(cur, next);
  else body.replaceChildren(next);
}
function docFigure(old: Node | undefined, kind: string, text: string, live: boolean): HTMLElement {
  let fig = old instanceof HTMLElement && old.dataset.kind === "doc" && old.querySelector<HTMLElement>(".doc-paper")?.dataset.doc === (isTex(kind) ? "tex" : "md") ? old : undefined;
  if (!fig) {
    fig = h("figure", "svg-fig doc-fig");
    fig.dataset.kind = "doc";
    fig.innerHTML = `<figcaption class="svg-bar"><span class="svg-state"></span><span class="svg-acts">` +
      ["code:Source", "copy:Copy", "save:Download"].map((a) => `<button type="button" data-doc="${a.split(":")[0]}">${a.split(":")[1]}</button>`).join("") +
      `</span></figcaption><div class="code svg-code"><div class="code-head"><span>${isTex(kind) ? "latex" : "markdown"}</span><button type="button" class="copy">Copy</button></div>` +
      `<pre><code class="hljs"></code></pre></div>`;
    fig.prepend(docPaper(kind));
  }
  svgSource.set(fig, text);
  fig.dataset.lang = kind;
  fig.classList.toggle("drawing", live);
  const title = isTex(kind) ? text.match(/\\title\s*\{([^}]*)\}/)?.[1] : text.match(/^#{1,3}\s+(.+)$/m)?.[1];
  fig.querySelector(".svg-state")!.textContent = live ? "Writing" : `${isTex(kind) ? "LaTeX" : "Markdown"}${title ? ` · ${title.replace(/\\[a-z]+\s*|[{}*_`]/gi, "").trim()}` : ""}`;
  if (fig.classList.contains("show-code")) docCode(fig);
  showDoc(fig.querySelector<HTMLElement>(".doc-paper")!, kind, text, live);
  return fig;
}
function docCode(fig: HTMLElement) {
  const code = fig.querySelector(".svg-code code")!, text = svgSource.get(fig) ?? "", lang = isTex(fig.dataset.lang ?? "") ? "latex" : "markdown";
  if (code.textContent !== text) code.innerHTML = hljs.getLanguage(lang) ? hljs.highlight(text, { language: lang, ignoreIllegals: true }).value : escapeHtml(text);
}
function docAction(btn: HTMLElement) {
  const fig = btn.closest<HTMLElement>(".doc-fig")!, text = svgSource.get(fig) ?? "", act = btn.dataset.doc;
  if (act === "code") {
    btn.textContent = fig.classList.toggle("show-code") ? "Document" : "Source";
    docCode(fig);
  } else if (act === "copy") copy(text, btn);
  else if (act === "save") download(`${fig.dataset.art ?? "document"}.${isTex(fig.dataset.lang ?? "") ? "tex" : "md"}`, text, "text/plain");
}

function docPoint(paper: HTMLElement, e: Event): Point | undefined {
  const host = paper.closest<HTMLElement>("[data-art]"), t = e.composedPath()[0], ver = shownVersion(paper);
  const el = t instanceof Element ? t.closest<HTMLElement>("[data-at]") : null;
  if (!host || !ver || !el || !paper.shadowRoot!.contains(el)) return undefined;
  const at = Number(el.dataset.at), end = Number(el.dataset.end), words = (el.textContent ?? "").replace(/\s+/g, " ").trim();
  return { art: host.dataset.art!, sel: `at:${at}`, at, markup: ver.src.slice(at, end), tag: el.tagName.toLowerCase(),
    label: words.length > 28 ? `${words.slice(0, 26).trimEnd()}…` : words, paper };
}
const docBlock = (paper: HTMLElement, at: number) => paper.shadowRoot!.querySelector<HTMLElement>(`[data-at="${at}"]`);
function docSpans(paper: HTMLElement): { spans: Span[]; els: HTMLElement[] } {
  const els = [...paper.shadowRoot!.querySelectorAll<HTMLElement>("[data-at]")];
  const spans = els.map((el): Span => {
    const up = el.parentElement?.closest<HTMLElement>("[data-at]");
    return { tag: el.tagName, start: Number(el.dataset.at), openEnd: Number(el.dataset.at), end: Number(el.dataset.end), parent: up ? els.indexOf(up) : -1 };
  });
  return { spans, els };
}
function docPointable(paper: HTMLElement) {
  const root = paper.shadowRoot!;
  paper.tabIndex = 0;
  addToolbar(paper, "doc");
  const inTools = (_e: Event) => false;
  const retyping = (e: Event) => e.composedPath().some((x) => x instanceof HTMLTextAreaElement);
  paper.addEventListener("pointerdown", (e) => { if (!inTools(e) && !retyping(e)) paper.focus({ preventScroll: true }); });
  paper.addEventListener("pointermove", (e) => {
    if (inTools(e) || retyping(e)) return;
    const p = docPoint(paper, e);
    if (!samePoint(p, hovered) || (p && hovered?.paper !== paper)) { hovered = p; glow(); }
  });
  paper.addEventListener("pointerleave", () => { if (hovered) { hovered = undefined; glow(); } });
  paper.addEventListener("click", (e) => {
    if (inTools(e) || retyping(e) || (e.composedPath()[0] as Element).closest?.("a")) return;
    const p = docPoint(paper, e);
    if (toolOf(paper) === "text") return void (p && retypeDoc(paper, p));
    pinned = p && !samePoint(p, pinned) ? p : undefined;
    glow();
    renderPointing();
  });
  paper.addEventListener("dblclick", (e) => {
    const p = editMode(paper) && !retyping(e) ? docPoint(paper, e) : undefined;
    if (p) { e.preventDefault(); retypeDoc(paper, p); }
  });
  paper.addEventListener("keydown", (e) => {
    if (retyping(e)) return;
    const mod = e.ctrlKey || e.metaKey, k = e.key, on = editMode(paper);
    if (k === "Escape" && (pinned || on)) escape(paper);
    else if (!on) return;
    else if (mod && k.toLowerCase() === "z") editDoc(paper, "undo");
    else if (mod && k.toLowerCase() === "d") editDoc(paper, "duplicate");
    else if ((k === "Delete" || k === "Backspace") && pinned?.paper === paper) editDoc(paper, "delete");
    else if (k === "Enter" && pinned?.paper === paper) retypeDoc(paper, pinned);
    else return;
    e.preventDefault();
    e.stopPropagation();
  });
}
function retypeDoc(paper: HTMLElement, p: Point) {
  const ver = shownVersion(paper), el = p.at === undefined ? null : docBlock(paper, p.at);
  if (!ver || !el || paper.classList.contains("busy")) return;
  const at = p.at!, end = Number(el.dataset.end), area = h("textarea", "retype");
  area.value = ver.src.slice(at, end);
  area.spellcheck = false;
  el.replaceWith(area);
  const fit = () => { area.style.height = "auto"; area.style.height = `${area.scrollHeight + 4}px`; };
  fit();
  area.addEventListener("input", fit);
  area.focus({ preventScroll: true });
  let keep = true;
  area.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { keep = false; area.blur(); }
    else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) area.blur();
    else return;
    e.preventDefault();
    e.stopPropagation();
  });
  area.addEventListener("blur", () => {
    const text = area.value;
    area.replaceWith(el);
    if (keep && text !== ver.src.slice(at, end)) {
      pinned = undefined;
      renderPointing();
      editByHand(ver.id, ver, ver.src.slice(0, at) + text + ver.src.slice(end));
    }
    paper.focus({ preventScroll: true });
  }, { once: true });
}
function editDoc(paper: HTMLElement, act: string) {
  const ver = shownVersion(paper);
  if (!ver) return;
  if (act === "undo") return undoHand(ver.id);
  const was = pinned?.paper === paper ? pinned : undefined;
  if (!was || was.at === undefined) return;
  const { spans } = docSpans(paper), span = spans.find((x) => x.start === was.at);
  if (!span) return;
  const sib = spans.filter((x) => x.parent === span.parent), next = sib[sib.indexOf(span) + 1];
  const gap = next ? ver.src.slice(span.end, next.start) : "", sep = /^\s+$/.test(gap) ? gap : "\n\n";
  const src = act === "delete" ? withoutElement(ver.src, span) : act === "duplicate" ? ver.src.slice(0, span.end) + sep + ver.src.slice(span.start, span.end) + ver.src.slice(span.end)
    : withOrder(ver.src, span, spans, act === "down");
  pinned = undefined;
  renderPointing();
  editByHand(ver.id, ver, src);
}

function placeSelection(paper: HTMLElement) {
  const box = paper.shadowRoot?.querySelector<HTMLElement>(".sel");
  if (!box) return;
  const el = pinned?.paper === paper && toolOf(paper) === "select" ? partIn(paper, pinned.sel) : null;
  box.hidden = !el;
  if (!el) return;
  const at = paper.getBoundingClientRect(), r = el.getBoundingClientRect();
  Object.assign(box.style, { left: `${r.left - at.left - 3}px`, top: `${r.top - at.top - 3}px`, width: `${r.width + 6}px`, height: `${r.height + 6}px` });
}
function glow() {
  if (pinned && !pinned.paper?.isConnected) pinned.paper = $("shelf-stage").querySelector<HTMLElement>(`.art-canvas[data-art="${CSS.escape(pinned.art)}"] :is(.svg-paper, .page-paper, .doc-paper)`) ?? undefined;
  for (const paper of document.querySelectorAll<HTMLElement>(".svg-paper")) {
    const root = paper.shadowRoot, art = paper.closest<HTMLElement>("[data-art]")?.dataset.art;
    if (!root || !art || paper.closest<HTMLElement>("[data-kind]")?.dataset.kind === "mermaid" || paper.dataset.kind === "mermaid" || paper.querySelector(".doc-paper")) continue;
    let style = root.querySelector<HTMLStyleElement>("style.point");
    if (!style) root.append(style = Object.assign(document.createElement("style"), { className: "point" }));
    const css = [`:host > svg { cursor: ${toolOf(paper) === "fill" ? "cell" : "pointer"}; }`];
    if (hovered?.art === art && hovered.paper === paper) css.push(`${ruleFor(hovered.sel)} { filter: drop-shadow(0 0 1.5px #e24d85) drop-shadow(0 0 5px rgb(226 77 133 / 0.75)); }`);
    if (pinned?.art === art) css.push(`${ruleFor(pinned.sel)} { filter: drop-shadow(0 0 2px #e24d85) drop-shadow(0 0 9px #e24d85); }`);
    if (style.textContent !== css.join("\n")) style.textContent = css.join("\n");
    placeSelection(paper);
  }
  for (const paper of document.querySelectorAll<HTMLElement>(".doc-paper")) {
    const root = paper.shadowRoot!, art = paper.closest<HTMLElement>("[data-art]")?.dataset.art;
    if (!art || !paper.dataset.edit) continue;
    const css = [`.doc-body [data-at] { cursor: ${toolOf(paper) === "text" ? "text" : "pointer"}; }`];
    if (hovered?.paper === paper && hovered.at !== undefined) css.push(`.doc-body [data-at="${hovered.at}"] { outline: 1.5px dashed #e24d85; outline-offset: 3px; }`);
    if (pinned?.paper === paper && pinned.at !== undefined) css.push(`.doc-body [data-at="${pinned.at}"] { outline: 2px solid #e24d85; outline-offset: 3px; background: rgb(226 77 133 / 0.07); }`);
    const point = root.querySelector<HTMLStyleElement>("style.point")!;
    if (point.textContent !== css.join("\n")) point.textContent = css.join("\n");
  }
  for (const paper of document.querySelectorAll<HTMLElement>(".page-paper[data-edit]")) {
    const frame = frameOf(paper);
    if (frame) flushPage(frame);
  }
  syncToolbars();
}
const pointName = (p: Point) => `${p.art}${p.partId ? ` · #${p.partId}` : p.label ? ` · “${p.label}”` : p.tag ? ` · <${p.tag}>` : ""}`;
function renderPointing() {
  const el = $("pointing");
  el.hidden = !pinned;
  if (!pinned) return el.replaceChildren();
  const x = h("button", "act");
  x.type = "button";
  x.innerHTML = icon("x");
  x.title = x.ariaLabel = "Stop pointing at it";
  x.onclick = () => { pinned = undefined; glow(); renderPointing(); };
  el.replaceChildren(h("span", "", `Pointing at ${pointName(pinned)}`), x);
}
function takeEdits(): Pick<Message, "edits"> {
  const p = conv.pending;
  if (!p || !Object.keys(p).length) return {};
  delete conv.pending;
  return { edits: Object.entries(p).map(([id, e]) => ({ id, kind: e.kind, src: e.srcs.at(-1)! })) };
}
function pointFields(said: boolean): Pick<Message, "point" | "pointAt"> {
  const p = pinned ?? (said ? hovered : undefined);
  if (!p) return {};
  pinned = undefined;
  glow();
  renderPointing();
  const latest = artifactsOf(conv).get(p.art)?.at(-1), part = (p.partId && latest && elementById(latest.src, p.partId)) || p.markup;
  return { pointAt: pointName(p), point: `The user is pointing at ${p.partId ? `the part #${p.partId} of ` : "this part of "}the artifact ${p.art}` +
    (part ? `:\n${part.length > 3000 ? part.slice(0, 3000) + "…" : part}` : ".") };
}

const fmtK = (n: number) => (n < 1000 ? String(n) : `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}K`);
const fmtS = (s: number) => `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function pref(key: string, fallback: string) {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}
function setPref(key: string, value: string) {
  try { localStorage.setItem(key, value); } catch {  }
}
function load<T>(key: string, fallback: T): T {
  try { return JSON.parse(sessionStorage.getItem(key) ?? "null") ?? fallback; } catch { return fallback; }
}
function store(key: string, value: unknown) {
  try { sessionStorage.setItem(key, JSON.stringify(value)); } catch {  }
}

const chats: Record<string, Conversation> = load("chats", {});
let settings = loadSettings();
let conv = blank();
const runs = new Map<string, Run>();
const running = (c: Conversation = conv) => runs.get(c.id);
let status: Status = { state: "offline" };
let build = "";
let follow = true;
const views = new Map<Message, View>();
const tokens = new Map<string, { reasoning: Tok[]; content: Tok[] }>();
const tokenView = new Set<string>();
let editing: Message | null = null;

function randomId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}

function blank(): Conversation {
  let tools: ToolChoice["tools"] | undefined;
  try { tools = JSON.parse(pref("tools", "null")) ?? undefined; } catch {  }
  return { id: randomId(), title: "New chat", updated: Date.now(), effort: pref("effort", "xhigh") as Effort, messages: [], ...(tools && Object.keys(tools).length ? { tools } : {}) };
}

function render() {
  views.clear();
  renderCastNext();
  const was = [...thread.children];
  thread.append(...(conv.messages.length ? conv.messages.map(messageView) : [emptyState()]));
  for (const el of was) el.remove();
  $("title").textContent = conv.title;
  $("export").hidden = !conv.messages.length;
  document.title = conv.messages.length ? `${conv.title} · Chrysalis Inference` : "Chrysalis Inference";
  renderComposer();
  renderList();
  paintGhost();
  renderShelf();
}

function messageView(m: Message, i: number): HTMLElement {
  if (m === editing) return editorView(m, i);
  if (m.role === "assistant") {
    const v = assistantView(i), by = speakerHead(conv);
    if (by) v.el.prepend(by);
    views.set(m, v);
    paint(v, m, [...runs.values()].find((r) => r.m === m)?.live);
    return v.el;
  }
  const el = h("article", m.fix ? "msg user fix" : "msg user"), foot = h("footer", "msg-foot");
  el.dataset.i = String(i);
  el.append(head(ME));
  foot.innerHTML = pager(m) + (m.voice ? `<span class="said" title="Said in voice mode">${icon("wave")}</span>` : "")
    + slots([["speak", !!m.content.trim() && !voice, reading === m ? `<button type="button" class="act on" data-act="speak" title="Stop reading">${icon("stop")}</button>` : undefined],
      ["copy", true], ["edit", !running()], ["trash", !running()]]);
  if (m.note) {
    const note = h("details", "msg-note"), sum = h("summary", "", m.voice ? "Voice mode note" : "Note to the model");
    sum.title = "What the app told the model with this message (Settings, Voice mode)";
    note.append(sum, h("div", "msg-note-text", m.note));
    el.append(note);
  }
  for (const e of m.edits ?? []) {
    const chip = h("div", "point-chip edit-chip", `you changed ${e.id}`);
    chip.title = "Changed by hand before this message: the model was sent it as it was then";
    el.append(chip);
  }
  if (m.pointAt) {
    const chip = h("div", "point-chip", m.pointAt);
    chip.title = m.point ?? "";
    el.append(chip);
  }
  if (m.files?.length) el.append(filesView(m.files));
  if (m.content || !m.files?.length) el.append(heardBubble(m));
  if (unsettled.has(m)) el.classList.add("unsettled");
  el.append(foot);
  return el;
}
function heardBubble(m: Message): HTMLElement {
  const b = h("div", "bubble");
  if (!m.heard?.length) b.innerHTML = userHtml(m.content);
  else for (const p of fixedParts(m.content, m.heard)) b.append(typeof p === "string" ? p : fixedSpan(p));
  return b;
}
function userHtml(s: string): string {
  const re = /```[\s\S]*?(?:```|$)|`[^`\n]*`|\*\*(?=\S)(.+?)(?<=\S)\*\*|\*(?=[^\s*])(.+?)(?<=[^\s*])\*|["“«「『＂]/g;
  let out = "", at = 0;
  for (let m; (m = re.exec(s)); at = re.lastIndex) {
    out += escapeHtml(s.slice(at, m.index));
    if (m[1] !== undefined) out += `<strong>${userHtml(m[1])}</strong>`;
    else if (m[2] !== undefined) out += `<em>${userHtml(m[2])}</em>`;
    else if (m[0][0] === "`") out += escapeHtml(m[0]);
    else {
      const close = QUOTES[m[0]], nl = s.indexOf("\n", m.index), eol = nl < 0 ? s.length : nl, end = s.indexOf(close, m.index + 1);
      const shut = end >= 0 && end < eol;
      re.lastIndex = shut ? end + 1 : eol;
      out += `<q>${escapeHtml(m[0])}${userHtml(s.slice(m.index + 1, shut ? end : eol))}${shut ? escapeHtml(close) : ""}</q>`;
    }
  }
  return out + escapeHtml(s.slice(at));
}
function fixedSpan(p: Fix): HTMLElement {
  const f = h("span", "fixed", p.said);
  f.title = `Speech recognition heard "${p.heard}"`;
  return f;
}

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
  const again = m.role === "user";
  const button = (label: string, cls: string, fn: () => void, title = "") => {
    const b = h("button", cls, label);
    b.type = "button";
    b.title = title;
    b.onclick = fn;
    acts.append(b);
  };
  const done = (keep: boolean, send = false) => {
    editing = null;
    if (send) {
      if (!text.value.trim() && !m.files?.length) return render();
      branch(conv, i, { role: "user", content: text.value.trim(), ...(m.files?.length ? { files: m.files } : {}) });
      void generate();
      return save(conv);
    }
    if (keep && (text.value !== m.content || (think && think.value !== m.reasoning))) {
      m.content = text.value;
      if (think) m.reasoning = think.value;
      m.edited = true;
      if (m.id) { tokens.delete(m.id); tokenView.delete(m.id); }
    }
    render();
    save(conv);
  };
  acts.append(h("span", "editor-hint", again ? "Ctrl+Enter sends · Esc cancels" : "Ctrl+Enter saves · Esc cancels"));
  button("Cancel", "btn", () => done(false));
  button("Save", again ? "btn" : "btn primary", () => done(true), "Change this message in place, keeping the rest of the chat");
  if (again) button("Send", "btn primary", () => done(true, true), "Send it as a new version of this message, with a new reply (the old version stays a click away)");
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

function versionsAt(c: Conversation, i: number): Message[][] {
  const vs = c.messages[i].versions, all = [...(vs?.others ?? [])];
  all.splice(vs?.shown ?? 0, 0, c.messages.slice(i));
  return all;
}
function showVersion(c: Conversation, i: number, all: Message[][], k: number) {
  c.messages.splice(i);
  for (const v of all) delete v[0].versions;
  const others = all.filter((_, j) => j !== k);
  if (others.length) all[k][0].versions = { shown: k, others };
  c.messages.push(...all[k]);
}
function branch(c: Conversation, i: number, head: Message) {
  if (i >= c.messages.length) return void c.messages.push(head);
  const all = versionsAt(c, i);
  all.push([head]);
  showVersion(c, i, all, all.length - 1);
}
function* walk(list: Message[]): Generator<Message> {
  for (const m of list) {
    yield m;
    for (const v of m.versions?.others ?? []) yield* walk(v);
  }
}
function pager(m: Message) {
  const vs = m.versions;
  if (!vs) return "";
  const n = vs.others.length + 1, k = vs.shown, b = (act: "prev" | "next", on: boolean, label: string) =>
    `<button type="button" class="act" data-act="${act}" title="${label}" aria-label="${label}"${on ? "" : " disabled"}>${icon(act)}</button>`;
  return `<span class="pager" role="group" aria-label="Version ${k + 1} of ${n}">${b("prev", k > 0, "Previous version")}` +
    `<span class="pager-n" aria-hidden="true">${k + 1} / ${n}</span>${b("next", k < n - 1, "Next version")}</span>`;
}

function deleteMessage(i: number) {
  const c = conv;
  if (c.messages[i].versions) {
    const all = versionsAt(c, i), k = c.messages[i].versions!.shown, [gone] = all.splice(k, 1), role = gone[0].role;
    showVersion(c, i, all, Math.min(k, all.length - 1));
    persist(c);
    render();
    return toast(role === "user" ? "This version of the message is deleted" : "This version of the reply is deleted", () => {
      if (!c.messages[i]) return;
      const back = versionsAt(c, i);
      back.splice(k, 0, gone);
      showVersion(c, i, back, k);
      persist(c);
      if (c === conv) render();
    });
  }
  const [m] = c.messages.splice(i, 1);
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
  const body = h("div", "markdown"), foot = h("footer", "msg-foot"), rounds = h("div", "rounds"), calls = h("div", "tool-cards"), notes = h("div", "notes");
  el.dataset.i = String(i);
  think.append(summary, thought);
  el.append(rounds, think, body, calls, notes, foot);
  thought.addEventListener("wheel", (e) => { if (e.deltaY < 0) thought.dataset.held = "1"; }, { passive: true });
  thought.addEventListener("scroll", () => { if (thought.scrollHeight - thought.scrollTop - thought.clientHeight < 4) delete thought.dataset.held; });
  return { el, think, summary, thought, body, foot, rounds, calls, notes };
}

function roundView(r: Round, m: Message): HTMLElement {
  const el = h("div", "round"), body = h("div", "markdown"), calls = h("div", "tool-cards");
  if (r.reasoning) {
    const think = h("details", "thinking"), summary = h("summary", "", "Thoughts"), thought = h("div", "thought", r.reasoning);
    think.append(summary, thought);
    el.append(think);
  }
  if (r.content) renderBlocks({ body }, r.content, false, { c: conv, holder: r });
  syncCards(calls, r.calls, m, false);
  el.append(body, calls);
  return el;
}

const openCards = new Set<string>();
const callSince = new Map<string, number>();
const deciding = new Set<string>();
const LONG_RESULT = 1200;
function argLine(c: Call) {
  let a: unknown;
  try { a = c.args.trim() ? JSON.parse(c.args) : {}; } catch { return c.args; }
  if (!a || typeof a !== "object" || Array.isArray(a)) return c.args;
  const e = Object.entries(a);
  if (e.length === 1 && typeof e[0][1] === "string") return e[0][1];
  return e.map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`).join(", ");
}
function stateText(c: Call, live: boolean) {
  const s = c.state;
  if (s === "running") return `running · ${fmtS((performance.now() - (callSince.get(c.id) ?? performance.now())) / 1000)}`;
  if (s === "done") return c.ms !== undefined ? `done in ${fmtS(c.ms / 1000)}` : "done";
  return s === "approval" ? "asks to run" : s === "failed" ? "failed" : s === "declined" ? "declined" : s === "stopped" ? "stopped" : live ? "asked for" : "not run";
}
function resultView(c: Call): HTMLElement {
  const box = h("div", "tool-result"), text = c.result ?? "";
  let pretty: string | undefined;
  try { const j = JSON.parse(text); if (j && typeof j === "object") pretty = JSON.stringify(j, null, 2); } catch {  }
  if (pretty !== undefined) box.append(Object.assign(h("pre", "tool-pre"), { textContent: pretty }));
  else {
    const md = h("div", "markdown");
    md.innerHTML = markdown(text);
    typeset(md);
    box.append(md);
  }
  if (text.length > LONG_RESULT || text.split("\n").length > 24) {
    box.classList.add("long");
    const more = h("button", "btn tool-more", "Show all");
    more.type = "button";
    more.dataset.act = "more";
    box.append(more);
  }
  return box;
}
const cardLists = new WeakMap<HTMLElement, Call[]>();
function syncCards(box: HTMLElement, calls: Call[], m: Message, live: boolean) {
  cardLists.set(box, calls);
  const kept = new Map([...box.children].map((el) => [(el as HTMLElement).dataset.call!, el as HTMLElement]));
  calls.forEach((c, k) => {
    let el = kept.get(c.id);
    if (c.name === AGENT_FN) {
      el ??= agentCard(c);
      if (box.children[k] !== el) box.insertBefore(el, box.children[k] ?? null);
      return paintAgent(el, c, m, live);
    }
    if (!el) {
      el = h("div", "tool-card");
      el.dataset.call = c.id;
      const id = `tc-${c.id.replace(/[^\w-]/g, "")}`;
      el.innerHTML = `<button type="button" class="tool-head" data-act="card" aria-controls="${id}"><span class="tool-ico"></span>` +
        `<span class="tool-name"></span><span class="tool-src"></span><span class="tool-args"></span><span class="tool-state"></span>${icon("chevron")}</button>` +
        `<div class="tool-ask" hidden></div><div class="tool-body" id="${id}" hidden></div>`;
    }
    if (box.children[k] !== el) box.insertBefore(el, box.children[k] ?? null);
    const open = openCards.has(c.id), fetchTool = c.name === "fetch_url";
    el.dataset.state = c.state ?? (live ? "asked" : "none");
    const head = el.querySelector<HTMLElement>(".tool-head")!;
    head.setAttribute("aria-expanded", String(open));
    const set = (sel: string, v: string) => { const x = el.querySelector(sel)!; if (x.textContent !== v) x.textContent = v; };
    const ico = el.querySelector(".tool-ico")!, want = fetchTool ? "globe" : "wrench";
    if (ico.getAttribute("data-i") !== want) { ico.setAttribute("data-i", want); ico.innerHTML = icon(want); }
    set(".tool-name", c.title ?? c.name);
    set(".tool-src", fetchTool ? "" : c.source ?? "");
    set(".tool-args", argLine(c));
    set(".tool-state", stateText(c, live));
    const ask = el.querySelector<HTMLElement>(".tool-ask")!, asking = c.state === "approval" && live && !!m.job;
    ask.hidden = !asking;
    if (asking && ask.dataset.for !== `${c.id}:${deciding.has(c.id)}`) {
      ask.dataset.for = `${c.id}:${deciding.has(c.id)}`;
      const dis = deciding.has(c.id) ? " disabled" : "";
      ask.innerHTML = `<pre class="tool-pre"></pre><div class="tool-ask-row"><span>Run this ${c.readOnly === false ? "tool, which can change things" : "tool"}?</span>` +
        `<button type="button" class="btn" data-act="decide" data-decision="deny"${dis}>Deny</button>` +
        `<button type="button" class="btn" data-act="decide" data-decision="always" title="Run ${escapeHtml(c.title ?? c.name)} without asking from now on (Settings lists it)"${dis}>Always allow</button>` +
        `<button type="button" class="btn primary" data-act="decide" data-decision="once"${dis}>Allow once</button></div>`;
      ask.querySelector("pre")!.textContent = prettyArgs(c.args);
    }
    const body = el.querySelector<HTMLElement>(".tool-body")!;
    body.hidden = !open;
    const key = `${c.state}:${c.result?.length ?? -1}`;
    if (open && body.dataset.key !== key) {
      body.dataset.key = key;
      body.replaceChildren(h("div", "tool-label", "Arguments"), Object.assign(h("pre", "tool-pre"), { textContent: prettyArgs(c.args) }));
      if (c.result !== undefined) body.append(h("div", "tool-label", "Result"), resultView(c));
    }
  });
  for (const el of [...box.children].slice(calls.length)) el.remove();
}
const prettyArgs = (args: string) => { try { return JSON.stringify(JSON.parse(args), null, 2); } catch { return args; } };

const AGENT_FN = "subagent";
interface AgentView { task: HTMLElement; log: HTMLElement; think: HTMLDetailsElement; thought: HTMLElement; calls: HTMLElement;
  answerLabel: HTMLElement; answer: { body: HTMLElement; blocks?: Blocks }; notes: HTMLElement; stop: HTMLButtonElement; logShown: number }
const agentViews = new WeakMap<HTMLElement, AgentView>();
function agentArgs(c: Call): { description?: unknown; prompt?: unknown } {
  try { const a = JSON.parse(c.args); return a && typeof a === "object" ? a : {}; } catch { return {}; }
}
const agentCalls = (a?: AgentRun) => [...(a?.rounds ?? []).flatMap((r) => r.calls), ...(a?.calls ?? [])];
function gist(text: string) {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  const line = lines.find((l) => !/^#{1,6}\s/.test(l) && !/^\*\*[^*]+\*\*:?$/.test(l)) ?? lines[0] ?? "";
  return line.replace(/^(([#>*+|-]|\d+\.)\s*)+/, "").replace(/\*\*|__|`/g, "").trim();
}
function agentNow(c: Call, live: boolean) {
  const a = c.agent;
  if (c.state === "done") return gist(a?.content || c.result || "") || "Done";
  if (c.state === "failed") return a?.error || gist(c.result ?? "") || "Failed";
  if (c.state === "stopped") return "Stopped before it finished";
  if (c.state !== "running") return live ? "Waiting to start" : "Not run";
  const x = a?.calls?.find((y) => y.state === "approval") ?? a?.calls?.find((y) => y.state === "running");
  if (x) return x.state === "approval" ? `Asks to run ${x.title ?? x.name}` : `${x.title ?? x.name}: ${argLine(x)}`;
  if (a?.content) return "Writing its answer";
  if (a?.reasoning) return "Thinking";
  if (a?.rounds?.length) return "Reading the results";
  return a?.read ? "Working" : "Starting";
}
function agentCard(c: Call): HTMLElement {
  const el = h("div", "tool-card agent-card"), id = `tc-${c.id.replace(/[^\w-]/g, "")}`;
  el.dataset.call = c.id;
  el.innerHTML = `<button type="button" class="tool-head agent-head" data-act="card" aria-controls="${id}"><span class="tool-ico">${icon("agent")}</span>` +
    `<span class="agent-text"><span class="agent-title"><span class="tool-name"></span><span class="tool-src">Sub-agent</span></span>` +
    `<span class="agent-now"></span></span><span class="agent-meta"></span><span class="tool-state"></span>${icon("chevron")}</button>` +
    `<div class="tool-body agent-body" id="${id}" hidden></div>`;
  return el;
}
function paintAgent(el: HTMLElement, c: Call, m: Message, live: boolean) {
  const a = c.agent, open = openCards.has(c.id), n = agentCalls(a).length, label = agentArgs(c).description;
  const set = (x: Element, v: string) => { if (x.textContent !== v) x.textContent = v; };
  el.dataset.state = c.state ?? (live ? "asked" : "none");
  set(el.querySelector(".tool-name")!, typeof label === "string" && label.trim() ? label.trim() : "Sub-agent");
  set(el.querySelector(".agent-now")!, agentNow(c, live));
  set(el.querySelector(".agent-meta")!, [n ? `${n} tool${n === 1 ? "" : "s"}` : "", a?.tokens ? `${fmtK(a.tokens)} tokens` : ""].filter(Boolean).join(" · "));
  set(el.querySelector(".tool-state")!, stateText(c, live));
  el.querySelector(".tool-head")!.setAttribute("aria-expanded", String(open));
  const body = el.querySelector<HTMLElement>(".agent-body")!;
  body.hidden = !open;
  if (!open) return;
  let v = agentViews.get(el);
  if (!v) {
    const task = agentArgs(c).prompt;
    v = { task: h("div", "agent-task", typeof task === "string" ? task : c.args), log: h("div", "agent-log"), think: h("details", "thinking"),
      thought: h("div", "thought"), calls: h("div", "tool-cards"), answerLabel: h("div", "tool-label", "Answer"), answer: { body: h("div", "markdown") },
      notes: h("div", "agent-notes"), stop: h("button", "btn small agent-stop", "Stop sub-agent"), logShown: -1 };
    v.think.append(h("summary", "", "Thoughts"), v.thought);
    v.stop.type = "button";
    v.stop.dataset.act = "agent-stop";
    body.append(h("div", "tool-label", "Task"), v.task, v.log, v.think, v.calls, v.answerLabel, v.answer.body, v.notes, v.stop);
    agentViews.set(el, v);
  }
  const working = c.state === "running" && live, rounds = a?.rounds ?? [];
  if (v.logShown !== rounds.length) {
    v.log.replaceChildren(...rounds.map((r) => roundView(r, m)));
    v.logShown = rounds.length;
  }
  v.think.hidden = !a?.reasoning;
  v.think.classList.toggle("live", working && !!a?.reasoning && !a.content && !a.calls?.length);
  set(v.think.firstElementChild!, v.think.classList.contains("live") ? "Thinking" : "Thoughts");
  set(v.thought, a?.reasoning ?? "");
  syncCards(v.calls, a?.calls ?? [], m, live);
  v.answerLabel.hidden = !a?.content;
  renderBlocks(v.answer, a?.content ?? "", working);
  const said = [...(a?.notes ?? []), ...(c.state === "failed" || c.state === "stopped" ? [c.result ?? ""] : [])].filter(Boolean);
  if (v.notes.childElementCount !== said.length) v.notes.replaceChildren(...said.map((x) => h("div", "note", x)));
  v.stop.hidden = !(working && m.job);
}

type Act = keyof typeof TITLES | "cont";
const slots = (list: [Act, boolean, string?][]) => list.map(([a, on, html]) => !on ? `<span class="act ghost" aria-hidden="true"></span>`
  : html ?? `<button type="button" class="act" data-act="${a}" title="${TITLES[a as keyof typeof TITLES]}">${icon(a)}</button>`).join("");

function paint(v: View, m: Message, live?: Live) {
  const elapsed = live ? (performance.now() - live.since) / 1000 : 0;
  if (v.roundsShown !== (m.rounds?.length ?? 0)) {
    v.rounds.replaceChildren(...(m.rounds ?? []).map((r) => roundView(r, m)));
    v.roundsShown = m.rounds?.length ?? 0;
  }
  syncCards(v.calls, m.calls ?? [], m, !!live);
  if (v.notes.childElementCount !== (m.notes?.length ?? 0)) v.notes.replaceChildren(...(m.notes ?? []).map((n) => h("div", "note", n)));
  if (live && v.foot.dataset.pager !== pager(m)) v.foot.innerHTML = v.foot.dataset.pager = pager(m);
  v.think.hidden = !m.reasoning;
  v.think.classList.toggle("live", live?.phase === "thinking");
  if (m.reasoning) {
    v.summary.textContent = live?.phase === "thinking" ? `Thinking · ${fmtS(elapsed)}`
      : m.stats?.thought ? `Thought for ${fmtS(m.stats.thought)}` : "Thoughts";
  }
  const asTokens = !!m.id && tokenView.has(m.id) && tokens.has(m.id);
  v.el.classList.toggle("as-tokens", asTokens);
  if (asTokens) {
    if (!live) v.tok = undefined;
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
  if (asTokens) {  } else if (live?.phase === "waiting") {
    v.body.innerHTML = `<p class="waiting">${live.loading ? "" : `${live.note} · ${fmtS(elapsed)}`}</p>`;
    v.shown = v.blocks = undefined;
  } else if (v.shown !== m.content) {
    renderBlocks(v, m.content, !!live, { c: conv, holder: m });
    v.shown = m.content;
  }
  if (live) return;
  if (m.error) v.body.append(h("div", "error", m.error));
  const s = m.stats;
  const facts = s ? [...(s.model ? [modelName(s.model)] : []), `${fmtS(s.ttft)} to first token`, `${Math.round(s.tps)} tok/s`, `${fmtK(s.tokens)} tokens`, s.cached === undefined ? `prompt ${fmtK(s.prompt)}` : `prompt ${fmtK(s.cached)}/${fmtK(s.prompt)} cached`]
    : m.stopped ? ["stopped"] : [];
  if (typeof m.voiceMs === "number") facts.unshift(`voice ${(m.voiceMs / 1000).toFixed(2)} s`);
  if (m.interrupted) facts.push("interrupted");
  if (m.params) facts.push(m.params);
  if (m.edited) facts.push("edited");
  if (s?.seed !== undefined) facts.push(`<button type="button" class="seed" data-act="seed" data-seed="${s.seed}" title="Use this seed for the next replies">seed ${s.seed}</button>`);
  const last = conv.messages.at(-1) === m, busy = !!running();
  const ranCalls = !!m.calls?.some((c) => c.result !== undefined), canCont = onChrysalis() || ranCalls;
  const cut = m.finish === "length" || m.stopped || (ranCalls && !m.error);
  const cont = cut ? `<button type="button" class="btn" data-act="cont" title="Continue">${icon("cont")}Continue</button>`
    : `<button type="button" class="act" data-act="cont" title="Continue">${icon("cont")}</button>`;
  v.foot.innerHTML = pager(m) + `<span class="stats">${facts.join(" · ")}</span>` + slots([["cont", last && !busy && canCont, cont], ["retry", !busy]])
    + slots([["tokens", !!m.id && tokens.has(m.id)], ["speak", !!m.content.trim() && !voice, reading === m ? `<button type="button" class="act on" data-act="speak" title="Stop reading">${icon("stop")}</button>` : undefined],
      ["copy", true], ["edit", !busy], ["trash", !busy]]);
  delete v.foot.dataset.pager;
  v.foot.querySelector("[data-act=tokens]")?.classList.toggle("on", asTokens);
}

const ocean = new Ocean($("sea")), seaReady = ocean.start(), seaSwitch = $<HTMLButtonElement>("sea-switch");
function emptyState() {
  const e = h("div", "empty");
  if (seaReady) ocean.butterfly(e);
  else e.innerHTML = `<img class="logo" src="${logo}" alt="">`;
  return e;
}
function showSea(on: boolean) {
  document.querySelector("main")!.toggleAttribute("data-sea", on && seaReady);
  seaSwitch.setAttribute("aria-checked", String(on && seaReady));
}
seaSwitch.disabled = !seaReady;
if (!seaReady) seaSwitch.title = "This browser cannot draw it";
seaSwitch.onclick = () => {
  const on = seaSwitch.getAttribute("aria-checked") !== "true";
  setPref("sea", on ? "on" : "off");
  showSea(on);
};
showSea(pref("sea", "on") === "on");
const themePick = $("theme-pick");
for (const [t, label] of [["system", "System"], ["light", "Light"], ["dark", "Dark"]] as [Theme, string][]) {
  const b = h("button", "", label);
  b.type = "button";
  b.setAttribute("role", "radio");
  b.dataset.theme = t;
  b.onclick = () => { choose(t); markTheme(); };
  themePick.append(b);
}
const connectView = $<HTMLDialogElement>("connect-view");
let known: Set<string> | undefined, joined = "", connectTimer = 0;
interface Paired { id: string; name: string; added: number; seen: number; kind: "browser" | "computer" }
interface Joining { code: string; name: string; at: number; allowed: boolean }
const since = (t: number) => {
  const m = Math.round((Date.now() - t) / 60_000);
  return m < 1 ? "just now" : m < 60 ? `${m} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : day(t);
};
const day = (t: number) => new Date(t).toLocaleDateString(undefined, { month: "short", day: "numeric" });
const spaced = (code: string) => `${code.slice(0, 3)} ${code.slice(3)}`;
async function answer(how: "allow" | "deny", code: string) {
  await fetch(`/api/pair/${how}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code }) }).catch(() => {});
  $("connect-body").dataset.key = "";
  void renderConnect();
}
async function renderConnect() {
  const p = await fetch("/api/pair").then((r) => r.ok ? r.json() : null).catch(() => null);
  if (!connectView.open) return;
  clearTimeout(connectTimer);
  connectTimer = window.setTimeout(renderConnect, 2000);
  const box = $("connect-body");
  if (!p) {
    box.dataset.key = "";
    return void box.replaceChildren(h("p", "set-note", "Cannot reach Chrysalis's web server. Trying again…"));
  }
  if (!p.lan) {
    const note = h("p", "set-note");
    note.append("Chrysalis is on this computer only. Run ", h("code", "", "chrysalis.sh --host 0.0.0.0"), " in a terminal to let your phone and other devices on this network in.");
    box.dataset.key = "";
    return void box.replaceChildren(note);
  }
  const list: Paired[] = p.devices, asking: Joining[] = p.waiting, fresh = known && list.find((d) => !known!.has(d.id));
  if (fresh) joined = fresh.name;
  known = new Set(list.map((d) => d.id));
  const key = JSON.stringify([joined, asking, list.map((d) => [d.id, since(d.seen)]), p.connect, !!p.qr]);
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  const row = h("div", "dev-row"), text = h("div", "dev-text"), other = h("p");
  if (p.qr) {
    const qr = h("img", "dev-qr");
    qr.src = `data:image/svg+xml;base64,${btoa(p.qr)}`;
    qr.alt = "QR code of Chrysalis's address on this network";
    row.append(qr);
  }
  other.append(p.qr ? "No camera? On the device, open " : "On the device, open ", h("code", "dev-url", p.connect[0].replace(/\/$/, "")), ".");
  text.append(h("p", "dev-lead", `${p.qr ? "Scan this with your phone's camera." : "The device connects there."} It warns once that the connection is not private, as it does not trust this computer yet: go on past it, and the page sets the device up. Then the device shows a code, which appears here to allow.`),
    other, h("p", "set-note", "A computer: open Chrysalis there. It finds this one, and asks with a code that appears here too."),
    ...(p.qr ? [] : [h("p", "set-note", "With qrencode installed, a QR code shows here for phones to scan.")]));
  row.append(text);
  const asks = h("div", "dev-list");
  if (asking.length) asks.append(h("div", "set-title", "Asking to join"));
  for (const a of asking) {
    const item = h("div", "dev-ask"), allow = h("button", "btn small primary", a.allowed ? "Allowed" : "Allow"), deny = h("button", "btn small", "Deny");
    allow.type = deny.type = "button";
    allow.disabled = a.allowed;
    allow.onclick = () => void answer("allow", a.code);
    deny.onclick = () => void answer("deny", a.code);
    item.append(h("span", "dev-name", a.name), h("code", "dev-ask-code", spaced(a.code)), deny, allow);
    asks.append(item);
  }
  if (asking.length) asks.append(h("p", "set-note", "Allow the one whose code your device shows: one code per device, good for 10 minutes."));
  const devs = h("div", "dev-list");
  devs.append(h("div", "set-title", "Let in"));
  if (!list.length) devs.append(h("p", "set-note", "None yet."));
  for (const d of list) {
    const item = h("div", "dev-item"), remove = h("button", "btn small", "Remove");
    remove.type = "button";
    remove.onclick = async () => { await fetch(`/api/pair/${d.id}`, { method: "DELETE" }); box.dataset.key = ""; void renderConnect(); };
    item.append(h("span", "dev-name", d.name), h("span", "dev-when", `${d.kind === "computer" ? "linked computer · " : ""}added ${day(d.added)} · used ${since(d.seen)}`), remove);
    devs.append(item);
  }
  const fp = h("p", "set-note dev-fp");
  fp.append(`To be sure a device trusts this computer and not an impostor on the network, its certificate details show the same SHA-256 fingerprint for ${p.name}: `,
    h("code", "", p.fingerprint));
  const done = h("p", "dev-joined");
  done.innerHTML = icon("check");
  done.append(`${joined} is connected.`);
  box.replaceChildren(...(joined ? [done] : []), row, ...(asking.length ? [asks] : []), devs, fp);
}
function openConnect() {
  joined = "";
  known = undefined;
  connectView.showModal();
  void renderConnect();
}
let checking: Promise<boolean> | undefined;
function checkTrust(): Promise<boolean> {
  return checking ??= new Promise<boolean>((done) => {
    const frame = h("iframe");
    const finish = (ok: boolean) => { removeEventListener("message", heard); clearTimeout(timer); frame.remove(); checking = undefined; done(ok); };
    const heard = (e: MessageEvent) => { if (e.source === frame.contentWindow && typeof e.data?.chrysalisCheck === "boolean") finish(e.data.chrysalisCheck); };
    const timer = setTimeout(() => finish(false), 10_000);
    addEventListener("message", heard);
    frame.hidden = true;
    frame.src = "/check-frame";
    document.body.append(frame);
  });
}
function markTheme() {
  for (const b of themePick.querySelectorAll<HTMLElement>("button")) b.setAttribute("aria-checked", String(b.dataset.theme === chosen()));
}
markTheme();

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
  v.shown = v.blocks = undefined;
}

const SPECIAL: Record<string, string> = { "<|im_end|>": "end of reply", "<|endoftext|>": "end of reply", "</think>": "end of thinking" };
const showTok = (t: string) => SPECIAL[t] ?? (t.replace(/\n/g, "↵").replace(/^ | $/g, "·") || "∅");

function closeTokPop() {
  $("tokpop").hidden = true;
  thread.querySelector(".tok.picked")?.classList.remove("picked");
}
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
  const c = conv, at = c.messages.indexOf(m);
  if (at < 0 || running(c)) return;
  const td = structuredClone(tokens.get(m.id!)!), before = td[ch].slice(0, i).map((x) => x.t).join("");
  const { versions, stats, edited, ...rest } = m, copy: Message = { ...structuredClone(rest), id: randomId() };
  tokens.set(copy.id!, td);
  if (tokenView.has(m.id!)) tokenView.add(copy.id!);
  branch(c, at, copy);
  if (SPECIAL[a.t] === "end of reply") {
    td[ch] = td[ch].slice(0, i);
    copy[ch] = before;
    if (ch === "reasoning") { td.content = []; copy.content = ""; }
    copy.finish = "stop";
    save(c);
    render();
    return toast("The reply ends there now, as a new version");
  }
  continueReply(copy, { ch, i, alt: a, before });
}


function renderComposer() {
  const busy = !!running();
  sendBtn.innerHTML = icon(busy || voice ? "stop" : "send");
  sendBtn.setAttribute("aria-label", voice ? "End voice mode" : busy ? "Stop" : "Send");
  sendBtn.title = voice ? "End voice mode" : "";
  sendBtn.classList.toggle("stop", busy || !!voice);
  const draftBtn = $<HTMLButtonElement>("draft");
  draftBtn.innerHTML = icon(drafting ? "stop" : "wand");
  draftBtn.title = drafting ? "Stop the draft" : input.value.trim() ? "Finish my message (draft)" : "Draft my next message";
  draftBtn.setAttribute("aria-label", draftBtn.title);
  draftBtn.classList.toggle("on", !!drafting);
  draftBtn.hidden = busy || !conv.messages.length || !onChrysalis();
  $("composer").classList.toggle("drafting", !!drafting);
  renderControls();
  if (modeOf(conv) !== chatMode) {
    chatMode = modeOf(conv);
    syncSettings();
  }
}

const customized = () => settings.system !== DEFAULT_SYSTEM
  || Object.keys(settings.think).length + Object.keys(settings.fast).length > 0
  || !!settings.max_tokens || settings.seed !== undefined || !!settings.allow?.length || settings.fetchChars !== undefined
  || Object.keys(settings.voice ?? {}).length > 0;

function renderList() {
  const folders = [...[...people.values()].filter((p) => p.kind !== "persona"), ...groups.values()].map((x) => ({ id: x.id, list: chatsOf(x.id), at: x.added }));
  $("conversations").replaceChildren(...folders.sort((a, b) => (b.list[0]?.updated ?? b.at) - (a.list[0]?.updated ?? a.at)).map(({ id, list }) => {
    const open = unfolded.has(id) && list.length > 0, box = h("div", `folder${open ? " open" : ""}`);
    box.append(castRow(id, list, open));
    if (open) {
      const kids = h("div", "folder-chats");
      kids.append(...list.map(convRow));
      box.append(kids);
    }
    return box;
  }));
}
function ago(t: number) {
  const s = (Date.now() - t) / 1000;
  return s < 60 ? "now" : s < 3600 ? `${Math.floor(s / 60)}m` : s < 86400 ? `${Math.floor(s / 3600)}h` : s < 7 * 86400 ? `${Math.floor(s / 86400)}d`
    : s < 30 * 86400 ? `${Math.floor(s / 604800)}w` : new Date(t).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
function convRow(s: Conversation) {
  const a = h("a", `conv${s.id === conv.id ? " active" : ""}${runs.has(s.id) ? " running" : ""}`), del = h("button", "conv-del", "×");
  a.href = `#${s.id}`;
  a.onclick = (e) => { if (!e.defaultPrevented) closeOver(panes.conversations); };
  del.type = "button";
  del.title = "Delete";
  del.onclick = (e) => {
    e.preventDefault();
    if (del.classList.contains("armed")) return void remove(s.id);
    del.classList.add("armed");
    del.textContent = "Delete";
    setTimeout(() => { del.classList.remove("armed"); del.textContent = "×"; }, 2500);
  };
  a.append(h("span", "conv-title", s.title), h("span", "conv-ago", ago(s.updated)), del);
  return a;
}

let picked = pref("model", MODELS[0][0]);
const onChrysalis = () => status.backend?.chrysalis ?? true;
const onService = () => status.backend?.service ?? true;
const listed = () => {
  const ids = status.backend?.listed?.ok ? status.backend.listed.models.map((m) => m.id) : [];
  return onChrysalis() ? ids.filter((id) => !(id.endsWith("-fast") && ids.includes(id.slice(0, -5)))) : ids;
};
const otherModel = () => {
  const mine = pref(`model@${status.backend?.url}`, "");
  return listed().includes(mine) ? mine : listed()[0] ?? mine;
};
let switching: string | undefined;
const current = () => !onService() ? otherModel() : switching ?? status.loading ?? (status.state === "online" ? status.model : undefined) ?? picked;
const quick = () => (onChrysalis() ? { model: `${current()}-fast` } : { model: current(), reasoning_effort: "none" });

async function start() {
  await fetch("/api/chrysalis/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: picked }) });
}

async function pickModel(model: string) {
  picked = model;
  setPref("model", model);
  switching = undefined;
  if (status.state !== "online" || status.model === model) return renderStatus();
  switching = model;
  status = { ...status, loading: model, progress: undefined };
  renderStatus();
  if (!conv.messages.length) render();
  try {
    const r = await fetch("/api/chrysalis/load", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model }) });
    if (!r.ok) console.warn("model switch failed:", await r.text());
  } finally {
    if (switching === model) switching = undefined;
    await refreshStatus();
  }
}

function renderStatus() {
  renderControls();
}

let asked = 0;
async function refreshStatus(again = true) {
  const was = current();
  try {
    const r = await fetch("/api/status");
    if (r.status === 401) {
      if ((await r.json().catch(() => ({}))).check && again && await checkTrust()) return refreshStatus(false);
      return location.replace("/");
    }
    status = await r.json();
  } catch { status = { state: "offline", backend: status.backend }; }
  if (status.check) void checkTrust();
  $("connect").hidden = !status.here;
  $("connect").classList.toggle("asking", !!status.waiting);
  if ((status.waiting ?? 0) > asked && !connectView.open) toast("A device is asking to join: Connect a device, the phone at the top");
  asked = status.waiting ?? 0;
  build ||= status.build ?? "";
  if (status.state === "online" && status.model && !status.loading && status.model !== picked) {
    picked = status.model;
    setPref("model", picked);
  }
  if (current() !== was && !conv.messages.length) render();
  if (status.build && status.build !== build && !runs.size && !input.value && !panel.contains(document.activeElement)
    && !editing && !drafting) return location.reload();
  renderStatus();
}

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
  const at = location.hash.slice(1);
  if (c === conv && at !== c.id && !chats[at]) history.replaceState(null, "", `#${c.id}`);
  renderList();
}

function request(c: Conversation, upto: Message[], o: { effort?: Effort; sampling?: Mode; final?: object; extra?: Record<string, unknown>; loop?: boolean; route?: boolean; as?: string } = {}) {
  const effort = o.effort ?? c.effort, mode: Mode = effort === "off" ? "fast" : "think";
  const messages: object[] = upto.flatMap(wire), system = systemFor(c), { members, group } = castOf(c);
  const writer = writerOf(c), char = writer?.card.data.name ?? members[0]?.card.data.name ?? "";
  const after = [writer, persona()].map((p) => p?.card.data.post_history_instructions?.trim()).filter(Boolean).map((t) => `[${fillCard(t!, char, userName())}]`);
  const asked = [...messages].reverse().find((x) => (x as { role: string }).role === "user") as { content: string | { type: string; text?: string }[] } | undefined;
  if (after.length && asked) {
    const note = `\n\n${after.join("\n\n")}`;
    if (typeof asked.content === "string") asked.content += note;
    else asked.content = [...asked.content, { type: "text", text: note }];
  }
  if ((members.length || c.messages[0]?.role === "assistant") && (messages[0] as { role: string } | undefined)?.role !== "user") messages.unshift({ role: "user", content: "[The scene begins.]" });
  const next = o.as && people.get(o.as)?.card.data.name, last = messages.at(-1) as { role: string; content: unknown } | undefined;
  if (next && last?.role === "user" && typeof last.content === "string") last.content += `\n\n[${next} next.]`;
  else if (next) messages.push({ role: "user", content: `[${next} next.]` });
  if (system) messages.unshift({ role: "system", content: system });
  if (o.final) messages.push(o.final);
  const web: ToolChoice | undefined = toolsOn(c)
    ? { tools: c.tools!, allow: settings.allow, fetchChars: settings.fetchChars, ...(o.loop === false ? { loop: false } : {}),
      ...(o.route ? { route: true } : {}) } : undefined;
  const chrysalis = onChrysalis();
  return {
    model: chrysalis && effort === "off" ? `${current()}-fast` : current(), reasoning_effort: chrysalis ? (effort === "off" ? undefined : effort) : OTHER_EFFORT[effort],
    ...samplerFields(o.sampling ?? mode), max_tokens: settings.max_tokens, seed: settings.seed, ...(chrysalis ? { logprobs: true, top_logprobs: 5 } : {}),
    messages, stream: true, stream_options: { include_usage: true }, ...(web ? { chrysalis_web: web } : {}),
    ...(group ? { stop: [`\n${userName()}:`, `\n**${userName()}`] } : {}), ...o.extra,
  };
}
const OTHER_EFFORT: Record<Effort, string | undefined> = { xhigh: undefined, medium: "medium", low: "low", off: "none" };
const ran = (c: Call) => c.result !== undefined;
const TALKED_OVER = " — [cut off: the user talked over this reply]";
function wire(m: Message): object[] {
  if (m.role === "user") return [{ role: "user", content: userContent(m) }];
  const ended = !!m.calls?.some(ran);
  const rounds = [...(m.rounds ?? []), ...(ended ? [{ content: m.content, reasoning: m.reasoning, calls: m.calls! }] : [])];
  const out = rounds.flatMap((r) => {
    const done = r.calls.filter(ran);
    return [assistantTurn(r.content, r.reasoning, done), ...done.map((x) => toolTurn(x.id, x.result!))];
  });
  if (!ended && (m.content || m.reasoning)) out.push(assistantTurn(m.interrupted ? m.content + TALKED_OVER : m.content, m.reasoning));
  return out;
}
const toolsOn = (c: Conversation) => !!c.tools && Object.keys(c.tools).length > 0;
const systemFor = (c: Conversation) => castOf(c).members.length ? castSystem(c)
  : [castPrompt(fill((settings.system.trim() === DEFAULT_SYSTEM && toolsOn(c) ? DEFAULT_SYSTEM_TOOLS : settings.system).trim()),
    [people.get(CHRYSALIS)!.card.data], userCard()),
    settings.artifacts === false ? "" : ARTIFACTS_PROMPT].filter(Boolean).join("\n\n");

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

async function ensureOnline(run: { stopping: boolean }) {
  if (!onService()) {
    if (status.state !== "online") await refreshStatus();
    const l = status.backend?.listed;
    if (l && !l.ok) throw new Error(l.error);
    if (!current()) throw new Error(`${status.backend?.name ?? "The backend"} lists no model to chat with`);
    return;
  }
  if (status.state === "offline") await start();
  for (const deadline = Date.now() + 180_000; status.state !== "online" || status.loading; await refreshStatus()) {
    if (run.stopping) throw new DOMException("stopped", "AbortError");
    if (Date.now() > deadline) throw new Error("Chrysalis did not start: its terminal says why");
    await sleep(100);
  }
}

const CUT_OFF = "The reply was cut off: the web server restarted while it was being written";

async function generate(c: Conversation = conv, at = c.messages.length, as = c.messages[at]?.as) {
  const m: Message = { id: randomId(), role: "assistant", content: "", reasoning: "", params: changed(modeOf(c)) || undefined, effort: c.effort, ...(as ? { as } : {}) };
  branch(c, at, m);
  const upto = c.messages.slice(0, at);
  await startRun(c, m, () => request(c, upto, { as }));
}

function continueReply(m: Message, pick?: { ch: "reasoning" | "content"; i: number; alt: Alt; before: string }) {
  const c = conv, at = c.messages.indexOf(m);
  if (at < 0 || running(c)) return;
  const effort = m.effort ?? c.effort;
  c.messages.splice(at + 1);
  m.error = m.finish = undefined;
  m.stopped = false;
  m.params = changed(effort === "off" ? "fast" : "think") || undefined;
  if (!pick && m.calls?.some((x) => x.result !== undefined)) {
    m.rounds = [...(m.rounds ?? []), { content: m.content, ...(m.reasoning ? { reasoning: m.reasoning } : {}), calls: m.calls }];
    m.content = m.reasoning = "";
    m.calls = undefined;
    if (m.id) tokens.delete(m.id);
    void startRun(c, m, () => request(c, c.messages.slice(0, at + 1), { effort }));
    return;
  }
  const original = { role: "assistant", content: m.content, reasoning_content: m.reasoning ?? "" };
  const final = !pick ? original : pick.ch === "content" ? { ...original, content: pick.before } : { role: "assistant", content: "", reasoning_content: pick.before };
  const extra = pick ? { continue_final_message: true, chrysalis_original: original, chrysalis_append_token: pick.alt.id } : { continue_final_message: true };
  const upto = [...c.messages.slice(0, at), { ...m, content: "", reasoning: "", calls: undefined }];
  void startRun(c, m, () => request(c, upto, { effort, final, extra }),
    pick && { t: pick.alt.t, p: pick.alt.p, alts: tokens.get(m.id!)?.[pick.ch][pick.i]?.alts ?? [] });
}

let leaving = false;
addEventListener("pagehide", () => { leaving = true; });
async function startRun(c: Conversation, m: Message, body: () => object, pick?: Tok) {
  const phase = m.content ? "writing" : m.reasoning ? "thinking" : "waiting";
  const run: Run = { conv: c, m, live: { phase, since: performance.now(), note: "Reading the prompt" }, at: 0, stopping: false, frame: 0, saved: 0, pick };
  m.base = m.rounds?.length || undefined;
  m.notes = undefined;
  runs.set(c.id, run);
  if (c === conv) {
    follow = true;
    render();
  }
  m.starting = true;
  save(c);
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
    m.job = res.headers.get("X-Chrysalis-Job") ?? undefined;
    delete m.starting;
    save(c);
    if (run.stopping) stopRun(run, true);
    await stream(run, res.body);
  } catch (e) {
    if (leaving) return;
    if (run.stopping) m.stopped = true;
    else m.error = e instanceof Error ? e.message : String(e);
  }
  finish(run);
}

async function resume(c: Conversation, m: Message) {
  const run: Run = { conv: c, m, live: { phase: "waiting", since: performance.now(), note: "Reading the prompt" }, at: 0, stopping: false, frame: 0, saved: 0 };
  runs.set(c.id, run);
  try {
    const res = await fetch(`/api/chat/${m.job}?from=0`);
    if (res.status === 404) throw new Error(CUT_OFF);
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    m.content = m.reasoning = "";
    m.rounds = m.rounds?.slice(0, m.base ?? 0);
    if (!m.rounds?.length) delete m.rounds;
    m.calls = m.notes = undefined;
    if (m.id) tokens.delete(m.id);
    await stream(run, res.body);
  } catch (e) {
    if (run.stopping) m.stopped = true;
    else m.error = e instanceof Error ? e.message : String(e);
  }
  finish(run);
}

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
      } catch {  }
    }
    body = null;
    if (run.stopping && run.at === 0 && !run.m.job) throw new DOMException("stopped", "AbortError");
    await sleep(Math.min(8000, 300 * 2 ** attempt++));
    const res = await fetch(`/api/chat/${run.m.job}?from=${run.at}`).catch(() => null);
    if (res?.status === 404) throw new Error(CUT_OFF);
    if (res?.ok) body = res.body;
    else if (run.stopping) throw new DOMException("stopped", "AbortError");
  }
}

function take(run: Run, ev: any) {
  const { m, live } = run, choice = ev.choices?.[0], d = choice?.delta ?? {}, now = performance.now(), v = views.get(m);
  if (ev.error) m.error = ev.error.message;
  if (choice?.finish_reason) m.finish = choice.finish_reason;
  if (d.chrysalis_prefix) {
    m.reasoning = d.chrysalis_prefix.reasoning_content?.trim() ? d.chrysalis_prefix.reasoning_content : "";
    m.content = d.chrysalis_prefix.content ?? "";
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
  if (ev.chrysalis_prompt) { run.read = ev.chrysalis_prompt.prompt_tokens; run.written = 0; }
  if (d.content || d.reasoning_content) run.written = (run.written ?? 0) + 1;
  if (ev.usage) {
    const u = run.usage ?? { prompt_tokens: 0, completion_tokens: 0 }, cached = ev.usage.prompt_tokens_details?.cached_tokens;
    run.usage = { prompt_tokens: u.prompt_tokens + ev.usage.prompt_tokens, completion_tokens: u.completion_tokens + ev.usage.completion_tokens,
      cached: cached === undefined ? u.cached : (u.cached ?? 0) + cached };
    run.last = ev.usage;
  }
  if (ev.seed !== undefined) run.seed ??= ev.seed;
  for (const c of d.tool_calls ?? []) {
    if (live.phase !== "writing") {
      if (live.phase === "thinking" && v) v.think.open = false;
      live.phase = "writing";
    }
    addCall(m, c);
  }
  if (ev.tool) toolEvent(m.calls, ev.tool, now);
  if (ev.agent) {
    const c = m.calls?.find((x) => x.id === ev.agent.id);
    if (c) feedAgent(c, ev.agent.ev, now);
  }
  if (ev.round) {
    m.rounds = [...(m.rounds ?? []), { content: m.content, ...(m.reasoning ? { reasoning: m.reasoning } : {}), calls: m.calls ?? [] }];
    m.content = m.reasoning = "";
    m.calls = undefined;
    if (m.id) tokens.delete(m.id);
    if (v) v.tok = undefined;
    Object.assign(live, { phase: "waiting", since: now, note: "Reading the results" });
  }
  if (ev.note) m.notes = [...(m.notes ?? []), ev.note];

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
  if (now - run.saved > 2000) {
    run.saved = now;
    save(run.conv);
  }
  feedVoice(m);
  schedule(run);
}

function addCall(holder: { calls?: Call[] }, t: any) {
  const calls = (holder.calls ??= []), at = (calls[t.index ?? calls.length] ??= { id: "", name: "", args: "" });
  if (t.id) at.id = t.id;
  if (t.function?.name) at.name += t.function.name;
  if (t.function?.arguments) at.args += t.function.arguments;
}
function toolEvent(calls: Call[] | undefined, tool: any, now: number): Call | undefined {
  const c = calls?.find((x) => x.id === tool.id);
  if (!c) return undefined;
  for (const k of ["title", "source", "readOnly", "state", "result", "ms"] as const) if (tool[k] !== undefined) (c as any)[k] = tool[k];
  if (c.state === "running") callSince.set(c.id, now);
  if (c.state === "approval") openCards.delete(c.id);
  deciding.delete(c.id);
  return c;
}
function feedAgent(c: Call, ev: any, now: number) {
  const a = (c.agent ??= { content: "", reasoning: "" }), d = ev.choices?.[0]?.delta ?? {};
  if (ev.error) a.error = ev.error.message;
  if (ev.chrysalis_prompt) a.read = true;
  if (ev.usage) a.tokens = (a.tokens ?? 0) + (ev.usage.completion_tokens ?? 0);
  if (d.reasoning_content) a.reasoning += d.reasoning_content;
  if (d.content) a.content += d.content;
  for (const t of d.tool_calls ?? []) addCall(a, t);
  if (ev.tool && toolEvent(a.calls, ev.tool, now)?.state === "approval") openCards.add(c.id);
  if (ev.round) {
    a.rounds = [...(a.rounds ?? []), { content: a.content, ...(a.reasoning ? { reasoning: a.reasoning } : {}), calls: a.calls ?? [] }];
    a.content = a.reasoning = "";
    a.calls = undefined;
  }
  if (ev.note) a.notes = [...(a.notes ?? []), ev.note];
}

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
    if (!v) return;
    paint(v, run.m, run.live);
    if (voice?.reply?.m === run.m) paintSpoken(voice, performance.now(), true);
    if (run.conv === conv) { renderMeter(); shelfSoon(); }
    keepUp();
  });
}

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
setInterval(() => { for (const r of runs.values()) schedule(r); }, 250);

function finish(run: Run) {
  const { conv: c, m, usage, timing: t } = run;
  if (usage && t?.first && t.last) {
    m.stats = {
      ttft: (t.first - t.start) / 1000, tps: usage.completion_tokens / Math.max((t.gen || t.last - t.first) / 1000, 1e-3),
      tokens: usage.completion_tokens, prompt: usage.prompt_tokens, cached: usage.cached,
      thought: m.reasoning && t.answer && !m.rounds?.length ? (t.answer - t.first) / 1000 : undefined, seed: run.seed, model: current(),
      ctx: run.last ? run.last.prompt_tokens + run.last.completion_tokens : undefined,
    };
  }
  const cut = cuts.get(m);
  if (cut !== undefined) {
    cuts.delete(m);
    keepHeard(c, m, cut);
  } else if (voice?.reply?.m === m) endSaying(voice.reply);
  if (voice?.c === c) tellVoice(voice);
  const unrun = (m.calls ?? []).filter((x) => !x.state);
  if (unrun.length && !m.error && !m.stopped && !toolsOn(c))
    m.error = `The model asked to run ${[...new Set(unrun.map((x) => x.name))].join(", ")}, but this chat has no tools on, so the reply ends here.`;
  delete m.job;
  delete m.base;
  delete m.starting;
  cancelAnimationFrame(run.frame);
  if (runs.get(c.id) === run) runs.delete(c.id);
  if (run.gone) return;
  c.updated = Date.now();
  if (c === conv) render();
  else renderList();
  save(c);
  if (!c.named && !m.error) void name(c);
  redoEdits(c, m);
  if (talking && c === conv && !m.error && !m.stopped) setTimeout(talkOn, 400);
  else if (talking) { talking = 0; renderCastNext(); }
}

function redoEdits(c: Conversation, m: Message) {
  const at = c.messages.indexOf(m);
  if (m.error || m.stopped || voice?.c === c || at !== c.messages.length - 1 || c.messages[at - 1]?.fix || runs.has(c.id)) return;
  const all = artifactsOf(c), mine = new Set<object>([m, ...(m.rounds ?? [])]);
  const failed = [...all.values()].flat().filter((v) => mine.has(v.block.at as object) && v.applied?.some((a) => !a.ok));
  if (!failed.length) return;
  const note = failed.map((v) => {
    const latest = all.get(v.id)!.at(-1)!, misses = v.applied!.filter((a) => !a.ok);
    return `The edit to ${v.id} did not apply in full: ${misses.length} of its changes did not (${misses.map((a) => a.how).join("; ")}). ` +
      `${v.id} as it is now:\n\`\`\`${latest.kind} ${v.id}\n${latest.src}\n\`\`\``;
  }).join("\n\n") + "\n\nWrite those changes again, as an edit block against this. Do not repeat the ones that applied.";
  const names = [...new Set(failed.map((v) => v.id))].join(", ");
  c.messages.push({ role: "user", content: `Part of the edit to ${names} did not apply: asked again.`, note, fix: true });
  save(c);
  void generate(c);
}

function stopRun(run: Run, now = false) {
  if (talking) { talking = 0; renderCastNext(); }
  if (run.stopping && !now) return;
  run.stopping = true;
  if (run.m.job) void fetch(`/api/chat/${run.m.job}/stop`, { method: "POST" }).catch(() => {});
}

async function name(c: Conversation) {
  c.named = true;
  const first = c.messages.find((m) => m.role === "user");
  if (!first) {
    c.title = nameOf(folderOf(c));
    return save(c);
  }
  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...quick(), max_tokens: 24, messages: [
        { role: "system", content: TITLE_PROMPT },
        { role: "user", content: (first.content || (first.files ?? []).map((f) => f.name).join(", ")).slice(0, 2000) },
      ] }),
    });
    const choice = (await res.json()).choices?.[0], cut = !onChrysalis() && choice?.finish_reason === "length";
    const title: string | undefined = cut ? undefined : choice?.message?.content?.trim().split("\n")[0]?.replace(/^[\s#*"']+|[\s*"'.]+$/g, "");
    if (title) c.title = title.slice(0, 80);
  } catch {  }
  save(c);
  if (c === conv) $("title").textContent = document.title = c.title;
}

function submit(text: string, reply = true) {
  text = text.trim();
  if ((!text && !pending.length) || running() || drafting) return;
  input.value = "";
  grow();
  const files = pending;
  pending = [];
  renderPending();
  if (!conv.messages.some((m) => m.role === "user")) conv.title = (text || files.map((f) => f.name).join(", ")).split("\n")[0].slice(0, 60);
  const note = noteFor(conv, false);
  conv.messages.push({ role: "user", content: text, ...(files.length ? { files } : {}), ...(note ? { note } : {}), ...pointFields(false), ...takeEdits() });
  if (reply) void generate();
  else save(conv);
}

type VoicePhase = "starting" | "listening" | "thinking" | "speaking" | "muted" | "error";
interface VoiceReply {
  m: Message; sent: number; spoken: number; audible: boolean; ended?: boolean;
  sounding?: { from: number; to: number; at: number; secs: number };
}
interface VoiceMode {
  client: VoiceClient; c: Conversation; ready?: VoiceReady; muted: boolean; partial: string; error?: string;
  reply?: VoiceReply;
  merge?: Message;
  latency?: number;
  now: { voice?: string; speed?: number };
  asked?: VoiceAsk; pushed?: string;
  defaults: { tts?: string; voices: Record<string, string> };
  level: number; frame: number;
  check: { role: string; content: string }[];
  asking: Asking;
  ahead?: Ahead;
}
let voice: VoiceMode | null = null;
let voiceCatalog: VoiceReady["catalog"] = (() => { try { return JSON.parse(pref("voice-catalog", "{}")) ?? {}; } catch { return {}; } })();
const cuts = new Map<Message, number | null>();
const voicePrompt = () => settings.voice?.prompt ?? DEFAULT_VOICE_PROMPT;
const voiceNote = (c: Conversation) => fill([voicePrompt().trim(), toolsOn(c) ? VOICE_TOOLS : ""].filter(Boolean).join(" "));
function noteFor(c: Conversation, said: boolean): string | undefined {
  const users = c.messages.filter((m) => m.role === "user"), prev = users.at(-1);
  if (!said) return prev?.voice ? VOICE_OFF : undefined;
  const last = users.findLast((m) => m.note && m.voice)?.note, note = voiceNote(c);
  return note && (!prev?.voice || last !== note) ? note : undefined;
}
function tellVoice(v: VoiceMode) {
  const chat = v.c.messages.map((m) => m.content).join("\n");
  v.client.context(`${modelName(current())}\n${systemFor(v.c)}\n${chat.slice(-4000)}`);
}

async function openVoice() {
  if (voice) return;
  stopReading();
  const mine = () => voice?.client === client;
  const client: VoiceClient = new VoiceClient({
    ready: (r) => {
      if (!mine()) return;
      const v = voice!, asked = v.asked ?? {}, first = !v.ready;
      if (first && !asked.stack?.tts) v.defaults.tts = r.stack.tts;
      if ((first || asked.stack?.tts) && !asked.voice && r.stack.tts) v.defaults.voices[r.stack.tts] = r.voice;
      v.ready = r;
      v.now = { voice: r.voice, speed: r.speed };
      voiceCatalog = r.catalog ?? {};
      setPref("voice-catalog", JSON.stringify(voiceCatalog));
      pushVoice();
      renderVoice();
      syncSettings();
    },
    speech: (on) => { if (mine() && on && voice!.ahead && !voice!.ahead.said) dropAhead(voice!); },
    partial: (text, settled) => {
      if (!mine()) return;
      voice!.partial = text;
      if (settled) voice!.asking.settled = text;
      paintGhost();
      checkSaying(voice!);
    },
    pause: (text) => { if (mine()) pauseAhead(voice!, text); },
    turn: (text) => { if (mine()) void voiceTurn(voice!, text); },
    interrupt: () => { if (mine()) cutVoice(false); },
    spoken: (id, from, to, at, secs) => {
      const r = mine() ? voice!.reply : undefined;
      if (r?.m.id === id) Object.assign(r, { spoken: to, audible: true, sounding: { from, to, at, secs } });
    },
    audible: (id, ms) => {
      if (!mine() || ms === undefined) return;
      const m = [...walk(voice!.c.messages)].find((x) => x.id === id);
      voice!.latency = ms;
      renderVoice();
      if (!m) return;
      m.voiceMs = Math.round(ms);
      save(voice!.c);
      const v = views.get(m);
      if (v && ![...runs.values()].some((r) => r.m === m)) paint(v, m);
    },
    played: (id) => { if (mine() && voice!.reply?.m.id === id) voice!.reply = undefined; },
    error: (message) => {
      if (!mine()) return;
      if (voice!.ready) return toast(`Voice: ${message}`);
      voice!.error = message;
      renderVoice();
    },
    closed: () => { if (mine()) { voice!.error ??= "The voice service stopped"; renderVoice(); } },
  });
  voice = { client, c: conv, muted: false, partial: "", level: 0, frame: 0, check: [], asking: asking(), now: {}, defaults: { voices: {} }, asked: voiceWant(null) };
  tellVoice(voice);
  renderVoice();
  if (status.state === "offline" && onService()) void start().then(() => refreshStatus());
  if (!navigator.mediaDevices?.getUserMedia) {
    voice.error = "This browser gives no page a microphone";
    return renderVoice();
  }
  void warmVoice(voice);
  try {
    await client.open(voice.asked!);
  } catch (e) {
    if (!mine()) return;
    voice!.error = e instanceof DOMException && e.name === "NotAllowedError" ? "The microphone is blocked for this page" : `No microphone: ${e instanceof Error ? e.message : e}`;
    renderVoice();
  }
}

function closeVoice() {
  const v = voice;
  if (!v) return;
  dropAhead(v);
  voice = null;
  cancelAnimationFrame(v.frame);
  (globalThis as any).CSS?.highlights?.delete("voice-unspoken");
  v.client.close();
  setVoicePop(false);
  paintGhost();
  renderVoice();
  input.focus();
}

interface Heard { text: string; fixes: Fix[]; tool: boolean; asked?: { role: string; content: string }[] }
interface Asking { checks: Map<string, Promise<Heard>>; done: Map<string, Heard>; busy?: string; settled?: string }
const asking = (): Asking => ({ checks: new Map(), done: new Map() });
interface Ahead {
  text: string;
  q: Message;
  sent: string; tool: boolean;
  said?: Message;
  check?: Heard;
  run: Run; reply: VoiceReply; ended: boolean; live: boolean;
}
const unsettled = new WeakSet<Message>();
const heardAs = new WeakMap<Message, string>();

function checkOf(v: VoiceMode, text: string): Promise<Heard> {
  const mine = v.asking, asked = mine.checks.get(text);
  if (asked) return asked;
  const c = v.c, system = fill(CHECK.replace("{{TOOLS}}", toolsOn(c) ? ", with web search and page reading" : "")) + (toolsOn(c) ? CHECK_TOOLS : "");
  if (v.check[0]?.content !== system || v.check.length > 121) v.check = [{ role: "system", content: system }];
  const at = c.messages.findLastIndex((m) => m.role === "assistant"), last = c.messages[at]?.content.slice(0, 300) ?? "";
  const before = at > 0 ? c.messages.slice(0, at).findLast((m) => m.role === "user")?.content.slice(0, 200) ?? "" : "";
  const turn = { role: "user", content: (before ? `The user said before: ${before}\n` : "") + (last ? `The assistant said: ${last}\n` : "") + `Transcript: ${text}` };
  const p = (async (): Promise<Heard> => {
    try {
      const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        ...quick(), messages: [...v.check, turn], temperature: 0, presence_penalty: 0, max_tokens: 48 }) });
      const answer: string = (await res.json())?.choices?.[0]?.message?.content ?? "";
      if (!res.ok || !answer) return { text, fixes: [], tool: false };
      return { text, ...readHeard(answer), asked: [turn, { role: "assistant", content: answer }] };
    } catch {
      return { text, fixes: [], tool: false };
    }
  })();
  mine.checks.set(text, p);
  void p.then((h) => mine.done.set(text, h));
  return p;
}
function knownFor(v: VoiceMode, text: string): Heard | undefined {
  let best: Heard | undefined;
  for (const h of v.asking.done.values())
    if ((text === h.text || text.startsWith(`${h.text} `)) && (!best || h.text.length > best.text.length)) best = h;
  return best;
}
function checkSaying(v: VoiceMode) {
  const mine = v.asking, text = mine.settled && questionFor(v, mine.settled).text;
  if (mine.busy || !text || mine.checks.has(text)) return;
  mine.busy = text;
  void checkOf(v, text).then(() => {
    mine.busy = undefined;
    if (voice !== v || v.asking !== mine) return;
    paintGhost();
    checkSaying(v);
  });
}

function questionFor(v: VoiceMode, text: string): { text: string; on?: Message } {
  const last = v.c.messages.at(-1), on = last && (last === v.merge || last === v.ahead?.said) ? last : undefined;
  return on ? { text: `${heardAs.get(on) ?? on.content} ${text}`, on } : { text };
}

function startAhead(v: VoiceMode, text: string, on?: Message): Ahead {
  const c = v.c, known = knownFor(v, text), fixed = applyFixes(text, known?.fixes ?? []), tool = known?.tool ?? false;
  const effort = v.c.effort, mode: Mode = effort === "off" ? "fast" : "think";
  const note = on ? undefined : noteFor(c, true);
  const q: Message = on ?? { role: "user", content: fixed.text, voice: true, ...(note ? { note } : {}), ...pointFields(true), ...takeEdits() };
  const sent = { ...q, content: fixed.text }, upto = on ? c.messages.map((x) => (x === on ? sent : x)) : [...c.messages, sent];
  const m: Message = { id: randomId(), role: "assistant", content: "", reasoning: "", effort, params: changed(mode) || undefined };
  const run: Run = { conv: c, m, live: { phase: "waiting", since: performance.now(), note: "Reading the prompt" }, at: 0, stopping: false, frame: 0, saved: 0 };
  const a: Ahead = { text, q, sent: fixed.text, tool, run, reply: { m, sent: 0, spoken: 0, audible: false }, ended: false, live: false };
  if (!on) heardAs.set(q, text);
  v.client.hold(m.id!);
  void (async () => {
    try {
      await filesReady(upto);
      const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request(c, upto, { effort, route: true, extra: { chrysalis_voice: { tool, ahead: true } } })) });
      if (!res.ok || !res.body) throw new Error((await res.json().catch(() => null))?.error?.message ?? `HTTP ${res.status}`);
      m.job = res.headers.get("X-Chrysalis-Job") ?? undefined;
      if (run.stopping) stopRun(run, true);
      else if (a.live) void fetch(`/api/chat/${m.job}/go`, { method: "POST" }).catch(() => {});
      await stream(run, res.body);
    } catch (e) {
      if (run.stopping) m.stopped = true;
      else m.error = e instanceof Error ? e.message : String(e);
    }
    a.ended = true;
    if (a.live) finish(run);
    else if (voice?.ahead === a) endSaying(a.reply);
  })();
  void checkOf(v, text).then((h) => {
    a.check = h;
    standAhead(v, a);
  });
  return a;
}

function dropAhead(v: VoiceMode) {
  const a = v.ahead;
  if (!a) return;
  v.ahead = undefined;
  stopRun(a.run, true);
  v.client.cancel(a.run.m.id!);
}

function pauseAhead(v: VoiceMode, text: string) {
  if (v.ahead?.said || running(v.c) || status.state !== "online" || status.loading) return;
  const q = questionFor(v, text.trim());
  if (v.ahead?.text === q.text) return;
  dropAhead(v);
  v.ahead = startAhead(v, q.text, q.on);
}

function standAhead(v: VoiceMode, a: Ahead) {
  if (voice !== v || v.ahead !== a || !a.said || !a.check) return;
  const c = v.c, said = a.said, fixed = applyFixes(a.text, a.check.fixes), m = a.run.m;
  if (fixed.text !== a.sent || a.check.tool !== a.tool) {
    dropAhead(v);
    said.content = fixed.text;
    const again = (v.ahead = startAhead(v, a.text, said));
    again.said = said;
    return;
  }
  v.ahead = undefined;
  a.live = true;
  const kept = (said.heard ?? []).filter((f) => fixed.text.includes(f.said)), heard = [...kept, ...fixed.made];
  said.heard = heard.length ? heard : undefined;
  unsettled.delete(said);
  heardAs.delete(said);
  if (a.check.asked) v.check.push(...a.check.asked);
  v.asking = asking();
  branch(c, c.messages.length, m);
  runs.set(c.id, a.run);
  v.reply = a.reply;
  if (m.job) void fetch(`/api/chat/${m.job}/go`, { method: "POST" }).catch(() => {});
  v.client.go(m.id!);
  if (c === conv) {
    follow = true;
    render();
  }
  save(c);
  renderVoice();
  tellVoice(v);
  if (a.ended) finish(a.run);
}

async function voiceTurn(v: VoiceMode, text: string) {
  v.partial = "";
  v.asking.settled = undefined;
  paintGhost();
  text = text.trim();
  if (!text) return;
  const c = v.c;
  for (let i = 0; running(c) && i < 150; i++) {
    if (i === 50) stopRun(running(c)!);
    await sleep(20);
  }
  if (voice !== v || running(c)) return;
  const q = questionFor(v, text);
  v.merge = undefined;
  let a = v.ahead;
  if (a && (a.said || a.text !== q.text)) {
    dropAhead(v);
    a = undefined;
  }
  if (status.state !== "online" || status.loading) return answerLater(v, q.text, q.on);
  a ??= v.ahead = startAhead(v, q.text, q.on);
  if (!q.on) {
    if (!c.messages.length) c.title = q.text.slice(0, 60);
    c.messages.push(a.q);
  } else heardAs.set(q.on, q.text);
  a.said = a.q;
  a.q.content = a.sent;
  if (!a.check) {
    unsettled.add(a.q);
    if (c === conv) render();
  }
  standAhead(v, a);
}

async function answerLater(v: VoiceMode, text: string, on?: Message) {
  const c = v.c, note = on ? undefined : noteFor(c, true);
  const said: Message = on ?? { role: "user", content: text, voice: true, ...(note ? { note } : {}), ...pointFields(true), ...takeEdits() };
  if (!on) {
    if (!c.messages.length) c.title = text.slice(0, 60);
    c.messages.push(said);
  }
  heardAs.set(said, text);
  unsettled.add(said);
  if (c === conv) render();
  const mine = v.asking, heard = await checkOf(v, text);
  if (v.asking !== mine || c.messages.at(-1) !== said) return;
  unsettled.delete(said);
  heardAs.delete(said);
  const fixed = applyFixes(text, heard.fixes);
  said.content = fixed.text;
  said.heard = fixed.made.length ? fixed.made : undefined;
  if (heard.asked) v.check.push(...heard.asked);
  v.asking = asking();
  if (voice !== v || running(c)) {
    persist(c);
    if (c === conv) render();
    return;
  }
  const effort = v.c.effort, mode: Mode = effort === "off" ? "fast" : "think";
  const m: Message = { id: randomId(), role: "assistant", content: "", reasoning: "", effort, params: changed(mode) || undefined };
  const upto = [...c.messages];
  branch(c, c.messages.length, m);
  v.reply = { m, sent: 0, spoken: 0, audible: false };
  await startRun(c, m, () => request(c, upto, { effort, route: true, extra: { chrysalis_voice: { tool: heard.tool } } }));
}

async function warmVoice(v: VoiceMode) {
  const effort = v.c.effort;
  await ensureOnline({ stopping: false }).catch(() => {});
  if (voice !== v || running(v.c)) return;
  const upto: Message[] = v.c.messages.length ? [...v.c.messages] : [{ role: "user", content: "Hi" }];
  const body = { ...request(v.c, upto, { effort }), max_tokens: 1, logprobs: false, top_logprobs: undefined };
  const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).catch(() => null);
  await res?.body?.cancel().catch(() => {});
}

const sayable = (m: Message) => [...(m.rounds ?? []).map((r) => r.content), m.content].filter(Boolean).join("\n\n");
function feedVoice(m: Message) {
  const r = voice?.reply?.m === m ? voice.reply : voice?.ahead?.reply.m === m ? voice.ahead.reply : undefined;
  if (!r || cuts.has(m)) return;
  const text = sayable(m);
  if (text.length > r.sent) {
    voice!.client.say(m.id!, text.slice(r.sent));
    r.sent = text.length;
  }
}

function endSaying(r: VoiceReply) {
  if (r.ended) return;
  feedVoice(r.m);
  r.ended = true;
  voice!.client.done(r.m.id!);
}

function cutVoice(here: boolean) {
  const v = voice;
  if (!v) return;
  if (here) v.client.stop();
  const a = v.ahead;
  dropAhead(v);
  if (a?.said) v.merge = a.said;
  const r = v.reply;
  v.reply = undefined;
  renderVoice();
  if (!r) return;
  const worked = !!(r.m.rounds?.length || r.m.calls?.length);
  const keep = r.audible || worked ? heardTo(r) : null, run = running(v.c);
  if (keep === null) v.merge = v.c.messages[v.c.messages.indexOf(r.m) - 1];
  if (run?.m === r.m) {
    cuts.set(r.m, keep);
    return stopRun(run);
  }
  keepHeard(v.c, r.m, keep);
  save(v.c);
  if (v.c === conv) render();
}
function heardTo(r: VoiceReply, now = performance.now()): number {
  const s = r.sounding;
  if (!s) return r.spoken;
  const text = sayable(r.m), f = Math.min(1, Math.max(0, (now - s.at) / (s.secs * 1000)));
  const at = unitOf(text, s.from + Math.round(f * (s.to - s.from))), word = shownWords(text).findLast((w) => w.start < at);
  return Math.min(s.to, Math.max(s.from, word ? pointOf(text, word.end) : 0));
}
const UNREAD_ELS = ".code, table, .katex, figure, svg, button";
let spokenAt = 0;
function paintSpoken(v: VoiceMode, now: number, painted = false) {
  const reg = (globalThis as any).CSS?.highlights as Map<string, unknown> | undefined, r = v.reply, view = r && views.get(r.m);
  if (!reg) return;
  if (!r || !view || cuts.has(r.m)) return void reg.delete("voice-unspoken");
  if (!painted && now - spokenAt < 80) return;
  spokenAt = now;
  const text = sayable(r.m), at = unitOf(text, r.sounding ? heardTo(r, now) : 0);
  const n = shownWords(text).filter((w) => w.start < at).length;
  const boxes = [...view.rounds.querySelectorAll<HTMLElement>(":scope > .round > .markdown"), view.body], from = afterWords(boxes, n);
  if (!from) return void reg.delete("voice-unspoken");
  const range = new Range();
  range.setStart(from.node, from.offset);
  range.setEnd(view.body, view.body.childNodes.length);
  reg.set("voice-unspoken", new (globalThis as any).Highlight(range));
}
function afterWords(boxes: HTMLElement[], n: number): { node: Text; offset: number } | null {
  let count = 0, inWord = false, wordy = false, block: Element | null = null, end: { node: Text; offset: number } | null = null;
  const ended = () => {
    inWord = false;
    return wordy && ++count === n ? end : null;
  };
  for (const box of boxes) {
    const walk = document.createTreeWalker(box, NodeFilter.SHOW_TEXT,
      { acceptNode: (t) => (t.parentElement?.closest(UNREAD_ELS) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT) });
    for (let t = walk.nextNode() as Text | null; t; t = walk.nextNode() as Text | null) {
      if (n === 0) return { node: t, offset: 0 };
      const b = t.parentElement?.closest("p, li, h1, h2, h3, h4, h5, h6, blockquote, td, th, div") ?? null;
      if (b !== block && inWord && ended()) return end;
      block = b;
      for (let i = 0; i < t.data.length; i++) {
        if (/\s/.test(t.data[i])) {
          if (inWord) {
            end = { node: t, offset: i };
            if (ended()) return end;
          }
        } else {
          if (!inWord) [inWord, wordy] = [true, false];
          wordy ||= /[\p{L}\p{N}]/u.test(t.data[i]);
        }
      }
      if (inWord) end = { node: t, offset: t.data.length };
    }
  }
  return inWord && ended() ? end : null;
}
function keepHeard(c: Conversation, m: Message, to: number | null) {
  if (to === null) {
    const at = c.messages.indexOf(m);
    if (at >= 0 && at === c.messages.length - 1) c.messages.splice(at, 1);
    return;
  }
  const before = sayable({ ...m, content: "" }), skip = before ? [...before].length + 2 : 0;
  m.content = [...m.content].slice(0, Math.max(0, to - skip)).join("").trimEnd();
  m.interrupted = true;
  if (m.id) tokens.delete(m.id);
}

function paintGhost() {
  let g = thread.querySelector<HTMLElement>(".msg.ghost");
  const v = voice, text = v && v.c === conv ? v.partial : "";
  if (!v || !text) return void g?.remove();
  if (!g) {
    g = h("article", "msg user ghost");
    g.append(h("div", "bubble"));
    thread.querySelector(".empty")?.remove();
    thread.append(g);
  }
  const b = g.firstElementChild as HTMLElement, known = knownFor(v, text);
  if (known) {
    const fixed = applyFixes(known.text, known.fixes), checked = h("span", "checked");
    for (const p of fixedParts(fixed.text, fixed.made)) checked.append(typeof p === "string" ? p : fixedSpan(p));
    b.replaceChildren(checked, text.slice(known.text.length));
  } else b.textContent = text;
  keepUp();
}

function voicePhase(v: VoiceMode): VoicePhase {
  if (v.error) return "error";
  if (!v.client.hearing) return "starting";
  if (v.client.playing) return "speaking";
  if (v.reply) return "thinking";
  return v.muted ? "muted" : "listening";
}
function voiceStatus(v: VoiceMode, phase: VoicePhase) {
  if (phase === "error") return v.error!;
  if (phase === "thinking" && running(v.c)?.live.loading) return `Loading ${modelName(current())}…`;
  return { starting: "Starting voice…", listening: "Listening", thinking: "Thinking", speaking: "Speaking", muted: "Muted" }[phase];
}

function renderVoice() {
  const v = voice, pill = $("voice-pill"), pick = $("voice-pick");
  $("composer").classList.toggle("voicing", !!v);
  pill.hidden = pick.hidden = !v;
  $("voice").hidden = !!v;
  renderComposer();
  if (!v) return void delete $("composer").dataset.phase;
  const mute = $("voice-mute");
  mute.innerHTML = icon(v.muted ? "micOff" : "mic");
  mute.title = v.muted ? "Unmute" : "Mute";
  mute.setAttribute("aria-label", mute.title);
  mute.setAttribute("aria-pressed", String(v.muted));
  mute.classList.toggle("on", v.muted);
  const cur = currentVoice(), name = voiceInfo(cur.tts, cur.id)?.label ?? cur.id ?? "Voice", model = modelLabel("tts", cur.tts);
  pick.innerHTML = `<span class="voice-pick-name">${escapeHtml(name)}</span>${model ? `<span class="voice-pick-model">${escapeHtml(model)}</span>` : ""}${icon("chevron")}`;
  pick.title = `Voice: ${name}${model ? ` (${model})` : ""}`;
  $("voice-meta").textContent = v.ready ? (["stt", "turn", "tts"] as Part[]).map((k) => modelLabel(k, v.ready!.stack[k])).filter(Boolean).join(" · ") : "";
  $("voice-meta").title = v.ready ? Object.entries(v.ready.stack).map(([k, n]) => `${k}: ${n}`).join("\n") : "";
  paintVoice();
  v.frame ||= requestAnimationFrame(animateVoice);
}
function paintVoice() {
  const v = voice!, phase = voicePhase(v), text = voiceStatus(v, phase), pill = $("voice-pill"), dots = $("voice-dots");
  pill.dataset.phase = $("composer").dataset.phase = phase;
  if ($("voice-status").textContent !== text) $("voice-status").textContent = text;
  const latency = v.latency !== undefined ? `\nLast reply: ${(v.latency / 1000).toFixed(2)} s from the end of your speech to its first sound` : "";
  pill.title = text + latency;
  dots.title = phase === "speaking" || phase === "thinking" ? "Talk over the reply (Esc)" : "";
  dots.setAttribute("aria-label", phase === "speaking" || phase === "thinking" ? `${text}: talk over the reply` : text);
}

const DOT_WEIGHT = [0.45, 0.8, 1, 0.8, 0.45];
function animateVoice(now: number) {
  const v = voice;
  if (!v) return;
  v.frame = requestAnimationFrame(animateVoice);
  paintVoice();
  paintSpoken(v, now);
  const phase = voicePhase(v), still = reducedMotion.matches || phase === "muted" || phase === "error", t = now / 1000;
  const lv = v.client.levels(), target = still ? 0 : Math.min(1, (phase === "speaking" ? lv.out : phase === "listening" ? lv.mic : 0) * 6);
  v.level += (target - v.level) * (target > v.level ? 0.5 : 0.12);
  const dots = $("voice-dots").children as HTMLCollectionOf<HTMLElement>;
  for (let i = 0; i < dots.length; i++) {
    const wobble = 0.75 + 0.25 * Math.sin(t * 9 + i * 1.7), w = DOT_WEIGHT[i] ?? 1;
    const h = still ? 0 : phase === "starting" ? 0.12 * (0.5 + 0.5 * Math.sin(t * 3 - i * 0.6)) : v.level * w * wobble;
    const o = still || phase !== "thinking" ? 1 : 0.35 + 0.65 * Math.max(0, Math.sin(t * 3.2 - i * 0.9));
    dots[i].style.setProperty("--h", h.toFixed(3));
    dots[i].style.setProperty("--o", o.toFixed(3));
  }
}

function setVoice(change: Partial<VoiceSettings>) {
  const next: VoiceSettings = { ...settings.voice, ...change };
  for (const k of Object.keys(next) as (keyof VoiceSettings)[]) if (next[k] === undefined) delete next[k];
  if (Object.keys(next).length) settings.voice = next;
  else delete settings.voice;
}
function voiceWant(v: VoiceMode | null): VoiceAsk {
  const { members, group } = v ? castOf(v.c) : { members: [] as Person[], group: undefined };
  const own = !group && members.length === 1 ? voiceOf(members[0].card.data) : undefined;
  const s = { ...(settings.voice ?? {}), ...(own ?? {}) }, d = v?.defaults, tts = s.tts ?? d?.tts, name = s.name ?? (tts ? d?.voices[tts] : undefined);
  const stack: VoiceAsk["stack"] = tts ? { tts } : {};
  return { ...(Object.keys(stack).length ? { stack } : {}), ...(name ? { voice: name } : {}), speed: s.speed ?? 1 };
}
function pushVoice() {
  const v = voice, r = v?.ready;
  if (!v || !r) return;
  const want = voiceWant(v), ask: VoiceAsk = {};
  const stack = Object.fromEntries(Object.entries(want.stack ?? {}).filter(([k, n]) => r.stack[k as Part] !== n));
  if (Object.keys(stack).length) ask.stack = stack;
  if (want.voice && want.voice !== v.now.voice) ask.voice = want.voice;
  if (want.speed !== v.now.speed) ask.speed = want.speed;
  const key = JSON.stringify(ask);
  if (!Object.keys(ask).length || key === v.pushed) return;
  v.pushed = key;
  v.asked = ask;
  Object.assign(v.now, ask.voice ? { voice: ask.voice } : {}, ask.speed ? { speed: ask.speed } : {});
  v.client.config(ask);
}
function currentVoice() {
  const s = settings.voice, r = voice?.ready, tts = s?.tts ?? r?.stack.tts;
  return { tts, id: s?.name ?? (r && tts === r.stack.tts ? voice!.now.voice : undefined) };
}
const voiceInfo = (tts?: string, id?: string) => voiceCatalog.tts?.find((m) => m.name === tts)?.voices?.find((x) => x.id === id);
const LANGS = new Intl.DisplayNames(["en"], { type: "language" });
const langName = (code: string) => { try { return LANGS.of(code) ?? code; } catch { return code; } };
const voiceGroup = (x: VoiceInfo) => [langName(x.lang), x.gender ? x.gender[0].toUpperCase() + x.gender.slice(1) : ""].filter(Boolean).join(" · ");
const modelLabel = (part: Part, name?: string) => (name && voiceCatalog[part]?.find((m) => m.name === name)?.label) || name;

function pickVoice(tts: string, x: VoiceInfo) {
  const before = { tts: settings.voice?.tts, name: settings.voice?.name };
  setVoice({ tts, name: x.id });
  saveSettings();
  setVoicePop(false, true);
  toast(`Voice: ${x.label}`, () => { setVoice(before); saveSettings(); });
}

let voicePop = false;
function setVoicePop(open: boolean, refocus = false) {
  voicePop = open && !!voice;
  $("voice-pop").hidden = !voicePop;
  $("voice-pick").classList.toggle("open", voicePop);
  $("voice-pick").setAttribute("aria-expanded", String(voicePop));
  delete $("voice-pop").dataset.key;
  renderVoicePop();
  if (voicePop) $("voice-pop").querySelector<HTMLElement>(".current .voice-item, .voice-item")?.focus();
  else if (refocus && voice) $("voice-pick").focus();
}
function renderVoicePop() {
  const pop = $("voice-pop"), cur = currentVoice(), key = JSON.stringify([voiceCatalog.tts, cur]);
  if (!voicePop || pop.dataset.key === key) return;
  pop.dataset.key = key;
  const rows: HTMLElement[] = [];
  for (const m of voiceCatalog.tts ?? []) {
    const head = h("div", m.installed ? "voice-model" : "voice-model missing");
    head.append(h("span", "", m.label));
    if (!m.installed) {
      const get = h("button", "btn small", "Download");
      get.type = "button";
      get.onclick = () => {
        get.disabled = true;
        get.textContent = "Downloading…";
        samples().get(m.name);
        void samples().catalog().then((c) => {
          if (c?.tts?.length) { voiceCatalog = c; setPref("voice-catalog", JSON.stringify(c)); }
          renderVoicePop();
        });
      };
      head.append(get);
    }
    rows.push(head);
    if (!m.installed) continue;
    const groups = new Map<string, VoiceInfo[]>();
    for (const x of m.voices ?? []) groups.set(voiceGroup(x), [...(groups.get(voiceGroup(x)) ?? []), x]);
    for (const [g, list] of groups) {
      rows.push(h("div", "voice-group", g));
      for (const x of list) {
        const on = m.name === cur.tts && x.id === cur.id, row = h("div", on ? "voice-row current" : "voice-row");
        const item = h("button", "model-item voice-item"), play = h("button", "act voice-play"), mark = h("span", "model-check");
        item.type = play.type = "button";
        item.setAttribute("role", "option");
        item.setAttribute("aria-selected", String(on));
        item.dataset.voice = `${m.name}/${x.id}`;
        if (on) mark.innerHTML = icon("check");
        item.append(h("span", "model-label", x.label), mark);
        item.onclick = () => pickVoice(m.name, x);
        play.innerHTML = icon("play");
        play.title = `Hear ${x.label}`;
        play.setAttribute("aria-label", play.title);
        play.onclick = () => voice?.client.preview(x.id);
        row.append(play, item);
        rows.push(row);
      }
    }
  }
  if (!rows.length) rows.push(h("div", "voice-group", "The voice service has not listed its voices yet"));
  $("voice-list").replaceChildren(...rows);
}

let pending: Attachment[] = [];
const bodies = new Map<string, string>();
const thumbs = new Map<string, Promise<string>>();

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
  for (const f of await welcome(list, true)) {
    const r = await attach(f, onChrysalis() ? undefined : status.backend?.name);
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

const editsNote = (m: Message) => !m.edits?.length ? undefined : `The user changed ${m.edits.map((e) => e.id).join(", ")} by hand since your ` +
  `last reply. As ${m.edits.length > 1 ? "they are" : "it is"} now:\n\n` + m.edits.map((e) => `\`\`\`${e.kind} ${e.id}\n${e.src}\n\`\`\``).join("\n\n");
function userContent(m: Message): string | object[] {
  const note = [m.note, editsNote(m), m.point].filter(Boolean).join("\n\n") || undefined, said = noted(note, m.content);
  if (!m.files?.length) return said;
  const parts: object[] = m.files.map((a) => {
    const body = bodies.get(a.id) ?? "";
    if (a.kind === "image") return { type: "image_url", image_url: { url: body } };
    if ((a.kind === "pdf" || a.kind === "video") && !onChrysalis())
      return { type: "text", text: `[${a.name}: a ${a.kind === "pdf" ? "PDF" : "video"}, which ${status.backend?.name} is not sent]\n\n` };
    if (a.kind === "pdf") return { type: "file", file: { file_data: body, filename: a.name } };
    if (a.kind === "video") return { type: "video_url", video_url: { url: body } };
    return { type: "text", text: `${textBlock(a.name, body)}\n\n` };
  });
  if (note) parts.unshift({ type: "text", text: noted(note, "") });
  if (m.content) parts.push({ type: "text", text: m.content });
  return parts;
}

interface ChatFile { chrysalis: "chats"; version: 1; exported: string; conversations: Conversation[]; files?: Record<string, string> }

function download(name: string, data: string, type: string) {
  const a = h("a");
  a.href = URL.createObjectURL(new Blob([data], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function exportChats(list: Conversation[], name: string) {
  const files: Record<string, string> = {};
  for (const a of list.flatMap((c) => [...walk(c.messages)].flatMap((m) => m.files ?? []))) {
    const data = await getFile(a.id);
    if (data !== undefined) files[a.id] = typeof data === "string" ? data : await dataUrl(data);
  }
  const bare = (ms: Message[]): Message[] => ms.map(({ job, base, versions, ...m }) =>
    (versions ? { ...m, versions: { shown: versions.shown, others: versions.others.map(bare) } } : m));
  const file: ChatFile = { chrysalis: "chats", version: 1, exported: new Date().toISOString(),
    conversations: list.map((c) => ({ ...structuredClone(c), messages: bare(structuredClone(c.messages)) })), files };
  download(name, JSON.stringify(file, null, 1), "application/json");
}

const blobOf = (url: string) => {
  const [head, b64] = url.split(",", 2), bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));
  return new Blob([bytes], { type: head.slice(5).split(";")[0] });
};
const fileName = (title: string) => `${title.replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "chat"}.chrysalis.json`;

const isCall = (c: any) => typeof c?.id === "string" && typeof c.name === "string" && typeof c.args === "string" && (c.result === undefined || typeof c.result === "string");
const isRound = (r: any) => typeof r?.content === "string" && (r.reasoning === undefined || typeof r.reasoning === "string") && Array.isArray(r.calls) && r.calls.every(isCall);
function importedMessages(list: unknown, depth = 0): Message[] | null {
  if (!Array.isArray(list) || !list.length || depth > 200) return null;
  const messages: Message[] = [];
  for (const m of list) {
    if (!m || (m.role !== "user" && m.role !== "assistant") || typeof m.content !== "string") return null;
    const { job, base, versions, ...rest } = m;
    if (rest.files !== undefined && !(Array.isArray(rest.files) && rest.files.every((a: any) => typeof a?.id === "string"
      && typeof a.name === "string" && KINDS.includes(a.kind)))) return null;
    if (rest.rounds !== undefined && !(Array.isArray(rest.rounds) && rest.rounds.every(isRound))) return null;
    if (rest.calls !== undefined && !(Array.isArray(rest.calls) && rest.calls.every(isCall))) return null;
    if (typeof rest.voiceMs !== "number") delete rest.voiceMs;
    if (rest.note !== undefined && typeof rest.note !== "string") return null;
    if (rest.heard !== undefined && !(Array.isArray(rest.heard) && rest.heard.every((f: any) => typeof f?.heard === "string" && typeof f.said === "string"))) return null;
    if (versions !== undefined) {
      const others = Array.isArray(versions?.others) ? versions.others.map((v: unknown) => importedMessages(v, depth + 1)) : null;
      if (!others?.length || others.some((v: Message[] | null) => !v) || !Number.isInteger(versions.shown) || versions.shown < 0 || versions.shown > others.length) return null;
      rest.versions = { shown: versions.shown, others };
    }
    messages.push(rest);
  }
  return messages;
}
function importedChat(x: any): Conversation | null {
  if (!x || typeof x !== "object") return null;
  const messages = importedMessages(x.messages);
  if (!messages) return null;
  const tools = x.tools && typeof x.tools === "object" ? Object.fromEntries(Object.entries(x.tools).filter(([, v]: [string, any]) =>
    v && typeof v === "object" && (v.off === undefined || (Array.isArray(v.off) && v.off.every((n: unknown) => typeof n === "string"))))) as ToolChoice["tools"] : undefined;
  return { id: randomId(), title: typeof x.title === "string" && x.title ? x.title.slice(0, 80) : "Imported chat", named: true,
    updated: Date.now(), effort: EFFORTS.includes(x.effort) ? x.effort : "xhigh", messages, ...(tools && Object.keys(tools).length ? { tools } : {}) };
}

async function importFiles(files: FileList) {
  let added: Conversation[] = [], bad = 0;
  for (const f of files) {
    try {
      const data = JSON.parse(await f.text());
      for (const [id, body] of Object.entries(data?.files ?? {}))
        if (typeof body === "string") await putFile(id, body.startsWith("data:image/") ? blobOf(body) : body);
      const list = (data?.chrysalis === "chats" && Array.isArray(data.conversations) ? data.conversations : [data]).map(importedChat);
      bad += list.filter((c: Conversation | null) => !c).length;
      added = added.concat(list.filter(Boolean));
    } catch { bad++; }
  }
  const theirs = added.filter(castless).length;
  added = added.filter((c) => !castless(c));
  for (const c of added) save(c);
  if (added.length) openConversation(added.at(-1)!.id);
  const why = [bad ? `${bad} not a Chrysalis chat` : "", theirs ? `${theirs} with a character not on this device: add their card first` : ""].filter(Boolean).join("; ");
  toast(added.length ? `Imported ${added.length} chat${added.length > 1 ? "s" : ""}${why ? ` (${why})` : ""}`
    : theirs && !bad ? `Nothing imported: ${why}` : "Nothing to import: not a chat exported from Chrysalis");
}

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
    sampling: "fast", final: begin.trim() ? { role: "assistant", content: begin } : undefined, loop: false,
    extra: { chrysalis_aside: true, max_tokens: 400, logprobs: false, seed: undefined, ...(begin.trim() ? { continue_final_message: true } : {}) },
  });
  (body.messages as object[]).splice(begin.trim() ? -1 : body.messages.length, 0, { role: "user", content: DRAFT_PROMPT });
  input.readOnly = true;
  renderComposer();
  try {
    const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok || !res.body) throw new Error((await res.json().catch(() => null))?.error?.message ?? `HTTP ${res.status}`);
    d.job = res.headers.get("X-Chrysalis-Job") ?? undefined;
    if (d.stop && d.job) void fetch(`/api/chat/${d.job}/stop`, { method: "POST" }).catch(() => {});
    let text = begin;
    for await (const ev of events(res.body)) {
      if (ev === DONE) break;
      if (ev.error) throw new Error(ev.error.message);
      const delta = ev.choices?.[0]?.delta ?? {};
      if (delta.chrysalis_prefix) text = delta.chrysalis_prefix.content ?? text;
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

function openConversation(id: string, reveal = true) {
  closeVoice();
  stopReading();
  conv = runs.get(id)?.conv ?? (chats[id] ? structuredClone(chats[id]) : blank());
  if (reveal) unfold(whoOf(conv));
  closeOver(panes.conversations);
  render();
  scroller.scrollTop = scroller.scrollHeight;
  follow = true;
  keepUp();
}

function newChat(who = "") {
  closeVoice();
  stopReading();
  conv = blank();
  unfold(who || CHRYSALIS);
  if (who && who !== CHRYSALIS) conv.who = who;
  greet(conv);
  const mine = persona().card.data.first_mes?.trim();
  if (mine && !input.value.trim()) {
    input.value = fillCard(mine, writerOf(conv)?.card.data.name ?? castOf(conv).members[0]?.card.data.name ?? "", userName());
    grow();
  }
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
  if (id === conv.id) newChat(folderOf(conv));
  else renderList();
}

function copy(text: string, button: HTMLElement) {
  const area = h("textarea");
  area.value = text;
  area.readOnly = true;
  area.style.cssText = "position: fixed; top: 0; opacity: 0; font-size: 16px";
  document.body.append(area);
  area.select();
  area.setSelectionRange(0, text.length);
  document.execCommand("copy");
  area.remove();
  const label = button.textContent;
  button.classList.add("done");
  if (button.classList.contains("copy")) button.textContent = "Copied";
  setTimeout(() => { button.classList.remove("done"); if (button.classList.contains("copy")) button.textContent = label; }, 1200);
}

$("svg-zoom").addEventListener("click", (e) => (e.currentTarget as HTMLDialogElement).close());
$("page-zoom").addEventListener("click", (e) => { if (e.target === e.currentTarget) (e.currentTarget as HTMLDialogElement).close(); });
$("page-zoom").addEventListener("close", () => $("page-zoom").querySelector(".page-zoom-box")!.replaceChildren());

function grow() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 240)}px`;
}

function figureClick(t: HTMLElement): boolean {
  const code = t.closest<HTMLElement>(".copy");
  if (code) return void copy(code.closest(".code")?.querySelector("code")?.textContent ?? "", code), true;
  const art = t.closest<HTMLElement>("[data-art]:is(button)");
  if (art) return artAction(art), true;
  const svg = t.closest<HTMLElement>("[data-svg]");
  if (svg) return svgAction(svg), true;
  const page = t.closest<HTMLElement>("[data-page]");
  if (page) return pageAction(page), true;
  const doc = t.closest<HTMLElement>("[data-doc]:is(button)");
  if (doc) return docAction(doc), true;
  return false;
}
$("shelf-stage").addEventListener("click", (e) => { figureClick(e.target as HTMLElement); });
thread.addEventListener("keydown", (e) => {
  const card = (e.target as HTMLElement).closest<HTMLElement>(".art-card");
  if (!card || (e.key !== "Enter" && e.key !== " ")) return;
  e.preventDefault();
  openShelf(card.dataset.art, Number(card.dataset.v));
});
thread.addEventListener("click", (e) => {
  const t = e.target as HTMLElement, act = t.closest<HTMLElement>("[data-act]");
  const tok = t.closest<HTMLElement>(".tok");
  if (tok) return void openTokPop(tok);
  const card = t.closest<HTMLElement>(".art-card");
  if (card) return openShelf(card.dataset.art, Number(card.dataset.v));
  if (figureClick(t)) return;
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
  if ((act.dataset.act === "prev" || act.dataset.act === "next") && m.versions) {
    const k = m.versions.shown + (act.dataset.act === "prev" ? -1 : 1);
    editing = null;
    showVersion(conv, i, versionsAt(conv, i), k);
    save(conv);
    render();
    const again = thread.querySelector<HTMLButtonElement>(`article[data-i="${i}"] [data-act=${act.dataset.act}]`);
    return void (again && !again.disabled ? again : thread.querySelector<HTMLElement>(`article[data-i="${i}"] .pager [data-act]:not(:disabled)`))?.focus();
  }
  if (act.dataset.act === "card" || act.dataset.act === "more") return toggleCard(act);
  if (act.dataset.act === "decide") return void decide(act);
  if (act.dataset.act === "agent-stop") return stopAgent(act, m);
  if (act.dataset.act === "speak") return readAloud(m);
  if (running()) return;
  if (act.dataset.act === "retry") void generate(conv, i);
  else if (act.dataset.act === "cont") continueReply(m);
  else if (act.dataset.act === "trash") deleteMessage(i);
  else if (act.dataset.act === "edit") {
    editing = m;
    render();
  }
});
document.addEventListener("pointerdown", (e) => {
  if (!$("tokpop").hidden && !(e.target as Element).closest("#tokpop, .tok")) closeTokPop();
});

function toggleCard(btn: HTMLElement) {
  const card = btn.closest<HTMLElement>(".tool-card")!, id = card.dataset.call!;
  if (btn.dataset.act === "more") return void btn.closest(".tool-result")!.classList.remove("long");
  if (!openCards.delete(id)) openCards.add(id);
  const m = conv.messages[Number(card.closest<HTMLElement>("article")?.dataset.i)];
  syncCards(card.parentElement!, cardLists.get(card.parentElement!) ?? [], m, running()?.m === m);
}
function findCall(calls: Call[] | undefined, id: string): Call | undefined {
  for (const c of calls ?? []) {
    const found = c.id === id ? c : c.agent && (findCall(c.agent.calls, id) ?? c.agent.rounds?.map((r) => findCall(r.calls, id)).find(Boolean));
    if (found) return found;
  }
  return undefined;
}
function stopAgent(btn: HTMLElement, m: Message) {
  const card = btn.closest<HTMLElement>(".agent-card")!;
  if (!m.job) return;
  (btn as HTMLButtonElement).disabled = true;
  void fetch(`/api/chat/${m.job}/agent/${encodeURIComponent(card.dataset.call!)}/stop`, { method: "POST" }).catch(() => {});
}

async function decide(btn: HTMLElement) {
  const card = btn.closest<HTMLElement>(".tool-card")!, id = card.dataset.call!, decision = btn.dataset.decision!;
  const m = conv.messages[Number(card.closest<HTMLElement>("article")?.dataset.i)], c = findCall(m?.calls, id);
  if (!m?.job || !c) return;
  if (decision === "always" && !settings.allow?.includes(c.name)) {
    settings.allow = [...(settings.allow ?? []), c.name].sort();
    saveSettings();
  }
  deciding.add(id);
  const r = running();
  if (r) schedule(r);
  const res = await fetch(`/api/chat/${m.job}/tool/${encodeURIComponent(id)}`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision }) }).catch(() => null);
  if (!res?.ok && res?.status !== 409) { deciding.delete(id); toast("The answer did not reach the web server: try again"); }
}
scroller.addEventListener("scroll", () => { if (!$("tokpop").hidden) closeTokPop(); }, { passive: true });
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (voice) return closeVoice();
  const r = running();
  if (r) stopRun(r);
  else submit(input.value);
});
input.addEventListener("input", () => { grow(); if (!drafting) $("draft").title = input.value.trim() ? "Finish my message (draft)" : "Draft my next message"; });
input.addEventListener("input", () => { if (talking) { talking = 0; renderCastNext(); } });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    submit(input.value);
  }
});
document.addEventListener("keydown", (e) => {
  const ctrl = (e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey, keyed = ctrl && PANES.find((p) => e.key.toLowerCase() === p.key);
  if (ctrl && e.key === ",") {
    e.preventDefault();
    return panel.open ? panel.close() : openSettings();
  }
  if (keyed) {
    e.preventDefault();
    if (keyed === panes.artifacts) return isOpen(keyed) ? closeShelf() : openShelf();
    return setPane(keyed, !isOpen(keyed));
  }
  if (e.key !== "Escape" || e.defaultPrevented) return;
  if (!$("tokpop").hidden) return closeTokPop();
  if (toolsOpen) return setToolsOpen(false, true);
  if (modelsOpen) return setModelsOpen(false, true);
  if (voicePop) return setVoicePop(false, true);
  if (document.querySelector("dialog[open]")) return;
  const over = PANES.find((p) => isOpen(p) && !p.beside.matches);
  if (over) return over === panes.artifacts ? closeShelf() : setPane(over, false);
  if (voice) return voice.reply || voice.client.playing ? cutVoice(true) : closeVoice();
  const r = running();
  if (r) stopRun(r);
});
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
  if (e.touches[0].clientY > touchY + 2) stopFollowing();
  touchY = e.touches[0].clientY;
}, { passive: true });
scroller.addEventListener("pointerdown", (e) => { if (e.target === scroller && e.offsetX >= scroller.clientWidth) stopFollowing(); });
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
$("new-chat").addEventListener("click", () => newChat());
$("add-cast").addEventListener("click", (e) => { e.stopPropagation(); openAdd(e.currentTarget as HTMLElement); });
$("export").addEventListener("click", () => void exportChats([conv], fileName(conv.title)));
$("export-all").addEventListener("click", () => {
  const all = Object.values(chats).sort((a, b) => b.updated - a.updated);
  if (all.length) void exportChats(all, `chrysalis-chats-${new Date().toISOString().slice(0, 10)}.json`);
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
let dragDepth = 0;
const dropsHere = (e: DragEvent) => !!e.dataTransfer?.types.includes("Files")
  || (!!e.dataTransfer?.types.includes(CAST_DRAG) && !!folderOf(conv) && !!(e.target as Element).closest?.("main"));
document.addEventListener("dragenter", (e) => { if (dropsHere(e) && dragDepth++ === 0) document.querySelector(".composer-box")!.classList.add("dropping"); });
document.addEventListener("dragleave", (e) => { if (dropsHere(e) && --dragDepth === 0) document.querySelector(".composer-box")!.classList.remove("dropping"); });
document.addEventListener("dragover", (e) => { if (dropsHere(e)) e.preventDefault(); });
document.addEventListener("drop", (e) => {
  if (!dropsHere(e)) return;
  e.preventDefault();
  dragDepth = 0;
  document.querySelector(".composer-box")!.classList.remove("dropping");
  const who = e.dataTransfer!.getData(CAST_DRAG);
  if (who) join([who]);
  else void addFiles(e.dataTransfer!.files);
});
document.addEventListener("dragend", () => { dragDepth = 0; document.querySelector(".composer-box")!.classList.remove("dropping"); });
$("import").addEventListener("click", () => $("import-file").click());
$("import-file").addEventListener("change", (e) => {
  const input = e.currentTarget as HTMLInputElement;
  if (input.files?.length) void importFiles(input.files).finally(() => { input.value = ""; });
});

const MAIN_MIN = 640;
const isOpen = (p: Pane) => p.el.hasAttribute("data-open");
const besideOpen = (p: Pane) => isOpen(p) && p.beside.matches;
let still = 0;
function hush() {
  still++;
  document.documentElement.classList.add("no-anim");
}
function unhush() {
  requestAnimationFrame(() => requestAnimationFrame(() => { if (--still === 0) document.documentElement.classList.remove("no-anim"); }));
}
function fitPanes() {
  const { conversations: c, artifacts: s } = panes, room = innerWidth - MAIN_MIN;
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
  if (p.beside.matches && p !== panes.artifacts) setPref(`pane-${p.name}`, open ? "open" : "closed");
  if (open && !p.beside.matches) for (const q of PANES) if (q !== p) closeOver(q);
  fitPanes();
  if (open) {
    p.onOpen?.();
    if (focus && !p.beside.matches) p.el.focus({ preventScroll: true });
  } else if (p.el.contains(document.activeElement)) p.toggle.focus();
  if (!animate) unhush();
}
function closeOver(p: Pane) {
  if (!isOpen(p) || p.beside.matches) return;
  if (p === panes.artifacts) return closeShelf();
  setPane(p, false, { focus: false });
}
function restorePane(p: Pane) {
  const saved = pref(`pane-${p.name}`, "");
  p.preferred = Number(pref(`pane-${p.name}-width`, "")) || p.def;
  setPane(p, p.beside.matches && (saved ? saved === "open" : p.byDefault), { focus: false, animate: false });
}
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
    const end = () => {
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
  if (p.toggle.classList.contains("pane-toggle")) p.toggle.innerHTML = icon(p.el.dataset.side === "left" ? "paneLeft" : "paneRight");
  p.toggle.title = `${label} (Ctrl+${p.key.toUpperCase()})`;
  p.toggle.addEventListener("click", () => p === panes.artifacts ? (isOpen(p) ? closeShelf() : openShelf()) : setPane(p, !isOpen(p)));
  const rail = p.el.querySelector<HTMLElement>(".pane-rail")!;
  rail.title = "Drag to resize · double-click for the default width";
  rail.setAttribute("aria-label", `Resize ${label.toLowerCase()}`);
  resizeWith(p, rail);
  p.beside.addEventListener("change", () => p === panes.artifacts ? renderShelf() : restorePane(p));
}
addEventListener("resize", () => { hush(); fitPanes(); unhush(); });
document.addEventListener("pointerdown", (e) => {
  const t = e.target as Node;
  for (const p of PANES) if (!p.el.contains(t) && !p.toggle.contains(t) && !$("toast").contains(t) && !(t as Element).closest?.(".art-card")) closeOver(p);
});
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
    const at = EFFORTS.indexOf(barEffort()), step = ({ ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 } as Record<string, number>)[e.key];
    if (step) setEffort(at + step);
    else if (e.key === "Home" || e.key === "End") setEffort(e.key === "Home" ? 0 : 3);
    else return;
    e.preventDefault();
  });
  $("model-btn").addEventListener("click", () => setModelsOpen(!modelsOpen));
  $("backend").addEventListener("submit", (e) => { e.preventDefault(); void setBackend(); });
  $("backend").addEventListener("input", () => { backendEdited = true; backendNote(""); renderBackend(); });
  $("backend").addEventListener("keydown", (e) => { if (e.key === "Escape") { resetBackend(); (e.target as HTMLElement).blur(); } });
  $("backend-far").addEventListener("click", () => void setBackend(true));
  $("power").addEventListener("click", async () => {
    if (status.state === "online") await fetch("/api/chrysalis/stop", { method: "POST" });
    else await start();
    await refreshStatus();
  });
  document.addEventListener("pointerdown", (e) => {
    if (modelsOpen && !(e.target as Element).closest("#model-pop, #model-btn")) setModelsOpen(false);
    if (voicePop && !(e.target as Element).closest("#voice-pop, #voice-pick")) setVoicePop(false);
  });
}

const barEffort = (): Effort => conv.effort;
function setEffort(i: number) {
  const e = EFFORTS[Math.min(3, Math.max(0, i))];
  if (e === barEffort()) return;
  conv.effort = e;
  setPref("effort", e);
  renderComposer();
}

let backendEdited = false;
function renderBackend() {
  const b = status.backend, form = $("backend");
  form.hidden = !b;
  if (!b) return;
  const url = $<HTMLInputElement>("backend-url"), key = $<HTMLInputElement>("backend-key"), here = !!status.here;
  const down = b.listed && !b.listed.ok ? b.listed.error : "", dot = $("backend-dot");
  dot.className = `dot ${down ? "offline" : b.remote ? "remote" : status.state}`;
  dot.title = down || (b.remote ? "Off this computer and its network" : status.state === "online" ? "Answering" : status.state === "loading" ? "Loading" : "Stopped");
  if (!backendEdited && document.activeElement !== url) url.value = b.url;
  url.readOnly = !here;
  url.title = here ? `${b.name}: replies come from here. Paste another backend's address and press Enter.` : "Set on the computer Chrysalis runs on";
  key.hidden = !here;
  key.placeholder = b.key ? "key set" : "API key";
  $("backend-go").hidden = !here || !backendEdited;
  let known: string[] = [];
  try { known = JSON.parse(pref("backends", "[]")); } catch {  }
  if (!known.includes(b.url)) setPref("backends", JSON.stringify((known = [b.url, ...known].slice(0, 8))));
  const list = $("backend-known");
  if (list.childElementCount !== known.length) list.replaceChildren(...known.map((u) => Object.assign(h("option"), { value: u })));
  renderLink();
}
function renderLink() {
  const l = status.link, p = l?.pairing, note = $("link-note"), parts: (string | Node)[] = [];
  if (l?.refused) parts.push(l.refused);
  else if (p?.state === "waiting" && p.code && !p.confirmed) {
    const yes = h("button", "btn small primary", "Yes");
    yes.type = "button";
    yes.onclick = () => void fetch("/api/link/confirm", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }).then(() => refreshStatus());
    parts.push(`Does ${p.name} show ${spaced(p.code)}? Allow it there, and here:`, yes);
  } else if (p?.state === "waiting") parts.push(p.code ? `Allow ${spaced(p.code)} on ${p.name}` : `Asking ${p.name}…`);
  else {
    if (p?.state === "gone") parts.push(`${p.name} did not allow this computer.`);
    if (p?.state === "error") parts.push(p.error ?? "");
    for (const f of !l?.linked && status.state === "offline" ? l?.found ?? [] : []) {
      const use = h("button", "btn small primary", "Use it");
      use.type = "button";
      use.onclick = () => void fetch("/api/link", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ fp: f.fp }) })
        .then(() => refreshStatus());
      parts.push(`Chrysalis is on ${f.name}.`, use);
    }
  }
  const key = JSON.stringify([parts.map((x) => typeof x === "string" ? x : x.textContent), l?.found]);
  if (note.dataset.key !== key) { note.dataset.key = key; note.replaceChildren(...parts); }
  note.hidden = !parts.length || !$("backend-note").hidden;
}
function backendNote(text: string, error = false, far = false) {
  $("backend-why").textContent = text;
  $("backend-note").classList.toggle("error", error);
  $("backend-note").hidden = !text;
  $("backend-far").hidden = !far;
}
function resetBackend() {
  backendEdited = false;
  $<HTMLInputElement>("backend-url").value = status.backend?.url ?? "";
  $<HTMLInputElement>("backend-key").value = "";
  backendNote("");
  renderBackend();
}
async function setBackend(confirmRemote = false) {
  const url = $<HTMLInputElement>("backend-url").value.trim(), key = $<HTMLInputElement>("backend-key").value, go = $<HTMLButtonElement>("backend-go");
  go.disabled = true;
  backendNote("Checking…");
  try {
    const r = await fetch("/api/backend", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, ...(key ? { key } : {}), ...(confirmRemote ? { confirmRemote: true } : {}) }) });
    const body = await r.json().catch(() => ({}));
    if (r.status === 409 && body.remote)
      return backendNote(`Every chat you send would go to ${body.remote}, off this computer and its network.`, true, true);
    if (!r.ok) return backendNote(body.error?.message ?? `HTTP ${r.status}`, true);
    status = { ...status, backend: body };
    resetBackend();
    (document.activeElement as HTMLElement | null)?.blur();
    await refreshStatus();
    if (!conv.messages.length) render();
  } catch (e) {
    backendNote(e instanceof Error ? e.message : String(e), true);
  } finally {
    go.disabled = false;
  }
}

function setModelsOpen(open: boolean, refocus = false) {
  if (open && toolsOpen) setToolsOpen(false);
  modelsOpen = open;
  $("model-pop").hidden = !open;
  delete $("model-pop").dataset.key;
  renderControls();
  if (open) $("model-pop").querySelector<HTMLElement>(".current")?.focus();
  else if (refocus) $("model-btn").focus();
}

let catalog: Catalog | null = null, toolsOpen = false;
const shownTools = new Set<string>();
let fetchKnob: HTMLElement | undefined;
const JSON_POST = { method: "POST", headers: { "Content-Type": "application/json" } };
const FETCH_CHARS = 20_000;
const EXAMPLE = '{"mcpServers": {"notes": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]}}}';

async function loadCatalog(how: { reload?: boolean; connect?: string } = {}) {
  try {
    const r = how.reload ? await fetch("/api/tools/reload", JSON_POST)
      : how.connect ? await fetch("/api/tools/connect", { ...JSON_POST, body: JSON.stringify({ source: how.connect }) }) : await fetch("/api/tools");
    if (r.ok) catalog = await r.json();
  } catch {  }
  renderTools();
  void watchTools();
}
let watching = false;
async function watchTools() {
  if (watching) return;
  watching = true;
  try {
    for (let k = 0; k < 120 && catalog?.sources.some((s) => s.state === "connecting"); k++) {
      await sleep(500);
      const r = await fetch("/api/tools").catch(() => null);
      if (r?.ok) catalog = await r.json();
      renderTools();
    }
  } finally { watching = false; }
}

const isOn = (id: string, c = conv) => !!c.tools?.[id];
const toolOn = (id: string, name: string, c = conv) => isOn(id, c) && !c.tools![id].off?.includes(name);
const onCount = (c = conv) => (catalog?.sources ?? []).reduce((n, s) => n + s.tools.filter((t) => toolOn(s.id, t.name, c)).length, 0);
function setTools(next: NonNullable<Conversation["tools"]>) {
  conv.tools = Object.keys(next).length ? next : undefined;
  setPref("tools", JSON.stringify(conv.tools ?? {}));
  save(conv);
  renderTools();
  if (panel.open) syncSettings();
}
function setSource(id: string, on: boolean, only?: string) {
  const next = { ...conv.tools }, s = catalog?.sources.find((x) => x.id === id);
  if (!on) delete next[id];
  else {
    next[id] = only && s ? { off: s.tools.map((t) => t.name).filter((n) => n !== only).sort() } : {};
    if (s && (s.state === "idle" || s.state === "failed")) void loadCatalog({ connect: id });
  }
  setTools(next);
}
function setTool(id: string, name: string, on: boolean) {
  if (!isOn(id)) return void (on && setSource(id, true, name));
  const off = new Set(conv.tools![id].off ?? []), s = catalog?.sources.find((x) => x.id === id);
  if (on) off.delete(name); else off.add(name);
  if (s && s.tools.every((t) => off.has(t.name))) return setSource(id, false);
  setTools({ ...conv.tools, [id]: off.size ? { off: [...off].sort() } : {} });
}

const dotOf = (s: SourceInfo) => (s.state === "connected" ? "online" : s.state === "connecting" ? "loading" : s.state === "failed" ? "failed" : "");
function stateLine(s: SourceInfo) {
  const n = s.tools.length, k = s.tools.filter((t) => toolOn(s.id, t.name)).length;
  if (s.kind === "fetch") return "Reads web pages at public addresses";
  if (s.kind === "agent") return "Hands tasks to agents that work side by side";
  return s.state === "connected" ? `${k} of ${n} tool${n === 1 ? "" : "s"} on` : s.state === "connecting" ? "Starting…"
    : s.state === "failed" ? "Could not start" : s.state === "disabled" ? "Disabled in the config file" : isOn(s.id) ? "Starts with the next reply" : "Not started";
}
const toolTag = (t: ToolInfo) => (t.readOnly ? ["read-only", "Changes nothing: runs without asking"]
  : settings.allow?.includes(t.fn) ? ["runs without asking", "You allowed it for good (Settings, Tools)"] : ["asks first", "Asks you before each run"]);
function keyed<T extends HTMLElement>(el: T, what: string, src = "", tool = ""): T {
  Object.assign(el.dataset, { do: what, src, tool, k: `${what}|${src}|${tool}` });
  return el;
}
function toolSwitch(what: string, src: string, tool: string, on: boolean, label: string, disabled = false) {
  const b = keyed(h("button", "switch"), what, src, tool);
  b.type = "button";
  b.setAttribute("role", "switch");
  b.setAttribute("aria-checked", String(on));
  b.setAttribute("aria-label", label);
  b.disabled = disabled;
  return b;
}
function sourceView(s: SourceInfo): HTMLElement {
  const box = h("div", `tp-src${isOn(s.id) ? " on" : ""}`), row = h("div", "tp-row"), text = h("span", "tp-text");
  const lead = s.kind === "mcp" ? h("span", `dot ${dotOf(s)}`) : h("span", "tp-ico");
  if (s.kind !== "mcp") lead.innerHTML = icon(s.kind === "fetch" ? "globe" : "agent");
  else lead.title = s.state;
  text.append(h("span", "tp-name", s.label), h("span", "tp-sub", stateLine(s)));
  row.append(lead, text);
  const open = shownTools.has(s.id) && s.kind === "mcp" && s.tools.length > 0;
  if (s.kind === "mcp" && s.tools.length) {
    const more = h("button", `act tp-more${open ? " open" : ""}`);
    more.type = "button";
    keyed(more, "more", s.id);
    more.setAttribute("aria-expanded", String(open));
    more.setAttribute("aria-label", `${open ? "Hide" : "Show"} the tools of ${s.label}`);
    more.title = open ? "Hide its tools" : "Show its tools";
    more.innerHTML = icon("chevron");
    row.append(more);
  }
  row.append(toolSwitch("src", s.id, "", isOn(s.id), `${s.label} in this chat`, s.state === "disabled"));
  box.append(row);
  if (s.kind === "fetch" && isOn(s.id) && fetchKnob) box.append(fetchKnob);
  if (s.state === "failed" && s.error) {
    const err = h("div", "tp-err"), retry = h("button", "btn small", "Retry");
    retry.type = "button";
    keyed(retry, "retry", s.id);
    err.append(h("span", "", s.error), retry);
    box.append(err);
  }
  if (open) {
    const list = h("div", "tp-tools");
    for (const t of s.tools) {
      const r = h("div", "tp-tool"), [tag, why] = toolTag(t), label = h("span", "tp-tname", t.title), badge = h("span", `tp-tag${t.readOnly ? "" : " asks"}`, tag);
      label.title = DESCRIPTIONS[t.name] ?? (t.description || t.name);
      badge.title = why;
      r.append(label, badge, toolSwitch("tool", s.id, t.name, toolOn(s.id, t.name), `${t.title} (${s.label})`));
      list.append(r);
    }
    box.append(list);
  }
  return box;
}
function renderTools() {
  const btn = $("tools-btn"), n = onCount(), on = toolsOn(conv);
  const starting = !!catalog?.sources.some((s) => isOn(s.id) && s.state === "connecting");
  const label = on ? `Tools: ${n} on in this chat` : "Tools: off in this chat";
  const html = `${icon("wrench")}${on ? `<span class="tools-n">${n || "…"}</span>` : ""}`;
  if (btn.innerHTML !== html) btn.innerHTML = html;
  btn.title = label;
  btn.setAttribute("aria-label", label);
  btn.setAttribute("aria-expanded", String(toolsOpen));
  btn.classList.toggle("on", on);
  btn.classList.toggle("open", toolsOpen);
  btn.classList.toggle("pending", starting);
  if (!toolsOpen) return;
  const pop = $("tools-pop"), key = JSON.stringify([catalog, conv.tools, [...shownTools], settings.allow]);
  if (pop.dataset.key === key) return;
  pop.dataset.key = key;
  const focus = (document.activeElement as HTMLElement | null)?.dataset?.k;
  const head = h("div", "tp-head");
  head.append(h("span", "tp-title", "Tools in this chat"), h("span", "tp-count", on ? `${n} on` : "off"));
  const parts: HTMLElement[] = [head];
  if (!catalog) parts.push(h("div", "tp-empty", "Asking the web server…"));
  else {
    if (catalog.error) parts.push(h("div", "tp-err", catalog.error));
    const builtIn = catalog.sources.filter((s) => s.kind !== "mcp"), mcp = catalog.sources.filter((s) => s.kind === "mcp");
    parts.push(...builtIn.map(sourceView), h("div", "tp-group", "MCP servers"));
    if (mcp.length) parts.push(...mcp.map(sourceView));
    else {
      const help = h("div", "tp-empty");
      help.append("None yet. Add servers to ", h("code", "", catalog.config), " (the shape Claude Desktop and Claude Code use), for example:",
        h("pre", "tp-code", EXAMPLE), "then Reload.");
      parts.push(help);
    }
    if (settings.allow?.length) {
      const allow = h("div", "tp-allow");
      allow.append(h("span", "tp-allow-label", "Run without asking"), ...settings.allow.map((fn) => {
        const b = keyed(h("button", "chip", fn), "unallow", "", fn);
        b.type = "button";
        b.title = b.ariaLabel = `${fn}: ask first again`;
        b.insertAdjacentHTML("beforeend", icon("x"));
        return b;
      }));
      parts.push(allow);
    }
    const foot = h("div", "tp-foot"), reload = h("button", "btn small", "Reload config");
    reload.type = "button";
    keyed(reload, "reload");
    reload.title = `Read ${catalog.config} again (servers that changed restart)`;
    foot.append(reload);
    parts.push(foot);
  }
  pop.replaceChildren(...parts);
  if (focus) pop.querySelector<HTMLElement>(`[data-k="${CSS.escape(focus)}"]`)?.focus();
}
function setToolsOpen(open: boolean, refocus = false) {
  toolsOpen = open;
  const pop = $("tools-pop");
  pop.hidden = !open;
  delete pop.dataset.key;
  if (open) {
    setModelsOpen(false);
    const btn = $("tools-btn"), room = pop.offsetParent?.clientWidth ?? innerWidth;
    pop.style.left = `${Math.max(8, Math.min(btn.offsetLeft - 8, room - pop.offsetWidth - 8))}px`;
    void loadCatalog();
  }
  renderTools();
  if (open) pop.querySelector<HTMLElement>("[data-do]")?.focus();
  else if (refocus) $("tools-btn").focus();
}
function buildTools() {
  $("tools-btn").addEventListener("click", () => setToolsOpen(!toolsOpen));
  $("tools-pop").addEventListener("click", (e) => {
    const el = (e.target as Element).closest<HTMLElement>("[data-do]");
    if (!el || (el as HTMLButtonElement).disabled) return;
    const { do: what, src = "", tool = "" } = el.dataset;
    if (what === "src") setSource(src, !isOn(src));
    else if (what === "tool") setTool(src, tool, !toolOn(src, tool));
    else if (what === "more") { if (!shownTools.delete(src)) shownTools.add(src); renderTools(); }
    else if (what === "retry") void loadCatalog({ connect: src });
    else if (what === "reload") void loadCatalog({ reload: true });
    else if (what === "unallow") {
      const before = settings.allow;
      settings.allow = settings.allow?.filter((x) => x !== tool);
      if (!settings.allow?.length) delete settings.allow;
      saveSettings();
      renderTools();
      toast(`${tool} asks first again`, () => { settings.allow = before; saveSettings(); renderTools(); });
    }
  });
  document.addEventListener("pointerdown", (e) => {
    if (toolsOpen && !(e.target as Element).closest("#tools-pop, #tools-btn")) setToolsOpen(false);
  });
}

const shortName = (id: string) => modelName(id).replace(/^Qwen3\.8-/, "");
function renderMeter() {
  const online = status.state === "online" && !status.loading, ctx = status.context ?? 262144;
  const used = status.kv_tokens && status.cache ? 1 - status.cache.free_kv_tokens / status.kv_tokens : 0;
  const r = running(), s = conv.messages.findLast((m) => m.stats)?.stats;
  const n = r?.read !== undefined ? r.read + (r.written ?? 0) : s ? s.ctx ?? s.prompt + s.tokens : undefined;
  if (!onService()) {
    $("meter").replaceChildren(...(n !== undefined ? [h("span", "", `${fmtK(n)} tokens`)] : []));
    $("meter").title = n !== undefined ? `This chat: ${n.toLocaleString("en-US")} tokens` : "";
    return;
  }
  const meter = $("meter"), parts = [n !== undefined ? `${fmtK(n)} / ${fmtK(ctx)}` : online ? `${fmtK(ctx)} context` : "", online ? `cache ${Math.round(used * 100)}%` : ""];
  meter.replaceChildren(...parts.filter(Boolean).map((t) => h("span", "", t)));
  meter.title = [n !== undefined ? `This chat: ${n.toLocaleString("en-US")} of ${ctx.toLocaleString("en-US")} tokens of context` : "",
    online ? `Cache: ${fmtK(status.cache?.free_kv_tokens ?? 0)} of ${fmtK(status.kv_tokens ?? 0)} KV tokens free (cached conversations are dropped as needed)` : ""].filter(Boolean).join("\n");
}
function renderControls() {
  renderTools();
  renderBackend();
  const effort = barEffort(), at = EFFORTS.indexOf(effort), energy = $("energy");
  energy.dataset.heat = HEAT[at];
  energy.style.setProperty("--v", String(at));
  energy.setAttribute("aria-valuenow", String(at));
  energy.setAttribute("aria-valuetext", LEVEL[effort][0]);
  energy.title = `Thinking: ${LEVEL[effort][0]}`;
  energy.querySelectorAll(".energy-dot").forEach((d, k) => d.classList.toggle("lit", at > 0 && k <= at));
  $("energy-stops").dataset.heat = HEAT[at];
  $("energy-stops").querySelectorAll("button").forEach((b, k) => b.classList.toggle("on", k === at));
  const state = status.loading || status.state === "loading" ? "loading" : status.state;
  const pct = status.progress == null ? "" : `${Math.round(status.progress * 100)}%`, name = modelName(current());
  const btn = $("model-btn");
  btn.innerHTML = `<span class="dot ${state}"></span>${shortName(current())}${state === "loading" && pct ? `<span class="model-pct">${pct}</span>` : ""}${icon("chevron")}`;
  const b = status.backend, down = b?.listed && !b.listed.ok ? b.listed.error : "";
  btn.title = !onService() ? (down ? `${b!.name}: ${down}` : `${name || "No model"} on ${b!.name}`)
    : state === "online" ? `${name}: ready` : state === "loading" ? `Loading ${name}` : `${name}: stopped`;
  btn.setAttribute("aria-expanded", String(modelsOpen));
  btn.classList.toggle("open", modelsOpen);
  renderMeter();
  const power = $<HTMLButtonElement>("power");
  power.innerHTML = icon("power");
  power.title = power.ariaLabel = state === "online" ? `Unload ${name}` : state === "loading" ? `Loading ${name}` : `Load ${name}`;
  power.disabled = state === "loading";
  power.hidden = !onService();
  if (!modelsOpen) return;
  const pop = $("model-pop");
  const key = JSON.stringify([current(), state, pct, listed()]);
  if (pop.dataset.key === key) return;
  pop.dataset.key = key;
  if (!onService()) return void pop.replaceChildren(...otherRows(down));
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
      if (!on || state === "offline") { picked = id; setPref("model", id); void (state === "offline" ? start().then(() => refreshStatus()) : pickModel(id)); }
      setModelsOpen(false, true);
    };
    return item;
  });
  pop.replaceChildren(...rows);
}

function otherRows(down: string): HTMLElement[] {
  if (!listed().length) return [h("div", "model-empty", down || `${status.backend?.name} lists no model to chat with`)];
  return listed().map((id) => {
    const on = id === current(), item = h("button", on ? "model-item current" : "model-item");
    item.type = "button";
    item.title = id;
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", String(on));
    const mark = h("span", "model-check");
    if (on) mark.innerHTML = icon("check");
    item.append(h("span", "model-label", modelName(id)), mark);
    item.onclick = () => {
      setPref(`model@${status.backend?.url}`, id);
      setModelsOpen(false, true);
      if (!conv.messages.length) render();
    };
    return item;
  });
}


const modeOf = (c: Conversation): Mode => (c.effort === "off" ? "fast" : "think");
let chatMode: Mode | undefined, settingsMode: Mode | undefined;
const shownMode = (): Mode => (panel.open && settingsMode) || modeOf(conv);

function loadSettings(): Settings {
  const s = load<(Settings & Record<string, unknown>) | null>("settings2", null);
  if (s) {
    if (s.system === PLAIN_SYSTEM) s.system = DEFAULT_SYSTEM;
    for (const k of ["systemTools", "prompts", "descriptions"]) delete s[k];
    if (s.v !== 5) for (const m of [s.think, s.fast]) if (m?.order && m.order.length < DEFAULT_STAGES.length) delete m.order;
    s.v = 5;
    return s;
  }
  const old = load<{ system?: string; sampling?: Record<string, number> } | null>("settings", null);
  const { max_tokens, seed, ...rest } = old?.sampling ?? {};
  return { v: 5, system: [DEFAULT_SYSTEM, old?.system?.trim()].filter(Boolean).join("\n\n"), think: { ...rest }, fast: { ...rest }, max_tokens, seed };
}
function saveSettings() {
  store("settings2", settings);
  pushVoice();
  if (voice) { renderVoice(); renderVoicePop(); }
  syncSettings();
}
const value = (mode: Mode, key: SamplerKey) => settings[mode][key] ?? RECOMMENDED[mode][key];
const orderOf = (mode: Mode) => settings[mode].order ?? DEFAULT_STAGES;
const fmtValue = (key: SamplerKey, v: number) => {
  const d = SAMPLERS[key];
  if (d.all !== undefined && v === d.all) return "all";
  if (d.off !== undefined && (v === d.off || (d.off < 0 && v <= 0))) return "off";
  return d.step >= 1 ? String(v) : v.toFixed(2);
};
function samplerFields(mode: Mode) {
  const { order, ...rest } = settings[mode];
  return { ...rest, ...(order ? { samplers: order } : {}) };
}
function changed(mode: Mode) {
  const out = (Object.keys(SAMPLERS) as SamplerKey[]).filter((k) => settings[mode][k] !== undefined)
    .map((k) => `${SAMPLERS[k].short} ${fmtValue(k, settings[mode][k]!)}`);
  if (settings[mode].order) out.push(settings[mode].order!.join("›"));
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


const FIRST: Cand[] = ODDS.map(([word, c], i) => ({ word, logit: Math.log(c), id: -1 - i }));
const pct = (p: number) => (p >= 0.995 ? "100%" : p >= 0.01 ? `${Math.round(p * 100)}%` : p > 0 ? "<1%" : "0%");
const CUT_BY: Record<string, string> = { penalties: "penalties", dry: "DRY", top_n_sigma: "top-nσ", top_k: "top-k", typ_p: "typical-p", top_p: "top-p",
  min_p: "min-p", xtc: "XTC", temperature: "temp 0", mirostat: "mirostat" };
const SHOWN = [0, 1, 2, 3, 4, 5];
function oddsChart(fig: HTMLElement) {
  const head = h("div", "odd-head"), count = h("span", "odd-count"), box = h("div", "odd-rows"), min = h("div", "odd-min"), caption = h("figcaption", "odd-caption");
  head.append(h("span", "odd-title", "Qwen names a cat"), count);
  const els = [...SHOWN, -1].map(() => {
    const el = h("div", "odd"), word = h("span", "odd-word"), track = h("span", "odd-track"), p = h("span", "odd-p");
    const base = h("i", "odd-base"), bar = h("i", "odd-bar");
    track.append(base, bar);
    el.append(word, track, p);
    box.append(el);
    return { el, word, base, bar, p };
  });
  min.append(h("span", "", "min-p"));
  box.append(min);
  fig.replaceChildren(head, box, caption);
  let at = -1, last: (() => void) | null = null;
  box.addEventListener("pointerover", (e) => {
    const i = els.findIndex((x) => x.el.contains(e.target as Node));
    if (i !== at) { at = i; last?.(); }
  });
  box.addEventListener("pointerleave", () => { at = -1; last?.(); });
  return (rows: Odds[], minLine: { at: number; tempered: boolean } | undefined, xtc: number) => {
    last = () => {
      const rest = rows.filter((_, i) => !SHOWN.includes(i)), restCut = rest.filter((r) => r.cut).length;
      const agg: Odds = { word: `${rest.length} more`, raw: rest.reduce((a, r) => a + r.raw, 0), p: rest.reduce((a, r) => a + r.p, 0),
        cut: restCut === rest.length ? rest[0].cut : "", penalty: 0 };
      [...SHOWN.map((i) => rows[i]), agg].forEach((r, k) => {
        const o = els[k];
        o.el.classList.toggle("cut", !!r.cut);
        o.el.classList.toggle("rest", k === SHOWN.length);
        o.el.classList.toggle("at", k === at);
        o.word.textContent = r.word;
        o.base.style.width = `${r.raw * 100}%`;
        o.bar.style.width = `${r.cut ? 0 : r.p * 100}%`;
        o.p.textContent = r.cut ? CUT_BY[r.cut] : pct(r.p);
      });
      const kept = rows.filter((r) => !r.cut), topNow = kept.reduce((a, r) => (r.p > a.p ? r : a), kept[0]);
      count.textContent = kept.length === 1 ? `always ${kept[0].word}` : `${kept.length} of ${rows.length} can be drawn`;
      min.classList.toggle("on", !!minLine);
      min.classList.toggle("untempered", !!minLine && !minLine.tempered);
      if (minLine) min.style.setProperty("--at", String(Math.min(1, minLine.at * topNow.p)));
      const by = Object.entries(Object.groupBy(rows.filter((r) => r.cut), (r) => r.cut)).map(([c, l]) => `${l!.length} by ${CUT_BY[c]}`);
      const r = at >= 0 && at < SHOWN.length ? rows[SHOWN[at]] : undefined;
      caption.innerHTML = r ? `<b>${r.word}</b> ${pct(r.raw)} → ${r.cut ? `cut by ${CUT_BY[r.cut]}` : pct(r.p)}`
        : at === SHOWN.length ? `<b>${rest.length} more names</b> ${pct(agg.p)} together${restCut ? ` · ${restCut} cut` : ""}`
        : `${by.length ? `Cut ${by.join(", ")}` : "Nothing cut"}${xtc > 0 && xtc < 1 ? ` · XTC shown cutting, as ${pct(xtc)} of draws do` : ""}<span> · faint bars: before sampling</span>`;
    };
    last();
  };
}
let drawOdds: ReturnType<typeof oddsChart> | null = null;
function drawCharts() {
  const mode = shownMode(), v = (k: SamplerKey) => value(mode, k);
  drawOdds ??= oddsChart($("odds"));
  const { rows, minLine } = sample(FIRST, [], v, orderOf(mode));
  drawOdds(rows, minLine, v("xtc_probability"));
}

interface KnobSpec {
  id: string; label: string; help: string; min: number; max: number; step: number; parse?(text: string): number;
  get(): number; rec(): number; custom(): boolean; set(v: number | undefined): void; fmt(v: number): string;
  pos(v: number): number; val(pos: number): number;
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
  range.tabIndex = -1;
  range.setAttribute("aria-hidden", "true");
  reset.type = "button";
  reset.innerHTML = icon("retry");
  track.append(rec, range);
  el.append(label, track, num, reset);
  const commit = (v: number) => {
    if (!Number.isFinite(v)) return sync(true);
    v = Math.min(k.max, Math.max(k.min, roundTo(v, k.step)));
    k.set(onChrysalis() && Math.abs(v - k.rec()) < k.step / 2 ? undefined : +v.toFixed(6));
    saveSettings();
    sync();
  };
  const typed = () => k.parse ? k.parse(num.value.trim()) : Number(num.value.replace(/[,_\s]/g, "").replace(/^off$/i, "0"));
  range.addEventListener("input", () => commit(k.val(range.valueAsNumber)));
  range.addEventListener("dblclick", () => { k.set(undefined); saveSettings(); sync(true); });
  reset.addEventListener("click", () => { k.set(undefined); saveSettings(); sync(true); });
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
    const v = k.get(), p = k.pos(v), custom = k.custom(), unset = !custom && !onChrysalis();
    el.classList.toggle("custom", custom);
    el.classList.toggle("unset", unset);
    num.placeholder = unset ? "default" : "";
    if (force || document.activeElement !== num) num.value = unset ? "" : k.fmt(v);
    range.value = String(p);
    range.style.setProperty("--fill", `${p * 100}%`);
    rec.style.setProperty("--at", String(k.pos(k.rec())));
    rec.title = reset.title = onChrysalis() ? `Recommended: ${k.fmt(k.rec())}` : "The backend's own value (not sent)";
  }
  knobs.push(sync);
  sync();
  return el;
}
function samplerKnob(key: SamplerKey) {
  const d = SAMPLERS[key], [lo, hi] = d.slider ?? [d.min, d.max];
  return knob({
    id: key, label: d.label, help: d.help, min: d.min, max: d.max, step: d.step, fmt: (v) => fmtValue(key, v),
    parse: (t) => /^off$/i.test(t) ? d.off ?? 0 : /^all$/i.test(t) && d.all !== undefined ? d.all : Number(t.replace(/[,_\s]/g, "")),
    get: () => value(shownMode(), key), rec: () => RECOMMENDED[shownMode()][key], custom: () => settings[shownMode()][key] !== undefined,
    set: (v) => { if (v === undefined) delete settings[shownMode()][key]; else settings[shownMode()][key] = v; },
    pos: (v) => d.all !== undefined && v === d.all ? 1 : Math.min(1, Math.max(0, (v - lo) / (hi - lo))),
    val: (p) => d.all !== undefined && p >= 0.999 ? d.all : lo + p * (hi - lo),
  });
}
function dims(row: HTMLElement, key: SamplerKey, on: (v: number) => boolean) {
  knobs.push(() => row.classList.toggle("off", !on(value(shownMode(), key))));
  return row;
}
function choiceKnob(key: SamplerKey, options: [number, string][]) {
  const d = SAMPLERS[key], row = h("div", "knob seg-knob"), label = h("span", "knob-label", d.label), seg = h("div", "segmented"), reset = h("button", "act knob-reset");
  row.title = d.help;
  seg.setAttribute("role", "radiogroup");
  seg.setAttribute("aria-label", d.label);
  reset.type = "button";
  reset.innerHTML = icon("retry");
  const set = (v: number | undefined) => { if (v === undefined) delete settings[shownMode()][key]; else settings[shownMode()][key] = v; saveSettings(); };
  for (const [v, text] of options) {
    const b = h("button", "", text);
    b.type = "button";
    b.setAttribute("role", "radio");
    b.onclick = () => set(onChrysalis() && v === RECOMMENDED[shownMode()][key] ? undefined : v);
    b.dataset.v = String(v);
    seg.append(b);
  }
  reset.onclick = () => set(undefined);
  row.append(label, seg, reset);
  knobs.push(() => {
    const v = value(shownMode(), key), custom = settings[shownMode()][key] !== undefined;
    row.classList.toggle("custom", custom);
    seg.querySelectorAll<HTMLElement>("button").forEach((b) => b.setAttribute("aria-checked", String((custom || onChrysalis()) && Number(b.dataset.v) === v)));
    reset.title = onChrysalis() ? `Recommended: ${options.find(([x]) => x === RECOMMENDED[shownMode()][key])?.[1]}` : "The backend's own value (not sent)";
  });
  return row;
}
function orderKnob() {
  const row = h("div", "knob order-knob"), label = h("label", "", "Samplers"), text = h("input", "knob-text"), reset = h("button", "act knob-reset");
  row.title = `The order the samplers are applied in, separated by semicolons. Default: ${DEFAULT_STAGES.join(";")}. Leaving one out turns it off.`;
  label.htmlFor = text.id = "knob-samplers";
  text.type = "text";
  text.spellcheck = false;
  text.autocomplete = "off";
  reset.type = "button";
  reset.innerHTML = icon("retry");
  const commit = () => {
    const names = text.value.split(/[;,\s]+/).filter(Boolean), order = names.map((n) => STAGE_ALIASES[n.toLowerCase()]);
    const bad = names.filter((_, i) => !order[i]);
    if (bad.length) return void toast(`Not a sampler: ${bad.join(", ")}. They are ${DEFAULT_STAGES.join(", ")}`);
    const uniq = [...new Set(order)];
    if (!uniq.length || uniq.join() === DEFAULT_STAGES.join()) delete settings[shownMode()].order;
    else settings[shownMode()].order = uniq;
    saveSettings();
  };
  text.addEventListener("change", commit);
  text.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit();
    else if (e.key === "Escape") { e.preventDefault(); text.value = orderOf(shownMode()).join(";"); text.blur(); }
  });
  reset.onclick = () => { delete settings[shownMode()].order; saveSettings(); };
  row.append(label, text, reset);
  knobs.push(() => {
    const custom = !!settings[shownMode()].order;
    row.classList.toggle("custom", custom);
    row.classList.toggle("unset", !custom && !onChrysalis());
    text.placeholder = onChrysalis() ? "" : "default";
    if (document.activeElement !== text) text.value = custom || onChrysalis() ? orderOf(shownMode()).join(";") : "";
    reset.title = onChrysalis() ? `Default: ${DEFAULT_STAGES.join(";")}` : "The backend's own order (not sent)";
  });
  return row;
}

function buildSettings() {
  for (const [m, text] of [["think", "With thinking"], ["fast", "Without thinking"]] as const) {
    const b = h("button", "", text);
    b.type = "button";
    b.setAttribute("role", "radio");
    b.dataset.mode = m;
    b.onclick = () => { settingsMode = m; syncSettings(); };
    $("mode-pick").append(b);
  }
  knobs.push(() => $("mode-pick").querySelectorAll<HTMLElement>("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === shownMode()))));
  const k = samplerKnob;
  $("samplers").replaceChildren(k("temperature"), k("dynatemp_range"), dims(k("dynatemp_exponent"), "dynatemp_range", (v) => v > 0), k("top_k"), k("top_p"),
    k("min_p"), k("xtc_probability"), dims(k("xtc_threshold"), "xtc_probability", (v) => v > 0), k("typical_p"), k("top_n_sigma"), k("min_keep"), orderKnob());
  const dry = (key: SamplerKey) => dims(k(key), "dry_multiplier", (v) => v > 0);
  $("penalties").replaceChildren(k("repeat_last_n"), k("repeat_penalty"), k("presence_penalty"), k("frequency_penalty"), k("dry_multiplier"),
    dry("dry_base"), dry("dry_allowed_length"), dry("dry_penalty_last_n"));
  const miro = (key: SamplerKey) => dims(k(key), "mirostat", (v) => v > 0);
  $("mirostat").replaceChildren(choiceKnob("mirostat", [[0, "Off"], [1, "Mirostat"], [2, "Mirostat 2.0"]]), miro("mirostat_tau"), miro("mirostat_eta"));

  const LO = 256, HI = () => status.context ?? 262144;
  const maxLen = knob({
    id: "max_tokens", label: "Max tokens", help: "The maximum number of tokens per reply; thinking counts toward it.", min: 1, max: 262144, step: 1,
    fmt: (v) => v.toLocaleString("en-US"), get: () => settings.max_tokens ?? MAX_REPLY, rec: () => MAX_REPLY,
    custom: () => settings.max_tokens !== undefined, set: (v) => { settings.max_tokens = v; },
    pos: (v) => Math.min(1, Math.max(0, Math.log(v / LO) / Math.log(HI() / LO))),
    val: (p) => 2 ** Math.round(Math.log2(LO) + p * Math.log2(HI() / LO)),
  });
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

  const FLO = 2000, FHI = 100_000;
  fetchKnob = knob({
    id: "fetchChars", label: "Page length", min: FLO, max: FHI, step: 1000,
    help: "Fetch URL: how much of a page the model gets at a time, in characters (it can ask for the rest).",
    fmt: (v) => v.toLocaleString("en-US"), get: () => settings.fetchChars ?? FETCH_CHARS, rec: () => FETCH_CHARS,
    custom: () => settings.fetchChars !== undefined, set: (v) => { settings.fetchChars = v; },
    pos: (v) => Math.min(1, Math.max(0, Math.log(v / FLO) / Math.log(FHI / FLO))), val: (p) => Math.round(FLO * (FHI / FLO) ** p / 1000) * 1000,
  });
  const voicePromptInput = $<HTMLTextAreaElement>("voice-prompt");
  voicePromptInput.addEventListener("input", () => {
    setVoice({ prompt: voicePromptInput.value === DEFAULT_VOICE_PROMPT ? undefined : voicePromptInput.value });
    saveSettings();
    fit(voicePromptInput);
  });
  $("voice-prompt-reset").onclick = () => {
    const before = settings.voice?.prompt;
    setVoice({ prompt: undefined });
    saveSettings();
    toast("The voice prompt is back to the default", () => { setVoice({ prompt: before }); saveSettings(); });
  };
  const choice = (id: string, label: string, help: string, change: (value: string) => void, reset: () => void) => {
    const row = h("div", "knob voice-choice"), lab = h("label", "", label), select = h("select", "voice-select"), back = h("button", "act knob-reset");
    row.title = help;
    lab.htmlFor = select.id = id;
    back.type = "button";
    back.innerHTML = icon("retry");
    back.title = "Back to the voice service's default";
    select.addEventListener("change", () => { change(select.value); saveSettings(); });
    back.onclick = () => { reset(); saveSettings(); };
    row.append(lab, select, back);
    return { row, select };
  };
  const pick = choice("voice-name", "Voice", "The voice replies are spoken in; one of another voice model switches to it (also in voice mode's dock)",
    (v) => { const [tts, name] = v ? JSON.parse(v) : []; setVoice({ tts, name }); }, () => setVoice({ tts: undefined, name: undefined }));
  knobs.push(() => {
    const s = settings.voice, now = s?.name ? JSON.stringify([s.tts ?? voice?.ready?.stack.tts ?? "", s.name]) : "", key = JSON.stringify([voiceCatalog.tts, now]);
    pick.row.classList.toggle("custom", !!now);
    if (pick.select.dataset.key === key) return;
    pick.select.dataset.key = key;
    const groups: HTMLElement[] = [new Option("Default", "")];
    for (const m of (voiceCatalog.tts ?? []).filter((x) => x.installed)) {
      const by = new Map<string, HTMLOptGroupElement>();
      for (const x of m.voices ?? []) {
        const g = `${m.label} · ${voiceGroup(x)}`;
        if (!by.has(g)) by.set(g, Object.assign(document.createElement("optgroup"), { label: g }));
        by.get(g)!.append(new Option(x.label, JSON.stringify([m.name, x.id])));
      }
      groups.push(...by.values());
    }
    const listed = groups.flatMap((g) => g instanceof HTMLOptionElement ? [g.value] : [...g.querySelectorAll("option")].map((o) => o.value));
    if (now && !listed.includes(now)) groups.push(new Option(s!.name!, now));
    pick.select.replaceChildren(...groups);
    pick.select.value = now;
  });
  const speed = knob({
    id: "voice_speed", label: "Speed", help: "Speed: how fast the voice speaks.", min: 0.5, max: 2, step: 0.05,
    fmt: (v) => `${v.toFixed(2)}×`, get: () => settings.voice?.speed ?? 1, rec: () => 1, custom: () => settings.voice?.speed !== undefined,
    set: (v) => setVoice({ speed: v }), pos: (v) => (v - 0.5) / 1.5, val: (p) => 0.5 + p * 1.5,
  });
  $("voice-knobs").replaceChildren(pick.row, speed);



  systemInput.addEventListener("input", () => {
    settings.system = systemInput.value;
    growSystem();
    saveSettings();
  });
  $("sent-open").onclick = () => void showSent();
  $("sent-close").innerHTML = icon("x");
  $("sent-close").onclick = () => $<HTMLDialogElement>("sent-view").close();
  $("sent-copy").onclick = () => copy($("sent-text").textContent ?? "", $("sent-copy"));
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
    settings = { v: 5, system: DEFAULT_SYSTEM, think: {}, fast: {} };
    saveSettings();
    growSystem();
    toast("Everything is back to the recommended values", () => { settings = before; saveSettings(); growSystem(); });
  };
  measureScrollbar();
  addEventListener("resize", measureScrollbar);
}

async function showSent() {
  const c = conv, said = voice?.c === c, effort = c.effort;
  await filesReady(c.messages);
  const body = request(c, [...c.messages], { effort, ...(said ? { route: true, ...(effort === "off" ? { extra: { chrysalis_voice: true } } : {}) } : {}) });
  const res = await fetch("/api/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).catch(() => null);
  const sent = res?.ok ? (await res.json()).body : null;
  if (!sent) return toast("The web server did not answer");
  const kb = (url: string) => `${Math.round((url.length * 3) / 4 / 1024)} KB`;
  const part = (p: any): string => p.type === "text" ? p.text : p.type === "image_url" ? `[picture, ${kb(p.image_url.url)}]`
    : p.type === "file" ? `[PDF ${p.file.filename}, ${kb(p.file.file_data)}]` : p.type === "video_url" ? `[video, ${kb(p.video_url.url)}]` : JSON.stringify(p);
  const lines = [`model: ${sent.model}${sent.reasoning_effort ? `, thinking ${sent.reasoning_effort}` : ""}`];
  if (sent.tools?.length) lines.push("", `tools (${sent.tools.length}):`, JSON.stringify(sent.tools, null, 2));
  for (const m of sent.messages) {
    lines.push("", `── ${m.role}${m.tool_call_id ? ` (${m.tool_call_id})` : ""}`);
    if (m.reasoning_content) lines.push(`[reasoning] ${m.reasoning_content}`);
    lines.push(typeof m.content === "string" ? m.content : (m.content ?? []).map(part).join(""));
    if (m.tool_calls) lines.push(JSON.stringify(m.tool_calls, null, 2));
  }
  const note = said ? noteFor(c, true) : undefined;
  if (note) lines.push("", "── your next question said aloud, after this note:", `[${note}]`);
  $("sent-about").textContent = (!sent.tools?.length ? "" : onChrysalis() ? "Chrysalis's chat template writes the tools first in the system turn (\"# Tools … <tools> … </tools>\"), " +
    "with how to call them, and the system prompt after them. " : "The backend's chat template writes the tools into the prompt. ") + "Your next message goes after these." +
    (said ? " Each spoken question is first checked, in a conversation of its own beside this one, for what speech recognition misheard (and with tools whether one is needed): the question is sent fixed." : "");
  $("sent-text").textContent = lines.join("\n");
  $<HTMLDialogElement>("sent-view").showModal();
}

function measureScrollbar() {
  document.documentElement.style.setProperty("--sb", `${scroller.offsetWidth - scroller.clientWidth}px`);
}

function growSystem() {
  systemInput.style.height = "auto";
  systemInput.style.height = `${Math.min(systemInput.scrollHeight + 2, innerHeight * 0.4)}px`;
}

function openSettings(sec?: string) {
  settingsMode = modeOf(conv);
  if (!panel.open) panel.showModal();
  showSettingsSec(sec ?? pref("settings-sec", "chat"));
  growSystem();
  void loadCatalog();
}
function showSettingsSec(sec: string) {
  if (!panel.querySelector(`.set-sec[data-sec="${sec}"]`)) sec = "chat";
  for (const b of panel.querySelectorAll<HTMLElement>(".set-nav [data-sec]")) b.setAttribute("aria-current", String(b.dataset.sec === sec));
  for (const el of panel.querySelectorAll<HTMLElement>(".set-sec")) el.hidden = el.dataset.sec !== sec;
  setPref("settings-sec", sec);
  syncSettings();
}

function syncSettings() {
  if (!panel.open) return;
  if (document.activeElement !== systemInput && systemInput.value !== settings.system) systemInput.value = settings.system;
  $("system-reset").classList.toggle("on", settings.system !== DEFAULT_SYSTEM);
  const vp = $<HTMLTextAreaElement>("voice-prompt");
  if (document.activeElement !== vp && vp.value !== voicePrompt()) { vp.value = voicePrompt(); fit(vp); }
  $("voice-prompt-reset").classList.toggle("on", settings.voice?.prompt !== undefined);
  $("artifacts-switch").setAttribute("aria-checked", String(settings.artifacts !== false));
  $("settings").classList.toggle("other-backend", !onChrysalis());
  for (const sync of knobs) sync();
  drawCharts();
}

document.addEventListener("visibilitychange", () => { if (!document.hidden) void refreshStatus(); });

$("new-chat").innerHTML = `${icon("plus")}New chat`;
$("add-cast").innerHTML = icon("personAdd");
$("import").innerHTML = icon("upload");
$("export-all").innerHTML = icon("download");
$("export").innerHTML = icon("download");
$("connect").innerHTML = icon("phone");
$("connect").onclick = openConnect;
$("connect-close").innerHTML = icon("x");
$("connect-close").onclick = () => connectView.close();
connectView.addEventListener("click", (e) => { if (e.target === connectView) connectView.close(); });
$("attach").innerHTML = icon("clip");
$("draft").addEventListener("click", () => void draft());
$("voice").innerHTML = icon("wave");
$("voice").addEventListener("click", () => void openVoice());
$("voice-mute").addEventListener("click", () => {
  if (!voice) return;
  voice.muted = !voice.muted;
  voice.client.mute(voice.muted);
  renderVoice();
});
$("voice-dots").addEventListener("click", () => { if (voice?.reply || voice?.client.playing) cutVoice(true); });
$("voice-pick").addEventListener("click", () => setVoicePop(!voicePop));
$("artifacts-switch").addEventListener("click", () => {
  if (settings.artifacts === false) delete settings.artifacts;
  else settings.artifacts = false;
  saveSettings();
});
$("shelf-close").innerHTML = icon("x");
$("shelf-close").addEventListener("click", closeShelf);
$("shelf-compare").addEventListener("click", () => { shelf.compare = !shelf.compare; shelf.other = undefined; renderShelf(); });
$("shelf").addEventListener("keydown", (e) => {
  if (e.key !== "Escape" || e.defaultPrevented || document.querySelector("[data-editing]")) return;
  e.preventDefault();
  closeShelf();
  $("artifacts-toggle").focus();
});
addEventListener("resize", () => glow());
buildControls();
buildTools();
if (location.protocol === "https:") await checkTrust();
void loadCatalog();
$("to-latest").innerHTML = icon("down");
$("settings-reset").insertAdjacentHTML("afterbegin", icon("retry"));
for (const id of ["system-reset", "voice-prompt-reset"]) $(id).innerHTML = icon("retry");
$("settings-open").innerHTML = icon("gear");
$("settings-open").addEventListener("click", () => openSettings());
$("settings-close").innerHTML = icon("x");
$("settings-close").addEventListener("click", () => panel.close());
panel.addEventListener("click", (e) => { if (e.target === panel) panel.close(); });
panel.addEventListener("close", () => { settingsMode = undefined; });
$("set-nav").addEventListener("click", (e) => {
  const b = (e.target as Element).closest<HTMLElement>("[data-sec]");
  if (b) showSettingsSec(b.dataset.sec!);
});
buildSettings();
restorePane(panes.conversations);
await refreshStatus();
for (const c of Object.values(chats)) {
  const m = [...walk(c.messages)].find((x) => x.role === "assistant" && x.job);
  if (m) void resume(c, m);
  for (const x of walk(c.messages)) if (x.starting) {
    delete x.starting;
    x.error = "The page was left before this reply began: regenerate it";
    save(c);
  }
}
{
  const got = await loadCast();
  for (const p of got?.people ?? []) {
    people.set(p.id, p);
    setFace(p);
  }
  for (const g of got?.groups ?? []) groups.set(g.id, g);
  const orphans = got ? Object.values(chats).filter(castless).map((c) => c.id) : [];
  if (orphans.length) dropChats(orphans);
  renderCast();
}
if (location.hash.length > 1) openConversation(location.hash.slice(1), false);
else render();
(async () => {
  for (;;) {
    await sleep(status.state === "loading" || status.loading ? 250 : 4000);
    if (!document.hidden) await refreshStatus();
  }
})();
