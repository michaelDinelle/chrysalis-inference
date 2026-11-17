import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/common";
import katex from "katex";
import "katex/dist/katex.min.css";
import { marked, type Links, type Token, type Tokens } from "marked";
import { type Attachment, KINDS, attach, dataUrl, fmtBytes, get as getFile, lasts, put as putFile, textBlock } from "./files";
import { type Cand, type Odds, type SamplerKey, type Stage, DEFAULT_STAGES, sample } from "./sampler";
import { type Part, type VoiceAsk, type VoiceInfo, type VoiceReady, Speaker, VoiceClient } from "./voice";
import { type Fix, applyFixes, fixedParts, readHeard } from "./heard";
import { pointOf, shownWords, unitOf } from "./spoken";
import { type Checkpoint, type WireCall, assistantTurn, checkpointText, noted, toolTurn, withText } from "./wire";
import { type Block, type Span, type Version, ARTIFACTS_PROMPT, PAGE_DOC, elementById, elementSpans, artifactOf, infoOf, type Info, markSource, replay, setOpen, withAttr, withBackground,
  withContent, withCopy, withOrder, withPaint, withoutElement, lineDiff } from "./artifacts";
import { LATEX_CSS, latexToHtml } from "./latex";
import logo from "./logo.svg";
import { type Card, type CardData, type Group, type Person, type Voice, cardPng, castPrompt, dropGroup, dropPerson, fill as fillCard, loadCast, readCast, saveGroup, savePerson,
  voiceOf } from "./cast";
import { Ocean } from "./ocean";
import { type Drop, sidebarDrag } from "./drag";
import type { SessionContextBreakdownKey, SessionContextBreakdownSegment } from "./breakdown";

type Effort = "xhigh" | "medium" | "low" | "off";
interface Stats { ttft: number; tps: number; tokens: number; prompt: number; cached?: number; thought?: number; seed?: number; model?: string; ctx?: number }
type CallState = "approval" | "asking" | "running" | "done" | "failed" | "declined" | "stopped";
interface Field { key: string; type: string; title?: string; description?: string; required?: boolean; hidden?: boolean; placeholder?: string; default?: unknown; custom?: boolean; url?: string;
  options?: { value: string; label: string; description?: string }[]; when?: { key: string; op: "eq" | "neq"; value: unknown }[] }
interface Form { title: string; fields: Field[]; error?: string }
interface Call extends WireCall { title?: string; source?: string; readOnly?: boolean; state?: CallState; result?: string; ms?: number; form?: Form; note?: string; asks?: string }
interface Kept extends Checkpoint { at: number; job?: string }
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
  checkpoints?: Kept[];
  base?: number;
  starting?: boolean;
  notes?: string[];
  versions?: Versions;
  voice?: boolean;
  voiceMs?: number;
  interrupted?: boolean; heardTo?: number; cut?: string;
  note?: string;
  heard?: Fix[];
  point?: string; pointAt?: string;
  fix?: boolean;
  sent?: string; oc?: string; ocs?: string[];
  edits?: { id: string; kind: string; src: string }[];
}
interface Conversation { id: string; title: string; named?: boolean; updated: number; effort: Effort; messages: Message[];
  unsynced?: boolean;
  incognito?: true;
  pending?: Record<string, { kind: string; srcs: string[] }>;
  who?: string;
  thread?: { hidden?: true; dir: string; id?: string; model?: string; at?: number; agent?: string; machine?: string; context?: { used?: number; compacted?: boolean; window: number; ceiling: number; auto: boolean; breakdown?: SessionContextBreakdownSegment[] } } }
type Sampler = Partial<Record<SamplerKey, number>> & { order?: Stage[] };
type Mode = "think" | "fast";
interface Settings {
  v?: 5;
  think: Sampler; fast: Sampler; max_tokens?: number; seed?: number; voice?: VoiceSettings;
  artifacts?: false;
  global?: string;
}
interface VoiceSettings { tts?: string; name?: string; stt?: string; speed?: number }
interface Status { state: "online" | "loading" | "offline"; got?: string[]; download?: { model: string; done: number; total: number; rate: number; paused?: boolean }; machine?: string; build?: string; model?: string; loading?: string; progress?: number; context?: number; kv_tokens?: number; cache?: { free_kv_tokens: number };
  check?: boolean; here?: boolean; waiting?: number;
  backend?: Backend; link?: Linking }
interface Linking { linked?: { name: string }; found: { name: string; fp: string }[]; refused?: string;
  pairing?: { name: string; code: string; state: "waiting" | "allowed" | "gone" | "error"; error?: string; theirs?: boolean; confirmed?: boolean } }
interface Backend { url: string; chrysalis: boolean; service: boolean; remote: boolean; name: string;
  listed?: { ok: true; models: { id: string }[] } | { ok: false; error: string } }
interface View {
  el: HTMLElement; think: HTMLDetailsElement; summary: HTMLElement; thought: HTMLElement; body: HTMLElement; foot: HTMLElement; shown?: string; blocks?: Blocks;
  tok?: { reasoning: number; content: number };
  rounds: HTMLElement; calls: HTMLElement; notes: HTMLElement; roundsShown?: string;
}
interface Alt { t: string; p: number; id: number }
interface Tok { t: string; p: number; alts: Alt[] }
interface Live { phase: "waiting" | "thinking" | "writing"; since: number; note: string; loading?: boolean }
interface Run {
  conv: Conversation; m: Message; live: Live;
  at: number; begun: number;
  stopping: boolean; gone?: boolean; ended?: () => void;
  heard?: number; pace?: number; shown?: number; painted?: number;
  usage?: { prompt_tokens: number; completion_tokens: number; cached?: number }; last?: any;
  read?: number; written?: number;
  seed?: number; timing?: { start: number; first?: number; answer?: number; last?: number; gen?: number };
  pick?: Tok;
  frame: number; saved: number;
  asking?: boolean;
}

const MODELS: [string, string, string][] = [
  ["qwen3.8-27b", "Qwen3.8-27B", "Quickest replies"],
  ["qwen3.8-flash-next", "Qwen3.8-Flash-Next", "Fast at any length"],
];
const modelName = (id?: string) => MODELS.find(([m]) => m === id?.replace(/^chrysalis\//, ""))?.[1] ?? id?.split("/").pop()?.replace(/\.gguf$/i, "") ?? "";
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
const OLD_SYSTEMS = ["You are {{MODEL}}, running locally in Chrysalis Inference. There are no tools or files in this chat. Answer directly.", "There are no tools or files in this chat. Answer directly."];
const VOICE_PROMPT = "The user is talking to you by voice; a word may still be misheard, so take what they most likely meant, by " +
  "how it sounds and by the conversation, without remarking on it. Your reply is read aloud, except code blocks and tables, which are " +
  "shown and not read. Talk the way a person does: answer in one or two short plain sentences and stop, with no lists, no summary of what " +
  "you did, no offer of more and no question back unless you cannot go on without the answer (but what they ask you to write, such as " +
  "an essay or a story, in full). Put code and tables in markdown blocks and say what you put there, and do what is asked rather than " +
  "asking whether to. When a question needs counting or logic, work it out loud in a few short steps, then answer.";
const CHECK = "You fix speech recognition errors in what a user says to {{MODEL}}, a voice assistant running locally in Chrysalis Inference. " +
  "For each transcript, write one line per misheard word or phrase: the transcript's exact words => the words that sound like them " +
  "which the user meant (names spelled right, numbers in digits). Read the transcript against the conversation: words that do not fit " +
  "what the user is asking were misheard, names of libraries, products and models as everyday words (\"pie torch\" for \"PyTorch\") " +
  "and everyday words as others that sound alike (\"read it\" for \"write it\", \"by\" for \"buy\"). Fix only what was misheard: " +
  "never rephrase, censor, explain or expand, and keep words that fit and names you have no reason to doubt. Write \"-\" when nothing " +
  "was misheard.";
const VOICE_OFF = "The user is typing again and reads your replies.";
const TITLE_PROMPT = "You are a title generator. Reply with a title of at most six words for the conversation that starts with the user's message below. Reply with the title only.";
const BREAKDOWN_COLOR: Record<SessionContextBreakdownKey, string> = {
  system: "var(--syntax-info)",
  user: "var(--syntax-success)",
  assistant: "var(--syntax-property)",
  tool: "var(--syntax-warning)",
  other: "var(--syntax-comment)",
};
const BREAKDOWN_LABEL: Record<SessionContextBreakdownKey, string> = { system: "System", user: "User", assistant: "Assistant", tool: "Tool Calls", other: "Other" };
const BREAKDOWN_NOTE = 'Approximate breakdown of the tokens in context. "Other" includes tool definitions and overhead.';
const RING = 2 * Math.PI * 6.5;
const DRAFT_FIRST = "Write a first message I might send to open this conversation, as me: one natural, brief line of plain text, a question or a request. Reply with only that message.";
const DRAFT_PROMPT = "Write the next message I would send in this conversation, as me: same voice, same goals, natural and brief. Reply with only that message.";
const fill = (text: string) => text.replaceAll("{{MODEL}}", modelName(current()) || "a Qwen model");
const EFFORTS: Effort[] = ["off", "low", "medium", "xhigh"];
const LEVEL: Record<Effort, [string]> = { off: ["Off"], low: ["Low"], medium: ["Medium"], xhigh: ["Max"] };
const HEAT = ["off", "low", "medium", "max"];
const ICONS = {
  plus: '<path d="M12 5v14M5 12h14"/>',
  palette: '<path d="M12 3a9 9 0 0 0 0 18c.9 0 1.6-.7 1.6-1.6 0-.4-.2-.8-.4-1.1-.3-.3-.4-.7-.4-1.100 0-.9.700-1.6 1.6-1.6H16.300A4.700 4.700 0 0 0 21 10.900C21 6.500 17 3 12 3z"/><circle cx="7.6" cy="11.5" r="1"/><circle cx="10" cy="7.600" r="1"/><circle cx="14.600" cy="7.200" r="1"/>',
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
  filter: '<path d="M3 6h18M7 12h10M10 18h4"/>',
  folderPlus: '<path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H9l2 2.5h7.5A2.5 2.5 0 0 1 21 10v6.5a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 16.5z"/><path d="M12 10.5v6M9 13.5h6"/>',
  pen: '<path d="M12 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-6"/><path d="M17.4 3.6a2 2 0 0 1 3 3L12 15l-4 1 1-4z"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>',
  pin: '<path d="M9 4h6l-1 6 3 3H7l3-3z"/><path d="M12 13v7"/>',
  approval: '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="M9.1 9a3 3 0 0 1 5.82 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
  input: '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
  failed: '<circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
  done: '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
  compose: '<path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.375 2.625a1 1 0 0 1 3 3l-9.013 9.014a2 2 0 0 1-.853.505l-2.873.84a.5.5 0 0 1-.62-.62l.84-2.873a2 2 0 0 1 .506-.852z"/>',
  grip: '<circle cx="9" cy="6" r="1.3" fill="currentColor" stroke="none"/><circle cx="15" cy="6" r="1.3" fill="currentColor" stroke="none"/><circle cx="9" cy="12" r="1.3" fill="currentColor" stroke="none"/><circle cx="15" cy="12" r="1.3" fill="currentColor" stroke="none"/><circle cx="9" cy="18" r="1.3" fill="currentColor" stroke="none"/><circle cx="15" cy="18" r="1.3" fill="currentColor" stroke="none"/>',
  power: '<path d="M12 3.5v8"/><path d="M6.6 6.8a7.5 7.5 0 1 0 10.8 0"/>',
  incognito: '<path d="M9 10h.01"/><path d="M15 10h.01"/><path d="M12 2.5a7.5 7.5 0 0 0-7.5 7.5v11.5l2.8-2.6 2.4 2.6 2.3-2.6 2.3 2.6 2.4-2.6 2.8 2.6V10A7.5 7.5 0 0 0 12 2.5z"/>',
  dice: '<rect x="4" y="4" width="16" height="16" rx="3.5"/><circle cx="9" cy="9" r="1.1" fill="currentColor"/><circle cx="15" cy="15" r="1.1" fill="currentColor"/><circle cx="15" cy="9" r="1.1" fill="currentColor"/><circle cx="9" cy="15" r="1.1" fill="currentColor"/>',
  prev: '<path d="M14.5 6.5 9 12l5.5 5.5"/>',
  next: '<path d="M9.5 6.5 15 12l-5.5 5.5"/>',
  wrench: '<path d="M14.7 6.3a4 4 0 0 0 5 5L21 12.6a6 6 0 0 1-7.4 1.6L7 20.8a2.1 2.1 0 0 1-3-3l6.6-6.6A6 6 0 0 1 12.2 3.8l1.3 1.3a4 4 0 0 0 1.2 1.2z"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.6 3.8 5.6 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.6-3.8-9S9.5 5.6 12 3z"/>',
  fork: '<circle cx="7" cy="5.5" r="2.2"/><circle cx="7" cy="18.5" r="2.2"/><circle cx="17" cy="9" r="2.2"/><path d="M7 7.7v8.6M17 11.2v.3a4 4 0 0 1-4 4H9.2"/>',
  agent: '<circle cx="12" cy="5.5" r="2.2"/><circle cx="6" cy="18.5" r="2.2"/><circle cx="18" cy="18.5" r="2.2"/><path d="M12 7.7v3.8M6 16.3v-1.8a3 3 0 0 1 3-3h6a3 3 0 0 1 3 3v1.8"/>',
  wave: '<path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2"/>',
  play: '<path fill="currentColor" stroke="none" d="M8 5.8v12.4a.8.8 0 0 0 1.2.7l10-6.2a.8.8 0 0 0 0-1.4l-10-6.2a.8.8 0 0 0-1.2.7z"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/>',
  phone: '<rect x="6.5" y="2.5" width="11" height="19" rx="2.5"/><path d="M10.5 18.5h3"/>',
  shapes: '<rect x="3.5" y="10.5" width="10" height="10" rx="2"/><circle cx="15.5" cy="8.5" r="5"/>',
  working: '<path d="M10.1 2.182a10 10 0 0 1 3.8 0"/><path d="M13.9 21.818a10 10 0 0 1-3.8 0"/><path d="M17.609 3.721a10 10 0 0 1 2.69 2.7"/><path d="M2.182 13.9a10 10 0 0 1 0-3.8"/><path d="M20.279 17.609a10 10 0 0 1-2.7 2.69"/><path d="M21.818 10.1a10 10 0 0 1 0 3.8"/><path d="M3.721 6.391a10 10 0 0 1 2.7-2.69"/><path d="M6.391 20.279a10 10 0 0 1-2.69-2.7"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
  page: '<rect x="3" y="4" width="18" height="16" rx="2.5"/><path d="M3 9h18M6.5 6.5h.01M9.5 6.5h.01"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2.5"/><circle cx="9" cy="10" r="1.8"/><path d="m21 16-5-5-9 9"/>',
  doc: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>',
  diagram: '<rect x="3" y="3" width="7" height="6" rx="1.5"/><rect x="14" y="15" width="7" height="6" rx="1.5"/><path d="M6.5 9v4.5a1.5 1.5 0 0 0 1.5 1.5h6"/>',
  code: '<path d="m8 8-4 4 4 4M16 8l4 4-4 4M13.5 5l-3 14"/>',
  speak: '<path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z"/><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/>',
  person: '<circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/>',
  folder: '<path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H9l2 2.5h7.5A2.5 2.5 0 0 1 21 10v6.5a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 16.5z"/>',
  lock: '<rect x="5" y="10.5" width="14" height="10" rx="2"/><path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5"/>',
  monitor: '<rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>',
  mac: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 15h.01M11 15h.01M15 15h.01"/>',
  warn: '<path d="M12 4 2.8 19.5h18.4z"/><path d="M12 10v4.5M12 17.3v.2"/>',
  personAdd: '<circle cx="10" cy="8" r="4"/><path d="M2.5 21c1.4-4 4.2-6 7.5-6 1.6 0 3 .5 4.3 1.3M19 14v6M16 17h6"/>',
  micOff: '<path d="M15 9.4V6a3 3 0 0 0-5.7-1.3M9 9v2a3 3 0 0 0 4.9 2.3M18.5 11a6.5 6.5 0 0 1-1 3.4M5.5 11a6.5 6.5 0 0 0 10.4 5.2M12 17.5V21M4 4l16 16"/>',
};
const TITLES = { copy: "Copy", edit: "Edit", trash: "Delete", tokens: "Token odds", retry: "Regenerate (a new version)", speak: "Read aloud",
  fork: "Fork: a new chat with everything before this message" };
const icon = (name: keyof typeof ICONS) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;
const OC_MARK = '<svg class="oc-mark" viewBox="0 0 16 20" aria-hidden="true"><path class="w" d="M12 16H4V8H12V16Z"/><path class="s" d="M12 4H4V16H12V4ZM16 20H0V0H16V20Z"/></svg>';
const OC_WORD = '<svg class="oc-word" viewBox="0 0 234 42" aria-hidden="true"><path class="w" d="M18 30H6V18H18V30ZM48 30H36V18H48V30ZM84 24V30H66V24H84ZM108 36H96V18H108V36ZM144 30H126V18H144V30ZM168 30H156V18H168V30ZM198 30H186V18H198V30ZM234 24V30H216V24H234Z"/>'
  + '<path class="b" d="M18 12H6V30H18V12ZM24 36H0V6H24V36ZM36 30H48V12H36V30ZM54 36H36V42H30V6H54V36ZM84 24H66V30H84V36H60V6H84V24ZM66 18H78V12H66V18ZM108 12H96V36H90V6H108V12ZM114 36H108V12H114V36Z"/>'
  + '<path class="s" d="M144 12H126V30H144V36H120V6H144V12ZM168 12H156V30H168V12ZM174 36H150V6H174V36ZM198 12H186V30H198V12ZM204 36H180V6H198V0H204V36ZM216 12V18H228V12H216ZM234 24H216V30H234V36H210V6H234V24Z"/></svg>';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const thread = $("thread"), scroller = $("scroller"), input = $<HTMLTextAreaElement>("input"), sendBtn = $<HTMLButtonElement>("send");
const panel = $<HTMLDialogElement>("settings");
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

const castReady = Promise.withResolvers<void>();
const people = new Map<string, Person>(), groups = new Map<string, Group>(), faces = new Map<string, string>();
const ME = "me", CHRYSALIS = "chrysalis";
people.set(CHRYSALIS, { id: CHRYSALIS, kind: "assistant", card: { spec: "chara_card_v2", spec_version: "2.0", data: { name: "Chrysalis" } }, added: 0 });
people.set(ME, { id: ME, kind: "persona", card: { spec: "chara_card_v2", spec_version: "2.0", data: { name: "User", description: "" } }, added: 0 });
let talking = 0;
const persona = () => people.get(ME)!;
const userName = () => nameOf(ME);
const userCard = () => ({ ...persona().card.data, name: userName() });
const CAST = "cast:";
interface Scene { cast: string[]; cards?: Card[]; group?: string; scenario?: string; name?: string; greeting?: string; greeter?: string }
interface Harness { state: "absent" | "off" | "on"; outside?: boolean; machines?: { id: string; name: string; home: string; kind: string }[]; name?: string; version?: string; install?: string; wanted?: boolean; home?: string; models?: { id: string; name: string; provider: string; variants: string[]; straight?: string }[]; agents?: { id: string; name: string }[];
  threads?: { id: string; title: string; dir: string; updated: number; idle: number; busy: boolean; archived: boolean; job?: string; model?: string; scene?: Scene; machine?: string; hidden?: boolean }[] }
let harness: Harness = { state: "off" };
const castable = (id: string) => !!people.get(id) && people.get(id)!.kind !== "persona";
const folderOf = (c: Conversation) => (c.who && c.who !== CHRYSALIS && (groups.has(c.who) || people.has(c.who)) ? c.who : "");
const DIR = "dir:", STUB = "oc-";
const machineOf = (id?: string) => harness.machines?.find((m) => m.id === (id ?? ""));
const homeOf = (id?: string) => machineOf(id)?.home ?? harness.home;
const dirMachines = (): Record<string, string> => { try { return JSON.parse(pref("dir-machines", "{}")); } catch { return {}; } };
const dirMachine = (dir: string) => harness.threads?.find((t) => t.dir === dir)?.machine ?? (machineOf(dirMachines()[dir]) ? dirMachines()[dir] : undefined);
const onMachine = (dir: string, machine?: string) => {
  const all = dirMachines();
  if (machine) all[dir] = machine;
  else delete all[dir];
  setPref("dir-machines", JSON.stringify(all));
};
const freshThread = () => {
  const m = machineOf(pref("machine", "")) ?? machineOf("");
  return { dir: m?.home ?? harness.home!, ...(m?.id ? { machine: m.id } : {}) };
};
const inDir = (dir: string) => ({ dir, ...(dirMachine(dir) ? { machine: dirMachine(dir) } : {}) });
const whoOf = (c: Conversation) => folderOf(c) || (c.thread && c.thread.dir !== homeOf(c.thread.machine) ? DIR + c.thread.dir : CHRYSALIS);
const membersOf = (g: Group) => g.members.map((m) => people.get(m)).filter((p): p is Person => !!p);
function castOf(c: Conversation): { members: Person[]; group?: Group } {
  const id = folderOf(c), group = groups.get(id);
  return group ? { members: membersOf(group), group } : id ? { members: [people.get(id)!] } : { members: [] };
}
const chatsOf = (id: string) => [...Object.values(chats), ...stubs()].filter((s) => !s.incognito && whoOf(s) === id).sort((a, b) => b.updated - a.updated);
const hue = (s: string) => [...s].reduce((a, ch) => (a * 31 + ch.charCodeAt(0)) % 360, 7);
const tinted = (h: number) => {
  const a = 0.55 * Math.min(0.42, 0.58), f = (n: number) => Math.round(255 * (0.42 - a * Math.max(-1, Math.min((n + h / 30) % 12 - 3, 9 - (n + h / 30) % 12, 1)))).toString(16).padStart(2, "0");
  return `#${f(0)}${f(8)}${f(4)}`;
};
const own = (p: Person) => ((p.card.data.extensions ??= {}).chrysalis ??= {}) as { voice?: Voice; color?: string; hue?: number };
const tintOf = (p: Person) => { const x = p.card.data.extensions?.chrysalis as { color?: string; hue?: number } | undefined; return x?.color ?? tinted(x?.hue ?? hue(p.card.data.name.trim() || "?")); };
function face(id: string): HTMLElement {
  if (groups.has(id)) {
    const g = groups.get(id), ms = (g ? membersOf(g).map((m) => face(m.id)) : []).slice(0, 4), s = h("span", `face mosaic m${Math.max(ms.length, 1)}`);
    s.append(...(ms.length ? ms : [initials("?")]));
    return s;
  }
  if (id.startsWith(DIR)) {
    const s = h("span", "face dir");
    s.innerHTML = icon("folder");
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
  return initials(name, p && tintOf(p));
}
function initials(name: string, tint = tinted(hue(name))) {
  const s = h("span", "face initials", name.slice(0, 1).toUpperCase());
  s.style.setProperty("--tint", tint);
  return s;
}
function setFace(p: Person) {
  const was = faces.get(p.id);
  if (was) URL.revokeObjectURL(was);
  if (p.image) faces.set(p.id, URL.createObjectURL(p.image));
  else faces.delete(p.id);
}
const andList = (n: string[]) => (n.length < 2 ? n.join("") : `${n.slice(0, -1).join(", ")} and ${n.at(-1)}`);
const names = (ids: string[]) => andList(ids.map(nameOf));
const castNames = (g: Group) => names(membersOf(g).map((m) => m.id));
function nameOf(id: string): string {
  if (id.startsWith(DIR)) return dirName(id.slice(DIR.length));
  const g = groups.get(id);
  return g ? g.name?.trim() || castNames(g) : people.get(id)?.card.data.name.trim() || (id === CHRYSALIS ? "Chrysalis" : id === ME ? "User" : "");
}

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
  const g = groups.has(id), d = id.startsWith(DIR), name = nameOf(id);
  const row = h("div", "cast-row");
  const pic = h("button", "cast-pic"), main = h("button", "cast-main"), add = h("button", "act cast-new");
  pic.type = main.type = add.type = "button";
  pic.title = pic.ariaLabel = d ? id.slice(DIR.length) : `${name || "Unnamed"}'s card`;
  pic.append(face(id));
  if (g) membersOf(groups.get(id)!).slice(0, 4).forEach((m, k) => {
    const tile = pic.firstElementChild!.children[k] as HTMLElement;
    tile.dataset.person = m.id;
    tile.dataset.group = id;
    tile.title = m.card.data.name.trim() || nameOf(m.id);
  });
  if (d) pic.onclick = () => main.click();
  else pic.dataset.card = id;
  if (renaming === id) {
    const box = h("input", "conv-rename");
    box.value = g ? groups.get(id)!.name ?? "" : name;
    box.placeholder = g ? castNames(groups.get(id)!) : "Their name";
    box.ariaLabel = "Name";
    box.onclick = (e) => e.stopPropagation();
    box.onkeydown = (e) => { if (e.key === "Enter") { renaming = ""; renameRow(id, box.value); } else if (e.key === "Escape") { renaming = ""; renderList(); } e.stopPropagation(); };
    box.onblur = () => { if (renaming === id) { renaming = ""; renameRow(id, box.value); } };
    queueMicrotask(() => { box.focus(); box.select(); });
    main.append(box);
  } else main.append(h("span", "cast-name", name || "Unnamed"));
  if (kept("pinned").includes(id)) main.insertAdjacentHTML("beforeend", `<span class="conv-pin">${icon("pin")}</span>`);
  if (list.length) {
    main.setAttribute("aria-expanded", String(open));
    const needs = open ? undefined : ["approval", "input", "failed", "working", "done"].find((k) => list.some((x) => statusOf(x)?.kind === k));
    if (needs) main.append(Object.assign(h("span", `cast-dot ${needs}`), { title: STATUS[needs as keyof typeof STATUS][0] }));
    if (!open) main.append(h("span", "cast-count", String(list.length)));
    main.insertAdjacentHTML("beforeend", icon("chevron"));
  }
  main.onclick = () => {
    if (renaming === id) return;
    if (!list.length || whoOf(conv) !== id) return newChat(id);
    unfold(id, !open);
    renderList();
  };
  add.title = add.ariaLabel = d ? `New thread in ${name}` : `New chat with ${name}`;
  add.innerHTML = icon("compose");
  add.onclick = () => newChat(id);
  row.append(pic, main, add);
  menuOn(row, () => menuOf(id));
  if (g || d || castable(id)) {
    const bring = h("button", "act cast-bring"), here = whoOf(conv) === id;
    bring.type = "button";
    bring.title = bring.ariaLabel = d ? (here ? "Who works in this thread" : `A new thread in ${name} with a character`) : g ? `Who is in ${name}` : here ? "Bring someone into this chat" : `A new group with ${name} and others`;
    bring.innerHTML = icon("personAdd");
    bring.onclick = () => ($("cast-invite").hidden ? openInvite(bring, here && !g ? null : id) : closeInvite());
    row.classList.toggle("here", here);
    row.classList.toggle("open", here && !(open && list.some((c) => c.id === conv.id)));
    row.append(bring);
  }
  if (g || castable(id)) row.dataset.person = id;
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
  const base = members.length > 1 ? "Write the next reply in a fictional scene between {{user}} and these characters."
    : "Write {{char}}'s next reply in a fictional chat between {{char}} and {{user}}.";
  return castPrompt(base, members.map((p) => p.card.data), userCard(), group?.scenario);
}

function stow(c: Conversation) {
  if (c.messages.length) return save(c);
  delete chats[c.id];
  keepChats();
  renderList();
}
const byName = (name: string) => [...people.values()].find((p) => p.kind !== "persona" && nameOf(p.id) === name.trim());
const keepGroup = (g: Group) => { groups.set(g.id, g); void saveGroup(g); };
function placeRow(id: string, beside: string, after = false) {
  const rows = kept("rows").filter((r) => r !== id), at = rows.indexOf(beside);
  if (at < 0) return;
  rows.splice(at + Number(after), 0, id);
  setPref("rows", JSON.stringify(rows));
}
function makeGroup(members: string[], more: Partial<Group> = {}, beside = members[0]): Group {
  const g: Group = { id: CAST + randomId(), members, added: Date.now(), ...more };
  keepGroup(g);
  placeRow(g.id, beside);
  unfold(g.id);
  return g;
}
function meetCard(card: Card, image?: Blob): Person {
  const had = byName(card.data.name);
  if (had) return had;
  const p: Person = { id: randomId(), kind: "character", card, ...(image ? { image } : {}), added: Date.now() };
  people.set(p.id, p);
  setFace(p);
  void savePerson(p);
  return p;
}
const called = (g: Group) => g.name?.trim() || "the group";
function setMembers(g: Group, members: string[], said: string) {
  const before = g.members;
  g.members = members;
  keepGroup(g);
  toast(said, () => { g.members = before; keepGroup(g); renderCast(); render(); });
  renderCast();
  render();
}

async function adopt(f: File): Promise<string | null> {
  const got = await readCast(f).catch(() => null);
  if (!got) return null;
  if ("group" in got) {
    for (const card of got.group.cards) if (got.group.members.includes(card.data.name)) meetCard(card);
    const found = got.group.members.map((n) => byName(n)?.id), members = found.filter((x): x is string => !!x), absent = got.group.members.filter((_, i) => !found[i]);
    cardless.push(...absent);
    if (!members.length) return "";
    const same = [...groups.values()].find((g) => nameOf(g.id) === got.group.name && g.members.length === members.length && members.every((m) => g.members.includes(m)));
    if (same) return same.id;
    const g = makeGroup(members, got.group.scenario ? { scenario: got.group.scenario } : {});
    if (got.group.name !== castNames(g)) { g.name = got.group.name; keepGroup(g); }
    return g.id;
  }
  const d = got.card.data, had = [...people.values()].find((p) => p.kind === "character" && p.card.data.name === d.name
    && (p.card.data.description ?? "") === (d.description ?? ""));
  if (had) return had.id;
  const p: Person = { id: randomId(), kind: "character", card: got.card, image: got.image, added: Date.now() };
  people.set(p.id, p);
  setFace(p);
  await savePerson(p);
  return p.id;
}
const cardless: string[] = [];
async function welcome(files: Iterable<File>, into: boolean): Promise<File[]> {
  const had = new Set([...people.keys(), ...groups.keys()]), ids: string[] = [], rest: File[] = [];
  for (const f of files) {
    const id = await adopt(f);
    if (id) ids.push(id); else if (id === null) rest.push(f);
  }
  const none = cardless.splice(0).filter((n) => !byName(n)), lacking = none.length ? `; no card here for ${andList([...new Set(none)])}` : "";
  const them = ids.filter((id) => people.has(id)), fresh = ids.filter((id) => !had.has(id));
  if (into && them.length && folderOf(conv)) join(them);
  else if (ids.length) {
    pickFolder(ids[0]);
    renderCast();
    if (fresh.length) toast(`${names(fresh)} ${fresh.length > 1 ? "are" : "is"} here${lacking}`, () => { for (const id of fresh) void forget(id).then((f) => f.done()); });
    else if (lacking) toast(`No card here for ${andList([...new Set(none)])}`);
  } else if (lacking) toast(`No card here for ${andList([...new Set(none)])}`);
  return rest;
}
function join(ids: string[]) {
  const c = conv, { members: named, group } = castOf(c), members = named.length || !c.messages.length ? named : [people.get(CHRYSALIS)!];
  const fresh = [...new Set(ids)].filter((id) => !members.some((m) => m.id === id));
  if (!fresh.length) return toast(`${names(ids)} ${ids.length > 1 ? "are" : "is"} in this chat already`);
  if (group) return setMembers(group, [...group.members, ...fresh], `${names(fresh)} joined ${called(group)}`);
  const was = c.who, had = new Set(groups.keys());
  if (!members.length && fresh.length === 1) c.who = fresh[0];
  else {
    for (const m of c.messages) if (m.role === "assistant" && members.length) m.as ??= members[0].id;
    const all = [...members.map((m) => m.id), ...fresh];
    c.who = (groupOf(all) ?? makeGroup(all)).id;
  }
  const now = c.who, made = !had.has(now);
  unfold(now);
  stow(c);
  toast(groups.has(now) ? `${names(fresh)} joined: a group now` : `This chat is ${names(fresh)}'s now`, () => {
    if (was) c.who = was; else delete c.who;
    if (made && groups.delete(now)) void dropGroup(now);
    stow(c);
    renderCast();
    render();
  });
  renderCast();
  render();
}
const canLeave = (g: Group) => g.members.length > 2 || !everyChat().some((c) => c.who === g.id);
function takeOut(g: Group, id: string) {
  if (g.members.length > 2) setMembers(g, g.members.filter((m) => m !== id), `${nameOf(id)} left ${called(g)}`);
  else if (canLeave(g)) void drop(g.id);
}
const groupOf = (ids: string[]) => [...groups.values()].find((g) => g.members.length === ids.length && ids.every((m) => g.members.includes(m)));
function together(a: string, b: string) {
  const g = groups.get(b), theirs = groupOf([a, b]);
  if (g) return setMembers(g, [...g.members, a], `${nameOf(a)} joined ${called(g)}`);
  if (theirs) return pickFolder(theirs.id);
  const made = makeGroup([b, a]);
  renderCast();
  toast(`${nameOf(made.id)} are a group`, () => void forget(made.id).then((f) => f.done()));
}
const doomed = new Set<string>();
async function forget(id: string) {
  const p = people.get(id), name = nameOf(id), held = [...groups.values()].filter((g) => g.id === id || g.members.includes(id)).map((g) => ({ g, members: g.members }));
  const out = new Set([id, ...held.filter(({ g }) => g.id === id || g.members.length === 1).map(({ g }) => g.id)]);
  const theirs = (c: Conversation) => out.has(c.who ?? "") || (id === CHRYSALIS && whoOf(c) === CHRYSALIS);
  const gone = [...new Set([...Object.values(chats), conv])].filter(theirs), threads = [...new Set(everyChat().filter((c) => c.thread?.id && theirs(c)).map((c) => c.thread!.id!))];
  const n = new Set([...gone.filter((c) => c.messages.length || c.thread?.id).map(keyOf), ...threads.map((t) => STUB + t)]).size;
  if (p) { people.delete(id); await dropPerson(id); }
  if (id === CHRYSALIS) {
    people.set(CHRYSALIS, { id: CHRYSALIS, kind: "assistant", card: { spec: "chara_card_v2", spec_version: "2.0", data: { name: "Chrysalis" } }, added: 0 });
    setFace(people.get(CHRYSALIS)!);
    setPref("chrysalis-gone", "1");
  }
  for (const { g } of held) {
    if (out.has(g.id)) { groups.delete(g.id); void dropGroup(g.id); }
    else { g.members = g.members.filter((m) => m !== id); void saveGroup(g); }
  }
  for (const t of threads) doomed.add(t);
  dropChats(gone.map((c) => c.id));
  renderCast();
  return { name, chats: n, undo: async () => {
    if (p) { people.set(id, p); setFace(p); await savePerson(p); }
    if (id === CHRYSALIS) setPref("chrysalis-gone", "");
    for (const { g, members } of held) { g.members = members; keepGroup(g); }
    for (const t of threads) doomed.delete(t);
    for (const c of gone) if (c.messages.length) chats[c.id] = c;
    keepChats();
    renderCast();
    render();
  }, done: () => {
    if (!threads.length) return;
    harness.threads = harness.threads?.filter((t) => !threads.includes(t.id));
    void Promise.all(threads.map((t) => fetch(`/api/threads/${t}`, { method: "DELETE", keepalive: true }).catch(() => null))).then(() => loadThreads()).then(() => { for (const t of threads) doomed.delete(t); });
  } };
}
async function drop(id: string) {
  const f = await forget(id);
  toast(`${f.name || "Unnamed"} deleted${f.chats ? `, with ${f.chats} chat${f.chats > 1 ? "s" : ""}` : ""}`, () => void f.undo(), f.done);
}
function dropChats(ids: string[]) {
  for (const id of ids) {
    const r = runs.get(id);
    if (r) { r.gone = true; stopRun(r); }
    delete chats[id];
  }
  keepChats();
  if (ids.includes(conv.id) || (conv.who && !folderOf(conv))) newChat();
  else render();
}
const castless = (c: Conversation) => !c.thread && !!c.who && c.who !== CHRYSALIS && !people.has(c.who) && !groups.has(c.who);

const dirName = (dir: string) => dir.replace(/\/+$/, "").split("/").pop() || dir;
const kept = (key: string): string[] => { try { const v = JSON.parse(pref(key, "[]")); return Array.isArray(v) ? v : []; } catch { return []; } };
const dirs = () => {
  if (harness.state !== "on") return [];
  const closed = new Set(kept("closed-folders"));
  return [...new Set([...kept("folders"), ...[...Object.values(chats), conv].flatMap((c) => (c.thread ? [c.thread.dir] : [])), ...(harness.threads ?? []).map((t) => t.dir)])]
    .filter((d) => d !== harness.home && !harness.machines?.some((m) => m.home === d) && !closed.has(d));
};
function closeFolder(dir: string, shut = true) {
  setPref("closed-folders", JSON.stringify(shut ? [...new Set([...kept("closed-folders"), dir])] : kept("closed-folders").filter((d) => d !== dir)));
  if (shut && conv.thread?.dir === dir) newChat();
  renderList();
  if (shut) toast(`${dirName(dir)} is closed`, () => closeFolder(dir, false));
}
function seated(scene?: Scene): Partial<Conversation> {
  if (!scene?.cast.length) return {};
  if (scene.group && groups.has(scene.group)) return { who: scene.group };
  const found = scene.cast.map((n) => byName(n)?.id);
  if (found.some((id) => !id)) return {};
  if (found.length === 1 && !scene.group) return { who: found[0] };
  const same = [...groups.values()].find((g) => g.members.length === found.length && found.every((id) => g.members.includes(id!)));
  return same ? { who: same.id } : {};
}
function meet(scene?: Scene) {
  if (!scene?.cast.length) return;
  for (const card of scene.cards ?? []) if (scene.cast.includes(card.data.name.trim())) meetCard(card);
  const members = scene.cast.map((n) => byName(n)?.id).filter((id): id is string => !!id);
  if ((scene.group || scene.cast.length > 1) && members.length && !seated(scene).who)
    keepGroup({ id: scene.group ?? CAST + randomId(), members, added: Date.now(), ...(scene.name ? { name: scene.name } : {}), ...(scene.scenario ? { scenario: scene.scenario } : {}) });
  const who = seated(scene).who;
  if (who) unfold(who);
}
const stubs = (): Conversation[] => { const mine = new Set([...Object.values(chats), conv, ...[...runs.values()].map((r) => r.conv)].map((c) => c.thread?.id)); return (harness.threads ?? []).filter((t) => !mine.has(t.id) && !doomed.has(t.id))
  .map((t) => ({ id: STUB + t.id, title: t.title || "Thread", named: true, updated: t.updated, effort: "xhigh" as Effort, messages: [], thread: { dir: t.dir, id: t.id, ...(t.machine ? { machine: t.machine } : {}), ...(t.hidden ? { hidden: true as const } : {}) }, ...(t.hidden ? { incognito: true as const } : {}), ...seated(t.scene) })); };
const keyOf = (c: Conversation) => (c.thread?.id ? STUB + c.thread.id : c.id);
const threadModel = (c: Conversation) => [c.thread?.model, pref("thread-model", "")].find((id) => id && harness.models?.some((m) => m.id === id))
  ?? harness.models?.find((m) => m.id === `chrysalis/${current()}`)?.id ?? harness.models?.find((m) => m.id.startsWith("chrysalis/"))?.id;
function base64(text: string) {
  const bytes = new TextEncoder().encode(text);
  let out = "";
  for (let i = 0; i < bytes.length; i += 0x8000) out += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(out);
}
const VARIANTS: Record<Effort, string[]> = { off: ["none", "off", "minimal"], low: ["low", "minimal"], medium: ["medium"], xhigh: [] };
const variantsOf = (c: Conversation) => harness.models?.find((m) => m.id === threadModel(c))?.variants ?? [];
const threadVariant = (c: Conversation) => VARIANTS[c.effort].find((v) => variantsOf(c).includes(v));
const greeted = (c: Conversation) => { const at = c.messages.findIndex((m) => m.role === "user"); return (at < 0 ? c.messages : c.messages.slice(0, at)).find((m) => m.role === "assistant" && !m.job)?.content; };
function threadPersona(c: Conversation) {
  const { members, group } = castOf(c), opened = greeted(c);
  if (!members.length) return [fill(castPrompt("", [people.get(CHRYSALIS)!.card.data], userCard())),
    opened ? `The chat opened with this greeting from you:\n${opened}` : "", globalPrompt()].filter(Boolean).join("\n\n");
  return [castSystem(c), `Stay in character in everything you say. ${group ? "Whoever is speaking" : members[0].card.data.name} can use the tools here to do what ${userName()} asks, for real.`,
    opened ? `The chat opened with this greeting:\n${opened}` : "", globalPrompt()].filter(Boolean).join("\n\n");
}
function sceneFor(c: Conversation): Scene | undefined {
  const { members, group } = castOf(c), opened = greeted(c), first = c.messages.find((m) => m.role === "assistant" && !m.job);
  const greeter = opened ? (people.get(first?.as ?? "") ?? (group ? undefined : members[0]))?.card.data.name.trim() : undefined;
  return members.length || opened ? { cast: members.map((p) => p.card.data.name.trim()), ...(members.length ? { cards: members.map((p) => p.card) } : {}), ...(group ? { group: group.id } : {}),
    ...(group?.scenario ? { scenario: group.scenario } : {}), ...(group?.name?.trim() ? { name: group.name.trim() } : {}), ...(opened ? { greeting: opened } : {}), ...(greeter ? { greeter } : {}) } : undefined;
}
let harnessAt = 0;
async function loadThreads() {
  const at = harnessAt, r = await fetch("/api/threads").catch(() => null);
  if (!r?.ok) return;
  const text = await r.text();
  if (at !== harnessAt) return;
  if (text === JSON.stringify(harness)) return;
  const before = new Map((harness.threads ?? []).map((t) => [t.id, t.idle]));
  harness = JSON.parse(text);
  for (const t of harness.threads ?? []) if (!t.hidden && before.has(t.id) && before.get(t.id) !== t.idle && !t.job && conv.thread?.id !== t.id && ![...runs.values()].some((x) => x.conv.thread?.id === t.id))
    mark({ id: STUB + t.id, thread: { dir: t.dir, id: t.id, ...(t.machine ? { machine: t.machine } : {}) } } as Conversation, "done");
  const now = (c: Conversation) => (c.thread?.id ? harness.threads?.find((x) => x.id === c.thread!.id) : undefined);
  if (harness.state === "on" && harness.threads) {
    const gone = Object.values(chats).filter((c) => c.thread?.id && !c.incognito && !now(c) && !runs.get(c.id) && c.id !== conv.id);
    for (const c of gone) delete chats[c.id];
    if (conv.thread?.id && !conv.incognito && !now(conv) && !runs.get(conv.id)) { delete chats[conv.id]; newChat(); }
  }
  for (const c of new Set([...Object.values(chats), conv])) {
    const t = c.thread?.id ? harness.threads?.find((x) => x.id === c.thread!.id) : undefined;
    if (t?.title && c.title !== t.title && !runs.get(c.id)) { c.title = t.title; c.named = true; }
  }
  keepChats();
  paintHarness();
  renderList();
  if (conv.thread) { $("title").textContent = conv.title; renderControls(); }
  if (harness.home && !conv.thread && !conv.messages.some((m) => m.role === "user") && !runs.get(conv.id) && pref("tools", "on") === "on") { conv.thread = conv.incognito ? hiddenThread() : freshThread(); renderControls(); }
  const t = runs.get(conv.id) ? undefined : now(conv);
  if (t && !voice && (t.job ? !conv.messages.some((m) => m.job === t.job) : t.idle !== (conv.thread!.at ?? t.idle))) showThread(t.id);
  if (harness.state === "off" && harness.wanted) void ensureHarness().catch(() => { harness.wanted = false; if (!$("oc-pop").hidden && !ocAt) void paintOcPop(); });
}
async function sentFile(key: string, f: { name: string; type: string; data: string }, keep: boolean): Promise<Attachment> {
  const bytes = Uint8Array.from(atob(f.data), (ch) => ch.charCodeAt(0)), kind = f.type.startsWith("image/") ? "image" : f.type === "application/pdf" ? "pdf" : "text";
  await putFile(key, kind === "text" ? new TextDecoder().decode(bytes) : new Blob([bytes], { type: f.type }), keep);
  return { id: key, name: f.name, type: f.type, size: bytes.length, kind };
}
async function openThread(sid: string): Promise<string | undefined> {
  const held = () => Object.values(chats).find((c) => c.thread?.id === sid);
  if (held() && runs.get(held()!.id)) return held()!.id;
  const r = await fetch(`/api/threads/${sid}`).catch(() => null), got = r?.ok ? await r.json() : null;
  if (!got) { if (!held()) toast("OpenCode could not give that thread"); return held()?.id; }
  await castReady.promise;
  const mine = held();
  meet(got.thread.scene);
  if (mine && !folderOf(mine)) Object.assign(mine, seated(got.thread.scene));
  const c: Conversation = mine ?? { id: STUB + sid, title: "", named: true, updated: 0, effort: pref("effort", "xhigh") as Effort, messages: [], thread: { dir: got.thread.dir, id: sid, ...(got.thread.machine ? { machine: got.thread.machine } : {}) }, ...seated(got.thread.scene) };
  if (runs.get(c.id)) return c.id;
  if (got.thread.hidden) { c.incognito = true; c.thread!.hidden = true; }
  c.thread!.at = got.thread.idle;
  c.thread!.dir = got.thread.dir;
  if (got.context) c.thread!.context = got.context;
  const said: Message[] = await Promise.all(got.messages.map(async ({ files, by, ...m }: Omit<Message, "files"> & { by?: string; files?: { name: string; type: string; data: string }[] }, i: number) => ({ id: randomId(), ...m,
    ...(by && byName(by) ? { as: byName(by)!.id } : {}),
    ...(files ? { files: await Promise.all(files.map((f, k) => sentFile(`${sid}-${i}-${k}`, f, !got.thread.hidden))) } : {}) })));
  if (got.job) {
    if (said.at(-1)?.role === "assistant") said.pop();
    said.push({ id: randomId(), role: "assistant", content: "", reasoning: "", job: got.job, ...(people.has(c.who ?? "") ? { as: c.who } : {}) });
  }
  const old = mine?.messages ?? [], was = old.at(-1), tail = said.at(-1);
  if (mine && was?.versions && was.role === "assistant" && tail?.role === "assistant" && !tail.job && said.length === old.length)
    said[said.length - 1] = mine.unsynced || tail.content.trim() === was.content.trim() ? { ...was, oc: tail.oc, ocs: tail.ocs } : { ...tail, versions: was.versions };
  else if (mine) delete mine.unsynced;
  said.forEach((m, i) => {
    const was = old.find((o) => o.oc && o.oc === m.oc) ?? (old[i]?.role === m.role && old[i]?.content === m.content ? old[i] : undefined);
    if (was?.stats && !m.stats) m.stats = was.stats;
  });
  Object.assign(c, { title: got.thread.title || c.title || "Thread", updated: got.thread.updated || c.updated, messages: said });
  chats[c.id] = c;
  keepChats();
  return c.id;
}
async function syncThread(c: Conversation, ended = false) {
  const sid = c.thread?.id, last = c.messages.at(-1);
  if (!sid || runs.get(c.id) || (!ended && (last?.error || last?.stopped))) return;
  const r = await fetch(`/api/threads/${sid}`).catch(() => null), got = r?.ok ? await r.json() : null;
  if (!got || got.job || runs.get(c.id) || (!chats[c.id] && c !== conv)) return;
  const theirs: Message[] = got.messages;
  if (theirs.length !== c.messages.length || theirs.some((m, i) => m.role !== c.messages[i].role)) { if (c === conv && !voice) showThread(sid); return; }
  theirs.forEach((m, i) => { if (m.oc) c.messages[i].oc = m.oc; });
  c.thread!.at = got.thread.idle;
  save(c);
}
async function threadAct(act: "fork" | "drop" | "edit" | "resend", m: Message, text = m.content) {
  const c = conv, sid = c.thread?.id;
  if (!sid || running(c)) return;
  if (!m.oc) await syncThread(c, true);
  if (c !== conv) return;
  const at = conv.messages.findIndex((x) => x === m || (x.oc && x.oc === m.oc) || (!m.oc && x.role === m.role && x.content === m.content && !!x.oc)), now = conv.messages[at];
  const asked = now?.role === "user" ? now : conv.messages.slice(0, Math.max(at, 0)).findLast((x) => x.role === "user");
  const target = act === "resend" ? asked : now;
  if (act === "drop" && now && !now.oc) {
    conv.messages.splice(at, 1);
    save(conv);
    return render();
  }
  if (!target?.oc) return toast("OpenCode does not have this message yet");
  const [path, body] = act === "fork" ? ["fork", { before: target.oc }] : act === "resend" ? ["undo", { message: target.oc }] : ["rewrite", act === "drop" ? { drop: now.ocs ?? [now.oc] } : { edit: { id: now.oc, text } }];
  const r = await fetch(`/api/threads/${sid}/${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).catch(() => null);
  if (!r?.ok) return toast((await r?.json().catch(() => null))?.error?.message ?? "OpenCode did not do that");
  if (act === "fork") {
    const { id } = await r.json();
    await loadThreads();
    return void (location.hash = `#${STUB}${id}`);
  }
  const files = act === "resend" ? asked!.files : undefined;
  await showThread(sid);
  if (act !== "resend" || c.id !== conv.id) return;
  if (files?.length) pending = files;
  submit(now.role === "user" ? text : asked!.content);
}
function showThread(sid: string) {
  const was = conv;
  return openThread(sid).then((at) => {
    if (!at || conv !== was || runs.get(at)) return renderList();
    conv = structuredClone(chats[at]);
    if (at !== location.hash.slice(1)) history.replaceState(null, "", `#${at}`);
    render();
    scroller.scrollTop = scroller.scrollHeight;
    const m = conv.messages.find((x) => x.role === "assistant" && x.job);
    if (m) { void resume(conv, m); renderComposer(); renderList(); }
  });
}
async function moveThread(c: Conversation, dir: string) {
  if (!c.thread || dir === c.thread.dir) return;
  if (runs.get(c.id)) return toast("OpenCode is still working in this chat");
  if (c.thread.id) {
    const r = await fetch(`/api/threads/${c.thread.id}/move`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dir }) }).catch(() => null);
    if (!r?.ok) return toast((await r?.json().catch(() => null))?.error?.message ?? "OpenCode did not move it");
    await loadThreads();
  }
  setPref("folders", JSON.stringify([...new Set([...kept("folders"), dir])]));
  setPref("closed-folders", JSON.stringify(kept("closed-folders").filter((d) => d !== dir)));
  onMachine(dir, c.thread.machine);
  for (const x of new Set([c, chats[c.id], c.id === conv.id ? conv : undefined])) if (x?.thread) x.thread.dir = dir;
  unfold(dir === homeOf(c.thread.machine) ? CHRYSALIS : DIR + dir);
  if (c.messages.length) save(c);
  renderControls();
  renderList();
}
const machineIcon = (kind: string): keyof typeof ICONS => (kind === "darwin" ? "mac" : "monitor");
async function handOff(c: Conversation, m: { id: string; name: string }) {
  if (!c.thread?.id) return;
  if (runs.get(c.id)) return toast("OpenCode is still working in this chat");
  const r = await fetch(`/api/threads/${c.thread.id}/handoff`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ machine: m.id }) }).catch(() => null);
  if (!r?.ok) return toast((await r?.json().catch(() => null))?.error?.message ?? "OpenCode did not hand it off");
  const got = await r.json();
  for (const x of new Set([c, chats[c.id], c.id === conv.id ? conv : undefined])) if (x?.thread) {
    x.thread.dir = got.dir;
    if (got.machine) x.thread.machine = got.machine; else delete x.thread.machine;
  }
  if (c.messages.length) save(c);
  await loadThreads();
  renderControls();
  renderList();
  toast(`Handed off to ${m.name}: the chat goes on there, in ${dirName(got.dir)}`);
}
function chooseMachine(at: HTMLElement) {
  if (!conv.thread) return;
  const r = at.getBoundingClientRect(), menu = $("row-menu"), here = conv.thread.machine ?? "";
  if (!menu.hidden) return void (menu.hidden = true);
  if (conv.thread.id) {
    const others = (harness.machines ?? []).filter((m) => m.id !== here);
    if (!others.length) return;
    openMenu(r.left, r.top, others.map((m): MenuItem => ({ label: m.name, icon: machineIcon(m.kind), run: () => void handOff(conv, m) })), "Hand off to");
    menu.style.top = "auto";
    menu.style.bottom = `${innerHeight - r.top + 6}px`;
    return;
  }
  openMenu(r.left, r.top, [...(harness.machines ?? []).map((m): MenuItem => ({ label: m.name, icon: machineIcon(m.kind), run: () => {
    setPref("machine", m.id);
    conv.thread = freshThread();
    renderCast();
    renderControls();
  } })), ...(status.here || !PHONE ? [{ label: "Add another computer", icon: "plus" as const, run: () => openConnect("computer") }] : [])], "Run on");
  menu.style.top = "auto";
  menu.style.bottom = `${innerHeight - r.top + 6}px`;
}
function chooseFolder(at: HTMLElement) {
  if (conv.thread) openFolders(at, { from: conv.thread.dir, to: (dir) => void moveThread(conv, dir) });
}
function newThread(dir: string, who = "") {
  setPref("folders", JSON.stringify([...new Set([...kept("folders"), dir])]));
  setPref("closed-folders", JSON.stringify(kept("closed-folders").filter((d) => d !== dir)));
  newChat(DIR + dir);
  if (who) conv.who = who;
  renderCast();
  render();
}

let ocAt = "", ocOn = "", ocShown = "", ocPick: (path: string) => void = () => {};
function paintHarness() {
  const b = $("opencode"), on = harness.state === "on";
  const dot = `dot ${on ? (harness.outside ? "outside" : "online") : harness.state === "off" ? "" : "failed"}`;
  if (b.lastElementChild?.className !== dot) b.innerHTML = `${OC_MARK}${OC_WORD}<span class="${dot}"></span>`;
  b.title = b.ariaLabel = on ? `OpenCode ${harness.version}: connected, ${harness.outside ? "models outside this computer allowed" : "Chrysalis's models only"}` : harness.state === "off" ? "OpenCode: disconnected" : "OpenCode is not installed on the computer Chrysalis runs on";
  b.classList.toggle("asking", !!status.waiting);
  const shown = JSON.stringify([harness.state, harness.version, harness.outside, status.waiting, status.here, dirs()]);
  if (!$("oc-pop").hidden && !ocAt && shown !== ocShown) void paintOcPop();
  ocShown = shown;
}
async function paintOcPop() {
  const pop = $("oc-pop"), row = (cls: string, html: string, onclick: () => void, title = "") => {
    const x = h("button", cls);
    x.type = "button";
    x.innerHTML = html;
    x.title = title;
    x.onclick = onclick;
    return x;
  };
  const host = status.here ? "" : status.machine ?? "the computer Chrysalis runs on";
  const rest = () => {
    const only = !harness.outside, note = h("p", only ? "oc-note" : "oc-note outside"), models: HTMLElement[] = [];
    if (!only) note.innerHTML = icon("warn");
    if (status.here) {
      const line = h("label", "oc-only"), sw = h("button", "switch");
      sw.type = "button";
      sw.id = "oc-only";
      sw.setAttribute("role", "switch");
      sw.setAttribute("aria-checked", String(only));
      sw.onclick = () => (only ? $<HTMLDialogElement>("outside-view").showModal() : void setOutside(false));
      line.append(h("span", "", "Chrysalis only"), sw);
      models.push(line);
    } else if (only) {
      const line = h("div", "oc-only locked");
      line.id = "oc-locked";
      line.innerHTML = icon("lock");
      line.append(h("span", "", "Chrysalis only"), h("span", "oc-set", `set on ${host}`));
      line.title = `Other models can be allowed only on ${host}: this switch in its own browser, or the OpenCode switch in Chrysalis's terminal there`;
      models.push(line);
    }
    note.append(only ? "Nothing leaves your computers." : "Other models are allowed. A thread on one sends it your prompts and files.");
    models.push(note);
    if (!status.here && !only) {
      const back = row("invite-row oc-back", `${icon("lock")}<span class="invite-name">Back to Chrysalis only</span>`, () => void setOutside(false));
      back.id = "oc-back";
      models.push(back);
    }
    if (!status.here && PHONE) return [h("div", "invite-head oc-part", "Models"), ...models];
    const link = (id: string, glyph: keyof typeof ICONS, label: string, mode: "devices" | "computer", count = 0) => {
      const b = row("invite-row oc-devices", `${icon(glyph)}<span class="invite-name"></span>${count ? `<span class="oc-asking">${count}</span>` : ""}${icon("chevron")}`, () => { pop.hidden = true; openConnect(mode); });
      b.id = id;
      b.querySelector(".invite-name")!.textContent = label;
      return b;
    };
    return [h("div", "invite-head oc-part", "Models"), ...models, h("div", "invite-head oc-part", "Devices"),
      ...(status.here ? [link("oc-devices", "phone", status.waiting ? "Asking to be let in" : "Add a phone or browser", "devices", status.waiting ?? 0)] : []),
      link("oc-computers", "monitor", "Add a computer to Run on", "computer")];
  };
  const head = (version = "") => h("div", "invite-head", `OpenCode ${version}${host ? ` on ${host}` : ""}`.replace(/  +/g, " "));
  if (harness.state === "absent") {
    const line = h("code", "oc-line", harness.install ?? "");
    return void pop.replaceChildren(head(), h("p", "oc-note", `OpenCode is not installed on the computer Chrysalis runs on. Install it there${harness.install?.startsWith("npm") ? ", in a terminal (npm comes with Node.js, from nodejs.org)" : ""}, then press the button again:`), line,
      row("invite-card", `${icon("copy")}Copy the line`, () => void navigator.clipboard.writeText(harness.install ?? "").then(() => toast("Copied"))), ...rest());
  }
  const connect = () => { harness.wanted = true; void paintOcPop(); void ensureHarness().catch((e) => { harness.wanted = false; toast(e instanceof Error ? e.message : String(e)); void paintOcPop(); }); };
  if (harness.state !== "on") return void pop.replaceChildren(head(), ...(harness.wanted ? [h("p", "oc-note", "Starting OpenCode…")]
    : [row("invite-card", `${icon("power")}Connect`, connect, "Start OpenCode: its folders and threads come back")]), ...rest());
  if (ocAt) {
    const r = await fetch(`/api/threads/folders?path=${encodeURIComponent(ocAt === "~" ? "" : ocAt)}${ocOn ? `&machine=${ocOn}` : ""}`).catch(() => null), got = r?.ok ? await r.json() : null;
    if (!got) { ocAt = ""; toast("That folder cannot be read"); return paintOcPop(); }
    ocAt = got.path;
    const pick = ocPick, here = got.path as string, home = homeOf(ocOn) ?? "", typed = pop.contains(document.activeElement);
    const use = (path: string) => { ocAt = ""; pop.hidden = true; pop.classList.remove("exploring"); pick(path); };
    const go = (path: string) => { pop.classList.add("loading"); ocAt = path; void paintOcPop(); };
    const crumbs = h("nav", "oc-crumbs"), under = !!home && (here === home || here.startsWith(`${home}/`));
    crumbs.ariaLabel = "Path";
    let at = under ? home : "";
    const steps: [string, string][] = [[under ? dirName(home) : "/", under ? home : "/"],
      ...(under ? here.slice(home.length) : here).split("/").filter(Boolean).map((p): [string, string] => [p, (at = `${at}/${p}`)])];
    steps.forEach(([name, path], k) => {
      if (k) crumbs.insertAdjacentHTML("beforeend", `<span class="oc-sep" aria-hidden="true">${icon("chevron")}</span>`);
      const c = row("oc-crumb", k ? "" : icon("folder"), () => go(path));
      c.append(h("span", "oc-crumb-name", name));
      c.ariaLabel = path;
      if (k === steps.length - 1) c.setAttribute("aria-current", "location");
      crumbs.append(c);
    });
    const entry = (name: string, path: string, note = "") => {
      const x = h("div", "oc-dir"), b = row("invite-row", `${icon("folder")}<span class="invite-name"></span>${note ? '<span class="oc-note-path"></span>' : ""}`, () => go(path), path);
      b.querySelector(".invite-name")!.textContent = name;
      if (note) b.querySelector(".oc-note-path")!.textContent = note.startsWith(`${home}/`) ? `~${note.slice(home.length)}` : note === home ? "~" : note;
      x.append(b, row("oc-open", "Use", () => use(path), `Work in ${path}`));
      x.onmouseenter = () => { sel = rows().indexOf(x); mark(); };
      return x;
    };
    const recent = [...new Set([home, ...kept("folders"), ...(harness.threads ?? []).map((t) => t.dir)])].filter((d) => d && d !== here && (d === home || (dirMachine(d) ?? "") === ocOn)).slice(0, 5);
    const list = h("div", "invite-list oc-list"), find = h("label", "oc-find"), filter = h("input", "oc-path"), clear = h("button", "oc-clear");
    filter.placeholder = "Filter, or type a path";
    filter.spellcheck = false;
    filter.autocomplete = "off";
    filter.ariaLabel = "Filter folders, or type a path";
    clear.type = "button";
    clear.title = clear.ariaLabel = "Clear";
    clear.innerHTML = icon("x");
    clear.onclick = () => { filter.value = ""; fill(); filter.focus(); };
    find.insertAdjacentHTML("afterbegin", icon("search"));
    find.append(filter, clear);
    let sel = -1;
    const rows = () => [...list.querySelectorAll<HTMLElement>(".oc-dir")];
    const mark = () => rows().forEach((r, k) => r.classList.toggle("sel", k === sel));
    const fill = () => {
      const q = filter.value.trim().toLowerCase(), path = /^[~/]/.test(q);
      clear.hidden = !filter.value;
      const dirs = (got.dirs as { name: string; path: string }[]).filter((d) => path || !q || d.name.toLowerCase().includes(q));
      list.replaceChildren(...(!q && recent.length ? [h("div", "oc-section", "Recent"), ...recent.map((d) => entry(dirName(d) || "/", d, d))] : []),
        ...(!q && recent.length && dirs.length ? [h("div", "oc-section", dirName(here) || "/")] : []),
        ...dirs.map((d) => entry(d.name, d.path)),
        ...(path ? [h("div", "oc-empty", "Press Enter to go to this path")] : !dirs.length ? [h("div", "oc-empty", q ? "No folder matches" : "No folders in here")] : []));
      sel = q && !path && dirs.length ? 0 : -1;
      mark();
    };
    filter.oninput = fill;
    filter.onkeydown = (e) => {
      const n = rows().length, q = filter.value.trim();
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); sel = n ? (sel + (e.key === "ArrowDown" ? 1 : n - 1 + (sel < 0 ? 1 : 0))) % n : -1; mark(); rows()[sel]?.scrollIntoView({ block: "nearest" }); }
      else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); use(sel >= 0 ? rows()[sel].querySelector(".invite-row")!.getAttribute("title")! : here); }
      else if (e.key === "Enter" && /^[~/]/.test(q)) { e.preventDefault(); go(q.replace(/^~(?=\/|$)/, home)); }
      else if (e.key === "Enter" && sel >= 0) { e.preventDefault(); rows()[sel].querySelector<HTMLElement>(".invite-row")!.click(); }
      else if (e.key === "Backspace" && !filter.value && steps.length > 1) { e.preventDefault(); go(steps.at(-2)![1]); }
      else if (e.key === "Escape") { e.preventDefault(); ocAt = ""; pop.hidden = true; pop.classList.remove("exploring"); }
    };
    fill();
    const foot = h("div", "oc-foot"), keys = h("span", "oc-keys"), mod = /Mac|iP(hone|ad)/.test(navigator.platform) ? "⌘" : "Ctrl";
    keys.innerHTML = `<kbd>↑</kbd><kbd>↓</kbd> move <kbd>↵</kbd> open <kbd>${mod}</kbd><kbd>↵</kbd> use`;
    foot.append(keys, row("oc-use", `Use ${escapeHtml(dirName(here) || "/")}`, () => use(here), here));
    pop.classList.add("exploring");
    pop.classList.remove("loading");
    const title = h("div", "oc-title", "Choose a folder");
    if ((harness.machines?.length ?? 0) > 1 && conv.thread && !conv.thread.id) {
      const pick = h("select", "oc-machine");
      pick.ariaLabel = "On";
      for (const m of harness.machines!) pick.append(Object.assign(h("option", "", m.name), { value: m.id, selected: m.id === ocOn }));
      pick.onchange = () => {
        setPref("machine", pick.value);
        conv.thread = freshThread();
        ocOn = conv.thread.machine ?? "";
        ocAt = conv.thread.dir;
        renderCast();
        renderControls();
        void paintOcPop();
      };
      title.append(" on ", pick);
    }
    pop.replaceChildren(title, crumbs, find, list, foot);
    crumbs.scrollLeft = crumbs.scrollWidth;
    crumbs.classList.toggle("clipped", crumbs.scrollWidth > crumbs.clientWidth);
    if (typed || matchMedia("(hover: hover)").matches) filter.focus();
    return;
  }
  pop.replaceChildren(head(harness.version),
    row("invite-card", `${icon("power")}Disconnect`, () => {
      harness = { state: "off", wanted: false, outside: harness.outside };
      harnessAt++;
      paintHarness();
      void paintOcPop();
      renderCast();
      renderList();
      if (conv.thread) newChat();
      void fetch("/api/threads", { method: "DELETE" }).then(() => loadThreads());
    }, "Stop OpenCode: threads stop until it is connected again"), ...rest());
}
async function setOutside(on: boolean) {
  const r = await fetch("/api/threads/outside", on ? { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" } : { method: "DELETE" }).catch(() => null);
  if (!r?.ok) toast((await r?.json().catch(() => null))?.error?.message ?? "OpenCode did not take that");
  await loadThreads();
  if (!$("oc-pop").hidden && !ocAt) void paintOcPop();
}
let ensuring: Promise<void> | undefined;
function ensureHarness(): Promise<void> {
  if (harness.state === "on" && harness.models?.length) return Promise.resolve();
  return (ensuring ??= (async () => {
    if (harness.state !== "on") {
      const started = await fetch("/api/threads", { method: "POST" }).catch(() => null);
      if (!started?.ok) throw new Error((await started?.json().catch(() => null))?.error?.message ?? "OpenCode did not start");
    }
    await loadThreads();
  })().finally(() => { ensuring = undefined; }));
}
async function openHarness() {
  const pop = $("oc-pop"), r = $("opencode").getBoundingClientRect();
  if (!pop.hidden) return void (pop.hidden = true);
  ocAt = "";
  pop.hidden = false;
  pop.classList.remove("exploring", "loading");
  pop.style.maxHeight = "";
  pop.style.bottom = "auto";
  pop.style.left = "auto";
  pop.style.right = `${Math.max(12, innerWidth - r.right)}px`;
  pop.style.top = `${r.bottom + 6}px`;
  void paintOcPop();
  await loadThreads();
}
function openFolders(at: HTMLElement, pick: { from: string; to: (path: string) => void }) {
  const pop = $("oc-pop"), r = at.getBoundingClientRect(), low = r.top > innerHeight / 2;
  ocAt = pick.from;
  ocOn = conv.thread?.machine ?? "";
  ocPick = pick.to;
  pop.hidden = false;
  pop.style.right = "auto";
  pop.style.left = `${Math.max(8, Math.min(r.right - Math.min(400, innerWidth - 16), innerWidth - Math.min(400, innerWidth - 16) - 8))}px`;
  pop.style.top = low ? "auto" : `${r.bottom + 6}px`;
  pop.style.bottom = low ? `${innerHeight - r.top + 6}px` : "auto";
  pop.style.maxHeight = `${Math.min(560, (low ? r.top : innerHeight - r.bottom) - 18)}px`;
  harness.wanted = true;
  void paintOcPop();
  if (harness.state !== "on") void ensureHarness().then(() => paintOcPop(), (e) => { pop.hidden = true; toast(e instanceof Error ? e.message : String(e)); });
}

const unwritten = (p?: Person) => !!p && p.kind === "character" && !p.image && ![p.card.data.name, p.card.data.description, p.card.data.scenario, p.card.data.first_mes].some((t) => t?.trim());
function newCharacter() {
  if (unwritten(people.get(sheetOf))) return closeSheet();
  const p: Person = { id: randomId(), kind: "character", card: { spec: "chara_card_v2", spec_version: "2.0", data: { name: "", description: "" } }, added: Date.now() };
  own(p).color = tinted(Math.floor(Math.random() * 360));
  people.set(p.id, p);
  openSheet(p.id);
  $("cast-sheet").querySelector<HTMLInputElement>("input.card-name")?.focus();
}
function dropBlank(id: string) {
  if (!unwritten(people.get(id)) || chatsOf(id).length || folderOf(conv) === id || [...groups.values()].some((g) => g.members.includes(id))) return;
  people.delete(id);
  void dropPerson(id);
  renderCast();
}
let fileInto = false, inviteFrom: string | null = null;
function openInvite(at: HTMLElement, from: string | null) {
  const box = $("cast-invite"), find = h("input", "invite-find"), list = h("div", "invite-list"), card = h("button", "invite-card");
  inviteFrom = from;
  find.type = "search";
  find.placeholder = "Find a character";
  find.ariaLabel = "Find a character";
  find.spellcheck = false;
  find.oninput = () => paintInvite();
  find.onkeydown = (e) => { if (e.key === "Enter") list.querySelector<HTMLElement>(".invite-row:not(:disabled)")?.click(); };
  card.type = "button";
  card.innerHTML = `${icon("personAdd")}A character from a card…`;
  card.onclick = () => {
    if (inviteFrom) { newChat(inviteFrom); inviteFrom = null; }
    fileInto = inviteFrom === null;
    $("cast-file").click();
  };
  box.replaceChildren(h("div", "invite-head"), find, list, card);
  box.hidden = false;
  paintInvite();
  const r = at.getBoundingClientRect(), up = r.top > innerHeight / 2;
  box.style.left = `${Math.max(12, Math.min(r.left, innerWidth - box.offsetWidth - 12))}px`;
  box.style.top = up ? "" : `${r.bottom + 6}px`;
  box.style.bottom = up ? `${innerHeight - r.top + 6}px` : "";
  if (matchMedia("(hover: hover)").matches) find.focus();
}
function paintInvite() {
  const box = $("cast-invite"), list = box.querySelector<HTMLElement>(".invite-list"), find = box.querySelector<HTMLInputElement>(".invite-find");
  if (box.hidden || !list || !find) return;
  const all = [...people.values()].filter((p) => p.kind !== "persona" && !unwritten(p)).sort((a, b) => nameOf(a.id).localeCompare(nameOf(b.id)));
  const from = inviteFrom, folder = !!from && from.startsWith(DIR), edited = from === null ? castOf(conv).group : groups.get(from ?? "");
  const plain = from === null && !folderOf(conv), q = find.value.trim().toLowerCase();
  const members = edited ? membersOf(edited) : from === null ? (plain ? [people.get(CHRYSALIS)!] : castOf(conv).members) : from && !folder ? [people.get(from)!] : [];
  box.querySelector(".invite-head")!.textContent = folder ? `A new thread in ${nameOf(from!)} with` : edited ? `Who is in ${nameOf(edited.id)}`
    : from === null ? (!plain || conv.messages.length ? "Bring into this chat" : whoOf(conv).startsWith(DIR) ? "Who works in this thread" : "Who this chat is with") : `A new group with ${nameOf(from!)} and`;
  const shown = all.filter((p) => p.card.data.name.toLowerCase().includes(q));
  find.hidden = all.length < 7;
  list.replaceChildren(...shown.map((p) => {
    const inside = members.some((m) => m.id === p.id), b = h("button", "invite-row"), name = p.card.data.name;
    b.type = "button";
    b.setAttribute("aria-pressed", String(inside));
    b.disabled = inside && (edited ? !canLeave(edited) : from !== null || plain);
    b.title = edited ? (inside ? (b.disabled ? `${name}: one of the last two in it, which has chats` : `${name}: in it (click to take out)`) : `Bring ${name} in`)
      : inside ? `${name}: this chat is theirs${from === null ? " (click for plain Chrysalis)" : ""}` : folder ? `${name} works in it, in character` : `Bring ${name} in`;
    b.append(face(p.id), h("span", "invite-name", name));
    if (inside) b.insertAdjacentHTML("beforeend", icon("check"));
    b.onclick = () => {
      if (edited && inside) takeOut(edited, p.id);
      else if (edited) setMembers(edited, [...edited.members, p.id], `${name} joined ${called(edited)}`);
      else if (from === null && inside) {
        delete conv.who;
        stow(conv);
        if (conv.thread?.id) void fetch(`/api/threads/${conv.thread.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scene: sceneFor(conv) ?? null }) }).then(() => loadThreads());
        renderCast();
        render();
      }
      else if (from === null) join([p.id]);
      else {
        inviteFrom = null;
        if (folder) { newChat(from); conv.who = p.id; renderCast(); render(); }
        else newChat((groupOf([from!, p.id]) ?? makeGroup([from!, p.id])).id);
      }
      paintInvite();
    };
    return b;
  }));
  if (!shown.length) list.append(h("p", "invite-none", all.length ? "No one by that name" : "No characters yet: add one from a card"));
}
const closeInvite = () => { $("cast-invite").hidden = true; inviteFrom = null; };
$("cast-file").addEventListener("change", async (e) => {
  const input = e.currentTarget as HTMLInputElement;
  for (const f of await welcome(input.files ?? [], fileInto)) toast(`${f.name} holds no character card`);
  input.value = "";
});

const hexOf = (h: number, s: number, v: number) => {
  const f = (n: number) => { const k = (n + h / 60) % 6; return Math.round(255 * (v - v * s * Math.max(0, Math.min(k, 4 - k, 1)))).toString(16).padStart(2, "0"); };
  return `#${f(5)}${f(3)}${f(1)}`;
};
function hsvOf(hex: string): [number, number, number] {
  const [r, g, b] = [1, 3, 5].map((k) => Number.parseInt(hex.slice(k, k + 2), 16) / 255), v = Math.max(r, g, b), d = v - Math.min(r, g, b);
  const h = !d ? 0 : v === r ? ((g - b) / d + 6) % 6 : v === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h * 60, v ? d / v : 0, v];
}
function tintPop(from: string, pick: (hex: string) => void, done: () => void) {
  const pop = h("div", "tint-pop"), field = h("div", "tint-field"), dot = h("span", "tint-dot"), rail = h("div", "tint-rail"), knob = h("span", "tint-dot"), code = h("input", "tint-code"), chip = h("span", "tint-chip");
  let [hue, sat, val] = hsvOf(from);
  const paint = (typed = false) => {
    const hex = hexOf(hue, sat, val);
    field.style.setProperty("--hue", String(hue));
    dot.style.left = `${sat * 100}%`;
    dot.style.top = `${(1 - val) * 100}%`;
    knob.style.left = `${(hue / 360) * 100}%`;
    for (const x of [dot, chip]) x.style.background = hex;
    knob.style.background = hexOf(hue, 1, 1);
    if (!typed) code.value = hex;
    field.setAttribute("aria-valuetext", `Saturation ${Math.round(sat * 100)}%, brightness ${Math.round(val * 100)}%`);
    rail.setAttribute("aria-valuenow", String(Math.round(hue)));
  };
  const set = (typed = false) => { paint(typed); pick(hexOf(hue, sat, val)); };
  const drag = (el: HTMLElement, move: (x: number, y: number) => void) => {
    el.onpointerdown = (e) => {
      el.setPointerCapture(e.pointerId);
      const at = (ev: PointerEvent) => { const r = el.getBoundingClientRect(); move(Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)), Math.max(0, Math.min(1, (ev.clientY - r.top) / r.height))); set(); };
      at(e);
      el.onpointermove = at;
      el.onpointerup = el.onpointercancel = () => { el.onpointermove = null; done(); };
      e.preventDefault();
      el.focus();
    };
  };
  drag(field, (x, y) => { sat = x; val = 1 - y; });
  drag(rail, (x) => { hue = Math.min(359.9, x * 360); });
  const keys = (el: HTMLElement, step: (dx: number, dy: number) => void) => {
    el.tabIndex = 0;
    el.setAttribute("role", "slider");
    el.onkeydown = (e) => {
      const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key];
      if (!d) return;
      e.preventDefault();
      step(d[0] * (e.shiftKey ? 10 : 1), d[1] * (e.shiftKey ? 10 : 1));
      set();
      done();
    };
  };
  keys(field, (dx, dy) => { sat = Math.max(0, Math.min(1, sat + dx / 100)); val = Math.max(0, Math.min(1, val + dy / 100)); });
  keys(rail, (dx, dy) => { hue = Math.max(0, Math.min(359.9, hue + (dx + dy) * 3.6)); });
  field.ariaLabel = "Saturation and brightness";
  rail.ariaLabel = "Hue";
  rail.setAttribute("aria-valuemin", "0");
  rail.setAttribute("aria-valuemax", "360");
  code.spellcheck = false;
  code.maxLength = 7;
  code.ariaLabel = "Hex color";
  code.oninput = () => {
    const m = /^#?([0-9a-f]{6})$/i.exec(code.value.trim());
    if (!m) return;
    [hue, sat, val] = hsvOf(`#${m[1].toLowerCase()}`);
    set(true);
  };
  code.onchange = () => { paint(); done(); };
  const foot = h("div", "tint-foot");
  foot.append(chip, code);
  field.append(dot);
  rail.append(knob);
  pop.append(field, rail, foot);
  paint();
  return pop;
}
let sheetOf = "", sampler: Speaker | undefined, sheetFile: { id: string; url: string } | undefined, sheetKeep = () => {};
function openSheet(id: string) {
  if (sheetOf && sheetOf !== id) dropBlank(sheetOf);
  const sheet = $("cast-sheet"), person = people.get(id), g = groups.get(id);
  if (!person && !g) return;
  const again = sheetOf === id, top = sheet.querySelector(".card-scroll")?.scrollTop ?? 0;
  sheetOf = id;
  sheetKeep();
  let saving = 0;
  const saved = () => {
    if (!saving) return;
    clearTimeout(saving);
    saving = 0;
    if (person && people.has(id)) { void savePerson(person); void cardFile(person); } else if (g && groups.has(id)) keepGroup(g);
  };
  const keep = () => {
    clearTimeout(saving);
    saving = window.setTimeout(saved, 300);
  };
  sheetKeep = saved;
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
  const art = h("div", "card-art"), title = h("div", "card-title"), body = h("div", "card-body"), scroll = h("div", "card-scroll");
  if (g) {
    const all = membersOf(g).slice(0, 4).map((m) => {
      const f = m.kind === "assistant" && !m.image && seaPortrait() ? cardSea!.layer : face(m.id);
      f.dataset.card = m.id;
      f.title = `${nameOf(m.id)}'s card`;
      return f;
    }), strips = h("div", `strips s${Math.max(all.length, 1)}`);
    strips.append(...(all.length ? all : [face(id)]));
    art.append(strips);
    if (all.includes(cardSea?.layer as HTMLElement)) {
      scroll.addEventListener("scroll", () => cardSea!.ocean.place(), { passive: true });
      requestAnimationFrame(() => cardSea!.ocean.place());
    }
  } else if (person!.kind === "assistant" && !person!.image && seaPortrait()) {
    delete cardSea!.layer.dataset.card;
    cardSea!.layer.title = "";
    art.append(cardSea!.layer);
    scroll.addEventListener("scroll", () => cardSea!.ocean.place(), { passive: true });
  } else {
    const pic = person!.image ? h("img") : face(id);
    if (pic instanceof HTMLImageElement) {
      pic.src = faces.get(id)!;
      pic.alt = "";
      pic.draggable = false;
    }
    pic.classList.add("card-portrait");
    art.append(pic);
    if (sheetFile?.id !== id) void cardFile(person!);
  }
  art.append(h("div", "card-scrim"), title);
  if (person) art.append(btn("card-float card-pic", person.image ? "Another picture" : "A picture", icon("edit"), () => picture(person)));
  if (person?.kind === "character" && !person.image) {
    const well = btn("card-float card-tint", "Their color", icon("palette"), () => {
      if (sheet.querySelector(".tint-pop")) return void sheet.querySelector(".tint-pop")!.remove();
      sheet.append(tintPop(tintOf(person), (hex) => { own(person).color = hex; keep(); relit(); renderCast(); }, () => render()));
    });
    art.append(well);
  }
  const relit = () => {
    const letter = art.querySelector(":scope > .face.card-portrait");
    if (letter) { const now = face(id); now.classList.add("card-portrait"); letter.replaceWith(now); }
  };
  const name = h("input", "card-name");
  name.value = person ? person.card.data.name : g!.name ?? "";
  name.placeholder = id === ME ? "User" : id === CHRYSALIS ? "Chrysalis" : g ? castNames(g) : "Their name";
  name.ariaLabel = "Name";
  name.spellcheck = false;
  name.oninput = () => {
    if (person?.kind === "character") own(person).color ??= tintOf(person);
    if (person) person.card.data.name = name.value;
    else if (name.value.trim()) g!.name = name.value; else delete g!.name;
    const file = body.querySelector<HTMLAnchorElement>("a.card-file");
    if (file) file.download = cardName(id);
    relit();
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
    body.append(voicePick(person, keep), field("Description", "description", person.kind === "assistant" ? "The system prompt: who they are and how they answer" : "Who they are: looks, past, manner, how they talk"),
      field("Scenario", "scenario", "Where and when the chat begins"), field("Greeting", "first_mes", "Their first message"));
    const acts = h("div", "card-acts"), file = h("a", "card-file");
    file.download = cardName(id);
    file.title = "Their card as a PNG any app that takes character cards can read";
    file.draggable = false;
    file.innerHTML = `${icon("download")}Download card`;
    if (sheetFile?.id === id) file.href = sheetFile.url;
    acts.append(file);
    body.append(acts);
  } else {
    const members = h("div", "card-members");
    members.append(...membersOf(g!).map((x) => {
      const row = h("div", "card-member"), open = h("button", "card-member-open"), name = x.card.data.name, last = !canLeave(g!);
      open.type = "button";
      open.title = `${name}'s card`;
      open.dataset.card = x.id;
      open.append(face(x.id), h("span", "", name));
      const out = btn("act card-member-out", last ? `${name}: one of the last two in it, which has chats` : `Take ${name} out`, icon("x"), () => {
        clearTimeout(saving);
        takeOut(g!, x.id);
        if (groups.has(id)) openSheet(id); else closeSheet();
      });
      out.disabled = last;
      row.append(open, out);
      return row;
    }));
    const add = btn("card-member-add", "Bring someone in", `${icon("personAdd")}Add someone`, () => openInvite(add, id));
    const who = h("div", "card-section"), where = h("label", "card-field"), acts = h("div", "card-acts");
    who.append(h("span", "card-label", "Members"), members, add);
    where.append(h("span", "card-label", "Scenario"), text(g!.scenario ?? "", (v) => { if (v) g!.scenario = v; else delete g!.scenario; }, "card-text",
      "Where they are and what is going on (in place of their own)"));
    acts.append(btn("", "This group as a file, with its members' cards in it", `${icon("download")}Download group`,
      () => download(`${nameOf(id) || "group"}.json`, JSON.stringify({ spec: "chrysalis_group", name: nameOf(id),
        members: membersOf(g!).map((m) => m.card.data.name), cards: membersOf(g!).map((m) => m.card), ...(g!.scenario ? { scenario: g!.scenario } : {}) }, null, 2), "application/json")));
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
}
const cardName = (id: string) => `${nameOf(id) || "character"}.png`;
function closeSheet() {
  if (menuAnchor?.closest("#cast-sheet")) voiceMenu.hidePopover();
  if ($("cast-sheet").contains(document.activeElement)) (document.activeElement as HTMLElement).blur();
  $("cast-sheet").hidden = true;
  if (sheetFile) URL.revokeObjectURL(sheetFile.url);
  sheetFile = undefined;
  const was = sheetOf;
  sheetOf = "";
  sheetKeep();
  dropBlank(was);
  sampler?.close();
  sampler = undefined;
}
function voicePick(p: Person, keep: () => void) {
  const box = h("div", "card-voice"), assistant = p.kind === "assistant";
  const mine = () => (assistant ? settings.voice : voiceOf(p.card.data));
  const spot: VoiceSpot = {
    now: () => voiceIdOf(mine()?.tts, mine()?.name),
    none: assistant ? "The default voice" : `${nameOf(CHRYSALIS)}'s voice`,
    pick: (id) => {
      const [tts, name] = voiceParts(id);
      if (assistant) {
        setVoice({ tts, name });
        saveSettings();
      } else {
        const ext = (p.card.data.extensions ??= {}), own = (ext.chrysalis ??= {}) as { voice?: Voice };
        if (name) own.voice = { tts, name }; else delete own.voice;
        keep();
      }
      if (voice && castOf(voice.c).members.some((m) => m.id === p.id)) pushVoice();
      trigger.paint();
    },
    hear: (id) => {
      const [tts, name] = id ? voiceParts(id) : assistant ? [] : [settings.voice?.tts, settings.voice?.name];
      return samples().preview(tts, name);
    },
  };
  const trigger = voiceTrigger(spot);
  box.title = "The voice they speak with in voice mode";
  box.append(trigger.el);
  return box;
}
document.addEventListener("pointerdown", (e) => {
  const t = e.target as HTMLElement;
  if (!$("cast-invite").hidden && !t.closest("#cast-invite, .cast-bring, .card-member-add, #toast")) closeInvite();
  if (!$("oc-pop").hidden && !t.closest("#oc-pop, #opencode, #toast, #outside-view")) $("oc-pop").hidden = true;
  if (!$("row-menu").hidden && !t.closest("#row-menu")) $("row-menu").hidden = true;
  if (!t.closest(".tint-pop, .card-tint")) $("cast-sheet").querySelector(".tint-pop")?.remove();
  if (!$("cast-sheet").hidden && !t.closest("#cast-sheet, [data-card], #voice-menu, #add-cast")) closeSheet();
});
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape" || voiceMenu.matches(":popover-open")) return;
  if (!$("row-menu").hidden) { $("row-menu").hidden = true; e.preventDefault(); }
  else if (!$("cast-invite").hidden) { closeInvite(); e.preventDefault(); }
  else if (!$("oc-pop").hidden) { $("oc-pop").hidden = true; e.preventDefault(); }
  else if ($("cast-sheet").querySelector(".tint-pop")) { $("cast-sheet").querySelector(".tint-pop")!.remove(); e.preventDefault(); }
  else if (!$("cast-sheet").hidden) { closeSheet(); e.preventDefault(); }
}, true);
document.addEventListener("click", (e) => {
  const id = (e.target as HTMLElement).closest<HTMLElement>("[data-card]")?.dataset.card;
  if (!id) return;
  if (sheetOf === id) closeSheet(); else openSheet(id);
});

function speakerHead(c: Conversation, m: Message) {
  const id = m.as && people.has(m.as) ? m.as : writerOf(c)?.id;
  return id ? head(id) : null;
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
  input.placeholder = members.length ? `Message ${nameOf(conv.who!)}` : "Message Chrysalis";
  box.hidden = !group || !members.length;
  paintInvite();
  if (box.hidden) return box.replaceChildren();
  const talk = h("button", `talk${talking ? " on" : ""}`), up = nextUp(conv);
  talk.type = "button";
  talk.title = talking ? "Stop them" : "They take turns by themselves for a few replies (typing stops them)";
  talk.innerHTML = talking ? `${icon("stop")}Stop them` : `${icon("play")}Let them talk`;
  talk.onclick = () => {
    if (talking) { talking = 0; return renderCastNext(); }
    talking = 4;
    renderCastNext();
    talkOn();
  };
  box.replaceChildren(h("span", "", "Next"), ...members.map((m) => {
    const b = h("button", m.id === up ? "up" : "");
    b.type = "button";
    b.title = b.ariaLabel = `${m.card.data.name} answers`;
    b.append(face(m.id));
    b.onclick = () => replyAs(m);
    return b;
  }), talk);
}
function nextUp(c: Conversation): string | undefined {
  const { members, group } = castOf(c);
  if (!group || !members.length) return undefined;
  const last = c.messages.findLast((m) => m.role === "assistant" && m.as)?.as;
  return members[(members.findIndex((m) => m.id === last) + 1) % members.length].id;
}
let lined: string[] = [];
function replyAs(p: Person, auto = false) {
  if (running()) return;
  if (!auto) { talking = 0; lined = []; }
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
    else parts.push({ text: m.content, who: (m.as && people.get(m.as)) || writerOf(c) });
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
  const next = lined.shift() ?? (talking ? nextUp(conv) : undefined), p = next && castOf(conv).members.find((m) => m.id === next);
  if (!p || running()) { talking = 0; lined = []; return renderCastNext(); }
  if (talking) talking--;
  replyAs(p, true);
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
const cellsOf = (row: string) => row.trim().replace(/^\|/, "").replace(/(?<!\\)\|$/, "").split(/(?<!\\)\|/).map((c) => c.trim());
function bareRows(src: string) {
  const row = /[^\n]*\|[^\n]*(?:\n|$)/y, rows: string[][] = [];
  let raw = "";
  for (let m = row.exec(src); m && !/^(?: {4}| {0,3}(?:[-*+] |\d+[.)] |#{1,6} |>|```|~~~))/.test(m[0]); m = row.exec(src)) {
    const cells = cellsOf(m[0]);
    if (cells.length < 2 || (rows.length && cells.length !== rows[0].length)) break;
    if (rows.length === 1 && cells.every((c) => /^:?-+:?$/.test(c))) return;
    rows.push(cells);
    raw += m[0];
  }
  return rows.length > 2 ? { raw, rows } : undefined;
}
marked.use({
  extensions: [{
    name: "bareTable",
    level: "block",
    start(src) { for (const m of src.matchAll(/\n(?=[^\n]*\|)/g)) if (bareRows(src.slice(m.index + 1))) return m.index + 1; },
    tokenizer(src) {
      const found = bareRows(src);
      if (!found) return;
      const cell = (text: string, header: boolean) => ({ text, tokens: this.lexer.inline(text), header, align: null });
      return { type: "table", raw: found.raw, align: found.rows[0].map(() => null), header: found.rows[0].map((c) => cell(c, true)),
        rows: found.rows.slice(1).map((r) => r.map((c) => cell(c, false))) };
    },
  }],
});

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
function fitTable(box: Element) {
  box.classList.remove("dense", "stacked");
  if (box.scrollWidth > box.clientWidth) box.classList.add("dense");
  if (box.scrollWidth > box.clientWidth) box.classList.replace("dense", "stacked");
}
const tableFit = new ResizeObserver((seen) => { for (const s of seen) fitTable(s.target.parentElement!); });
function typeset(root: ParentNode) {
  for (const el of root.querySelectorAll<HTMLElement>(".math[data-tex]"))
    katex.render(el.dataset.tex!, el, { displayMode: el.classList.contains("display"), throwOnError: false, strict: "ignore", output: "htmlAndMathml" });
  for (const table of root.querySelectorAll("table")) {
    const box = h("div", "table"), ruler = h("i", "table-ruler"), heads = [...table.querySelectorAll("thead th")].map((c) => c.textContent!);
    for (const row of table.querySelectorAll("tbody tr")) [...row.children].forEach((c, k) => { if (heads[k]) (c as HTMLElement).dataset.label = heads[k]; });
    table.replaceWith(box);
    box.append(ruler, table);
    tableFit.observe(ruler);
  }
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
  if (/^\s*\|/.test(line) || /\|.*\n$/.test(head)) return head;
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
  const b = v.blocks, tokens = chatTokens(live ? settle(src) : src), blocks = tokens.filter((t) => t.type !== "space");
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
  frame.setAttribute("sandbox", "allow-scripts allow-modals allow-pointer-lock allow-orientation-lock");
  frame.allow = "fullscreen; clipboard-write";
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
    fig.querySelector(":scope > .page-fail")?.remove();
    if (fig.classList.contains("show-code")) {
      const code = fig.querySelector<HTMLElement>(".svg-code code")!, pre = code.parentElement!;
      code.innerHTML = hljs.highlight(text, { language: "xml", ignoreIllegals: true }).value;
      if (live) pre.scrollTop = pre.scrollHeight;
    }
  }
  const title = text.match(/<title[^>]*>([^<]*)<\/title>/i)?.[1].trim();
  fig.querySelector(".svg-state")!.textContent = live ? "" : title ? `Page · ${title}` : "Page";
  fig.classList.toggle("drawing", live);
  const state = fig.querySelector<HTMLElement>(".svg-state")!;
  let badge = fig.querySelector<HTMLElement>(".svg-bar > .page-writing");
  if (live && !badge) {
    state.before(badge = h("button", "page-writing"));
    (badge as HTMLButtonElement).type = "button";
    badge.dataset.page = "code";
    badge.innerHTML = `${icon("gear")}<span>Writing the page…</span><span class="page-peek"></span>`;
    pagePeek(fig);
  } else if (!live) badge?.remove();
  showPage(fig.querySelector<HTMLElement>(".page-paper")!, text, live);
  return fig;
}

const PAGE_FAILS = "The page's script fails in the browser with";
function pageFailed(paper: HTMLElement, what: string, at: string) {
  const fig = paper.closest<HTMLElement>(".page-fig");
  if (!fig || fig.classList.contains("drawing") || fig.querySelector(":scope > .page-fail")) return;
  const fail = h("div", "page-fail");
  fail.append(h("span", "page-fail-what", `This page's script stopped: ${what}`));
  if (at) fail.append(h("code", "page-fail-at", at));
  const fix = h("button", "page-fail-fix", "Ask for a fix");
  fix.type = "button";
  fix.dataset.page = "fix";
  fix.dataset.ask = `${PAGE_FAILS} "${what}"${at ? ` at this line:\n\n\`\`\`\n${at}\n\`\`\`\n` : "."}\nFix it.`;
  fail.append(fix);
  paper.after(fail);
  const last = conv.messages.at(-1), src = svgSource.get(fig);
  if (last?.role !== "assistant" || last.error || last.stopped || running() || drafting) return;
  if (![...artifactsOf(conv).values()].some((vs) => vs.at(-1)!.block.at === last && vs.at(-1)!.src === src)) return;
  let tried = 0;
  for (let i = conv.messages.length - 2; i >= 0 && conv.messages[i].role === "user" && conv.messages[i].content.startsWith(PAGE_FAILS); i -= 2) tried++;
  if (tried >= 2) return;
  const typed = input.value;
  submit(fix.dataset.ask);
  input.value = typed;
  grow();
}

function pagePeek(fig: HTMLElement) {
  const coded = fig.classList.contains("show-code"), badge = fig.querySelector<HTMLElement>(".page-writing");
  fig.querySelector(".svg-acts [data-page=code]")!.textContent = coded ? "Page" : "Code";
  if (!badge) return;
  badge.title = coded ? "Show the page as it is written" : "Show the code as it is written";
  badge.querySelector(".page-peek")!.textContent = coded ? "Page" : "Code";
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
  if (typeof d.chrysalisFail === "string") return pageFailed(paper, d.chrysalisFail, typeof d.at === "string" ? d.at : "");
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
    fig.classList.toggle("show-code");
    pagePeek(fig);
    const code = fig.querySelector<HTMLElement>(".svg-code code")!, pre = code.parentElement!;
    if (code.textContent !== text) code.innerHTML = hljs.highlight(text, { language: "xml", ignoreIllegals: true }).value;
    if (fig.classList.contains("drawing")) pre.scrollTop = pre.scrollHeight;
  } else if (act === "copy") copy(text, btn);
  else if (act === "save") download("page.html", text, "text/html");
  else if (act === "fix") submit(btn.dataset.ask ?? "");
  else if (act === "reload") {
    const paper = fig.querySelector<HTMLElement>(".page-paper")!;
    fig.querySelector(":scope > .page-fail")?.remove();
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
const PICTURE_FILE = /\.(?:html?|svg)$/i;
function fileBlocks(calls: Call[] | undefined, files: Map<string, string>, label: string, live: boolean): Block[] {
  const out: Block[] = [];
  for (const c of calls ?? []) {
    if ((c.name !== "write" && c.name !== "edit") || c.state === "failed") continue;
    let a: any;
    try { a = JSON.parse(c.args); } catch { continue; }
    const path = typeof a?.path === "string" ? a.path : "";
    if (!PICTURE_FILE.test(path)) continue;
    if (c.name === "write" && typeof a.content === "string") {
      const base = path.split("/").pop()!.replace(/[^\w.-]/g, "-"), taken = new Set(files.values());
      let id = files.get(path) ?? base;
      for (let k = 2; !files.has(path) && taken.has(id); k++) id = base.replace(/(\.[^.]+)?$/, `-${k}$1`);
      files.set(path, id);
      out.push({ at: c, n: 0, info: { kind: /\.svg$/i.test(path) ? "svg" : "html", id }, text: a.content, live, label });
    } else if (c.name === "edit" && files.has(path) && typeof a.oldString === "string" && a.oldString && typeof a.newString === "string") {
      const id = files.get(path)!;
      out.push({ at: c, n: 0, info: { edit: id, from: id }, text: `<<<<<<< SEARCH\n${a.oldString}\n=======\n${a.newString}\n>>>>>>> REPLACE`, live, label });
    }
  }
  return out;
}
function artifactsOf(c: Conversation): Map<string, Version[]> {
  const blocks: Block[] = [], writing = runs.get(c.id)?.m, files = new Map<string, string>();
  let label = "";
  for (const m of c.messages) {
    if (m.role === "user") {
      label = m.content;
      for (const [n, e] of (m.edits ?? []).entries())
        blocks.push({ at: m, n, info: { kind: e.kind, id: e.id }, text: e.src, live: false, label: "Your edit", by: "user" });
      continue;
    }
    const parts: [object, string, Call[] | undefined][] = [...(m.rounds ?? []).map((r): [object, string, Call[]] => [r, r.content, r.calls]),
      [m, m === writing ? settle(m.content) : m.content, m.calls]];
    for (const [holder, text, calls] of parts) {
      for (const b of artBlocksOf(holder, text))
        blocks.push({ at: holder, n: b.n, info: b.info, text: b.text, live: m === writing && holder === m && !b.closed, label });
      blocks.push(...fileBlocks(calls, files, label, m === writing));
    }
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
const WRITING: Record<string, string> = { page: "Writing the page", svg: "Drawing", mermaid: "Drawing the diagram", doc: "Writing", code: "Writing the code" };
function cardText(ver: Version, count: number) {
  const done = (ver.applied ?? []).filter((a) => a.ok).length, shape = canvasKind(ver), [, kind] = KINDS_SHOWN[shape];
  const what = ver.block.live ? (ver.applied ? `Changing${done ? `, ${done} so far` : ""}…` : `${WRITING[shape]}…`) : ver.block.by === "user"
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
  const kind = canvasKind(ver), shape = ver.block.live ? "live" : kind;
  if (el.dataset.shape !== shape) {
    el.dataset.shape = shape;
    el.querySelector(".art-icon")!.innerHTML = icon(ver.block.live ? "gear" : KINDS_SHOWN[kind][0]);
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
  for (const x of made) shelf.known!.add(liveKey(x));
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
  if (el.id) return spans.find((x) => x.attrs.id === el.id);
  const geo = ["cx", "cy", "r", "rx", "ry", "x", "y", "width", "height", "d", "points", "x1", "y1", "x2", "y2"].filter((a) => el.hasAttribute(a));
  const hits = geo.length ? spans.filter((x) => x.name === tag && geo.every((a) => x.attrs[a] === el.getAttribute(a))) : [];
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
  return { art: host.dataset.art!, sel: `cx:${cx}`, cx: cx as number, partId: span.attrs.id, markup: ver.src.slice(span.start, span.end), tag: span.tag, paper };
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
  const at = was.partId ? elementSpans(src).findIndex((x) => x.attrs.id === was.partId)
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
  const open = ver.src.slice(span.start, span.openEnd), mat = span.attrs.material;
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
    return { tag: el.tagName, name: el.tagName.toLowerCase(), attrs: {}, start: Number(el.dataset.at), openEnd: Number(el.dataset.at), end: Number(el.dataset.end), parent: up ? els.indexOf(up) : -1, closed: true };
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
  try { sessionStorage.setItem(key, JSON.stringify(value)); return true; } catch { return false; }
}

const chats: Record<string, Conversation> = load("chats", {});
let tabFull = false;
function keepChats() {
  const kept = store("chats", Object.fromEntries(Object.entries(chats).filter(([, c]) => !c.incognito).map(([id, c]) => [id, { ...c, messages: lean(c.messages) }])));
  if (!kept && !tabFull) toast("This tab has no room left to keep its chats through a reload: export or delete some");
  tabFull = !kept;
}
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
  return { id: randomId(), title: "New chat", updated: Date.now(), effort: pref("effort", "xhigh") as Effort, messages: [] };
}

function render() {
  views.clear();
  lasts(!conv.incognito);
  document.documentElement.toggleAttribute("data-incognito", !!conv.incognito);
  $("incognito").setAttribute("aria-pressed", String(!!conv.incognito));
  renderCastNext();
  const was = [...thread.children];
  thread.append(...(conv.messages.length ? conv.messages.map(messageView) : [emptyState()]));
  for (const el of was) el.remove();
  if (conv.messages.length) ocean.butterfly(null);
  $("title").textContent = conv.title;
  document.title = conv.messages.length && !conv.incognito ? `${conv.title} · Chrysalis Inference` : "Chrysalis Inference";
  renderComposer();
  renderList();
  paintGhost();
  renderShelf();
}

function messageView(m: Message, i: number): HTMLElement {
  if (m === editing) return editorView(m, i);
  if (m.role === "assistant") {
    const v = assistantView(i), by = speakerHead(conv, m);
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
      ["copy", true], ["edit", !running() && !m.sent], ["trash", !running()], ...(conv.thread ? [["fork", !running() && !m.sent]] as [Act, boolean][] : [])]);
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
  if (m.sent) el.append(Object.assign(h("div", "point-chip", m.sent), { title: "OpenCode sent this to the thread, not you" }));
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
  const think = m.role === "assistant" && m.reasoning && !conv.thread?.id ? area(m.reasoning, "Thinking") : null;
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
    if (conv.thread?.id) {
      render();
      if (send && text.value.trim()) void threadAct("resend", m, text.value.trim());
      else if (keep && text.value !== m.content) void threadAct("edit", m, text.value);
      return;
    }
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
      stale(conv, i);
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
function pager(m: Message, side = false) {
  const vs = m.versions;
  if (!vs && !side) return "";
  const n = (vs?.others.length ?? 0) + 1, k = vs?.shown ?? 0, b = (act: "prev" | "next", on: boolean, label: string) =>
    `<button type="button" class="act${side ? ` swipe ${act}` : ""}" data-act="${act}" title="${label}" aria-label="${label}"${on ? "" : " disabled"}>${icon(act)}</button>`;
  const prev = b("prev", k > 0, "Previous version"), next = b("next", k < n - 1 || side, k < n - 1 ? "Next version" : "A new version");
  const count = `<span class="pager-n" aria-hidden="true">${k + 1} / ${n}</span>`;
  return side ? `${prev}<span class="pager swipe-end" aria-label="Version ${k + 1} of ${n}">${count}</span>${next}`
    : `<span class="pager" role="group" aria-label="Version ${k + 1} of ${n}">${prev}${count}${next}</span>`;
}
const syncing = new Map<string, Promise<void>>(), syncTimers = new Map<string, Timer>();
const chatOf = (id: string) => (conv.id === id ? conv : chats[id]);
function syncThreadText(id: string): Promise<void> {
  clearTimeout(syncTimers.get(id));
  syncTimers.delete(id);
  const c = chatOf(id), sid = c?.thread?.id, last = c?.messages.at(-1), before = syncing.get(id) ?? Promise.resolve();
  if (!c || !sid || !c.unsynced || runs.get(id) || last?.role !== "assistant") return before;
  const text = last.content;
  const done: Promise<void> = before.then(async () => {
    const r = await fetch(`/api/threads/${sid}/rewrite`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ last: text }) }).catch(() => null);
    if (r?.ok) for (const x of [conv.id === id ? conv : undefined, chats[id]]) if (x?.unsynced && x.messages.findLast((m) => m.role === "assistant" && !m.job && !m.starting)?.content === text) delete x.unsynced;
    if (r?.ok) keepChats();
    if (syncing.get(id) === done) syncing.delete(id);
  });
  syncing.set(id, done);
  return done;
}
function settleSoon(id: string) {
  clearTimeout(syncTimers.get(id));
  syncTimers.set(id, setTimeout(() => void syncThreadText(id), 400));
}
async function renew(i: number) {
  const c = conv, run = running(c), as = c.messages[i].as;
  if (run) {
    if (run.m !== c.messages[i]) return;
    stopRun(run);
    await new Promise<void>((over) => { run.ended = over; });
    if (c !== conv || running(c)) return;
    const all = versionsAt(c, i);
    all.splice(c.messages[i].versions?.shown ?? 0, 1);
    if (all.length) showVersion(c, i, all, all.length - 1);
    else c.messages.splice(i);
  }
  delete c.unsynced;
  void generate(c, i, as, !!c.thread);
}

const stale = (c: Conversation, i: number) => { for (const m of c.messages.slice(i + 1)) delete m.checkpoints; };
function deleteMessage(i: number) {
  const c = conv;
  stale(c, i);
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

function checkpointView(k: Kept): HTMLElement {
  const el = h("details", "checkpoint"), body = h("div", "markdown");
  el.append(h("summary", "", "Context compacted"), body);
  renderBlocks({ body }, k.summary, false);
  return el;
}
const roundViews = (rounds: Round[], marks: Kept[] | undefined, m: Message) => [...rounds.flatMap((r, i) => [...(marks ?? []).filter((k) => k.at === i).map(checkpointView), roundView(r, m)]),
  ...(marks ?? []).filter((k) => k.at >= rounds.length).map(checkpointView)];
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
  if (s === "running") return `running · ${fmtS((performance.now() - (callSince.get(c.id) ?? performance.now())) / 1000)}${c.note ? ` · ${c.note}` : ""}`;
  if (s === "done") return c.ms !== undefined ? `done in ${fmtS(c.ms / 1000)}` : "done";
  return s === "approval" ? "asks to run" : s === "asking" ? "asks you" : s === "failed" ? "failed" : s === "declined" ? "declined" : s === "stopped" ? "stopped" : live ? "asked for" : "not run";
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
  let all: Map<string, Version[]> | undefined;
  const kept = new Map([...box.children].map((el) => [(el as HTMLElement).dataset.call!, el as HTMLElement]));
  calls.forEach((c, k) => {
    let el = kept.get(c.id);
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
    const ver = c.name === "write" || c.name === "edit" ? [...(all ??= artifactsOf(conv)).values()].flat().find((v) => v.block.at === c) : undefined;
    const made = el.querySelector<HTMLElement>(":scope > .art-card");
    if (ver) { const now = artCard(made ?? undefined, ver, all!.get(ver.id)!.length); if (now !== made) head.after(now); }
    else made?.remove();
    const ask = el.querySelector<HTMLElement>(".tool-ask")!, asking = c.state === "approval" && live && !!m.job, form = c.state === "asking" && live && m.job ? c.form : undefined;
    ask.hidden = !asking && !form;
    if (form) {
      const key = `form:${c.id}:${form.error ?? ""}`;
      if (ask.dataset.for !== key) { ask.dataset.for = key; ask.replaceChildren(formView(form)); }
      ask.querySelectorAll<HTMLButtonElement>(".tool-ask-row button").forEach((b) => { b.disabled = deciding.has(c.id); });
    }
    if (asking && ask.dataset.for !== `${c.id}:${deciding.has(c.id)}:${c.asks ?? ""}`) {
      ask.dataset.for = `${c.id}:${deciding.has(c.id)}:${c.asks ?? ""}`;
      const dis = deciding.has(c.id) ? " disabled" : "";
      ask.innerHTML = `<pre class="tool-pre"></pre><div class="tool-ask-row"><span>Run this ${c.readOnly === false ? "tool, which can change things" : "tool"}?</span>` +
        `<button type="button" class="btn" data-act="decide" data-decision="deny"${dis}>Deny</button>` +
        `<button type="button" class="btn" data-act="decide" data-decision="always" title="Run ${escapeHtml(c.title ?? c.name)} without asking from now on (Settings lists it)"${dis}>Always allow</button>` +
        `<button type="button" class="btn primary" data-act="decide" data-decision="once"${dis}>Allow once</button></div>`;
      ask.querySelector("pre")!.textContent = prettyArgs(c.asks ?? c.args);
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
function formView(form: Form) {
  const el = h("form", "ask-form"), rows: { f: Field; row: HTMLElement; read: () => unknown }[] = [], foot = h("div", "tool-ask-row");
  const answer = () => Object.fromEntries(rows.map(({ f, read }) => [f.key, read()]));
  const sync = () => { const now = answer(); for (const { f, row } of rows) row.hidden = !(f.when ?? []).every((w) => (w.op === "eq") === (now[w.key] === w.value)); };
  for (const f of form.fields) {
    if (f.hidden) continue;
    const row = h("div", "ask-field"), text = h("input", "ask-own");
    let read: () => unknown = () => undefined;
    row.append(h("div", "ask-title", f.title ?? f.description ?? f.key));
    if (f.title && f.description) row.append(h("div", "ask-note", f.description));
    text.placeholder = f.placeholder ?? (f.options ? "Something else" : "");
    text.ariaLabel = f.title ?? f.key;
    if (f.type === "external") row.append(Object.assign(h("a", "", f.url ?? ""), { href: f.url ?? "", target: "_blank", rel: "noreferrer" }));
    else if (f.type === "boolean") {
      text.type = "checkbox";
      text.checked = f.default === true;
      row.append(text);
      read = () => text.checked;
    } else if (f.options) {
      const many = f.type === "multiselect", picks = h("div", "ask-options"), first = ([] as unknown[]).concat(f.default ?? []);
      const pressed = () => [...picks.children].filter((b) => b.getAttribute("aria-pressed") === "true");
      for (const o of f.options) {
        const b = h("button", "btn ask-option", o.label);
        b.type = "button";
        b.title = o.description ?? "";
        b.dataset.value = o.value;
        b.setAttribute("aria-pressed", String(first.includes(o.value)));
        b.onclick = () => {
          const on = b.getAttribute("aria-pressed") !== "true";
          if (!many) { for (const x of pressed()) x.setAttribute("aria-pressed", "false"); text.value = ""; }
          b.setAttribute("aria-pressed", String(on));
          sync();
        };
        picks.append(b);
      }
      text.oninput = () => { if (!many && text.value) for (const x of pressed()) x.setAttribute("aria-pressed", "false"); };
      row.append(picks, ...(f.custom ? [text] : []));
      read = () => { const got = pressed().map((b) => (b as HTMLElement).dataset.value!), typed = text.value.trim(); return many ? [...got, ...(typed ? [typed] : [])] : typed || got[0]; };
    } else {
      const number = f.type === "number" || f.type === "integer";
      if (number) text.type = "number";
      text.value = f.default === undefined ? "" : String(f.default);
      row.append(text);
      read = () => (text.value.trim() === "" ? undefined : number ? Number(text.value) : text.value.trim());
    }
    rows.push({ f, row, read });
    el.append(row);
  }
  foot.innerHTML = `<span></span><button type="button" class="btn" data-act="decide" data-decision="deny">Skip</button><button type="submit" class="btn primary">Answer</button>`;
  foot.querySelector("span")!.textContent = form.error ?? "";
  el.append(foot);
  el.oninput = sync;
  el.onsubmit = (e) => {
    e.preventDefault();
    const given = Object.entries(answer()).filter(([key, v]) => !rows.find((r) => r.f.key === key)!.row.hidden && v !== undefined && !(Array.isArray(v) && !v.length));
    const missed = rows.find(({ f, row }) => f.required && !row.hidden && !given.some(([key]) => key === f.key));
    if (missed) return toast(`${missed.f.title ?? missed.f.key} needs an answer`);
    void decide(foot, Object.fromEntries(given));
  };
  sync();
  return el;
}
const prettyArgs = (args: string) => { try { return JSON.stringify(JSON.parse(args), null, 2); } catch { return args; } };

type Act = keyof typeof TITLES | "cont";
const slots = (list: [Act, boolean, string?][]) => list.map(([a, on, html]) => !on ? `<span class="act ghost" aria-hidden="true"></span>`
  : html ?? `<button type="button" class="act" data-act="${a}" title="${TITLES[a as keyof typeof TITLES]}">${icon(a)}</button>`).join("");

function paint(v: View, m: Message, live?: Live) {
  const elapsed = live ? (performance.now() - live.since) / 1000 : 0;
  const shown = `${m.rounds?.length ?? 0}/${m.checkpoints?.length ?? 0}`;
  if (v.roundsShown !== shown) {
    v.rounds.replaceChildren(...roundViews(m.rounds ?? [], m.checkpoints, m));
    v.roundsShown = shown;
  }
  syncCards(v.calls, m.calls ?? [], m, !!live);
  if (v.notes.childElementCount !== (m.notes?.length ?? 0)) v.notes.replaceChildren(...(m.notes ?? []).map((n) => h("div", "note", n)));
  const drawn = live ? pager(m, conv.messages.at(-1) === m && m.role === "assistant") : "";
  if (live && v.foot.dataset.pager !== drawn) v.foot.innerHTML = v.foot.dataset.pager = drawn;
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
  if (m.heardTo !== undefined && !live) requestAnimationFrame(paintUnheard);
  if (m.params) facts.push(m.params);
  if (m.edited) facts.push("edited");
  if (s?.seed !== undefined) facts.push(`<button type="button" class="seed" data-act="seed" data-seed="${s.seed}" title="Use this seed for the next replies">seed ${s.seed}</button>`);
  const last = conv.messages.at(-1) === m, busy = !!running();
  const ranCalls = !!m.calls?.some((c) => c.result !== undefined), canCont = onChrysalis() || ranCalls;
  const cut = m.finish === "length" || m.stopped || (ranCalls && !m.error);
  const cont = cut ? `<button type="button" class="btn" data-act="cont" title="Continue">${icon("cont")}Continue</button>`
    : `<button type="button" class="act" data-act="cont" title="Continue">${icon("cont")}</button>`;
  v.foot.innerHTML = pager(m, last) + `<span class="stats">${facts.join(" · ")}</span>` + slots([["cont", last && !busy && canCont, cont], ["retry", !busy]])
    + slots([["tokens", !!m.id && tokens.has(m.id)], ["speak", !!m.content.trim() && !voice, reading === m ? `<button type="button" class="act on" data-act="speak" title="Stop reading">${icon("stop")}</button>` : undefined],
      ["copy", true], ["edit", !busy], ["trash", !busy]]);
  delete v.foot.dataset.pager;
  v.foot.querySelector("[data-act=tokens]")?.classList.toggle("on", asTokens);
}

const ocean = new Ocean($("sea")), seaReady = ocean.start(), seaSwitch = $<HTMLButtonElement>("sea-switch");
function emptyState() {
  const e = h("div", "empty");
  if (seaReady && document.querySelector("main")!.hasAttribute("data-sea")) ocean.butterfly(e);
  return e;
}
function showSea(on: boolean) {
  document.querySelector("main")!.toggleAttribute("data-sea", on && seaReady);
  seaSwitch.setAttribute("aria-checked", String(on && seaReady));
  thread.querySelector(":scope > .empty")?.replaceWith(emptyState());
}
seaSwitch.disabled = !seaReady;
if (!seaReady) seaSwitch.title = "This browser cannot draw it";
seaSwitch.onclick = () => {
  const on = seaSwitch.getAttribute("aria-checked") !== "true";
  setPref("sea", on ? "on" : "off");
  showSea(on);
};
showSea(pref("sea", "on") === "on");
const PHONE = /iPhone|iPad|Android/.test(navigator.userAgent) || (navigator.maxTouchPoints > 1 && /Mac/.test(navigator.platform));
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
  const [p, oc] = await Promise.all(["/api/pair", "/api/opencode"].map((u) => fetch(u).then((r) => r.ok ? r.json() : null).catch(() => null)));
  if (!connectView.open) return;
  clearTimeout(connectTimer);
  connectTimer = window.setTimeout(renderConnect, 2000);
  const box = $("connect-body"), computers = connectMode === "computer", host = status.machine ?? "the computer Chrysalis runs on";
  if (status.here && !p) {
    box.dataset.key = "";
    return void box.replaceChildren(h("p", "set-note", "Cannot reach Chrysalis's web server. Trying again…"));
  }
  if (status.here ? !p.lan : !oc?.line) {
    const note = h("p", "set-note");
    note.append("Chrysalis is on this computer only. Type ", h("code", "", "network"), " in Chrysalis's terminal to let other devices on this network in.");
    box.dataset.key = "";
    return void box.replaceChildren(note);
  }
  if (computers) await loadThreads();
  const list: Paired[] = p?.devices ?? [], asking: Joining[] = p?.waiting ?? [], on = harness.machines ?? [];
  const ids = computers ? on.map((m) => m.id) : list.map((d) => d.id), fresh = known && ids.find((id) => !known!.has(id));
  if (fresh) joined = computers ? on.find((m) => m.id === fresh)!.name : list.find((d) => d.id === fresh)!.name;
  known = new Set(ids);
  const key = JSON.stringify([connectMode, joined, asking, list.map((d) => [d.id, since(d.seen)]), on, p?.connect, !!p?.qr, oc?.line]);
  if (box.dataset.key === key) return;
  box.dataset.key = key;
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
  if (asking.length) asks.append(h("p", "set-note", "Allow only a code your own device is showing."));
  const entry = (name: string, note: string, id?: string) => {
    const item = h("div", "dev-item");
    item.append(h("span", "dev-name", name), h("span", "dev-when", note));
    if (id) {
      const remove = h("button", "btn small", "Remove");
      remove.type = "button";
      remove.onclick = async () => { await fetch(`/api/pair/${id}`, { method: "DELETE" }); box.dataset.key = ""; void renderConnect(); };
      item.append(remove);
    }
    return item;
  };
  const done = h("p", "dev-joined");
  done.innerHTML = icon("check");
  if (computers) {
    done.append(`${joined} is under Run on.`);
    const add = h("div", "dev-list"), copyIt = h("button", "btn small dev-copy", "Copy"), all = h("div", "dev-list");
    copyIt.type = "button";
    copyIt.onclick = () => copy(oc.line, copyIt);
    add.append(h("p", "dev-lead", "On that computer, with OpenCode installed, paste this into a terminal:"), h("code", "dev-url", oc.line), copyIt,
      ...(asking.length ? [] : [h("p", "set-note", status.here ? "Its code then shows here to allow." : `Then allow its code on ${host}.`)]));
    all.append(h("div", "set-title", "Run on"));
    for (const m of on) all.append(entry(m.name, m.id ? "connected" : "Chrysalis runs here", m.id || undefined));
    for (const d of list) if (d.kind === "computer" && !on.some((m) => m.id === d.id)) all.append(entry(d.name, `not connected · used ${since(d.seen)}`, d.id));
    return void box.replaceChildren(...(joined ? [done] : []), add, ...(asking.length ? [asks] : []), all);
  }
  done.append(`${joined} is connected.`);
  const row = h("div", "dev-row"), text = h("div", "dev-text"), other = h("p");
  const qr = h("img", "dev-qr");
  qr.src = `data:image/svg+xml;base64,${btoa(p.qr)}`;
  qr.alt = "QR code of Chrysalis's address on this network";
  row.append(qr);
  other.append("No camera? On the device, open ", h("code", "dev-url", p.connect[0].replace(/\/$/, "")), ".");
  text.append(h("p", "dev-lead", "Scan this with your phone. Go on past its one warning that the connection is not private, then allow the code it shows."), other);
  row.append(text);
  const devs = h("div", "dev-list"), browsers = list.filter((d) => d.kind !== "computer");
  devs.append(h("div", "set-title", "Let in"));
  if (!browsers.length) devs.append(h("p", "set-note", "None yet."));
  for (const d of browsers) devs.append(entry(d.name, `added ${day(d.added)} · used ${since(d.seen)}`, d.id));
  const fp = h("p", "set-note dev-fp");
  fp.append(`To be sure a device trusts this computer and not an impostor on the network, its certificate details show the same SHA-256 fingerprint for ${p.name}: `,
    h("code", "", p.fingerprint));
  box.replaceChildren(...(joined ? [done] : []), row, ...(asking.length ? [asks] : []), devs, fp);
}
let connectMode: "devices" | "computer" = "devices";
function openConnect(mode: "devices" | "computer" = "devices") {
  connectMode = mode;
  joined = "";
  known = undefined;
  $("connect-body").dataset.key = "";
  $("connect-title").textContent = mode === "computer" ? "Add a computer" : "Phones and browsers";
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


const draftTitle = () => (drafting ? "Stop the draft" : input.value.trim() ? "Finish my message (draft)" : "Draft my next message");
function renderComposer() {
  const busy = !!running();
  sendBtn.innerHTML = icon(busy || voice ? "stop" : "send");
  sendBtn.setAttribute("aria-label", voice ? "End voice mode" : busy ? "Stop" : "Send");
  sendBtn.title = voice ? "End voice mode" : "";
  sendBtn.classList.toggle("stop", busy || !!voice);
  const draftBtn = $<HTMLButtonElement>("draft");
  draftBtn.innerHTML = icon(drafting ? "stop" : "wand");
  draftBtn.title = draftTitle();
  draftBtn.setAttribute("aria-label", draftBtn.title);
  draftBtn.classList.toggle("on", !!drafting);
  $("composer").classList.toggle("drafting", !!drafting);
  renderControls();
  if (modeOf(conv) !== chatMode) {
    chatMode = modeOf(conv);
    syncSettings();
  }
}

const customized = () => Object.keys(settings.think).length + Object.keys(settings.fast).length > 0
  || !!settings.max_tokens || settings.seed !== undefined
  || Object.keys(settings.voice ?? {}).length > 0;

function putAway(c: Conversation, on: boolean, quiet = false) {
  const sid = c.thread?.id, t = harness.threads?.find((x) => x.id === sid);
  if (sid) {
    if (t) t.archived = on;
    void fetch(`/api/threads/${sid}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ archived: on }) })
      .then((r) => { if (!r.ok) toast("OpenCode did not change that thread"); return loadThreads(); });
    renderList();
    if (!quiet) toast(on ? `${c.title} is settled` : `${c.title} is active again`, () => putAway(c, !on, true));
    return;
  }
  const now = new Set(kept("settled"));
  if (on) now.add(keyOf(c)); else now.delete(keyOf(c));
  setPref("settled", JSON.stringify([...now]));
  renderList();
  if (!quiet) toast(on ? `${c.title} is settled` : `${c.title} is active again`, () => putAway(c, !on, true));
}
let sorting = false, listStale = false, sideQuery = "", sideAt = 0;
const fold = (t: string) => t.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase().trim();
const everyChat = () => [...new Map([...Object.values(chats), ...stubs()].filter((c) => !c.incognito && (c.messages.length || c.thread?.id)).map((c) => [keyOf(c), c])).values()];
const scopeOf = () => {
  const at = pref("scope", "");
  return at && (at === CHRYSALIS || people.has(at) || groups.has(at) || dirs().includes(at.slice(DIR.length))) ? at : "";
};
const inScope = (c: Conversation, at = scopeOf()) => !at || whoOf(c) === at;
function found() {
  const q = fold(sideQuery), score = (c: Conversation) => {
    const title = fold(c.title);
    return title.startsWith(q) ? 0 : title.includes(q) ? 1 : fold(nameOf(whoOf(c))).includes(q) || fold(c.thread?.dir ?? "").includes(q) || fold(machineOf(c.thread?.machine)?.name ?? "").includes(q) ? 2 : -1;
  };
  return everyChat().map((c) => ({ c, at: score(c) })).filter((x) => x.at >= 0 && inScope(x.c)).sort((a, b) => a.at - b.at || b.c.updated - a.c.updated).map((x) => x.c);
}
const isRow = (key: string) => key === CHRYSALIS || key.startsWith(DIR) || groups.has(key) || people.has(key);
function rowOrder(): string[] {
  if (pref("chrysalis-gone", "") && chatsOf(CHRYSALIS).some((c) => c.messages.length || c.thread?.id)) setPref("chrysalis-gone", "");
  const all = [...[...people.values()].filter((p) => p.kind !== "persona" && !unwritten(p) && !(p.id === CHRYSALIS && pref("chrysalis-gone", ""))).map((p) => ({ id: p.id, at: p.added })), ...[...groups.values()].map((g) => ({ id: g.id, at: g.added })), ...dirs().map((d) => ({ id: DIR + d, at: 0 }))];
  const was = kept("rows"), fresh = all.filter((x) => !was.includes(x.id)).sort((a, b) => (chatsOf(b.id)[0]?.updated ?? b.at) - (chatsOf(a.id)[0]?.updated ?? a.at)).map((x) => x.id);
  if (fresh.length) setPref("rows", JSON.stringify([...fresh, ...was]));
  const now = [...fresh, ...was];
  return all.map((x) => x.id).sort((a, b) => now.indexOf(a) - now.indexOf(b));
}
function keepRow(key: "pinned" | "settled", id: string, on: boolean) {
  for (const k of ["pinned", "settled"] as const) setPref(k, JSON.stringify([...(on && k === key ? [id] : []), ...kept(k).filter((x) => x !== id)]));
  renderList();
  const name = nameOf(id) || "Unnamed";
  toast(key === "pinned" ? (on ? `${name} is pinned` : `${name} is unpinned`) : on ? `${name} is settled` : `${name} is active again`, () => keepRow(key, id, !on));
}
function renameRow(id: string, value: string) {
  const p = people.get(id), g = groups.get(id), name = value.trim().slice(0, 100);
  if (p && name) { p.card.data.name = name; void savePerson(p); }
  if (g) { if (name) g.name = name; else delete g.name; keepGroup(g); }
  renderCast();
  render();
}
function forkRow(id: string) {
  const p = people.get(id), g = groups.get(id), taken = new Set([...[...people.values()].map((x) => x.card.data.name.trim()), ...[...groups.keys()].map(nameOf)]);
  let n = 2;
  while (taken.has(`${nameOf(id)} ${n}`)) n++;
  const name = `${nameOf(id)} ${n}`;
  let made = "";
  if (p) {
    const copy: Person = { id: randomId(), kind: "character", card: structuredClone(p.card), ...(p.image ? { image: p.image } : {}), added: Date.now() };
    copy.card.data.name = name;
    people.set(copy.id, copy);
    setFace(copy);
    void savePerson(copy);
    made = copy.id;
  } else if (g) made = makeGroup([...g.members], { name, ...(g.scenario ? { scenario: g.scenario } : {}) }).id;
  if (!made) return;
  placeRow(made, id, true);
  renderCast();
  toast(`${name} is a copy of ${nameOf(id)}`, () => void forget(made).then((f) => f.done()));
}
function renderList() {
  if (sorting) return void (listStale = true);
  const scope = scopeOf(), scopeBtn = $("side-scope");
  scopeBtn.classList.toggle("on", !!scope);
  scopeBtn.title = scopeBtn.ariaLabel = scope ? `Showing ${nameOf(scope)} only` : "Filter";
  $("side-clear").hidden = !sideQuery;
  if (fold(sideQuery)) {
    const hits = found(), box = h("div", "folder results open"), kids = h("div", "folder-chats");
    sideAt = Math.max(0, Math.min(sideAt, hits.length - 1));
    kids.id = "side-results";
    kids.setAttribute("role", "listbox");
    kids.append(...hits.map((c, k) => {
      const row = convRow(c, false, true);
      row.id = `side-result-${k}`;
      row.setAttribute("role", "option");
      row.setAttribute("aria-selected", String(k === sideAt));
      row.classList.toggle("sel", k === sideAt);
      return row;
    }));
    box.append(hits.length ? kids : h("p", "side-none", "No threads match"));
    $("side-search").setAttribute("aria-expanded", String(hits.length > 0));
    $("side-pins").replaceChildren();
    $("side-shelf").replaceChildren();
    return void $("side-list").replaceChildren(box);
  }
  $("side-search").setAttribute("aria-expanded", "false");
  const settled = new Set([...kept("settled"), ...(harness.threads ?? []).filter((t) => t.archived).map((t) => STUB + t.id)]), shelf: Conversation[] = [], pins: Conversation[] = [];
  const order = kept("pinned"), pinned = new Set(order), rowAt = rowOrder();
  const shown = (id: string) => chatsOf(id).filter((c) => (c.messages.length || c.thread?.id) && inScope(c, scope) && (!settled.has(keyOf(c)) || (shelf.push(c), false)) && (!pinned.has(keyOf(c)) || (pins.push(c), false)));
  const boxes = rowAt.map((id) => ({ id, list: shown(id) })).filter((x) => !scope || x.id === scope).map(({ id, list }) => {
    const open = unfolded.has(id) && list.length > 0, box = h("div", `folder${open ? " open" : ""}`);
    box.dataset.who = id;
    box.append(castRow(id, list, open));
    if (open) {
      const kids = h("div", "folder-chats");
      kids.append(...list.map((c) => convRow(c)));
      box.append(kids);
    }
    return { id, box };
  });
  const held = boxes.filter((x) => pinned.has(x.id)).sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id)).map((x) => x.box), away = boxes.filter((x) => !pinned.has(x.id) && settled.has(x.id)).map((x) => x.box);
  const rows = boxes.filter((x) => !pinned.has(x.id) && !settled.has(x.id)).map((x) => x.box);
  const above = h("div", pins.length || held.length ? "folder pinned open" : "pin-none"), below = h("div", shelf.length || away.length ? "folder settled" : "shelf-none");
  if (pins.length || held.length) {
    const kids = h("div", "folder-chats pin-chats");
    kids.append(...pins.sort((a, b) => order.indexOf(keyOf(a)) - order.indexOf(keyOf(b))).map((c) => convRow(c, false, true)));
    above.append(h("div", "settled-head pin-head", "Pinned"), ...held, kids);
  } else above.append(h("div", "pin-head", "Pinned"));
  if (shelf.length || away.length) {
    const open = pref("settled-open", "") === "1", head = h("button", "settled-head");
    below.classList.toggle("open", open);
    head.type = "button";
    head.setAttribute("aria-expanded", String(open));
    head.innerHTML = `<span>Settled</span><span class="cast-count">${shelf.length + away.length}</span>${icon("chevron")}`;
    head.onclick = () => { setPref("settled-open", open ? "" : "1"); renderList(); };
    below.append(head);
    if (open) {
      const kids = h("div", "folder-chats shelf-chats");
      kids.append(...shelf.sort((a, b) => b.updated - a.updated).map((c) => convRow(c, true, true)));
      below.append(...away, kids);
    }
  } else below.append(h("div", "shelf-head", "Settled"));
  $("side-pins").replaceChildren(above);
  $("side-shelf").replaceChildren(below);
  $("side-list").replaceChildren(...rows);
}
interface PaletteItem { icon: string | HTMLElement; title: string; note?: (string | HTMLElement)[]; keys?: string; run: () => void }
interface PaletteGroup { label: string; items: PaletteItem[] }
const palette = $<HTMLDialogElement>("palette");
let paletteGroups: PaletteGroup[] = [], paletteAt = 0, paletteBack: (() => void) | undefined;
const paletteShown = () => {
  const q = fold($<HTMLInputElement>("palette-input").value);
  return paletteGroups.map((g) => ({ ...g, items: g.items.filter((it) => !q || fold(`${it.title} ${(it.note ?? []).map((n) => (typeof n === "string" ? n : n.textContent)).join(" ")}`).includes(q)) })).filter((g) => g.items.length);
};
function paintPalette() {
  const groups = paletteShown(), flat = groups.flatMap((g) => g.items), list = $("palette-list");
  paletteAt = Math.max(0, Math.min(paletteAt, flat.length - 1));
  let k = 0;
  list.replaceChildren(...(flat.length ? groups.flatMap((g) => [h("div", "palette-label", g.label), ...g.items.map((it) => {
    const at = k++, b = h("button", at === paletteAt ? "palette-item on" : "palette-item"), text = h("span", "palette-text"), title = h("span", "palette-title", it.title);
    b.type = "button";
    b.id = `palette-item-${at}`;
    b.setAttribute("role", "option");
    b.setAttribute("aria-selected", String(at === paletteAt));
    const pic = h("span", "palette-icon");
    if (typeof it.icon === "string") pic.innerHTML = it.icon; else pic.append(it.icon);
    text.append(title);
    if (it.note?.length) {
      const note = h("span", "palette-note");
      it.note.forEach((n, j) => { if (j) note.append(h("span", "palette-dot", "·")); note.append(n); });
      text.append(note);
    }
    b.append(pic, text);
    if (it.keys) b.append(h("kbd", "palette-keys", it.keys));
    b.onmousemove = () => { if (paletteAt !== at) { paletteAt = at; paintPalette(); } };
    b.onclick = () => { palette.close(); it.run(); };
    return b;
  })]) : [h("div", "palette-none", "No results")]));
  document.getElementById(`palette-item-${paletteAt}`)?.scrollIntoView({ block: "nearest" });
}
function openPalette(groups: PaletteGroup[], back?: () => void) {
  paletteGroups = groups;
  paletteBack = back;
  paletteAt = 0;
  $<HTMLInputElement>("palette-input").value = "";
  if (!palette.open) palette.showModal();
  paintPalette();
  $("palette-input").focus();
}
const MOD = /Mac|iP(hone|ad)/.test(navigator.platform) ? "⌘" : "Ctrl+";
const where = (dir: string, machine?: string, lead = ""): (string | HTMLElement)[] => {
  const m = (harness.machines?.length ?? 0) > 1 ? machineOf(machine) : undefined, on = h("span", "palette-on");
  if (m) { on.innerHTML = icon(machineIcon(m.kind)); on.append(m.name); }
  return [...(lead ? [lead] : []), ...(m ? [on] : []), dir];
};
const NO_PROJECT = '<svg class="i" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3a2 2 0 0 0-2 2M19 3a2 2 0 0 1 2 2M21 19a2 2 0 0 1-2 2M5 21a2 2 0 0 1-2-2M9 3h1M9 21h1M14 3h1M14 21h1M3 9v1M21 9v1M3 14v1M21 14v1"/></svg>';
function newThreadIn() {
  const now = conv.thread && conv.thread.dir !== homeOf(conv.thread.machine) ? conv.thread.dir : "", all = [...(now ? [now] : []), ...dirs().filter((d) => d !== now)];
  const plain = (harness.machines ?? [{ id: "", name: "", home: harness.home ?? "", kind: "" }]).map((m): PaletteItem => ({ icon: NO_PROJECT, title: "No project", note: where(m.home, m.id),
    run: () => { setPref("machine", m.id); newChat(); } }));
  const projects = [...all.slice(0, now ? 1 : 0), ...plain.slice(0, 1), ...all.slice(now ? 1 : 0), ...plain.slice(1)].map((x, k): PaletteItem => (typeof x === "string"
    ? { icon: face(DIR + x), title: dirName(x), note: where(x, dirMachine(x), x === now ? "Current" : ""), keys: k < 9 ? `${MOD}${k + 1}` : undefined, run: () => newChat(DIR + x) }
    : { ...x, keys: k < 9 ? `${MOD}${k + 1}` : undefined }));
  const cast = [...[...people.values()].filter((p) => p.kind === "character").map((p) => p.id), ...groups.keys()].map((id): PaletteItem => ({ icon: face(id), title: nameOf(id), run: () => newChat(id) }));
  openPalette([{ label: "Projects", items: harness.state === "on" ? projects : [{ icon: NO_PROJECT, title: "No project", run: () => newChat() }] }, ...(cast.length ? [{ label: "Characters", items: cast }] : [])]);
}
function addProject() {
  if (harness.state !== "on") return void openHarness();
  const browse = (id: string) => {
    setPref("machine", id);
    newChat();
    if (!conv.thread || (conv.thread.machine ?? "") !== id) conv.thread = freshThread();
    renderCast();
    renderControls();
    openFolders($("add-project"), { from: conv.thread.dir, to: (dir) => void moveThread(conv, dir) });
  };
  const all = harness.machines ?? [];
  if (all.length < 2) return browse("");
  openPalette([{ label: "Environments", items: all.map((m): PaletteItem => ({ icon: icon(machineIcon(m.kind)), title: m.name, note: [m.id ? m.home : "Chrysalis runs here"], run: () => browse(m.id) })) }]);
}
$<HTMLInputElement>("palette-input").oninput = () => { paletteAt = 0; paintPalette(); };
$("palette-back").onclick = () => { palette.close(); paletteBack?.(); };
palette.addEventListener("click", (e) => { if (e.target === palette) palette.close(); });
palette.addEventListener("keydown", (e) => {
  const flat = paletteShown().flatMap((g) => g.items), n = flat.length, input = $<HTMLInputElement>("palette-input");
  if ((e.key === "ArrowDown" || e.key === "ArrowUp") && n) { e.preventDefault(); paletteAt = (paletteAt + (e.key === "ArrowDown" ? 1 : n - 1)) % n; paintPalette(); }
  else if (e.key === "Enter" && n) { e.preventDefault(); document.getElementById(`palette-item-${paletteAt}`)?.click(); }
  else if (e.key === "Backspace" && !input.value) { e.preventDefault(); palette.close(); paletteBack?.(); }
  else if ((e.metaKey || e.ctrlKey) && /^[1-9]$/.test(e.key)) {
    const hit = flat.find((it) => it.keys === `${MOD}${e.key}`);
    if (hit) { e.preventDefault(); palette.close(); hit.run(); }
  }
});
function ago(t: number) {
  const s = (Date.now() - t) / 1000;
  return s < 60 ? "now" : s < 3600 ? `${Math.floor(s / 60)}m` : s < 86400 ? `${Math.floor(s / 3600)}h` : s < 7 * 86400 ? `${Math.floor(s / 86400)}d`
    : s < 30 * 86400 ? `${Math.floor(s / 604800)}w` : new Date(t).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
function dropChat(c: Conversation) {
  const sid = c.thread?.id;
  if (sid) {
    harness.threads = harness.threads?.filter((t) => t.id !== sid);
    void fetch(`/api/threads/${sid}`, { method: "DELETE" }).then((r) => { if (!r.ok) toast("OpenCode did not delete that chat"); return loadThreads(); });
  }
  remove(c.id);
}
function pin(c: Conversation, on: boolean, at = 0) {
  const now = kept("pinned").filter((k) => k !== keyOf(c)), among = now.filter((k) => !isRow(k));
  if (on) now.splice(at < among.length ? now.indexOf(among[at]) : now.length, 0, keyOf(c));
  setPref("pinned", JSON.stringify(now));
  renderList();
}
const canMove = (c: Conversation, who: string) => {
  if (who === whoOf(c) || runs.get(c.id)) return false;
  if (who.startsWith(DIR)) return !!c.thread && (dirMachine(who.slice(DIR.length)) ?? "") === (c.thread.machine ?? "");
  return who === CHRYSALIS || people.get(who)?.kind === "character" || groups.has(who);
};
async function moveChat(c: Conversation, who: string) {
  if (who.startsWith(DIR)) return moveThread(c, who.slice(DIR.length));
  const mine = c.id === conv.id ? conv : chats[c.id] ?? c;
  if (who === CHRYSALIS) delete mine.who; else mine.who = who;
  stow(mine);
  unfold(who);
  if (mine.thread?.id) {
    const r = await fetch(`/api/threads/${mine.thread.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scene: sceneFor(mine) ?? null }) }).catch(() => null);
    if (!r?.ok) toast("OpenCode did not move that chat");
    await loadThreads();
  }
  renderCast();
  renderList();
  if (mine.id === conv.id) render();
}
const chatAt = (row: HTMLElement) => everyChat().find((c) => keyOf(c) === row.dataset.key);
const dropVerb = (row: HTMLElement, drop: Drop): string | null => {
  const person = row.dataset.person, from = groups.get(row.dataset.group ?? "");
  if (person && from) return drop.kind === "out" && canLeave(from) ? (from.members.length > 2 ? "Take out" : "Split up") : null;
  if (person) {
    const pinned = kept("pinned").includes(person), settled = kept("settled").includes(person);
    if (drop.kind === "folder") {
      const g = groups.get(drop.who);
      if (!people.has(person)) return null;
      if (g) return g.members.includes(person) ? null : `Join ${nameOf(g.id)}`;
      if (!castable(drop.who) || drop.who === person) return null;
      const theirs = groupOf([person, drop.who]);
      return theirs ? `Open ${nameOf(theirs.id)}` : `Group with ${nameOf(drop.who)}`;
    }
    if (drop.kind === "chat") return !people.has(person) || running() || castOf(conv).members.some((m) => m.id === person) || (person === CHRYSALIS && !folderOf(conv)) ? null
      : folderOf(conv) || conv.messages.length ? "Join this chat" : `${nameOf(person)}'s chat`;
    return drop.kind === "pin" ? (pinned ? "" : "Pin") : drop.kind === "unpin" ? "Unpin" : drop.kind === "settle" ? (settled ? "" : "Settle") : drop.kind === "active" ? "Un-settle" : null;
  }
  const c = chatAt(row), isPinned = !!c && kept("pinned").includes(keyOf(c));
  if (!c) return null;
  const away = !!row.closest(".shelf-chats");
  if (away) return drop.kind === "pin" ? "Pin" : drop.kind === "active" ? "Un-settle" : null;
  return drop.kind === "pin" ? (isPinned ? "" : "Pin") : drop.kind === "unpin" ? "Unpin" : drop.kind === "settle" ? "Settle" : drop.kind === "folder" && canMove(c, drop.who) ? nameOf(drop.who) : null;
};
sidebarDrag($("conversations"), { pins: $("side-pins"), list: $("side-list"), shelf: $("side-shelf") }, {
  verb: dropVerb,
  started: () => { sorting = true; },
  ended: () => { sorting = false; listStale = false; renderList(); },
  done: (row, drop) => {
    const person = row.dataset.person, from = groups.get(row.dataset.group ?? "");
    if (person && from) return void (drop.kind === "out" && takeOut(from, person));
    if (person) {
      if (drop.kind === "folder") together(person, drop.who);
      else if (drop.kind === "chat") join([person]);
      else if (drop.kind === "pin" || drop.kind === "unpin") keepRow("pinned", person, drop.kind === "pin");
      else if (drop.kind === "settle" || drop.kind === "active") keepRow("settled", person, drop.kind === "settle");
      return;
    }
    const c = chatAt(row);
    if (!c) return;
    if (row.closest(".shelf-chats")) putAway(c, false, true);
    if (drop.kind === "pin") pin(c, true, drop.index);
    else if (drop.kind === "unpin") pin(c, false);
    else if (drop.kind === "settle") putAway(c, true);
    else if (drop.kind === "folder") void moveChat(c, drop.who);
  },
});
function rename(c: Conversation, title: string) {
  title = title.trim().slice(0, 100);
  if (!title || title === c.title) return renderList();
  const mine = chats[c.id] ?? (c.id === conv.id ? conv : undefined), sid = c.thread?.id, t = harness.threads?.find((x) => x.id === sid);
  for (const x of [c, mine, c.id === conv.id ? conv : undefined]) if (x) Object.assign(x, { title, named: true });
  if (t) t.title = title;
  if (sid) void fetch(`/api/threads/${sid}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title }) })
    .then((r) => { if (!r.ok) toast("OpenCode did not rename that chat"); });
  keepChats();
  if (c.id === conv.id) $("title").textContent = document.title = title;
  renderList();
}
interface MenuItem { label: string; icon: keyof typeof ICONS; run: () => void; danger?: boolean; quiet?: boolean }
let menuAt = 0;
function openMenu(x: number, y: number, items: MenuItem[], head = "") {
  const menu = $("row-menu");
  menu.replaceChildren(...(head ? [h("div", "menu-head", head)] : []), ...items.map((it) => {
    const b = h("button", it.danger ? "danger" : it.quiet ? "quiet" : "");
    b.type = "button";
    b.setAttribute("role", "menuitem");
    b.innerHTML = `${icon(it.icon)}<span></span>`;
    b.querySelector("span")!.textContent = it.label;
    b.onclick = () => {
      menu.hidden = true;
      it.run();
    };
    return b;
  }));
  menu.hidden = false;
  menu.style.bottom = "auto";
  menu.style.left = `${Math.max(8, Math.min(x, innerWidth - menu.offsetWidth - 8))}px`;
  menu.style.top = `${Math.max(8, Math.min(y, innerHeight - menu.offsetHeight - 8))}px`;
  menuAt = performance.now();
}
function menuOf(at: string | Conversation, away = false): MenuItem[] {
  const c = typeof at === "string" ? undefined : at, id = c ? whoOf(c) : at as string, key = c ? keyOf(c) : id, sid = c?.thread?.id;
  const folder = id.startsWith(DIR), mine = !c && (groups.has(id) || castable(id)), pinned = kept("pinned").includes(key), settled = c ? away : kept("settled").includes(id);
  const fork = () => void fetch(`/api/threads/${sid}/fork`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })
    .then(async (r) => { if (!r.ok) return toast("OpenCode did not fork that chat"); const { id } = await r.json(); await loadThreads(); location.hash = `#${STUB}${id}`; });
  const items: (MenuItem | false | undefined)[] = [
    { label: folder ? `New thread in ${nameOf(id)}` : id === CHRYSALIS ? "New chat" : `New chat with ${nameOf(id) || "Unnamed"}`, icon: "compose", run: () => newChat(id) },
    { label: pinned ? "Unpin" : "Pin", icon: "pin", run: () => (c ? pin(c, !pinned) : keepRow("pinned", id, !pinned)) },
    { label: settled ? "Un-settle" : "Settle", icon: settled ? "retry" : "check", run: () => (c ? putAway(c, !settled) : keepRow("settled", id, !settled)) },
    (c || !folder) && { label: "Rename", icon: "edit", run: () => { renaming = key; renderList(); } },
    (!!sid || mine) && { label: "Fork", icon: "fork", run: () => (c ? fork() : forkRow(id)) },
    !c && (folder ? { label: "Copy path", icon: "copy", run: () => void navigator.clipboard.writeText(id.slice(DIR.length)).then(() => toast("Copied")) } : { label: "Card", icon: "agent", run: () => openSheet(id) }),
    c ? { label: "Delete", icon: "trash", danger: true, run: () => dropChat(c) } : folder ? { label: "Close folder", icon: "x", danger: true, run: () => closeFolder(id.slice(DIR.length)) }
      : mine && { label: "Delete", icon: "trash", danger: true, run: () => void drop(id) },
  ];
  return items.filter((x): x is MenuItem => !!x);
}
function menuOn(el: HTMLElement, items: () => MenuItem[]) {
  let timer = 0, from = [0, 0], held = false;
  el.addEventListener("contextmenu", (e) => { e.preventDefault(); if ($("row-menu").hidden || performance.now() - menuAt > 400) openMenu(e.clientX, e.clientY, items()); });
  el.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse") return;
    from = [e.clientX, e.clientY];
    held = false;
    timer = window.setTimeout(() => { timer = 0; held = true; openMenu(from[0], from[1], items()); }, 280);
  });
  el.addEventListener("pointermove", (e) => { if (timer && Math.hypot(e.clientX - from[0], e.clientY - from[1]) > 8) { clearTimeout(timer); timer = 0; } });
  for (const end of ["pointerup", "pointercancel", "pointerleave"]) el.addEventListener(end, () => { clearTimeout(timer); timer = 0; });
  el.addEventListener("click", (e) => { if (held) { held = false; e.preventDefault(); e.stopImmediatePropagation(); } }, true);
}
let renaming = "";
function convRow(s: Conversation, settled = false, faced = false) {
  const a = h("a", `conv${s.id === conv.id ? " active" : ""}${settled ? "" : " card"}`), put = h("button", "conv-settle"), title = h("span", "conv-title", s.title);
  put.type = "button";
  put.title = put.ariaLabel = settled ? "Make it active again" : "Settle: put it away";
  put.innerHTML = `${icon(settled ? "retry" : "check")}${settled ? "Un-settle" : "Settle"}`;
  put.onclick = (e) => { e.preventDefault(); putAway(s, !settled); };
  a.href = `#${s.id}`;
  a.dataset.key = keyOf(s);
  a.draggable = false;
  a.onclick = (e) => { if (!e.defaultPrevented) { mark(s); closeOver(panes.conversations); if (sideQuery) clearSearch(); else renderList(); } };
  menuOn(a, () => menuOf(s, settled));
  const st = statusOf(s), who = whoOf(s), pinned = kept("pinned").includes(keyOf(s));
  if (st) a.classList.add("fresh");
  const pic = face(who);
  pic.classList.add("conv-face");
  if (!who.startsWith(DIR)) {
    pic.dataset.card = who;
    pic.title = `${nameOf(who)}'s card`;
    pic.setAttribute("role", "button");
    pic.onclick = (e) => e.preventDefault();
  }
  const label = st ? statusLabel(st) : h("span", "conv-ago", ago(s.updated));
  let body: HTMLElement = title;
  if (renaming === keyOf(s)) {
    const box = h("input", "conv-rename");
    box.value = s.title;
    box.ariaLabel = "Chat title";
    box.onclick = (e) => e.preventDefault();
    box.onkeydown = (e) => { if (e.key === "Enter") { renaming = ""; rename(s, box.value); } else if (e.key === "Escape") { renaming = ""; renderList(); } e.stopPropagation(); };
    box.onblur = () => { if (renaming === keyOf(s)) { renaming = ""; rename(s, box.value); } };
    queueMicrotask(() => { box.focus(); box.select(); });
    body = box;
  }
  if (settled) {
    if (faced) a.append(pic);
    a.append(body);
    if (body === title) a.append(label, put);
    return a;
  }
  const top = h("div", "conv-top"), slot = h("span", "conv-slot"), mid = h("div", "conv-mid");
  top.append(pic);
  if (pinned) top.insertAdjacentHTML("beforeend", `<span class="conv-pin">${icon("pin")}</span>`);
  slot.append(label, put);
  top.append(h("span", "conv-who", nameOf(who)), slot);
  mid.append(body);
  a.append(top, mid);
  const where = s.thread?.dir && !who.startsWith(DIR) ? dirName(s.thread.dir) : "", model = s.thread?.model?.split("/").pop() ?? "";
  const at = s.thread && (harness.machines?.length ?? 0) > 1 ? machineOf(s.thread.machine) : undefined;
  if (where || model || at) {
    const meta = h("div", "conv-meta");
    if (at) {
      meta.insertAdjacentHTML("beforeend", icon(machineIcon(at.kind)));
      meta.append(Object.assign(h("span", "conv-mach", at.name), { title: `Runs on ${at.name}` }));
    }
    if (where) meta.insertAdjacentHTML("beforeend", icon("folder"));
    meta.append(h("span", "conv-where", where), h("span", "conv-model", model));
    a.append(meta);
  }
  return a;
}
const workingFor = (ms: number) => {
  const sec = Math.max(0, Math.floor(ms / 1000)), min = Math.floor(sec / 60);
  return sec < 60 ? `${sec}s` : min < 60 ? `${min}m` : `${Math.floor(min / 60)}h ${min % 60}m`;
};
const STATUS = { working: ["Working", "working"], approval: ["Approval", "approval"], input: ["Input", "input"], failed: ["Failed", "failed"], done: ["Done", "done"] } as const;
const unseen = (): Record<string, "done" | "failed"> => { try { return JSON.parse(pref("unseen", "{}")) ?? {}; } catch { return {}; } };
function mark(c: Conversation, as?: "done" | "failed") {
  if (c.incognito) return;
  const now = unseen(), key = keyOf(c);
  if (now[key] === as) return;
  if (as) now[key] = as; else delete now[key];
  setPref("unseen", JSON.stringify(now));
}
function statusOf(c: Conversation): { kind: keyof typeof STATUS; since?: number } | null {
  const r = runs.get(c.id), away = !r && c.thread?.id ? harness.threads?.find((t) => t.id === c.thread!.id && t.job) : undefined;
  if (r?.asking) return { kind: r.m.calls?.some((x) => x.state === "asking") ? "input" : "approval" };
  if (r || away) return { kind: "working", since: r?.begun ?? away!.updated };
  const was = unseen()[keyOf(c)];
  return was ? { kind: was } : null;
}
function statusLabel(st: { kind: keyof typeof STATUS; since?: number }) {
  const el = h("span", `conv-status ${st.kind}`);
  el.innerHTML = icon(STATUS[st.kind][1]);
  el.append(h("span", "", STATUS[st.kind][0]));
  if (st.since !== undefined) {
    const dur = h("span", "conv-dur", workingFor(Date.now() - st.since));
    dur.dataset.since = String(st.since);
    dur.ariaHidden = "true";
    el.append(dur);
    workingTick ||= window.setInterval(tickWorking, 1000);
  }
  return el;
}
let workingTick = 0;
function tickWorking() {
  const els = document.querySelectorAll<HTMLElement>(".conv-dur[data-since]");
  for (const el of els) el.textContent = workingFor(Date.now() - Number(el.dataset.since));
  if (els.length) return;
  clearInterval(workingTick);
  workingTick = 0;
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

async function start(get = false, model = picked) {
  const r = await fetch("/api/chrysalis/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model, ...(get ? { get: true } : {}) }) });
  return r.status === 409 ? ((await r.json().catch(() => ({}))).error?.message as string | undefined) : undefined;
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
  renderComposer();
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
  paintHarness();
  if ((status.waiting ?? 0) > asked && !connectView.open) toast("A device is asking to join: the OpenCode button has its code");
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
  keepChats();
  renderList();
}

const STEPS_KEPT = 20, STEP_CHARS = 200;
const clipped = (s: string) => (s.length > STEP_CHARS ? `${s.slice(0, STEP_CHARS)}…` : s);
const leanStep = (r: Round): Round => ({ content: clipped(r.content), calls: r.calls.map(({ result, ...c }) => ({ ...c, args: clipped(c.args) })) });
function leanCall({ result, ...c }: Call, behind: boolean): Call {
  return { ...c, ...(result !== undefined && !behind ? { result } : {}) };
}
const leanRound = (r: Round, behind: boolean): Round => ({ content: r.content, ...(r.reasoning && !behind ? { reasoning: r.reasoning } : {}), calls: r.calls.map((c) => leanCall(c, behind)) });
function lean(list: Message[]): Message[] {
  const at = list.findLastIndex((m) => m.checkpoints?.length), from = at < 0 ? 0 : list[at].checkpoints!.at(-1)!.at;
  return list.map((m, i) => ({ ...m, ...(i < at && m.reasoning ? { reasoning: "" } : {}), ...(m.calls ? { calls: m.calls.map((c) => leanCall(c, i < at)) } : {}),
    ...(m.checkpoints ? { checkpoints: m.checkpoints.map((k, j, all) => (i === at && j === all.length - 1 ? k : { ...k, recent: "" })) } : {}),
    ...(m.rounds ? { rounds: m.rounds.map((r, k) => leanRound(r, i < at || (i === at && k < from))) } : {}),
    ...(m.versions ? { versions: { ...m.versions, others: m.versions.others.map(lean) } } : {}) }));
}
function save(c: Conversation) {
  if (!c.messages.length) return;
  chats[c.id] = structuredClone(c);
  keepChats();
  const at = location.hash.slice(1);
  if (c === conv && at !== c.id && !chats[at]) history.replaceState(null, "", `#${c.id}`);
  renderList();
}

const bypassed = new Set<string>();
function request(c: Conversation, upto: Message[], o: { effort?: Effort; sampling?: Mode; final?: object; extra?: Record<string, unknown>; as?: string; again?: boolean } = {}) {
  if (c.thread) {
    const asked = upto.at(-1)?.role === "user" ? upto.at(-1) : undefined;
    const variant = threadVariant(c), files = (asked?.files ?? []).map((a) => {
      const body = bodies.get(a.id) ?? "";
      return { name: a.name, uri: a.kind === "text" ? `data:text/plain;base64,${base64(body)}` : body };
    });
    const next = o.as && people.get(o.as)?.card.data.name.trim(), scene = sceneFor(c), cast = castOf(c), by = next || (cast.group ? undefined : cast.members[0]?.card.data.name.trim());
    const said = asked?.content ?? "", text = [noted([asked?.note, asked?.cut, editsNote(asked ?? { role: "user", content: "" }), asked?.point].filter(Boolean).join("\n\n") || undefined, said), ...(next ? [`[${next} next.]`] : [])].filter(Boolean).join("\n\n");
    const straight = harness.models?.find((m) => m.id === threadModel(c))?.straight, mode: Mode = c.effort === "off" ? "fast" : "think";
    if (straight && (changed(mode) || settings.seed !== undefined) && !bypassed.has(`${c.thread.id}:${straight}`)) {
      bypassed.add(`${c.thread.id}:${straight}`);
      toast(`This OpenCode sends ${threadModel(c)} straight to ${straight}, so your sampler settings, max tokens and seed don't reach it`);
    }
    const tuned = threadModel(c) ? { sampling: { ...samplerFields(mode), ...(settings.max_tokens ? { max_tokens: settings.max_tokens } : {}), ...(settings.seed !== undefined ? { seed: settings.seed } : {}) } } : {};
    return { stream: true, chrysalis_thread: { ...(o.again && c.thread.id ? { again: true } : {}), session: c.thread.id, ...(c.thread.machine ? { machine: c.thread.machine } : {}), ...(c.thread.hidden ? { hidden: true } : {}), dir: c.thread.dir, text, ...tuned, ...(text !== said ? { said } : {}), ...(by ? { next: by } : {}), ...(scene ? { scene } : {}),
      ...(c.thread.id ? {} : { title: c.title }), model: `${threadModel(c)}${variant ? `#${variant}` : ""}`, persona: threadPersona(c), ...(c.thread.agent && harness.agents?.some((a) => a.id === c.thread!.agent) ? { agent: c.thread.agent } : {}), ...(files.length ? { files } : {}) } };
  }
  const effort = o.effort ?? c.effort, mode: Mode = effort === "off" ? "fast" : "think";
  const { members, group } = castOf(c), next = o.as && people.get(o.as)?.card.data.name, system = systemFor(c);
  const messages = cued([...sentHistory(upto, !!group), ...(next ? [cue(next)] : [])]);
  if ((members.length || c.messages[0]?.role === "assistant") && (messages[0] as { role: string } | undefined)?.role !== "user") messages.unshift({ role: "user", content: "[The scene begins.]" });
  if (system) messages.unshift({ role: "system", content: system });
  if (o.final) messages.push(o.final);
  const chrysalis = onChrysalis();
  return {
    model: chrysalis && effort === "off" ? `${current()}-fast` : current(), reasoning_effort: chrysalis ? (effort === "off" ? undefined : effort) : OTHER_EFFORT[effort],
    ...samplerFields(o.sampling ?? mode), max_tokens: settings.max_tokens, seed: settings.seed, ...(chrysalis ? { logprobs: true, top_logprobs: 5 } : {}),
    messages, stream: true, stream_options: { include_usage: true }, ...(c.incognito ? { chrysalis_incognito: true } : {}),
    ...(group ? { stop: [`\n${userName()}:`, `\n**${userName()}`] } : {}), ...o.extra,
  };
}
const OTHER_EFFORT: Record<Effort, string | undefined> = { xhigh: undefined, medium: "medium", low: "low", off: "none" };
const ran = (c: Call) => c.result !== undefined;
const TALKED_OVER = " — [cut off: the user talked over this reply]";
const cue = (name: string) => ({ role: "user", content: `[${name} next.]`, cue: true });
function cued(messages: object[]): object[] {
  const out: { role: string; content: unknown; cue?: boolean }[] = [];
  for (const m of messages as { role: string; content: unknown; cue?: boolean }[]) {
    const last = out.at(-1);
    if (!m.cue) out.push(m);
    else if (last?.role === "user" && typeof last.content === "string") last.content += `\n\n${m.content}`;
    else if (last?.role === "user" && Array.isArray(last.content)) last.content = [...last.content, { type: "text", text: `\n\n${m.content}` }];
    else out.push({ role: "user", content: m.content });
  }
  return out;
}
function sentHistory(upto: Message[], scene = false): object[] {
  const at = upto.findLastIndex((m) => m.checkpoints?.length), by = (m: Message) => (scene && m.role === "assistant" && m.as && people.get(m.as) ? [cue(people.get(m.as)!.card.data.name), ...wire(m)] : wire(m));
  if (at < 0) return upto.flatMap(by);
  const m = upto[at], k = m.checkpoints!.at(-1)!, asked = upto.slice(0, at).findLast((x) => x.role === "user");
  return [withText(asked ? wire(asked)[0] : { role: "user" }, checkpointText(k), k.first), ...wire(m, k.at), ...upto.slice(at + 1).flatMap(by)];
}
function wire(m: Message, from = 0): object[] {
  if (m.role === "user") return [{ role: "user", content: userContent(m) }];
  const ended = !!m.calls?.some(ran);
  const rounds = [...(m.rounds ?? []), ...(ended ? [{ content: m.content, reasoning: m.reasoning, calls: m.calls! }] : [])].slice(from).filter((r) => r.content || r.calls.some(ran));
  const out = rounds.flatMap((r) => {
    const done = r.calls.filter(ran);
    return [assistantTurn(r.content, r.reasoning, done), ...done.map((x) => toolTurn(x.id, x.result!))];
  });
  if (!ended && (m.content || m.reasoning)) out.push(assistantTurn(m.interrupted && m.heardTo === undefined ? m.content + TALKED_OVER : m.content, m.reasoning));
  return out;
}
const globalPrompt = () => fill(settings.global?.trim() ?? "");
const systemFor = (c: Conversation) => (castOf(c).members.length ? [castSystem(c), globalPrompt()]
  : [fill(castPrompt("", [people.get(CHRYSALIS)!.card.data], userCard())), globalPrompt(),
    settings.artifacts === false ? "" : ARTIFACTS_PROMPT]).filter(Boolean).join("\n\n");

const DONE = Symbol("done");
async function* events(body: ReadableStream<Uint8Array>, seen?: (id: number) => void): AsyncGenerator<any> {
  const reader = body.getReader(), decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    for (let end = buf.indexOf("\n\n"); end >= 0; end = buf.indexOf("\n\n")) {
      const block = buf.slice(0, end), data = block.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n"), id = /^id: (\d+)$/m.exec(block)?.[1];
      buf = buf.slice(end + 2);
      if (id) seen?.(Number(id));
      if (data === "[DONE]") return void (yield DONE);
      if (data) yield JSON.parse(data);
    }
  }
}

async function ensureOnline(run: { stopping: boolean }, model = picked) {
  if (!onService()) {
    if (status.state !== "online") await refreshStatus();
    const l = status.backend?.listed;
    if (l && !l.ok) throw new Error(l.error);
    if (!current()) throw new Error(`${status.backend?.name ?? "The backend"} lists no model to chat with`);
    return;
  }
  const none = status.state !== "online" || status.loading || status.model !== model ? await start(false, model) : undefined;
  if (none) throw new Error(none);
  for (const deadline = Date.now() + 180_000; status.state !== "online" || status.loading || status.model !== model; await refreshStatus()) {
    if (run.stopping) throw new DOMException("stopped", "AbortError");
    if (Date.now() > deadline && !status.download) throw new Error("Chrysalis did not start: its terminal says why");
    await sleep(100);
  }
}

const CUT_OFF = "The reply was cut off: the web server restarted while it was being written";

async function generate(c: Conversation = conv, at = c.messages.length, as = c.messages[at]?.as, again = false) {
  const m: Message = { id: randomId(), role: "assistant", content: "", reasoning: "", params: changed(modeOf(c)) || undefined, effort: c.effort, ...(as ? { as } : {}) };
  branch(c, at, m);
  if (as && voice?.c === c) pushVoice();
  const upto = c.messages.slice(0, at);
  await startRun(c, m, () => request(c, upto, { as, again }));
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
addEventListener("pagehide", () => { leaving = true; sheetKeep(); });
async function startRun(c: Conversation, m: Message, body: () => object, pick?: Tok) {
  const phase = m.content ? "writing" : m.reasoning ? "thinking" : "waiting";
  const run: Run = { conv: c, m, live: { phase, since: performance.now(), note: "Reading the prompt" }, at: 0, begun: Date.now(), stopping: false, frame: 0, saved: 0, pick };
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
    if (c.thread) {
      delete c.thread.at;
      await ensureHarness();
      if (!threadModel(c)) throw new Error("Choose a model for this thread: OpenCode lists none of Chrysalis's own");
    }
    const want = c.thread ? threadModel(c)?.slice(10) : current();
    if ((!c.thread || threadModel(c)?.startsWith("chrysalis/")) && (status.state !== "online" || status.loading || (onService() && status.model !== want))) {
      run.live.loading = true;
      await ensureOnline(run, want);
      run.live.note = "Reading the prompt";
      run.live.loading = false;
      run.live.since = performance.now();
    }
    if (run.stopping) throw new DOMException("stopped", "AbortError");
    await filesReady(c.messages);
    await syncing.get(c.id);
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

function begin(m: Message) {
  m.content = m.reasoning = "";
  m.rounds = m.rounds?.slice(0, m.base ?? 0);
  if (!m.rounds?.length) delete m.rounds;
  m.calls = m.notes = undefined;
  m.checkpoints = m.checkpoints?.filter((k) => k.job !== m.job);
  if (m.id) tokens.delete(m.id);
}
async function resume(c: Conversation, m: Message) {
  const run: Run = { conv: c, m, live: { phase: "waiting", since: performance.now(), note: "Reading the prompt" }, at: 0, begun: Date.now(), stopping: false, frame: 0, saved: 0 };
  runs.set(c.id, run);
  try {
    const res = await fetch(`/api/chat/${m.job}?from=0`);
    if (res.status === 404) throw new Error(CUT_OFF);
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    begin(m);
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
        for await (const ev of events(body, (id) => { run.at = id + 1; })) {
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
  if (ev.reset) {
    begin(m);
    run.usage = run.read = run.written = undefined;
    if (v) v.tok = undefined;
    return;
  }
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
  if (ev.thread && run.gone && run.conv.incognito) forgetThread(ev.thread);
  if (ev.thread && run.conv.thread && run.conv.thread.id !== ev.thread) { run.conv.thread.id = ev.thread; save(run.conv); }
  if (ev.idle && run.conv.thread) run.conv.thread.at = ev.idle;
  if (ev.context && run.conv.thread) { run.conv.thread.context = ev.context; if (run.conv === conv) renderMeter(); }
  if (ev.timing) run.timing = ev.timing;
  if (typeof ev.started === "number" && ev.started !== run.begun) { run.begun = ev.started; renderList(); }
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
  const waiting = waitNote(ev, "the context");
  if (waiting) Object.assign(live, { phase: "waiting", since: now, note: waiting });
  if (ev.compacted) m.checkpoints = [...(m.checkpoints ?? []), keptFrom(ev.compacted, m.rounds, m.job)];
  if (ev.round) {
    if (m.content || m.reasoning || m.calls) m.rounds = [...(m.rounds ?? []), { content: m.content, ...(m.reasoning ? { reasoning: m.reasoning } : {}), calls: m.calls ?? [] }];
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
  const asking = needsYou(m);
  if (asking !== !!run.asking) {
    run.asking = asking;
    renderList();
    if (asking && run.conv !== conv) toast(`${run.conv.title} is waiting for you`);
  }
  if (now - run.saved > 2000) {
    run.saved = now;
    save(run.conv);
  }
  if (d.content || d.reasoning_content) {
    const gap = now - (run.heard ?? now);
    if (gap > 20 || run.heard === undefined) {
      if (run.heard !== undefined) run.pace = run.pace ? run.pace * 0.7 + Math.min(gap, 400) * 0.3 : Math.min(gap, 400);
      run.heard = now;
    }
  }
  feedVoice(m);
  schedule(run);
}

const needsYou = (m: Message) => !!m.calls?.some((c) => c.state === "approval" || c.state === "asking");
const waitNote = (ev: any, what: string) => ev.compacting ? `Compacting ${what}` : ev.compacted ? "Reading the prompt" : ev.waiting ?? (ev.retry ? `${ev.retry.message}: trying again` : undefined);
const keptFrom = ({ kept, ...k }: Checkpoint & { kept: number }, rounds: Round[] | undefined, job?: string): Kept => ({ ...k, at: (rounds?.length ?? 0) - kept, ...(job ? { job } : {}) });
function addCall(holder: { calls?: Call[] }, t: any) {
  const calls = (holder.calls ??= []), at = (calls[t.index ?? calls.length] ??= { id: "", name: "", args: "" });
  if (t.id) at.id = t.id;
  if (t.function?.name) at.name += t.function.name;
  if (t.function?.arguments) at.args += t.function.arguments;
}
function toolEvent(calls: Call[] | undefined, tool: any, now: number): Call | undefined {
  const c = calls?.find((x) => x.id === tool.id);
  if (!c) return undefined;
  for (const k of ["title", "source", "readOnly", "state", "result", "ms", "form", "note", "asks"] as const) if (tool[k] !== undefined) (c as any)[k] = tool[k];
  if (tool.state === "running" && !callSince.has(c.id)) callSince.set(c.id, now);
  if (c.state === "approval" || c.state === "asking") openCards.delete(c.id);
  deciding.delete(c.id);
  return c;
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
    const m = run.m, now = performance.now(), whole = { content: m.content, reasoning: m.reasoning ?? "" };
    const size = whole.content.length + whole.reasoning.length, pace = run.pace ?? 0;
    run.shown = Math.min(run.shown ?? 0, size);
    if (pace > 50 && run.heard !== undefined) {
      const left = size - run.shown, dt = now - (run.painted ?? now);
      run.shown += Math.min(left, Math.max(1, Math.ceil(left * dt / Math.max(pace - (now - run.heard), dt, 16))));
    } else run.shown = size;
    run.painted = now;
    const think = Math.min(run.shown, whole.reasoning.length);
    if (m.reasoning !== undefined) m.reasoning = whole.reasoning.slice(0, think);
    m.content = whole.content.slice(0, run.shown - think);
    try { paint(v, m, run.live); } finally {
      m.content = whole.content;
      if (m.reasoning !== undefined) m.reasoning = whole.reasoning;
    }
    if (run.shown < size) schedule(run);
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
      thought: m.reasoning && t.answer && !m.rounds?.length ? (t.answer - t.first) / 1000 : undefined, seed: run.seed, model: c.thread ? threadModel(c) : current(),
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
  if (unrun.length && !m.error && !m.stopped)
    m.error = `The model asked to run ${[...new Set(unrun.map((x) => x.name))].join(", ")}, but a chat has no tools, so the reply ends here.`;
  delete m.job;
  delete m.base;
  delete m.starting;
  cancelAnimationFrame(run.frame);
  if (runs.get(c.id) === run) runs.delete(c.id);
  run.ended?.();
  if (run.gone) return;
  if (c.thread && c.messages.at(-1) !== m && !m.stopped) { c.unsynced = true; settleSoon(c.id); }
  c.updated = Date.now();
  if (c === conv) render();
  else renderList();
  save(c);
  mark(c, m.error ? "failed" : m.stopped ? undefined : "done");
  renderList();
  if (c.thread) void loadThreads().then(() => syncThread(c));
  else if (!c.named && !m.error) void name(c);
  redoEdits(c, m);
  if ((talking || lined.length) && c === conv && !m.error && !m.stopped) setTimeout(talkOn, 400);
  else { talking = 0; lined = []; renderCastNext(); }
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
  if (talking || lined.length) { talking = 0; lined = []; renderCastNext(); }
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

function addressed(c: Conversation, text: string): string[] {
  const { members, group } = castOf(c);
  if (!group) return [];
  const low = text.toLowerCase(), words = (p: Person) => p.card.data.name.toLowerCase().split(/\s+/).filter((w) => w.length > 2);
  const at = (p: Person) => Math.min(...[p.card.data.name.toLowerCase(), ...words(p).filter((w) => !members.some((o) => o !== p && words(o).includes(w)))]
    .map((w) => low.search(new RegExp(`(?<![\\p{L}\\p{N}])${w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?![\\p{L}\\p{N}])`, "u"))).filter((k) => k >= 0));
  return members.map((p) => ({ id: p.id, at: at(p) })).filter((x) => Number.isFinite(x.at)).sort((a, b) => a.at - b.at).map((x) => x.id);
}
function steer(text: string) {
  const sid = conv.thread!.id!;
  input.value = "";
  grow();
  void fetch(`/api/threads/${sid}/say`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: text.trim() }) }).then(async (r) => {
    if (r.ok) return toast("Sent: OpenCode reads it at its next step");
    toast((await r.json().catch(() => null))?.error?.message ?? "OpenCode did not take that message");
    if (!input.value) { input.value = text; grow(); }
  });
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
  const note = noteFor(conv, false), cut = unheardNote(conv);
  if (conv.thread) {
    void syncThreadText(conv.id);
    for (const m of conv.messages) delete m.versions;
  }
  conv.messages.push({ role: "user", content: text, ...(files.length ? { files } : {}), ...(note ? { note } : {}), ...(cut ? { cut } : {}), ...pointFields(false), ...takeEdits() });
  if (!reply) return save(conv);
  const [first, ...rest] = addressed(conv, text);
  talking = 0;
  lined = rest;
  void generate(conv, conv.messages.length, first ?? nextUp(conv));
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
  defaults: { tts?: string; stt?: string; voices: Record<string, string> };
  level: number; frame: number;
  check: { role: string; content: string }[];
  asking: Asking;
  ahead?: Ahead;
}
let voice: VoiceMode | null = null;
let voiceCatalog: VoiceReady["catalog"] = (() => { try { return JSON.parse(pref("voice-catalog", "{}")) ?? {}; } catch { return {}; } })();
const cuts = new Map<Message, number | null>();
const voiceNote = () => fill(VOICE_PROMPT);
function noteFor(c: Conversation, said: boolean): string | undefined {
  const users = c.messages.filter((m) => m.role === "user"), prev = users.at(-1);
  if (!said) return prev?.voice ? VOICE_OFF : undefined;
  const last = users.findLast((m) => m.note && m.voice)?.note, note = voiceNote();
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
      if (first && !asked.stack?.stt) v.defaults.stt = r.stack.stt;
      if ((first || asked.stack?.tts) && !asked.voice && r.stack.tts) v.defaults.voices[r.stack.tts] = r.voice;
      v.ready = r;
      v.now = { voice: r.voice, speed: r.speed };
      voiceCatalog = r.catalog ?? {};
      setPref("voice-catalog", JSON.stringify(voiceCatalog));
      pushVoice();
      renderVoice();
      renderVoiceMenu();
      if (modelsOpen) renderControls();
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
    played: (id) => { if (mine() && voice!.reply?.m.id === id) { voice!.reply = undefined; (globalThis as any).CSS?.highlights?.delete("voice-unspoken"); } },
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
    voice.error = isSecureContext ? "This browser gives no page a microphone" : "Voice needs an encrypted page: Chrysalis is on the network without encryption";
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
  if (menuSpot === composerVoice) voiceMenu.hidePopover();
  paintGhost();
  renderVoice();
  input.focus();
}

interface Heard { text: string; fixes: Fix[]; asked?: { role: string; content: string }[] }
interface Asking { checks: Map<string, Promise<Heard>>; done: Map<string, Heard>; busy?: string; settled?: string }
const asking = (): Asking => ({ checks: new Map(), done: new Map() });
interface Ahead {
  text: string;
  q: Message;
  sent: string;
  said?: Message;
  check?: Heard;
  run: Run; reply: VoiceReply; ended: boolean; live: boolean;
}
const unsettled = new WeakSet<Message>();
const heardAs = new WeakMap<Message, string>();

function checkOf(v: VoiceMode, text: string): Promise<Heard> {
  const mine = v.asking, asked = mine.checks.get(text);
  if (asked) return asked;
  const c = v.c, system = fill(CHECK);
  if (v.check[0]?.content !== system || v.check.length > 121) v.check = [{ role: "system", content: system }];
  const at = c.messages.findLastIndex((m) => m.role === "assistant"), last = c.messages[at]?.content.slice(0, 300) ?? "";
  const before = at > 0 ? c.messages.slice(0, at).findLast((m) => m.role === "user")?.content.slice(0, 200) ?? "" : "";
  const turn = { role: "user", content: (before ? `The user said before: ${before}\n` : "") + (last ? `The assistant said: ${last}\n` : "") + `Transcript: ${text}` };
  const p = (async (): Promise<Heard> => {
    try {
      const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        ...quick(), messages: [...v.check, turn], temperature: 0, presence_penalty: 0, max_tokens: 48 }) });
      const answer: string = (await res.json())?.choices?.[0]?.message?.content ?? "";
      if (!res.ok || !answer) return { text, fixes: [] };
      return { text, ...readHeard(answer), asked: [turn, { role: "assistant", content: answer }] };
    } catch {
      return { text, fixes: [] };
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
  const c = v.c, known = knownFor(v, text), fixed = applyFixes(text, known?.fixes ?? []);
  const effort = v.c.effort, mode: Mode = effort === "off" ? "fast" : "think";
  const note = on ? undefined : noteFor(c, true), cut = on ? undefined : unheardNote(c);
  const q: Message = on ?? { role: "user", content: fixed.text, voice: true, ...(note ? { note } : {}), ...(cut ? { cut } : {}), ...pointFields(true), ...takeEdits() };
  const sent = { ...q, content: fixed.text }, upto = on ? c.messages.map((x) => (x === on ? sent : x)) : [...c.messages, sent];
  const m: Message = { id: randomId(), role: "assistant", content: "", reasoning: "", effort, params: changed(mode) || undefined };
  const run: Run = { conv: c, m, live: { phase: "waiting", since: performance.now(), note: "Reading the prompt" }, at: 0, begun: Date.now(), stopping: false, frame: 0, saved: 0 };
  const a: Ahead = { text, q, sent: fixed.text, run, reply: { m, sent: 0, spoken: 0, audible: false }, ended: false, live: false };
  if (!on) heardAs.set(q, text);
  v.client.hold(m.id!);
  void (async () => {
    try {
      await filesReady(upto);
      const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request(c, upto, { effort })) });
      if (!res.ok || !res.body) throw new Error((await res.json().catch(() => null))?.error?.message ?? `HTTP ${res.status}`);
      m.job = res.headers.get("X-Chrysalis-Job") ?? undefined;
      if (run.stopping) stopRun(run, true);
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
  if (v.ahead?.said || running(v.c) || status.state !== "online" || status.loading || v.c.thread) return;
  const q = questionFor(v, text.trim());
  if (v.ahead?.text === q.text) return;
  dropAhead(v);
  v.ahead = startAhead(v, q.text, q.on);
}

function standAhead(v: VoiceMode, a: Ahead) {
  if (voice !== v || v.ahead !== a || !a.said || !a.check) return;
  const c = v.c, said = a.said, fixed = applyFixes(a.text, a.check.fixes), m = a.run.m;
  if (fixed.text !== a.sent) {
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
  if (c.thread || status.state !== "online" || status.loading) return answerLater(v, q.text, q.on);
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
  const c = v.c, note = on ? undefined : noteFor(c, true), cut = on ? undefined : unheardNote(c);
  const said: Message = on ?? { role: "user", content: text, voice: true, ...(note ? { note } : {}), ...(cut ? { cut } : {}), ...pointFields(true), ...takeEdits() };
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
  await startRun(c, m, () => request(c, upto, { effort }));
}

async function warmVoice(v: VoiceMode) {
  const effort = v.c.effort;
  await ensureOnline({ stopping: false }).catch(() => {});
  if (voice !== v || running(v.c) || v.c.thread) return;
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
  if (to === null && !c.thread) {
    const at = c.messages.indexOf(m);
    if (at >= 0 && at === c.messages.length - 1) c.messages.splice(at, 1);
    return;
  }
  m.interrupted = true;
  m.heardTo = to ?? 0;
}
function unheardNote(c: Conversation): string | undefined {
  const m = c.messages.at(-1);
  if (m?.role !== "assistant" || m.heardTo === undefined) return undefined;
  const heard = [...sayable(m)].slice(0, m.heardTo).join("").trim().split(/\s+/).filter(Boolean);
  if ([...sayable(m)].length <= m.heardTo) return undefined;
  return heard.length ? `The user talked over your last reply, which is read aloud. They heard it up to "${heard.slice(-12).join(" ")}" and nothing after that.`
    : "The user talked over your last reply, which is read aloud, before hearing any of it.";
}
function paintUnheard() {
  const reg = (globalThis as any).CSS?.highlights as Map<string, unknown> | undefined;
  if (!reg) return;
  const ranges = conv.messages.flatMap((m) => {
    const view = m.heardTo === undefined ? undefined : views.get(m);
    if (!view) return [];
    const text = sayable(m), at = unitOf(text, m.heardTo!), n = shownWords(text).filter((w) => w.start < at).length;
    const from = afterWords([...view.rounds.querySelectorAll<HTMLElement>(":scope > .round > .markdown"), view.body], n);
    if (!from) return [];
    const range = new Range();
    range.setStart(from.node, from.offset);
    range.setEnd(view.body, view.body.childNodes.length);
    return [range];
  });
  if (ranges.length) reg.set("voice-unheard", new (globalThis as any).Highlight(...ranges));
  else reg.delete("voice-unheard");
}

function paintGhost() {
  let g = thread.querySelector<HTMLElement>(".msg.ghost");
  const v = voice, text = v && v.c === conv ? v.partial : "";
  if (!v || !text) return void g?.remove();
  if (!g) {
    g = h("article", "msg user ghost");
    g.append(h("div", "bubble"));
    thread.querySelector(".empty")?.remove();
    ocean.butterfly(null);
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
  const v = voice, pill = $("voice-pill");
  $("composer").classList.toggle("voicing", !!v);
  pill.hidden = !v;
  composerPick ??= voiceTrigger(composerVoice, $<HTMLButtonElement>("voice-pick"));
  composerPick.el.hidden = !v;
  $("voice").hidden = !!v;
  renderComposer();
  if (!v) return void delete $("composer").dataset.phase;
  const mute = $("voice-mute");
  mute.innerHTML = icon(v.muted ? "micOff" : "mic");
  mute.title = v.muted ? "Unmute" : "Mute";
  mute.setAttribute("aria-label", mute.title);
  mute.setAttribute("aria-pressed", String(v.muted));
  mute.classList.toggle("on", v.muted);
  composerPick.paint();
  if (menuSpot === composerVoice) renderVoiceMenu();
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
  const speaks = group ? members.find((p) => p.id === v!.c.messages.findLast((m) => m.role === "assistant")?.as) : members[0];
  const own = speaks ? voiceOf(speaks.card.data) : undefined;
  const s = { ...(settings.voice ?? {}), ...(own ?? {}) }, d = v?.defaults, tts = s.tts ?? d?.tts, name = s.name ?? (tts ? d?.voices[tts] : undefined);
  const stt = settings.voice?.stt ?? d?.stt, stack: VoiceAsk["stack"] = { ...(tts ? { tts } : {}), ...(stt ? { stt } : {}) };
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

const composerVoice: VoiceSpot = {
  now: () => { const c = currentVoice(); return voiceIdOf(c.tts, c.id); },
  none: "Default",
  pick: (id) => {
    const before = { tts: settings.voice?.tts, name: settings.voice?.name }, [tts, name] = voiceParts(id);
    setVoice({ tts, name });
    saveSettings();
    toast(`Voice: ${voiceShown(id, "Default").name}`, () => { setVoice(before); saveSettings(); });
  },
  hear: (id) => { const [tts, name] = voiceParts(id); return voice ? voice.client.preview(name, tts) : Promise.resolve(); },
  meta: () => voice?.ready ? (["stt", "turn", "tts"] as Part[]).map((k) => modelLabel(k, voice!.ready!.stack[k])).filter(Boolean).join(" · ") : "",
};
let composerPick: ReturnType<typeof voiceTrigger> | undefined;

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
    const r = await attach(f, conv.thread ? "OpenCode" : onChrysalis() ? undefined : status.backend?.name, !!conv.thread);
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
  const note = [m.note, m.cut, editsNote(m), m.point].filter(Boolean).join("\n\n") || undefined, said = noted(note, m.content);
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


function download(name: string, data: string, type: string) {
  const a = h("a");
  a.href = URL.createObjectURL(new Blob([data], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

let drafting: { job?: string; stop: boolean } | null = null;

async function draft() {
  if (drafting) {
    drafting.stop = true;
    if (drafting.job) void fetch(`/api/chat/${drafting.job}/stop`, { method: "POST" }).catch(() => {});
    return;
  }
  const begin = input.value, c: Conversation = conv.thread ? { ...conv, thread: undefined, messages: conv.messages.filter((m) => m.content.trim()).map(({ rounds, calls, reasoning, checkpoints, ...m }) => m) } : conv;
  const d: { job?: string; stop: boolean } = (drafting = { stop: false });
  try { await filesReady(c.messages); } catch (e) { drafting = null; return toast(`The draft failed: ${e instanceof Error ? e.message : e}`); }
  const body = request(c, c.messages, {
    sampling: "fast", final: begin.trim() ? { role: "assistant", content: begin } : undefined,
    extra: { chrysalis_aside: true, max_tokens: 400, logprobs: false, seed: undefined, ...(begin.trim() ? { continue_final_message: true } : {}) },
  }) as { messages: object[] };
  (body.messages as object[]).splice(begin.trim() ? -1 : body.messages.length, 0, { role: "user", content: c.messages.length ? DRAFT_PROMPT : DRAFT_FIRST });
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
  if (id !== conv.id) leave();
  conv = runs.get(id)?.conv ?? (chats[id] ? structuredClone(chats[id]) : blank());
  const sid = conv.thread?.id ?? (id.startsWith(STUB) ? id.slice(STUB.length) : undefined);
  if (sid) showThread(sid);
  if (reveal) unfold(whoOf(conv));
  closeOver(panes.conversations);
  render();
  scroller.scrollTop = scroller.scrollHeight;
  follow = true;
  keepUp();
}

const hiddenThread = () => ({ dir: harness.home!, hidden: true as const });
const forgetThread = (id: string) => void fetch(`/api/threads/${id}`, { method: "DELETE", keepalive: true }).catch(() => {});
function leave() {
  const c = conv, run = runs.get(c.id);
  if (!c.incognito) return;
  if (run) { run.gone = true; stopRun(run); }
  delete chats[c.id];
  if (c.thread?.id) forgetThread(c.thread.id);
}
addEventListener("pagehide", leave);
function incognito() {
  if (conv.incognito) return newChat();
  closeVoice();
  stopReading();
  conv = { ...blank(), named: true, incognito: true };
  if (harness.state === "on" && harness.home && pref("tools", "on") === "on") conv.thread = hiddenThread();
  history.replaceState(null, "", location.pathname);
  closeOver(panes.conversations);
  render();
  input.focus();
}
function newChat(who = "") {
  closeVoice();
  stopReading();
  leave();
  conv = blank();
  unfold(who || CHRYSALIS);
  if (who.startsWith(DIR)) conv.thread = inDir(who.slice(DIR.length));
  else if (who && who !== CHRYSALIS) conv.who = who;
  if (!conv.thread && harness.state === "on" && harness.home && pref("tools", "on") === "on") conv.thread = freshThread();
  if (!who.startsWith(DIR)) greet(conv);
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
  keepChats();
  if (id === conv.id) newChat(whoOf(conv));
  else renderList();
}

function copy(text: string, button: HTMLElement) {
  const area = h("textarea");
  area.value = text;
  area.readOnly = true;
  area.style.cssText = "position: fixed; top: 0; opacity: 0; font-size: 16px";
  (button.closest("dialog") ?? document.body).append(area);
  area.select();
  area.setSelectionRange(0, text.length);
  document.execCommand("copy");
  area.remove();
  const label = button.textContent;
  button.classList.add("done");
  const named = button.matches(".copy, .dev-copy");
  if (named) button.textContent = "Copied";
  setTimeout(() => { button.classList.remove("done"); if (named) button.textContent = label; }, 1200);
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
  const turn = act.dataset.act === "prev" ? -1 : act.dataset.act === "next" ? 1 : 0;
  if (turn > 0 && (m.versions?.shown ?? 0) === (m.versions?.others.length ?? 0)) return void (m.role === "assistant" && conv.messages.at(-1) === m && renew(i));
  if (turn && m.versions) {
    const k = m.versions.shown + turn;
    editing = null;
    showVersion(conv, i, versionsAt(conv, i), k);
    if (conv.thread) { conv.unsynced = true; settleSoon(conv.id); }
    save(conv);
    render();
    const again = thread.querySelector<HTMLButtonElement>(`article[data-i="${i}"] [data-act=${act.dataset.act}]`);
    return void (again && !again.disabled ? again : thread.querySelector<HTMLElement>(`article[data-i="${i}"] .msg-foot :is([data-act="prev"], [data-act="next"]):not(:disabled)`))?.focus();
  }
  if (act.dataset.act === "card" || act.dataset.act === "more") return toggleCard(act);
  if (act.dataset.act === "decide") return void decide(act);
  if (act.dataset.act === "speak") return readAloud(m);
  if (running()) return;
  if (act.dataset.act === "fork") return void threadAct("fork", m);
  if (conv.thread?.id && act.dataset.act === "retry") return void (conv.messages.at(-1) === m ? renew(i) : threadAct("resend", m));
  if (conv.thread?.id && act.dataset.act === "trash") return void threadAct("drop", m);
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
async function decide(btn: HTMLElement, answer?: object) {
  const card = btn.closest<HTMLElement>(".tool-card")!, id = card.dataset.call!, decision = btn.dataset.decision;
  const m = conv.messages[Number(card.closest<HTMLElement>("article")?.dataset.i)], c = m?.calls?.find((x) => x.id === id);
  if (!m?.job || !c) return;
  deciding.add(id);
  const r = running();
  if (r) schedule(r);
  const res = await fetch(`/api/chat/${m.job}/tool/${encodeURIComponent(id)}`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(answer ? { answer } : { decision }) }).catch(() => null);
  if (!res?.ok && res?.status !== 409) { deciding.delete(id); toast("The answer did not reach the web server: try again"); }
}
scroller.addEventListener("scroll", () => { if (!$("tokpop").hidden) closeTokPop(); }, { passive: true });
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (voice) return closeVoice();
  const r = running();
  if (r && conv.thread?.id && input.value.trim()) steer(input.value);
  else if (r) stopRun(r);
  else submit(input.value);
});
input.addEventListener("input", () => { grow(); if (!drafting) $("draft").title = $("draft").ariaLabel = draftTitle(); });
input.addEventListener("input", () => { if (talking || lined.length) { talking = 0; lined = []; renderCastNext(); } });
interface Slash { name: string; aliases?: string[]; hint?: string; about: string; run?: (arg: string) => void | Promise<void> }
let slashFrom: { key: string; commands: { name: string; description: string }[]; skills: { id: string; name: string; description: string }[] } | undefined, slashAt = 0;
async function slashSource() {
  const t = conv.thread, key = t ? `${t.machine ?? ""}:${t.dir}` : "";
  if (!t || harness.state !== "on" || slashFrom?.key === key) return;
  slashFrom = { key, commands: [], skills: [] };
  const r = await fetch(`/api/threads/commands?dir=${encodeURIComponent(t.dir)}${t.machine ? `&machine=${t.machine}` : ""}`).catch(() => null), got = r?.ok ? await r.json() : null;
  if (got && slashFrom?.key === key) { slashFrom = { key, commands: got.commands ?? [], skills: got.skills ?? [] }; paintSlash(); }
}
const sidOf = () => conv.thread?.id;
const needThread = () => (sidOf() ? true : (toast("This works once the thread has a message"), false));
async function undoLast() {
  const sid = sidOf(), asked = conv.messages.findLast((m) => m.role === "user" && m.oc);
  if (!sid || !asked || running()) return;
  const r = await fetch(`/api/threads/${sid}/undo`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: asked.oc }) }).catch(() => null);
  if (!r?.ok) return toast((await r?.json().catch(() => null))?.error?.message ?? "OpenCode did not undo that");
  await showThread(sid);
  input.value = asked.content;
  input.dispatchEvent(new Event("input"));
}
function exportThread() {
  const text = [`# ${conv.title}`, ...conv.messages.map((m) => `## ${m.role === "user" ? userName() : nameOf(m.as ?? whoOf(conv))}\n\n${m.content}`)].join("\n\n"), a = h("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
  a.download = `${conv.title.replace(/[^\w.-]+/g, "-").slice(0, 60) || "thread"}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
}
const PAGE_SLASH: Slash[] = [
  { name: "new", aliases: ["clear"], about: "a new thread here", run: () => newChat(whoOf(conv)) },
  { name: "compact", about: "summarize the context now", run: async () => {
    const sid = sidOf();
    if (!sid || !needThread()) return;
    toast("Compacting the context…");
    const r = await fetch(`/api/threads/${sid}/compact`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }).catch(() => null);
    if (!r?.ok) return toast((await r?.json().catch(() => null))?.error?.message ?? "OpenCode did not compact it");
    if (conv.thread?.id === sid) { await showThread(sid); renderControls(); }
    toast("Context compacted");
  } },
  { name: "models", about: "choose the model", run: () => setModelsOpen(true) },
  { name: "agents", about: "choose the agent", run: () => setModelsOpen(true) },
  { name: "variants", aliases: ["thinking", "effort"], hint: "off | low | medium | max", about: "set the thinking level", run: (arg) => {
    const at = ["off", "low", "medium", "max"].indexOf(arg.trim().toLowerCase());
    if (at < 0) return toast("Say which: off, low, medium or max");
    setEffort(at);
  } },
  { name: "undo", about: "take back your last message and what it changed", run: () => { if (needThread()) void undoLast(); } },
  { name: "fork", about: "a new thread from this one", run: () => {
    const sid = sidOf();
    if (!sid || !needThread()) return;
    void fetch(`/api/threads/${sid}/fork`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })
      .then(async (r) => { if (!r.ok) return toast("OpenCode did not fork that chat"); const { id } = await r.json(); await loadThreads(); location.hash = `#${STUB}${id}`; });
  } },
  { name: "rename", hint: "title", about: "name this thread", run: (arg) => { if (arg.trim()) rename(conv, arg); else toast("Say the new title after /rename"); } },
  { name: "sessions", aliases: ["resume", "continue"], about: "find a thread", run: () => { setPane(panes.conversations, true); $("side-search").focus(); } },
  { name: "open", aliases: ["projects", "project"], about: "start a thread in a project", run: () => newThreadIn() },
  { name: "cd", hint: "folder", about: "work in another folder", run: (arg) => (arg.trim() ? void moveThread(conv, arg.trim().replace(/^~(?=\/|$)/, homeOf(conv.thread?.machine) ?? "~")) : chooseFolder($("folder-btn"))) },
  { name: "copy", about: "copy the last reply", run: () => {
    const last = conv.messages.findLast((m) => m.role === "assistant" && m.content);
    if (last) void navigator.clipboard.writeText(last.content).then(() => toast("Copied"));
  } },
  { name: "export", about: "save this thread as Markdown", run: exportThread },
  { name: "skills", about: "use a skill", run: () => {
    const all = slashFrom?.skills ?? [], sid = sidOf();
    if (!all.length) return toast("OpenCode lists no skill here");
    if (!sid || !needThread()) return;
    openPalette([{ label: "Skills", items: all.map((k): PaletteItem => ({ icon: icon("agent"), title: k.name, note: k.description ? [k.description] : [], run: () => void fetch(`/api/threads/${sid}/skill`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: k.id }) }).then(async (r) => { if (!r.ok) toast("OpenCode did not start that skill"); await loadThreads(); }) })) }]);
  } },
  { name: "status", aliases: ["usage"], about: "context used, and when it compacts", run: () => { const b = $("meter").querySelector<HTMLElement>(".ctx"); if (b) b.click(); else toast("Nothing measured yet"); } },
  { name: "settings", about: "open Settings", run: () => openSettings() },
  { name: "help", about: "list the commands", run: () => { input.value = "/"; paintSlash(); input.focus(); } },
];
const slashAll = (): Slash[] => [...(slashFrom?.commands ?? []).filter((c) => !PAGE_SLASH.some((p) => p.name === c.name)).map((c): Slash => ({ name: c.name, about: c.description, hint: "…" })), ...PAGE_SLASH];
const slashTyped = () => (conv.thread && harness.state === "on" ? /^\/(\S*)$/.exec(input.value)?.[1] : undefined);
const slashShown = () => {
  const q = slashTyped();
  return q === undefined ? [] : slashAll().filter((c) => [c.name, ...(c.aliases ?? [])].some((n) => n.startsWith(q.toLowerCase())));
};
function paintSlash() {
  const pop = $("slash-pop"), list = slashShown();
  pop.hidden = !list.length;
  if (pop.hidden) return;
  void slashSource();
  slashAt = Math.max(0, Math.min(slashAt, list.length - 1));
  pop.replaceChildren(...list.map((c, k) => {
    const b = h("button", k === slashAt ? "slash-item on" : "slash-item");
    b.type = "button";
    b.setAttribute("role", "option");
    b.setAttribute("aria-selected", String(k === slashAt));
    b.append(h("span", "slash-name", `/${c.name}`), ...(c.hint && c.hint !== "…" ? [h("span", "slash-hint", c.hint)] : []), h("span", "slash-about", c.about));
    b.onmousedown = (e) => e.preventDefault();
    b.onclick = () => pickSlash(c);
    return b;
  }));
  pop.querySelector(".on")?.scrollIntoView({ block: "nearest" });
}
function pickSlash(c: Slash) {
  if (c.hint || !c.run) {
    input.value = `/${c.name} `;
    input.dispatchEvent(new Event("input"));
    return input.focus();
  }
  input.value = "";
  input.dispatchEvent(new Event("input"));
  void c.run("");
}
function runSlash(text: string): boolean {
  const [, name, arg = ""] = /^\/(\S+)\s*([^]*)$/.exec(text.trim()) ?? [];
  if (!name || !conv.thread || harness.state !== "on" || (slashFrom?.commands ?? []).some((c) => c.name === name)) return false;
  const c = PAGE_SLASH.find((p) => p.name === name || p.aliases?.includes(name));
  if (!c?.run) return false;
  input.value = "";
  input.dispatchEvent(new Event("input"));
  void c.run(arg);
  return true;
}
input.addEventListener("input", () => { slashAt = 0; paintSlash(); });
input.addEventListener("blur", () => { $("slash-pop").hidden = true; });
input.addEventListener("keydown", (e) => {
  const list = $("slash-pop").hidden ? [] : slashShown();
  if (list.length && (e.key === "ArrowDown" || e.key === "ArrowUp")) { e.preventDefault(); slashAt = (slashAt + (e.key === "ArrowDown" ? 1 : list.length - 1)) % list.length; return paintSlash(); }
  if (list.length && e.key === "Escape") { e.preventDefault(); e.stopPropagation(); return void ($("slash-pop").hidden = true); }
  if (list.length && (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey && !e.isComposing))) { e.preventDefault(); return pickSlash(list[slashAt]); }
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (runSlash(input.value)) return;
    if (running() && conv.thread?.id && input.value.trim()) steer(input.value);
    else submit(input.value);
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
  if (modelsOpen) return setModelsOpen(false, true);
  if (voiceMenu.matches(":popover-open")) return;
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
function swipe(act: "prev" | "next") {
  const last = [...thread.querySelectorAll<HTMLElement>(":scope > article")].at(-1), was = document.activeElement;
  const to = last?.classList.contains("assistant") ? last.querySelector<HTMLButtonElement>(`.msg-foot [data-act="${act}"]`) : null;
  if (!to || to.disabled) return false;
  to.click();
  if (was === $("input")) $("input").focus();
  return true;
}
document.addEventListener("keydown", (e) => {
  if ((e.key !== "ArrowLeft" && e.key !== "ArrowRight") || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey || editing || document.querySelector("dialog[open]")) return;
  const t = e.target as HTMLElement, box = $<HTMLTextAreaElement>("input");
  if (t !== document.body && !(t === box && !box.value) && !t.closest?.('[data-act="prev"], [data-act="next"]')) return;
  if (swipe(e.key === "ArrowLeft" ? "prev" : "next")) e.preventDefault();
});
let swipeFrom: { x: number; y: number } | null = null;
thread.addEventListener("touchstart", (e) => {
  const t = e.target as Element, last = [...thread.querySelectorAll(":scope > article")].at(-1);
  swipeFrom = e.touches.length === 1 && last?.contains(t) && !t.closest("pre, table, .katex-display, iframe, input, textarea, button")
    ? { x: e.touches[0].clientX, y: e.touches[0].clientY } : null;
}, { passive: true });
thread.addEventListener("touchend", (e) => {
  const from = swipeFrom, at = e.changedTouches[0];
  swipeFrom = null;
  if (!from || !at || getSelection()?.toString()) return;
  const dx = at.clientX - from.x, dy = at.clientY - from.y;
  if (Math.abs(dx) > 70 && Math.abs(dx) > 2 * Math.abs(dy)) swipe(dx < 0 ? "next" : "prev");
}, { passive: true });
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
$("new-chat").addEventListener("click", (e) => {
  const projects = dirs().length + [...people.values()].filter((p) => p.kind === "character").length + groups.size + Math.max(0, (harness.machines?.length ?? 1) - 1);
  if (e.shiftKey) newChat(whoOf(conv));
  else if (scopeOf()) newChat(scopeOf());
  else if (!projects) newChat();
  else newThreadIn();
});
const side = $<HTMLInputElement>("side-search");
function clearSearch() {
  sideQuery = side.value = "";
  sideAt = 0;
  renderList();
}
side.oninput = () => { sideQuery = side.value; sideAt = 0; renderList(); };
side.onkeydown = (e) => {
  const n = document.querySelectorAll("#side-results > a").length;
  if (e.key === "Escape" && sideQuery) { e.preventDefault(); clearSearch(); }
  else if ((e.key === "ArrowDown" || e.key === "ArrowUp") && n) { e.preventDefault(); sideAt = (sideAt + (e.key === "ArrowDown" ? 1 : n - 1)) % n; renderList(); document.getElementById(`side-result-${sideAt}`)?.scrollIntoView({ block: "nearest" }); }
  else if (e.key === "Enter" && n) { e.preventDefault(); document.getElementById(`side-result-${sideAt}`)?.click(); }
};
$("side-clear").onclick = () => { clearSearch(); side.focus(); };
function paintScope() {
  const pop = $("scope-pop"), q = fold($<HTMLInputElement>("scope-input").value), now = scopeOf();
  const ids = [...[...people.values()].filter((p) => p.kind !== "persona").map((p) => p.id), ...groups.keys(), ...dirs().map((d) => DIR + d)].filter((id) => !q || fold(nameOf(id)).includes(q));
  const row = (id: string, label: string, pic: HTMLElement | string) => {
    const b = h("button", id === now ? "scope-item on" : "scope-item"), mark = h("span", "palette-icon");
    b.type = "button";
    b.setAttribute("role", "option");
    b.setAttribute("aria-selected", String(id === now));
    if (typeof pic === "string") mark.innerHTML = pic; else mark.append(pic);
    b.append(mark, h("span", "scope-name", label));
    const m = id.startsWith(DIR) && (harness.machines?.length ?? 0) > 1 ? machineOf(dirMachine(id.slice(DIR.length))) : undefined;
    if (m) b.insertAdjacentHTML("beforeend", `<span class="scope-on" title="${escapeHtml(m.name)}">${icon(machineIcon(m.kind))}</span>`);
    if (id === now) b.insertAdjacentHTML("beforeend", `<span class="scope-check">${icon("check")}</span>`);
    b.onclick = () => { setPref("scope", id); pop.hidden = true; renderList(); };
    return b;
  };
  $("scope-list").replaceChildren(...(!q ? [row("", "All projects", icon("folder"))] : []), ...ids.map((id) => row(id, nameOf(id), face(id))), ...(q && !ids.length ? [h("div", "palette-none", "No projects match")] : []));
}
$("side-scope").onclick = () => {
  const pop = $("scope-pop"), r = $("side-scope").getBoundingClientRect();
  if (!pop.hidden) return void (pop.hidden = true);
  pop.hidden = false;
  pop.style.left = `${Math.max(8, Math.min(r.left, innerWidth - 296))}px`;
  pop.style.top = `${r.bottom + 6}px`;
  $<HTMLInputElement>("scope-input").value = "";
  paintScope();
  $("scope-input").focus();
};
$<HTMLInputElement>("scope-input").oninput = paintScope;
$("scope-input").onkeydown = (e) => {
  if (e.key === "Escape") { e.preventDefault(); $("scope-pop").hidden = true; }
  else if (e.key === "Enter") { e.preventDefault(); $("scope-list").querySelector<HTMLElement>(".scope-item")?.click(); }
};
document.addEventListener("pointerdown", (e) => { if (!$("ctx-pop").hidden && !(e.target as Element).closest("#ctx-pop, #meter")) $("ctx-pop").hidden = true; });
document.addEventListener("pointerdown", (e) => { if (!$("scope-pop").hidden && !(e.target as Element).closest("#scope-pop, #side-scope")) $("scope-pop").hidden = true; });
$("add-project").onclick = addProject;
$("add-cast").addEventListener("click", (e) => { e.stopPropagation(); newCharacter(); });
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
const dropsHere = (e: DragEvent) => !!e.dataTransfer?.types.includes("Files");
document.addEventListener("dragenter", (e) => { if (dropsHere(e) && dragDepth++ === 0) document.querySelector(".composer-box")!.classList.add("dropping"); });
document.addEventListener("dragleave", (e) => { if (dropsHere(e) && --dragDepth === 0) document.querySelector(".composer-box")!.classList.remove("dropping"); });
document.addEventListener("dragover", (e) => { if (dropsHere(e)) e.preventDefault(); });
document.addEventListener("drop", (e) => {
  if (!dropsHere(e)) return;
  e.preventDefault();
  dragDepth = 0;
  document.querySelector(".composer-box")!.classList.remove("dropping");
  void addFiles(e.dataTransfer!.files);
});
document.addEventListener("dragend", () => { dragDepth = 0; document.querySelector(".composer-box")!.classList.remove("dropping"); });

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
  $("incognito").innerHTML = icon("incognito");
  $("incognito").title = $("incognito").ariaLabel = "Incognito chat: kept in memory only, in OpenCode too, and gone when you leave it";
  $("incognito").addEventListener("click", incognito);
  $("model-btn").addEventListener("click", () => setModelsOpen(!modelsOpen));
  $("machine-btn").addEventListener("click", () => chooseMachine($("machine-btn")));
  $("folder-btn").addEventListener("click", () => ($("oc-pop").hidden ? chooseFolder($("folder-btn")) : void ($("oc-pop").hidden = true)));
  document.addEventListener("pointerdown", (e) => {
    if (modelsOpen && !(e.target as Element).closest("#model-pop, #model-btn")) setModelsOpen(false);
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
  note.hidden = !parts.length;
}
function setModelsOpen(open: boolean, refocus = false) {
  modelsOpen = open;
  $("model-pop").hidden = !open;
  delete $("model-pop").dataset.key;
  renderControls();
  if (open) {
    const pop = $("model-pop"), room = pop.offsetParent?.clientWidth ?? innerWidth;
    pop.style.left = `${Math.max(8, Math.min($("model-btn").offsetLeft, room - pop.offsetWidth - 8))}px`;
    pop.querySelector<HTMLElement>(".current")?.focus();
    if (!voiceCatalog.stt?.length) void loadVoices();
  } else if (refocus) $("model-btn").focus();
}

const JSON_POST = { method: "POST", headers: { "Content-Type": "application/json" } };

const shortName = (id: string) => modelName(id).replace(/^Qwen3\.8-/, "");
interface Gauge { used?: number; top?: number; window?: number; mark?: number; pct: string; left: string; title: string[]; rows: [string, string][]; breakdown?: SessionContextBreakdownSegment[]; thread: boolean }
function gauge(): Gauge | undefined {
  const n = (x: number) => x.toLocaleString("en-US");
  if (conv.thread) {
    const at = conv.thread.context;
    if (harness.state !== "on") return undefined;
    if (!at) return { pct: "Context", left: "", title: ["The next reply measures this thread's context"], rows: [["Tokens", "not measured yet"]], thread: true };
    const used = at.used ?? 0, top = at.auto ? at.ceiling : at.window, left = Math.max(0, top - used);
    return { ...(at.used === undefined ? {} : { used }), top, window: at.window, ...(at.auto ? { mark: at.ceiling } : {}), breakdown: at.used === undefined ? undefined : at.breakdown, thread: true,
      pct: at.used === undefined ? (at.compacted ? "Compacted" : "Context") : `${Math.round(Math.min(1, used / top) * 100)}%`, left: at.used === undefined ? `${fmtK(top)} free` : `${fmtK(left)} left`,
      title: [at.used === undefined ? (at.compacted ? "The context was just compacted: the next reply measures it" : "The next reply measures this thread's context") : `Context: ${n(used)} tokens used`,
        at.auto ? `OpenCode compacts it at ${n(at.ceiling)}${at.used === undefined ? "" : ` (${n(left)} to go)`}` : "OpenCode does not compact it by itself", `The model's window is ${n(at.window)}`],
      rows: [["Tokens", at.used === undefined ? (at.compacted ? "just compacted" : "not measured yet") : n(used)], ["Context", `${Math.round((used / at.window) * 100)}% of ${n(at.window)}`],
        [at.auto ? "Compacts at" : "Compacts", at.auto ? n(at.ceiling) : "only when asked"]] };
  }
  const online = status.state === "online" && !status.loading, ctx = status.context ?? 262144;
  const r = running(), s = conv.messages.findLast((m) => m.stats)?.stats;
  const used = r?.read !== undefined ? r.read + (r.written ?? 0) : s ? s.ctx ?? s.prompt + s.tokens : undefined;
  if (!onService()) return { ...(used === undefined ? {} : { used }), thread: false, pct: used === undefined ? "Context" : `${fmtK(used)} tokens`, left: "",
    title: [used === undefined ? "The next reply measures this chat's context" : `This chat: ${n(used)} tokens`], rows: [["Tokens", used === undefined ? "not measured yet" : n(used)]] };
  const free = status.cache?.free_kv_tokens ?? 0, kv = status.kv_tokens ?? 0;
  return { ...(used === undefined ? {} : { used }), top: ctx, window: ctx, thread: false, pct: `${Math.round(Math.min(1, (used ?? 0) / ctx) * 100)}%`, left: `${fmtK(Math.max(0, ctx - (used ?? 0)))} left`,
    title: [used === undefined ? `This chat: nothing of ${n(ctx)} tokens of context yet` : `This chat: ${n(used)} of ${n(ctx)} tokens of context`,
      ...(online ? [`Cache: ${fmtK(free)} of ${fmtK(kv)} KV tokens free (cached conversations are dropped as needed)`] : [])],
    rows: [["Tokens", n(used ?? 0)], ["Context", `${Math.round(((used ?? 0) / ctx) * 100)}% of ${n(ctx)}`], ...(online && kv ? [["Cache free", `${fmtK(free)} of ${fmtK(kv)}`] as [string, string]] : [])] };
}
function renderContext() {
  const pop = $("ctx-pop"), g = gauge();
  if (!g) return void (pop.hidden = true);
  const key = JSON.stringify([g, !!running()]);
  if (pop.dataset.key === key) return;
  pop.dataset.key = key;
  const share = g.top ? Math.min(1, (g.used ?? 0) / g.top) : 0, head = h("div", "ctx-head"), parts: HTMLElement[] = [head];
  head.append(h("div", "invite-head", "Context"));
  if (g.thread) {
    const go = h("button", "btn small", "Compact now");
    go.type = "button";
    go.disabled = !conv.thread?.id || !!running();
    go.onclick = () => { pop.hidden = true; void PAGE_SLASH.find((c) => c.name === "compact")!.run!(""); };
    head.append(go);
  }
  if (g.window) {
    const bar = h("div", `ctx-track${share >= 0.9 ? " near" : share >= 0.7 ? " warm" : ""}`), fill = h("i"), mark = h("b");
    fill.style.width = `${(Math.min(g.used ?? 0, g.window) / g.window) * 100}%`;
    mark.style.left = `${((g.mark ?? 0) / g.window) * 100}%`;
    mark.hidden = !g.mark;
    bar.append(fill, mark);
    parts.push(bar);
  }
  parts.push(...g.rows.map(([label, value]) => { const x = h("div", "ctx-row"); x.append(h("span", "", label), h("span", "ctx-num", value)); return x; }));
  if (g.breakdown?.length) {
    const block = h("div", "ctx-block"), bar = h("div", "ctx-breakdown"), legend = h("div", "ctx-legend");
    block.title = BREAKDOWN_NOTE;
    for (const segment of g.breakdown) {
      const piece = h("div"), item = h("div", "ctx-key"), dot = h("div", "ctx-dot");
      piece.style.width = `${segment.width}%`;
      piece.style.backgroundColor = dot.style.backgroundColor = BREAKDOWN_COLOR[segment.key];
      bar.append(piece);
      item.append(dot, h("div", "", BREAKDOWN_LABEL[segment.key]), h("div", "ctx-weaker", `${segment.percent.toLocaleString("en-US")}%`));
      legend.append(item);
    }
    block.append(h("div", "ctx-label", "Context Breakdown"), bar, legend);
    parts.push(block);
  }
  pop.replaceChildren(...parts);
}
function renderMeter() {
  const meter = $("meter"), g = gauge();
  if (!g) { $("ctx-pop").hidden = true; return void meter.replaceChildren(); }
  let b = meter.querySelector<HTMLButtonElement>(".ctx");
  if (!b) {
    b = h("button", "ctx");
    b.type = "button";
    b.setAttribute("aria-haspopup", "dialog");
    b.setAttribute("aria-controls", "ctx-pop");
    b.innerHTML = `<svg class="ctx-ring" width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><circle cx="8" cy="8" r="6.5" stroke-width="3"/><circle class="on" cx="8" cy="8" r="6.5" stroke-width="3" stroke-dasharray="${RING}" stroke-dashoffset="${RING}"/></svg><span class="ctx-pct"></span><span class="ctx-left"></span>`;
    b.onclick = () => {
      const pop = $("ctx-pop"), r = b!.getBoundingClientRect();
      if (!pop.hidden) return void (pop.hidden = true);
      delete pop.dataset.key;
      renderContext();
      pop.hidden = false;
      pop.style.left = `${Math.max(8, Math.min(r.left, innerWidth - pop.offsetWidth - 8))}px`;
      pop.style.bottom = `${innerHeight - r.top + 8}px`;
    };
    meter.replaceChildren(b);
  }
  const share = g.top ? Math.min(1, (g.used ?? 0) / g.top) : 0;
  b.className = `ctx${share >= 0.9 ? " near" : share >= 0.7 ? " warm" : ""}`;
  b.querySelector(".on")!.setAttribute("stroke-dashoffset", String(RING * (1 - share)));
  b.querySelector(".ctx-pct")!.textContent = g.pct;
  b.querySelector(".ctx-left")!.textContent = g.left;
  b.title = b.ariaLabel = g.title.join("\n");
  if (!$("ctx-pop").hidden) renderContext();
}
function threadControls() {
  const id = threadModel(conv), btn = $("model-btn"), on = harness.state === "on";
  const label = id ? shortName(id.startsWith("chrysalis/") ? id.slice(10) : harness.models?.find((m) => m.id === id)?.name ?? id) : "OpenCode";
  const away = !!id && !id.startsWith("chrysalis/"), local = !conv.thread?.machine, dl = status.download;
  const absent = (model: string) => local && model.startsWith("chrysalis/") && !!status.got && !status.got.includes(model.slice(10));
  const coming = (model: string) => local && dl?.model === model.slice(10), got = (d: NonNullable<Status["download"]>) => Math.round(100 * (d.total ? Math.min(1, d.done / d.total) : 0));
  const missing = !!id && absent(id) && !coming(id);
  const which = id?.startsWith("chrysalis/") ? id.slice(10) : undefined, loaded = !which || !local || (status.state === "online" && !status.loading && status.model === which);
  const busy = !!which && local && (status.loading === which || coming(id!));
  btn.innerHTML = `${away ? icon("warn") : `<span class="dot ${!on || missing ? "offline" : busy ? "loading" : loaded ? "online" : "offline"}"></span>`}<span class="model-name">${escapeHtml(missing ? "Download a model" : label)}</span>${id && coming(id) ? `<span class="model-pct">${icon("download")}${got(dl!)}%</span>` : ""}${icon("chevron")}`;
  btn.classList.toggle("outside", away);
  btn.classList.toggle("getting", !!id && coming(id));
  btn.classList.toggle("paused", !!id && coming(id) && !!dl?.paused);
  btn.style.setProperty("--got", id && coming(id) ? `${got(dl!)}%` : "0%");
  btn.title = missing ? `${modelName(which!)} is not on this computer yet: choose it here and it downloads` : !loaded && !busy ? `${modelName(which!)} is not loaded: the first message loads it` : id ? `${id}, through OpenCode${away ? ": this thread leaves this computer" : ""}` : "OpenCode lists no model";
  btn.setAttribute("aria-expanded", String(modelsOpen));
  btn.classList.toggle("open", modelsOpen);
  renderMeter();
  if (!modelsOpen) return;
  const pop = $("model-pop"), agent = conv.thread!.agent ?? harness.agents?.[0]?.id, key = JSON.stringify(["thread", id, agent, harness.models, harness.agents, speechKey(), status.got, dl && [dl.model, got(dl)]]);
  if (pop.dataset.key === key) return;
  pop.dataset.key = key;
  const groups = Object.entries(Object.groupBy(harness.models ?? [], (m) => m.provider)).sort(([, a], [, b]) => Number(b![0].id.startsWith("chrysalis/")) - Number(a![0].id.startsWith("chrysalis/")));
  if (!groups.length) return void pop.replaceChildren(h("div", "model-empty", "OpenCode lists no model: sign in to a provider with opencode on that computer"), ...speechRows());
  const option = (name: string, title: string, cur: boolean, pick: () => void, model = "") => {
    const item = h("button", cur ? "model-item current" : "model-item"), gone = !!model && absent(model) && !coming(model);
    const mark = h("span", gone ? "model-check model-get" : "model-check");
    item.type = "button";
    item.title = gone ? `Download ${name}, then use it` : title;
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", String(cur));
    if (model && coming(model)) mark.textContent = `Downloading ${got(dl!)}%`;
    else if (gone) mark.textContent = "Download";
    else if (cur) mark.innerHTML = icon("check");
    item.append(h("span", "model-label", name), mark);
    item.onclick = () => {
      pick();
      if (gone) { picked = model.slice(10); setPref("model", picked); void start(true).then(() => refreshStatus()); }
      if (conv.messages.length) save(conv);
      setModelsOpen(false, true);
    };
    return item;
  };
  const fresh = !conv.messages.some((m) => m.role === "user") && !whoOf(conv).startsWith(DIR);
  const plain = fresh ? [option("Chat only", `No tools: a plain chat with the model, without ${harness.name ?? "OpenCode"}`, false, () => { delete conv.thread; setPref("tools", "off"); })] : [];
  pop.replaceChildren(...((harness.agents?.length ?? 0) > 1 || plain.length ? [h("div", "model-group", harness.name ?? "OpenCode"),
    ...(harness.agents ?? []).map((a) => option(a.name, `${harness.name ?? "OpenCode"}'s ${a.name} agent`, a.id === agent, () => { conv.thread!.agent = a.id; })), ...plain] : []),
    ...groups.flatMap(([provider, list]) => [list![0].id.startsWith("chrysalis/") ? h("div", "model-group", provider) : h("div", "model-group outside", `${provider}: leaves this computer`), ...list!.map((m) => option(m.name, m.id, m.id === id, () => { conv.thread!.model = m.id; setPref("thread-model", m.id); }, m.id))]),
    ...speechRows());
}
function renderControls() {
  renderLink();
  document.querySelector("main")!.toggleAttribute("data-thread", !!conv.thread);
  document.querySelector("main")!.toggleAttribute("data-plain", !!conv.thread && !variantsOf(conv).length);
  const effort = barEffort(), at = EFFORTS.indexOf(effort), energy = $("energy");
  energy.dataset.heat = HEAT[at];
  energy.style.setProperty("--v", String(at));
  energy.setAttribute("aria-valuenow", String(at));
  energy.setAttribute("aria-valuetext", LEVEL[effort][0]);
  const level = !!conv.thread && !variantsOf(conv).length;
  energy.title = level ? "This model has no thinking levels" : `Thinking: ${LEVEL[effort][0]}`;
  energy.tabIndex = level ? -1 : 0;
  energy.setAttribute("aria-disabled", String(level));
  energy.querySelectorAll(".energy-dot").forEach((d, k) => d.classList.toggle("lit", at > 0 && k <= at));
  $("energy-stops").dataset.heat = HEAT[at];
  $("energy-stops").querySelectorAll("button").forEach((b, k) => b.classList.toggle("on", k === at));
  const folder = $<HTMLButtonElement>("folder-btn");
  folder.hidden = !conv.thread || harness.state !== "on";
  const run = $<HTMLButtonElement>("machine-btn"), m = machineOf(conv.thread?.machine), fixed = !!conv.thread?.id && (harness.machines?.length ?? 0) < 2;
  run.hidden = folder.hidden || !m || !!conv.thread?.hidden;
  if (m && conv.thread) {
    run.innerHTML = `${icon(machineIcon(m.kind))}<span class="folder-name"></span>${fixed ? "" : icon("chevron")}`;
    run.querySelector(".folder-name")!.textContent = m.name;
    run.title = run.ariaLabel = fixed ? `Runs on ${m.name}` : conv.thread.id ? `Runs on ${m.name}: hand it off to another computer` : `Run on ${m.name}`;
    run.classList.toggle("fixed", fixed);
  }
  if (conv.thread) {
    folder.innerHTML = `${icon("folder")}<span class="folder-name"></span>${icon("chevron")}`;
    folder.querySelector(".folder-name")!.textContent = dirName(conv.thread.dir);
    folder.title = folder.ariaLabel = `Working in ${conv.thread.dir}`;
  }
  if (conv.thread) return threadControls();
  const state = status.loading || status.state === "loading" ? "loading" : status.state;
  const pct = status.progress == null ? "" : `${Math.round(status.progress * 100)}%`, name = modelName(current());
  const btn = $("model-btn");
  const dl = status.download, got = dl?.total ? Math.min(1, dl.done / dl.total) : 0, left = dl?.rate ? Math.round((dl.total - dl.done) / dl.rate) : 0;
  const getting = dl ? `${dl.paused ? "Paused" : "Downloading"} ${modelName(dl.model)}: ${(dl.done / 1e9).toFixed(1)} of ${(dl.total / 1e9).toFixed(1)} GB${dl.rate ? `, ${Math.round(dl.rate / 1e6)} MB/s, ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")} left` : ""}` : "";
  const empty = onService() && !dl && status.got?.length === 0 && state === "offline";
  btn.innerHTML = `<span class="dot ${state}"></span><span class="model-name">${empty ? "Download a model" : shortName(current())}</span>${dl ? `<span class="model-pct">${icon("download")}${Math.round(got * 100)}%</span>` : state === "loading" && pct ? `<span class="model-pct">${pct}</span>` : ""}${icon("chevron")}`;
  btn.style.setProperty("--got", dl ? `${got * 100}%` : "0%");
  btn.classList.toggle("getting", !!dl);
  btn.classList.toggle("paused", !!dl?.paused);
  const b = status.backend, down = b?.listed && !b.listed.ok ? b.listed.error : "";
  btn.title = getting ? getting : empty ? "No model is on this computer yet: choose one here and it downloads" : !onService() ? (down ? `${b!.name}: ${down}` : `${name || "No model"} on ${b!.name}`)
    : state === "online" ? `${name}: ready` : state === "loading" ? `Loading ${name}` : `${name}: stopped`;
  btn.setAttribute("aria-expanded", String(modelsOpen));
  btn.classList.toggle("open", modelsOpen);
  btn.classList.remove("outside");
  renderMeter();
  if (!modelsOpen) return;
  const pop = $("model-pop");
  const tools = harness.state === "on" && !!harness.home && !conv.messages.some((m) => m.role === "user");
  const key = JSON.stringify([current(), state, pct, listed(), tools, speechKey(), status.got, status.download && [status.download.model, !!status.download.paused, Math.round(100 * (status.download.total ? status.download.done / status.download.total : 0))]]);
  if (pop.dataset.key === key) return;
  pop.dataset.key = key;
  const withTools = () => {
    const item = h("button", "model-item");
    item.type = "button";
    item.title = `Answer through ${harness.name ?? "OpenCode"}, with its tools`;
    item.setAttribute("role", "option");
    item.append(h("span", "model-label", `With ${harness.name ?? "OpenCode"}'s tools`), h("span", "model-check"));
    item.onclick = () => { conv.thread = freshThread(); setPref("tools", "on"); setModelsOpen(false, true); };
    return [h("div", "model-group", harness.name ?? "OpenCode"), item, h("div", "model-group", "Chat only")];
  };
  if (!onService()) return void pop.replaceChildren(...(tools ? withTools() : []), ...otherRows(down), ...speechRows());
  const rows = MODELS.map(([id, label]) => {
    const on = id === current(), item = h("button", on ? "model-item current" : "model-item");
    item.type = "button";
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", String(on));
    const absent = !!status.got && !status.got.includes(id), coming = status.download?.model === id;
    const mark = h("span", absent && !coming ? "model-check model-get" : "model-check");
    if (coming) {
      const d = status.download!, tell = (how: string) => (ev: Event) => { ev.stopPropagation(); void fetch("/api/chrysalis/download", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ how }) }).then(() => refreshStatus()); };
      const chip = (label: string, how: string) => { const b = h("span", "model-get model-do", label); b.setAttribute("role", "button"); b.tabIndex = 0; b.onclick = tell(how); return b; };
      mark.className = "model-check model-doing";
      mark.append(h("span", "model-pct", `${Math.round(100 * (d.total ? Math.min(1, d.done / d.total) : 0))}%`), chip(d.paused ? "Resume" : "Pause", d.paused ? "resume" : "pause"), chip("Cancel", "cancel"));
    } else if (absent) mark.textContent = "Download";
    else if (on && state === "loading") mark.textContent = pct || "…";
    else if (on) mark.innerHTML = icon("check");
    item.title = absent ? `Download ${label}, then use it` : label;
    item.append(h("span", "model-label", label), mark);
    item.onclick = () => {
      if (coming) return setModelsOpen(false, true);
      if (absent) { picked = id; setPref("model", id); void start(true).then(() => refreshStatus()); }
      else if (!on || state === "offline") { picked = id; setPref("model", id); void (state === "offline" ? start(true).then(() => refreshStatus()) : pickModel(id)); }
      setModelsOpen(false, true);
    };
    return item;
  });
  pop.replaceChildren(...(tools ? withTools() : []), ...rows, ...speechRows());
}

const SPEECH: [Part, string][] = [["stt", "Speech to text"], ["tts", "Text to speech"]];
const speechNow = (part: Part) => (part === "tts" ? settings.voice?.tts : settings.voice?.stt) ?? voice?.ready?.stack[part] ?? voiceCatalog[part]?.find((m) => m.default)?.name;
const speechKey = () => JSON.stringify(SPEECH.map(([p]) => [speechNow(p), voiceCatalog[p]?.map((m) => [m.name, m.installed])]).concat([[...fetching]]));
function pickSpeech(part: Part, name: string) {
  if (part === "stt") setVoice({ stt: name });
  else if (name !== speechNow("tts")) setVoice({ tts: name, name: undefined });
  saveSettings();
}
function speechRows(): HTMLElement[] {
  return SPEECH.flatMap(([part, title]) => {
    const list = voiceCatalog[part] ?? [], now = speechNow(part);
    return list.length ? [h("div", "model-group", title), ...list.map((m) => {
      const on = m.name === now, item = h("button", on ? "model-item current" : "model-item"), mark = h("span", m.installed ? "model-check" : "model-check model-get");
      item.type = "button";
      item.dataset.speech = `${part}/${m.name}`;
      item.title = m.installed ? `${title}: ${m.label}` : `Download ${m.label}, then use it`;
      item.setAttribute("role", "option");
      item.setAttribute("aria-selected", String(on));
      if (fetching.has(m.name)) mark.textContent = "Downloading…";
      else if (!m.installed) mark.textContent = "Download";
      else if (on) mark.innerHTML = icon("check");
      item.append(h("span", "model-label", m.label), mark);
      item.onclick = async () => {
        if (fetching.has(m.name)) return;
        if (!m.installed) {
          await loadVoices(m.name, part);
          if (!voiceCatalog[part]?.find((x) => x.name === m.name)?.installed) return;
        }
        pickSpeech(part, m.name);
        if (modelsOpen) setModelsOpen(false, true);
      };
      return item;
    })] : [];
  });
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
    if (OLD_SYSTEMS.includes(String(s.system))) delete s.system;
    if (s.voice) delete (s.voice as Record<string, unknown>).prompt;
    for (const k of ["systemTools", "prompts", "descriptions"]) delete s[k];
    if (s.v !== 5) for (const m of [s.think, s.fast]) if (m?.order && m.order.length < DEFAULT_STAGES.length) delete m.order;
    s.v = 5;
    return s;
  }
  const old = load<{ system?: string; sampling?: Record<string, number> } | null>("settings", null);
  const { max_tokens, seed, ...rest } = old?.sampling ?? {};
  return { v: 5, ...(old?.system?.trim() ? { system: old.system.trim() } : {}), think: { ...rest }, fast: { ...rest }, max_tokens, seed };
}
function saveSettings() {
  store("settings2", settings);
  pushVoice();
  if (voice) renderVoice();
  renderVoiceMenu();
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
let toastDone: (() => void) | undefined;
function settleToast() {
  const done = toastDone;
  toastDone = undefined;
  done?.();
}
function toast(text: string, undo?: () => void, done?: () => void) {
  settleToast();
  toastDone = done;
  const t = $("toast"), b = h("button", "btn", "Undo");
  b.type = "button";
  b.onclick = () => { t.hidden = true; toastDone = undefined; undo?.(); };
  t.replaceChildren(h("span", "", text), ...(undo ? [b] : []));
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { t.hidden = true; settleToast(); }, 6000);
}
addEventListener("pagehide", settleToast);

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

  knobs.push(voiceTrigger(settingsVoice, $<HTMLButtonElement>("voice-settings")).paint);
  knobs.push(voiceTrigger(settingsListen, $<HTMLButtonElement>("listen-settings")).paint);
  const speed = knob({
    id: "voice_speed", label: "Speed", help: "Speed: how fast the voice speaks.", min: 0.5, max: 2, step: 0.05,
    fmt: (v) => `${v.toFixed(2)}×`, get: () => settings.voice?.speed ?? 1, rec: () => 1, custom: () => settings.voice?.speed !== undefined,
    set: (v) => setVoice({ speed: v }), pos: (v) => (v - 0.5) / 1.5, val: (p) => 0.5 + p * 1.5,
  });
  $("voice-knobs").replaceChildren(speed);



  $<HTMLTextAreaElement>("system-input").oninput = (e) => {
    const text = (e.target as HTMLTextAreaElement).value;
    if (text.trim()) settings.global = text;
    else delete settings.global;
    saveSettings();
  };
  $("settings-reset").onclick = () => {
    if (!customized()) return toast("Everything is already at the recommended values");
    const before = structuredClone(settings);
    settings = { v: 5, think: {}, fast: {}, ...(settings.global ? { global: settings.global } : {}) };
    saveSettings();
    toast("Everything is back to the recommended values", () => { settings = before; saveSettings(); });
  };
  measureScrollbar();
  addEventListener("resize", measureScrollbar);
}

function measureScrollbar() {
  document.documentElement.style.setProperty("--sb", `${scroller.offsetWidth - scroller.clientWidth}px`);
}

interface VoiceSpot { now(): string; none: string; pick(id: string): void; hear?(id: string): Promise<void>; meta?(): string; part?: Part; what?: string }
const voiceMenu = $("voice-menu"), PLAY = `${icon("wave")}<span class="eq"><i></i><i></i><i></i><i></i><i></i></span>`, fetching = new Set<string>();
let menuSpot: VoiceSpot | undefined, menuAnchor: HTMLElement | undefined, playingVoice: string | undefined;
const voiceParts = (id: string) => (id ? [id.slice(0, id.indexOf("/")), id.slice(id.indexOf("/") + 1)] : [undefined, undefined]);
const voiceIdOf = (tts?: string, name?: string) => (name ? `${tts ?? (voiceCatalog.tts ?? []).find((m) => m.installed)?.name ?? ""}/${name}` : "");
function voiceShown(id: string, none: string, part: Part = "tts") {
  if (part !== "tts") return { name: id ? modelLabel(part, id)! : none, model: "" };
  const [tts, name] = voiceParts(id);
  return id ? { name: (tts && voiceInfo(tts, name!)?.label) || name!, model: modelLabel("tts", tts) ?? "" } : { name: none, model: "" };
}
function playButton(spot: VoiceSpot, id: string, label: string) {
  const b = h("button", "voice-play");
  b.type = "button";
  b.dataset.voice = id;
  b.title = b.ariaLabel = `Hear ${label}`;
  b.innerHTML = PLAY;
  b.addEventListener("click", (e) => { e.stopPropagation(); void playVoice(spot, b.dataset.voice!); });
  return b;
}
async function playVoice(spot: VoiceSpot, id: string) {
  playingVoice = id;
  paintPlaying();
  await spot.hear?.(id);
  if (playingVoice !== id) return;
  playingVoice = undefined;
  paintPlaying();
}
function paintPlaying() {
  for (const b of document.querySelectorAll<HTMLElement>(".voice-play[data-voice]")) b.classList.toggle("playing", b.dataset.voice === playingVoice);
}
function voiceTrigger(spot: VoiceSpot, name: HTMLButtonElement = h("button", "voice-name")) {
  const el = h("span", "voice-pick"), play = spot.hear && playButton(spot, spot.now(), "this voice"), mine = () => voiceMenu.matches(":popover-open") && menuSpot === spot;
  let wasOpen = false;
  name.type = "button";
  name.hidden = false;
  name.setAttribute("aria-haspopup", "dialog");
  name.setAttribute("aria-expanded", "false");
  name.addEventListener("pointerdown", () => { wasOpen = mine(); });
  name.addEventListener("click", () => {
    const close = wasOpen || mine();
    wasOpen = false;
    if (close) voiceMenu.hidePopover(); else openVoiceMenu(spot, name);
  });
  if (name.isConnected) name.replaceWith(el);
  el.append(...(play ? [play] : []), name);
  const paint = () => {
    const id = spot.now(), shown = voiceShown(id, spot.none, spot.part);
    if (play) play.dataset.voice = id;
    name.replaceChildren(h("span", "voice-pick-name", shown.name), ...(shown.model ? [h("span", "voice-pick-model", shown.model)] : []));
    name.insertAdjacentHTML("beforeend", icon("chevron"));
    name.title = `${spot.what ?? "Voice"}: ${shown.name}${shown.model ? ` (${shown.model})` : ""}`;
    paintPlaying();
  };
  paint();
  return { el, paint };
}
function openVoiceMenu(spot: VoiceSpot, anchor: HTMLElement) {
  const host = anchor.closest("dialog") ?? document.body;
  if (voiceMenu.parentElement !== host) host.append(voiceMenu);
  menuSpot = spot;
  menuAnchor = anchor;
  anchor.setAttribute("aria-expanded", "true");
  delete voiceMenu.dataset.key;
  renderVoiceMenu();
  voiceMenu.showPopover();
  placeVoiceMenu();
  if (!voiceCatalog.tts?.length) void loadVoices();
}
function voiceChip(spot: VoiceSpot, id: string, label: string, about = "") {
  const el = h("div", spot.hear ? "voice-chip" : "voice-chip bare");
  el.dataset.voice = id;
  el.tabIndex = 0;
  el.title = about;
  el.setAttribute("role", "radio");
  el.append(...(spot.hear ? [playButton(spot, id, label)] : []), h("span", "", label));
  return el;
}
function renderVoiceMenu() {
  const spot = menuSpot;
  if (!spot) return;
  const part = spot.part ?? "tts", models = voiceCatalog[part] ?? [], cur = spot.now(), meta = spot.meta?.() ?? "";
  const key = JSON.stringify([part, models.map((m) => [m.name, m.installed, m.voices?.map((x) => x.id)]), cur, meta, [...fetching]]);
  if (voiceMenu.dataset.key !== key) {
    voiceMenu.dataset.key = key;
    const out: HTMLElement[] = [], top = h("div", "voice-chips voice-top");
    if (meta) out.push(h("div", "voice-menu-meta", meta));
    top.append(voiceChip(spot, "", spot.none));
    if (part !== "tts") top.append(...models.filter((x) => x.installed).map((m) => voiceChip(spot, m.name, m.label)));
    for (const m of models.filter((x) => !x.installed)) {
      const b = h("button", "voice-chip voice-get", fetching.has(m.name) ? `${m.label} · Downloading…` : `${m.label} · Download`);
      b.type = "button";
      b.dataset.get = m.name;
      b.dataset.part = part;
      b.disabled = fetching.has(m.name);
      top.append(b);
    }
    if (!models.length) top.append(h("span", "voice-wait", `Asking the voice service for its ${part === "tts" ? "voices" : "models"}…`));
    out.push(top);
    for (const m of models.filter((x) => part === "tts" && x.installed && x.voices?.length)) {
      const langs = new Map<string, VoiceInfo[]>();
      for (const x of m.voices!) langs.set(x.lang ? langName(x.lang) : "", [...(langs.get(x.lang ? langName(x.lang) : "") ?? []), x]);
      out.push(h("div", "voice-model", m.label));
      for (const [lang, list] of langs) {
        const chips = h("div", "voice-chips");
        chips.append(...list.map((x) => voiceChip(spot, `${m.name}/${x.id}`, x.label, voiceGroup(x))));
        out.push(h("div", "voice-group", lang), chips);
      }
    }
    if (cur && !out.some((e) => e.querySelector(`[data-voice="${CSS.escape(cur)}"]`))) top.append(voiceChip(spot, cur, part === "tts" ? voiceParts(cur)[1]! : cur, "Not in the voice service's list"));
    voiceMenu.replaceChildren(...out);
    placeVoiceMenu();
  }
  for (const c of voiceMenu.querySelectorAll<HTMLElement>(".voice-chip[data-voice]")) c.setAttribute("aria-checked", String(c.dataset.voice === cur));
  paintPlaying();
}
function placeVoiceMenu() {
  if (!menuAnchor || !voiceMenu.matches(":popover-open")) return;
  const a = menuAnchor.getBoundingClientRect(), m = voiceMenu.getBoundingClientRect(), room = innerHeight - a.bottom - 16;
  const top = m.height <= room || room >= a.top - 16 ? a.bottom + 8 : a.top - 8 - m.height;
  voiceMenu.style.top = `${Math.max(8, Math.min(top, innerHeight - m.height - 8))}px`;
  voiceMenu.style.left = `${Math.max(8, Math.min(a.left, innerWidth - m.width - 8))}px`;
}
voiceMenu.addEventListener("click", (e) => {
  const t = e.target as HTMLElement, chip = t.closest<HTMLElement>(".voice-chip[data-voice]"), get = t.closest<HTMLButtonElement>("[data-get]");
  if (get) return void loadVoices(get.dataset.get, get.dataset.part as Part);
  if (!chip || !menuSpot) return;
  menuSpot.pick(chip.dataset.voice!);
  voiceMenu.hidePopover();
});
voiceMenu.addEventListener("keydown", (e) => {
  const chip = (e.target as HTMLElement).closest<HTMLElement>(".voice-chip[data-voice]");
  if (!chip || !menuSpot || (e.key !== "Enter" && e.key !== " ")) return;
  e.preventDefault();
  menuSpot.pick(chip.dataset.voice!);
  voiceMenu.hidePopover();
});
voiceMenu.addEventListener("toggle", () => {
  for (const b of document.querySelectorAll(".voice-name[aria-expanded]")) b.setAttribute("aria-expanded", String(b === menuAnchor && voiceMenu.matches(":popover-open")));
  if (!voiceMenu.matches(":popover-open")) menuSpot = menuAnchor = undefined;
});
addEventListener("resize", placeVoiceMenu);
async function loadVoices(get?: string, part?: Part) {
  if (get) {
    fetching.add(get);
    samples().get(get, part);
    renderVoiceMenu();
    if (modelsOpen) renderControls();
  }
  const c = await samples().catalog();
  if (get) fetching.delete(get);
  if (c?.tts?.length) { voiceCatalog = c; setPref("voice-catalog", JSON.stringify(c)); }
  renderVoiceMenu();
  if (modelsOpen) renderControls();
  for (const sync of knobs) sync();
  if (voice) renderVoice();
}
const settingsVoice: VoiceSpot = {
  now: () => voiceIdOf(settings.voice?.tts ?? voice?.ready?.stack.tts, settings.voice?.name),
  none: "Default",
  pick: (id) => { const [tts, name] = voiceParts(id); setVoice({ tts, name }); saveSettings(); },
  hear: (id) => { const [tts, name] = voiceParts(id); return samples().preview(tts, name); },
};
const settingsListen: VoiceSpot = {
  part: "stt",
  what: "Speech to text",
  now: () => settings.voice?.stt ?? "",
  none: "Default",
  pick: (id) => { setVoice({ stt: id || undefined }); saveSettings(); },
};


function openSettings(sec?: string) {
  settingsMode = modeOf(conv);
  if (!panel.open) panel.showModal();
  showSettingsSec(sec ?? pref("settings-sec", "chat"));
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
  $("artifacts-switch").setAttribute("aria-checked", String(settings.artifacts !== false));
  const system = $<HTMLTextAreaElement>("system-input");
  if (document.activeElement !== system) system.value = settings.global ?? "";
  $("settings").classList.toggle("other-backend", !onChrysalis());
  for (const sync of knobs) sync();
  drawCharts();
}

document.addEventListener("visibilitychange", () => { if (!document.hidden) void refreshStatus(); });

$("new-chat").innerHTML = icon("pen");
$("side-scope").innerHTML = icon("filter");
$("add-project").innerHTML = icon("folderPlus");
$("side-clear").innerHTML = icon("x");
document.querySelector(".side-search")!.insertAdjacentHTML("afterbegin", icon("search"));
$("add-cast").innerHTML = icon("personAdd");
$("outside-title").insertAdjacentHTML("afterbegin", icon("warn"));
$("outside-keep").onclick = () => $<HTMLDialogElement>("outside-view").close();
$("outside-allow").onclick = () => { $<HTMLDialogElement>("outside-view").close(); void setOutside(true); };
$("opencode").onclick = () => void openHarness();
paintHarness();
void loadThreads();
addEventListener("focus", () => { if (harness.state === "on") void loadThreads(); });
setInterval(() => { if (harness.state === "on" && !document.hidden) void loadThreads(); }, 3000);
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
if (location.protocol === "https:") await checkTrust();
$("to-latest").innerHTML = icon("down");
$("settings-reset").insertAdjacentHTML("afterbegin", icon("retry"));
$("settings-open").innerHTML = icon("gear");
$("settings-open").addEventListener("click", () => openSettings());
$("settings-close").innerHTML = icon("x");
$("settings-close").addEventListener("click", () => panel.close());
panel.addEventListener("click", (e) => { if (e.target === panel) panel.close(); });
panel.addEventListener("close", () => {
  settingsMode = undefined;
  if (menuAnchor?.closest("#settings")) voiceMenu.hidePopover();
  if (playingVoice !== undefined) sampler?.stop();
});
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
  const old = settings as { system?: string }, system = old.system?.trim();
  if ("system" in old) {
    delete old.system;
    if (system) {
      const d = people.get(CHRYSALIS)!.card.data;
      d.description = [system, d.description?.trim()].filter(Boolean).join("\n\n");
      await savePerson(people.get(CHRYSALIS)!);
    }
    saveSettings();
  }
  for (const g of got?.groups ?? []) groups.set(g.id, { id: g.id, members: (g.members ?? []).filter((m) => people.has(m)), added: g.added ?? 0, ...(g.name ? { name: g.name } : {}), ...(g.scenario ? { scenario: g.scenario } : {}) });
  for (const c of Object.values(chats) as (Conversation & { cast?: string[]; castName?: string; scenario?: string; missing?: string[] })[]) if (c.cast) {
    const members = c.cast.filter((m) => people.has(m));
    if (c.who && members.length && !groups.has(c.who)) keepGroup({ id: c.who, members, added: Date.now(), ...(c.castName ? { name: c.castName } : {}), ...(c.scenario ? { scenario: c.scenario } : {}) });
    for (const k of ["cast", "castName", "scenario", "missing"] as const) delete c[k];
  }
  if (got) keepChats();
  const orphans = got ? Object.values(chats).filter(castless).map((c) => c.id) : [];
  if (orphans.length) dropChats(orphans);
  castReady.resolve();
  renderCast();
}
if (location.hash.length > 1) openConversation(location.hash.slice(1), false);
else render();
$("app").removeAttribute("aria-busy");
(async () => {
  for (;;) {
    await sleep(status.state === "loading" || status.loading ? 250 : 4000);
    if (!document.hidden) await refreshStatus();
  }
})();
