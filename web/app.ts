// Chrysalis web UI: chat with Chrysalis's local models (Qwen3.8-27B, Qwen3.8-Flash-Next). Replies stream in with their reasoning and
// tool calls and are rendered as markdown with highlighted code; conversations (every version of them) live in this tab's
// sessionStorage (a reload keeps them, closing the tab ends them). The settings (system prompt, sampling, reply length,
// seed, tools that run without asking), kept the same way, apply to every conversation.
import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/common";
import katex from "katex";
import "katex/dist/katex.min.css";
import { marked, type Links, type Token } from "marked";
import { type Attachment, KINDS, attach, dataUrl, fmtBytes, get as getFile, put as putFile, textBlock } from "./files";
import { type Cand, type Odds, type SamplerKey, type Stage, dryRuns, sample } from "./sampler";
import { type Part, type VoiceAsk, type VoiceInfo, type VoiceReady, VoiceClient } from "./voice";
import { type Fix, applyFixes, fixedParts, readHeard } from "./heard";
import { pointOf, shownWords, unitOf } from "./spoken";
import { DESCRIPTIONS, type ToolChoice, type WireCall, assistantTurn, noted, toolTurn } from "./wire";
import logoThin from "./logo-thin.svg";
import { Ocean } from "./ocean";

type Effort = "xhigh" | "medium" | "low" | "off";
// prompt and cached: summed over a reply's rounds; ctx: the context its last round left (prompt + reply)
interface Stats { ttft: number; tps: number; tokens: number; prompt: number; cached: number; thought?: number; seed?: number; model?: string; ctx?: number }
// A tool call as Chrysalis wrote it (WireCall: its id and argument string, kept byte for byte), and what became of it.
type CallState = "approval" | "running" | "done" | "failed" | "declined" | "stopped";
interface Call extends WireCall { title?: string; source?: string; readOnly?: boolean; state?: CallState; result?: string; ms?: number }
// A reply with tools is rounds: each but the last ended in tool calls, whose results the next one read.
interface Round { content: string; reasoning?: string; calls: Call[] }
// A conversation is a tree, kept as the path on screen: the first message of a stretch that has other versions holds
// them, each with the rest of the conversation that followed it (and their own versions further down).
interface Versions { shown: number; others: Message[][] }
interface Message {
  id?: string; role: "user" | "assistant"; content: string; reasoning?: string; stats?: Stats; error?: string; stopped?: boolean; job?: string;
  params?: string;                                  // the sampling values that differed from the recommended ones
  effort?: Effort;                                  // the thinking setting it was written with (Continue keeps it)
  finish?: string; edited?: boolean;
  files?: Attachment[];                             // a user message's attached files (their contents: files.ts)
  rounds?: Round[];                                 // a reply's earlier rounds; content, reasoning and calls are its last
  calls?: Call[];
  base?: number;                                    // rounds it had before its job began (a reload replays the job's)
  notes?: string[];                                 // what the web server said about the reply (a tool left out, a limit)
  versions?: Versions;
  voice?: boolean;                                  // a question said in voice mode
  voiceMs?: number;                                 // a spoken reply: from the end of the user's speech to its first sound
  interrupted?: boolean;                            // talked over: it keeps only what was heard
  note?: string;                                    // the app's words to the model before a question (voice mode on, off): sent, not shown
  heard?: Fix[];                                    // a question said aloud: the misheard words fixed in it (what was heard, on hover)
}
interface Conversation { id: string; title: string; named?: boolean; updated: number; effort: Effort; messages: Message[]; tools?: ToolChoice["tools"] }
type Sampler = Partial<Record<SamplerKey, number>> & { order?: Stage[] };
type Mode = "think" | "fast";                       // thinking on (Max, Medium, Low) or off: Qwen recommends other values
// Only what differs from the recommended values is kept, so whatever is left alone follows them. allow: tools that run
// without asking (by function name); fetchChars: how much of a page Fetch URL gives the model at a time.
interface Settings {
  v?: 4;                                            // the settings' shape (loadSettings brings older ones to it)
  system: string; think: Sampler; fast: Sampler; max_tokens?: number; seed?: number; allow?: string[]; fetchChars?: number; voice?: VoiceSettings;
}
// Voice mode's, every one of them (Settings and the dock's voice picker both show these): the voice model (a name from
// the service's catalog), the voice (an id of that voice model's), its speed, and what its replies are told instead of
// the system prompt. Left unset: what the service runs when it is not told. They think as the chat's bar says.
interface VoiceSettings { tts?: string; name?: string; speed?: number; prompt?: string }
// The tool sources server.ts offers (tools.ts catalog()).
interface ToolInfo { fn: string; name: string; title: string; description: string; readOnly: boolean }
interface SourceInfo { id: string; label: string; kind: "fetch" | "mcp"; detail: string; state: "idle" | "connecting" | "connected" | "failed" | "disabled"; error?: string; tools: ToolInfo[] }
interface Catalog { config: string; error?: string; sources: SourceInfo[] }
interface Status { state: "online" | "loading" | "offline"; build?: string; model?: string; loading?: string; progress?: number; context?: number; kv_tokens?: number; cache?: { free_kv_tokens: number } }
interface View {
  el: HTMLElement; think: HTMLDetailsElement; summary: HTMLElement; thought: HTMLElement; body: HTMLElement; foot: HTMLElement; shown?: string; blocks?: Blocks;
  tok?: { reasoning: number; content: number };     // tokens drawn so far, shown as tokens
  rounds: HTMLElement; calls: HTMLElement; notes: HTMLElement; roundsShown?: number;   // a reply's earlier rounds, its last round's calls
}
// A token of a reply, with the model's probability for it and its likeliest alternatives (log probs from Chrysalis).
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
  usage?: { prompt_tokens: number; completion_tokens: number; cached: number }; last?: any;   // summed over rounds; the last round's
  seed?: number; timing?: { start: number; first?: number; answer?: number; last?: number; gen?: number };
  pick?: Tok;                                       // the alternative a rewrite starts from
  frame: number; saved: number;
}

// Chrysalis keeps one model on the GPU. The picker by Load / Unload chooses it (switching takes 5-20 s); every chat
// then talks to the model that is loaded.
const MODELS: [string, string, string][] = [
  ["qwen3.8-27b", "Qwen3.8-27B", "Quickest replies · 4 s to load"],
  ["qwen3.8-flash-next", "Qwen3.8-Flash-Next", "Fast at any length · 20 s to load"],
];
const modelName = (id?: string) => MODELS.find(([m]) => m === id)?.[1] ?? id ?? "";
// What Chrysalis uses for a value it is not sent, with thinking on and off (DEFAULTS and MAX_OUTPUT in
// chrysalis/server.py: Qwen's recommended values). Without a seed Chrysalis draws a random one and reports it with the
// reply.
const RECOMMENDED: Record<Mode, Record<SamplerKey, number>> = {
  think: { temperature: 1, top_p: 0.95, top_k: 20, min_p: 0, presence_penalty: 0, dry_multiplier: 0, dry_allowed_length: 2, dry_base: 1.75 },
  fast: { temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0, presence_penalty: 1.5, dry_multiplier: 0, dry_allowed_length: 2, dry_base: 1.75 },
};
const MAX_REPLY = 131072;
const STAGES: Stage[] = ["top_k", "temperature", "top_p", "min_p"];   // Chrysalis's default order (engine/sampler.h)
// `slider` narrows the slider to the useful part of the range; typing still reaches all of it.
const SAMPLERS: Record<SamplerKey, { label: string; short: string; min: number; max: number; step: number; slider?: [number, number]; help: string }> = {
  temperature: { label: "Temperature", short: "temp", min: 0, max: 2, step: 0.01, help: "Temperature: under 1 sharpens the odds, over 1 flattens them; 0 always takes the likeliest." },
  top_p: { label: "Top P", short: "top-p", min: 0, max: 1, step: 0.01, help: "Top P: keeps the likeliest tokens until together they reach this share." },
  top_k: { label: "Top K", short: "top-k", min: 1, max: 64, step: 1, help: "Top K: keeps at most this many tokens (Chrysalis looks at the 64 likeliest)." },
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
// {{MODEL}} is the chat's model by name: told nothing, the model does not know it is Qwen3.8 and says so when asked.
const DEFAULT_SYSTEM = "You are {{MODEL}}, running locally in Chrysalis Inference. There are no tools or files in this chat. Answer directly.";
// What the default says instead in a chat with tools on (a prompt the user wrote goes as written); Chrysalis's template
// writes the tools themselves before it in the system turn (What the model is sent, in Settings, shows them).
const DEFAULT_SYSTEM_TOOLS = "You are {{MODEL}}, running locally in Chrysalis Inference. The tools offered here are the only ones in this chat, and there are no files or projects to explore. Use a tool when it helps; otherwise answer directly.";
// the default of builds before {{MODEL}}: settings that still hold it follow the default
const PLAIN_SYSTEM = "There are no tools or files in this chat. Answer directly.";
// What voice mode tells the model, in a note before the first question said aloud (and again when it changes): the
// system prompt stays as it is, so a chat's history is never read again for voice mode and the chat's own prompt and
// tools still hold (Chrysalis's template allows a system message only first). What speech recognition misheard is fixed
// before the reply, apart from it (checkOf); what is left the model takes as meant, without remarking on it ("I think
// you meant 'wrote'"). "Briefly" alone cut an essay asked for to a sentence or two. Told to write no code or markdown, the
// model wrote code as plain text, which was read out, and held back ("Let me write that for you", then nothing): so what
// is shown (code blocks, tables) is apart from what is said. The sentence on reasoning lets it work out loud while it
// already speaks: on Big Bench Audio (Flash-Next, thinking off, spoken questions through Parakeet, as the system prompt)
// 92.5% of 200 with it, 77.7% of all 1,000 without, where thinking first would hold the first word back by seconds.
const DEFAULT_VOICE_PROMPT = "The user is talking to you by voice; a word may still be misheard, so take what they most likely meant, by " +
  "how it sounds and by the conversation, without remarking on it. Your reply is read aloud, except code blocks and tables, which are " +
  "shown and not read: speak briefly in plain sentences (but what they ask you to write, such as an essay or a story, in full), put code " +
  "and tables in markdown blocks and say what you put there, and do what is asked rather than asking whether to. When a question needs " +
  "counting or logic, work it out loud in a few short steps, then answer.";
// The voice note's sentence for a chat with tools on: the server settles whether a voice reply runs a tool (route in
// server.ts), this keeps the words around a call true to it. The note on the first typed message after voice mode:
// replies are read again, so markdown is back.
const VOICE_TOOLS = "Only say you searched or did something once the tool call for it has returned, and never describe results you " +
  "have not received from a tool. When results come back, give the key points briefly, and say the links are on screen rather than reading them.";
// What speech recognition misheard of a question said aloud, and whether answering it needs a tool: asked at temperature
// 0 before the reply, in a conversation of its own beside the chat (a voice session's: these instructions, then a turn
// per question with the one before it and what the assistant had just said), which Chrysalis keeps cached, so a check
// reads ~40 tokens (~0.3 s on Flash-Next). Nothing of it is part of the chat: asked inside the reply, the model copied
// the fixes into a later reply's words, where they were shown and read out; told in the chat how to answer a check, it
// began replies with one; asked beside the chat with all of it, a check took 0.7-1.2 s. Fixes are swaps of the
// transcript's own words (asked to repeat the whole message, it dropped sentences), by sound (asked by meaning, it made
// "pie game" "Pong game"); heard.ts makes only those that sound like what was heard. Words that do not fit what the
// user asks, everyday ones too, and the question before: of 17 misheard transcripts (twice each) 29 came out right
// where 25 had ("read it" for "write it" fixed), and none of 14 right ones was changed; an example such as "due to" for
// "do a" was copied into right speech ("Do a the rain"), so the examples are words that are rarely right where they
// were heard.
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
// Chrysalis answers a prompt that starts "You are a title generator" as a side request (low priority, no thinking)
const TITLE_PROMPT = "You are a title generator. Reply with a title of at most six words for the conversation that starts with the user's message below. Reply with the title only.";
const DRAFT_PROMPT = "Write the next message I would send in this conversation, as me: same voice, same goals, natural and brief. Reply with only that message.";
const fill = (text: string) => text.replaceAll("{{MODEL}}", modelName(current()) || "a Qwen model");
// Thinking, a slider in the message box: no thinking (the model's no-thinking mode), then Qwen's three efforts.
const EFFORTS: Effort[] = ["off", "low", "medium", "xhigh"];
const LEVEL: Record<Effort, [string]> = { off: ["Off"], low: ["Low"], medium: ["Medium"], xhigh: ["Max"] };
const HEAT = ["off", "low", "medium", "max"];      // each stop's look, soft to vivid
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
  wave: '<path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2"/>',
  play: '<path fill="currentColor" stroke="none" d="M8 5.8v12.4a.8.8 0 0 0 1.2.7l10-6.2a.8.8 0 0 0 0-1.4l-10-6.2a.8.8 0 0 0-1.2.7z"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/>',
  micOff: '<path d="M15 9.4V6a3 3 0 0 0-5.7-1.3M9 9v2a3 3 0 0 0 4.9 2.3M18.5 11a6.5 6.5 0 0 1-1 3.4M5.5 11a6.5 6.5 0 0 0 10.4 5.2M12 17.5V21M4 4l16 16"/>',
};
const TITLES = { copy: "Copy", edit: "Edit", trash: "Delete", tokens: "Token odds", retry: "Regenerate (a new version)" };
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
  settings: pane("settings", panel, ",", 1460, "--panel-open", 340, () => { syncSettings(); growSystem(); void loadCatalog(); }),
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

function renderBlocks(v: { body: HTMLElement; blocks?: Blocks }, src: string, live: boolean) {
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
const SVG_SHADOW = ":host { display: block; color: #1f1b1c; } svg { display: block; width: var(--fig-w, 640px); max-width: 100%; height: auto; } " +
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
const REPORT = `<script>(() => { const html = document.documentElement, post = () => parent.postMessage({ chrysalisHeight: Math.ceil(html.getBoundingClientRect().height) }, "*");
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
  if (e.data?.chrysalisReady) frame.contentWindow!.postMessage({ chrysalisPage: pageText.get(frame) + REPORT }, "*");
  else if (typeof e.data?.chrysalisHeight === "number" && !frame.dataset.fixed) {
    const px = `${Math.round(Math.min(Math.max(e.data.chrysalisHeight, 60), pageRoom()))}px`;
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

// A new chat starts with the thinking and the tools the last one was left with.
function blank(): Conversation {
  let tools: ToolChoice["tools"] | undefined;
  try { tools = JSON.parse(pref("tools", "null")) ?? undefined; } catch { /* none */ }
  return { id: randomId(), title: "New chat", updated: Date.now(), effort: pref("effort", "xhigh") as Effort, messages: [], ...(tools && Object.keys(tools).length ? { tools } : {}) };
}

function render() {
  views.clear();
  thread.replaceChildren(...(conv.messages.length ? conv.messages.map(messageView) : [emptyState()]));
  $("title").textContent = conv.title;
  $("export").hidden = !conv.messages.length;
  document.title = conv.messages.length ? `${conv.title} · Chrysalis Inference` : "Chrysalis Inference";
  renderComposer();
  renderList();
  paintGhost();
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
  foot.innerHTML = pager(m) + (m.voice ? `<span class="said" title="Said in voice mode">${icon("wave")}</span>` : "")
    + slots([["copy", true], ["edit", !running()], ["trash", !running()]]);
  if (m.note) {                                      // what the app told the model with it: sent, so shown
    const note = h("details", "msg-note"), sum = h("summary", "", m.voice ? "Voice mode note" : "Note to the model");
    sum.title = "What the app told the model with this message (Settings, Voice mode)";
    note.append(sum, h("div", "msg-note-text", m.note));
    el.append(note);
  }
  if (m.files?.length) el.append(filesView(m.files));
  if (m.content || !m.files?.length) el.append(heardBubble(m));
  if (unsettled.has(m)) el.classList.add("unsettled");
  el.append(foot);
  return el;
}
function heardBubble(m: Message): HTMLElement {
  const b = h("div", "bubble");
  for (const p of m.heard?.length ? fixedParts(m.content, m.heard) : [m.content]) b.append(typeof p === "string" ? p : fixedSpan(p));
  return b;
}
function fixedSpan(p: Fix): HTMLElement {
  const f = h("span", "fixed", p.said);
  f.title = `Speech recognition heard "${p.heard}"`;
  return f;
}

// A message edited in place: its text, and a reply's thinking. Save keeps the rest of the chat as it is; a question's
// Send makes it a new version, with a new reply (the old one keeps what followed it). Ctrl+Enter takes the main
// action, Esc cancels.
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

// The versions of the conversation from message i on, the one on screen among them at its place.
function versionsAt(c: Conversation, i: number): Message[][] {
  const vs = c.messages[i].versions, all = [...(vs?.others ?? [])];
  all.splice(vs?.shown ?? 0, 0, c.messages.slice(i));
  return all;
}
// Puts version k of `all` (the versions from message i on) on screen; the others go with its first message.
function showVersion(c: Conversation, i: number, all: Message[][], k: number) {
  c.messages.splice(i);
  for (const v of all) delete v[0].versions;
  const others = all.filter((_, j) => j !== k);
  if (others.length) all[k][0].versions = { shown: k, others };
  c.messages.push(...all[k]);
}
// A new version from message i on, starting with `head`, made the one on screen (the others keep all that followed them).
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
// "‹ 2 / 3 ›" under a message that has other versions; nothing under one that has none.
function pager(m: Message) {
  const vs = m.versions;
  if (!vs) return "";
  const n = vs.others.length + 1, k = vs.shown, b = (act: "prev" | "next", on: boolean, label: string) =>
    `<button type="button" class="act" data-act="${act}" title="${label}" aria-label="${label}"${on ? "" : " disabled"}>${icon(act)}</button>`;
  return `<span class="pager" role="group" aria-label="Version ${k + 1} of ${n}">${b("prev", k > 0, "Previous version")}` +
    `<span class="pager-n" aria-hidden="true">${k + 1} / ${n}</span>${b("next", k < n - 1, "Next version")}</span>`;
}

// A message's delete. One with other versions takes this version with it (all that followed it), and the next
// version shows; the rest just loses the message.
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
  // while it streams the thinking box keeps its newest line in view, until it is scrolled up by hand (back at its end,
  // it follows again)
  thought.addEventListener("wheel", (e) => { if (e.deltaY < 0) thought.dataset.held = "1"; }, { passive: true });
  thought.addEventListener("scroll", () => { if (thought.scrollHeight - thought.scrollTop - thought.clientHeight < 4) delete thought.dataset.held; });
  return { el, think, summary, thought, body, foot, rounds, calls, notes };
}

// A finished round of a reply with tools: its thinking (closed), its text and its calls, where they happened.
function roundView(r: Round, m: Message): HTMLElement {
  const el = h("div", "round"), body = h("div", "markdown"), calls = h("div", "tool-cards");
  if (r.reasoning) {
    const think = h("details", "thinking"), summary = h("summary", "", "Thoughts"), thought = h("div", "thought", r.reasoning);
    think.append(summary, thought);
    el.append(think);
  }
  if (r.content) renderBlocks({ body }, r.content, false);
  syncCards(calls, r.calls, m, false);
  el.append(body, calls);
  return el;
}

// Tool calls as cards: what was called, on what (a line of its arguments), and how it went; a click shows the
// arguments and the result. A call waiting for the user shows its arguments with Allow once, Always allow and Deny.
// Cards are kept by call id and only their changed parts redrawn, so one stays open (and focused) while it runs.
const openCards = new Set<string>();                // calls shown with their arguments and result
const callSince = new Map<string, number>();        // when a call began running, as this page saw it
const deciding = new Set<string>();                 // calls whose answer is on its way to the server
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
  try { const j = JSON.parse(text); if (j && typeof j === "object") pretty = JSON.stringify(j, null, 2); } catch { /* not JSON: markdown */ }
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
function syncCards(box: HTMLElement, calls: Call[], m: Message, live: boolean) {
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

// Message actions in fixed places: from the right, delete, edit and copy on every message, then a reply's token odds,
// regenerate and continue. An action that is not available keeps its place empty, so none ever moves.
type Act = keyof typeof TITLES | "cont";
const slots = (list: [Act, boolean, string?][]) => list.map(([a, on, html]) => !on ? `<span class="act ghost" aria-hidden="true"></span>`
  : html ?? `<button type="button" class="act" data-act="${a}" title="${TITLES[a as keyof typeof TITLES]}">${icon(a)}</button>`).join("");

// Fill an assistant message's view; `live` while it streams (cheap to call repeatedly).
function paint(v: View, m: Message, live?: Live) {
  const elapsed = live ? (performance.now() - live.since) / 1000 : 0;
  if (v.roundsShown !== (m.rounds?.length ?? 0)) {
    v.rounds.replaceChildren(...(m.rounds ?? []).map((r) => roundView(r, m)));
    v.roundsShown = m.rounds?.length ?? 0;
  }
  syncCards(v.calls, m.calls ?? [], m, !!live);
  if (v.notes.childElementCount !== (m.notes?.length ?? 0)) v.notes.replaceChildren(...(m.notes ?? []).map((n) => h("div", "note", n)));
  if (live && v.foot.dataset.pager !== pager(m)) v.foot.innerHTML = v.foot.dataset.pager = pager(m);   // while it is written, only its versions
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
  if (typeof m.voiceMs === "number") facts.unshift(`voice ${(m.voiceMs / 1000).toFixed(2)} s`);
  if (m.interrupted) facts.push("interrupted");
  if (m.params) facts.push(m.params);
  if (m.edited) facts.push("edited");
  if (s?.seed !== undefined) facts.push(`<button type="button" class="seed" data-act="seed" data-seed="${s.seed}" title="Use this seed for the next replies">seed ${s.seed}</button>`);
  const last = conv.messages.at(-1) === m, busy = !!running();
  // Continue is spelled out when the reply was cut short (its length limit, Stop, or the tool rounds' limit); after
  // tool calls it carries on from their results
  const cut = m.finish === "length" || m.stopped || (!!m.calls?.some((c) => c.result !== undefined) && !m.error);
  const cont = cut ? `<button type="button" class="btn" data-act="cont" title="Continue">${icon("cont")}Continue</button>`
    : `<button type="button" class="act" data-act="cont" title="Continue">${icon("cont")}</button>`;
  v.foot.innerHTML = pager(m) + `<span class="stats">${facts.join(" · ")}</span>` + slots([["cont", last && !busy, cont], ["retry", !busy]])
    + slots([["tokens", !!m.id && tokens.has(m.id)], ["copy", true], ["edit", !busy], ["trash", !busy]]);
  delete v.foot.dataset.pager;
  v.foot.querySelector("[data-act=tokens]")?.classList.toggle("on", asTokens);
}

// A new chat: the butterfly in its sea of characters (ocean.ts), or the logo alone where WebGL2 is missing.
const ocean = new Ocean();
function emptyState() {
  const e = h("div", "empty");
  if (!ocean.attach(e)) e.innerHTML = `<img class="logo" src="${logoThin}" alt="">`;
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

// A picked token rewrites the reply from there as a new version of it (the one it was stays a click away).
function pickToken(m: Message, ch: "reasoning" | "content", i: number, a: Alt) {
  const c = conv, at = c.messages.indexOf(m);
  if (at < 0 || running(c)) return;
  const td = structuredClone(tokens.get(m.id!)!), before = td[ch].slice(0, i).map((x) => x.t).join("");
  const { versions, stats, edited, ...rest } = m, copy: Message = { ...structuredClone(rest), id: randomId() };
  tokens.set(copy.id!, td);
  if (tokenView.has(m.id!)) tokenView.add(copy.id!);
  branch(c, at, copy);
  if (SPECIAL[a.t] === "end of reply") {            // nothing to write: the reply just ends there
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
  // in voice mode it ends voice mode (Enter still sends what is typed; Esc talks over the reply)
  sendBtn.innerHTML = icon(busy || voice ? "stop" : "send");
  sendBtn.setAttribute("aria-label", voice ? "End voice mode" : busy ? "Stop" : "Send");
  sendBtn.title = voice ? "End voice mode" : "";
  sendBtn.classList.toggle("stop", busy || !!voice);
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
const customized = () => settings.system !== DEFAULT_SYSTEM
  || Object.keys(settings.think).length + Object.keys(settings.fast).length > 0
  || !!settings.max_tokens || settings.seed !== undefined || !!settings.allow?.length || settings.fetchChars !== undefined
  || Object.keys(settings.voice ?? {}).length > 0;

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

// The model picker: what Chrysalis has loaded (or is loading); while Chrysalis is stopped, what Load will start.
let picked = pref("model", MODELS[0][0]);
const current = () => status.loading ?? (status.state === "online" ? status.model : undefined) ?? picked;

async function start() {
  await fetch("/api/chrysalis/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: picked }) });
}

async function pickModel(model: string) {
  picked = model;
  setPref("model", model);
  if (status.state !== "online" || status.model === model) return renderStatus();
  status = { ...status, loading: model, progress: undefined };   // shown at once; the page polls while Chrysalis switches
  renderStatus();
  if (!conv.messages.length) render();
  try {
    const r = await fetch("/api/chrysalis/load", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model }) });
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
    picked = status.model;                         // the picker follows Chrysalis (another client may have switched it)
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

// The request body for a reply after the messages `upto`. Assistant turns go back exactly as Chrysalis sent them, tool
// rounds and all (wire.ts): Chrysalis then replays its own tokens and every turn is a cache hit (only the new message
// is read). The reply's tokens come with their odds (log probs, five alternatives each) for the token view. effort: the
// thinking setting (the chat's by default); sampling: whose values (by default the effort's mode). With tools on,
// chrysalis_web tells the web server which to offer (it runs them: server.ts); loop: false offers them without running
// any.
function request(c: Conversation, upto: Message[], o: { effort?: Effort; sampling?: Mode; final?: object; extra?: Record<string, unknown>; loop?: boolean; route?: boolean } = {}) {
  const effort = o.effort ?? c.effort, mode: Mode = effort === "off" ? "fast" : "think";
  const messages: object[] = upto.flatMap(wire), system = systemFor(c);
  if (system) messages.unshift({ role: "system", content: system });
  if (o.final) messages.push(o.final);
  const web: ToolChoice | undefined = toolsOn(c)
    ? { tools: c.tools!, allow: settings.allow, fetchChars: settings.fetchChars, ...(o.loop === false ? { loop: false } : {}),
      ...(o.route ? { route: true } : {}) } : undefined;
  return {
    model: effort === "off" ? `${current()}-fast` : current(), reasoning_effort: effort === "off" ? undefined : effort,
    ...samplerFields(o.sampling ?? mode), max_tokens: settings.max_tokens, seed: settings.seed, logprobs: true, top_logprobs: 5,
    messages, stream: true, stream_options: { include_usage: true }, ...(web ? { chrysalis_web: web } : {}), ...o.extra,
  };
}
const ran = (c: Call) => c.result !== undefined;
// The end of a reply talked over, as Chrysalis is sent it: kept as it was heard, cut mid-sentence, it taught the model
// to stop so itself (an essay asked for next came back as a sentence or two in 6 of 6 tries; with this, whole in 5 of
// 6).
const TALKED_OVER = " — [cut off: the user talked over this reply]";
// A message as Chrysalis is sent it: a question as its files and text; a reply as each round's turn with the calls that
// ran and their results, then its last text (none when it ended in calls). Calls that never ran are left out.
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
// The system prompt as sent: the user's as written; the default says what is true of the chat (no tools, or these).
const systemFor = (c: Conversation) => fill((settings.system.trim() === DEFAULT_SYSTEM && toolsOn(c) ? DEFAULT_SYSTEM_TOOLS : settings.system).trim());

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

async function ensureOnline(run: { stopping: boolean }) {
  if (status.state === "offline") await start();
  for (const deadline = Date.now() + 180_000; status.state !== "online" || status.loading; await refreshStatus()) {
    if (run.stopping) throw new DOMException("stopped", "AbortError");
    if (Date.now() > deadline) throw new Error("Chrysalis did not start: see `chrysalis logs`");
    await sleep(100);
  }
}

const CUT_OFF = "The reply was cut off: the web server restarted while it was being written";

// A new reply after the messages before `at`: at the end, or as a new version of the reply at `at` (Regenerate).
async function generate(c: Conversation = conv, at = c.messages.length) {
  const m: Message = { id: randomId(), role: "assistant", content: "", reasoning: "", params: changed(modeOf(c)) || undefined, effort: c.effort };
  branch(c, at, m);
  const upto = c.messages.slice(0, at);
  await startRun(c, m, () => request(c, upto));
}

// Writes on from the end of the last reply, or (pick) from one of its tokens replaced by an alternative. Chrysalis is
// given the reply as it generated it (chrysalis_original), finds the exact tokens to keep and serves them from its
// cache; it restates the reply as it now stands before the new text. A reply that ended in tool calls (stopped, or at
// the rounds' limit) carries on from their results instead, in a new round. Both write into the same reply: it grows,
// it is not another version (a picked token's rewrite is: pickToken).
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
  const upto = [...c.messages.slice(0, at), { ...m, content: "", reasoning: "", calls: undefined }];   // with the reply's earlier rounds
  void startRun(c, m, () => request(c, upto, { effort, final, extra }),
    pick && { t: pick.alt.t, p: pick.alt.p, alts: tokens.get(m.id!)?.[pick.ch][pick.i]?.alts ?? [] });
}

// body: the request, made once the conversation's files are read (a file that is gone ends the reply with an error)
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
  const run: Run = { conv: c, m, live: { phase: "waiting", since: performance.now(), note: "Reading the prompt" }, at: 0, stopping: false, frame: 0, saved: 0 };
  runs.set(c.id, run);
  try {
    const res = await fetch(`/api/chat/${m.job}?from=0`);
    if (res.status === 404) throw new Error(CUT_OFF);
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    m.content = m.reasoning = "";                   // the job's events bring them back, its rounds and calls too
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
  if (d.chrysalis_prefix) {                             // a continued reply as it now stands
    m.reasoning = d.chrysalis_prefix.reasoning_content ?? "";
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
  if (ev.usage) {
    const u = run.usage ?? { prompt_tokens: 0, completion_tokens: 0, cached: 0 };
    run.usage = { prompt_tokens: u.prompt_tokens + ev.usage.prompt_tokens, completion_tokens: u.completion_tokens + ev.usage.completion_tokens,
      cached: u.cached + (ev.usage.prompt_tokens_details?.cached_tokens ?? 0) };
    run.last = ev.usage;
  }
  if (ev.seed !== undefined) run.seed ??= ev.seed;
  // tool calls as Chrysalis streams them (by index, their arguments possibly in pieces), the same way server.ts adds
  // them up
  for (const c of d.tool_calls ?? []) {
    if (live.phase === "waiting") live.phase = "writing";
    const calls = (m.calls ??= []), at = (calls[c.index ?? calls.length] ??= { id: "", name: "", args: "" });
    if (c.id) at.id = c.id;
    if (c.function?.name) at.name += c.function.name;
    if (c.function?.arguments) at.args += c.function.arguments;
  }
  if (ev.tool) {
    const c = m.calls?.find((x) => x.id === ev.tool.id);
    if (c) {
      for (const k of ["title", "source", "readOnly", "state", "result", "ms"] as const) if (ev.tool[k] !== undefined) (c as any)[k] = ev.tool[k];
      if (c.state === "running") callSince.set(c.id, now);
      if (c.state === "approval") openCards.delete(c.id);
      deciding.delete(c.id);
    }
  }
  if (ev.round) {                                  // the calls' results are in: Chrysalis reads them in a new round
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
  if (now - run.saved > 2000) {                    // a reload shows the text so far even if the server forgets it
    run.saved = now;
    save(run.conv);
  }
  feedVoice(m);
  schedule(run);
}

// After Chrysalis restates a continued reply: keep the tokens it still starts with; the rest (a picked alternative)
// stands in as one token.
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
    if (voice?.reply?.m === run.m) paintSpoken(voice, performance.now(), true);   // its text's nodes are new
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
  // A reply with tool rounds counts as one: time to its first token, its tokens over the time spent writing them
  // (each round's first token to its last: not the tools' time, nor reading their results), all its prompts.
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
  if (unrun.length && !m.error && !m.stopped && !toolsOn(c))   // shown, not dropped: the reply would otherwise just stop
    m.error = `The model asked to run ${[...new Set(unrun.map((x) => x.name))].join(", ")}, but this chat has no tools on, so the reply ends here.`;
  delete m.job;
  delete m.base;
  cancelAnimationFrame(run.frame);
  if (runs.get(c.id) === run) runs.delete(c.id);
  if (run.gone) return;
  c.updated = Date.now();
  if (c === conv) render();
  else renderList();
  save(c);
  if (!c.named && !m.error) void name(c);
}

// Stop: Chrysalis stops and the reply ends with what it has. (Without a job yet, generate() sends this once it has
// one.)
function stopRun(run: Run, now = false) {
  if (run.stopping && !now) return;
  run.stopping = true;
  if (run.m.job) void fetch(`/api/chat/${run.m.job}/stop`, { method: "POST" }).catch(() => {});
}

// Titles come from the -fast model with a title-generator prompt, which Chrysalis runs as a low-priority request.
async function name(c: Conversation) {
  c.named = true;
  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: `${current()}-fast`, max_tokens: 24, messages: [
        { role: "system", content: TITLE_PROMPT },
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
  const note = noteFor(conv, false);
  conv.messages.push({ role: "user", content: text, ...(files.length ? { files } : {}), ...(note ? { note } : {}) });
  void generate();
}

// Voice mode: the message box becomes a dock (the orb, what it is doing, mute and end) and the conversation goes on in
// the chat as ordinary messages. voice.ts carries the sound; here what the user said becomes a question, its reply's
// text is spoken as it is written, and talking over a reply keeps only what was heard of it.
type VoicePhase = "starting" | "listening" | "thinking" | "speaking" | "muted" | "error";
// sent and spoken: code points of its spoken text (sayable) given to the service, and through the piece last heard to
// begin; sounding: that piece, its text's range and its speech's time (VoiceHandlers.spoken); ended: say_end sent
interface VoiceReply {
  m: Message; sent: number; spoken: number; audible: boolean; ended?: boolean;
  sounding?: { from: number; to: number; at: number; secs: number };
}
interface VoiceMode {
  client: VoiceClient; c: Conversation; ready?: VoiceReady; muted: boolean; partial: string; error?: string;
  reply?: VoiceReply;                               // being spoken, or about to be
  merge?: Message;                                  // a question whose reply was talked over before it said a word: the next turn joins it
  latency?: number;
  now: { voice?: string; speed?: number };          // what the service speaks with: its ready's, then what it was asked
  asked?: VoiceAsk; pushed?: string;                // the last hello or config; the last config (a refused one is not sent again as is)
  defaults: { tts?: string; voices: Record<string, string> };   // what it runs when not told (voices: by voice model)
  level: number; frame: number;                    // the pill's dots: the level they show, their animation frame
  check: { role: string; content: string }[];      // the session's check conversation (CHECK)
  asking: Asking;                                   // the checks of the question being said
  ahead?: Ahead;                                    // its reply, begun before the question stands
}
let voice: VoiceMode | null = null;
// The service's models and voices as its last ready listed them, so Settings can offer them while voice mode is closed.
let voiceCatalog: VoiceReady["catalog"] = (() => { try { return JSON.parse(pref("voice-catalog", "{}")) ?? {}; } catch { return {}; } })();
const cuts = new Map<Message, number | null>();     // replies talked over while still written: what to keep once they end (null: none of it)
const voicePrompt = () => settings.voice?.prompt ?? DEFAULT_VOICE_PROMPT;
const voiceNote = (c: Conversation) => fill([voicePrompt().trim(), toolsOn(c) ? VOICE_TOOLS : ""].filter(Boolean).join(" "));
// The note a new question carries, if any: voice mode's where it starts or its note changed, the typing one where it
// ends (a note marks which: a voice question's note is the voice one).
function noteFor(c: Conversation, said: boolean): string | undefined {
  const users = c.messages.filter((m) => m.role === "user"), prev = users.at(-1);
  if (!said) return prev?.voice ? VOICE_OFF : undefined;
  const last = users.findLast((m) => m.note && m.voice)?.note, note = voiceNote(c);
  return note && (!prev?.voice || last !== note) ? note : undefined;
}
// The conversation as speech recognition is told it, to listen for its names (voice/parts.py, terms): the model's name
// (one the user says), the system prompt, and the chat's last few thousand characters.
function tellVoice(v: VoiceMode) {
  const chat = v.c.messages.map((m) => m.content).join("\n");
  v.client.context(`${modelName(current())}\n${systemFor(v.c)}\n${chat.slice(-4000)}`);
}

async function openVoice() {
  if (voice) return;
  const mine = () => voice?.client === client;
  const client: VoiceClient = new VoiceClient({
    ready: (r) => {
      if (!mine()) return;
      const v = voice!, asked = v.asked ?? {}, first = !v.ready;
      // what it chose for what it was not asked: the parts after the hello, a voice after a voice model's switch
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
    // before it is ready the dock says why it is not; after, a refused ask (a model not installed) is a note and the rest goes on
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
  if (status.state === "offline") void start().then(refreshStatus);   // the first question finds the model loaded
  // http://<this machine's address> is not a secure page, and there the browser has no microphone to give at all
  if (!navigator.mediaDevices?.getUserMedia) {
    voice.error = isSecureContext ? "This browser gives no page a microphone"
      : `The microphone needs a secure page: open https://${location.host} (after chrysalis start --lan)`;
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

// Ending voice mode silences it; a reply still being written goes on as text.
function closeVoice() {
  const v = voice;
  if (!v) return;
  dropAhead(v);
  voice = null;
  cancelAnimationFrame(v.frame);
  (globalThis as any).CSS?.highlights?.delete("voice-unspoken");   // a reply still being written shows whole
  v.client.close();
  setVoicePop(false);
  paintGhost();
  renderVoice();
  input.focus();
}

// What a check found for a transcript, and the turn of the check conversation that asked it (kept once the question
// it checked stands, so the next check is an append to Chrysalis's cache).
interface Heard { text: string; fixes: Fix[]; tool: boolean; asked?: { role: string; content: string }[] }
// The checks of the question being said, by transcript: its words as they settle, a pause's, its turn's (each asked
// once); settled: the newest words speech recognition settled.
interface Asking { checks: Map<string, Promise<Heard>>; done: Map<string, Heard>; busy?: string; settled?: string }
const asking = (): Asking => ({ checks: new Map(), done: new Map() });
// A voice reply begun before its question stands: at a pause (the turn may yet go on), or at the turn while the turn's
// own check is out. Chrysalis writes it (its tools held: server.ts, HELD_MS) and the service speaks it into the
// client's hold, so once the turn ended with its words and the check found what it was sent with, it shows and sounds
// at once. Otherwise it is dropped, and begun again on the question as checked.
interface Ahead {
  text: string;                                     // the transcript it answers
  q: Message;                                       // its question (in the chat from the turn on)
  sent: string; tool: boolean;                      // the question's words as sent (the fixes known then made), and chrysalis_voice.tool
  said?: Message;                                   // the question in the chat, once the turn ended with these words
  check?: Heard;                                    // the check of exactly `text`
  run: Run; reply: VoiceReply; ended: boolean; live: boolean;
}
const unsettled = new WeakSet<Message>();                // questions said aloud whose check is not in yet: grey
const heardAs = new WeakMap<Message, string>();          // such a question's words as they were heard

// CHECK of a transcript: a turn of the session's check conversation (begun again when its instructions change: the
// model, tools on or off; and past 60 turns). A check that fails leaves the words as they were heard.
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
        model: `${current()}-fast`, messages: [...v.check, turn], temperature: 0, presence_penalty: 0, max_tokens: 48 }) });
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
// What is known of a transcript: the check of it, or else of the most of its first words.
function knownFor(v: VoiceMode, text: string): Heard | undefined {
  let best: Heard | undefined;
  for (const h of v.asking.done.values())
    if ((text === h.text || text.startsWith(`${h.text} `)) && (!best || h.text.length > best.text.length)) best = h;
  return best;
}
// The words as speech recognition settles them, checked as they come (one check at a time, the newest words next): they
// show fixed while the user is still talking, and the turn's own, settled as the speech stops, are checked before the
// turn ends. Only settled ones: a check of each look's words (every 0.3 s of speech) kept Chrysalis busy beside the
// reply and its slots turning over, so a new chat's reply read its whole prompt again (Flash-Next: 1.1 s to its first
// token).
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

// The question a turn (or a pause) would make: what was heard, joined to the question it goes on (one still being
// checked, or one whose reply was talked over before it made a sound), if any.
function questionFor(v: VoiceMode, text: string): { text: string; on?: Message } {
  const last = v.c.messages.at(-1), on = last && (last === v.merge || last === v.ahead?.said) ? last : undefined;
  return on ? { text: `${heardAs.get(on) ?? on.content} ${text}`, on } : { text };
}

function startAhead(v: VoiceMode, text: string, on?: Message): Ahead {
  const c = v.c, known = knownFor(v, text), fixed = applyFixes(text, known?.fixes ?? []), tool = known?.tool ?? false;
  const effort = v.c.effort, mode: Mode = effort === "off" ? "fast" : "think";
  const note = on ? undefined : noteFor(c, true);
  const q: Message = on ?? { role: "user", content: fixed.text, voice: true, ...(note ? { note } : {}) };
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
    else if (voice?.ahead === a) endSaying(a.reply);   // its last words are spoken into the hold too
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

// A pause: a reply on its words begins now, so a turn that ends with them is answered at once.
function pauseAhead(v: VoiceMode, text: string) {
  if (v.ahead?.said || running(v.c) || status.state !== "online" || status.loading) return;
  const q = questionFor(v, text.trim());
  if (v.ahead?.text === q.text) return;
  dropAhead(v);
  v.ahead = startAhead(v, q.text, q.on);
}

// The reply begun ahead stands once its words are the turn's and its check found what it was sent with: it joins the
// chat as it is so far and is heard. Otherwise it goes, and the question as checked is answered.
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
  v.asking.settled = undefined;                     // the turn's own: what settles next is said after it
  paintGhost();
  text = text.trim();
  if (!text) return;
  const c = v.c;
  for (let i = 0; running(c) && i < 150; i++) {     // a reply talked over ends within a few ms of its stop
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

// With Chrysalis still starting (or loading another model), a turn waits for its check and then for Chrysalis, as a
// reply does.
async function answerLater(v: VoiceMode, text: string, on?: Message) {
  const c = v.c, note = on ? undefined : noteFor(c, true);
  const said: Message = on ?? { role: "user", content: text, voice: true, ...(note ? { note } : {}) };
  if (!on) {
    if (!c.messages.length) c.title = text.slice(0, 60);
    c.messages.push(said);
  }
  heardAs.set(said, text);
  unsettled.add(said);
  if (c === conv) render();
  const mine = v.asking, heard = await checkOf(v, text);
  if (v.asking !== mine || c.messages.at(-1) !== said) return;   // the user went on: the next turn answers for these words
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

// The chat as the next voice turn will send it, read by Chrysalis while the user starts talking: the voice prompt
// replaces the system prompt (an open chat's whole history is read again) and the first request after a while idle is
// slow (Flash-Next: 1.7-3.3 s to the first token where 0.07-0.25 s warm), so the first turn would wait for both. One
// token, through the same job path as a reply (tools and all, so the prompt is the turn's own), and nothing is kept of
// it.
async function warmVoice(v: VoiceMode) {
  const effort = v.c.effort;
  await ensureOnline({ stopping: false }).catch(() => {});
  if (voice !== v || running(v.c)) return;
  // Chrysalis wants a question to answer: a new chat's stands in for the one to come (its snapshot ends with the
  // prompt)
  const upto: Message[] = v.c.messages.length ? [...v.c.messages] : [{ role: "user", content: "Hi" }];
  const body = { ...request(v.c, upto, { effort }), max_tokens: 1, logprobs: false, top_logprobs: undefined };
  const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).catch(() => null);
  await res?.body?.cancel().catch(() => {});
}

// What of a reply is spoken: each round's text and the text being written (thinking never is). It grows only at its
// end, so what was sent stays a prefix and the service's chunk offsets point into it.
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

// Talked over (the service heard the user, or the orb or Esc): silent at once, the reply stopped, and the chat keeps
// what was heard of it, which is also what the model is told it said. A reply that had not said a word yet is taken
// out, and the user's next words join the question it answered, as one.
function cutVoice(here: boolean) {
  const v = voice;
  if (!v) return;
  if (here) v.client.stop();
  const a = v.ahead;                                // not heard yet: the next words join its question
  dropAhead(v);
  if (a?.said) v.merge = a.said;
  const r = v.reply;
  v.reply = undefined;
  renderVoice();
  if (!r) return;
  // taken out (and the next words joined to its question) only when nothing of it happened yet: a reply that ran tools
  // keeps them and their results, and what was heard of it
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
// How far a reply has been heard, in code points of its spoken text: in the piece sounding, at its speech's pace (a
// word takes about as long as its letters), to the end of the word it is on.
function heardTo(r: VoiceReply, now = performance.now()): number {
  const s = r.sounding;
  if (!s) return r.spoken;
  const text = sayable(r.m), f = Math.min(1, Math.max(0, (now - s.at) / (s.secs * 1000)));
  const at = unitOf(text, s.from + Math.round(f * (s.to - s.from))), word = shownWords(text).findLast((w) => w.start < at);
  return Math.min(s.to, Math.max(s.from, word ? pointOf(text, word.end) : 0));
}
// The words of a reply not yet heard, grey (the text being read, style.css ::highlight(voice-unspoken)): those after the
// place heardTo finds, counted among its shown words as the page shows them (spoken.ts).
const UNREAD_ELS = ".code, table, .katex, figure, svg, button";
let spokenAt = 0;
function paintSpoken(v: VoiceMode, now: number, painted = false) {
  const reg = (globalThis as any).CSS?.highlights as Map<string, unknown> | undefined, r = v.reply, view = r && views.get(r.m);
  if (!reg) return;
  if (!r || !view || cuts.has(r.m)) return void reg.delete("voice-unspoken");
  if (!painted && now - spokenAt < 80) return;
  spokenAt = now;
  const text = sayable(r.m), at = unitOf(text, r.sounding ? heardTo(r, now) : 0);
  const n = shownWords(text).filter((w) => w.start < at).length;   // the word it is on is heard
  const boxes = [...view.rounds.querySelectorAll<HTMLElement>(":scope > .round > .markdown"), view.body], from = afterWords(boxes, n);
  if (!from) return void reg.delete("voice-unspoken");
  const range = new Range();
  range.setStart(from.node, from.offset);
  range.setEnd(view.body, view.body.childNodes.length);
  reg.set("voice-unspoken", new (globalThis as any).Highlight(range));
}
// The place after the first n words of these boxes' text as it is shown (runs with a letter or digit, across the
// inline elements in a block; nothing of what is not read), or null when they hold no more.
function afterWords(boxes: HTMLElement[], n: number): { node: Text; offset: number } | null {
  let count = 0, inWord = false, wordy = false, block: Element | null = null, end: { node: Text; offset: number } | null = null;
  const ended = () => {                             // the word under way ended at `end`
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
// `to` counts code points of the spoken text, as the service does.
function keepHeard(c: Conversation, m: Message, to: number | null) {
  if (to === null) {
    const at = c.messages.indexOf(m);
    if (at >= 0 && at === c.messages.length - 1) c.messages.splice(at, 1);
    return;
  }
  const before = sayable({ ...m, content: "" }), skip = before ? [...before].length + 2 : 0;
  m.content = [...m.content].slice(0, Math.max(0, to - skip)).join("").trimEnd();
  m.interrupted = true;
  if (m.id) tokens.delete(m.id);                    // its token odds no longer match its text
}

// The user's words while they speak: a faint message at the end of the chat, until the turn makes it a real one. The
// words checked so far are in their own color, fixed.
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

// Listening once the microphone gives audio, whether or not the service has its models yet: it keeps what comes before
// them, and what is said before the microphone opens (a phone's can take 3 s) is not heard at all.
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
  // the stack heads the voice list; its names, kind by kind, are the meta line's title
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

// The pill's dots: bars the voice being heard stretches, the user's while listening and the reply's while speaking, each
// a little apart so they read as a voice and not a meter; a wave of light runs along them while thinking and they
// pulse while voice mode starts. Still dots when muted, with reduced motion, or on an error.
const DOT_WEIGHT = [0.45, 0.8, 1, 0.8, 0.45];
function animateVoice(now: number) {
  const v = voice;
  if (!v) return;
  v.frame = requestAnimationFrame(animateVoice);
  paintVoice();
  paintSpoken(v, now);
  const phase = voicePhase(v), still = reducedMotion.matches || phase === "muted" || phase === "error", t = now / 1000;
  const lv = v.client.levels(), target = still ? 0 : Math.min(1, (phase === "speaking" ? lv.out : phase === "listening" ? lv.mic : 0) * 6);
  v.level += (target - v.level) * (target > v.level ? 0.5 : 0.12);    // quick to rise, slower to fall: it follows syllables
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
// What the settings ask the service for; with voice mode open, what is unset is filled in with what the service runs
// when not told (so taking a choice back goes back to that).
function voiceWant(v: VoiceMode | null): VoiceAsk {
  const s = settings.voice ?? {}, d = v?.defaults, tts = s.tts ?? d?.tts, name = s.name ?? (tts ? d?.voices[tts] : undefined);
  const stack: VoiceAsk["stack"] = tts ? { tts } : {};
  return { ...(Object.keys(stack).length ? { stack } : {}), ...(name ? { voice: name } : {}), speed: s.speed ?? 1 };
}
// The settings to the service, as far as they differ from what it runs: a model switch it loads (and is ready again),
// a voice or speed from its next sentence.
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
// The voice in use: the settings' pick, else what the service speaks with for its voice model.
function currentVoice() {
  const s = settings.voice, r = voice?.ready, tts = s?.tts ?? r?.stack.tts;
  return { tts, id: s?.name ?? (r && tts === r.stack.tts ? voice!.now.voice : undefined) };
}
const voiceInfo = (tts?: string, id?: string) => voiceCatalog.tts?.find((m) => m.name === tts)?.voices?.find((x) => x.id === id);
const LANGS = new Intl.DisplayNames(["en"], { type: "language" });
const langName = (code: string) => { try { return LANGS.of(code) ?? code; } catch { return code; } };
const voiceGroup = (x: VoiceInfo) => [langName(x.lang), x.gender ? x.gender[0].toUpperCase() + x.gender.slice(1) : ""].filter(Boolean).join(" · ");
const modelLabel = (part: Part, name?: string) => (name && voiceCatalog[part]?.find((m) => m.name === name)?.label) || name;
const setupHint = (part: Part, name: string) => voiceCatalog[part]?.find((m) => m.name === name)?.hint ?? `tools/voice_setup.sh ${name}`;

// A voice from the dock's picker: from the next sentence, and a voice of another voice model switches to that model.
function pickVoice(tts: string, x: VoiceInfo) {
  const before = { tts: settings.voice?.tts, name: settings.voice?.name };
  setVoice({ tts, name: x.id });
  saveSettings();
  setVoicePop(false, true);
  toast(`Voice: ${x.label}`, () => { setVoice(before); saveSettings(); });
}

// The voice picker beside the pill, as the message box's model picker: the voice's name, opening the voice models' voices by
// language and gender, each with a play button for its sample. A model that is not installed shows how to set it up.
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
    if (!m.installed) head.append(h("code", "voice-hint", setupHint("tts", m.name)));
    head.title = m.installed ? "" : "Not installed: this command sets it up";
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

// Picked with the clip, dropped anywhere on the page or pasted: shown as chips in the message box until it is sent,
// then above the message (pictures as pictures). Chrysalis is sent each file before the text that was typed.
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

// A user message as Chrysalis gets it: its files (pictures as images, PDFs as files, videos as videos, texts as fenced
// blocks), then the text typed.
function userContent(m: Message): string | object[] {
  const said = noted(m.note, m.content);
  if (!m.files?.length) return said;
  const parts: object[] = m.files.map((a) => {
    const body = bodies.get(a.id) ?? "";
    if (a.kind === "image") return { type: "image_url", image_url: { url: body } };
    if (a.kind === "pdf") return { type: "file", file: { file_data: body, filename: a.name } };
    if (a.kind === "video") return { type: "video_url", video_url: { url: body } };
    return { type: "text", text: `${textBlock(a.name, body)}\n\n` };
  });
  if (m.note) parts.unshift({ type: "text", text: noted(m.note, "") });
  if (m.content) parts.push({ type: "text", text: m.content });
  return parts;
}

// Chats live and die with the tab; a file keeps one. Export writes the open chat (or all of them) as JSON, Import adds
// chats from such files under new ids, so nothing already here is replaced. Token odds stay behind (this page only).
interface ChatFile { chrysalis: "chats"; version: 1; exported: string; conversations: Conversation[]; files?: Record<string, string> }

function download(name: string, data: string, type: string) {
  const a = h("a");
  a.href = URL.createObjectURL(new Blob([data], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// Attached files go along (every version's): texts as they are, pictures as data URLs.
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

const blobOf = (url: string) => {                       // a data URL's bytes (fetch would need connect-src data:)
  const [head, b64] = url.split(",", 2), bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));
  return new Blob([bytes], { type: head.slice(5).split(";")[0] });
};
const fileName = (title: string) => `${title.replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "chat"}.chrysalis.json`;

// One chat from a file, or null if it is not one: roles and text only where Chrysalis expects them, tool rounds with
// their calls as Chrysalis wrote them, versions as the tree they were. Files from before versions and tools have
// neither.
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
  for (const c of added) save(c);
  if (added.length) openConversation(added.at(-1)!.id);
  toast(added.length ? `Imported ${added.length} chat${added.length > 1 ? "s" : ""}${bad ? ` (${bad} not a Chrysalis chat)` : ""}`
    : "Nothing to import: not a chat exported from Chrysalis");
}

// Chrysalis writes the message the user would likely send next, into the composer to edit and send. It is asked as the
// assistant (Qwen will not write a user turn: at one it just ends it), without thinking, from the conversation
// Chrysalis has cached, which it leaves as it was (chrysalis_aside). Text already in the composer is how the draft
// begins.
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

function openConversation(id: string) {
  closeVoice();
  conv = runs.get(id)?.conv ?? (chats[id] ? structuredClone(chats[id]) : blank());   // a reply being written keeps going
  closeOver(panes.conversations);
  render();
  scroller.scrollTop = scroller.scrollHeight;
  follow = true;
  keepUp();
}

function newChat() {
  closeVoice();
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
  // switching versions works while a reply is written (in any of them: it goes on, and shows again when its version does)
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

// A card's head opens and closes it in place (no redraw, so the focus stays); "Show all" opens a long result.
function toggleCard(btn: HTMLElement) {
  const card = btn.closest<HTMLElement>(".tool-card")!, id = card.dataset.call!;
  if (btn.dataset.act === "more") return void btn.closest(".tool-result")!.classList.remove("long");
  if (!openCards.delete(id)) openCards.add(id);
  const m = conv.messages[Number(card.closest<HTMLElement>("article")?.dataset.i)], round = card.closest<HTMLElement>(".round");
  const calls = round ? m.rounds?.[[...round.parentElement!.children].indexOf(round)]?.calls : m.calls;
  syncCards(card.parentElement!, calls ?? [], m, running()?.m === m);
}

// The user's answer to a call that asks first, to the web server (whichever page answers, the reply goes on).
// Always allow also lets the tool run without asking from now on, in every chat of the tab (Settings lists it).
async function decide(btn: HTMLElement) {
  const card = btn.closest<HTMLElement>(".tool-card")!, id = card.dataset.call!, decision = btn.dataset.decision!;
  const m = conv.messages[Number(card.closest<HTMLElement>("article")?.dataset.i)], c = m?.calls?.find((x) => x.id === id);
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
  if (toolsOpen) return setToolsOpen(false, true);
  if (modelsOpen) return setModelsOpen(false, true);
  if (voicePop) return setVoicePop(false, true);
  if (document.querySelector("dialog[open]")) return;
  const over = PANES.find((p) => isOpen(p) && !p.beside.matches);
  if (over) return setPane(over, false);
  if (voice) return voice.reply || voice.client.playing ? cutVoice(true) : closeVoice();
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
// Thinking is an energy bar of four stops, Off to Max, always in view: it grows more vivid to the right (Medium flows;
// Max is brightest: it flows faster, a shine sweeps it, sparks rise from it). The model's name beside the send button
// opens the models, a flat list; it is the only place a model is chosen (the sidebar shows the engine's state).
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
    const at = EFFORTS.indexOf(barEffort()), step = ({ ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 } as Record<string, number>)[e.key];
    if (step) setEffort(at + step);
    else if (e.key === "Home" || e.key === "End") setEffort(e.key === "Home" ? 0 : 3);
    else return;
    e.preventDefault();
  });
  $("model-btn").addEventListener("click", () => setModelsOpen(!modelsOpen));
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

// The thinking bar sets what the chat's next reply thinks with, spoken or written.
const barEffort = (): Effort => conv.effort;
function setEffort(i: number) {
  const e = EFFORTS[Math.min(3, Math.max(0, i))];
  if (e === barEffort()) return;
  conv.effort = e;
  setPref("effort", e);
  renderComposer();
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

// Tools, beside the thinking bar: a wrench with how many tools this chat has on. It opens them by source: Fetch URL,
// then each MCP server of the config file with its state, a switch for the whole server and, a click further, one for
// each of its tools. The choice is the chat's (a new chat starts from the last one, as the thinking bar does) and goes
// with each request; the web server starts a server the first time a chat turns it on. What a server runs is the
// config file's alone: nothing here can change it (tools.ts).
let catalog: Catalog | null = null, toolsOpen = false;
const shownTools = new Set<string>();               // servers shown with their tools in the popover
const JSON_POST = { method: "POST", headers: { "Content-Type": "application/json" } };
const FETCH_CHARS = 20_000;                         // what Fetch URL gives at a time unless Settings says otherwise (tool-fetch.ts' default)
const EXAMPLE = '{"mcpServers": {"notes": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]}}}';

async function loadCatalog(how: { reload?: boolean; connect?: string } = {}) {
  try {
    const r = how.reload ? await fetch("/api/tools/reload", JSON_POST)
      : how.connect ? await fetch("/api/tools/connect", { ...JSON_POST, body: JSON.stringify({ source: how.connect }) }) : await fetch("/api/tools");
    if (r.ok) catalog = await r.json();
  } catch { /* the web server is restarting: keep what is shown */ }
  renderTools();
  void watchTools();
}
// While a server starts, its state (then its tools) twice a second.
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
  if (isOpen(panes.settings)) syncSettings();
}
// A whole source on (all its tools; `only`: just that one) or off. One that is not running starts now.
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
  if (s && s.tools.every((t) => off.has(t.name))) return setSource(id, false);   // the last one off: the server is off
  setTools({ ...conv.tools, [id]: off.size ? { off: [...off].sort() } : {} });
}

const dotOf = (s: SourceInfo) => (s.state === "connected" ? "online" : s.state === "connecting" ? "loading" : s.state === "failed" ? "failed" : "");
function stateLine(s: SourceInfo) {
  const n = s.tools.length, k = s.tools.filter((t) => toolOn(s.id, t.name)).length;
  if (s.kind === "fetch") return "Reads web pages at public addresses";
  return s.state === "connected" ? `${k} of ${n} tool${n === 1 ? "" : "s"} on` : s.state === "connecting" ? "Starting…"
    : s.state === "failed" ? "Could not start" : s.state === "disabled" ? "Disabled in the config file" : isOn(s.id) ? "Starts with the next reply" : "Not started";
}
const toolTag = (t: ToolInfo) => (t.readOnly ? ["read-only", "Changes nothing: runs without asking"]
  : settings.allow?.includes(t.fn) ? ["runs without asking", "You allowed it for good (Settings, Tools)"] : ["asks first", "Asks you before each run"]);
// The popover's controls say what they do in data-do (and on which source and tool); data-k names each one, so it
// keeps the focus across a redraw.
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
  const lead = s.kind === "fetch" ? h("span", "tp-ico") : h("span", `dot ${dotOf(s)}`);
  if (s.kind === "fetch") lead.innerHTML = icon("globe");
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
      label.title = t.description || t.name;
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
  renderToolSettings();
  if (!toolsOpen) return;
  // redrawn only when something in it changed (a poll never takes the focus from it), and the focus is kept
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
    const fetchSrc = catalog.sources.filter((s) => s.kind === "fetch"), mcp = catalog.sources.filter((s) => s.kind === "mcp");
    parts.push(...fetchSrc.map(sourceView), h("div", "tp-group", "MCP servers"));
    if (mcp.length) parts.push(...mcp.map(sourceView));
    else {
      const help = h("div", "tp-empty");
      help.append("None yet. Add servers to ", h("code", "", catalog.config), " (the shape Claude Desktop and Claude Code use), for example:",
        h("pre", "tp-code", EXAMPLE), "then Reload.");
      parts.push(help);
    }
    const foot = h("div", "tp-foot"), reload = h("button", "btn small", "Reload config");
    reload.type = "button";
    keyed(reload, "reload");
    reload.title = `Read ${catalog.config} again`;
    foot.append(reload);
    parts.push(foot);
  }
  pop.replaceChildren(...parts);
  if (focus) pop.querySelector<HTMLElement>(`[data-k="${CSS.escape(focus)}"]`)?.focus();
}
// Settings' Tools: the config file's servers as they stand (what each runs or reaches, read-only), what Fetch URL
// may reach, and the tools allowed to run without asking (a click makes one ask again).
const STATE_WORD: Record<SourceInfo["state"], string> = { idle: "not started", connecting: "starting…", connected: "running", failed: "failed", disabled: "disabled" };
// Each tool opens to its description as Chrysalis is sent it: Chrysalis's own for that tool where it has one (wire.ts),
// else its server's. A description is what tells the model when to call the tool and when not.
function toolDescriptions(tools: ToolInfo[]): HTMLElement[] {
  return [...tools].sort((a, b) => (a.fn < b.fn ? -1 : 1)).map((t) => {
    const d = h("details", "ts-tool"), sum = h("summary");
    sum.append(h("span", "ts-tool-name", t.title || t.name), h("code", "ts-tool-fn", t.fn));
    d.dataset.fn = t.fn;
    d.append(sum, h("p", "ts-tool-desc", DESCRIPTIONS[t.name] ?? t.description));
    if (DESCRIPTIONS[t.name] && DESCRIPTIONS[t.name] !== t.description) {
      const own = h("p", "set-note");
      own.append("Chrysalis's description for it; the server's own: ", h("q", "", t.description));
      d.append(own);
    }
    return d;
  });
}
function renderToolSettings() {
  if (!isOpen(panes.settings)) return;
  const list = $("tools-list"), key = JSON.stringify(catalog);
  if (list.dataset.key !== key) {
    list.dataset.key = key;
    const rows: HTMLElement[] = [];
    if (!catalog) rows.push(h("p", "set-note", "Asking the web server…"));
    else {
      if (catalog.error) rows.push(h("div", "tp-err", catalog.error));
      const mcp = catalog.sources.filter((s) => s.kind === "mcp");
      for (const s of mcp) {
        const r = h("div", "ts-row"), top = h("div", "ts-top"), n = s.tools.length;
        top.append(h("span", `dot ${dotOf(s)}`), h("span", "ts-name", s.label),
          h("span", "ts-state", s.state === "connected" ? `${n} tool${n === 1 ? "" : "s"}` : STATE_WORD[s.state]));
        const detail = h("code", "ts-detail", s.detail || "(no command or URL)");
        detail.title = s.detail;
        r.append(top, detail);
        if (s.error) r.append(h("div", "ts-err", s.error));
        r.append(...toolDescriptions(s.tools));
        rows.push(r);
      }
      if (!mcp.length) {
        const help = h("p", "set-note");
        help.append("No MCP servers. Add them to the config file, for example: ", h("code", "", EXAMPLE));
        rows.push(help);
      }
      const f = catalog.sources.find((s) => s.kind === "fetch");
      if (f) {
        const r = h("div", "ts-row"), top = h("div", "ts-top");
        top.append(h("span", `dot ${dotOf(f)}`), h("span", "ts-name", f.label), h("span", "ts-state", "built in"));
        r.append(top, ...toolDescriptions(f.tools));
        rows.push(r);
      }
      const note = h("p", "set-note");
      note.append(`Fetch URL reaches ${f?.detail ?? "public addresses only"}. Config file: `, h("code", "", catalog.config));
      rows.push(note);
    }
    list.replaceChildren(...rows);
  }
  const allow = $("allow-list"), akey = JSON.stringify(settings.allow ?? []);
  if (allow.dataset.key === akey) return;
  allow.dataset.key = akey;
  allow.hidden = !settings.allow?.length;
  allow.replaceChildren(h("span", "set-note", "Run without asking:"), ...(settings.allow ?? []).map((fn) => {
    const b = h("button", "chip", fn);
    b.type = "button";
    b.dataset.allow = fn;
    b.title = b.ariaLabel = `${fn}: ask first again`;
    b.insertAdjacentHTML("beforeend", icon("x"));
    return b;
  }));
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
  });
  document.addEventListener("pointerdown", (e) => {
    if (toolsOpen && !(e.target as Element).closest("#tools-pop, #tools-btn")) setToolsOpen(false);
  });
}

const shortName = (id: string) => modelName(id).replace(/^Qwen3\.8-/, "");
function renderControls() {
  renderTools();
  const effort = barEffort(), at = EFFORTS.indexOf(effort), energy = $("energy");
  energy.dataset.heat = HEAT[at];
  energy.style.setProperty("--v", String(at));
  energy.setAttribute("aria-valuenow", String(at));
  energy.setAttribute("aria-valuetext", LEVEL[effort][0]);
  energy.title = `Thinking: ${LEVEL[effort][0]}`;
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
  // usage, in view: this chat's length against the context, and how full Chrysalis's cache of conversations is
  const used = status.kv_tokens && status.cache ? 1 - status.cache.free_kv_tokens / status.kv_tokens : 0, ctx = fmtK(status.context ?? 262144);
  const s = conv.messages.findLast((m) => m.stats)?.stats, meter = $("meter");
  meter.textContent = [s ? `${fmtK(s.ctx ?? s.prompt + s.tokens)} / ${ctx} context` : state === "online" ? `${ctx} context` : "",
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
// reply and can be undone; each value shows what Chrysalis is sent, marked where it differs from the recommended one.

const modeOf = (c: Conversation): Mode => (c.effort === "off" ? "fast" : "think");
let chatMode: Mode | undefined;                     // the Think setting Settings last showed
const shownMode = (): Mode => modeOf(conv);             // the values Settings shows: those of the chat's Think setting
let systemEdited = false;

function loadSettings(): Settings {
  const s = load<(Settings & Record<string, unknown>) | null>("settings2", null);
  if (s) {
    if (s.system === PLAIN_SYSTEM) s.system = DEFAULT_SYSTEM;   // the plain default (before {{MODEL}}) follows the default
    // v3 kept a second system prompt for chats with tools, the other texts sent and each tool's description: gone
    for (const k of ["systemTools", "prompts", "descriptions"]) delete s[k];
    s.v = 4;
    return s;
  }
  // the earlier shape: one set of values for both modes, and a system prompt added after a built-in line
  const old = load<{ system?: string; sampling?: Record<string, number> } | null>("settings", null);
  const { max_tokens, seed, ...rest } = old?.sampling ?? {};
  return { v: 4, system: [DEFAULT_SYSTEM, old?.system?.trim()].filter(Boolean).join("\n\n"), think: { ...rest }, fast: { ...rest }, max_tokens, seed };
}
function saveSettings() {
  store("settings2", settings);
  pushVoice();
  if (voice) { renderVoice(); renderVoicePop(); }
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

// Two real moments of Qwen3.8-Flash-Next run through Chrysalis's sampler (make_dist in engine/sampler.h) with the
// settings as they stand: the first name of a reply (what the stages do) and the next name of a list that has started
// to repeat itself (what presence and DRY do). Each sits above the values it shows.

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

  // Fetch URL's page length: what is safe to change from here (the private-address rule is the config file's)
  const FLO = 2000, FHI = 100_000;
  $("tools-knobs").replaceChildren(knob({
    id: "fetchChars", label: "Page length", min: FLO, max: FHI, step: 1000,
    help: "Fetch URL: how much of a page the model gets at a time, in characters (it can ask for the rest).",
    fmt: (v) => v.toLocaleString("en-US"), get: () => settings.fetchChars ?? FETCH_CHARS, rec: () => FETCH_CHARS,
    custom: () => settings.fetchChars !== undefined, set: (v) => { settings.fetchChars = v; },
    pos: (v) => Math.min(1, Math.max(0, Math.log(v / FLO) / Math.log(FHI / FLO))), val: (p) => Math.round(FLO * (FHI / FLO) ** p / 1000) * 1000,
  }));
  // Voice mode: its prompt, the voice (the service's list, kept from the last time it said), its speed, its thinking.
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
  // a choice row: label, the select, ↺ back to what the service runs when not told
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
    if (now && !listed.includes(now)) groups.push(new Option(s!.name!, now));   // a voice the service no longer lists
    pick.select.replaceChildren(...groups);
    pick.select.value = now;
  });
  const speed = knob({
    id: "voice_speed", label: "Speed", help: "Speed: how fast the voice speaks.", min: 0.5, max: 2, step: 0.05,
    fmt: (v) => `${v.toFixed(2)}×`, get: () => settings.voice?.speed ?? 1, rec: () => 1, custom: () => settings.voice?.speed !== undefined,
    set: (v) => setVoice({ speed: v }), pos: (v) => (v - 0.5) / 1.5, val: (p) => 0.5 + p * 1.5,
  });
  $("voice-knobs").replaceChildren(pick.row, speed);

  $("tools-reload").onclick = () => void loadCatalog({ reload: true }).then(() => toast(catalog?.error ? "The config file has an error (shown under Tools)"
    : `Config read: ${catalog?.sources.filter((s) => s.kind === "mcp").length ?? 0} MCP server(s)`));
  $("allow-list").addEventListener("click", (e) => {
    const fn = (e.target as Element).closest<HTMLElement>("[data-allow]")?.dataset.allow;
    if (!fn) return;
    const before = settings.allow;
    settings.allow = settings.allow?.filter((x) => x !== fn);
    if (!settings.allow?.length) delete settings.allow;
    saveSettings();
    toast(`${fn} asks first again`, () => { settings.allow = before; saveSettings(); });
  });


  systemInput.addEventListener("input", () => {
    settings.system = systemInput.value;
    systemEdited = true;
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
    settings = { v: 4, system: DEFAULT_SYSTEM, think: {}, fast: {} };
    saveSettings();
    growSystem();
    toast("Everything is back to the recommended values", () => { settings = before; saveSettings(); growSystem(); });
  };
  measureScrollbar();
  addEventListener("resize", measureScrollbar);
}

// What the model is sent: the open chat's next request as the web server builds it for Chrysalis (/api/preview, the
// same code as a reply's), so it is exactly that: the tools, which Chrysalis's chat template writes first in the system
// turn, the system prompt after them, then every message as sent, notes included. A file shows as what it is and its
// size, not its data.
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
  $("sent-about").textContent = (sent.tools?.length ? "Chrysalis's chat template writes the tools first in the system turn (\"# Tools … <tools> … </tools>\"), " +
    "with how to call them, and the system prompt after them. " : "") + "Your next message goes after these." +
    (said ? " Each spoken question is first checked, in a conversation of its own beside this one, for what speech recognition misheard (and with tools whether one is needed): the question is sent fixed." : "");
  $("sent-text").textContent = lines.join("\n");
  $<HTMLDialogElement>("sent-view").showModal();
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
  const vp = $<HTMLTextAreaElement>("voice-prompt");
  if (document.activeElement !== vp && vp.value !== voicePrompt()) { vp.value = voicePrompt(); fit(vp); }
  $("voice-prompt-reset").classList.toggle("on", settings.voice?.prompt !== undefined);

  const sent = systemFor(conv), note = $("system-note");
  note.hidden = false;
  note.textContent = sent ? `This chat (${toolsOn(conv) ? "tools on" : "no tools"}) is sent: “${sent}”. A change makes each chat read its history again once.`
    : `This chat (${toolsOn(conv) ? "tools on" : "no tools"}) is sent no system prompt.`;
  renderToolSettings();
  $("mode-note").textContent = shownMode() === "think" ? "with thinking" : "thinking off";
  $("order-reset").classList.toggle("on", !!settings[shownMode()].order);
  placeStages();
  for (const sync of knobs) sync();
  drawCharts();
}

document.addEventListener("visibilitychange", () => { if (!document.hidden) void refreshStatus(); });

$("new-chat").innerHTML = `${icon("plus")}New chat`;
$("export").innerHTML = icon("download");
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
buildControls();
buildTools();
void loadCatalog();
$("to-latest").innerHTML = icon("down");
$("settings-reset").insertAdjacentHTML("afterbegin", icon("retry"));
for (const id of ["system-reset", "order-reset", "voice-prompt-reset"]) $(id).innerHTML = icon("retry");
buildSettings();
for (const p of PANES) restorePane(p);
await refreshStatus();
for (const c of Object.values(chats)) {                // a reply being written may be in a version not on screen
  const m = [...walk(c.messages)].find((x) => x.role === "assistant" && x.job);
  if (m) void resume(c, m);
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
