import { useEffect, useReducer, useRef, useSyncExternalStore } from "react";
import { Box, render, Text, useAnimation, useApp, useInput, usePaste, useStdout } from "ink";
import * as core from "./cli-core";
import { butterfly, paint } from "./terminal";

const { state, ui } = core;
const pink = (s: string) => `\x1b[38;2;255;111;165m${s}\x1b[39m`;
const SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const MODEL_NAMES = ["Qwen3.8-27B", "Qwen3.8-Flash-Next"];

interface Row { label: string; options: string[]; current: number; set: (i: number) => unknown }
function rows(): Row[] {
  const e = state.engine;
  return [
    { label: "Reach", options: ["this computer", "network", "network, unencrypted"], current: core.MODES.indexOf(state.mode), set: (i) => core.setMode(core.MODES[i]) },
    ...(core.HAS_MODEL ? [
      { label: "Model", options: [...MODEL_NAMES.map((name, i) => (state.here.includes(core.MODELS[i]) ? name : state.download?.model === core.MODELS[i] ? `${name} · downloading` : `${name} · download`)), ...(state.none ? [] : [e.state === "stopped" ? "unloaded" : "unload"])],
        current: e.state === "stopped" ? 2 : state.none ? MODEL_NAMES.length + 1 : core.MODELS.indexOf(e.model),
        set: (i: number) => (i === 2 ? core.unload() : core.setModel(core.MODELS[i])) },
    ] : []),
    { label: "Shows", options: ["requests only", "everything"], current: state.verbose ? 1 : 0, set: (i) => core.setVerbose(i === 1) },
    ...(state.incognito ? [{ label: "Incognito", options: ["forget its chats"], current: -1, set: () => core.forget() }] : []),
    ...(state.opencode ? [{ label: "OpenCode", options: ["Chrysalis's models only", "outside models too"], current: state.opencode.outside ? 1 : 0, set: (i: number) => core.setOutside(i === 1) }] : []),
    { label: "Restart", options: ["model server", "web chat", "both"], current: -1, set: (i) => core.restart(["model", "web", ""][i]) },
    { label: "Print", options: ["addresses", "QR code", "OpenCode line"], current: -1, set: (i) => [core.links, core.qr, core.opencodeLine][i]() },
    { label: "Uninstall", options: ["remove Chrysalis, its models and its settings"], current: -1, set: () => core.submit("uninstall") },
  ];
}

const PINK = "#ff6fa5", ROSE = "#b04a72", SIZES = [[64, 23], [46, 17], [33, 12]] as const;
const wings = (below: number, rows = process.stdout.rows ?? 0, cols = process.stdout.columns ?? 0) => SIZES.find(([w, h]) => rows > h + below && cols >= w + 8);
function Butterfly({ size }: { size: readonly [number, number] }) {
  const { frame } = useAnimation({ interval: 110 });
  return (
    <Box flexDirection="column" marginBottom={1}>
      {butterfly(size[0], size[1], frame * 0.11).map((line, i) => (
        <Text key={i}>{"    "}{(line.match(/[@%]+|[^@%]+/g) ?? [" "]).map((run, j) => <Text key={j} color={/[@%]/.test(run) ? PINK : ROSE} bold={/[@%]/.test(run)}>{run}</Text>)}</Text>
      ))}
    </Box>
  );
}

function Status() {
  const e = state.engine, d = state.download, starting = e.state === "down" && core.HAS_MODEL && state.serving;
  const { frame } = useAnimation({ interval: 100, isActive: !!d || e.state === "loading" || starting || (e.state === "ready" && e.writing > 0) });
  const got = d?.total ? Math.min(1, d.done / d.total) : 0, bar = Math.round(got * 20), left = d?.rate ? Math.round((d.total - d.done) / d.rate) : 0;
  const spin = <Text color="magenta">{SPIN[frame % SPIN.length]}</Text>, filled = Math.round(Math.min(1, Math.max(0, e.progress)) * 20);
  return (
    <Text wrap="truncate-end">
      {"  "}
      {d && e.state !== "ready" ? <>{spin} <Text bold>{d.model}</Text>{"  "}<Text color="magenta">{"━".repeat(bar)}</Text><Text color="gray">{"━".repeat(20 - bar)}</Text>
        {` ${String(Math.round(got * 100)).padStart(3)}%  `}<Text dimColor>downloading · {(d.done / 1e9).toFixed(1)} of {(d.total / 1e9).toFixed(1)} GB{d.rate > 0 ? ` · ${Math.round(d.rate / 1e6)} MB/s · ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")} left` : ""}</Text></>
        : e.state === "loading" ? <>{spin} <Text bold>{e.model || "model"}</Text>{"  "}<Text color="magenta">{"━".repeat(filled)}</Text><Text color="gray">{"━".repeat(20 - filled)}</Text>
        {` ${String(Math.round(e.progress * 100)).padStart(3)}%  `}<Text dimColor>{e.step} · {Math.round((Date.now() - e.since) / 1000)} s</Text></>
        : e.state === "ready" ? <><Text color="green">●</Text> <Text bold>{e.model}</Text>{e.writing ? <><Text dimColor> · </Text>{spin} {e.writing} writing</> : null}
          {d ? <><Text dimColor> · </Text>{d.model} <Text color="magenta">{Math.round(got * 100)}%</Text><Text dimColor> downloaded</Text></> : null}</>
        : e.state === "stopped" ? <><Text color="gray">○</Text> {e.model} <Text dimColor>is unloaded</Text></>
        : !core.HAS_MODEL ? <><Text color="gray">○</Text> <Text dimColor>no model on this computer</Text></>
        : state.none ? <><Text color="gray">○</Text> <Text dimColor>no model yet · the Model row downloads one</Text></>
        : starting ? <>{spin} <Text dimColor>the model server is starting</Text></>
        : <><Text color="red">○</Text> <Text dimColor>the model server is not running · restart starts it</Text></>}
      <Text dimColor> · </Text>
      {state.mode === "local" ? <Text dimColor>this computer only</Text> : state.mode === "network" ? <>network<Text dimColor> · encrypted</Text></> : <Text color="yellow">network · unencrypted</Text>}
      {state.verbose ? <><Text dimColor> · </Text>shows everything</> : null}
      {state.incognito ? <><Text dimColor> · </Text>incognito chats open</> : null}
    </Text>
  );
}

function App() {
  useSyncExternalStore(core.subscribe, core.snapshot);
  const { write } = useStdout(), { suspendTerminal } = useApp();
  const [, redraw] = useReducer((n: number) => n + 1, 0);
  const ed = useRef({ chars: [] as string[], cursor: 0, history: [] as string[], at: -1, draft: [] as string[], row: -1, picks: {} as Record<string, number> }).current;
  useEffect(() => {
    ui.log = (text) => write(`${text}\x1b[0m\n`);
    ui.clear = () => write("\x1b[2J\x1b[3J\x1b[H");
    ui.suspend = (run) => suspendTerminal(async () => { await run(); });
    for (const text of waiting.splice(0)) ui.log(text);
  }, []);

  const all = rows(), set = (chars: string[]) => { ed.chars = chars; ed.cursor = chars.length; };
  const insert = (text: string) => {
    const add = [...text.replace(/\s+/g, " ")].filter((ch) => ch.codePointAt(0)! >= 0x20 && ch !== "\x7f");
    ed.chars.splice(ed.cursor, 0, ...add);
    ed.cursor += add.length;
  };
  const recall = (step: number) => {
    const to = ed.at + step;
    if (to < -1 || to >= ed.history.length) return;
    if (ed.at === -1) ed.draft = ed.chars;
    ed.at = to;
    set(to === -1 ? ed.draft : [...ed.history[ed.history.length - 1 - to]]);
  };
  const complete = () => {
    const typed = ed.chars.join(""), fits = core.WORDS.filter((w) => w.startsWith(typed));
    if (typed.includes(" ") || !fits.length) return;
    let shared = fits[0];
    for (const w of fits) while (!w.startsWith(shared)) shared = shared.slice(0, -1);
    set([...shared]);
    if (fits.length > 1) ui.log(paint.dim(`    ${fits.join("  ")}`));
  };
  const enter = () => {
    const line = ed.chars.join("").trim();
    set([]);
    ed.at = -1;
    if (!line) return;
    if (ed.history.at(-1) !== line) ed.history.push(line);
    ui.log(paint.dim(`  › ${line}`));
    core.submit(line);
  };
  const tab = () => { if (ed.chars.length) complete(); };
  if (ed.row < 0) ed.row = Math.max(0, all.findIndex((r) => r.label === "Model"));
  const pickOf = (r: Row) => ed.picks[r.label] ?? (r.current >= 0 && r.current < r.options.length ? r.current : 0);

  usePaste((text) => { insert(text); redraw(); });
  useInput((input, key) => {
    const c = ed.chars, at = ed.cursor;
    const row = all[Math.min(ed.row, all.length - 1)], n = row.options.length, pick = pickOf(row);
    if (key.ctrl && input === "c") {
      if (c.length) set([]);
      else core.quit();
    } else if (key.upArrow) ed.row = (Math.min(ed.row, all.length - 1) + all.length - 1) % all.length;
    else if (key.downArrow) ed.row = (Math.min(ed.row, all.length - 1) + 1) % all.length;
    else if (!c.length && key.leftArrow) ed.picks[row.label] = (pick + n - 1) % n;
    else if (!c.length && key.rightArrow) ed.picks[row.label] = (pick + 1) % n;
    else if (!c.length && key.return) {
      if (pick !== row.current) core.act(() => row.set(pick));
      delete ed.picks[row.label];
    } else if (key.escape) { ed.picks = {}; set([]); }
    else if (key.return) enter();
    else if (key.tab) tab();
    else if (key.backspace) { if (at) { c.splice(at - 1, 1); ed.cursor--; } }
    else if (key.delete || (key.ctrl && input === "d")) { if (!c.length && key.ctrl) core.quit(); else c.splice(at, 1); }
    else if (key.leftArrow || (key.ctrl && input === "b")) ed.cursor = Math.max(0, at - 1);
    else if (key.rightArrow || (key.ctrl && input === "f")) ed.cursor = Math.min(c.length, at + 1);
    else if (key.home || (key.ctrl && input === "a")) ed.cursor = 0;
    else if (key.end || (key.ctrl && input === "e")) ed.cursor = c.length;
    else if (key.ctrl && input === "p") recall(1);
    else if (key.ctrl && input === "n") recall(-1);
    else if (key.ctrl && input === "u") { c.splice(0, at); ed.cursor = 0; }
    else if (key.ctrl && input === "k") c.splice(at);
    else if (key.ctrl && input === "w") {
      let to = at;
      while (to && c[to - 1] === " ") to--;
      while (to && c[to - 1] !== " ") to--;
      c.splice(to, at - to);
      ed.cursor = to;
    } else if (key.ctrl && input === "l") ui.clear();
    else if (!key.ctrl && !key.meta && !key.escape && input)
      for (const part of input.split(/([\r\n\t])/)) part === "\t" ? tab() : part === "\r" || part === "\n" ? enter() : insert(part);
    redraw();
  });

  const asking = [...state.asks, ...state.questions], row = Math.min(ed.row, all.length - 1);
  const waking = core.HAS_MODEL && !state.woke, size = wings(all.length + asking.length + 5);
  useEffect(() => {
    if (state.woke && size) ui.log(`\n${butterfly(size[0], size[1], -9).map((line) => `    ${pink(line)}`).join("\n")}\n`);
  }, [state.woke]);
  return (
    <Box flexDirection="column">
      {asking.map((q, i) => <Text key={q.key}>{"  "}<Text color="yellow">?</Text> {q.text}{i ? null : <Text dimColor>{"  y or n"}</Text>}</Text>)}
      {waking && size ? <Butterfly size={size} /> : null}
      <Box flexDirection="column" marginBottom={1}>
        {all.map((r, i) => (
          <Text key={r.label} wrap="truncate-end">
            {i === row ? <Text color="magenta">{"❯ "}</Text> : "  "}<Text bold={i === row} dimColor={i !== row}>{r.label.padEnd(10)}</Text>
            {r.options.map((o, j) => {
              const on = j === r.current, at = i === row && j === pickOf(r);
              return <Text key={o}><Text inverse={at} color={on ? "green" : undefined} dimColor={!on && !at}>{` ${on ? "● " : r.current < 0 ? "" : "○ "}${o} `}</Text>{" "}</Text>;
            })}
          </Text>
        ))}
        <Text dimColor wrap="truncate-end">{"  ↑↓ row · ←→ choice · Enter sets it"}{state.web ? `  ·  chat ${state.web.local}` : ""}</Text>
      </Box>
      <Status />
      <Text>
        {"  "}<Text color="magenta">›</Text>{" "}
        {ed.chars.slice(0, ed.cursor).join("")}<Text inverse>{ed.chars[ed.cursor] ?? " "}</Text>{ed.chars.slice(ed.cursor + 1).join("")}
        {ed.chars.length ? null : <Text dimColor>or type a command · help lists them</Text>}
      </Text>
    </Box>
  );
}

const waiting: string[] = [];
export async function mount() {
  ui.live = true;
  ui.log = (text) => void waiting.push(text);
  let app = render(<App />, { exitOnCtrlC: false }), settle: Timer | undefined;
  process.stdout.on("resize", () => {
    clearTimeout(settle);
    settle = setTimeout(() => {
      app.unmount();
      process.stdout.write("\x1b[2J\x1b[3J\x1b[H");
      ui.log = (text) => void waiting.push(text);
      core.header();
      app = render(<App />, { exitOnCtrlC: false });
    }, 150);
  });
  ui.close = () => { clearTimeout(settle); app.clear(); app.unmount(); };
  await app.waitUntilRenderFlush?.();
}
