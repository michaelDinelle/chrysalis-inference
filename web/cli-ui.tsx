import { useEffect, useReducer, useRef, useSyncExternalStore } from "react";
import { Box, render, Text, useAnimation, useApp, useInput, usePaste, useStdout } from "ink";
import * as core from "./cli-core";
import { paint } from "./terminal";

const { state, ui } = core;
const SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const MODEL_NAMES = ["Qwen3.8-27B", "Qwen3.8-Flash-Next"];

interface Row { label: string; options: string[]; current: number; set: (i: number) => unknown }
function rows(): Row[] {
  const e = state.engine;
  return [
    { label: "Reach", options: ["this computer", "network", "network, unencrypted"], current: core.MODES.indexOf(state.mode), set: (i) => core.setMode(core.MODES[i]) },
    ...(core.HAS_MODEL ? [
      { label: "Model", options: [...MODEL_NAMES, e.state === "stopped" ? "unloaded" : "unload"], current: e.state === "stopped" ? 2 : core.MODELS.indexOf(e.model),
        set: (i: number) => (i === 2 ? core.unload() : core.setModel(core.MODELS[i])) },
    ] : []),
    { label: "Shows", options: ["requests only", "everything"], current: state.verbose ? 1 : 0, set: (i) => core.setVerbose(i === 1) },
    ...(state.incognito ? [{ label: "Incognito", options: ["forget its chats"], current: -1, set: () => core.forget() }] : []),
    ...(state.opencode ? [{ label: "OpenCode", options: ["Chrysalis's models only", "outside models too"], current: state.opencode.outside ? 1 : 0, set: (i: number) => core.setOutside(i === 1) }] : []),
    { label: "Restart", options: ["model server", "web chat", "both"], current: -1, set: (i) => core.restart(["model", "web", ""][i]) },
    { label: "Print", options: ["addresses", "QR code", "OpenCode line"], current: -1, set: (i) => [core.links, core.qr, core.opencodeLine][i]() },
  ];
}

function Status() {
  const e = state.engine, starting = e.state === "down" && core.HAS_MODEL && state.serving;
  const { frame } = useAnimation({ interval: 100, isActive: e.state === "loading" || starting || (e.state === "ready" && e.writing > 0) });
  const spin = <Text color="magenta">{SPIN[frame % SPIN.length]}</Text>, filled = Math.round(Math.min(1, Math.max(0, e.progress)) * 20);
  return (
    <Text wrap="truncate-end">
      {"  "}
      {e.state === "loading" ? <>{spin} <Text bold>{e.model || "model"}</Text>{"  "}<Text color="magenta">{"━".repeat(filled)}</Text><Text color="gray">{"━".repeat(20 - filled)}</Text>
        {` ${String(Math.round(e.progress * 100)).padStart(3)}%  `}<Text dimColor>{e.step} · {Math.round((Date.now() - e.since) / 1000)} s</Text></>
        : e.state === "ready" ? <><Text color="green">●</Text> <Text bold>{e.model}</Text>{e.writing ? <><Text dimColor> · </Text>{spin} {e.writing} writing</> : null}</>
        : e.state === "stopped" ? <><Text color="gray">○</Text> {e.model} <Text dimColor>is unloaded</Text></>
        : !core.HAS_MODEL ? <><Text color="gray">○</Text> <Text dimColor>no model on this computer</Text></>
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
  const ed = useRef({ chars: [] as string[], cursor: 0, history: [] as string[], at: -1, draft: [] as string[], panel: false, row: 0, picks: [] as number[] }).current;
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
  const openPanel = () => {
    ed.panel = true;
    ed.row = 0;
    ed.picks = all.map((r) => Math.max(0, r.current));
    void core.readOpencode();
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
  const tab = () => (ed.chars.length ? complete() : openPanel());

  usePaste((text) => { if (!ed.panel) { insert(text); redraw(); } });
  useInput((input, key) => {
    const c = ed.chars, at = ed.cursor;
    if (key.ctrl && input === "c") {
      if (ed.panel) ed.panel = false;
      else if (c.length) set([]);
      else core.quit();
    } else if (ed.panel) {
      const row = all[Math.min(ed.row, all.length - 1)], n = row.options.length;
      if (key.escape || key.tab || input === "q") ed.panel = false;
      else if (key.upArrow) ed.row = (ed.row + all.length - 1) % all.length;
      else if (key.downArrow) ed.row = (ed.row + 1) % all.length;
      else if (key.leftArrow) ed.picks[ed.row] = ((ed.picks[ed.row] ?? 0) + n - 1) % n;
      else if (key.rightArrow) ed.picks[ed.row] = ((ed.picks[ed.row] ?? 0) + 1) % n;
      else if (key.return || input === " ") { const pick = ed.picks[ed.row] ?? 0; if (pick !== row.current) core.act(() => row.set(pick)); }
    } else if (key.return) enter();
    else if (key.tab) tab();
    else if (key.backspace) { if (at) { c.splice(at - 1, 1); ed.cursor--; } }
    else if (key.delete || (key.ctrl && input === "d")) { if (!c.length && key.ctrl) core.quit(); else c.splice(at, 1); }
    else if (key.leftArrow || (key.ctrl && input === "b")) ed.cursor = Math.max(0, at - 1);
    else if (key.rightArrow || (key.ctrl && input === "f")) ed.cursor = Math.min(c.length, at + 1);
    else if (key.home || (key.ctrl && input === "a")) ed.cursor = 0;
    else if (key.end || (key.ctrl && input === "e")) ed.cursor = c.length;
    else if (key.upArrow || (key.ctrl && input === "p")) recall(1);
    else if (key.downArrow || (key.ctrl && input === "n")) recall(-1);
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
  return (
    <Box flexDirection="column" marginTop={1}>
      {asking.map((q, i) => <Text key={q.key}>{"  "}<Text color="yellow">?</Text> {q.text}{i ? null : <Text dimColor>{"  y or n"}</Text>}</Text>)}
      {ed.panel ? (
        <Box flexDirection="column" marginBottom={1}>
          {all.map((r, i) => (
            <Text key={r.label} wrap="truncate-end">
              {"  "}<Text bold={i === row} dimColor={i !== row}>{r.label.padEnd(10)}</Text>
              {r.options.map((o, j) => {
                const on = j === r.current, at = i === row && j === (ed.picks[i] ?? 0);
                return <Text key={o}><Text inverse={at} color={on ? "green" : undefined} dimColor={!on && !at}>{` ${on ? "● " : r.current < 0 ? "" : "○ "}${o} `}</Text>{" "}</Text>;
              })}
            </Text>
          ))}
          <Text dimColor>{"  ↑↓ row · ←→ choice · Enter sets it · Esc closes"}</Text>
        </Box>
      ) : null}
      <Status />
      <Text>
        {"  "}<Text color="magenta">›</Text>{" "}
        {ed.panel ? <Text dimColor>the switches are open</Text> : <>
          {ed.chars.slice(0, ed.cursor).join("")}<Text inverse>{ed.chars[ed.cursor] ?? " "}</Text>{ed.chars.slice(ed.cursor + 1).join("")}
          {ed.chars.length ? null : <Text dimColor>Tab opens the switches · help lists the commands</Text>}
        </>}
      </Text>
    </Box>
  );
}

const waiting: string[] = [];
export async function mount() {
  ui.live = true;
  ui.log = (text) => void waiting.push(text);
  const app = render(<App />, { exitOnCtrlC: false });
  ui.close = () => { app.clear(); app.unmount(); };
  await app.waitUntilRenderFlush?.();
}
