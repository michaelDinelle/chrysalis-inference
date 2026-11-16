import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { letIn } from "./firewall";
import { gpuEnv, gpuFound } from "./gpu";
import { getCommand, here, HOME, MODELS, ourScript, python, ROOT, serverCommand, serverEnv, utf8Console, type Machine } from "./platform";
import { tlsPath } from "./tls";

const ROOMY_GB = 58, NAME = process.platform === "win32" ? "Chrysalis.exe" : "chrysalis.sh";
const USAGE = `usage: ${NAME} [--host 0.0.0.0 | --insecure] [--model <model>]

  Runs Chrysalis in this terminal: the model and the web chat, each request as it finishes, and the questions when a
  phone or computer wants in. Type help there for what it takes: local, network and insecure move it without stopping,
  verbose shows what the servers print. Ctrl+C stops it. Nothing it shows is written to disk.
  Requests pick the model (qwen3.8-27b or qwen3.8-flash-next); Chrysalis switches between them.
  --host 0.0.0.0: on the local network too, encrypted, for the phones and computers let in: a QR code connects a
  phone, and Chrysalis on another computer finds this one and uses its GPU
  --insecure: on the local network over plain HTTP, as llama.cpp serves: no certificates and nobody to let in, the
  chat on port 8081 and the API on 8080. Anyone on the network can use it and read what is sent, and other devices
  get no voice mode and no OpenCode
  --model qwen3.8-27b or qwen3.8-flash-next: start with that model; without it, the one last used, or the 27B

  ${NAME} get <model>   download a model's files now (Chrysalis gets the one it loads by itself)
  ${NAME} uninstall     remove Chrysalis from this computer: its models, its caches and its settings too
  ${NAME} opencode      say whether OpenCode here is kept to Chrysalis's models (it is, until you change it)
  ${NAME} opencode outside   allow models outside this computer, after a warning; takes effect at once
  ${NAME} opencode only      keep OpenCode here to Chrysalis's models again`;

export type Mode = "local" | "network" | "insecure";
export function parse(args: string[]): { mode: Mode; model?: string } | undefined {
  let mode: Mode = "local", model: string | undefined;
  for (let i = 0; i < args.length; ) {
    const [a, b] = [args[i], args[i + 1]];
    if (a === "--host" && b === "0.0.0.0") { mode = "network"; i += 2; }
    else if (a === "--host" && b === "127.0.0.1") i += 2;
    else if (a === "--insecure") { mode = "insecure"; i++; }
    else if (a === "--model" && MODELS.includes(b)) { model = b; i += 2; }
    else return undefined;
  }
  return { mode, model };
}

export interface Facts { gpu: boolean; roomy: boolean; last?: string; first?: string; missing: (model: string) => boolean }
export interface Plan { model?: string; background?: string }
export function plan(f: Facts): Plan {
  if (!f.gpu) return {};
  const fits = f.roomy ? MODELS : MODELS.slice(0, 1), model = f.first ?? (f.last && fits.includes(f.last) ? f.last : fits[0]);
  return { model, ...(f.roomy && model !== MODELS[1] && f.missing(MODELS[1]) ? { background: MODELS[1] } : {}) };
}

const read = (path: string) => { try { return readFileSync(path, "utf8").trim(); } catch { return ""; } };
const running = (port: string) => fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(1000) }).then((r) => r.ok, () => false);
const wait = async (cmd: string[], o: { cwd?: string; env?: Record<string, string | undefined>; quiet?: boolean } = {}) => {
  try {
    const p = Bun.spawn(cmd, { cwd: o.cwd ?? ROOT, env: o.env as Record<string, string>, stdin: "inherit", stdout: o.quiet ? "ignore" : "inherit", stderr: o.quiet ? "ignore" : "inherit" });
    return await p.exited;
  } catch (e) {
    if (!o.quiet) console.error(`chrysalis: ${cmd[0]} could not be started: ${e instanceof Error ? e.message : String(e)}`);
    return 127;
  }
};

type Found = () => { here: boolean; env: Record<string, string>; why?: string };
async function start(args: string[], m: Machine, gpus: Found) {
  const asked = parse(args);
  if (!asked) return console.log(USAGE), 1;
  const port = process.env.PORT ?? "8081";
  if (await running(port)) return console.log("Chrysalis is running already, in another terminal (Ctrl+C there stops it)"), 1;
  if (asked.mode !== "local") await letIn(asked.mode);
  const tls = tlsPath(), last = read(`${tls}/model`) || undefined, env = { ...process.env, ...serverEnv(m) };
  const missing = (model: string) => { try { return Bun.spawnSync([...getCommand(model, m), "--check"], { cwd: ROOT, env, stdout: "ignore", stderr: "ignore" }).exitCode !== 0; } catch { return true; } };
  const found = gpus(), gpu = found.here;
  const p = plan({ gpu, roomy: Math.floor(m.memory / 2 ** 30) >= ROOMY_GB, last, first: asked.model, missing });
  const run = [...ourScript(existsSync(join(import.meta.dir, "cli.js")) ? join(import.meta.dir, "cli.js") : join(import.meta.dir, "cli.ts")).cmd, "--mode", asked.mode];
  if (!existsSync(join(import.meta.dir, "cli.js")))
    await wait([process.execPath, "install", "--frozen-lockfile"], { cwd: import.meta.dir, env: { ...process.env, BUN_BE_BUN: "1" }, quiet: true });
  if (p.model) {
    mkdirSync(tls, { recursive: true });
    writeFileSync(`${tls}/model`, p.model);
    if (p.background) run.push("--get", p.background);
  } else run.push("--no-model", ...(found.why ? [found.why] : []));
  const cli = Bun.spawn(run, { cwd: import.meta.dir, env: { ...process.env, ...found.env, BUN_BE_BUN: "1" }, stdio: ["inherit", "inherit", "inherit"] });
  const opener = m.platform === "win32" ? ["cmd", "/c", "start", ""] : HOME !== ROOT ? ["open"] : undefined;
  if (opener && asked.mode !== "insecure") void (async () => {
    while (cli.exitCode === null && !(await running(port))) await Bun.sleep(250);
    if (cli.exitCode === null) Bun.spawn([...opener, `http://localhost:${port}`], { stdin: "ignore", stdout: "ignore", stderr: "ignore" });
  })();
  for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"] as const) process.on(sig, () => cli.kill(sig));
  return cli.exited;
}

async function opencode(to: string | undefined) {
  const web = `http://127.0.0.1:${process.env.PORT ?? "8081"}`, flag = `${tlsPath()}/opencode-outside`, up = await running(process.env.PORT ?? "8081");
  const call = (method: string) => fetch(`${web}/api/threads/outside`, { method, headers: { "Content-Type": "application/json" }, body: "{}" }).then((r) => r.ok, () => false);
  if (!to) {
    const outside = up ? (await fetch(`${web}/api/threads`).then((r) => r.json(), () => ({}))).outside === true : existsSync(flag);
    console.log(outside ? `Models outside this computer are allowed: a thread on one sends that company your prompts, your files and what its tools read.\n${NAME} opencode only keeps OpenCode here to Chrysalis's models again.`
      : `Chrysalis only: OpenCode here runs on this computer's own models, and nothing is sent anywhere else.\n${NAME} opencode outside allows the others.`);
    return 0;
  }
  if (to === "only") {
    if (up ? !(await call("DELETE")) : (rmSync(flag, { force: true }), false)) return console.error("Chrysalis did not take that"), 1;
    console.log("Chrysalis only: OpenCode here is kept to this computer's own models.");
    return 0;
  }
  if (to !== "outside") return console.log(USAGE), 1;
  if (!process.stdin.isTTY) return console.error("run this in a terminal: it asks before it allows models outside this computer"), 1;
  console.log("Models outside this computer\nA thread on one of these models sends that company your prompts, your files and whatever its tools read\non this computer. Chrysalis cannot take any of it back. Threads on Chrysalis's own models stay here as before.");
  if (!/^(y|yes)$/i.test(prompt("Allow them? [y/N]")?.trim() ?? "")) return console.log("Kept to Chrysalis only."), 0;
  if (up ? !(await call("POST")) : (mkdirSync(dirname(flag), { recursive: true }), writeFileSync(flag, ""), false)) return console.error("Chrysalis did not take that"), 1;
  console.log(`Allowed. ${NAME} opencode only takes it back.`);
  return 0;
}

export function uninstall(root = HOME, settings = dirname(tlsPath()), setup = join(dirname(process.execPath), "unins000.exe")): string[] {
  const gone = [join(root, "data", "models"), join(root, "data", "cache"), settings].filter((dir) => existsSync(dir));
  for (const dir of gone) rmSync(dir, { recursive: true, force: true });
  if (existsSync(setup)) Bun.spawn([setup, "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], { stdin: "ignore", stdout: "ignore", stderr: "ignore", detached: true } as never).unref();
  return gone;
}

async function remove() {
  if (await running(process.env.PORT ?? "8081")) return console.error("Chrysalis is running: stop it first (Ctrl+C in its terminal, or quit there)"), 1;
  if (!process.stdin.isTTY) return console.error("run this in a terminal: it asks before it removes anything"), 1;
  console.log("Uninstall Chrysalis\nThis removes the models it downloaded, its caches and its settings from this computer. Nothing of it is kept.");
  if (!/^(y|yes)$/i.test(prompt("Remove them? [y/N]")?.trim() ?? "")) return console.log("Nothing was removed."), 0;
  const setup = existsSync(join(dirname(process.execPath), "unins000.exe"));
  for (const dir of uninstall()) console.log(`  removed ${dir}`);
  console.log(setup ? "Windows is removing Chrysalis itself now." : `What is left is the program's own folder, ${ROOT}: delete it and Chrysalis is gone.`);
  return 0;
}

export async function main(args: string[], m: Machine = here(), gpus: Found = () => gpuFound(m)): Promise<number> {
  await utf8Console(m);
  const env = { ...process.env, ...serverEnv(m) };
  switch (args[0]) {
    case "serve": {
      const model = read(`${tlsPath()}/model`) || undefined;
      mkdirSync(join(ROOT, "data", "cache"), { recursive: true });
      return wait([...serverCommand(MODELS.includes(model ?? "") ? model : undefined, m), ...args.slice(1)], { env: { ...env, ...gpuEnv(env) } });
    }
    case "get": return wait(args.length > 1 ? [...getCommand(args[1], m), ...args.slice(2)] : [python(m), "-m", "chrysalis", "get"], { env });
    case "gpu": {
      const found = gpus();
      if (found.why) console.error(`chrysalis: ${found.why}`);
      return found.here ? 0 : 1;
    }
    case "opencode": return opencode(args[1]);
    case "uninstall": return remove();
    case "stats": return console.log(await fetch(`${process.env.CHRYSALIS_URL ?? "http://127.0.0.1:8080"}/chrysalis/stats`).then((r) => r.text(), () => "")), 0;
    case "help": case "-h": case "--help": return console.log(USAGE), 0;
  }
  return start(args, m, gpus);
}

if (import.meta.main) {
  const beside = dirname(process.execPath);
  if (!existsSync(join(ROOT, "web", "launch.ts")) && existsSync(join(beside, "web", "launch.ts"))) {
    const p = Bun.spawn([process.execPath, join(beside, "web", "launch.ts"), ...process.argv.slice(2)], { stdio: ["inherit", "inherit", "inherit"], env: { ...process.env, BUN_BE_BUN: "1" } });
    for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"] as const) process.on(sig, () => p.kill(sig));
    process.exit(await p.exited);
  }
  process.exit(await main(process.argv.slice(2)));
}
