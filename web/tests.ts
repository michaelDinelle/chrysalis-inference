import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { availableParallelism, freemem } from "node:os";
import puppeteer from "puppeteer-core";
import { BROWSER_ARGS, CHAT_ARGS } from "./browser-harness";

const KEEP_MB = 2048, WORKER_MB = 400;
const browser = await puppeteer.launch({ executablePath: process.env.CHROMIUM ?? "/usr/bin/chromium", headless: true, args: [...CHAT_ARGS, ...BROWSER_ARGS] });
const free = process.platform === "linux" ? Number(/MemAvailable:\s+(\d+)/.exec(readFileSync("/proc/meminfo", "utf8"))![1]) >> 10 : freemem() >> 20;
const workers = Math.max(1, Math.min(availableParallelism(), Math.floor((free - KEEP_MB) / WORKER_MB)));
const TIMINGS = `${import.meta.dir}/../build/web-timings.json`;
mkdirSync(`${import.meta.dir}/../build`, { recursive: true });
if (!existsSync(TIMINGS)) writeFileSync(TIMINGS, JSON.stringify({ version: 1, files: {} }));
const run = Bun.spawn([process.execPath, "test", `--parallel=${workers}`, `--timings=${TIMINGS}`, "--update-timings", ...process.argv.slice(2)], {
  cwd: import.meta.dir, stdio: ["inherit", "inherit", "inherit"], env: { ...process.env, CHROMIUM_WS: browser.wsEndpoint() },
});
const code = await run.exited;
await browser.close();
process.exit(code);
