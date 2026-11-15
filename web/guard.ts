import { stopTree } from "./platform";

const { BUN_BE_BUN: _, ...env } = process.env;
const child = Bun.spawn(process.argv.slice(2), { stdin: "ignore", stdout: "inherit", stderr: "inherit", env });
let over = false;
const end = () => { if (!over) { over = true; stopTree(child); } };
for (const sig of ["SIGTERM", "SIGINT", "SIGHUP"] as const) process.on(sig, end);
void Bun.stdin.stream().pipeTo(new WritableStream()).finally(end);
process.exit(await child.exited);
