import { appendFileSync, mkdirSync, readFileSync } from "node:fs";
import { homedir, networkInterfaces } from "node:os";
import { dirname, join } from "node:path";
import { routeAddress } from "./tls";

const NOTE = join(homedir(), ".config", "chrysalis", "ufw");
const read = (path: string) => { try { return readFileSync(path, "utf8"); } catch { return ""; } };
const ran = (cmd: string[], stdio: "ignore" | "inherit" = "ignore") => {
  if (!Bun.which(cmd[0])) return false;
  try { return Bun.spawnSync(cmd, { stdin: stdio, stdout: stdio, stderr: stdio }).exitCode === 0; } catch { return false; }
};

export function subnetOf(ip: string | undefined, nets = networkInterfaces()): string | undefined {
  const at = Object.values(nets).flat().find((a) => a && a.family === "IPv4" && a.address === ip);
  if (!at) return undefined;
  const bits = (s: string) => s.split(".").reduce((n, p) => (n << 8) + Number(p), 0) >>> 0;
  const mask = bits(at.netmask), net = (bits(at.address) & mask) >>> 0, len = mask.toString(2).replace(/0+$/, "").length;
  return `${[24, 16, 8, 0].map((s) => (net >>> s) & 255).join(".")}/${len}`;
}

async function ufw(mode: "network" | "insecure") {
  if (!ran(["systemctl", "is-active", "-q", "ufw"])) return;
  const net = subnetOf(await routeAddress()), from = mode === "insecure" ? 8080 : 8081;
  if (!net) return;
  const ports = mode === "insecure" ? "8080" : "808[01]", rules = read("/etc/ufw/user.rules");
  const open = new RegExp(`allow tcp ${ports}:8083 0\\.0\\.0\\.0/0 any ${net.replaceAll(".", "\\.")} in`).test(rules) && rules.includes(`allow udp 5353 0.0.0.0/0 any ${net} in`);
  if (open || new RegExp(`^${net.replaceAll(".", "\\.")} ${ports}:8083 5353$`, "m").test(read(NOTE))) return;
  console.log(`ufw keeps other devices out: letting ${net} reach Chrysalis (TCP ${from}-8083, mDNS) takes your password, once`);
  const allow = `ufw allow from ${net} to any port ${from}:8083 proto tcp comment 'Chrysalis Inference' && ` +
    `ufw allow from ${net} to any port 5353 proto udp comment 'Chrysalis Inference: found on the network'`;
  if (process.stdin.isTTY && ran(["sudo", "sh", "-c", `${allow} >/dev/null`], "inherit")) {
    mkdirSync(dirname(NOTE), { recursive: true });
    appendFileSync(NOTE, `${net} ${from}:8083 5353\n`);
  } else console.log("  not changed: Chrysalis started in a terminal asks for the password");
}

export async function letIn(mode: "network" | "insecure") {
  if (process.platform !== "linux") return;
  await ufw(mode);
}
