import { createHash, randomBytes, randomInt } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { linkCode } from "./link";

type Kind = "browser" | "computer";
export interface Device { id: string; name: string; hash: string; added: number; seen: number; kind?: Kind }
interface Ask { code: string; name: string; at: number; allowed: boolean; kind: Kind; commit?: string; nonce?: string }

const ASK_MS = 10 * 60_000, MAX_ASKS = 8;
const NONCE_MS = 2 * 60_000, CHECK_MS = 15 * 60_000;
const COOKIE = "chrysalis-device", YEAR = 400 * 86400;
const sha = (s: string) => createHash("sha256").update(s).digest("hex");
const keyOf = (req: Request) => req.headers.get("cookie")?.split(/;\s*/).find((c) => c.startsWith(`${COOKIE}=`))?.slice(COOKIE.length + 1)
  ?? req.headers.get("authorization")?.match(/^Bearer\s+(\S+)$/i)?.[1] ?? req.headers.get("x-api-key") ?? undefined;
export const keyCookie = (key: string) => `${COOKIE}=${key}; Path=/; Max-Age=${YEAR}; Secure; HttpOnly; SameSite=Strict`;
const digits = (code: string) => code.replace(/\D/g, "");

export class Devices {
  private list: Device[];
  private asks = new Map<string, Ask>();
  private nonces = new Map<string, number>();
  private checked = new Map<string, number>();
  constructor(private file: string) {
    this.list = existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : [];
  }
  private save() { writeFileSync(this.file, JSON.stringify(this.list, null, 1), { mode: 0o600 }); }
  private prune() {
    const now = Date.now();
    for (const [s, a] of this.asks) if (now - a.at > ASK_MS) this.asks.delete(s);
    for (const [n, at] of this.nonces) if (now - at > NONCE_MS) this.nonces.delete(n);
  }

  check(req: Request, nonce: string | null) {
    this.prune();
    if (nonce && /^[\w-]{16,64}$/.test(nonce) && this.nonces.size < 256) this.nonces.set(nonce, Date.now());
    const found = this.find(req);
    if (found && found.device.kind !== "computer") this.checked.set(found.device.id, Date.now());
  }
  trusted(device: Device) { return Date.now() - (this.checked.get(device.id) ?? 0) < CHECK_MS; }
  stale(device: Device) { return Date.now() - (this.checked.get(device.id) ?? 0) > CHECK_MS / 2; }

  ask(nonce: string, name: string): { secret: string; code: string } | undefined {
    this.prune();
    if (!this.nonces.delete(nonce)) return undefined;
    return this.coded(name.slice(0, 40) || "A device", "browser");
  }
  computerAsk(name: string): { secret: string; code: string } {
    this.prune();
    return this.coded(name.replace(/[^\w .-]/g, "").slice(0, 40) || "A computer", "computer");
  }
  private coded(name: string, kind: Kind): { secret: string; code: string } {
    while (this.asks.size >= MAX_ASKS) this.asks.delete(this.asks.keys().next().value!);
    const taken = new Set([...this.asks.values()].map((a) => a.code));
    let code: string;
    do code = String(randomInt(1_000_000)).padStart(6, "0"); while (taken.has(code));
    const secret = randomBytes(24).toString("base64url");
    this.asks.set(secret, { code, name, at: Date.now(), allowed: false, kind });
    return { secret, code };
  }
  linkAsk(name: string, commit: string): { secret: string; nonce: string } | undefined {
    this.prune();
    if (!/^[0-9a-f]{64}$/.test(commit)) return undefined;
    while (this.asks.size >= MAX_ASKS) this.asks.delete(this.asks.keys().next().value!);
    const secret = randomBytes(24).toString("base64url"), nonce = randomBytes(16).toString("hex");
    this.asks.set(secret, { code: "", name: name.slice(0, 40) || "A computer", at: Date.now(), allowed: false, kind: "computer", commit, nonce });
    return { secret, nonce };
  }
  linkReveal(secret: string, nonce: string, fp: string): boolean {
    this.prune();
    const ask = this.asks.get(secret);
    if (!ask || ask.kind !== "computer" || ask.code || sha(nonce) !== ask.commit) return false;
    ask.code = linkCode(fp, ask.nonce!, nonce);
    return true;
  }
  waiting() {
    this.prune();
    return [...this.asks.values()].filter((a) => a.code).map(({ code, name, at, allowed, kind }) => ({ code, name, at, allowed, kind }));
  }
  private byCode(code: string) { this.prune(); return [...this.asks].find(([, a]) => a.code && a.code === digits(code)); }
  allow(code: string) { const ask = this.byCode(code)?.[1]; if (ask) ask.allowed = true; return ask?.name; }
  deny(code: string) { const hit = this.byCode(code); return hit ? this.asks.delete(hit[0]) : false; }
  collect(secret: string): { state: "waiting" } | { state: "allowed"; key: string } | undefined {
    this.prune();
    const ask = this.asks.get(secret);
    if (!ask) return undefined;
    if (!ask.allowed) return { state: "waiting" };
    this.asks.delete(secret);
    const { key, id } = this.add(ask.name, ask.kind);
    if (ask.kind === "browser") this.checked.set(id, Date.now());
    return { state: "allowed", key };
  }

  add(name: string, kind: Kind): { key: string; id: string } {
    const key = randomBytes(32).toString("base64url"), now = Date.now(), id = randomBytes(6).toString("hex");
    this.list.push({ id, name: name.slice(0, 40), hash: sha(key), added: now, seen: now, kind });
    this.save();
    return { key, id };
  }

  find(req: Request): { device: Device; key: string } | undefined {
    const key = keyOf(req), hash = key && sha(key), device = hash ? this.list.find((d) => d.hash === hash) : undefined;
    if (!device) return undefined;
    if (Date.now() - device.seen > 60_000) { device.seen = Date.now(); this.save(); }
    return { device, key: key! };
  }
  all() { return this.list.map(({ hash: _, ...d }) => ({ ...d, kind: d.kind ?? "browser" })); }
  remove(id: string) {
    const n = this.list.length;
    this.list = this.list.filter((d) => d.id !== id);
    this.checked.delete(id);
    if (this.list.length !== n) this.save();
    return this.list.length !== n;
  }
}
