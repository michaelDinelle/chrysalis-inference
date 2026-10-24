// Who may use the web chat and Chrysalis's API from the network (`chrysalis.sh start --host 0.0.0.0`): the devices this
// computer's own screen let in. A device asks to join and shows a code; Connect a device here (or `chrysalis.sh allow
// <code>`) shows the same code beside Allow. Allowed, the device gets a key of its own: a browser keeps it as a cookie
// that only the chat's HTTPS gets, and a computer linked to this one (link.ts) sends it over the link with every request.
// This machine needs no key: what runs here can reach Chrysalis anyway.
//
// A browser has to show it trusts Chrysalis's authority too, to ask and every few minutes after. No server can see whether
// a browser checked the certificate it was shown, and a page loads all the same past a warning clicked through. So the
// trust check (server.ts) listens on a port of its own with a certificate made anew at every start: a browser reaches it
// only where it trusts the authority itself, since a warning clicked past covers the one certificate it was shown (Chromium
// and Firefox, browser.test.ts), and an image cannot show a warning. An app checks certificates itself: every HTTP client
// does unless told not to, which no server can tell either.
//
// A device key is 256 bits and is kept here as its SHA-256 only, so the file lets nobody in, and comparing hashes tells a
// timing attacker nothing about a key.
import { createHash, randomBytes, randomInt } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { linkCode } from "./link";

// a browser's, or a linked computer's, which pins this computer's link authority itself (older entries have no kind:
// browsers')
type Kind = "browser" | "computer";
export interface Device { id: string; name: string; hash: string; added: number; seen: number; kind?: Kind }
// A computer's ask (link.ts) has its code only once it has shown the nonce it committed to: the desktop's nonce is
// fixed before the laptop's is known, so no one in between can pick an authority whose code comes out the same.
interface Ask { code: string; name: string; at: number; allowed: boolean; kind: Kind; commit?: string; nonce?: string }

const ASK_MS = 10 * 60_000, MAX_ASKS = 8;                 // an ask lasts 10 minutes; a few at once, the newest kept
const NONCE_MS = 2 * 60_000, CHECK_MS = 15 * 60_000;      // a pairing page's check, then a browser's, stay good this long
const COOKIE = "chrysalis-device", YEAR = 400 * 86400;    // 400 days: the longest browsers keep a cookie
const sha = (s: string) => createHash("sha256").update(s).digest("hex");
// a browser's cookie, or an app's key as OpenAI's SDKs send it (Authorization: Bearer) or Anthropic's (x-api-key)
const keyOf = (req: Request) => req.headers.get("cookie")?.split(/;\s*/).find((c) => c.startsWith(`${COOKIE}=`))?.slice(COOKIE.length + 1)
  ?? req.headers.get("authorization")?.match(/^Bearer\s+(\S+)$/i)?.[1] ?? req.headers.get("x-api-key") ?? undefined;
// Secure: never over plain HTTP; HttpOnly: no script reads it; Strict: no other site's page sends it
export const keyCookie = (key: string) => `${COOKIE}=${key}; Path=/; Max-Age=${YEAR}; Secure; HttpOnly; SameSite=Strict`;
// what an ask's code is matched by: its digits ("482 913", "482913")
const digits = (code: string) => code.replace(/\D/g, "");

export class Devices {
  private list: Device[];
  private asks = new Map<string, Ask>();                  // by the asking page's secret
  private nonces = new Map<string, number>();             // a pairing page's checks, when each came through
  private checked = new Map<string, number>();            // a browser's last check: memory only, the check's certificate is new each start
  constructor(private file: string) {
    this.list = existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : [];
  }
  private save() { writeFileSync(this.file, JSON.stringify(this.list, null, 1), { mode: 0o600 }); }
  private prune() {
    const now = Date.now();
    for (const [s, a] of this.asks) if (now - a.at > ASK_MS) this.asks.delete(s);
    for (const [n, at] of this.nonces) if (now - at > NONCE_MS) this.nonces.delete(n);
  }

  // A request that came through the trust check: a pairing page's nonce, and the browser's own key when it has one
  check(req: Request, nonce: string | null) {
    this.prune();
    if (nonce && /^[\w-]{16,64}$/.test(nonce) && this.nonces.size < 256) this.nonces.set(nonce, Date.now());
    const found = this.find(req);
    if (found && found.device.kind !== "computer") this.checked.set(found.device.id, Date.now());
  }
  // through the gate: a browser that came through the trust check lately (never a linked computer, whose key is the
  // link's alone: Chrysalis's API, not the chat or its tools)
  trusted(device: Device) { return Date.now() - (this.checked.get(device.id) ?? 0) < CHECK_MS; }
  // past half its time: the page checks again before the gate asks it to
  stale(device: Device) { return Date.now() - (this.checked.get(device.id) ?? 0) > CHECK_MS / 2; }

  // A device asks to join, from a pairing page whose nonce came through the trust check (once): the secret the page
  // collects its key with, and the code it shows, which no other ask has
  ask(nonce: string, name: string): { secret: string; code: string } | undefined {
    this.prune();
    if (!this.nonces.delete(nonce)) return undefined;
    while (this.asks.size >= MAX_ASKS) this.asks.delete(this.asks.keys().next().value!);
    const taken = new Set([...this.asks.values()].map((a) => a.code));
    let code: string;
    do code = String(randomInt(1_000_000)).padStart(6, "0"); while (taken.has(code));
    const secret = randomBytes(24).toString("base64url");
    this.asks.set(secret, { code, name: name.slice(0, 40) || "A device", at: Date.now(), allowed: false, kind: "browser" });
    return { secret, code };
  }
  // A computer asks to link (link.ts), committing to a nonce it shows only once it has this computer's: the secret it
  // reveals and collects its key with, and this computer's nonce
  linkAsk(name: string, commit: string): { secret: string; nonce: string } | undefined {
    this.prune();
    if (!/^[0-9a-f]{64}$/.test(commit)) return undefined;
    while (this.asks.size >= MAX_ASKS) this.asks.delete(this.asks.keys().next().value!);
    const secret = randomBytes(24).toString("base64url"), nonce = randomBytes(16).toString("hex");
    this.asks.set(secret, { code: "", name: name.slice(0, 40) || "A computer", at: Date.now(), allowed: false, kind: "computer", commit, nonce });
    return { secret, nonce };
  }
  // its nonce, which must be the one it committed to: then its code, from this computer's link authority as it is
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
  // The asking page's answer: its key once allowed, once (the ask is gone then); it came through the check to ask
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

  // the device whose key a request brings, and the key, to set again (its 400 days count from the last visit)
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
