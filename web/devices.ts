// Who may use the web chat from the network (`chrysalis start --lan`): the devices this computer's own screen let in.
// Its QR code carries a pairing code (or the code is typed, where there is no camera), and a device that brings one gets
// a key of its own, kept by its browser as a cookie that only the chat's HTTPS gets. This machine needs no key: what
// runs here can reach Chrysalis anyway.
//
// A pairing code is 8 characters (40 bits), short enough to type, since it is only ever tried against this server and
// never offline: it works once, for 10 minutes, and 20 wrong codes void every live one, so each code shown gives a
// guesser at most 20 tries at odds of 1 in 2^40. A device key is 256 bits and is kept here as its SHA-256 only, so the
// file lets nobody in, and comparing hashes tells a timing attacker nothing about a key.
import { createHash, randomBytes, randomInt } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";

export interface Device { id: string; name: string; hash: string; added: number; seen: number }

const ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ";       // no 0, O, 1 or I: read off a screen and typed
const CODE_MS = 10 * 60_000, MAX_FAILS = 20, MAX_CODES = 8;
const COOKIE = "chrysalis-device", YEAR = 400 * 86400;    // 400 days: the longest browsers keep a cookie
const sha = (s: string) => createHash("sha256").update(s).digest("hex");
const cookie = (req: Request) => req.headers.get("cookie")?.split(/;\s*/).find((c) => c.startsWith(`${COOKIE}=`))?.slice(COOKIE.length + 1);
// Secure: never over plain HTTP; HttpOnly: no script reads it; Strict: no other site's page sends it
export const keyCookie = (key: string) => `${COOKIE}=${key}; Path=/; Max-Age=${YEAR}; Secure; HttpOnly; SameSite=Strict`;

export class Devices {
  private list: Device[];
  private codes = new Map<string, number>();              // a live code and when it ends
  private fails = 0;
  constructor(private file: string) {
    this.list = existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : [];
  }
  private save() { writeFileSync(this.file, JSON.stringify(this.list, null, 1), { mode: 0o600 }); }

  code(): { code: string; expires: number } {
    const now = Date.now();
    for (const [c, end] of this.codes) if (end <= now) this.codes.delete(c);
    while (this.codes.size >= MAX_CODES) this.codes.delete(this.codes.keys().next().value!);
    let code = "";
    for (let i = 0; i < 8; i++) code += ALPHABET[randomInt(ALPHABET.length)];
    this.codes.set(code, now + CODE_MS);
    return { code, expires: now + CODE_MS };
  }
  live() { return [...this.codes].filter(([, end]) => end > Date.now()).map(([c]) => c); }

  // a new device's key for a live code, which then is not
  redeem(typed: string, name: string): string | undefined {
    const code = typed.toUpperCase().replace(/[^0-9A-Z]/g, ""), end = this.codes.get(code);
    if (!end || end <= Date.now()) {
      if (++this.fails >= MAX_FAILS) { this.codes.clear(); this.fails = 0; }
      return undefined;
    }
    this.codes.delete(code);
    const key = randomBytes(32).toString("base64url"), now = Date.now();
    this.list.push({ id: randomBytes(6).toString("hex"), name: name.slice(0, 40) || "A device", hash: sha(key), added: now, seen: now });
    this.save();
    return key;
  }

  // the device whose key a request brings, and the key, to set again (its 400 days count from the last visit)
  find(req: Request): { device: Device; key: string } | undefined {
    const key = cookie(req), hash = key && sha(key), device = hash ? this.list.find((d) => d.hash === hash) : undefined;
    if (!device) return undefined;
    if (Date.now() - device.seen > 60_000) { device.seen = Date.now(); this.save(); }
    return { device, key: key! };
  }
  all() { return this.list.map(({ hash: _, ...d }) => d); }
  remove(id: string) {
    const n = this.list.length;
    this.list = this.list.filter((d) => d.id !== id);
    if (this.list.length !== n) this.save();
    return this.list.length !== n;
  }
}
