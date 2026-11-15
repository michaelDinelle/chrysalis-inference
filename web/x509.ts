import { createHash, createPublicKey, generateKeyPairSync, randomBytes, sign, X509Certificate, type KeyObject } from "node:crypto";

type Bytes = Uint8Array;
const cat = (parts: Bytes[]) => Buffer.concat(parts);
function tlv(tag: number, ...parts: Bytes[]): Buffer {
  const body = cat(parts), n = body.length;
  const size = n < 0x80 ? [n] : (() => { const b: number[] = []; for (let x = n; x > 0; x >>= 8) b.unshift(x & 0xff); return [0x80 | b.length, ...b]; })();
  return cat([Uint8Array.of(tag, ...size), body]);
}
const seq = (...p: Bytes[]) => tlv(0x30, ...p);
const set = (...p: Bytes[]) => tlv(0x31, ...p);
const ctx = (n: number, ...p: Bytes[]) => tlv(0xa0 | n, ...p);
const tag = (n: number, b: Bytes) => tlv(0x80 | n, b);
const bool = (v: boolean) => tlv(0x01, Uint8Array.of(v ? 0xff : 0));
const octets = (b: Bytes) => tlv(0x04, b);
const bits = (b: Bytes, unused = 0) => tlv(0x03, Uint8Array.of(unused), b);
function int(b: Bytes) {
  let i = 0;
  while (i < b.length - 1 && b[i] === 0 && !(b[i + 1] & 0x80)) i++;
  const v = b.subarray(i);
  return tlv(0x02, v[0] & 0x80 ? cat([Uint8Array.of(0), v]) : v);
}
function oid(dotted: string) {
  const [a, b, ...rest] = dotted.split(".").map(Number), out = [a * 40 + b];
  for (const n of rest) { const g: number[] = []; for (let x = n; ; x = Math.floor(x / 128)) { g.unshift((x % 128) | (g.length ? 0x80 : 0)); if (x < 128) break; } out.push(...g); }
  return tlv(0x06, Uint8Array.from(out));
}
function time(d: Date) {
  const iso = d.toISOString().replace(/[-:T]/g, "").slice(0, 14) + "Z";
  return d.getUTCFullYear() < 2050 ? tlv(0x17, Buffer.from(iso.slice(2))) : tlv(0x18, Buffer.from(iso));
}
const ip = (s: string) => Uint8Array.from(s.split(".").map(Number));

function children(der: Bytes): Bytes[] {
  const out: Bytes[] = [];
  for (let at = 0; at < der.length; ) {
    let n = der[at + 1], head = 2;
    if (n & 0x80) { const k = n & 0x7f; n = 0; for (let i = 0; i < k; i++) n = n * 256 + der[at + 2 + i]; head += k; }
    out.push(der.subarray(at, at + head + n));
    at += head + n;
  }
  return out;
}
const inside = (t: Bytes) => t.subarray(t[1] & 0x80 ? 2 + (t[1] & 0x7f) : 2);
const content = (t: Bytes) => children(inside(t));

const ECDSA_SHA256 = seq(oid("1.2.840.10045.4.3.2"));
const ext = (id: string, critical: boolean, value: Bytes) => seq(oid(id), ...(critical ? [bool(true)] : []), octets(value));
const nameOf = (cn: string) => seq(set(seq(oid("2.5.4.3"), tlv(0x0c, Buffer.from(cn)))));
const spkiOf = (key: KeyObject) => (key.type === "private" ? createPublicKey(key) : key).export({ type: "spki", format: "der" });
export const keyId = (key: KeyObject) => createHash("sha1").update(inside(content(spkiOf(key))[1]).subarray(1)).digest();
export const subjectOf = (pem: string) => {
  const tbs = content(content(new X509Certificate(pem).raw)[0]);
  return tbs[tbs[0][0] === 0xa0 ? 5 : 4];
};
const pem = (label: string, der: Bytes) => `-----BEGIN ${label}-----\n${Buffer.from(der).toString("base64").match(/.{1,64}/g)!.join("\n")}\n-----END ${label}-----\n`;

export const newKey = () => generateKeyPairSync("ec", { namedCurve: "prime256v1" }).privateKey;
export const keyPem = (key: KeyObject) => key.export({ type: "pkcs8", format: "pem" }).toString();

export interface Issue {
  subject: string | Bytes;
  key: KeyObject;
  issuer?: { cert: string; key: KeyObject };
  days: number;
  authority?: { ips?: string[]; dns?: string[] };
  names?: string;
}

export function issue(o: Issue): string {
  const subject = typeof o.subject === "string" ? nameOf(o.subject) : o.subject, signer = o.issuer?.key ?? o.key;
  const issuerName = o.issuer ? subjectOf(o.issuer.cert) : subject, own = keyId(o.key), theirs = keyId(signer);
  const serial = randomBytes(20);
  serial[0] = (serial[0] & 0x7f) || 1;
  const now = new Date(), until = new Date(now.getTime() + o.days * 86_400_000);
  const general = (s: string) => (s.startsWith("IP:") ? tag(7, ip(s.slice(3))) : tag(2, Buffer.from(s.slice(4))));
  const exts = [ext("2.5.29.14", false, octets(own)), ext("2.5.29.35", false, seq(tag(0, theirs)))];
  if (o.names) exts.push(ext("2.5.29.17", false, seq(...o.names.split(",").map(general))), ext("2.5.29.37", false, seq(oid("1.3.6.1.5.5.7.3.1"))));
  exts.push(ext("2.5.29.19", true, o.authority ? seq(bool(true), int(Uint8Array.of(0))) : seq()));
  exts.push(ext("2.5.29.15", true, o.authority ? bits(Uint8Array.of(0x06), 1) : bits(Uint8Array.of(0x80), 7)));
  if (o.authority) {
    const within = [...(o.authority.ips ?? []).map((n) => { const [a, m] = n.split("/"); return seq(tag(7, cat([ip(a), ip(m)]))); }), ...(o.authority.dns ?? []).map((d) => seq(tag(2, Buffer.from(d))))];
    exts.push(ext("2.5.29.30", true, seq(ctx(0, ...within))));
  }
  const tbs = seq(ctx(0, int(Uint8Array.of(2))), int(serial), ECDSA_SHA256, issuerName, seq(time(now), time(until)), subject, spkiOf(o.key), ctx(3, seq(...exts)));
  return pem("CERTIFICATE", seq(tbs, ECDSA_SHA256, bits(sign("sha256", tbs, signer))));
}
