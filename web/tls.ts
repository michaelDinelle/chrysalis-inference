import { createPrivateKey, X509Certificate } from "node:crypto";
import { createSocket } from "node:dgram";
import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir, hostname, networkInterfaces } from "node:os";
import { issue, keyPem, newKey, subjectOf, type Issue } from "./x509";

export const lanAddresses = () =>
  Object.values(networkInterfaces()).flat().flatMap((a) => (a && !a.internal && a.family === "IPv4" ? [a.address] : []));

export function routeAddress(): Promise<string | undefined> {
  return new Promise((resolve) => {
    const s = createSocket("udp4"), done = (a?: string) => { s.close(); resolve(a ?? lanAddresses()[0]); };
    s.on("error", () => done());
    s.connect(9, "192.0.2.1", () => done(s.address().address));
  });
}

const PRIVATE = ["10.0.0.0/255.0.0.0", "172.16.0.0/255.240.0.0", "192.168.0.0/255.255.0.0", "100.64.0.0/255.192.0.0",
  "169.254.0.0/255.255.0.0", "127.0.0.0/255.0.0.0"];

export const tlsPath = () => process.env.CHRYSALIS_TLS_DIR ?? `${homedir()}/.config/chrysalis/tls`;

export function tlsDir(dir = tlsPath()) {
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  return dir;
}

export const netName = () => `chrysalis-${hostname().toLowerCase().replace(/\.local$/, "").replace(/[^a-z0-9-]/g, "-")}.local`;

export function webTls(dir = tlsDir()) {
  const host = hostname().toLowerCase(), at = (f: string) => `${dir}/${f}`;
  const keyed = (name: string) => {
    const key = newKey();
    writeFileSync(at(`${name}.key`), keyPem(key), { mode: 0o600 });
    return key;
  };
  const made = (name: string, o: Omit<Issue, "key">) => writeFileSync(at(`${name}.pem`), issue({ ...o, key: keyed(name) }));
  const loaded = (name: string) => createPrivateKey(readFileSync(at(`${name}.key`), "utf8"));
  if (!existsSync(at("ca.pem"))) made("ca", { subject: `Chrysalis Inference on ${host}`, days: 3650, authority: { ips: PRIVATE, dns: ["local", "localhost", host] } });
  const names = [`DNS:${host}`, `DNS:${host}.local`, `DNS:${netName()}`, "DNS:localhost", "IP:127.0.0.1", ...lanAddresses().map((a) => `IP:${a}`)].join(",");
  const lasting = (name: string) => { try { return new Date(new X509Certificate(readFileSync(at(`${name}.pem`), "utf8")).validTo).getTime() > Date.now() + 30 * 86_400_000; } catch { return false; } };
  const stale = !existsSync(at("names")) || readFileSync(at("names"), "utf8") !== names || !lasting("server");
  const ca = readFileSync(at("ca.pem"), "utf8"), signer = { cert: ca, key: loaded("ca") };
  if (stale) {
    made("server", { subject: host, issuer: signer, days: 825, names });
    writeFileSync(at("names"), names);
  }
  made("check", { subject: `${host} trust check`, issuer: signer, days: 825, names });
  if (!existsSync(at("link-ca.pem"))) made("link-ca", { subject: `Chrysalis link on ${host}`, days: 3650, authority: {} });
  const linkCa = readFileSync(at("link-ca.pem"), "utf8");
  if (stale || !existsSync(at("link.pem"))) made("link", { subject: `${host} link`, issuer: { cert: linkCa, key: loaded("link-ca") }, days: 825, names });
  for (const k of ["ca.key", "server.key", "check.key", "link-ca.key", "link.key"]) chmodSync(at(k), 0o600);
  const x = new X509Certificate(ca);
  if (!existsSync(at("dns-ca.pem")))
    writeFileSync(at("dns-ca.pem"), issue({ subject: subjectOf(ca), key: signer.key, days: 3650, authority: { dns: ["local", "localhost", host] } }));
  return { dir, cert: readFileSync(at("server.pem"), "utf8"), key: readFileSync(at("server.key"), "utf8"), ca, dnsCa: readFileSync(at("dns-ca.pem"), "utf8"), caDer: Uint8Array.from(x.raw),
    caName: x.subject.replace(/^CN=/, ""), fingerprint: x.fingerprint256,
    check: { cert: readFileSync(at("check.pem"), "utf8"), key: readFileSync(at("check.key"), "utf8") },
    link: { cert: readFileSync(at("link.pem"), "utf8"), key: readFileSync(at("link.key"), "utf8"), ca: linkCa, fingerprint: fingerprint(linkCa) } };
}

export const fingerprint = (pem: string) => new X509Certificate(pem).fingerprint256.replaceAll(":", "").toLowerCase();
