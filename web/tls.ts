import { X509Certificate } from "node:crypto";
import { createSocket } from "node:dgram";
import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir, hostname, networkInterfaces } from "node:os";

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

function openssl(...args: string[]) {
  const r = Bun.spawnSync(["openssl", ...args], { stdout: "pipe", stderr: "pipe" });
  return { ok: r.exitCode === 0, err: r.stderr.toString().trim() };
}

export const tlsPath = () => process.env.CHRYSALIS_TLS_DIR ?? `${homedir()}/.config/chrysalis/tls`;

export function tlsDir(dir = tlsPath()) {
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  return dir;
}

export const netName = () => `chrysalis-${hostname().toLowerCase().replace(/\.local$/, "").replace(/[^a-z0-9-]/g, "-")}.local`;

export function webTls(dir = tlsDir()) {
  const host = hostname().toLowerCase(), ec = ["-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes"];
  const run = (...args: string[]) => {
    const r = openssl(...args);
    if (!r.ok) throw new Error(`openssl ${args[0]}: ${r.err}`);
  };
  if (!existsSync(`${dir}/ca.pem`)) {
    const allow = [...PRIVATE.map((n) => `permitted;IP:${n}`), "permitted;DNS:local", "permitted;DNS:localhost", `permitted;DNS:${host}`];
    run("req", "-x509", ...ec, "-keyout", `${dir}/ca.key`, "-out", `${dir}/ca.pem`, "-days", "3650", "-subj", `/CN=Chrysalis Inference on ${host}`,
      "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0", "-addext", "keyUsage=critical,keyCertSign,cRLSign",
      "-addext", `nameConstraints=critical,${allow.join(",")}`);
  }
  const names = [`DNS:${host}`, `DNS:${host}.local`, `DNS:${netName()}`, "DNS:localhost", "IP:127.0.0.1", ...lanAddresses().map((a) => `IP:${a}`)].join(",");
  const stale = !existsSync(`${dir}/server.pem`) || !existsSync(`${dir}/names`) || readFileSync(`${dir}/names`, "utf8") !== names
    || !openssl("x509", "-checkend", String(30 * 86400), "-noout", "-in", `${dir}/server.pem`).ok;
  if (stale) {
    run("req", "-x509", ...ec, "-keyout", `${dir}/server.key`, "-out", `${dir}/server.pem`, "-days", "825", "-subj", `/CN=${host}`,
      "-CA", `${dir}/ca.pem`, "-CAkey", `${dir}/ca.key`, "-addext", `subjectAltName=${names}`, "-addext", "extendedKeyUsage=serverAuth",
      "-addext", "basicConstraints=critical,CA:FALSE", "-addext", "keyUsage=critical,digitalSignature");
    writeFileSync(`${dir}/names`, names);
  }
  run("req", "-x509", ...ec, "-keyout", `${dir}/check.key`, "-out", `${dir}/check.pem`, "-days", "825", "-subj", `/CN=${host} trust check`,
    "-CA", `${dir}/ca.pem`, "-CAkey", `${dir}/ca.key`, "-addext", `subjectAltName=${names}`, "-addext", "extendedKeyUsage=serverAuth",
    "-addext", "basicConstraints=critical,CA:FALSE", "-addext", "keyUsage=critical,digitalSignature");
  if (!existsSync(`${dir}/link-ca.pem`)) {
    run("req", "-x509", ...ec, "-keyout", `${dir}/link-ca.key`, "-out", `${dir}/link-ca.pem`, "-days", "3650", "-subj", `/CN=Chrysalis link on ${host}`,
      "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0", "-addext", "keyUsage=critical,keyCertSign,cRLSign");
  }
  if (stale || !existsSync(`${dir}/link.pem`)) {
    run("req", "-x509", ...ec, "-keyout", `${dir}/link.key`, "-out", `${dir}/link.pem`, "-days", "825", "-subj", `/CN=${host} link`,
      "-CA", `${dir}/link-ca.pem`, "-CAkey", `${dir}/link-ca.key`, "-addext", `subjectAltName=${names}`, "-addext", "extendedKeyUsage=serverAuth",
      "-addext", "basicConstraints=critical,CA:FALSE", "-addext", "keyUsage=critical,digitalSignature");
  }
  for (const k of ["ca.key", "server.key", "check.key", "link-ca.key", "link.key"]) chmodSync(`${dir}/${k}`, 0o600);
  const ca = readFileSync(`${dir}/ca.pem`, "utf8"), x = new X509Certificate(ca), linkCa = readFileSync(`${dir}/link-ca.pem`, "utf8");
  if (!existsSync(`${dir}/dns-ca.pem`)) {
    run("req", "-x509", "-new", "-key", `${dir}/ca.key`, "-out", `${dir}/dns-ca.pem`, "-days", "3650", "-subj", `/${x.subject}`,
      "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0", "-addext", "keyUsage=critical,keyCertSign,cRLSign",
      "-addext", `nameConstraints=critical,permitted;DNS:local,permitted;DNS:localhost,permitted;DNS:${host}`);
  }
  return { dir, cert: readFileSync(`${dir}/server.pem`, "utf8"), key: readFileSync(`${dir}/server.key`, "utf8"), ca, dnsCa: readFileSync(`${dir}/dns-ca.pem`, "utf8"), caDer: Uint8Array.from(x.raw),
    caName: x.subject.replace(/^CN=/, ""), fingerprint: x.fingerprint256,
    check: { cert: readFileSync(`${dir}/check.pem`, "utf8"), key: readFileSync(`${dir}/check.key`, "utf8") },
    link: { cert: readFileSync(`${dir}/link.pem`, "utf8"), key: readFileSync(`${dir}/link.key`, "utf8"), ca: linkCa, fingerprint: fingerprint(linkCa) } };
}

export const fingerprint = (pem: string) => new X509Certificate(pem).fingerprint256.replaceAll(":", "").toLowerCase();
