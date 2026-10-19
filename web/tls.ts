// The web chat over HTTPS on the local network (`kiln start --lan`). Browsers give the microphone (voice mode) only to
// secure pages, and http://<this machine's address> is not one (localhost is). A certificate authority of this
// machine's own signs the server's certificate; its name constraints let it vouch only for private addresses and local
// names, so a device that trusts it (kiln-ca.crt, installed once) trusts nothing else because of it.
import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir, hostname, networkInterfaces } from "node:os";

// IPv4 only: the CA's constraints name the private IPv4 ranges, and a network's IPv6 addresses are usually global
export const lanAddresses = () =>
  Object.values(networkInterfaces()).flat().flatMap((a) => (a && !a.internal && a.family === "IPv4" ? [a.address] : []));

const PRIVATE = ["10.0.0.0/255.0.0.0", "172.16.0.0/255.240.0.0", "192.168.0.0/255.255.0.0", "100.64.0.0/255.192.0.0",
  "169.254.0.0/255.255.0.0", "127.0.0.0/255.0.0.0"];                  // 100.64/10: Tailscale's addresses

function openssl(...args: string[]) {
  const r = Bun.spawnSync(["openssl", ...args], { stdout: "pipe", stderr: "pipe" });
  return { ok: r.exitCode === 0, err: r.stderr.toString().trim() };
}

export function lanTls(dir = process.env.KILN_TLS_DIR ?? `${homedir()}/.config/kiln/tls`) {
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  const host = hostname().toLowerCase(), ec = ["-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes"];
  const run = (...args: string[]) => {
    const r = openssl(...args);
    if (!r.ok) throw new Error(`openssl ${args[0]}: ${r.err}`);
  };
  if (!existsSync(`${dir}/ca.pem`)) {
    const allow = [...PRIVATE.map((n) => `permitted;IP:${n}`), "permitted;DNS:local", "permitted;DNS:localhost", `permitted;DNS:${host}`];
    run("req", "-x509", ...ec, "-keyout", `${dir}/ca.key`, "-out", `${dir}/ca.pem`, "-days", "3650", "-subj", `/CN=kiln on ${host}`,
      "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0", "-addext", "keyUsage=critical,keyCertSign,cRLSign",
      "-addext", `nameConstraints=critical,${allow.join(",")}`);
  }
  // A new certificate when the addresses change (another network) or within 30 days of its end. 825 days is the longest
  // Apple's devices accept from an authority the user installed.
  const names = [`DNS:${host}`, `DNS:${host}.local`, "DNS:localhost", "IP:127.0.0.1", ...lanAddresses().map((a) => `IP:${a}`)].join(",");
  const stale = !existsSync(`${dir}/server.pem`) || !existsSync(`${dir}/names`) || readFileSync(`${dir}/names`, "utf8") !== names
    || !openssl("x509", "-checkend", String(30 * 86400), "-noout", "-in", `${dir}/server.pem`).ok;
  if (stale) {
    run("req", "-x509", ...ec, "-keyout", `${dir}/server.key`, "-out", `${dir}/server.pem`, "-days", "825", "-subj", `/CN=${host}`,
      "-CA", `${dir}/ca.pem`, "-CAkey", `${dir}/ca.key`, "-addext", `subjectAltName=${names}`, "-addext", "extendedKeyUsage=serverAuth",
      "-addext", "basicConstraints=critical,CA:FALSE", "-addext", "keyUsage=critical,digitalSignature");
    writeFileSync(`${dir}/names`, names);
  }
  for (const k of ["ca.key", "server.key"]) chmodSync(`${dir}/${k}`, 0o600);
  return { cert: readFileSync(`${dir}/server.pem`, "utf8"), key: readFileSync(`${dir}/server.key`, "utf8"), ca: readFileSync(`${dir}/ca.pem`, "utf8") };
}
