import { expect, test } from "bun:test";
import { hostname } from "node:os";
import { chmodSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import mdns from "multicast-dns";
import { advertise, discover } from "./link";
import { fingerprint, lanAddresses, netName, webTls } from "./tls";
import { harness, RUN, TLS, announcement, ask, base, chrysalisAborted, fake, json, onNetwork, port, post, seen, shown, until, web, withoutEngine } from "./server-harness";

harness();

test("on the network (CHRYSALIS_HOST 0.0.0.0): each address gets HTTPS whose certificate the server's own authority signs, and no plain HTTP, while this machine gets plain HTTP; that authority vouches for nothing public; a device without a key gets the connect page and the authority as its system takes it, and none of the chat's files; the trust check has a certificate of its own from the same authority", async () => {
  const { proc, dir, ips, urls, checks } = await onNetwork();
  try {
    const port = new URL(urls[0]).port;
    expect(urls).toEqual([`http://127.0.0.1:${port}`, ...ips.map((ip) => `https://${ip}:${port}`)]);
    const ca = await (await fetch(`${urls[0]}/chrysalis-ca.crt`)).text();
    writeFileSync(`${dir}/ca-served.pem`, ca);
    const tlsCheck = (ip: string, ...with_: string[]) => Bun.spawnSync(["openssl", "s_client", "-connect", `${ip}:${port}`, "-verify_return_error",
      "-verify_ip", ip, ...with_], { stdin: new Uint8Array(), stdout: "pipe", stderr: "pipe" });
    for (const u of urls.slice(1)) {
      const ip = new URL(u).hostname, good = tlsCheck(ip, "-CAfile", `${dir}/ca-served.pem`);
      expect(good.stdout.toString()).toContain("Verify return code: 0 (ok)");
      expect(tlsCheck(ip).exitCode).not.toBe(0);
      expect((await fetch(`${u}/`, { tls: { rejectUnauthorized: false } })).status).toBe(200);
      expect(await fetch(`${u.replace("https:", "http:")}/`).then((r) => `plain ${r.status}`, () => "refused")).toBe("refused");
    }
    const sign = Bun.spawnSync(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-keyout", `${dir}/x.key`,
      "-out", `${dir}/x.pem`, "-days", "30", "-subj", "/CN=example.com", "-CA", `${dir}/ca.pem`, "-CAkey", `${dir}/ca.key`, "-addext", "subjectAltName=DNS:example.com"]);
    expect(sign.exitCode).toBe(0);
    const verify = Bun.spawnSync(["openssl", "verify", "-CAfile", `${dir}/ca-served.pem`, `${dir}/x.pem`], { stderr: "pipe" });
    expect(verify.exitCode).not.toBe(0);
    expect(verify.stdout.toString() + verify.stderr.toString()).toContain("permitted subtree violation");

    const der = Bun.spawnSync(["openssl", "x509", "-in", `${dir}/ca.pem`, "-outform", "DER"]).stdout;
    const fingerprint = Bun.spawnSync(["openssl", "x509", "-in", `${dir}/ca.pem`, "-noout", "-fingerprint", "-sha256"]).stdout.toString().trim().split("=")[1];
    const insecure = { tls: { rejectUnauthorized: false } };
    const chatJs = (await (await fetch(`${urls[0]}/`)).text()).match(/src="(\/[^"]+\.js)"/)![1];
    for (const [k, u] of urls.slice(1).entries()) {
      const page = await fetch(`${u}/`, insecure);
      expect([page.status, page.headers.get("content-type"), page.headers.get("cache-control")]).toEqual([200, "text/html; charset=utf-8", "no-store"]);
      const html = await page.text();
      expect([html.includes("<title>Connect to Chrysalis</title>"), html.includes(chatJs), html.includes(JSON.stringify(`${checks[k]}/check.gif`))]).toEqual([true, false, true]);
      expect([(await fetch(`${u}${chatJs}`, insecure)).status, (await fetch(`${urls[0]}${chatJs}`)).status]).toEqual([404, 200]);
      const other = await fetch(`${u}/pair`, { ...insecure, redirect: "manual", headers: { "Sec-Fetch-Mode": "navigate" } });
      expect([other.status, other.headers.get("location")]).toEqual([302, "/"]);
      const profile = await fetch(`${u}/chrysalis.mobileconfig`, insecure);
      expect(profile.headers.get("content-type")).toBe("application/x-apple-aspen-config");
      writeFileSync(`${dir}/served.mobileconfig`, Buffer.from(await profile.arrayBuffer()));
      const read = Bun.spawnSync([`${import.meta.dir}/../data/venv/bin/python`, "-c", "import plistlib, sys; p = plistlib.load(open(sys.argv[1], 'rb')); "
        + "c = p['PayloadContent'][0]; print(p['PayloadType'], c['PayloadType']); sys.stdout.flush(); sys.stdout.buffer.write(c['PayloadContent'])",
        `${dir}/served.mobileconfig`], { stdout: "pipe", stderr: "pipe" });
      const [types, ...rest] = Buffer.from(read.stdout).toString("latin1").split("\n");
      expect(types).toBe("Configuration com.apple.security.root");
      expect(Buffer.from(rest.join("\n"), "latin1").equals(Buffer.from(der))).toBe(true);
      expect(await (await fetch(`${u}/chrysalis-ca.crt`, insecure)).text()).toBe(readFileSync(`${dir}/ca.pem`, "utf8"));
      const firefox = await fetch(`${u}/chrysalis-ca-firefox.crt`, insecure);
      expect([firefox.headers.get("content-type"), Buffer.from(await firefox.arrayBuffer()).equals(Buffer.from(der))]).toEqual(["application/x-x509-ca-cert", true]);
      const gif = await fetch(`${checks[k]}/check.gif`, { tls: { rejectUnauthorized: false } });
      expect([gif.status, gif.headers.get("content-type")]).toEqual([200, "image/gif"]);
      const atCheck = shown(ips[k], new URL(checks[k]).port, `${dir}/ca-served.pem`), atChat = shown(ips[k], port, `${dir}/ca-served.pem`);
      expect([atCheck.ok, atChat.ok, !!atCheck.cert && atCheck.cert !== atChat.cert]).toEqual([true, true, true]);
    }
    const pair = await (await fetch(`${urls[0]}/api/pair`)).json();
    expect([pair.lan, pair.fingerprint, pair.connect.map((u: string) => u.replace(/\/$/, "")).sort()]).toEqual([true, fingerprint, urls.slice(1).sort()]);
  } finally {
    proc.kill();
  }
});

test("on the network, this machine gets the chat with no key and no connect page, at 127.0.0.1 over plain HTTP and at its own address over HTTPS, and lets devices in", async () => {
  const { proc, urls } = await onNetwork(undefined, true);
  try {
    const insecure = { tls: { rejectUnauthorized: false } }, [here, net] = urls;
    for (const u of [here, net]) {
      const html = await (await fetch(`${u}/`, insecure)).text();
      expect([html.includes("<title>Connect to Chrysalis</title>"), /src="\/[^"]+\.js"/.test(html)]).toEqual([false, true]);
    }
    const status = await (await fetch(`${net}/api/status`, insecure)).json(), pair = await fetch(`${net}/api/pair`, insecure);
    expect([status.here, pair.status, (await pair.json()).lan]).toEqual([true, 200, true]);
  } finally {
    proc.kill();
  }
});

test("on the network only the devices this computer let in get past the pages, and a browser only while it shows it trusts Chrysalis: a device asks once its nonce came through the trust check, shows its own code, and is let in when that code is allowed here, or not when denied; after a restart it shows it again; a device let in cannot change the backend; a device removed is out again", async () => {
  const first = await onNetwork(), { dir, ips } = first;
  let proc = first.proc;
  try {
    const [here, net] = first.urls, check = first.checks[0], insecure = { tls: { rejectUnauthorized: false } };
    const asked = (path: string, init: RequestInit = {}) => fetch(`${net}${path}`, { ...insecure, redirect: "manual", ...init });
    const through = (base: string, query = "", cookie = "") => fetch(`${base}/check.gif${query}`, { ...insecure, headers: cookie ? { cookie } : {} });
    const ask = (nonce: string) => asked("/pair/ask", { method: "POST", body: JSON.stringify({ nonce, name: "Test phone" }), headers: json });
    const answer = (secret: string) => asked(`/pair/ask/${secret}`);
    const local = (path: string, body: object) => fetch(`${here}${path}`, { method: "POST", body: JSON.stringify(body), headers: json });
    const nonce = () => crypto.randomUUID().replaceAll("-", "");
    const voice = (cookie = "") => new Promise<string>((done) => {
      const ws = new WebSocket(`${net.replace("https:", "wss:")}/api/voice`, { headers: { Origin: net, cookie }, ...insecure } as unknown as string[]);
      ws.onopen = () => (ws.close(), done("open"));
      ws.onerror = () => done("refused");
    });

    const connecting = async (cookie = "", base = net) => (await (await fetch(`${base}/`, { ...insecure, headers: cookie ? { cookie } : {} })).text())
      .includes("<title>Connect to Chrysalis</title>");
    expect(await connecting()).toBe(true);
    expect([(await asked("/api/status")).status, (await asked("/api/threads")).status, (await asked("/api/pair")).status,
      (await asked("/api/chat", { method: "POST", body: JSON.stringify({ stream: true, messages: [] }), headers: json })).status,
      (await asked("/api/chrysalis/load", { method: "POST", body: JSON.stringify({ model: "qwen3.8-27b" }), headers: json })).status]).toEqual([401, 401, 401, 401, 401]);
    expect(await voice()).toBe("refused");
    expect((await fetch(`${here}/api/status`)).status).toBe(200);

    const n = nonce();
    expect((await ask(n)).status).toBe(403);
    expect((await through(check, `?n=${n}`)).status).toBe(200);
    const asking = await ask(n), { secret, code } = await asking.json();
    expect([asking.status, /^\d{6}$/.test(code)]).toEqual([200, true]);
    expect((await ask(n)).status).toBe(403);
    expect(await (await answer(secret)).json()).toEqual({ state: "waiting" });

    expect((await asked("/api/pair/allow", { method: "POST", body: JSON.stringify({ code }), headers: json })).status).toBe(401);
    expect((await (await fetch(`${here}/api/pair`)).json()).waiting.map((a: any) => [a.code, a.name, a.allowed])).toEqual([[code, "Test phone", false]]);
    expect((await local("/api/pair/allow", { code: code === "000000" ? "111111" : "000000" })).status).toBe(404);
    expect((await local("/api/pair/allow", { code: `${code.slice(0, 3)} ${code.slice(3)}` })).status).toBe(200);
    const got = await answer(secret);
    expect(await got.clone().json()).toEqual({ state: "allowed" });
    const set = got.headers.get("set-cookie")!, cookie = set.split(";")[0], key = cookie.split("=")[1];
    expect(set.split(/;\s*/).slice(1).filter((a) => /^(Secure|HttpOnly|SameSite=Strict)$/.test(a)).sort()).toEqual(["HttpOnly", "SameSite=Strict", "Secure"]);
    expect((await answer(secret)).status).toBe(404);
    const status = await asked("/api/status", { headers: { cookie } });
    expect([status.status, status.headers.get("set-cookie")?.split(";")[0]]).toEqual([200, cookie]);
    expect([(await asked("/api/threads", { headers: { cookie } })).status, await voice(cookie)]).toEqual([200, "open"]);
    expect(await connecting(cookie)).toBe(false);
    const twice = (path: string, ...args: string[]) => Bun.spawnSync(["curl", "-sk", "-v", ...args, "-o", "/dev/null", `${net}${path}`, ...args, "-o", "/dev/null", `${net}${path}`],
      { stderr: "pipe" }).stderr.toString().match(/Re-?using existing/) !== null;
    expect([twice("/"), twice("/pair/ask", "-H", "Content-Type: application/json", "-d", "{}"), twice("/chrysalis-ca.crt"), twice("/", "-H", `Cookie: ${cookie}`)])
      .toEqual([false, false, false, true]);
    expect(readFileSync(`${dir}/devices.json`, "utf8")).not.toContain(key);
    expect((await asked("/api/backend", { method: "POST", body: JSON.stringify({ url: "http://127.0.0.1:1/v1" }), headers: { ...json, cookie } })).status).toBe(403);
    expect((await (await asked("/api/status", { headers: { cookie } })).json()).backend.chrysalis).toBe(true);

    const m = nonce();
    await through(check, `?n=${m}`);
    const denied = await (await ask(m)).json();
    expect((await asked("/api/pair/allow", { method: "POST", body: JSON.stringify({ code: denied.code }), headers: { ...json, cookie } })).status).toBe(403);
    expect((await local("/api/pair/deny", { code: denied.code })).status).toBe(204);
    expect((await answer(denied.secret)).status).toBe(404);

    const before = shown(ips[0], new URL(check).port, `${dir}/ca.pem`).cert;
    proc.kill();
    await proc.exited;
    const second = await onNetwork(dir);
    proc = second.proc;
    const [here2, net2] = second.urls, check2 = second.checks[0];
    const stale = await fetch(`${net2}/api/status`, { ...insecure, headers: { cookie } });
    expect([stale.status, (await stale.json()).check]).toEqual([401, true]);
    expect((await through(check2, "", cookie)).status).toBe(200);
    expect((await fetch(`${net2}/api/status`, { ...insecure, headers: { cookie } })).status).toBe(200);
    const after = shown(ips[0], new URL(check2).port, `${dir}/ca.pem`);
    expect([after.ok, !!before && after.cert !== before]).toEqual([true, true]);

    const listed = await (await fetch(`${here2}/api/pair`)).json();
    expect(listed.devices.map((d: { name: string; kind: string }) => [d.name, d.kind])).toEqual([["Test phone", "browser"]]);
    for (const d of listed.devices) expect((await fetch(`${here2}/api/pair/${d.id}`, { method: "DELETE" })).status).toBe(204);
    expect([(await fetch(`${net2}/api/status`, { ...insecure, headers: { cookie } })).status, await connecting(cookie, net2)]).toEqual([401, true]);
  } finally {
    proc.kill();
  }
});

test("a computer without the engine finds this one on the network, and is not found itself, and links to it once both screens show the same code and this computer allows it: its apps then reach Chrysalis's API at its own 127.0.0.1, a reply streamed through and stopped there as here, and nothing else of this computer; a nonce other than the one committed to gets no code; a computer removed here is out; one in between, with a link authority of its own, gets a code of its own", async () => {
  const desk = await onNetwork(), lap = await withoutEngine(), [here, net] = desk.urls, door = desk.links[0];
  const fp = fingerprint(readFileSync(`${desk.dir}/link-ca.pem`, "utf8"));
  const state = async () => (await fetch(`${lap.url}/api/link`)).json();
  const until = async (ok: (s: any) => boolean) => {
    for (const end = Date.now() + 10_000; ; await Bun.sleep(200)) {
      const s = await state();
      if (ok(s) || Date.now() > end) return s;
    }
  };
  const pair = (to: string) => fetch(`${lap.url}/api/link`, { method: "POST", headers: json, body: JSON.stringify({ fp: to }) });
  const shownHere = async () => (await (await fetch(`${here}/api/pair`)).json()).waiting.map((a: { code: string; kind: string }) => [a.code, a.kind]);
  try {
    await Bun.sleep(300);
    const announced = (await discover()).map((f) => f.fp), itself = fingerprint(readFileSync(`${lap.dir}/link-ca.pem`, "utf8"));
    expect((await until((s) => s.found.some((f: { fp: string }) => f.fp === fp))).found.map((f: { fp: string }) => f.fp)).toContain(fp);
    expect([announced.includes(fp), announced.includes(itself)]).toEqual([true, false]);
    const records = await announcement(fp);
    expect([records.map((r) => r.type).sort().join(), records.filter((r) => !(r.ttl! > 0))]).toEqual([expect.stringContaining("PTR,SRV,TXT"), []]);
    await pair(fp);
    const { code } = (await until((s) => !!s.pairing?.code)).pairing;
    expect(await shownHere()).toEqual([[code, "computer"]]);
    expect((await fetch(`${here}/api/pair/allow`, { method: "POST", headers: json, body: JSON.stringify({ code }) })).status).toBe(200);
    expect((await until((s) => !!s.pairing?.theirs)).linked).toBeUndefined();
    expect((await fetch(`${lap.url}/api/link/confirm`, { method: "POST", headers: json, body: "{}" })).status).toBe(200);
    expect((await until((s) => !!s.linked)).linked).toBeTruthy();

    expect(await (await fetch(`${lap.chrysalis}/chrysalis/stats`)).json()).toEqual({ model: "fake-model" });
    const completion = (headers: Record<string, string> = {}, signal?: AbortSignal) => fetch(`${lap.chrysalis}/v1/chat/completions`,
      { method: "POST", body: JSON.stringify({ stream: true, messages: [] }), headers: { ...json, ...headers }, signal });
    const streamed = await completion({ "x-api-key": "anything" });
    expect([streamed.status, (await streamed.text()).trim().endsWith("data: [DONE]")]).toEqual([200, true]);
    const aborts = chrysalisAborted, ctrl = new AbortController(), live = await completion({}, ctrl.signal);
    await live.body!.getReader().read();
    ctrl.abort();
    for (let t = Date.now(); chrysalisAborted === aborts && Date.now() - t < 3000; ) await Bun.sleep(50);
    expect(chrysalisAborted).toBe(aborts + 1);
    const { key } = JSON.parse(readFileSync(`${lap.dir}/link.json`, "utf8")), bearer = { authorization: `Bearer ${key}` };
    expect([(await fetch(`${net}/api/status`, { headers: bearer })).status, (await fetch(`${net}/api/threads`, { headers: bearer })).status,
      (await fetch(`${door}/chrysalis/stats`)).status, (await fetch(`${door}/chrysalis/stats`, { headers: bearer })).status]).toEqual([401, 401, 401, 200]);
    expect(readFileSync(`${desk.dir}/devices.json`, "utf8")).not.toContain(key);

    const sha = (s: string) => createHash("sha256").update(s).digest("hex");
    const asked = await (await fetch(`${door}/link/ask`, { method: "POST", headers: json, body: JSON.stringify({ name: "Another", commit: sha("one") }) })).json();
    expect((await fetch(`${door}/link/reveal`, { method: "POST", headers: json, body: JSON.stringify({ secret: asked.secret, nonce: "two" }) })).status).toBe(404);

    const listed = (await (await fetch(`${here}/api/pair`)).json()).devices;
    expect(listed.map((d: { kind: string }) => d.kind)).toEqual(["computer"]);
    await fetch(`${here}/api/pair/${listed[0].id}`, { method: "DELETE" });
    expect((await fetch(`${lap.chrysalis}/chrysalis/stats`)).status).toBe(401);
    expect((await state()).linked).toBeUndefined();

    const mid = webTls(TLS());
    const relay = async (req: Request) => fetch(`${door}${new URL(req.url).pathname}`,
      { method: req.method, headers: json, body: req.method === "POST" ? await req.text() : undefined });
    const between = Bun.serve({ hostname: desk.ips[0], port: 0, tls: { cert: mid.link.cert, key: mid.link.key },
      fetch: (req) => new URL(req.url).pathname === "/link/ca" ? new Response(mid.link.ca) : relay(req) });
    const quiet = advertise("in-between", between.port!, mid.link.fingerprint);
    try {
      await until((s) => s.found.some((f: { fp: string }) => f.fp === mid.link.fingerprint));
      await pair(mid.link.fingerprint);
      const theirs = (await until((s) => !!s.pairing?.code)).pairing.code;
      const ours = await shownHere();
      expect([ours.length, ours[0]?.[1]]).toEqual([1, "computer"]);
      expect(ours[0][0]).not.toBe(theirs);
    } finally {
      quiet();
      between.stop(true);
    }

    const fake = webTls(TLS());
    const yes = Bun.serve({ hostname: desk.ips[0], port: 0, tls: { cert: fake.link.cert, key: fake.link.key },
      fetch: (req) => {
        const path = new URL(req.url).pathname;
        return path === "/link/ca" ? new Response(fake.link.ca) : path === "/link/ask" ? Response.json({ secret: "s", nonce: "n" })
          : path === "/link/reveal" ? new Response(null, { status: 204 }) : Response.json({ state: "allowed", key: "theirs" });
      } });
    const quiet2 = advertise("omarchy", yes.port!, fake.link.fingerprint);
    try {
      await fetch(`${lap.url}/api/link`, { method: "DELETE" });
      await until((s) => s.found.some((f: { fp: string }) => f.fp === fake.link.fingerprint));
      await pair(fake.link.fingerprint);
      expect((await until((s) => !!s.pairing?.theirs)).linked).toBeUndefined();
      await Bun.sleep(1500);
      expect([(await state()).linked, existsSync(`${lap.dir}/link.json`)]).toEqual([undefined, false]);
    } finally {
      quiet2();
      yes.stop(true);
    }
  } finally {
    desk.proc.kill();
    lap.proc.kill();
  }
}, 60_000);

test("a computer with only OpenCode gets in from one pinned line and an approval: the line refuses any other server, leaves OpenCode trusting only local names, and its key opens the API and nothing else", async () => {
  const dir = TLS(), home = mkdtempSync(`${RUN}/home-`), bin = mkdtempSync(`${RUN}/bin-`), log = `${home}/calls`, browserKey = "a-browser-key";
  writeFileSync(`${dir}/devices.json`, JSON.stringify([{ id: "b", name: "Phone", hash: createHash("sha256").update(browserKey).digest("hex"), added: 1, seen: 1, kind: "browser" }]));
  writeFileSync(`${home}/corporate.pem`, "corporate roots\n");
  const carries = `${home}/service-carries-ca`, window = `${home}/window`;
  writeFileSync(`${bin}/opencode`, `#!/bin/sh\n[ -z "$BUN_BE_BUN" ] || exec "${process.execPath}" "$@"\necho "$*" >> "${log}"\n[ "$1 $2" != "service get" ] || echo '{ "NODE_EXTRA_CA_CERTS": "${home}/corporate.pem" }'\n[ "$1 $2" != "service restart" ] || touch "${carries}"\n`);
  writeFileSync(`${bin}/ps`, `#!/bin/sh\nif [ "$1" = -axo ]; then echo "/opt/x/opencode.exe serve --service"; [ ! -f "${window}" ] || echo "opencode2"; else\n[ ! -f "${carries}" ] || echo "1 ?? S 0:00 opencode.exe serve --service NODE_EXTRA_CA_CERTS=${home}/.config/chrysalis/opencode-ca.pem"; fi\n`);
  writeFileSync(`${bin}/systemctl`, `#!/bin/sh\necho "$*" >> "${home}/units"\n`);
  chmodSync(`${bin}/systemctl`, 0o755);
  const MACHINE_LABEL = hostname().replace(/\.local$/, "").replace(/[^\w.-]/g, "-").slice(0, 60);
  const realCurl = Bun.which("curl")!, noName = `${home}/no-local-name`;
  writeFileSync(noName, "");
  writeFileSync(`${bin}/curl`, `#!/bin/sh\ncase "$*" in *.local:*) [ ! -f "${noName}" ] || exit 6; exit 0 ;; esac\nexec "${realCurl}" "$@"\n`);
  chmodSync(`${bin}/opencode`, 0o755);
  chmodSync(`${bin}/ps`, 0o755);
  chmodSync(`${bin}/curl`, 0o755);
  const desk = await onNetwork(dir), [here, net] = desk.urls, json = { "Content-Type": "application/json" };
  const paste = (line: string, path = `${bin}:${process.env.PATH}`) => Bun.spawnSync(["sh", "-c", line], { env: { ...process.env, HOME: home, XDG_CONFIG_HOME: "", XDG_STATE_HOME: "", PATH: path }, stdout: "pipe", stderr: "pipe" });
  const calls = () => existsSync(log) ? readFileSync(log, "utf8").trim().split("\n") : [];
  let forward: Bun.Subprocess<"ignore", "pipe", "inherit"> | undefined;
  try {
    const { line } = await (await fetch(`${here}/api/opencode`)).json();
    expect(line.endsWith(`${net}/opencode | sh`)).toBe(true);

    paste(line.replace(/sha256\/\/\S+/, `sha256//${"A".repeat(43)}=`));
    expect(calls()).toEqual([]);
    const missing = paste(line, "/usr/bin:/bin");
    expect([missing.exitCode, String(missing.stderr)]).toEqual([1, expect.stringContaining("OpenCode is not on this computer")]);

    writeFileSync(window, "");
    const held = Bun.spawn(["sh", "-c", line], { env: { ...process.env, HOME: home, XDG_CONFIG_HOME: "", XDG_STATE_HOME: "", PATH: `${bin}:${process.env.PATH}` }, stdout: "ignore", stderr: "pipe" });
    const said = held.stderr.getReader(), text = new TextDecoder();
    expect(text.decode((await said.read()).value)).toContain("Close OpenCode to go on");
    await Bun.sleep(1200);
    expect(calls().filter((c) => c !== "service get env")).toEqual([]);
    rmSync(window);
    let asked = "";
    while (!/Allow \d{6} on /.test(asked)) asked += text.decode((await said.read()).value);
    expect((await fetch(`${here}/api/pair/allow`, { method: "POST", headers: json, body: JSON.stringify({ code: /Allow (\d{6})/.exec(asked)![1] }) })).status).toBe(200);
    void (async () => { while (!(await said.read()).done); })();
    expect(await held.exited).toBe(0);
    const units = () => (existsSync(`${home}/units`) ? readFileSync(`${home}/units`, "utf8").trim().split("\n") : []);
    const unit = readFileSync(`${home}/.config/systemd/user/chrysalis-machine-${MACHINE_LABEL}.service`, "utf8");
    expect([units().slice(-2), /^ExecStart="\S+\/opencode" "(\S+)" "(\S+)" "(\S+)"$/m.exec(unit)!.slice(1), /^Restart=on-failure$/m.test(unit), readFileSync(`${home}/.config/chrysalis/machine-${MACHINE_LABEL}.js`, "utf8")])
      .toEqual([[`--user enable chrysalis-machine-${MACHINE_LABEL}.service`, `--user restart chrysalis-machine-${MACHINE_LABEL}.service`],
        [`${home}/.config/chrysalis/machine-${MACHINE_LABEL}.js`, net, `${home}/.config/chrysalis/key-${MACHINE_LABEL}`], true, readFileSync(`${import.meta.dir}/machine.js`, "utf8")]);
    const waiting = async () => (await (await fetch(`${here}/api/pair`)).json()).waiting.length;
    expect(await waiting()).toBe(0);
    const trust = `${home}/.config/chrysalis/opencode-ca.pem`, restarts = () => calls().filter((c) => c === "service restart").length;
    expect(calls().filter((c) => c === `service set env NODE_EXTRA_CA_CERTS ${trust}`).length).toBe(1);
    expect(calls().at(-1)).toBe(`auth login ${net} --method login`);
    expect(restarts()).toBe(1);
    expect(paste(line).exitCode).toBe(0);
    expect(restarts()).toBe(1);
    rmSync(carries);
    expect(paste(line).exitCode).toBe(0);
    expect(restarts()).toBe(2);
    expect(calls().filter((c) => c.startsWith("auth login")).length).toBe(3);
    rmSync(noName);
    expect(paste(line).exitCode).toBe(0);
    expect(calls().at(-1)).toBe(`auth login https://${netName()}:${new URL(net).port} --method login`);

    const state = `${home}/.local/state/opencode/model.json`, chosen = () => JSON.parse(readFileSync(state, "utf8"));
    const manifestNow = await (await fetch(`${net}/.well-known/opencode`, { tls: { rejectUnauthorized: false } })).json();
    const [providerID, provided]: [string, any] = Object.entries(manifestNow.config.providers)[0] as [string, any];
    const both = Object.keys(provided.models).map((modelID) => ({ providerID, modelID }));
    expect(provided.name).toBe("Chrysalis");
    expect(both.length).toBe(2);
    const alone = `${home}/.config/chrysalis/opencode-${providerID.replace(/^chrysalis-/, "")}`, kept = JSON.parse(/OPENCODE_CONFIG_CONTENT='([^']+)'/.exec(readFileSync(alone, "utf8"))![1]);
    expect([kept.enabled_providers, both.some((m) => `${m.providerID}/${m.modelID}` === kept.model)]).toEqual([[providerID], true]);
    expect(Bun.spawnSync([alone, "run", "two words"], { env: { PATH: `${bin}:${process.env.PATH}`, HOME: home } }).exitCode).toBe(0);
    expect(calls().at(-1)).toBe("--standalone run two words");
    expect(chosen().favorite).toEqual(both);
    const mine = { providerID: "other", modelID: "theirs" };
    writeFileSync(state, JSON.stringify({ recent: [mine], favorite: [mine, both[1]], variant: { "other/theirs": "high" } }));
    expect(paste(line).exitCode).toBe(0);
    expect(chosen()).toEqual({ recent: [mine], favorite: [mine, both[1], both[0]], variant: { "other/theirs": "high" } });
    expect(paste(line).exitCode).toBe(0);
    expect(chosen().favorite.length).toBe(3);

    const mac = mkdtempSync(`${RUN}/mac-`), jobs = `${home}/jobs`, refused = `${home}/launchd-refuses`, onMac = `${mac}:${bin}:${process.env.PATH}`;
    expect(existsSync(jobs)).toBe(false);
    writeFileSync(`${mac}/uname`, "#!/bin/sh\necho Darwin\n");
    writeFileSync(`${mac}/launchctl`, `#!/bin/sh\necho "$*" >> "${jobs}"\n[ "$1" != bootstrap ] || [ ! -f "${refused}" ]\n`);
    chmodSync(`${mac}/uname`, 0o755);
    chmodSync(`${mac}/launchctl`, 0o755);
    writeFileSync(refused, "");
    expect(paste(line, onMac).exitCode).toBe(0);
    expect(calls().at(-1)).toBe(`auth login https://${netName()}:${new URL(net).port} --method login`);
    rmSync(refused);
    expect(paste(line, onMac).exitCode).toBe(0);
    const job = JSON.parse(String(Bun.spawnSync(["python3", "-c", "import json, plistlib, sys; print(json.dumps(plistlib.load(open(sys.argv[1], 'rb'))))",
      readFileSync(jobs, "utf8").trim().split("\n").filter((l) => l.includes("chrysalis-opencode-")).at(-1)!.split(" ").at(-1)!], { stdout: "pipe" }).stdout));
    const listener = job.Sockets.Listeners, local = `https://127.0.0.1:${listener.SockServiceName}`;
    expect([job.ProgramArguments, job.inetdCompatibility, listener.SockNodeName]).toEqual([["/usr/bin/nc", netName(), new URL(net).port], { Wait: false }, "127.0.0.1"]);
    expect(calls().at(-1)).toBe(`auth login ${local} --method login`);
    const helper = JSON.parse(String(Bun.spawnSync(["python3", "-c", "import json, plistlib, sys; print(json.dumps(plistlib.load(open(sys.argv[1], 'rb'))))",
      readFileSync(jobs, "utf8").trim().split("\n").filter((l) => l.includes("chrysalis-machine-")).at(-1)!.split(" ").at(-1)!], { stdout: "pipe" }).stdout));
    expect([helper.ProgramArguments, helper.EnvironmentVariables.BUN_BE_BUN, helper.EnvironmentVariables.NODE_EXTRA_CA_CERTS, helper.KeepAlive, helper.RunAtLoad])
      .toEqual([[`${bin}/opencode`, `${home}/.config/chrysalis/machine-${MACHINE_LABEL}.js`, local, `${home}/.config/chrysalis/key-${MACHINE_LABEL}`], "1", trust, { SuccessfulExit: false }, true]);
    forward = Bun.spawn([process.execPath, "-e", `require("net").createServer((c) => { const far = require("net").connect(${new URL(net).port}, ${JSON.stringify(new URL(net).hostname)}); c.on("error", () => {}); far.on("error", () => {}); c.pipe(far).pipe(c); }).listen(${listener.SockServiceName}, "127.0.0.1", () => console.log("up"))`], { stdout: "pipe" });
    await forward.stdout.getReader().read();

    const heard = await new Promise<{ type: string; data: unknown }[]>((done) => {
      const m = mdns(), got: { type: string; data: unknown }[] = [];
      m.on("response", (r) => got.push(...r.answers.filter((a) => a.name === netName()).map((a) => ({ type: a.type, data: (a as { data: unknown }).data }))));
      m.query({ questions: [{ name: netName(), type: "A" }, { name: netName(), type: "AAAA" }] });
      setTimeout(() => { m.destroy(); done(got); }, 1000);
    });
    const addresses = heard.filter((a) => a.type === "A");
    expect(addresses.length).toBeGreaterThan(0);
    expect(addresses.every((a) => lanAddresses().includes(String(a.data)))).toBe(true);
    expect(heard.filter((a) => a.type === "AAAA")).toEqual([]);
    expect(heard.some((a) => a.type === "NSEC" && (a.data as { rrtypes: string[] }).rrtypes.join() === "A")).toBe(true);
    expect(Bun.spawnSync(["openssl", "x509", "-in", `${dir}/server.pem`, "-noout", "-ext", "subjectAltName"], { stdout: "pipe" }).stdout.toString()).toContain(netName());
    const bundle = readFileSync(trust, "utf8");
    expect([bundle.startsWith("corporate roots\n"), bundle.split("BEGIN CERTIFICATE").length - 1]).toEqual([true, 1]);

    const asBun = (roots: Record<string, string>, at = net) => String(Bun.spawnSync([process.execPath, "-e", `const r = await fetch(${JSON.stringify(`${at}/.well-known/opencode`)}); console.log(r.status)`],
      { env: { ...process.env, NODE_TLS_REJECT_UNAUTHORIZED: "1", ...roots }, stdout: "pipe", stderr: "pipe" }).stdout).trim();
    expect([asBun({ NODE_EXTRA_CA_CERTS: trust }), asBun({})]).toEqual(["200", ""]);
    expect([asBun({ NODE_EXTRA_CA_CERTS: trust }, local), asBun({}, local)]).toEqual(["200", ""]);
    const leaf = `${home}/public`;
    for (const args of [["req", "-new", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-keyout", `${leaf}.key`, "-out", `${leaf}.csr`, "-subj", "/CN=api.example.com", "-addext", "subjectAltName=DNS:api.example.com"],
      ["x509", "-req", "-in", `${leaf}.csr`, "-CA", `${dir}/dns-ca.pem`, "-CAkey", `${dir}/ca.key`, "-CAcreateserial", "-days", "2", "-copy_extensions", "copy", "-out", `${leaf}.pem`]])
      expect(Bun.spawnSync(["openssl", ...args], { stdout: "ignore", stderr: "ignore" }).exitCode).toBe(0);
    expect(String(Bun.spawnSync(["openssl", "verify", "-CAfile", `${dir}/dns-ca.pem`, `${leaf}.pem`], { stdout: "pipe", stderr: "pipe" }).stderr)).toContain("permitted subtree violation");

    const before = readFileSync(`${dir}/server.pem`, "utf8"), trusted = readFileSync(`${dir}/dns-ca.pem`, "utf8");
    writeFileSync(`${dir}/names`, "an address this computer had before");
    webTls(dir);
    expect(readFileSync(`${dir}/server.pem`, "utf8")).not.toBe(before);
    expect(readFileSync(`${dir}/dns-ca.pem`, "utf8")).toBe(trusted);
    expect(Bun.spawnSync(["openssl", "verify", "-CAfile", `${dir}/dns-ca.pem`, `${dir}/server.pem`], { stdout: "ignore", stderr: "ignore" }).exitCode).toBe(0);

    const ca = readFileSync(`${dir}/ca.pem`, "utf8"), manifest = await (await fetch(`${net}/.well-known/opencode`, { tls: { ca } })).json();
    const provider: any = Object.values(manifest.config.providers)[0];
    expect([provider.settings.baseURL, provider.settings.apiKey]).toEqual([`${net}/v1`, `{env:${manifest.auth.env}}`]);
    const forwarded = await (await fetch(`${local}/.well-known/opencode`, { tls: { ca } })).json();
    expect((Object.values(forwarded.config.providers)[0] as any).settings.baseURL).toBe(`${local}/v1`);
    const login = async () => {
      const proc = Bun.spawn(manifest.auth.command, { stdout: "pipe", stderr: "pipe" });
      const code = /Allow (\d{6}) on /.exec(new TextDecoder().decode((await proc.stderr.getReader().read()).value))?.[1];
      expect(code).toBeTruthy();
      return { proc, code };
    };
    const yes = await login();
    expect((await fetch(`${here}/api/pair/allow`, { method: "POST", headers: json, body: JSON.stringify({ code: yes.code }) })).status).toBe(200);
    expect(await yes.proc.exited).toBe(0);
    const key = (await new Response(yes.proc.stdout).text()).trim();
    const chat = (token?: string, at = net) => fetch(`${at}/v1/chat/completions`, { method: "POST", tls: { ca }, headers: { ...json, ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      body: JSON.stringify({ messages: [{ role: "user", content: "hi" }] }) });
    expect([(await chat(key)).status, (await chat(browserKey)).status, (await chat()).status, (await chat("wrong")).status]).toEqual([200, 401, 401, 401]);
    expect([(await chat(key, local)).status, (await chat(undefined, local)).status]).toEqual([200, 401]);
    expect((await fetch(`${net}/api/status`, { tls: { ca }, headers: { Authorization: `Bearer ${key}` } })).status).toBe(401);
    expect((await fetch(`${local}/api/status`, { tls: { ca }, headers: { Authorization: `Bearer ${key}` } })).status).toBe(401);

    const no = await login();
    expect((await fetch(`${here}/api/pair/deny`, { method: "POST", headers: json, body: JSON.stringify({ code: no.code }) })).status).toBe(204);
    expect(await no.proc.exited).toBe(1);
    expect(await new Response(no.proc.stdout).text()).toBe("");
  } finally {
    forward?.kill();
    desk.proc.kill();
  }
});

test("the page hears Chrysalis is off where its address takes a connection and never answers, as a port no one listens on does in WSL's mirrored networking", async () => {
  const mute = Bun.listen({ hostname: "127.0.0.1", port: 0, socket: { data() {} } }), run = mkdtempSync(`${RUN}/run-`);
  const proc = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PORT: "0", CHRYSALIS_URL: `http://127.0.0.1:${mute.port}`, NODE_ENV: "production",
      CHRYSALIS_TLS_DIR: TLS(), XDG_RUNTIME_DIR: run, DBUS_SESSION_BUS_ADDRESS: `unix:path=${run}/bus` },
  });
  try {
    const reader = proc.stdout.getReader(), dec = new TextDecoder();
    let out = "";
    while (!/chrysalis web: \S+/.test(out)) {
      const { value, done } = await reader.read();
      if (done) throw new Error(`server.ts exited: ${out}`);
      out += dec.decode(value);
    }
    const url = /chrysalis web: (\S+)/.exec(out)![1].replace(/\/$/, "");
    expect((await (await fetch(`${url}/api/status`, { signal: AbortSignal.timeout(10_000) })).json()).state).toBe("offline");
  } finally {
    proc.kill();
    mute.stop(true);
  }
});

test("a computer with neither systemd nor openssl, as a Mac is, runs the chat: the page hears Chrysalis is off there and gets the computers found, and its Stop and model picker answer", async () => {
  const free = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response() }), chrysalis = `http://127.0.0.1:${free.port}`;
  free.stop(true);
  const proc = Bun.spawn([process.execPath, "server.ts"], {
    cwd: import.meta.dir, stdout: "pipe", stderr: "inherit",
    env: { ...process.env, PATH: mkdtempSync(`${RUN}/bin-`), PORT: "0", CHRYSALIS_URL: chrysalis, CHRYSALIS_HOST: "127.0.0.1", NODE_ENV: "production",
      CHRYSALIS_TLS_DIR: TLS() },
  });
  try {
    const reader = proc.stdout.getReader(), dec = new TextDecoder();
    let out = "";
    while (!/chrysalis web: \S+/.test(out)) {
      const { value, done } = await reader.read();
      if (done) throw new Error(`server.ts exited: ${out}`);
      out += dec.decode(value);
    }
    const url = /chrysalis web: (\S+)/.exec(out)![1].replace(/\/$/, ""), post = (path: string, body: object) =>
      fetch(`${url}${path}`, { method: "POST", headers: json, body: JSON.stringify(body) });
    const status = await fetch(`${url}/api/status`), seen = status.ok ? await status.json() : {};
    expect([status.status, seen.state, Array.isArray(seen.link?.found)]).toEqual([200, "offline", true]);
    expect((await post("/api/chrysalis/load", { model: "qwen3.8-27b" })).status).toBe(502);
  } finally {
    proc.kill();
  }
});
