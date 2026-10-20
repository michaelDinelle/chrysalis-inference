// Setting a device up for the web chat on the network (`chrysalis start --lan`). The chat is HTTPS there (tls.ts), and a
// device warns about its certificate until it trusts this machine's authority, so a device starts at plain HTTP on a
// port of its own (the QR code of Connect a device and `chrysalis status`): a page that offers the authority as the
// device's system takes it (a profile on Apple's, the certificate elsewhere, Firefox's own prompt), the taps that
// trust it there, and goes on to the chat itself once the device trusts it (it keeps loading an image from the trust
// check, devices.ts: the load succeeds only then). There the device asks to join and shows its own code, which the
// computer's screen shows beside Allow (pairPage): nothing secret crosses plain HTTP.
//
// Over plain HTTP, someone on the network at that moment could serve a page and an authority of their own, so nothing
// this page says can vouch for it: the check is the device's own certificate details against the computer's screen.
import { readFileSync } from "node:fs";

export interface Authority { caDer: Uint8Array; caName: string; fingerprint: string }

const TOKENS = readFileSync(`${import.meta.dir}/tokens.css`, "utf8");
// the aurora the chat's panels wear on their heads (index.html's sprite), for the card's
const INDEX = readFileSync(`${import.meta.dir}/index.html`, "utf8"), SPRITE_AT = INDEX.indexOf('<svg class="sprite"');
const AURORA = INDEX.slice(SPRITE_AT, INDEX.indexOf("</svg>", SPRITE_AT) + "</svg>".length);
if (SPRITE_AT < 0 || !AURORA.includes('id="aurora"')) throw new Error("index.html has no aurora sprite for the setup pages");
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const b64 = (b: Uint8Array) => Buffer.from(b).toString("base64");
// a profile's identifiers from the authority's fingerprint: the same authority is the same profile, so setting a device
// up again replaces it rather than adding another
function uuid(fingerprint: string, k: number) {
  const h = fingerprint.replace(/:/g, "").toLowerCase().slice(k * 32, k * 32 + 32);
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-5${h.slice(13, 16)}-${((parseInt(h[16], 16) & 3) | 8).toString(16)}${h.slice(17, 20)}-${h.slice(20, 32)}`;
}

// Apple's configuration profile: the authority as a root certificate (iOS and iPadOS install it from Safari; macOS from
// System Settings). Unsigned: nothing the device trusts yet could sign it, so it shows as not verified.
export function appleProfile(a: Authority): string {
  const id = `local.chrysalis.${uuid(a.fingerprint, 0)}`;
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>PayloadContent</key>
  <array>
    <dict>
      <key>PayloadCertificateFileName</key><string>chrysalis-ca.crt</string>
      <key>PayloadContent</key><data>${b64(a.caDer)}</data>
      <key>PayloadDescription</key><string>Lets this device trust the web chat of ${esc(a.caName)} on your network. It vouches only for private addresses and local names.</string>
      <key>PayloadDisplayName</key><string>${esc(a.caName)}</string>
      <key>PayloadIdentifier</key><string>${id}.ca</string>
      <key>PayloadType</key><string>com.apple.security.root</string>
      <key>PayloadUUID</key><string>${uuid(a.fingerprint, 1)}</string>
      <key>PayloadVersion</key><integer>1</integer>
    </dict>
  </array>
  <key>PayloadDescription</key><string>Trust for ${esc(a.caName)}: its web chat over HTTPS on your local network.</string>
  <key>PayloadDisplayName</key><string>${esc(a.caName)}</string>
  <key>PayloadIdentifier</key><string>${id}</string>
  <key>PayloadRemovalDisallowed</key><false/>
  <key>PayloadType</key><string>Configuration</string>
  <key>PayloadUUID</key><string>${uuid(a.fingerprint, 0)}</string>
  <key>PayloadVersion</key><integer>1</integer>
</dict>
</plist>
`;
}

// A 1x1 transparent GIF: the image the setup page loads from the chat's HTTPS address to see whether the device trusts it
export const TRUSTED_GIF = Uint8Array.from(atob("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"), (c) => c.charCodeAt(0));

// The device as both pages tell it: its system (the guide it gets) and, for the list of devices let in, a name
const DETECT = `const ua = navigator.userAgent, touchMac = /Macintosh/.test(ua) && navigator.maxTouchPoints > 1;
const os = /iPhone|iPad|iPod/.test(ua) || touchMac ? "ios" : /Android/.test(ua) ? "android" : /CrOS/.test(ua) ? "chromeos"
  : /Macintosh/.test(ua) ? "mac" : /Windows/.test(ua) ? "windows" : "linux";
const browser = /Edg/.test(ua) ? "Edge" : /Firefox\\/|FxiOS/.test(ua) ? "Firefox" : /CriOS|Chrome\\//.test(ua) ? "Chrome" : /Safari\\//.test(ua) ? "Safari" : "";
const NAME = { ios: /iPad/.test(ua) || touchMac ? "iPad" : "iPhone", android: "Android", chromeos: "Chromebook", mac: "Mac", windows: "Windows", linux: "Linux" }[os]
  + (browser ? " · " + browser : "");
const $ = (id) => document.getElementById(id);`;

// The pages' look is the chat's (tokens.css): its sea of characters fills the window (device.ts, whose built script each
// page carries: `sea`), and the page is a card floating on it, as the message box does, with the butterfly above it and
// a night sky over the page's top: the aurora the chat's panels wear, fading into the sea.
const page = (body: string, script: string, sea: string) => `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>Connect to Chrysalis</title>
<style>
${TOKENS}
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 16px/1.55 var(--sans); -webkit-font-smoothing: antialiased; }
  .sea { position: fixed; inset: 0; overflow: hidden; }
  .sea > .ocean { position: absolute; touch-action: manipulation; }
  main { position: relative; display: flex; flex-direction: column; align-items: center; min-height: 100dvh;
    padding: 0 16px max(28px, env(safe-area-inset-bottom)); pointer-events: none; }
  /* the whole art across a wide window (its own proportions), its middle on a phone's, under the status bar too */
  .sky { position: relative; display: grid; place-items: center; align-self: stretch; margin: 0 -16px; box-sizing: content-box;
    height: clamp(68px, 100vw * 51 / 640, 120px); padding-top: env(safe-area-inset-top); }
  .sky > .aurora { position: absolute; inset: 0; width: 100%; height: 100%; mask-image: linear-gradient(#000 45%, transparent); }
  .sky > .brand { position: relative; color: #f6edf1; font-size: 16px; font-weight: 600; letter-spacing: 0.015em;
    text-shadow: 0 0 2px var(--aurora-sky), 0 0 10px var(--aurora-sky); }
  .stage { display: grid; flex: none; place-items: center; width: 100%; height: clamp(200px, 38vh, 440px); }
  .stage > .logo { width: min(78%, 320px); filter: drop-shadow(0 0 40px rgb(226 77 133 / 0.14)); }
  .card { width: 100%; max-width: 520px; padding: 22px 20px 18px; border: 1px solid var(--line); border-radius: 20px; background: var(--raised);
    box-shadow: var(--shadow); pointer-events: auto; }
  .sprite { position: absolute; width: 0; height: 0; }
  h1 { margin: 0; font-size: 22px; font-weight: 650; text-align: center; letter-spacing: -0.01em; }
  .sub { margin: 4px 0 18px; color: var(--muted); font-size: 15px; text-align: center; }
  .status { display: flex; align-items: center; justify-content: center; gap: 10px; margin: 0 0 18px; padding: 10px 14px; border: 1px solid var(--line);
    border-radius: 12px; background: var(--panel); color: var(--muted); font-size: 14.5px; }
  .status::before { content: ""; flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--faint); animation: pulse 1.4s ease-in-out infinite; }
  .status.ok { color: var(--text); border-color: color-mix(in srgb, var(--ok) 50%, var(--line)); }
  .status.ok::before { background: var(--ok); animation: none; }
  .status.bad { color: var(--text); border-color: color-mix(in srgb, var(--accent) 50%, var(--line)); }
  .status.bad::before { background: var(--accent); animation: none; }
  .status[hidden] { display: none; }
  @keyframes pulse { 50% { opacity: 0.35; } }
  @media (prefers-reduced-motion: reduce) { .status::before { animation: none; } }
  .go { display: block; width: 100%; padding: 14px 18px; border: 0; border-radius: 12px; background: var(--accent); color: #fff; font-family: inherit;
    font-size: 16.5px; font-weight: 600; line-height: 1.2; text-align: center; text-decoration: none; cursor: pointer; }
  .go:hover { filter: brightness(1.06); }
  .go:active { filter: brightness(0.92); }
  .code { margin: 0 0 18px; color: var(--text); font: 600 40px/1.1 var(--mono); letter-spacing: 0.08em; text-align: center; }
  ol { margin: 20px 0 0; padding: 0; list-style: none; counter-reset: step; }
  li { position: relative; margin: 0 0 14px; padding-left: 38px; counter-increment: step; }
  li::before { content: counter(step); position: absolute; left: 0; top: 0; display: grid; place-items: center; width: 26px; height: 26px; border-radius: 50%;
    background: var(--panel); border: 1px solid var(--line); color: var(--accent); font-size: 13.5px; font-weight: 650; }
  li:has(.aside) { counter-increment: none; }
  li:has(.aside)::before { content: none; }
  .aside { color: var(--muted); font-size: 14px; }
  p.aside { margin: 16px 0 0; text-align: center; }
  code { padding: 1px 5px; border-radius: 6px; background: var(--code); font: 13px var(--mono); overflow-wrap: anywhere; }
  .warn { margin: 0 0 16px; padding: 12px 14px; border-radius: 12px; background: var(--accent-soft); font-size: 15px; }
  .warn button { margin-top: 8px; padding: 8px 12px; border: 1px solid var(--line); border-radius: 9px; background: var(--raised); color: var(--text); font: inherit; }
  details { margin-top: 20px; border-top: 1px solid var(--line); padding-top: 14px; color: var(--muted); font-size: 14.5px; }
  summary { cursor: pointer; }
  .others { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .others button { padding: 6px 12px; border: 1px solid var(--line); border-radius: 999px; background: var(--raised); color: var(--text); font: inherit; font-size: 14px; cursor: pointer; }
  .others button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent-hi); background: var(--accent-soft); }
  .about { margin: 12px 0 0; }
</style>
</head>
<body>
${AURORA}
<div id="sea" class="sea"></div>
<main>
  <header class="sky">
    <svg class="aurora" aria-hidden="true" viewBox="0 0 640 51" preserveAspectRatio="xMidYMax slice"><use href="#aurora" /></svg>
    <span class="brand">Chrysalis</span>
  </header>
  <div class="stage" id="stage"></div>
  <section class="card">
    <h1>Connect to Chrysalis</h1>
${body}
  </section>
</main>
<script>
${DETECT}
${script}
</script>
<script type="module">${sea}</script>
</body>
</html>
`;

// device.ts as a script of its own, its drawing in it as a data: URL: the pages are served whole, from a port of their own
export async function deviceScript(): Promise<string> {
  const built = await Bun.build({ entrypoints: [`${import.meta.dir}/device.ts`], target: "browser", minify: true, plugins: [{ name: "svg as a data: URL",
    setup(b) {
      b.onLoad({ filter: /\.svg$/ }, async (f) => ({ loader: "js",
        contents: `export default ${JSON.stringify(`data:image/svg+xml;base64,${Buffer.from(await Bun.file(f.path).arrayBuffer()).toString("base64")}`)};` }));
    } }] });
  if (!built.success) throw new AggregateError(built.logs, "device.ts did not build");
  return (await built.outputs[0].text()).replaceAll("</script", "<\\/script");
}

// The page: the guide for the device it detects (any other one a tap away), its download, and the check that goes on
// to the chat. chat: the chat's HTTPS address at the name or address the device used; check: the trust check's image.
export function setupPage(a: Authority, chat: string, check: string, sea: string): string {
  const name = esc(a.caName), guides = {
    ios: { title: "iPhone and iPad", button: ["Download the profile", "/chrysalis.mobileconfig"], steps: [
      "Tap <b>Download the profile</b>, then <b>Allow</b>.",
      "Open <b>Settings</b>, tap <b>Profile Downloaded</b> near the top, then <b>Install</b> and enter your passcode.",
      `In <b>Settings → General → About → Certificate Trust Settings</b>, turn on <b>${name}</b>.`,
      "Come back here: Chrysalis opens by itself."] },
    android: { title: "Android", button: ["Download the certificate", "/chrysalis-ca.crt"], steps: [
      "Tap <b>Download the certificate</b>.",
      "Open <b>Settings</b> and search for <b>CA certificate</b>. Tap it, then <b>Install anyway</b>, and pick <b>chrysalis-ca.crt</b> from Downloads.",
      "Come back here: Chrysalis opens by itself.",
      "<span class=\"aside\">Firefox: also turn on <b>Use third party CA certificates</b> (Settings → About Firefox, tap the logo five times, then Secret settings).</span>"] },
    mac: { title: "Mac", button: ["Download the profile", "/chrysalis.mobileconfig"], steps: [
      "Click <b>Download the profile</b> (and open it, if the browser only saves it).",
      "Open <b>System Settings</b>, click <b>Profile Downloaded</b> (or General → Device Management), then <b>Install</b>, and enter your password.",
      "Come back here: Chrysalis opens by itself.",
      `<span class="aside">Still waiting? In Keychain Access, open <b>${name}</b> and set <b>Trust → When using this certificate</b> to <b>Always Trust</b>.</span>`] },
    windows: { title: "Windows", button: ["Download the certificate", "/chrysalis-ca.crt"], steps: [
      "Click <b>Download the certificate</b> and open it.",
      "Click <b>Install Certificate…</b>, choose <b>Current User</b>, then <b>Place all certificates in the following store → Browse… → Trusted Root Certification Authorities</b>.",
      "Click <b>Next</b>, <b>Finish</b>, and <b>Yes</b> to the warning.",
      "Come back here: Chrysalis opens by itself (or restart the browser if it does not)."] },
    linux: { title: "Linux", button: ["Download the certificate", "/chrysalis-ca.crt"], steps: [
      "Click <b>Download the certificate</b>.",
      "Chrome or Chromium: open <code>chrome://certificate-manager</code>, then <b>Local certificates → Installed by you → Trusted certificates → Import</b>, and pick <b>chrysalis-ca.crt</b>.",
      "Come back here: Chrysalis opens by itself.",
      "<span class=\"aside\">For the whole system (curl and others): <code>sudo trust anchor --store chrysalis-ca.crt</code>, or on Debian and Ubuntu copy it to <code>/usr/local/share/ca-certificates/</code> and run <code>sudo update-ca-certificates</code>.</span>"] },
    chromeos: { title: "ChromeOS", button: ["Download the certificate", "/chrysalis-ca.crt"], steps: [
      "Click <b>Download the certificate</b>.",
      "Open <b>Settings → Privacy and security → Security → Manage certificates → Authorities → Import</b>, pick <b>chrysalis-ca.crt</b> and tick <b>Trust this certificate for identifying websites</b>.",
      "Come back here: Chrysalis opens by itself."] },
    firefox: { title: "Firefox", button: ["Trust in Firefox", "/chrysalis-ca-firefox.crt"], steps: [
      "Click <b>Trust in Firefox</b>.",
      "Tick <b>Trust this CA to identify websites</b>, then <b>OK</b>.",
      "Chrysalis opens by itself."] },
  };
  return page(`  <p class="sub">Set this device up once, and Chrysalis opens securely on it from then on.</p>
  <div class="status" id="status" role="status" aria-live="polite">Checking the secure connection…</div>
  <div id="guide"></div>
  <details>
    <summary>Another device, or what this does</summary>
    <div class="others" id="others"></div>
    <p class="about">This device will trust <b>${name}</b>, an authority this computer made for its own web chat. It can only vouch
      for private addresses and local names, so it changes nothing for any other site. This page came over plain HTTP, so
      someone else on the network could have sent one of their own: to be sure, compare the SHA-256 fingerprint this device
      shows for <b>${name}</b> in its certificate details with the one under <b>Connect a device</b> on the computer.</p>
  </details>`, `const CHAT = ${JSON.stringify(chat)}, CHECK = ${JSON.stringify(check)}, GUIDES = ${JSON.stringify(guides)};
// desktop Firefox keeps an authority list of its own, and asks itself; on phones it goes by the system's
const detected = /Firefox\\//.test(ua) && os !== "ios" && os !== "android" ? "firefox" : os;
function show(key) {
  const g = GUIDES[key], guide = $("guide");
  guide.innerHTML = "";
  // a profile installs from Safari only
  if (key === "ios" && os === "ios" && /CriOS|FxiOS|EdgiOS|OPiOS/.test(ua)) {
    const w = document.createElement("div");
    w.className = "warn";
    w.innerHTML = "Open this page in <b>Safari</b>: iPhone and iPad install profiles from Safari only.<br><button type=\\"button\\">Copy this page's address</button>";
    w.querySelector("button").onclick = (e) => navigator.clipboard?.writeText(location.href).then(() => { e.target.textContent = "Copied"; });
    guide.append(w);
  }
  const go = document.createElement("a");
  go.className = "go";
  go.textContent = g.button[0];
  go.href = g.button[1];
  const steps = document.createElement("ol");
  steps.innerHTML = g.steps.map((s) => "<li>" + s + "</li>").join("");
  guide.append(go, steps);
  for (const b of $("others").children) b.setAttribute("aria-pressed", String(b.dataset.key === key));
}
for (const [key, g] of Object.entries(GUIDES)) {
  const b = document.createElement("button");
  b.type = "button";
  b.dataset.key = key;
  b.textContent = g.title;
  b.onclick = () => show(key);
  $("others").append(b);
}
show(detected);
// The device trusts Chrysalis once the trust check's image loads: then the chat's page that lets it in opens. Checked
// again at once when the page comes back into view (after Settings).
let checking = false, timer = 0;
function probe() {
  return new Promise((done) => {
    const img = new Image();
    img.onload = () => done(true);
    img.onerror = () => done(false);
    img.src = CHECK + "?t=" + Date.now();
  });
}
async function check() {
  if (checking) return;
  checking = true;
  clearTimeout(timer);
  const ok = await probe();
  checking = false;
  const status = $("status");
  if (ok) {
    status.classList.add("ok");
    status.textContent = "Trusted. Opening Chrysalis…";
    document.dispatchEvent(new Event("chrysalis-opening"));
    setTimeout(() => location.replace(CHAT + "pair"), 900);
    return;
  }
  status.textContent = "Waiting for this device to trust Chrysalis…";
  timer = setTimeout(check, 1500);
}
document.addEventListener("visibilitychange", () => { if (!document.hidden) check(); });
check();`, sea);
}

// The chat's page for a browser it has not let in, or that has not shown lately it trusts Chrysalis (server.ts /pair).
// It checks first, through the trust check's own certificate. Trusted, it asks to join and shows the device's code, which
// Connect a device on the computer shows beside Allow, and waits; not trusted (a warning clicked past included), it sends
// the browser to set the device up. check: the trust check's image; setup: the setup page.
export function pairPage(sea: string, check: string, setup: string): string {
  return page(`  <p class="sub" id="sub">Checking that this device trusts Chrysalis…</p>
  <div class="status" id="status" role="status" aria-live="polite">Checking the secure connection…</div>
  <div class="code" id="code" hidden></div>
  <div id="next"></div>`, `const CHECK = ${JSON.stringify(check)}, SETUP = ${JSON.stringify(setup)};
const status = $("status");
function say(text, kind) {
  status.className = "status" + (kind ? " " + kind : "");
  status.textContent = text;
}
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
// the image loads only where this browser trusts Chrysalis's authority: a warning clicked past covers the chat's
// certificate alone, and an image cannot show one
function trusts(nonce) {
  return new Promise((done) => {
    const img = new Image();
    img.onload = () => done(true);
    img.onerror = () => done(false);
    img.src = CHECK + "?n=" + nonce + "&t=" + Date.now();
  });
}
function open() {
  say("Allowed. Opening Chrysalis…", "ok");
  document.dispatchEvent(new Event("chrysalis-opening"));
  setTimeout(() => location.replace("/"), 900);
}
function untrusted() {
  $("sub").textContent = "This device does not trust Chrysalis yet, so its connection is not verified.";
  say("Not secure: set this device up first", "bad");
  const go = document.createElement("a"), aside = document.createElement("p");
  go.className = "go";
  go.href = SETUP;
  go.textContent = "Set this device up";
  aside.className = "aside";
  aside.textContent = "A warning clicked past does not count: the device itself has to trust Chrysalis. The setup page shows how, then comes back here.";
  $("next").replaceChildren(go, aside);
}
async function start() {
  $("code").hidden = true;
  $("next").replaceChildren();
  say("Checking the secure connection…");
  const nonce = Array.from(crypto.getRandomValues(new Uint8Array(18)), (b) => b.toString(16).padStart(2, "0")).join("");
  if (!(await trusts(nonce))) return untrusted();
  const r = await fetch("/pair/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ nonce, name: NAME }) })
    .then((x) => x.json()).catch(() => null);
  if (r?.in) return open();
  if (!r?.code) return untrusted();
  $("sub").innerHTML = "Allow this device on the computer running Chrysalis: <b>Connect a device</b>, the phone at the top of the chat, shows this code.";
  $("code").textContent = r.code.slice(0, 3) + " " + r.code.slice(3);
  $("code").hidden = false;
  say("Waiting for you to allow it…");
  const aside = document.createElement("p"), cmd = document.createElement("code");
  aside.className = "aside";
  cmd.textContent = "chrysalis allow " + r.code;
  aside.append("In a terminal there: ", cmd);
  $("next").replaceChildren(aside);
  for (;;) {
    await sleep(1500);
    const got = await fetch("/pair/ask/" + r.secret).then((x) => x.json()).catch(() => null);
    if (got?.state === "allowed") return open();
    if (got?.state === "gone") {
      $("code").hidden = true;
      say("Not allowed, or it waited too long", "bad");
      const again = document.createElement("button");
      again.className = "go";
      again.type = "button";
      again.textContent = "Ask again";
      again.onclick = start;
      return void $("next").replaceChildren(again);
    }
  }
}
start();`, sea);
}
