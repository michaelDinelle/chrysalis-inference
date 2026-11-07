import { readFileSync } from "node:fs";

export interface Authority { caDer: Uint8Array; caName: string; fingerprint: string }

const TOKENS = readFileSync(`${import.meta.dir}/tokens.css`, "utf8");
const INDEX = readFileSync(`${import.meta.dir}/index.html`, "utf8"), SPRITE_AT = INDEX.indexOf('<svg class="sprite"');
const AURORA = INDEX.slice(SPRITE_AT, INDEX.indexOf("</svg>", SPRITE_AT) + "</svg>".length);
if (SPRITE_AT < 0 || !AURORA.includes('id="aurora"')) throw new Error("index.html has no aurora sprite for the connect page");
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const b64 = (b: Uint8Array) => Buffer.from(b).toString("base64");
function uuid(fingerprint: string, k: number) {
  const h = fingerprint.replace(/:/g, "").toLowerCase().slice(k * 32, k * 32 + 32);
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-5${h.slice(13, 16)}-${((parseInt(h[16], 16) & 3) | 8).toString(16)}${h.slice(17, 20)}-${h.slice(20, 32)}`;
}

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

export const TRUSTED_GIF = Uint8Array.from(atob("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"), (c) => c.charCodeAt(0));

const DETECT = `const ua = navigator.userAgent, touchMac = /Macintosh/.test(ua) && navigator.maxTouchPoints > 1;
const os = /iPhone|iPad|iPod/.test(ua) || touchMac ? "ios" : /Android/.test(ua) ? "android" : /CrOS/.test(ua) ? "chromeos"
  : /Macintosh/.test(ua) ? "mac" : /Windows/.test(ua) ? "windows" : "linux";
const browser = /Edg/.test(ua) ? "Edge" : /Firefox\\/|FxiOS/.test(ua) ? "Firefox" : /CriOS|Chrome\\//.test(ua) ? "Chrome" : /Safari\\//.test(ua) ? "Safari" : "";
const NAME = { ios: /iPad/.test(ua) || touchMac ? "iPad" : "iPhone", android: "Android", chromeos: "Chromebook", mac: "Mac", windows: "Windows", linux: "Linux" }[os]
  + (browser ? " · " + browser : "");
const $ = (id) => document.getElementById(id);`;

const page = (body: string, script: string, sea: string) => `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="dark">
<title>Connect to Chrysalis</title>
<style>
${TOKENS}
  * { box-sizing: border-box; }
  html, body { overscroll-behavior: none; background: var(--bg); }
  body { margin: 0; color: var(--text); font: 16px/1.55 var(--sans); -webkit-font-smoothing: antialiased; }
  .sea { position: fixed; inset: 0; overflow: hidden; }
  .sea > .ocean { position: absolute; touch-action: manipulation; }
  main { position: relative; display: flex; flex-direction: column; align-items: center; min-height: 100dvh;
    padding: 0 16px max(28px, env(safe-area-inset-bottom)); pointer-events: none; }
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
  @keyframes pulse { 50% { opacity: 0.35; } }
  @media (prefers-reduced-motion: reduce) { .status::before { animation: none; } }
  .go { display: block; width: 100%; padding: 14px 18px; border: 0; border-radius: 12px; background: var(--accent); color: #fff; font-family: inherit;
    font-size: 16.5px; font-weight: 600; line-height: 1.2; text-align: center; text-decoration: none; cursor: pointer; }
  .go:hover { filter: brightness(1.06); }
  .go:active { filter: brightness(0.92); }
  .code { margin: 0 0 18px; color: var(--text); font: 600 40px/1.1 var(--mono); letter-spacing: 0.08em; text-align: center; }
  .now { margin: 0 0 16px; font-size: 17px; line-height: 1.5; text-align: center; }
  .copyable { display: flex; align-items: center; gap: 8px; margin-top: 12px; padding: 6px 6px 6px 12px; border: 1px solid var(--line); border-radius: 12px;
    background: var(--panel); text-align: left; }
  .copyable code { flex: 1; padding: 0; background: none; color: var(--text); font-size: 13.5px; }
  .copy { flex: none; padding: 8px 16px; border: 0; border-radius: 9px; background: var(--accent); color: #fff; font: inherit; font-size: 14.5px; font-weight: 600;
    cursor: pointer; }
  .steps { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; margin-top: 12px; color: var(--muted); font-size: 14px; }
  .steps button { padding: 6px 2px; border: 0; background: none; color: var(--accent-hi); font: inherit; cursor: pointer; }
  .steps button[hidden] { display: block; visibility: hidden; }
  #back { justify-self: start; }
  #on { justify-self: end; }
  [hidden] { display: none; }
  .aside { color: var(--muted); font-size: 14px; }
  div.aside { margin: 16px 0 0; text-align: center; }
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

export async function deviceScript(): Promise<string> {
  const built = await Bun.build({ entrypoints: [`${import.meta.dir}/device.ts`], target: "browser", minify: true, plugins: [{ name: "svg as a data: URL",
    setup(b) {
      b.onLoad({ filter: /\.svg$/ }, async (f) => ({ loader: "js",
        contents: `export default ${JSON.stringify(`data:image/svg+xml;base64,${Buffer.from(await Bun.file(f.path).arrayBuffer()).toString("base64")}`)};` }));
    } }] });
  if (!built.success) throw new AggregateError(built.logs, "device.ts did not build");
  return (await built.outputs[0].text()).replaceAll("</script", "<\\/script");
}

export function connectPage(a: Authority, check: string, sea: string): string {
  const name = esc(a.caName);
  const copyable = (text: string) => `<span class="copyable"><code>${esc(text)}</code><button type="button" class="copy">Copy</button></span>`;
  const guides = {
    ios: { title: "iPhone and iPad", button: ["Download", "/chrysalis.mobileconfig"], steps: [
      ["Download", "Then tap <b>Allow</b>."],
      ["Install", "Open <b>Settings</b>, tap <b>Profile Downloaded</b>, then <b>Install</b>."],
      ["Trust", `In <b>Settings → General → About → Certificate Trust Settings</b>, turn on <b>${name}</b>. Then come back here.`]] },
    android: { title: "Android", button: ["Download", "/chrysalis-ca.crt"], steps: [
      ["Download", ""],
      ["Trust", "Open <b>Settings</b>, search for <b>CA certificate</b>, tap <b>Install anyway</b> and pick <b>chrysalis-ca.crt</b>. Then come back here."]],
      firefox: ["Trust in Firefox", "In Firefox, <b>Settings → About Firefox</b>, tap the logo five times, then in <b>Secret settings</b> turn on <b>Use third party CA certificates</b>."] },
    mac: { title: "Mac", button: ["Download", "/chrysalis.mobileconfig"], steps: [
      ["Download", "Then open the file, if the browser only saves it."],
      ["Install", "Open <b>System Settings</b>, click <b>Profile Downloaded</b> (or <b>General → Device Management</b>), then <b>Install</b>."],
      ["Trust", `Open <b>Keychain Access</b>, double-click <b>${name}</b> and set <b>Trust</b> to <b>Always Trust</b>. Then come back here.`]] },
    windows: { title: "Windows", button: ["Download", "/chrysalis-ca.crt"], steps: [
      ["Download", "Then open the file."],
      ["Trust", "Click <b>Install Certificate…</b> and <b>Next</b>, choose <b>Place all certificates in the following store → Browse… → Trusted Root Certification Authorities</b>, then <b>Next</b>, <b>Finish</b> and <b>Yes</b>."]] },
    linux: { title: "Linux", button: ["Download", "/chrysalis-ca.crt"], steps: [
      ["Download", ""],
      ["Trust", `Open this address in a new tab, click <b>Import</b> beside <b>Trusted Certificates</b> and pick <b>chrysalis-ca.crt</b>.${copyable("chrome://certificate-manager/localcerts/usercerts")}`]] },
    chromeos: { title: "ChromeOS", button: ["Download", "/chrysalis-ca.crt"], steps: [
      ["Download", ""],
      ["Trust", "In <b>Settings → Privacy and security → Security → Manage certificates → Authorities</b>, click <b>Import</b>, pick <b>chrysalis-ca.crt</b> and tick <b>Trust this certificate for identifying websites</b>."]] },
    firefox: { title: "Firefox", button: ["Trust in Firefox", "/chrysalis-ca-firefox.crt"], steps: [
      ["Open", ""],
      ["Trust", "Tick <b>Trust this CA to identify websites</b>, then <b>OK</b>."]] },
  };
  return page(`  <p class="sub" id="sub">Checking this device…</p>
  <div class="status" id="status" role="status" aria-live="polite">Checking the secure connection…</div>
  <div class="code" id="code" hidden></div>
  <div id="setup" hidden>
    <div class="warn" id="safari" hidden>Open this page in <b>Safari</b>: iPhone and iPad install profiles from Safari only.<br><button type="button">Copy this page's address</button></div>
    <div class="now" id="now" aria-live="polite"></div>
    <a class="go" id="go"></a>
    <div class="steps"><button type="button" id="back">‹ Back</button><span id="count"></span><button type="button" id="on">Next ›</button></div>
  </div>
  <div id="after"></div>
  <details id="more" hidden>
    <summary>Another device, or what this does</summary>
    <div class="others" id="others"></div>
    <p class="about">This device will trust <b>${name}</b>, an authority this computer made for its own web chat. It can only vouch
      for private addresses and local names, so it changes nothing for any other site. Until this device trusts it, nothing
      proves this page came from your computer (that is what the browser's warning said): to be sure, compare the SHA-256
      fingerprint this device shows for <b>${name}</b> in its certificate details with the one under <b>Connect a device</b>
      on the computer.</p>
  </details>`, `const CHECK = ${JSON.stringify(check)}, GUIDES = ${JSON.stringify(guides)};
const detected = /Firefox\\//.test(ua) && os !== "ios" && os !== "android" ? "firefox" : os;
if (os === "android" && browser === "Firefox") GUIDES.android.steps.push(GUIDES.android.firefox);
const status = $("status");
function say(text, kind) {
  status.className = "status" + (kind ? " " + kind : "");
  status.textContent = text;
}
function copied(button, text) {
  const done = () => { button.textContent = "Copied"; setTimeout(() => { button.textContent = "Copy"; }, 2000); };
  navigator.clipboard?.writeText(text).then(done, () => {});
}
function copies(root) {
  for (const b of root.querySelectorAll(".copy")) b.onclick = () => {
    getSelection().selectAllChildren(b.previousElementSibling);
    copied(b, b.previousElementSibling.textContent);
  };
}
let guide = GUIDES[detected], at = 0, returned = false;
function render() {
  const n = guide.steps.length, now = $("now"), go = $("go");
  now.innerHTML = guide.steps[at][1];
  now.hidden = !guide.steps[at][1];
  copies(now);
  go.hidden = at > 0;
  go.textContent = guide.button[0];
  go.href = guide.button[1];
  $("back").hidden = at === 0;
  $("on").hidden = at === n - 1;
  $("count").textContent = "Step " + (at + 1) + " of " + n + " · " + guide.steps[at][0];
  $("safari").hidden = !(guide === GUIDES.ios && os === "ios" && /CriOS|FxiOS|EdgiOS|OPiOS/.test(ua));
  for (const b of $("others").children) b.setAttribute("aria-pressed", String(GUIDES[b.dataset.key] === guide));
}
const move = (by) => { at = Math.max(0, Math.min(guide.steps.length - 1, at + by)); render(); };
$("back").onclick = () => move(-1);
$("on").onclick = () => move(1);
$("go").onclick = () => setTimeout(() => { if (at === 0) move(1); }, 500);
$("safari").querySelector("button").onclick = (e) => copied(e.target, location.href);
for (const [key, g] of Object.entries(GUIDES)) {
  const b = document.createElement("button");
  b.type = "button";
  b.dataset.key = key;
  b.textContent = g.title;
  b.onclick = () => { guide = g; at = 0; render(); };
  $("others").append(b);
}
render();
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
function trusts(nonce) {
  return new Promise((done) => {
    const img = new Image(), late = setTimeout(() => done(null), 6000);
    img.onload = () => (clearTimeout(late), done(true));
    img.onerror = () => (clearTimeout(late), done(false));
    img.src = CHECK + "?n=" + nonce + "&t=" + Date.now();
  });
}
function setUp(on) {
  $("setup").hidden = !on;
  $("more").hidden = !on;
  status.hidden = on;
  if (on) $("sub").textContent = "Set this device up once, and it opens Chrysalis from then on.";
}
function opening(text) {
  say(text, "ok");
  document.dispatchEvent(new Event("chrysalis-opening"));
  setTimeout(() => location.replace("/"), 900);
}
let asking = false, timer = 0;
async function check() {
  if (asking) return;
  asking = true;
  clearTimeout(timer);
  const nonce = Array.from(crypto.getRandomValues(new Uint8Array(18)), (b) => b.toString(16).padStart(2, "0")).join("");
  const trusted = await trusts(nonce);
  if (trusted === null) {
    asking = false;
    setUp(false);
    $("sub").innerHTML = "This device cannot reach Chrysalis's trust check (port " + new URL(CHECK).port + "). On the computer running Chrysalis, "
      + "<code>chrysalis.sh --host 0.0.0.0</code> lets it through the firewall.";
    say("Cannot reach this computer", "bad");
    timer = setTimeout(check, 1500);
    return;
  }
  if (!trusted) {
    asking = false;
    setUp(true);
    if (returned && at > 0) move(1);
    returned = false;
    timer = setTimeout(check, 1500);
    return;
  }
  setUp(false);
  const r = await fetch("/pair/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ nonce, name: NAME }) })
    .then((x) => x.json()).catch(() => null);
  if (r?.in) return opening("Trusted. Opening Chrysalis…");
  if (!r?.code) return again("Could not ask to join");
  $("sub").innerHTML = "Allow this device on your computer: Chrysalis's terminal asks with this code, and <b>Connect a device</b>, the phone at the top of the chat, shows it.";
  $("code").textContent = r.code.slice(0, 3) + " " + r.code.slice(3);
  $("code").hidden = false;
  say("Waiting for you to allow it…");
  $("after").replaceChildren();
  for (;;) {
    await sleep(1500);
    const got = await fetch("/pair/ask/" + r.secret).then((x) => x.json()).catch(() => null);
    if (got?.state === "allowed") return opening("Allowed. Opening Chrysalis…");
    if (got?.state === "gone") return again("Not allowed, or it waited too long");
  }
}
function again(why) {
  $("code").hidden = true;
  say(why, "bad");
  const b = document.createElement("button");
  b.className = "go";
  b.type = "button";
  b.textContent = "Ask again";
  b.onclick = () => { $("after").replaceChildren(); asking = false; check(); };
  $("after").replaceChildren(b);
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden) return;
  returned = true;
  if (!asking) check();
});
check();`, sea);
}
