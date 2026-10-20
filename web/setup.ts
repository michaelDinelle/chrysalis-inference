// Setting a device up for the web chat on the network (`chrysalis start --lan`). The chat is HTTPS there (tls.ts), and a
// device warns about its certificate until it trusts this machine's authority, so a device starts at plain HTTP on a
// port of its own (the QR code chrysalis.sh prints, Settings' Other devices): a page that offers the authority as the
// device's system takes it (a profile on Apple's, the certificate elsewhere, Firefox's own prompt), the taps that
// trust it there, and opens the chat itself once the device trusts it (it keeps loading an image from the chat's HTTPS
// address: the load succeeds only then). The page serves the authority's certificate and nothing else. Over plain HTTP
// it can be swapped by someone on the network at that moment: the authority's fingerprint is on this page and in
// `chrysalis status`, and a device's certificate details show it, for anyone who wants to compare.
import { readFileSync } from "node:fs";

export interface Authority { caDer: Uint8Array; caName: string; fingerprint: string }

const LOGO = readFileSync(`${import.meta.dir}/logo-thin.svg`, "utf8").replace(/<!--[\s\S]*?-->/g, "");
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

// The page: the guide for the device it detects (any other one a tap away), its download, and the check that opens the
// chat. chat: the chat's HTTPS address at the name or address the device used.
export function setupPage(a: Authority, chat: string): string {
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
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>Connect to Chrysalis</title>
<style>
  :root { --bg: #121010; --panel: #1a1617; --line: #2d2628; --text: #efe4e7; --muted: #9c8e92; --faint: #6b5f62; --accent: #e24d85; --ok: #6ccf8e; }
  @media (prefers-color-scheme: light) { :root { --bg: #fcf8f9; --panel: #ffffff; --line: #e5d6da; --text: #1d181a; --muted: #6e6164; --faint: #a39497; --ok: #2f9e5b; } }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 16px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; -webkit-font-smoothing: antialiased; }
  main { max-width: 520px; margin: 0 auto; padding: max(28px, env(safe-area-inset-top)) 20px 40px; }
  .logo { display: block; width: 112px; height: auto; margin: 0 auto 14px; }
  h1 { margin: 0; font-size: 24px; font-weight: 650; text-align: center; letter-spacing: -0.01em; }
  .sub { margin: 4px 0 22px; color: var(--muted); text-align: center; }
  .status { display: flex; align-items: center; justify-content: center; gap: 10px; margin: 0 0 22px; padding: 10px 14px; border: 1px solid var(--line);
    border-radius: 12px; background: var(--panel); color: var(--muted); font-size: 14.5px; }
  .status::before { content: ""; flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--faint); animation: pulse 1.4s ease-in-out infinite; }
  .status.ok { color: var(--text); border-color: color-mix(in srgb, var(--ok) 50%, var(--line)); }
  .status.ok::before { background: var(--ok); animation: none; }
  @keyframes pulse { 50% { opacity: 0.35; } }
  .go { display: block; width: 100%; padding: 15px 18px; border: 0; border-radius: 14px; background: var(--accent); color: #fff; font: 600 17px/1.2 inherit;
    text-align: center; text-decoration: none; }
  .go:active { filter: brightness(0.92); }
  ol { margin: 22px 0 0; padding: 0; list-style: none; counter-reset: step; }
  li { position: relative; margin: 0 0 14px; padding-left: 38px; counter-increment: step; }
  li::before { content: counter(step); position: absolute; left: 0; top: 0; display: grid; place-items: center; width: 26px; height: 26px; border-radius: 50%;
    background: var(--panel); border: 1px solid var(--line); color: var(--accent); font-size: 13.5px; font-weight: 650; }
  li:has(.aside) { counter-increment: none; padding-left: 38px; }
  li:has(.aside)::before { content: none; }
  .aside { color: var(--muted); font-size: 14px; }
  code { padding: 1px 5px; border-radius: 6px; background: var(--panel); font: 13.5px ui-monospace, monospace; overflow-wrap: anywhere; }
  .warn { margin: 0 0 18px; padding: 12px 14px; border-radius: 12px; background: color-mix(in srgb, var(--accent) 14%, transparent); font-size: 15px; }
  .warn button { margin-top: 8px; padding: 8px 12px; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); color: var(--text); font: inherit; }
  details { margin-top: 26px; border-top: 1px solid var(--line); padding-top: 14px; color: var(--muted); font-size: 14.5px; }
  summary { cursor: pointer; }
  .others { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .others button { padding: 7px 12px; border: 1px solid var(--line); border-radius: 999px; background: var(--panel); color: var(--text); font: inherit; font-size: 14px; }
  .others button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
  .about { margin: 12px 0 0; }
  .fp { display: block; margin-top: 6px; color: var(--faint); font: 12px/1.5 ui-monospace, monospace; overflow-wrap: anywhere; }
</style>
</head>
<body>
<main>
  ${LOGO.replace("<svg ", '<svg class="logo" aria-hidden="true" ')}
  <h1>Connect to Chrysalis</h1>
  <p class="sub">Set this device up once, and Chrysalis opens securely on it from then on.</p>
  <div class="status" id="status" role="status" aria-live="polite">Checking the secure connection…</div>
  <div id="guide"></div>
  <details>
    <summary>Another device, or what this does</summary>
    <div class="others" id="others"></div>
    <p class="about">This device will trust <b>${name}</b>, an authority this computer made for its own web chat. It can only vouch
      for private addresses and local names, so it changes nothing for any other site. To compare it with the one the
      computer shows (<code>chrysalis status</code>), its SHA-256 fingerprint:<span class="fp" id="fp">${esc(a.fingerprint)}</span></p>
  </details>
</main>
<script>
const CHAT = ${JSON.stringify(chat)}, GUIDES = ${JSON.stringify(guides)};
const ua = navigator.userAgent, touchMac = /Macintosh/.test(ua) && navigator.maxTouchPoints > 1;
const os = /iPhone|iPad|iPod/.test(ua) || touchMac ? "ios" : /Android/.test(ua) ? "android" : /CrOS/.test(ua) ? "chromeos"
  : /Macintosh/.test(ua) ? "mac" : /Windows/.test(ua) ? "windows" : "linux";
// desktop Firefox keeps an authority list of its own, and asks itself; on phones it goes by the system's
const detected = /Firefox\\//.test(ua) && os !== "ios" && os !== "android" ? "firefox" : os;
const $ = (id) => document.getElementById(id);
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
// The device trusts the chat once an image from its HTTPS address loads: then the chat opens. Checked again at once when
// the page comes back into view (after Settings).
let checking = false, timer = 0;
function probe() {
  return new Promise((done) => {
    const img = new Image();
    img.onload = () => done(true);
    img.onerror = () => done(false);
    img.src = CHAT + "trusted.gif?" + Date.now();
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
    setTimeout(() => location.replace(CHAT), 700);
    return;
  }
  status.textContent = "Waiting for this device to trust Chrysalis…";
  timer = setTimeout(check, 1500);
}
document.addEventListener("visibilitychange", () => { if (!document.hidden) check(); });
check();
</script>
</body>
</html>
`;
}
