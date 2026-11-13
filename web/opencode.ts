import { createHash, X509Certificate } from "node:crypto";

const MODELS = { "qwen3.8-27b": "Qwen3.8 27B", "qwen3.8-flash-next": "Qwen3.8 Flash-Next" };
const KEY = "CHRYSALIS_KEY";
const label = (name: string) => name.replace(/[^\w.-]/g, "-").slice(0, 60) || "chrysalis";
const providerId = (name: string) => `chrysalis-${label(name)}`;
const favorites = (name: string) => `const fs = require("fs"), f = process.env.FAVORITES, m = { recent: [], favorite: [], variant: {}, ...(fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, "utf8")) : {}) };
for (const modelID of ${JSON.stringify(Object.keys(MODELS))})
    if (!m.favorite.some((x) => x.providerID === "${providerId(name)}" && x.modelID === modelID)) m.favorite.push({ providerID: "${providerId(name)}", modelID });
fs.mkdirSync(require("path").dirname(f), { recursive: true });
fs.writeFileSync(f + ".chrysalis", JSON.stringify(m));
fs.renameSync(f + ".chrysalis", f);`;
const pinned = (cert: string) => {
  const spki = new X509Certificate(cert).publicKey.export({ type: "spki", format: "der" });
  return `curl -fsSk --pinnedpubkey sha256//${createHash("sha256").update(spki).digest("base64")}`;
};

export const opencodeLine = (origin: string, cert: string) => `${pinned(cert)} ${origin}/opencode | sh`;

export function opencodeScript(origin: string, name: string, ca: string, fingerprint: string, cert: string, host: string): string {
  const named = `https://${host}:${new URL(origin).port}`;
  return `#!/bin/sh
set -e
command -v opencode >/dev/null || { echo "OpenCode is not on this computer: get it from https://opencode.ai, then run this again" >&2; exit 1; }
dir="\${XDG_CONFIG_HOME:-$HOME/.config}/chrysalis"
ca="$dir/opencode-ca.pem"
mkdir -p "$dir"
windows() { ps -axo command= | awk '{ n = $1; sub(/.*\\//, "", n) } (n == "opencode" || n == "opencode2" || n == "opencode.exe") && $2 != "serve"' | grep -q .; }
fresh=
if ! grep -qF ${fingerprint} "$ca" 2>/dev/null; then
    was=$(opencode service get env 2>/dev/null | sed -n 's/.*"NODE_EXTRA_CA_CERTS": *"\\([^"]*\\)".*/\\1/p')
    if [ -n "$was" ] && [ "$was" != "$ca" ] && [ -f "$was" ]; then cat "$was" >> "$ca"; fi
    cat >> "$ca" <<'CHRYSALIS_CA'
# Chrysalis on ${label(name)} ${fingerprint}
${ca.trim()}
CHRYSALIS_CA
    fresh=1
fi
if [ -n "$fresh" ] || ! ps axeww | grep -F 'serve --service' | grep -qF "NODE_EXTRA_CA_CERTS=$ca"; then
    if windows; then
        echo "Close OpenCode to go on: an open window keeps its old settings. Waiting..." >&2
        while windows; do sleep 1; done
    fi
    opencode service set env NODE_EXTRA_CA_CERTS "$ca" >/dev/null
    opencode service restart >/dev/null
fi
origin=${origin}
if ${pinned(cert)} -m 3 ${named}/.well-known/opencode >/dev/null 2>&1; then origin=${named}; fi
if [ "$(uname)" = Darwin ]; then
    port=$(($(printf %s ${host} | cksum | cut -d' ' -f1) % 20000 + 20000))
    job=chrysalis-opencode-${host}
    plist="$HOME/Library/LaunchAgents/$job.plist"
    target=\${origin#https://}
    mkdir -p "$HOME/Library/LaunchAgents"
    cat > "$plist" <<CHRYSALIS_JOB
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$job</string>
<key>ProgramArguments</key><array><string>/usr/bin/nc</string><string>\${target%:*}</string><string>\${target##*:}</string></array>
<key>inetdCompatibility</key><dict><key>Wait</key><false/></dict>
<key>Sockets</key><dict><key>Listeners</key><dict><key>SockNodeName</key><string>127.0.0.1</string><key>SockServiceName</key><string>$port</string><key>SockFamily</key><string>IPv4</string></dict></dict>
</dict></plist>
CHRYSALIS_JOB
    launchctl bootout "gui/$(id -u)/$job" 2>/dev/null || true
    if launchctl bootstrap "gui/$(id -u)" "$plist"; then origin=https://127.0.0.1:$port; fi
fi
opencode auth login "$origin" --method login </dev/null
FAVORITES="\${XDG_STATE_HOME:-$HOME/.local/state}/opencode/model.json" BUN_BE_BUN=1 opencode -e '${favorites(name)}' || true
echo "OpenCode here now has Chrysalis's models from ${label(name)}: run opencode in a project folder and pick one. Its other models are still there, and a session on one of those leaves this network." >&2
`;
}

export function opencodeManifest(origin: string, name: string, cert: string, vision: boolean) {
  const curl = pinned(cert), who = label(name);
  const ask = `a=$(${curl} -H 'Content-Type: application/json' -d "{\\"name\\":\\"$(hostname -s | tr -cd 'A-Za-z0-9.-')\\"}" ${origin}/opencode/ask)
echo "Allow \${a#* } on ${who}" >&2
while :; do
    k=$(${curl} ${origin}/opencode/ask/\${a%% *}) || { echo "${who} did not allow this computer" >&2; exit 1; }
    [ -z "$k" ] || { echo "$k"; exit 0; }
    sleep 1
done`;
  const models = Object.fromEntries(Object.entries(MODELS).map(([id, title]) => [id, {
    modelID: id, name: title,
    capabilities: { tools: true, input: vision ? ["text", "image"] : ["text"], output: ["text"] },
    limit: { context: 262144, output: 32768 },
    compatibility: { reasoningField: "reasoning_content" },
    variants: ["low", "medium", "high"].map((effort) => ({ id: effort, settings: { reasoningEffort: effort } })),
  }]));
  return { auth: { command: ["sh", "-c", ask], env: KEY },
    config: { providers: { [providerId(name)]: { name: "Chrysalis", package: "@opencode-ai/ai/providers/openai-compatible",
      settings: { baseURL: `${origin}/v1`, apiKey: `{env:${KEY}}` }, models } } } };
}
