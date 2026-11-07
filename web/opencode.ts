import { createHash, X509Certificate } from "node:crypto";

const MODELS = { "qwen3.8-27b": "Qwen3.8 27B", "qwen3.8-flash-next": "Qwen3.8 Flash-Next" };
const KEY = "CHRYSALIS_KEY";
const label = (name: string) => name.replace(/[^\w.-]/g, "-").slice(0, 60) || "chrysalis";
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
windows() { ps -axo command= | awk '{ n = $1; sub(/.*\\//, "", n) } (n == "opencode" || n == "opencode.exe") && $2 != "serve"' | grep -q .; }
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
opencode auth login "$origin" --method login </dev/null
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
    config: { providers: { [`chrysalis-${who}`]: { name: `Chrysalis on ${who}`, package: "@opencode-ai/ai/providers/openai-compatible",
      settings: { baseURL: `${origin}/v1`, apiKey: `{env:${KEY}}` }, models } } } };
}
