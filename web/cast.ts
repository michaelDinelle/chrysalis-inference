
export interface CardData {
  name: string; description?: string; personality?: string; scenario?: string; first_mes?: string; mes_example?: string;
  system_prompt?: string; post_history_instructions?: string; alternate_greetings?: string[]; creator_notes?: string;
  tags?: string[]; creator?: string; character_version?: string; extensions?: Record<string, unknown>; [k: string]: unknown;
}
export interface Card { spec?: string; spec_version?: string; data: CardData; [k: string]: unknown }
export interface Person { id: string; kind: "character" | "persona" | "assistant"; card: Card; image?: Blob; added: number }
export interface Group { id: string; name: string; members: string[]; missing?: string[]; scenario?: string; added: number }
export interface Voice { tts?: string; name?: string }

const enc = new TextEncoder(), dec = new TextDecoder();
const SIG = [137, 80, 78, 71, 13, 10, 26, 10];
const isPng = (b: Uint8Array) => SIG.every((x, i) => b[i] === x);

const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
const crc32 = (b: Uint8Array) => { let c = ~0; for (const x of b) c = CRC[(c ^ x) & 255] ^ (c >>> 8); return ~c >>> 0; };

interface Chunk { type: string; start: number; end: number; data: Uint8Array }
function chunks(b: Uint8Array): Chunk[] {
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength), out: Chunk[] = [];
  for (let at = 8; at + 12 <= b.length;) {
    const n = v.getUint32(at), type = String.fromCharCode(...b.subarray(at + 4, at + 8));
    if (at + 12 + n > b.length) break;
    out.push({ type, start: at, end: at + 12 + n, data: b.subarray(at + 8, at + 8 + n) });
    if (type === "IEND") break;
    at += 12 + n;
  }
  return out;
}
const textChunk = (c: Chunk) => {
  const z = c.data.indexOf(0);
  return z < 0 ? null : { key: String.fromCharCode(...c.data.subarray(0, z)), text: String.fromCharCode(...c.data.subarray(z + 1)) };
};
const fromBase64 = (s: string) => dec.decode(Uint8Array.from(atob(s.replace(/\s+/g, "")), (ch) => ch.charCodeAt(0)));
const toBase64 = (s: string) => { let bin = ""; for (const x of enc.encode(s)) bin += String.fromCharCode(x); return btoa(bin); };

function asCard(j: any): Card | null {
  if (!j || typeof j !== "object") return null;
  const data = j.spec === "chara_card_v2" || j.spec === "chara_card_v3" ? j.data : !("data" in j) && typeof j.name === "string" ? j : null;
  if (!data || typeof data.name !== "string" || !data.name.trim()) return null;
  if (!["description", "personality", "scenario", "first_mes", "mes_example"].some((k) => typeof data[k] === "string")) return null;
  return data === j ? { spec: "chara_card_v2", spec_version: "2.0", data: { ...data } } : j;
}

export async function readCast(f: File): Promise<{ card: Card; image?: Blob } | { group: { name: string; members: string[]; scenario?: string } } | null> {
  const b = new Uint8Array(await f.slice(0, 64 << 20).arrayBuffer());
  if (isPng(b)) {
    const texts = chunks(b).filter((c) => c.type === "tEXt").map(textChunk).filter((t) => !!t);
    const raw = texts.find((t) => t.key === "ccv3") ?? texts.find((t) => t.key === "chara");
    if (!raw) return null;
    try {
      const card = asCard(JSON.parse(fromBase64(raw.text)));
      return card && { card, image: new Blob([b], { type: "image/png" }) };
    } catch { return null; }
  }
  if (b[0] !== 0x7b && !/\.json$/i.test(f.name)) return null;
  try {
    const j = JSON.parse(dec.decode(b));
    if (j?.spec === "chrysalis_group" && typeof j.name === "string" && Array.isArray(j.members))
      return { group: { name: j.name, members: j.members.filter((x: unknown) => typeof x === "string"), ...(typeof j.scenario === "string" ? { scenario: j.scenario } : {}) } };
    const card = asCard(j);
    return card && { card };
  } catch { return null; }
}

export async function cardPng(card: Card, image?: Blob): Promise<Blob> {
  let png = image && isPng(new Uint8Array(await image.slice(0, 8).arrayBuffer())) ? new Uint8Array(await image.arrayBuffer()) : null;
  if (!png) png = new Uint8Array(await (await asPng(image)).arrayBuffer());
  const keep = chunks(png).filter((c) => !(c.type === "tEXt" && ["chara", "ccv3"].includes(textChunk(c)?.key ?? "")));
  const v2 = { ...card, spec: "chara_card_v2", spec_version: "2.0" }, v3 = { ...card, spec: "chara_card_v3", spec_version: "3.0",
    data: { group_only_greetings: [], ...card.data } };
  const text = (key: string, value: string) => {
    const data = new Uint8Array([...enc.encode(key), 0, ...enc.encode(value)]), out = new Uint8Array(12 + data.length), v = new DataView(out.buffer);
    v.setUint32(0, data.length);
    out.set(enc.encode("tEXt"), 4);
    out.set(data, 8);
    v.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
    return out;
  };
  const parts: Uint8Array[] = [new Uint8Array(SIG)];
  for (const c of keep) {
    if (c.type === "IEND") parts.push(text("chara", toBase64(JSON.stringify(v2))), text("ccv3", toBase64(JSON.stringify(v3))));
    parts.push(png.subarray(c.start, c.end));
  }
  return new Blob(parts as BlobPart[], { type: "image/png" });
}
async function asPng(image?: Blob): Promise<Blob> {
  const canvas = document.createElement("canvas");
  if (image) {
    const bmp = await createImageBitmap(image);
    canvas.width = bmp.width; canvas.height = bmp.height;
    canvas.getContext("2d")!.drawImage(bmp, 0, 0);
  } else {
    canvas.width = canvas.height = 256;
    const g = canvas.getContext("2d")!;
    g.fillStyle = "#3a1a2a"; g.fillRect(0, 0, 256, 256);
  }
  return new Promise((ok) => canvas.toBlob((b) => ok(b!), "image/png"));
}

export const fill = (text: string, char: string, user: string) =>
  text.replace(/\{\{char\}\}|<BOT>/gi, char).replace(/\{\{user\}\}|<USER>/gi, user);
const examples = (s: string) => s.replace(/^\s*<START>\s*$/gim, "").trim();

export function castPrompt(base: string, cast: CardData[], user: CardData, scenario?: string): string {
  const one = cast.length === 1, out: string[] = [base];
  if (!one) out.push(`This is a scene with ${list(cast.map((c) => c.name))}. Write it one paragraph per line, each beginning with the ` +
    `speaker's name and a colon:\n\n${cast[0].name}: *What they do.* What they say.\n\nWrite only for them; {{user}} speaks ` +
    `for themself. Everyone here takes part: whoever would answer or react does, often more than one, then stop when it is {{user}}'s turn.`);
  const about = (c: CardData) => [c.description?.trim() ?? "", c.personality?.trim() ? `${c.name}'s personality: ${c.personality.trim()}` : "",
    !scenario && c.scenario?.trim() ? `Scenario: ${c.scenario.trim()}` : "", c.mes_example?.trim() ? `How ${c.name} talks:\n${examples(c.mes_example)}` : ""].filter(Boolean);
  for (const c of cast) out.push(fill([one ? "" : `## ${c.name}`, ...about(c)].filter(Boolean).join("\n\n"), c.name, user.name));
  if (scenario?.trim()) out.push(`Scenario: ${scenario.trim()}`);
  const you = about(user);
  if (you.length) out.push([one ? "" : `## ${user.name}`, ...you].filter(Boolean).join("\n\n"));
  return fill(out.filter(Boolean).join("\n\n"), cast[0]?.name ?? "", user.name);
}
const list = (names: string[]) => names.length < 2 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;

export function speakerOf(text: string, names: string[]): { name: string; rest: number } | null {
  for (const name of [...names].sort((a, b) => b.length - a.length)) {
    const m = text.match(new RegExp(`^\\s*(?:\\*\\*)?${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:\\*\\*)?\\s*:(?:\\*\\*)?\\s*`));
    if (m) return { name, rest: m[0].length };
  }
  return null;
}

const ask = <T>(r: IDBRequest<T>) => new Promise<T>((ok, fail) => { r.onsuccess = () => ok(r.result); r.onerror = () => fail(r.error); });
const db: Promise<IDBDatabase | null> = new Promise((resolve) => {
  try {
    const open = indexedDB.open("chrysalis-cast", 1);
    open.onupgradeneeded = () => {
      open.result.createObjectStore("people", { keyPath: "id" });
      open.result.createObjectStore("groups", { keyPath: "id" });
    };
    open.onsuccess = () => resolve(open.result);
    open.onerror = () => resolve(null);
  } catch { resolve(null); }
});
const store = async (name: "people" | "groups", mode: IDBTransactionMode) => (await db)?.transaction(name, mode).objectStore(name);
export async function loadCast(): Promise<{ people: Person[]; groups: Group[] } | null> {
  const d = await db, t = d?.transaction(["people", "groups"], "readonly");
  if (!t) return null;
  const [people, groups] = await Promise.all([ask(t.objectStore("people").getAll()), ask(t.objectStore("groups").getAll())]);
  return { people: people as Person[], groups: groups as Group[] };
}
export const savePerson = async (x: Person) => { const s = await store("people", "readwrite"); if (s) await ask(s.put(x)); };
export const dropPerson = async (id: string) => { const s = await store("people", "readwrite"); if (s) await ask(s.delete(id)); };
export const voiceOf = (c: CardData): Voice | undefined => (c.extensions?.chrysalis as { voice?: Voice } | undefined)?.voice;
