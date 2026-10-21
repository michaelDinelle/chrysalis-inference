// Characters as SillyTavern's cards hold them (Character Card V2 and V3: the card's JSON in base64 in a PNG's tEXt chunk,
// "chara" or "ccv3", or the JSON as a file), the user's persona (the same shape: a name, a description, a picture) and
// groups of characters (a small file of our own: SillyTavern keeps none). They live in this browser's IndexedDB, on this
// device: never on the server. A card is kept whole, so exporting it loses nothing this chat does not use (a lorebook,
// another app's extensions); the voice picked for it is kept in it too (extensions.chrysalis.voice).

export interface CardData {
  name: string; description?: string; personality?: string; scenario?: string; first_mes?: string; mes_example?: string;
  system_prompt?: string; post_history_instructions?: string; alternate_greetings?: string[]; creator_notes?: string;
  tags?: string[]; creator?: string; character_version?: string; extensions?: Record<string, unknown>; [k: string]: unknown;
}
export interface Card { spec?: string; spec_version?: string; data: CardData; [k: string]: unknown }
export interface Person { id: string; kind: "character" | "persona"; card: Card; image?: Blob; added: number }
export interface Group { id: string; name: string; members: string[]; scenario?: string; added: number }
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
// tEXt: a Latin-1 keyword, a zero byte, the text (here base64, so ASCII)
const textChunk = (c: Chunk) => {
  const z = c.data.indexOf(0);
  return z < 0 ? null : { key: String.fromCharCode(...c.data.subarray(0, z)), text: String.fromCharCode(...c.data.subarray(z + 1)) };
};
const fromBase64 = (s: string) => dec.decode(Uint8Array.from(atob(s.replace(/\s+/g, "")), (ch) => ch.charCodeAt(0)));
const toBase64 = (s: string) => { let bin = ""; for (const x of enc.encode(s)) bin += String.fromCharCode(x); return btoa(bin); };

// A card as V2 and V3 both have it (V3 is V2's fields and more): a version 1 card (the fields at the top level) is wrapped
function asCard(j: any): Card | null {
  if (!j || typeof j !== "object") return null;
  const data = j.spec === "chara_card_v2" || j.spec === "chara_card_v3" ? j.data : !("data" in j) && typeof j.name === "string" ? j : null;
  if (!data || typeof data.name !== "string" || !data.name.trim()) return null;
  if (!["description", "personality", "scenario", "first_mes", "mes_example"].some((k) => typeof data[k] === "string")) return null;
  return data === j ? { spec: "chara_card_v2", spec_version: "2.0", data: { ...data } } : j;
}

// What a dropped or picked file is: a character's card (a PNG with one in it, or a card's JSON), a group (our file), or
// neither (an ordinary picture or file, which a message takes as an attachment)
export async function readCast(f: File): Promise<{ card: Card; image?: Blob } | { group: { name: string; members: string[]; scenario?: string } } | null> {
  const b = new Uint8Array(await f.slice(0, 64 << 20).arrayBuffer());
  if (isPng(b)) {
    const texts = chunks(b).filter((c) => c.type === "tEXt").map(textChunk).filter((t) => !!t);
    const raw = texts.find((t) => t.key === "ccv3") ?? texts.find((t) => t.key === "chara");   // V3 when it has both
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

// The card as a PNG for any app that reads them: its picture (as a PNG) with the card in both chunks, "ccv3" and, for apps
// that only read V2, "chara"
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
// A picture as a PNG (a JPEG or WebP one, drawn again), or a plain square when there is none
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

// {{char}} and {{user}} (and version 1's <BOT> and <USER>) in a card's text, as SillyTavern fills them
export const fill = (text: string, char: string, user: string) =>
  text.replace(/\{\{char\}\}|<BOT>/gi, char).replace(/\{\{user\}\}|<USER>/gi, user);
const examples = (s: string) => s.replace(/^\s*<START>\s*$/gim, "").trim();

// The system prompt of a chat with characters: the first one's system prompt in place of the chat's own if its card has
// one ({{original}} in it is the chat's own), then for each character what its card says of it (with the group's
// scenario in place of theirs), and the user's persona. Nothing in it changes from turn to turn, so every reply reads
// it from Chrysalis's cache. With several, the reply is the scene: each line its speaker's name and a colon (app.ts sceneText).
export function castPrompt(base: string, cast: CardData[], user: { name: string; description?: string }, scenario?: string): string {
  const one = cast.length === 1, out: string[] = [];
  const own = cast[0]?.system_prompt?.trim();
  out.push(own ? own.replace(/\{\{original\}\}/gi, base) : base);
  if (!one) out.push(`This is a scene with ${list(cast.map((c) => c.name))}. Write it one paragraph per line, each beginning with the ` +
    `speaker's name and a colon:\n\n${cast[0].name}: *What they do.* What they say.\n\nWrite only for them; {{user}} speaks ` +
    `for themself. Everyone here takes part: whoever would answer or react does, often more than one, then stop when it is {{user}}'s turn.`);
  for (const c of cast) {
    const parts = [one ? "" : `## ${c.name}`, c.description?.trim() ?? "", c.personality?.trim() ? `{{char}}'s personality: ${c.personality.trim()}` : "",
      !scenario && c.scenario?.trim() ? `Scenario: ${c.scenario.trim()}` : "",
      c.mes_example?.trim() ? `How {{char}} talks:\n${examples(c.mes_example)}` : ""].filter(Boolean);
    out.push(fill(parts.join("\n\n"), c.name, user.name));
  }
  if (scenario?.trim()) out.push(`Scenario: ${scenario.trim()}`);
  if (user.description?.trim()) out.push(user.description.trim());
  return fill(out.filter(Boolean).join("\n\n"), cast[0]?.name ?? "", user.name);
}
const list = (names: string[]) => names.length < 2 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;

// Who speaks a scene's paragraph: one of the cast's names at its start, then a colon (models also bold the name)
export function speakerOf(text: string, names: string[]): { name: string; rest: number } | null {
  for (const name of [...names].sort((a, b) => b.length - a.length)) {
    const m = text.match(new RegExp(`^\\s*(?:\\*\\*)?${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:\\*\\*)?\\s*:(?:\\*\\*)?\\s*`));
    if (m) return { name, rest: m[0].length };
  }
  return null;
}

// This device's cast, in IndexedDB of its own (the tab's files are another database, deleted with the tab)
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
export async function loadCast(): Promise<{ people: Person[]; groups: Group[] }> {
  const d = await db, t = d?.transaction(["people", "groups"], "readonly");   // one transaction: it ends at the first await
  if (!t) return { people: [], groups: [] };
  const [people, groups] = await Promise.all([ask(t.objectStore("people").getAll()), ask(t.objectStore("groups").getAll())]);
  return { people: people as Person[], groups: groups as Group[] };
}
export const savePerson = async (x: Person) => { const s = await store("people", "readwrite"); if (s) await ask(s.put(x)); };
export const dropPerson = async (id: string) => { const s = await store("people", "readwrite"); if (s) await ask(s.delete(id)); };
export const saveGroup = async (x: Group) => { const s = await store("groups", "readwrite"); if (s) await ask(s.put(x)); };
export const dropGroup = async (id: string) => { const s = await store("groups", "readwrite"); if (s) await ask(s.delete(id)); };
export const voiceOf = (c: CardData): Voice | undefined => (c.extensions?.chrysalis as { voice?: Voice } | undefined)?.voice;
