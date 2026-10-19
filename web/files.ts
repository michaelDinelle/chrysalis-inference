// Files attached to messages: text files (sent to the model as their text), pictures (sent as images), PDFs (Chrysalis
// reads their pages as pictures and text) and videos (frames Chrysalis takes 2 a second). A
// conversation keeps each file's name, kind and size; what is in it lives in IndexedDB, which holds far more than
// sessionStorage (one screenshot is most of sessionStorage's 5 MB). It belongs to this tab like the conversations do:
// every record carries the tab's id (kept in sessionStorage, so a reload keeps it), the tab marks itself alive once a
// minute, and records of tabs not seen for ten minutes are deleted whenever a Chrysalis page loads.
export type Kind = "text" | "image" | "pdf" | "video";
export const KINDS: Kind[] = ["text", "image", "pdf", "video"];
export interface Attachment { id: string; name: string; type: string; size: number; kind: Kind }
interface Stored { id: string; tab: string; data: Blob | string }

export const MAX_TEXT = 4 << 20, MAX_IMAGE = 24 << 20, MAX_PDF = 32 << 20, MAX_VIDEO = 64 << 20;
const IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp"]);
const UNREADABLE = "Chrysalis reads text files, PDFs, pictures (PNG, JPEG, WebP, GIF, BMP) and videos";

const randomId = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
const tab = (() => {
  try {
    let id = sessionStorage.getItem("tab");
    if (!id) sessionStorage.setItem("tab", (id = randomId()));
    return id;
  } catch { return randomId(); }
})();

const ask = <T>(r: IDBRequest<T>) => new Promise<T>((ok, fail) => { r.onsuccess = () => ok(r.result); r.onerror = () => fail(r.error); });
const db: Promise<IDBDatabase | null> = new Promise((resolve) => {
  try {
    const open = indexedDB.open("chrysalis", 1);
    open.onupgradeneeded = () => {
      open.result.createObjectStore("files", { keyPath: "id" }).createIndex("tab", "tab");
      open.result.createObjectStore("tabs", { keyPath: "tab" });
    };
    open.onsuccess = () => resolve(open.result);
    open.onerror = () => resolve(null);
  } catch { resolve(null); }
});
const store = async (name: string, mode: IDBTransactionMode) => (await db)?.transaction(name, mode).objectStore(name);

async function alive() {
  await ask((await store("tabs", "readwrite"))!.put({ tab, seen: Date.now() }));
}
// Drop what tabs that are gone left behind.
async function sweep() {
  const tabs = await store("tabs", "readwrite");
  if (!tabs) return;
  const gone = ((await ask(tabs.getAll())) as { tab: string; seen: number }[]).filter((t) => t.tab !== tab && Date.now() - t.seen > 10 * 60_000);
  for (const t of gone) {
    const keys = await ask((await store("files", "readonly"))!.index("tab").getAllKeys(t.tab));
    const files = (await store("files", "readwrite"))!;   // all requests at once: a transaction ends once none is pending
    await Promise.all(keys.map((k) => ask(files.delete(k))));
    await ask((await store("tabs", "readwrite"))!.delete(t.tab));
  }
}
void db.then(async (d) => {
  if (!d) return;
  await alive();
  await sweep();
  setInterval(() => void alive(), 60_000);
});

const memo = new Map<string, Blob | string>();     // contents read this page (and kept for its requests)

export async function put(id: string, data: Blob | string) {
  memo.set(id, data);
  const files = await store("files", "readwrite");
  if (files) await ask(files.put({ id, tab, data } satisfies Stored));
}

export async function get(id: string): Promise<Blob | string | undefined> {
  if (memo.has(id)) return memo.get(id);
  const files = await store("files", "readonly");
  const rec = files && ((await ask(files.get(id))) as Stored | undefined);
  if (rec) memo.set(id, rec.data);
  return rec?.data;
}

export const dataUrl = (b: Blob) => new Promise<string>((ok, fail) => {
  const r = new FileReader();
  r.onload = () => ok(r.result as string);
  r.onerror = () => fail(r.error);
  r.readAsDataURL(b);
});

// A file the user picked, dropped or pasted, stored and described; or why it cannot be attached.
export async function attach(f: File): Promise<Attachment | string> {
  const binary: [Kind, number, string][] = [["image", MAX_IMAGE, "pictures"], ["pdf", MAX_PDF, "PDFs"], ["video", MAX_VIDEO, "videos"]];
  const kind: Kind | null = IMAGE_TYPES.has(f.type) ? "image" : f.type === "application/pdf" || /\.pdf$/i.test(f.name) ? "pdf"
    : f.type.startsWith("video/") ? "video" : null;
  if (kind) {
    const [, max, what] = binary.find(([k]) => k === kind)!;
    if (f.size > max) return `${f.name}: ${what} can be up to ${max >> 20} MB`;
    const a: Attachment = { id: randomId(), name: f.name || "pasted image", type: f.type, size: f.size, kind };
    await put(a.id, f);
    return a;
  }
  if (f.size > MAX_TEXT) return `${f.name}: text files can be up to ${MAX_TEXT >> 20} MB`;
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(await f.arrayBuffer());
  } catch {
    return `${f.name}: ${UNREADABLE}`;
  }
  if (text.includes("\0")) return `${f.name}: ${UNREADABLE}`;
  const a: Attachment = { id: randomId(), name: f.name, type: f.type || "text/plain", size: f.size, kind: "text" };
  await put(a.id, text);
  return a;
}

// A text file as the model reads it: its name, then its text in a fence longer than any backtick run inside it.
export function textBlock(name: string, text: string) {
  const fence = "`".repeat(Math.max(3, ...[...text.matchAll(/`+/g)].map((m) => m[0].length + 1)));
  const lang = name.includes(".") ? name.split(".").pop()!.toLowerCase().replace(/[^a-z0-9+#-]/g, "") : "";
  return `${name}\n${fence}${lang}\n${text.replace(/\n$/, "")}\n${fence}`;
}

export const fmtBytes = (n: number) => n < 1024 ? `${n} B` : n < 1 << 20 ? `${Math.round(n / 1024)} KB` : `${(n / (1 << 20)).toFixed(1)} MB`;
