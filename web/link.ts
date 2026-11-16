import { createHash, randomBytes } from "node:crypto";
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import mdns from "multicast-dns";
import { fingerprint } from "./tls";

const SERVICE = "_chrysalis._tcp.local", TTL = 120, ASK_MS = 10 * 60_000, REFIND_MS = Number(process.env.CHRYSALIS_REFIND_MS ?? 5000);
const sha = (s: string) => createHash("sha256").update(s).digest("hex");
const OFF_NETWORK = "macOS keeps Chrysalis off your network: allow it in System Settings › Privacy & Security › Local Network";
const refused = (e: unknown) => process.platform === "darwin" && (e as { code?: string } | null)?.code === "EHOSTUNREACH";

export function linkCode(fp: string, theirs: string, mine: string): string {
  return String(createHash("sha256").update(`chrysalis link\0${fp}\0${theirs}\0${mine}`).digest().readUInt32BE(0) % 1_000_000).padStart(6, "0");
}

export interface Found { name: string; address: string; port: number; fp: string }
interface Linked extends Found { ca: string; key: string }

export function advertise(name: string, port: number, fp: string, serving: () => boolean = () => true): () => void {
  const m = mdns(), label = name.replace(/[^\w-]/g, "-").slice(0, 60) || "chrysalis", instance = `${label}.${SERVICE}`;
  m.on("query", (q) => {
    if (!q.questions.some((x) => x.name === SERVICE && (x.type === "PTR" || (x.type as string) === "ANY")) || !serving()) return;
    m.respond({ answers: [{ name: SERVICE, type: "PTR", ttl: TTL, data: instance }],
      additionals: [{ name: instance, type: "SRV", ttl: TTL, data: { port, target: `${label}.local` } }, { name: instance, type: "TXT", ttl: TTL, data: [`fp=${fp}`, `name=${name}`] }] });
  });
  return () => m.destroy();
}

export function answerName(host: string, addresses: () => string[]): () => void {
  const m = mdns();
  m.on("query", (q) => {
    if (q.questions.some((x) => x.name.toLowerCase() === host && ["A", "AAAA", "ANY"].includes(x.type as string)))
      m.respond({ answers: [...addresses().map((data) => ({ name: host, type: "A" as const, ttl: TTL, data })),
        { name: host, type: "NSEC" as const, ttl: TTL, data: { nextDomain: host, rrtypes: ["A"] } }] });
  });
  return () => m.destroy();
}

export function discover(off?: () => void): Promise<Found[]> {
  return new Promise((done) => {
    const m = mdns(), found = new Map<string, Found>();
    m.on("response", (r, from) => {
      const all = [...r.answers, ...(r.additionals ?? [])];
      for (const srv of all) {
        if (srv.type !== "SRV" || !srv.name.endsWith(SERVICE)) continue;
        const txt = (all.find((x) => x.type === "TXT" && x.name === srv.name) as { data?: Buffer[] } | undefined)?.data;
        const kv = Object.fromEntries((txt ?? []).map((b) => String(b).split(/=(.*)/s).slice(0, 2)));
        if (/^[0-9a-f]{64}$/.test(kv.fp ?? "")) found.set(kv.fp, { name: kv.name || srv.name.split(".")[0], address: from.address, port: srv.data.port, fp: kv.fp });
      }
    });
    m.on("error", () => {});
    m.query({ questions: [{ name: SERVICE, type: "PTR" }] }, (e) => { if (refused(e)) off?.(); });
    setTimeout(() => { m.destroy(); done([...found.values()]); }, Number(process.env.CHRYSALIS_FIND_MS) || 1500);
  });
}

const base = (f: Found) => `https://${f.address.includes(":") ? `[${f.address}]` : f.address}:${f.port}`;

export class Link {
  private linked?: Linked;
  pairing?: { name: string; code: string; state: "waiting" | "allowed" | "gone" | "error"; error?: string; theirs?: boolean; confirmed?: boolean };
  private pending?: Linked;
  private found: Found[] = [];
  private finding?: Promise<void>;
  private lastFind = 0;
  private offNetwork = false;

  constructor(private file: string, private ownFp: string, private onNetwork: boolean) {
    if (existsSync(file)) this.linked = JSON.parse(readFileSync(file, "utf8"));
  }
  get name() { return this.linked?.name; }
  private save(l: Linked | undefined) {
    this.linked = l;
    if (l) writeFileSync(this.file, JSON.stringify(l, null, 1), { mode: 0o600 });
    else rmSync(this.file, { force: true });
  }
  forget() { this.save(undefined); this.pairing = undefined; this.pending = undefined; }
  confirm(): boolean {
    if (this.pairing?.state !== "waiting" || !this.pairing.code) return false;
    this.pairing.confirmed = true;
    this.settle();
    return true;
  }
  private settle() {
    if (!this.pairing?.confirmed || !this.pending) return;
    this.save(this.pending);
    this.pending = undefined;
    this.pairing.state = "allowed";
  }

  state() {
    if (this.onNetwork && !this.linked && !this.finding && Date.now() - this.lastFind > REFIND_MS) {
      let off = false;
      this.finding = discover(() => (off = true)).then((f) => {
        this.found = f.filter((x) => x.fp !== this.ownFp);
        this.offNetwork = off;
        this.lastFind = Date.now();
        this.finding = undefined;
      });
    }
    return { linked: this.linked ? { name: this.linked.name } : undefined, pairing: this.pairing,
      found: this.linked ? [] : this.found.map(({ name, fp }) => ({ name, fp })), ...(this.offNetwork ? { refused: OFF_NETWORK } : {}) };
  }

  async pair(fp: string, myName: string): Promise<void> {
    const to = this.found.find((f) => f.fp === fp);
    if (!to) return void (this.pairing = { name: "", code: "", state: "error", error: "that computer is not on the network now" });
    this.pairing = { name: to.name, code: "", state: "waiting" };
    try {
      const ca = await (await fetch(`${base(to)}/link/ca`, { tls: { rejectUnauthorized: false } })).text();
      const post = async (path: string, body: object) => {
        const r = await fetch(`${base(to)}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), tls: { ca } });
        if (!r.ok) throw new Error(`${to.name} said ${r.status} to ${path}`);
        return r.status === 204 ? {} : r.json();
      };
      const mine = randomBytes(16).toString("hex");
      const { secret, nonce } = await post("/link/ask", { name: myName, commit: sha(mine) });
      await post("/link/reveal", { secret, nonce: mine });
      this.pairing.code = linkCode(fingerprint(ca), nonce, mine);
      for (const end = Date.now() + ASK_MS; Date.now() < end && this.pairing?.state === "waiting"; await Bun.sleep(1000)) {
        const r = await fetch(`${base(to)}/link/ask/${secret}`, { tls: { ca } });
        if (r.status === 404) { this.pairing.state = "gone"; return; }
        const got = await r.json();
        if (got.state === "allowed") {
          this.pending = { ...to, fp: fingerprint(ca), ca, key: got.key };
          this.pairing.theirs = true;
          this.settle();
          while (this.pairing?.state === "waiting" && Date.now() < end) await Bun.sleep(250);
          return;
        }
      }
    } catch (e) {
      if (refused(e)) {
        this.offNetwork = true;
        this.pairing = undefined;
      } else if (this.pairing) Object.assign(this.pairing, { state: "error", error: (e as Error).message });
    }
  }

  async forward(req: Request): Promise<Response> {
    const l = this.linked;
    if (!l) return Response.json({ error: { message: "Chrysalis is not on this computer, nor linked to one on the network" } }, { status: 502 });
    const url = new URL(req.url), headers = new Headers(req.headers);
    for (const h of ["host", "cookie", "authorization", "x-api-key", "connection", "content-length"]) headers.delete(h);
    headers.set("Authorization", `Bearer ${l.key}`);
    const body = req.method === "GET" || req.method === "HEAD" ? undefined : await req.arrayBuffer();
    const send = (to: Found) => fetch(`${base(to)}${url.pathname}${url.search}`, { method: req.method, headers, body, signal: req.signal, tls: { ca: l.ca } });
    let r: Response;
    try {
      r = await send(l);
    } catch (e) {
      if (req.signal.aborted) throw e;
      this.offNetwork = refused(e);
      if (this.offNetwork) return Response.json({ error: { message: OFF_NETWORK } }, { status: 502 });
      const now = this.onNetwork ? (await discover()).find((f) => f.fp === l.fp) : undefined;
      if (!now || (now.address === l.address && now.port === l.port)) {
        return Response.json({ error: { message: `Chrysalis on ${l.name} does not answer: is it on, and on this network?` } }, { status: 502 });
      }
      this.save({ ...l, address: now.address, port: now.port });
      try { r = await send(now); } catch { return Response.json({ error: { message: `Chrysalis on ${l.name} does not answer` } }, { status: 502 }); }
    }
    this.offNetwork = false;
    if (r.status === 401) this.forget();
    const back = new Headers(r.headers);
    for (const h of ["content-encoding", "content-length", "transfer-encoding", "connection"]) back.delete(h);
    return new Response(r.body, { status: r.status, headers: back });
  }
}
