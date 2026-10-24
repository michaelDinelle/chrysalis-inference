// A computer without Chrysalis's engine (a laptop) uses another's on the network as its own: its apps and its chat ask
// http://127.0.0.1:8080 here, as they would on the computer with the GPU, and this forwards them there over HTTPS. The
// link finds that computer by mDNS (its service, `_chrysalis._tcp`), and pairs once: both screens show a code, and the
// other computer's Allow gives this one a key (devices.ts). From then on every request carries the key, and this one
// trusts the other's link authority (tls.ts) for that computer alone, which no one in between can show.
//
// The code covers the authority this computer was shown. This one commits to a nonce, the other answers with its own,
// and only then does this one show its nonce: the code is a hash of the authority and both nonces, so one in between,
// whose authority is another, gets a code of its own, and cannot pick an authority whose code comes out the same
// without knowing a nonce it is shown only after it chose (as Bluetooth's numeric comparison does).
import { createHash, randomBytes } from "node:crypto";
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import mdns from "multicast-dns";
import { fingerprint } from "./tls";

const SERVICE = "_chrysalis._tcp.local", TTL = 120, FIND_MS = 1500, ASK_MS = 10 * 60_000;
const sha = (s: string) => createHash("sha256").update(s).digest("hex");
// macOS turns a program's packets for the local network away (EHOSTUNREACH) until the user allows its app there: the
// alert it shows the first time, and System Settings after
const OFF_NETWORK = "macOS keeps Chrysalis off your network: allow it in System Settings › Privacy & Security › Local Network";
const refused = (e: unknown) => process.platform === "darwin" && (e as { code?: string } | null)?.code === "EHOSTUNREACH";

export function linkCode(fp: string, theirs: string, mine: string): string {
  return String(createHash("sha256").update(`chrysalis link\0${fp}\0${theirs}\0${mine}`).digest().readUInt32BE(0) % 1_000_000).padStart(6, "0");
}

// A Chrysalis on the network: its name, where it answers, and its link authority's fingerprint (what it says of itself,
// until the code has covered it)
export interface Found { name: string; address: string; port: number; fp: string }
interface Linked extends Found { ca: string; key: string }

// This computer's link, answering queries for the service on the network (the computers' that look for one) while
// `serving` says Chrysalis runs here
export function advertise(name: string, port: number, fp: string, serving: () => boolean = () => true): () => void {
  const m = mdns(), label = name.replace(/[^\w-]/g, "-").slice(0, 60) || "chrysalis", instance = `${label}.${SERVICE}`;
  m.on("query", (q) => {
    if (!q.questions.some((x) => x.name === SERVICE && (x.type === "PTR" || (x.type as string) === "ANY")) || !serving()) return;
    // a record's TTL is 0 unless set, which mDNS reads as a goodbye: Bonjour (macOS's dns-sd) drops the service
    m.respond({ answers: [{ name: SERVICE, type: "PTR", ttl: TTL, data: instance }],
      additionals: [{ name: instance, type: "SRV", ttl: TTL, data: { port, target: `${label}.local` } }, { name: instance, type: "TXT", ttl: TTL, data: [`fp=${fp}`, `name=${name}`] }] });
  });
  return () => m.destroy();
}

// The Chrysalis computers that answer within FIND_MS, at the address each answered from
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
    setTimeout(() => { m.destroy(); done([...found.values()]); }, FIND_MS);
  });
}

const base = (f: Found) => `https://${f.address.includes(":") ? `[${f.address}]` : f.address}:${f.port}`;

export class Link {
  private linked?: Linked;
  // the pairing on screen: the code to compare, until the other computer allows or denies it
  pairing?: { name: string; code: string; state: "waiting" | "allowed" | "gone" | "error"; error?: string };
  private found: Found[] = [];
  private finding?: Promise<void>;
  private lastFind = 0;
  private offNetwork = false;

  constructor(private file: string, private ownFp: string) {
    if (existsSync(file)) this.linked = JSON.parse(readFileSync(file, "utf8"));
  }
  get name() { return this.linked?.name; }
  private save(l: Linked | undefined) {
    this.linked = l;
    if (l) writeFileSync(this.file, JSON.stringify(l, null, 1), { mode: 0o600 });
    else rmSync(this.file, { force: true });
  }
  forget() { this.save(undefined); this.pairing = undefined; }

  // What the page shows: this computer's link, the pairing under way, and the computers found (looked for again in
  // the background, at most every 5 s, while none is linked)
  state() {
    if (!this.linked && !this.finding && Date.now() - this.lastFind > 5000) {
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
      // its link authority, over a connection nothing vouches for yet: the code covers what was shown here
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
          this.save({ ...to, fp: fingerprint(ca), ca, key: got.key });
          this.pairing.state = "allowed";
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

  // A request for Chrysalis, as it came to http://127.0.0.1:8080 here, sent on to the linked computer with this one's
  // key. Where it does not answer, it is looked for again (its address may have changed), and asked once more.
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
      const now = (await discover()).find((f) => f.fp === l.fp);
      if (!now || (now.address === l.address && now.port === l.port)) {
        return Response.json({ error: { message: `Chrysalis on ${l.name} does not answer: is it on, and on this network?` } }, { status: 502 });
      }
      this.save({ ...l, address: now.address, port: now.port });
      try { r = await send(now); } catch { return Response.json({ error: { message: `Chrysalis on ${l.name} does not answer` } }, { status: 502 }); }
    }
    this.offNetwork = false;
    if (r.status === 401) this.forget();                     // removed there
    const back = new Headers(r.headers);
    for (const h of ["content-encoding", "content-length", "transfer-encoding", "connection"]) back.delete(h);   // fetch undid them
    return new Response(r.body, { status: r.status, headers: back });
  }
}
