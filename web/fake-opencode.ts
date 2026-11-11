interface Session { id: string; title: string; dir: string; created: number; updated: number; model?: object; messages: any[]; persona?: string }

export function fakeOpenCode(secret: string) {
  const sessions = new Map<string, Session>(), listeners = new Set<(line: string) => void>(), asks = new Map<string, (reply: string) => void>();
  const stops = new Map<string, { now: boolean }>();
  const log: { prompts: { session: string; text: string; persona?: string; model?: object }[]; replies: string[]; interrupts: number } = { prompts: [], replies: [], interrupts: 0 };
  let ids = 0;
  const emit = (type: string, data: object) => { for (const fn of listeners) fn(`data: ${JSON.stringify({ id: `evt_${++ids}`, type, data })}\n\n`); };
  const json = (data: unknown, status = 200) => Response.json(data, { status });
  const info = (s: Session) => ({ id: s.id, title: s.title, time: { created: s.created, updated: s.updated }, location: { directory: s.dir }, ...(s.model ? { model: s.model } : {}) });

  async function run(s: Session, text: string) {
    const sessionID = s.id, stopped = { now: false }, lead = s.persona ? `${/You are (\w+)/.exec(s.persona)?.[1] ?? "Someone"} here. ` : "";
    stops.set(s.id, stopped);
    s.messages.push({ id: `msg_${++ids}`, type: "user", text, time: { created: Date.now() } });
    emit("session.execution.started", { sessionID });
    const one = `msg_${++ids}`, call = `call_${++ids}`, input = { path: "note.txt" }, out = "hello\nhoney\n";
    emit("session.step.started", { sessionID, assistantMessageID: one });
    emit("session.reasoning.delta", { sessionID, assistantMessageID: one, delta: "Look at the file first." });
    emit("session.tool.input.started", { sessionID, assistantMessageID: one, id: call, name: "read" });
    emit("session.tool.input.ended", { sessionID, assistantMessageID: one, id: call, text: JSON.stringify(input) });
    emit("session.tool.called", { sessionID, assistantMessageID: one, id: call, input });
    let allowed = true;
    if (text.includes("ask first")) {
      const request = `per_${++ids}`;
      emit("permission.asked", { id: request, sessionID, action: "read", resources: ["note.txt"], save: ["*"], source: { type: "tool", id: call } });
      const reply = await new Promise<string>((done) => asks.set(request, done));
      log.replies.push(reply);
      allowed = reply !== "reject";
    }
    if (text.includes("slowly")) for (let k = 0; k < 40 && !stopped.now; k++) await Bun.sleep(50);
    const state = allowed ? { status: "completed", input, output: out } : { status: "error", input, error: "rejected" };
    if (allowed) emit("session.tool.success", { sessionID, assistantMessageID: one, id: call, content: [{ type: "text", text: out }] });
    else emit("session.tool.failed", { sessionID, assistantMessageID: one, id: call, error: { message: "rejected" } });
    emit("session.step.ended", { sessionID, assistantMessageID: one, finish: "tool-calls", tokens: { input: 30, output: 12, cache: { read: 70 } } });
    s.messages.push({ id: one, type: "assistant", time: { created: Date.now() },
      content: [{ type: "reasoning", text: "Look at the file first." }, { type: "tool", id: call, name: "read", state }] });
    const end = (outcome: string) => {
      s.messages.push({ id: `msg_${++ids}`, type: "idle", outcome, time: { created: Date.now() } });
      s.updated = Date.now();
      stops.delete(s.id);
    };
    if (stopped.now) { end("interrupted"); return emit("session.execution.interrupted", { sessionID, reason: "user" }); }
    const two = `msg_${++ids}`, answer = `${lead}${allowed ? "The file says hello and honey." : "I did not read it."} You asked: ${text}`;
    emit("session.step.started", { sessionID, assistantMessageID: two });
    for (let k = 0; k < answer.length; k += 12) emit("session.text.delta", { sessionID, assistantMessageID: two, delta: answer.slice(k, k + 12) });
    emit("session.step.ended", { sessionID, assistantMessageID: two, finish: "stop", tokens: { input: 40, output: 20, cache: { read: 100 } } });
    s.messages.push({ id: two, type: "assistant", time: { created: Date.now() }, content: [{ type: "text", text: answer }] });
    if (s.title === "") {
      s.title = `About ${text.split(" ").slice(0, 3).join(" ")}`;
      emit("session.renamed", { sessionID, title: s.title });
    }
    end("succeeded");
    emit("session.execution.succeeded", { sessionID });
  }

  const server = Bun.serve({
    hostname: "127.0.0.1", port: 0, idleTimeout: 0,
    async fetch(req) {
      const url = new URL(req.url), path = url.pathname, method = req.method;
      if (req.headers.get("authorization") !== `Basic ${Buffer.from(`opencode:${secret}`).toString("base64")}`) return new Response("", { status: 401 });
      if (path === "/api/info") return json({ version: "2.0.25" });
      if (path === "/api/model") return json({ data: [{ id: "qwen3.8-27b", providerID: "chrysalis", name: "Qwen3.8 27B" }, { id: "cloud-1", providerID: "elsewhere", name: "Cloud One" }] });
      if (path === "/api/provider") return json({ data: [{ id: "chrysalis", name: "Chrysalis" }, { id: "elsewhere", name: "Elsewhere" }] });
      if (path === "/api/event") {
        let fn: (line: string) => void;
        const enc = new TextEncoder();
        return new Response(new ReadableStream({
          start(ctl) { fn = (line) => ctl.enqueue(enc.encode(line)); listeners.add(fn); fn(`data: ${JSON.stringify({ type: "server.connected", data: {} })}\n\n`); },
          cancel() { listeners.delete(fn); },
        }), { headers: { "Content-Type": "text/event-stream" } });
      }
      if (path === "/api/session" && method === "GET") return json({ data: [...sessions.values()].sort((a, b) => b.updated - a.updated).map(info), cursor: {} });
      if (path === "/api/session" && method === "POST") {
        const body = await req.json(), s: Session = { id: `ses_${++ids}`, title: "", dir: body.location.directory, created: Date.now(), updated: Date.now(), messages: [] };
        sessions.set(s.id, s);
        return json({ data: info(s) });
      }
      const [, id, rest = ""] = /^\/api(?:\/experimental)?\/session\/([^/]+)(\/.*)?$/.exec(path) ?? [], s = id ? sessions.get(id) : undefined;
      if (!s) return json({ message: "not found" }, 404);
      if (!rest && method === "GET") return json({ data: info(s) });
      if (rest === "/message") return json({ data: s.messages, cursor: {} });
      if (rest === "/model") { s.model = (await req.json()).model; return json({ data: info(s) }); }
      if (rest === "/instructions/entries/character") { s.persona = method === "PUT" ? (await req.json()).value : undefined; return new Response(null, { status: 204 }); }
      if (rest === "/interrupt") { log.interrupts++; const stop = stops.get(s.id); if (stop) stop.now = true; return new Response(null, { status: 204 }); }
      if (rest === "/prompt") {
        const { text } = await req.json();
        log.prompts.push({ session: s.id, text, persona: s.persona, model: s.model });
        void run(s, text);
        return json({ data: { id: `msg_${++ids}`, sessionID: s.id } });
      }
      const asked = /^\/permission\/([^/]+)\/reply$/.exec(rest)?.[1];
      if (asked) { asks.get(asked)?.((await req.json()).decision); asks.delete(asked); return new Response(null, { status: 204 }); }
      return json({ message: "not found" }, 404);
    },
  });
  return { server, log, sessions };
}
