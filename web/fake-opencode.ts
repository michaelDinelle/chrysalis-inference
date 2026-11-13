interface Session { id: string; title: string; dir: string; created: number; updated: number; idle?: number; archived?: number; fork?: object; metadata?: object; model?: object; messages: any[]; persona?: string; parentID?: string; agent?: string }

const PAGE = '<!doctype html>\n<html><head><title>Counter</title></head>\n<body>\n<h1 id="title">Clicks</h1>\n<button id="inc">Add</button>\n</body></html>\n';

export function fakeOpenCode(secret: string) {
  const sessions = new Map<string, Session>(), listeners = new Set<(line: string) => void>(), asks = new Map<string, (reply: string, at: string) => void>();
  const stops = new Map<string, { now: boolean }>(), staged = new Map<string, string>();
  const log: { prompts: { session: string; text: string; persona?: string; model?: object; agent?: string; command?: string; meta?: any; files?: { uri: string; name?: string }[] }[]; replies: string[]; interrupts: number; answers: (object | string)[] } =
    { prompts: [], replies: [], interrupts: 0, answers: [] };
  const forms = new Map<string, (answer: object | undefined) => void>();
  let ids = 0;
  const held: (() => void)[] = [];
  const release = () => { for (const go of held.splice(0)) go(); };
  const emit = (type: string, data: object) => { for (const fn of listeners) fn(`data: ${JSON.stringify({ id: `evt_${++ids}`, type, data })}\n\n`); };
  const json = (data: unknown, status = 200) => Response.json(data, { status });
  const info = (s: Session) => ({ id: s.id, title: s.title, time: { created: s.created, updated: s.updated, idle: s.idle, ...(s.archived ? { archived: s.archived } : {}) }, ...(s.fork ? { fork: s.fork } : {}), ...(s.metadata ? { metadata: s.metadata } : {}), location: { directory: s.dir }, ...(s.model ? { model: s.model } : {}), ...(s.parentID ? { parentID: s.parentID } : {}) });

  async function run(s: Session, text: string, more: object = {}) {
    const sessionID = s.id, stopped = { now: false }, who = /Write (\w+)'s next reply/.exec(s.persona ?? "")?.[1] ?? /\[(\w+) next\.\]$/.exec(text)?.[1], lead = who ? `${who} here. ` : "";
    stops.set(s.id, stopped);
    s.messages.push({ id: `msg_${++ids}`, type: "user", text, time: { created: Date.now() }, ...more });
    emit("session.execution.started", { sessionID });
    const one = `msg_${++ids}`, call = `call_${++ids}`, page = text.includes("page file"), tool = page ? "write" : "read";
    const input: object = page ? { path: `${s.dir}/counter.html`, content: PAGE } : { path: "note.txt" }, out = page ? "Wrote file successfully." : "hello\nhoney\n";
    emit("session.step.started", { sessionID, assistantMessageID: one });
    emit("session.reasoning.delta", { sessionID, assistantMessageID: one, delta: "Look at the file first." });
    emit("session.tool.input.started", { sessionID, assistantMessageID: one, id: call, name: tool });
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
    if (text.includes("a helper")) {
      const child: Session = { id: `ses_${++ids}`, title: "Count the jars", dir: s.dir, created: Date.now(), updated: Date.now(), messages: [], parentID: s.id }, sub = `call_${++ids}`, read = `call_${++ids}`;
      const later = text.includes("in the background"), mine = { sessionID: child.id, assistantMessageID: `msg_${++ids}` };
      sessions.set(child.id, child);
      emit("session.tool.input.started", { sessionID, assistantMessageID: one, id: sub, name: "subagent" });
      emit("session.tool.called", { sessionID, assistantMessageID: one, id: sub, input: { agent: "explore", description: "Count the jars" } });
      emit("session.created", { sessionID: child.id, parentID: s.id });
      emit("session.tool.progress", { sessionID, assistantMessageID: one, id: sub, metadata: { sessionID: child.id, status: "running" } });
      emit("session.execution.started", { sessionID: child.id });
      const work = async () => {
        emit("session.tool.input.started", { ...mine, id: read, name: "shell" });
        emit("session.tool.called", { ...mine, id: read, input: { command: "ls jars" } });
        const request = `per_${++ids}`;
        emit("permission.asked", { id: request, sessionID: child.id, action: "shell", resources: ["ls jars"], save: ["*"], source: { type: "tool", id: read } });
        log.replies.push(await new Promise<string>((done) => asks.set(request, (reply, at) => done(`${at === child.id ? "to the helper" : "to the wrong session"}: ${reply}`))));
        emit("session.tool.success", { ...mine, id: read, content: [{ type: "text", text: "12 jars" }] });
        emit("session.execution.succeeded", { sessionID: child.id });
      };
      if (later) {
        emit("session.tool.success", { sessionID, assistantMessageID: one, id: sub, content: [{ type: "text", text: "The helper is working in the background." }] });
        void Bun.sleep(300).then(work).then(() => Bun.sleep(300)).then(async () => {
          emit("session.inbox.enqueued", { sessionID, inboxID: `msg_${++ids}`, item: { type: "user" } });
          s.messages.push({ id: `msg_${++ids}`, type: "user", text: "<subagent>12 jars</subagent>", metadata: { source: "subagent", agent: "Explore" }, time: { created: Date.now() } });
          emit("session.execution.started", { sessionID });
          emit("session.inbox.delivered", { sessionID });
          const three = `msg_${++ids}`, said = "The helper counted 12 jars.";
          emit("session.step.started", { sessionID, assistantMessageID: three });
          emit("session.text.delta", { sessionID, assistantMessageID: three, delta: said });
          emit("session.step.ended", { sessionID, assistantMessageID: three, finish: "stop", tokens: { input: 10, output: 5, cache: { read: 0 } } });
          s.messages.push({ id: three, type: "assistant", time: { created: Date.now() }, content: [{ type: "text", text: said }] }, { id: `msg_${++ids}`, type: "idle", outcome: "succeeded", time: { created: Date.now() } });
          s.updated = s.idle = Date.now();
          emit("session.execution.succeeded", { sessionID });
        });
      } else {
        await work();
        emit("session.tool.success", { sessionID, assistantMessageID: one, id: sub, content: [{ type: "text", text: "<subagent>12 jars</subagent>" }] });
      }
    }
    let chosen = "";
    if (text.includes("which one")) {
      const form = `frm_${++ids}`;
      emit("form.created", { form: { id: form, sessionID, title: "Questions", metadata: { kind: "question", tool: { messageID: one, id: call } },
        fields: [{ key: "q0", title: "Which jar", description: "Which jar should I read?", type: "string", custom: true, options: [{ value: "Clover", label: "Clover" }, { value: "Heather", label: "Heather" }] }] } });
      const answer = await new Promise<object | undefined>((done) => forms.set(form, done));
      log.answers.push(answer ?? "skipped");
      chosen = answer ? ` You chose ${(answer as { q0?: string }).q0}.` : " You did not choose.";
    }
    if (text.includes("until let go")) await new Promise<void>((go) => held.push(go));
    else if (text.includes("slowly")) for (let k = 0; k < 40 && !stopped.now; k++) await Bun.sleep(50);
    const state = allowed ? { status: "completed", input, output: out } : { status: "error", input, error: "rejected" };
    if (allowed) emit("session.tool.success", { sessionID, assistantMessageID: one, id: call, content: [{ type: "text", text: out }] });
    else emit("session.tool.failed", { sessionID, assistantMessageID: one, id: call, error: { message: "rejected" } });
    const edit = `call_${++ids}`, change = { path: `${s.dir}/counter.html`, oldString: "<h1 id=\"title\">Clicks</h1>", newString: "<h1 id=\"title\">Taps</h1>" };
    if (page) {
      emit("session.tool.input.started", { sessionID, assistantMessageID: one, id: edit, name: "edit" });
      emit("session.tool.input.ended", { sessionID, assistantMessageID: one, id: edit, text: JSON.stringify(change) });
      emit("session.tool.called", { sessionID, assistantMessageID: one, id: edit, input: change });
      emit("session.tool.success", { sessionID, assistantMessageID: one, id: edit, content: [{ type: "text", text: "Edit applied successfully." }] });
    }
    emit("session.step.ended", { sessionID, assistantMessageID: one, finish: "tool-calls", tokens: { input: 30, output: 12, cache: { read: 70 } } });
    if (text.includes("page file until released")) await new Promise<void>((go) => held.push(go));
    s.messages.push({ id: one, type: "assistant", time: { created: Date.now() },
      content: [{ type: "reasoning", text: "Look at the file first." }, { type: "tool", id: call, name: tool, state },
        ...(page ? [{ type: "tool", id: edit, name: "edit", state: { status: "completed", input: change, output: "Edit applied successfully." } }] : [])] });
    const end = (outcome: string) => {
      s.messages.push({ id: `msg_${++ids}`, type: "idle", outcome, time: { created: Date.now() } });
      s.updated = s.idle = Date.now();
      stops.delete(s.id);
    };
    if (stopped.now) { end("interrupted"); return emit("session.execution.interrupted", { sessionID, reason: "user" }); }
    const two = `msg_${++ids}`, answer = `${lead}${allowed ? "The file says hello and honey." : "I did not read it."}${chosen} You asked: ${text}`;
    emit("session.step.started", { sessionID, assistantMessageID: two });
    for (let k = 0; k < answer.length; k += 12) emit("session.text.delta", { sessionID, assistantMessageID: two, delta: answer.slice(k, k + 12) });
    emit("session.step.ended", { sessionID, assistantMessageID: two, finish: "stop", tokens: { input: 40, output: 20, cache: { read: 100 } } });
    s.messages.push({ id: two, type: "assistant", time: { created: Date.now() }, tokens: { input: 40, output: 20, cache: { read: 100 } }, content: [{ type: "text", text: answer }] });
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
      if (path === "/api/info") return json({ version: "2.2.7" });
      if (path === "/api/model") return json({ data: [{ id: "qwen3.8-27b", providerID: "chrysalis", name: "Qwen3.8 27B", variants: [{ id: "none" }, { id: "low" }, { id: "medium" }] }, { id: "cloud-1", providerID: "elsewhere", name: "Cloud One", settings: { baseURL: "https://cloud.example/v1" } }] });
      if (path === "/api/agent") return json({ data: [{ id: "build", name: "Build", mode: "primary", hidden: false }, { id: "plan", name: "Plan", mode: "primary", hidden: false },
        { id: "explore", name: "Explore", mode: "subagent", hidden: false }, { id: "title", name: "Title", mode: "primary", hidden: true }] });
      if (path === "/api/command") return json({ data: [{ name: "review", description: "review changes" }] });
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
      if (path === "/api/experimental/session/import") {
        const { info: was, messages, location } = await req.json();
        if (sessions.has(was.id)) return json({ message: "Session already exists" }, 409);
        const back: Session = { id: was.id, title: was.title, dir: location?.directory ?? process.cwd(), created: was.time.created, updated: was.time.updated, idle: was.time.idle, archived: was.time.archived, metadata: was.metadata, model: was.model, messages };
        sessions.set(back.id, back);
        return json({ data: info(back) });
      }
      if (path === "/api/session" && method === "POST") {
        const body = await req.json(), s: Session = { id: `ses_${++ids}`, title: body.title ?? "", dir: body.location.directory, created: Date.now(), updated: Date.now(), messages: [], ...(body.metadata ? { metadata: body.metadata } : {}) };
        sessions.set(s.id, s);
        return json({ data: info(s) });
      }
      const [, id, rest = ""] = /^\/api(?:\/experimental)?\/session\/([^/]+)(\/.*)?$/.exec(path) ?? [], s = id ? sessions.get(id) : undefined;
      if (!s) return json({ message: "not found" }, 404);
      if (!rest && method === "GET") return json({ data: info(s) });
      if (rest === "/export") return json({ data: { info: info(s), messages: s.messages } });
      if (rest === "/compact" && method === "POST") {
        const sessionID = s.id, text = `Summary: ${s.messages.filter((m) => m.type === "user").length} requests so far.`;
        emit("session.compaction.started", { sessionID, reason: "manual" });
        s.messages.push({ id: `msg_${++ids}`, type: "compaction", status: "completed", reason: "manual", summary: text, time: { created: Date.now() } });
        setTimeout(() => emit("session.compaction.ended", { sessionID, reason: "manual", text }), 30);
        return json({ data: { id: `msg_${ids}`, sessionID, type: "compaction" } });
      }
      if (rest === "/move" && method === "POST") { s.dir = (await req.json()).directory; return new Response(null, { status: 204 }); }
      if (!rest && method === "DELETE") {
        sessions.delete(s.id);
        return new Response(null, { status: 204 });
      }
      if (!rest && method === "PATCH") {
        const to = await req.json();
        if (typeof to.title === "string") s.title = to.title;
        if (typeof to.archived === "boolean") s.archived = to.archived ? Date.now() : undefined;
        if (to.metadata) s.metadata = to.metadata;
        return new Response(null, { status: 204 });
      }
      if (rest === "/fork") {
        const { before } = await req.json(), at = s.messages.findIndex((m) => m.id === before);
        const copy: Session = { id: `ses_${++ids}`, title: `${s.title} (fork)`, dir: s.dir, created: Date.now(), updated: Date.now(), idle: Date.now(), messages: s.messages.slice(0, at < 0 ? undefined : at),
          fork: { sessionID: s.id } };
        sessions.set(copy.id, copy);
        return json({ data: info(copy) });
      }
      if (rest === "/revert/stage") { staged.set(s.id, (await req.json()).messageID); return json({ data: { messageID: staged.get(s.id), files: [] } }); }
      if (rest === "/revert/commit") {
        const at = s.messages.findIndex((m) => m.id === staged.get(s.id));
        if (at >= 0) s.messages.length = at;
        s.updated = s.idle = Date.now();
        return new Response(null, { status: 204 });
      }
      if (rest === "/agent") { s.agent = (await req.json()).agent; return new Response(null, { status: 204 }); }
      if (rest === "/command") {
        const { name, text } = await req.json();
        log.prompts.push({ session: s.id, text, persona: s.persona, model: s.model, agent: s.agent, command: name });
        void run(s, `/${name} ${text}`);
        return new Response(null, { status: 204 });
      }
      if (rest === "/message") return json({ data: s.messages, cursor: {} });
      if (rest === "/model") { s.model = (await req.json()).model; return new Response(null, { status: 204 }); }
      if (rest === "/instructions/entries/character") { s.persona = method === "PUT" ? (await req.json()).value : undefined; return new Response(null, { status: 204 }); }
      if (rest === "/interrupt") { log.interrupts++; const stop = stops.get(s.id); if (stop) stop.now = true; return new Response(null, { status: 204 }); }
      if (rest === "/prompt") {
        const { text, files, metadata } = await req.json();
        log.prompts.push({ session: s.id, text, persona: s.persona, model: s.model, agent: s.agent, ...(metadata ? { meta: metadata.chrysalis } : {}), ...(files ? { files } : {}) });
        if (stops.has(s.id)) { s.messages.push({ id: `msg_${++ids}`, type: "user", text, time: { created: Date.now() } }); return json({ data: { id: `msg_${ids}`, sessionID: s.id } }); }
        void run(s, text, { ...(metadata ? { metadata } : {}), ...(files ? { files: files.map((f: { uri: string; name?: string }) => ({ data: f.uri.slice(f.uri.indexOf(",") + 1), mime: f.uri.slice(5, f.uri.indexOf(";")), name: f.name, source: { type: "inline" } })) } : {}) });
        return json({ data: { id: `msg_${++ids}`, sessionID: s.id } });
      }
      const asked = /^\/permission\/([^/]+)\/reply$/.exec(rest)?.[1];
      if (asked) { asks.get(asked)?.((await req.json()).decision, s.id); asks.delete(asked); return new Response(null, { status: 204 }); }
      const form = /^\/form\/([^/]+)(\/reply)?$/.exec(rest);
      if (form && forms.has(form[1])) { forms.get(form[1])!(form[2] ? (await req.json()).answer : undefined); forms.delete(form[1]); return new Response(null, { status: 204 }); }
      return json({ message: "not found" }, 404);
    },
  });
  const wake = (session: string, agent: string, text: string) => run(sessions.get(session)!, text, { metadata: { source: "subagent", agent } });
  return { server, log, sessions, wake, release };
}
