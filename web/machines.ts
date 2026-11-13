import type { Send } from "./harness";

interface Waiting { head: (status: number) => void; fail: (e: Error) => void; body?: ReadableStreamDefaultController<Uint8Array> }

export function tunnel(post: (text: string) => void) {
  const waiting = new Map<number, Waiting>(), bytes = new TextEncoder();
  let ids = 0;
  const send: Send = (path, init) => new Promise((resolve, reject) => {
    const id = ++ids, drop = () => { if (waiting.delete(id)) post(JSON.stringify({ id, cancel: true })); };
    const entry: Waiting = {
      head: (status) => resolve([204, 205, 304].includes(status) ? new Response(null, { status }) : new Response(stream, { status })),
      fail: (e) => { reject(e); try { entry.body?.error(e); } catch {} },
    };
    const stream = new ReadableStream<Uint8Array>({ start(c) { entry.body = c; }, cancel: drop });
    waiting.set(id, entry);
    init.signal?.addEventListener("abort", () => { drop(); entry.fail(new DOMException("stopped", "AbortError")); }, { once: true });
    post(JSON.stringify({ id, method: init.method, path, headers: init.headers, body: init.body }));
  });
  const got = (m: { id: number; status?: number; chunk?: string; end?: boolean; error?: string }) => {
    const w = waiting.get(m.id);
    if (!w) return;
    if (m.status) w.head(m.status);
    else if (m.chunk !== undefined) w.body?.enqueue(bytes.encode(m.chunk));
    else if (m.end) { waiting.delete(m.id); try { w.body?.close(); } catch {} }
    else if (m.error) { waiting.delete(m.id); w.fail(new Error(m.error)); }
  };
  const closed = () => {
    for (const w of waiting.values()) w.fail(new Error("That computer went away"));
    waiting.clear();
  };
  return { send, got, closed };
}
