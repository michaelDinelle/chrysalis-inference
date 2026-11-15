const say = (line: string) => process.stderr.write(`${line}\n`);
let model = "qwen3.8-flash-next", asleep = false, writing = 0;
const gap = Number(process.env.FAKE_ENGINE_GAP_MS ?? 0);

say(`chrysalis: loading ${model}`);
say("chrysalis-fn: CPU experts on 16 threads");
say("chrysalis-fn: weights 20.10 GB VRAM (4.1 s), cold experts 42.60 GB RAM (9.0 s), buffers 2.00 GB (passes of 2048 tokens), KV pool 4030 pages = 257920 tokens (2.50 GB), load 13.2 s");
Bun.serve({
  hostname: "127.0.0.1",
  port: Number(process.env.FAKE_ENGINE_PORT),
  async fetch(req) {
    const path = new URL(req.url).pathname;
    if (path === "/chrysalis/stats") return Response.json(asleep ? { model, asleep: true } : { model, context: 257920 });
    if (path === "/chrysalis/env") return Response.json({ pid: process.pid, snapshots: process.env.CHRYSALIS_SNAPSHOTS, tmp: process.env.TMPDIR });
    if (path === "/metrics") return new Response(`chrysalis_replies_writing ${writing}\n`);
    if (path === "/chrysalis/unload") { asleep = true; return Response.json({ asleep: true }); }
    if (path === "/chrysalis/load") {
      const to = (await req.json()).model;
      if (asleep) say(`chrysalis: awake: ${to} loaded in 0.2 s`);
      else say(`chrysalis: switched ${model} -> ${to} in 0.3 s (context 131072 tokens)`);
      model = to;
      asleep = false;
      return Response.json({ model });
    }
    if (path === "/v1/chat/completions") {
      const body = await req.json(), secret = String(body.messages?.at(-1)?.content ?? "");
      writing++;
      if (gap) await Bun.sleep(gap);
      writing--;
      if (secret.startsWith("probe ")) {
        let wrote = "written";
        if (req.headers.get("x-chrysalis-unkept") === "1") wrote = "unkept";
        else try { const fs = await import("node:fs"); fs.mkdirSync(`${process.env.CHRYSALIS_SNAPSHOTS}/${model}`, { recursive: true }); fs.writeFileSync(`${process.env.CHRYSALIS_SNAPSHOTS}/${model}/${secret.slice(6)}.snap`, "x"); } catch { wrote = "refused"; }
        say("chrysalis: prompt 14982 (cached 14272) ttft 1.08s | gen 1088 42.6 tok/s 1.80/step | stop | slot 1");
        return Response.json({ choices: [{ message: { content: wrote } }] });
      }
      if (secret === "fail") { say("chrysalis: request failed: ValueError('boom')"); return Response.json({ error: { message: "internal error" } }, { status: 500 }); }
      if (secret === "leave") say("chrysalis: prompt 14982 (cached 14272) | stopped by the client after 12 tokens | slot 1");
      else say(`chrysalis: prompt 14982 (cached ${secret === "cold" ? 0 : 14272}) ttft 1.08s | gen 1088 42.6 tok/s 1.80/step | ${secret === "long" ? "length" : "stop"} | slot 1`);
      return Response.json({ choices: [{ message: { content: `heard ${secret}` } }] });
    }
    return new Response("not found", { status: 404 });
  },
});
say(`chrysalis: serving ${model} on http://127.0.0.1:${process.env.FAKE_ENGINE_PORT}/v1 (context 257920 tokens)`);
say("chrysalis: warmed up in 0.4 s");
