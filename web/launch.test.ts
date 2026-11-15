import { expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { main, parse, plan } from "./launch";
import { here } from "./platform";

const none = () => false, all = () => true;

test("a first start asks nothing when one model is sensible: the 27B where RAM is short of Flash-Next, the model last chosen or named, and no model where no GPU runs Chrysalis; with room for both it asks once, the 27B first, and gets Flash-Next in the background", () => {
  expect(plan({ gpu: false, roomy: true, asks: true, missing: all })).toEqual({ ask: false, choices: [], at: 0 });
  const small = plan({ gpu: true, roomy: false, asks: true, missing: all });
  expect([small.ask, small.model, small.background, small.choices.length]).toEqual([false, "qwen3.8-27b", undefined, 1]);
  expect(plan({ gpu: true, roomy: false, asks: true, last: "qwen3.8-flash-next", missing: all }).model).toBe("qwen3.8-27b");
  const roomy = plan({ gpu: true, roomy: true, asks: true, missing: all });
  expect([roomy.ask, roomy.at, roomy.model, roomy.background, roomy.choices.map((c) => c.id)]).toEqual([true, 0, "qwen3.8-27b", "qwen3.8-flash-next", ["qwen3.8-27b", "qwen3.8-flash-next"]]);
  expect([plan({ gpu: true, roomy: true, asks: true, missing: none }).background, plan({ gpu: true, roomy: true, asks: true, last: "qwen3.8-flash-next", missing: none }).at]).toEqual([undefined, 1]);
  const named = plan({ gpu: true, roomy: true, asks: true, first: "qwen3.8-flash-next", missing: all });
  expect([named.ask, named.model, named.background]).toEqual([false, "qwen3.8-flash-next", undefined]);
  const piped = plan({ gpu: true, roomy: true, asks: false, last: "qwen3.8-flash-next", missing: all });
  expect([piped.ask, piped.model]).toEqual([false, "qwen3.8-flash-next"]);
  expect([parse([]), parse(["--host", "0.0.0.0", "--model", "qwen3.8-27b"]), parse(["--insecure"]), parse(["--host", "127.0.0.1"]), parse(["--model", "other"]), parse(["--host"])])
    .toEqual([{ mode: "local" }, { mode: "network", model: "qwen3.8-27b" }, { mode: "insecure" }, { mode: "local" }, undefined, undefined]);
});

test("the launcher starts the terminal without a model on a computer with no GPU Chrysalis runs on, asking nothing and choosing no model, and ends with it; a second launcher finds Chrysalis running and stops", async () => {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-launch-`), free = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response() }), port = String(free.port);
  free.stop(true);
  const was = { ...process.env };
  Object.assign(process.env, { PORT: port, CHRYSALIS_URL: "http://127.0.0.1:1", CHRYSALIS_TLS_DIR: `${dir}/tls`, HOME: dir, CHRYSALIS_CHECK_PORT: "0", CHRYSALIS_LINK_PORT: "0",
    CHRYSALIS_ENGINE_CMD: JSON.stringify([process.execPath, "-e", "process.exit(9)"]), XDG_DATA_HOME: `${dir}/share`, XDG_STATE_HOME: `${dir}/state` });
  try {
    const ended = main(["--model", "qwen3.8-27b"], here(), () => ({ here: false, env: {} }));
    let up = false;
    for (const end = Date.now() + 20_000; !up && Date.now() < end; ) up = await fetch(`http://127.0.0.1:${port}/api/status`).then((r) => r.ok, () => false) || (await Bun.sleep(50), false);
    expect(up).toBe(true);
    expect(await main([], here(), () => ({ here: false, env: {} }))).toBe(1);
    expect(existsSync(`${dir}/tls/model`)).toBe(false);
    process.emit("SIGTERM");
    expect(await ended).toBe(0);
    expect(await fetch(`http://127.0.0.1:${port}/`).then(() => "answered", () => "refused")).toBe("refused");
  } finally {
    for (const k of Object.keys(process.env)) if (!(k in was)) delete process.env[k];
    Object.assign(process.env, was);
    rmSync(dir, { recursive: true, force: true });
  }
}, 60_000);
