import { expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readlinkSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { main, parse, plan, uninstall } from "./launch";
import { here, settle } from "./platform";

const none = () => false, all = () => true;

test("a start asks nothing and downloads nothing by itself: it runs the model last used when that one is here, and none the first time or when it is gone; a model named on the command line is run, and fetched first when it is not here; no GPU, no plan", () => {
  expect(plan({ gpu: false, last: "qwen3.8-27b", missing: none })).toBeUndefined();
  expect([plan({ gpu: true, missing: all }), plan({ gpu: true, missing: none })]).toEqual([{}, {}]);
  expect([plan({ gpu: true, last: "qwen3.8-flash-next", missing: none }), plan({ gpu: true, last: "qwen3.8-flash-next", missing: all }), plan({ gpu: true, last: "other", missing: none })]).toEqual([{ model: "qwen3.8-flash-next" }, {}, {}]);
  expect([plan({ gpu: true, last: "qwen3.8-27b", first: "qwen3.8-flash-next", missing: none }), plan({ gpu: true, first: "qwen3.8-flash-next", missing: all })]).toEqual([{ model: "qwen3.8-flash-next" }, { get: "qwen3.8-flash-next" }]);
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

test("uninstalling removes the models, the caches and the settings Chrysalis wrote, says which were there, and leaves the folders beside them alone", () => {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-uninstall-`);
  try {
    for (const d of ["home/data/models/qwen3.8-27b", "home/data/cache/huggingface", "home/data/venv", "config/chrysalis/tls", "config/other"]) mkdirSync(`${dir}/${d}`, { recursive: true });
    writeFileSync(`${dir}/home/data/models/qwen3.8-27b/qwen3.8-27b.gguf`, "weights");
    expect(uninstall(`${dir}/home`, `${dir}/config/chrysalis`, `${dir}/none.exe`, `${dir}/home`).sort()).toEqual([`${dir}/config/chrysalis`, `${dir}/home/data/cache`, `${dir}/home/data/models`]);
    expect(["home/data/models", "home/data/cache", "config/chrysalis", "home/data/venv", "config/other"].map((d) => existsSync(`${dir}/${d}`))).toEqual([false, false, false, true, true]);
    expect(uninstall(`${dir}/home`, `${dir}/config/chrysalis`, `${dir}/none.exe`, `${dir}/home`)).toEqual([]);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("in a Mac app the program is read where it was installed and kept where its data is: the engines, the model files and the version are copied out, Python and the Python code are linked, weights are never copied, and uninstalling removes the data folder and the app", () => {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-app-`), app = `${dir}/Applications/Chrysalis.app/Contents/Resources`, home = `${dir}/Support/Chrysalis`;
  try {
    for (const d of ["build", "chrysalis", "data/venv/bin", "data/models/qwen3.8-27b"]) mkdirSync(`${app}/${d}`, { recursive: true });
    writeFileSync(`${app}/build/libchrysalis.dylib`, "engine");
    writeFileSync(`${app}/build/chrysalis.metallib.archive`, "cache");
    writeFileSync(`${app}/data/models/qwen3.8-27b/config.json`, "{}");
    writeFileSync(`${app}/data/models/qwen3.8-27b/qwen3.8-27b.gguf`, "weights");
    writeFileSync(`${app}/VERSION`, "v2");
    settle(app, home);
    settle(app, home);
    expect([readFileSync(`${home}/build/libchrysalis.dylib`, "utf8"), existsSync(`${home}/build/chrysalis.metallib.archive`), readFileSync(`${home}/data/models/qwen3.8-27b/config.json`, "utf8"),
      existsSync(`${home}/data/models/qwen3.8-27b/qwen3.8-27b.gguf`), readlinkSync(`${home}/chrysalis`), readlinkSync(`${home}/data/venv`), readFileSync(`${home}/VERSION`, "utf8")])
      .toEqual(["engine", false, "{}", false, `${app}/chrysalis`, `${app}/data/venv`, "v2"]);
    writeFileSync(`${app}/build/libchrysalis.dylib`, "engine 3");
    writeFileSync(`${app}/VERSION`, "v3");
    settle(app, home);
    expect(readFileSync(`${home}/build/libchrysalis.dylib`, "utf8")).toBe("engine 3");
    mkdirSync(`${dir}/config/chrysalis`, { recursive: true });
    expect(uninstall(home, `${dir}/config/chrysalis`, `${dir}/none.exe`, app).sort()).toEqual([`${dir}/Applications/Chrysalis.app`, `${dir}/Support/Chrysalis`, `${dir}/config/chrysalis`].sort());
    expect([existsSync(home), existsSync(`${dir}/Applications/Chrysalis.app`), existsSync(`${dir}/config/chrysalis`)]).toEqual([false, false, false]);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
