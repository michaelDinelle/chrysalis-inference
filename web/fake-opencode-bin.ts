import { appendFileSync } from "node:fs";
import { homedir } from "node:os";
import { fakeOpenCode } from "./fake-opencode";

const port = Number(process.argv[process.argv.indexOf("--port") + 1]);
if (process.env.FAKE_OPENCODE_PIDS) appendFileSync(process.env.FAKE_OPENCODE_PIDS, `${process.pid} ${process.env.OPENCODE_DB ?? "file"} ${process.env.XDG_DATA_HOME ?? ""}\n`);
const config = JSON.parse(process.env.OPENCODE_CONFIG_CONTENT || "{}");
fakeOpenCode(process.env.OPENCODE_PASSWORD ?? "", port, process.env.OPENCODE_DB === ":memory:" ? ":memory:" : `${process.env.XDG_DATA_HOME ?? `${homedir()}/.local/share`}/opencode/fake.json`,
  config.providers?.chrysalis?.settings?.baseURL, Object.keys(config.providers?.chrysalis?.models ?? {}).map((id) => ({ id, name: id })));
