import { homedir } from "node:os";
import { fakeOpenCode } from "./fake-opencode";

const port = Number(process.argv[process.argv.indexOf("--port") + 1]);
fakeOpenCode(process.env.OPENCODE_PASSWORD ?? "", port, `${process.env.XDG_DATA_HOME ?? `${homedir()}/.local/share`}/opencode/fake.json`);
