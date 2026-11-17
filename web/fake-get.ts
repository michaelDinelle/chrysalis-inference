import { existsSync, mkdirSync, writeFileSync } from "node:fs";

const model = process.argv.at(-1)!, models = `${process.env.CHRYSALIS_HOME}/data/models`, done_ = `${models}/${model}/${model}.gguf`;
if (process.argv.includes("--check")) process.exit(existsSync(done_) ? 0 : 1);
const part = `${models}/.cache/huggingface/download/${model}`;
mkdirSync(part, { recursive: true });
writeFileSync(`${part}/weights.incomplete`, "part");
for (let done = Number(process.env.FAKE_GET_FROM ?? 0); ; done = Math.min(16.8, done + 2)) {
  if (process.env.FAKE_GET_FINISH && existsSync(process.env.FAKE_GET_FINISH) && done >= 16.8) {
    mkdirSync(`${models}/${model}`, { recursive: true });
    writeFileSync(done_, "weights");
    process.stderr.write(`\r\x1b[K  ${model}  \x1b[35m━━━\x1b[0m  16.8 / 16.8 GB  done\n`);
    process.exit(0);
  }
  process.stderr.write(`\r\x1b[K  ${model}  \x1b[35m━━━\x1b[0m  ${done.toFixed(1)} / 16.8 GB  100 MB/s, 1:00 left`);
  await Bun.sleep(100);
}
