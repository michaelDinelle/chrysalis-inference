import { mkdirSync, writeFileSync } from "node:fs";

const model = process.argv.at(-1)!, dir = `${process.env.CHRYSALIS_HOME}/data/models/${model}`;
mkdirSync(dir, { recursive: true });
writeFileSync(`${dir}/${model}.gguf`, "part");
for (let done = Number(process.env.FAKE_GET_FROM ?? 0); ; done += 0.5) {
  process.stderr.write(`\r\x1b[K  ${model}  \x1b[35m━━━\x1b[0m  ${done.toFixed(1)} / 16.8 GB  100 MB/s, 1:00 left`);
  await Bun.sleep(100);
}
