import { createInterface } from "node:readline";
import { start, submit } from "./cli-core";
import { mount } from "./cli-ui";

if (process.stdout.isTTY && process.stdin.isTTY) await mount();
else createInterface({ input: process.stdin }).on("line", (l) => { if (l.trim()) submit(l.trim()); });
start();
