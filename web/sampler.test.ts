import { expect, test } from "bun:test";
import { type Cand, type SamplerKey, type Stage, sample } from "./sampler";

const STAGES: Stage[] = ["top_k", "temperature", "top_p", "min_p"];
const HELPER = `${import.meta.dir}/../build/sampler_cases`;

function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test("the web sampler gives make_dist's odds", () => {
  const r = mulberry32(7), gauss = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  const f32 = (x: number) => Math.fround(x), num = (x: number) => x.toPrecision(9);
  const text = (id: number) => (id % 23 === 0 ? "\n" : id % 29 === 0 ? ":" : id % 31 === 0 ? "*" : `w${id}`);
  const N = 2000;
  const cases: { cands: Cand[]; reply: [string, number][]; vals: Record<SamplerKey, number>; order: Stage[] }[] = [];
  let input = `${N}\n`;
  for (let c = 0; c < N; c++) {
    const logits = Array.from({ length: 64 }, () => f32(gauss() * 3)).sort((a, b) => b - a);
    const ids = new Set<number>();
    while (ids.size < 64) ids.add(Math.floor(r() * 3000));
    const idl = [...ids];
    const cands = idl.map((id, i) => ({ word: text(id), logit: logits[i], id }));
    const pattern = Array.from({ length: 2 + Math.floor(r() * 4) }, () => (r() < 0.15 ? 23 * (1 + Math.floor(r() * 9)) : idl[Math.floor(r() * 16)]));
    const reply: [string, number][] = [];
    for (let i = 0, len = Math.floor(r() * 41); i < len; i++) {
      const id = r() < 0.1 ? idl[Math.floor(r() * 64)] : pattern[i % pattern.length];
      reply.push([text(id), id]);
    }
    const vals: Record<SamplerKey, number> = {
      temperature: r() < 1 / 17 ? 0 : f32(r() * 2), top_p: f32(0.05 + 0.95 * r()), min_p: r() < 0.66 ? 0 : f32(0.4 * r()),
      top_k: Math.floor(r() * 70), presence_penalty: r() < 0.5 ? 0 : 1.5, dry_multiplier: r() < 0.5 ? 0 : f32(0.5 + 1.5 * r()),
      dry_base: f32(1.5 + 0.5 * r()), dry_allowed_length: 2 + Math.floor(r() * 3),
    };
    const order = [...STAGES].sort(() => r() - 0.5);
    cases.push({ cands, reply, vals, order });
    const breakers = [...new Set([...idl, ...reply.map(([, id]) => id)])].filter((id) => /[\n:"*]/.test(text(id)));
    input += `${logits.map(num).join(" ")}\n${idl.join(" ")}\n${reply.length} ${reply.map(([, id]) => id).join(" ")}\n` +
      `${breakers.length} ${breakers.join(" ")}\n${[vals.temperature, vals.top_p, vals.min_p, vals.presence_penalty].map(num).join(" ")} ` +
      `${vals.top_k} ${num(vals.dry_multiplier)} ${num(vals.dry_base)} ${vals.dry_allowed_length} ${order.map((s) => STAGES.indexOf(s)).join(" ")}\n`;
  }
  const run = Bun.spawnSync([HELPER], { stdin: Buffer.from(input) });
  expect(run.exitCode).toBe(0);
  const lines = run.stdout.toString().trim().split("\n");
  expect(lines.length).toBe(N);
  let worst = 0, first = "";
  cases.forEach(({ cands, reply, vals, order }, c) => {
    const f = lines[c].split(" ").map(Number), engine = new Map<number, number>();
    for (let i = 0; i < f[0]; i++) engine.set(f[1 + 2 * i], f[2 + 2 * i]);
    const { rows } = sample(cands, reply, (k) => vals[k], order);
    rows.forEach((row, i) => {
      const d = Math.abs(row.p - (engine.get(cands[i].id) ?? 0));
      if (d > worst) worst = d;
      if (d > 1e-5 && !first) first = `case ${c}, id ${cands[i].id}: web ${row.p}, engine ${engine.get(cands[i].id) ?? 0} (${order.join(" ")})`;
    });
  });
  expect(first).toBe("");
  expect(worst).toBeLessThan(1e-5);
});
