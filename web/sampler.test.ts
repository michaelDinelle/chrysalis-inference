import { expect, test } from "bun:test";
import { type Cand, DEFAULT_STAGES, type SamplerKey, type Stage, sample } from "./sampler";

const HELPER = `${import.meta.dir}/../build/sampler_cases`;
const INPUT: SamplerKey[] = ["temperature", "dynatemp_range", "dynatemp_exponent", "top_k", "top_p", "min_p", "typical_p", "xtc_probability",
  "xtc_threshold", "top_n_sigma", "repeat_last_n", "repeat_penalty", "presence_penalty", "frequency_penalty", "dry_multiplier", "dry_base",
  "dry_allowed_length", "dry_penalty_last_n", "mirostat", "mirostat_tau", "mirostat_eta", "min_keep"];
const WHOLE = new Set<SamplerKey>(["top_k", "repeat_last_n", "dry_allowed_length", "dry_penalty_last_n", "mirostat", "min_keep"]);

function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Case = { cands: Cand[]; context: [string, number][]; vals: Record<SamplerKey, number>; order: Stage[]; xtc: boolean; mu: number };

function cases(n: number) {
  const r = mulberry32(7), gauss = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  const f32 = (x: number) => Math.fround(x), pick = <T>(xs: readonly T[]) => xs[Math.floor(r() * xs.length)];
  const text = (id: number) => (id % 23 === 0 ? "\n" : id % 29 === 0 ? ":" : id % 31 === 0 ? " *" : `w${id}`);
  const out: Case[] = [];
  for (let c = 0; c < n; c++) {
    const logits = Array.from({ length: 64 }, () => f32(gauss() * 3)).sort((a, b) => b - a);
    const ids = new Set<number>();
    while (ids.size < 64) ids.add(Math.floor(r() * 3000));
    const idl = [...ids];
    const cands = idl.map((id, i) => ({ word: text(id), logit: logits[i], id }));
    const pattern = Array.from({ length: 2 + Math.floor(r() * 5) }, () => (r() < 0.15 ? 23 * (1 + Math.floor(r() * 9)) : idl[Math.floor(r() * 16)]));
    const context: [string, number][] = [];
    for (let i = 0, len = Math.floor(r() * 120); i < len; i++) {
      const id = r() < 0.15 ? idl[Math.floor(r() * 64)] : pattern[i % pattern.length];
      context.push([text(id), id]);
    }
    const vals: Record<SamplerKey, number> = {
      temperature: r() < 0.1 ? 0 : r() < 0.15 ? 1 : f32(0.1 + 1.9 * r()), dynatemp_range: r() < 0.25 ? f32(1.5 * r()) : 0,
      dynatemp_exponent: f32(0.5 + 1.5 * r()), top_k: r() < 0.25 ? 0 : 1 + Math.floor(r() * 80), top_p: r() < 0.3 ? 1 : f32(0.05 + 0.95 * r()),
      min_p: r() < 0.5 ? 0 : f32(0.5 * r()), typical_p: r() < 0.3 ? f32(0.1 + 0.9 * r()) : 1, xtc_probability: r() < 0.3 ? f32(r()) : 0,
      xtc_threshold: f32(0.6 * r()), top_n_sigma: r() < 0.3 ? f32(0.1 + 3.9 * r()) : -1, repeat_last_n: pick([0, 1, 8, 64, -1]),
      repeat_penalty: r() < 0.5 ? f32(0.5 + 1.5 * r()) : 1, presence_penalty: r() < 0.5 ? f32(3 * r() - 1) : 0,
      frequency_penalty: r() < 0.3 ? f32(2 * r() - 1) : 0, dry_multiplier: r() < 0.5 ? f32(0.1 + 2.9 * r()) : 0, dry_base: f32(0.9 + 2.1 * r()),
      dry_allowed_length: Math.floor(r() * 5), dry_penalty_last_n: pick([1, 4, 16, 64, -1]), mirostat: r() < 0.2 ? 1 + Math.floor(r() * 2) : 0,
      mirostat_tau: f32(1 + 7 * r()), mirostat_eta: f32(0.01 + 0.49 * r()), min_keep: r() < 0.3 ? 1 + Math.floor(r() * 10) : 0,
    };
    const order = r() < 0.3 ? [...DEFAULT_STAGES] : Array.from({ length: 1 + Math.floor(r() * 12) }, () => pick(DEFAULT_STAGES));
    out.push({ cands, context, vals, order, xtc: r() < 0.5, mu: f32(20 * r()) });
  }
  return out;
}

function engine(all: Case[]) {
  const num = (k: SamplerKey, x: number) => (WHOLE.has(k) ? String(x) : x.toPrecision(9));
  let input = `${all.length}\n`;
  for (const { cands, context, vals, order, xtc, mu } of all) {
    const words = [...cands.map((c) => [c.word, c.id] as const), ...context];
    const breakers = [...new Set(words.filter(([w]) => /[\n:"*]/.test(w)).map(([, id]) => id))];
    input += `${cands.map((c) => c.logit.toPrecision(9)).join(" ")}\n${cands.map((c) => c.id).join(" ")}\n` +
      `${context.length} ${context.map(([, id]) => id).join(" ")}\n${breakers.length} ${breakers.join(" ")}\n` +
      `${INPUT.map((k) => num(k, vals[k])).join(" ")} ${xtc ? 1 : 0} ${mu.toPrecision(9)} ${order.length} ${order.map((s) => DEFAULT_STAGES.indexOf(s)).join(" ")}\n`;
  }
  const run = Bun.spawnSync([HELPER], { stdin: Buffer.from(input) });
  expect(run.exitCode).toBe(0);
  const lines = run.stdout.toString().trim().split("\n");
  expect(lines.length).toBe(all.length);
  return lines.map((line) => {
    const f = line.split(" ").map(Number), p = new Map<number, number>();
    for (let i = 0; i < f[0]; i++) p.set(f[1 + 2 * i], f[2 + 2 * i]);
    return p;
  });
}

test("the web sampler gives make_dist's odds for every sampler, order and mirostat", () => {
  const all = cases(4000), got = engine(all), used = new Map<string, number>();
  let worst = 0, first = "";
  all.forEach(({ cands, context, vals, order, xtc, mu }, c) => {
    const { rows } = sample(cands, context, (k) => vals[k], order, { xtc, mu });
    for (const s of vals.mirostat ? [`mirostat ${vals.mirostat}`] : order) used.set(s, (used.get(s) ?? 0) + 1);
    rows.forEach((row, i) => {
      const d = Math.abs(row.p - (got[c].get(cands[i].id) ?? 0));
      if (d > worst) worst = d;
      if (d > 1e-5 && !first) first = `case ${c}, id ${cands[i].id}: web ${row.p}, engine ${got[c].get(cands[i].id) ?? 0} (${order.join(" ")})`;
      if (row.cut && row.p > 0 && !first) first = `case ${c}, id ${cands[i].id}: cut by ${row.cut} yet p ${row.p}`;
    });
  });
  expect(first).toBe("");
  expect(worst).toBeLessThan(1e-5);
  for (const s of [...DEFAULT_STAGES, "mirostat 1", "mirostat 2"]) expect(used.get(s) ?? 0).toBeGreaterThan(300);
});

test("each row's penalty is the logit drop the engine's penalties and DRY give it", () => {
  const front = (order: Stage[]) => { const i = order.findIndex((s) => s !== "penalties" && s !== "dry"); return i < 0 ? order.length : i; };
  const picked = cases(6000).filter(({ vals, order }) => !vals.mirostat && front(order) > 0 &&
    !order.slice(front(order)).some((s) => s === "penalties" || s === "dry"));
  const moved = picked.map(({ cands, context, vals, order, xtc, mu }) => {
    const { rows } = sample(cands, context, (k) => vals[k], order, { xtc, mu });
    return { cands: cands.map((c, i) => ({ ...c, logit: Math.fround(c.logit - rows[i].penalty) })).sort((a, b) => b.logit - a.logit),
      context, vals, order: order.slice(front(order)), xtc, mu, dropped: rows.filter((r) => r.penalty > 0).length };
  });
  expect(moved.filter((c) => c.dropped > 0).length).toBeGreaterThan(300);
  const whole = engine(picked), rest = engine(moved);
  let worst = 0;
  picked.forEach(({ cands }, c) => {
    for (const { id } of cands) worst = Math.max(worst, Math.abs((whole[c].get(id) ?? 0) - (rest[c].get(id) ?? 0)));
  });
  expect(worst).toBeLessThan(1e-5);
});
