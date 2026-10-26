export type Stage = "penalties" | "dry" | "top_n_sigma" | "top_k" | "typ_p" | "top_p" | "min_p" | "xtc" | "temperature";
export type SamplerKey = "temperature" | "dynatemp_range" | "dynatemp_exponent" | "top_k" | "top_p" | "min_p" | "typical_p" |
  "xtc_probability" | "xtc_threshold" | "top_n_sigma" | "repeat_last_n" | "repeat_penalty" | "presence_penalty" | "frequency_penalty" |
  "dry_multiplier" | "dry_base" | "dry_allowed_length" | "dry_penalty_last_n" | "mirostat" | "mirostat_tau" | "mirostat_eta" | "min_keep";
export interface Cand { word: string; logit: number; id: number }
export interface Odds { word: string; raw: number; p: number; cut: Stage | "mirostat" | ""; penalty: number }
export const DEFAULT_STAGES: Stage[] = ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"];

const LLAMA: Record<SamplerKey, number> = {
  temperature: 0.8, dynatemp_range: 0, dynatemp_exponent: 1, top_k: 40, top_p: 0.95, min_p: 0.05, typical_p: 1, xtc_probability: 0,
  xtc_threshold: 0.1, top_n_sigma: -1, repeat_last_n: 64, repeat_penalty: 1, presence_penalty: 0, frequency_penalty: 0, dry_multiplier: 0,
  dry_base: 1.75, dry_allowed_length: 2, dry_penalty_last_n: -1, mirostat: 0, mirostat_tau: 5, mirostat_eta: 0.1, min_keep: 0,
};
const f = Math.fround;
const breaker = (t: string) => /[\n:"*]/.test(t);
type Live = { l: number; r: number };

function repeats(ids: number[], lone: (at: number) => boolean, window: number) {
  const n = ids.length, N = Math.min(n, window), at = (i: number) => ids[n - 1 - i];
  let limit = N;
  for (let i = 0; i < N; i++) if (lone(n - 1 - i)) { limit = i; break; }
  const count = new Array<number>(Math.max(N, 1)).fill(0), last = N - 1;
  for (let k = 1, lt = 0, rt = 0; k < N; k++) {
    let z = 0;
    if (k > rt) {
      while (z + k < N && at(z) === at(z + k)) z++;
      if (z > 0) { lt = k; rt = k + z - 1; }
    } else if (count[last - (k - lt)] < rt - k + 1) z = count[last - (k - lt)];
    else {
      let i = rt + 1;
      while (i < N && at(i) === at(i - k)) i++;
      z = i - k; lt = k; rt = i - 1;
    }
    count[last - k] = Math.min(z, limit);
  }
  return { N, limit, lo: n - N, count };
}

export function sample(cands: Cand[], context: [string, number][], v: (k: SamplerKey) => number, stages: readonly Stage[],
  opts: { xtc?: boolean; mu?: number; vocab?: number } = {}) {
  const val = (k: SamplerKey) => { const x = v(k); return Number.isFinite(x) ? x : LLAMA[k]; };
  const top = Math.max(...cands.map((c) => c.logit)), z = cands.reduce((a, c) => a + Math.exp(c.logit - top), 0);
  const rows: Odds[] = cands.map((c) => ({ word: c.word, raw: Math.exp(c.logit - top) / z, p: 0, cut: "", penalty: 0 }));
  let c: Live[] = cands.map((x, r) => ({ l: f(x.logit), r }));
  const sort = () => { c = c.map((x, i) => [x, i] as const).sort((a, b) => b[0].l - a[0].l || a[1] - b[1]).map(([x]) => x); };
  sort();
  const keep = (n: number, why: Odds["cut"]) => { for (const x of c.slice(n)) rows[x.r].cut ||= why; c = c.slice(0, n); };
  const kill = (x: Live, why: Odds["cut"]) => { x.l = -Infinity; rows[x.r].cut ||= why; };
  const softmax = () => {
    const m = c[0].l, e = c.map((x) => f(Math.exp(f(x.l - m))));
    let cum = 0;
    for (const y of e) cum = f(cum + y);
    return { p: e.map((y) => f(y / cum)), cum };
  };
  const entropy = (p: number[]) => { let h = 0; for (const y of p) if (y > 0) h = f(h - f(y * f(Math.log(y)))); return h; };
  const scale = (t: number) => {
    if (t <= 0) { c.slice(1).forEach((x) => kill(x, "temperature")); return; }
    for (const x of c) x.l = f(x.l / t);
  };
  const minKeep = val("min_keep"), miro = val("mirostat") === 1 || val("mirostat") === 2;
  const inChain = (s: Stage) => !miro && stages.includes(s);
  const ids = context.map(([, id]) => id);
  const window = (n: number) => (n < 0 ? Infinity : n);
  const penN = inChain("penalties") && val("repeat_last_n") !== 0 &&
    !(f(val("repeat_penalty")) === 1 && f(val("frequency_penalty")) === 0 && f(val("presence_penalty")) === 0) ? window(val("repeat_last_n")) : 0;
  const dryN = inChain("dry") && f(val("dry_multiplier")) !== 0 && f(val("dry_base")) >= 1 && val("dry_penalty_last_n") !== 0 ?
    window(val("dry_penalty_last_n")) : 0;
  const counts = new Map<number, number>();
  for (const id of ids.slice(Math.max(0, ids.length - penN))) counts.set(id, (counts.get(id) ?? 0) + 1);
  const lone = new Set([...context, ...cands.map((x) => [x.word, x.id] as [string, number])].filter(([w]) => breaker(w)).map(([, id]) => id));
  let minLine: { at: number; tempered: boolean } | undefined, tempered = false;
  if (miro) {
    const t = f(val("temperature")), tau = f(val("mirostat_tau")), mu = f(opts.mu ?? f(2 * tau));
    if (t !== 1) scale(t);
    const { p } = softmax(), finite = c.filter((x) => x.l !== -Infinity).length;
    let n = c.length;
    if (val("mirostat") === 1) {
      let tb = 0, tt = 0;
      for (let i = 0; i < 99 && i < finite - 1; i++) {
        const ti = f(Math.log(f(f(i + 2) / f(i + 1)))), bi = f(Math.log(f(p[i] / p[i + 1])));
        tb = f(tb + f(ti * bi));
        tt = f(tt + f(ti * ti));
      }
      const s = f(tb / tt), eps = f(s - 1), vocab = opts.vocab ?? cands.length;
      const k = f(Math.pow(f(f(eps * f(Math.pow(2, mu))) / f(1 - f(Math.pow(vocab, -eps)))), f(1 / s)));
      n = Math.min(n, k >= 1 && k < 2147483648 ? Math.floor(k) : 1);
    } else {
      let m = 0;
      while (m < n && !(f(-Math.log2(p[m])) > mu)) m++;
      n = Math.max(m, 1);
    }
    keep(n, "mirostat");
    const q = softmax().p;
    c.forEach((x, i) => { rows[x.r].p = q[i]; });
    return { rows, minLine };
  }
  for (const s of stages) {
    if (s === "penalties") {
      if (!penN) continue;
      const rp = f(val("repeat_penalty")), fp = f(val("frequency_penalty")), pp = f(val("presence_penalty"));
      for (const x of c) {
        const k = counts.get(cands[x.r].id) ?? 0;
        if (!k) continue;
        const before = x.l;
        x.l = x.l <= 0 ? f(x.l * rp) : f(x.l / rp);
        x.l = f(x.l - f(f(k * fp) + f((k > 0 ? 1 : 0) * pp)));
        if (Number.isFinite(before)) rows[x.r].penalty += before - x.l;
      }
      sort();
    } else if (s === "dry") {
      const A = val("dry_allowed_length");
      if (!dryN) continue;
      const { N, limit, lo, count } = repeats(ids, (i) => lone.has(ids[i]), dryN);
      if (N <= A || limit < A) continue;
      const rep = new Map<number, number>();
      for (let i = 0; i < N - 1; i++) if (count[i] >= A) rep.set(ids[lo + i + 1], Math.max(rep.get(ids[lo + i + 1]) ?? 0, count[i]));
      const base = f(val("dry_base")), mult = f(val("dry_multiplier"));
      const maxExp = base > 1.000001 ? Math.trunc(f(f(88.7228391) / f(Math.log(base)))) : 0;
      for (const x of c) {
        const id = cands[x.r].id, r = rep.get(id);
        if (r === undefined || lone.has(id)) continue;
        const e = maxExp > 0 && r - A > maxExp ? maxExp : r - A, before = x.l;
        x.l = f(x.l - f(mult * Math.pow(base, e)));
        if (Number.isFinite(before)) rows[x.r].penalty += before - x.l;
      }
      sort();
    } else if (s === "top_n_sigma") {
      const sigma = f(val("top_n_sigma"));
      if (sigma <= 0 || c.length <= 1) continue;
      let mx = c[0].l, sum = 0, cnt = 0, acc = 0;
      for (const x of c) if (x.l !== -Infinity) { mx = Math.max(mx, x.l); sum = f(sum + x.l); cnt++; }
      const mean = cnt > 0 ? f(sum / cnt) : 0;
      for (const x of c) if (x.l !== -Infinity) { const d = f(x.l - mean); acc = f(acc + d * d); }
      const sd = cnt > 0 ? f(Math.sqrt(f(acc / cnt))) : 0, thr = f(mx - f(sigma * sd));
      for (const x of c) if (x.l < thr && x.l !== -Infinity) kill(x, "top_n_sigma");
    } else if (s === "top_k") {
      const k = val("top_k");
      if (k > 0 && k < c.length) keep(k, "top_k");
    } else if (s === "typ_p") {
      const tp = f(val("typical_p"));
      if (tp >= 1) continue;
      const { p } = softmax(), h = entropy(p);
      const score = p.map((y) => f(Math.abs(f(f(-Math.log(y)) - h))));
      const idx = p.map((_, i) => i).sort((a, b) => score[a] - score[b] || a - b);
      let acc = 0, last = c.length;
      for (let i = 0; i < idx.length; i++) {
        acc = f(acc + p[idx[i]]);
        if (acc > tp && (minKeep === 0 || i >= minKeep - 1)) { last = i + 1; break; }
      }
      const kept = new Set(idx.slice(0, last));
      c.forEach((x, i) => { if (!kept.has(i)) rows[x.r].cut ||= "typ_p"; });
      c = c.filter((_, i) => kept.has(i));
    } else if (s === "top_p") {
      const tp = f(val("top_p"));
      if (tp >= 1) continue;
      const { p } = softmax();
      let acc = 0;
      for (let i = 0; i < c.length; i++) {
        acc = f(acc + p[i]);
        if (acc >= tp && i + 1 >= minKeep) { keep(i + 1, "top_p"); break; }
      }
    } else if (s === "min_p") {
      const mp = f(val("min_p"));
      if (mp <= 0 || !c.length) continue;
      const thr = f(c[0].l + f(Math.log(mp)));
      let n = 0;
      while (n < c.length && c[n].l >= thr) n++;
      keep(Math.min(c.length, Math.max(n, 1, minKeep)), "min_p");
      minLine = { at: mp, tempered };
    } else if (s === "xtc") {
      const xp = f(val("xtc_probability")), xt = f(val("xtc_threshold"));
      if (xp <= 0 || xt > 0.5 || c.length < 2 || !(opts.xtc ?? xp > 0)) continue;
      const { p } = softmax();
      let last = 0;
      for (let i = 0; i < c.length && p[i] >= xt; i++) last = i;
      if (c.length - last >= minKeep && last > 0) {
        for (const x of c.slice(0, last)) rows[x.r].cut ||= "xtc";
        c = c.slice(last);
      }
    } else {
      const t = f(val("temperature")), range = f(val("dynatemp_range"));
      if (range > 0) {
        if (c.length <= 1) continue;
        const maxEnt = f(-Math.log(f(1 / c.length))), h = entropy(softmax().p);
        const lo = Math.max(0, f(t - range)), hi = f(t + range);
        scale(f(lo + f(f(hi - lo) * f(Math.pow(f(h / maxEnt), f(val("dynatemp_exponent")))))));
        tempered = true;
      } else if (t !== 1) {
        scale(t);
        tempered = true;
      }
    }
  }
  const m = c[0].l, e = c.map((x) => f(Math.exp(f(x.l - m)))), sum = e.reduce((a, y) => a + y, 0);
  c.forEach((x, i) => { rows[x.r].p = f(e[i] / sum); });
  return { rows, minLine };
}
