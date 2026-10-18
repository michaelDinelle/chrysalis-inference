// kiln's sampler (make_dist and dry_runs in engine/sampler.h) as Settings' odds charts run it on named candidates.
// web/sampler.test.ts holds it to the C++ one.
export type Stage = "top_k" | "temperature" | "top_p" | "min_p";
export type SamplerKey = Stage | "presence_penalty" | "dry_multiplier" | "dry_allowed_length" | "dry_base";
export interface Cand { word: string; logit: number; id: number }
export interface Odds { word: string; raw: number; p: number; cut: Stage | ""; pres: number; dry: number }
const breaker = (t: string) => /[\n:"*]/.test(t);    // DRY's sequence breakers (the server's set)

// DRY's runs (dry_runs in engine/sampler.h): for each token that followed an earlier occurrence of the context's last
// token, how many tokens before it match the end of the context; and where the longest run sits, to show it.
export function dryRuns(ctx: [string, number][]) {
  const runs = new Map<number, number>(), n = ctx.length;
  let best: { run: number; at: number } | null = null;
  if (n < 2 || breaker(ctx[n - 1][0])) return { runs, best };
  for (let i = n - 2; i >= 0; i--) {
    if (ctx[i][1] !== ctx[n - 1][1] || breaker(ctx[i + 1][0])) continue;
    let run = 1;
    while (run < 48 && i >= run && ctx[i - run][1] === ctx[n - 1 - run][1] && !breaker(ctx[i - run][0])) run++;
    const next = ctx[i + 1][1];
    runs.set(next, Math.max(runs.get(next) ?? 0, run));
    if (!best || run > best.run) best = { run, at: i };
  }
  return { runs, best };
}

// The sampler on a set of candidates after a reply so far: penalties, then the stages in their order. Each
// candidate's odds before (the model's) and after, what cut it, and its penalties; and min P's line.
export function sample(cands: Cand[], reply: [string, number][], v: (k: SamplerKey) => number, stages: readonly Stage[]) {
  const T = v("temperature");
  const seen = new Set(reply.map(([, id]) => id)), { runs } = dryRuns(reply);
  const top = Math.max(...cands.map((c) => c.logit)), z = cands.reduce((a, c) => a + Math.exp(c.logit - top), 0);
  const rows: Odds[] = cands.map((c) => {
    const run = runs.get(c.id) ?? 0, allowed = v("dry_allowed_length");
    const dry = v("dry_multiplier") > 0 && run >= allowed ? v("dry_multiplier") * v("dry_base") ** (run - allowed) : 0;
    return { word: c.word, raw: Math.exp(c.logit - top) / z, p: 0, cut: "", pres: seen.has(c.id) ? v("presence_penalty") : 0, dry };
  });
  const x = cands.map((c, i) => c.logit - rows[i].dry - rows[i].pres);
  const order = rows.map((_, i) => i).sort((a, b) => x[b] - x[a]);
  let n = order.length, t = 1, minLine: { at: number; tempered: boolean } | undefined;
  if (T <= 0) {
    order.forEach((r, j) => { if (j) rows[r].cut = "temperature"; });
    rows[order[0]].p = 1;
    return { rows, minLine };
  }
  const probs = (temp: number) => {
    const e = order.slice(0, n).map((r) => Math.exp((x[r] - x[order[0]]) / temp)), sum = e.reduce((a, y) => a + y, 0);
    return e.map((y) => y / sum);
  };
  const cut = (to: number, stage: Stage) => { for (let j = to; j < n; j++) rows[order[j]].cut ||= stage; n = to; };
  for (const s of stages) {
    if (s === "top_k") { const k = v("top_k"); cut(Math.max(1, Math.min(k > 0 ? k : n, n)), s); }
    else if (s === "temperature") t = T;
    else if (s === "top_p") {
      const P = probs(t);
      let cum = 0, keep = n;
      for (let j = 0; j < n; j++) { cum += P[j]; if (cum >= v("top_p")) { keep = j + 1; break; } }
      cut(keep, s);
    } else {
      const P = probs(t), min = v("min_p");
      let m = 1;
      while (m < n && P[m] >= min * P[0]) m++;
      cut(m, s);
      if (min > 0) minLine = { at: min, tempered: t === T };
    }
  }
  probs(T).forEach((p, j) => { rows[order[j]].p = p; });
  return { rows, minLine };
}
