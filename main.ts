// gta6-twitch-ccu-forecast-01 entry (Node 22, standard library only).
//
// Forecast: per-hour kernel-weighted log-log regression across comparable launches.
// A launch's hourly Twitch viewers are predicted from its hour-1 print: for each
// comparable launch, log(viewers at hour h) is regressed on log(viewers at hour 1),
// weighting launches by how close their hour-1 size is to this case's (Gaussian
// kernel, bandwidth 1.2 in log units). The peak of a launch day is nearly
// scale-free (a small launch rises many-fold, a huge one barely rises), so a plain
// ratio-to-hour-1 model misleads; the local regression learns that slope per hour.
// Quantiles are the fitted log-mean plus/minus the normal quantiles times 0.7x the
// weighted residual spread (floored at 0.10, capped at 0.90), rounded to 2 decimals.
//
// Reads every <id>.in in the input directory and writes <id>.out to the output
// directory: {"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}, 23 rows.
// Offline, deterministic, standard library only.
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Kernel bandwidth for the local regression, in natural-log units of hour-1 viewers. */
const BANDWIDTH = 1.2;
/** The fitted spread is scaled by this much: the raw residual spread overstates how
 *  far a big console launch strays from its neighbours. */
const SPREAD_SCALE = 0.7;
/** Floor and cap for the log spread of an hour's forecast. */
const SPREAD_MIN = 0.1;
const SPREAD_MAX = 0.9;
/** Fallback spread (persistence) when too few comparables cover an hour. */
const FALLBACK_SIGMA = 0.3;

interface CaseInput {
  levels: number[];
  game: { name: string };
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Array<{
    name: string;
    day_start: string;
    unlock: string;
    hourly: Array<{ from: string; viewers: number }>;
  }>;
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/**
 * Gaussian-kernel weighted least squares of y on x, centred at x0.
 * Returns the fitted value at x0 and the weighted residual spread.
 */
function wols(points: Array<[number, number]>, x0: number): { mu: number; s: number } {
  const w = points.map(([x]) => Math.exp(-((x - x0) ** 2) / (2 * BANDWIDTH * BANDWIDTH)));
  let sw = 0;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < points.length; i++) {
    sw += w[i];
    mx += w[i] * points[i][0];
    my += w[i] * points[i][1];
  }
  mx /= sw;
  my /= sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < points.length; i++) {
    sxx += w[i] * (points[i][0] - mx) ** 2;
    sxy += w[i] * (points[i][0] - mx) * (points[i][1] - my);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  const a = my - b * mx;
  let ss = 0;
  for (let i = 0; i < points.length; i++) {
    const r = points[i][1] - (a + b * points[i][0]);
    ss += w[i] * r * r;
  }
  return { mu: a + b * x0, s: Math.sqrt(ss / sw) };
}

/** Keep a quantile row non-decreasing and inside (0, 1e9], whatever rounding does. */
function finalize(qs: number[]): number[] {
  const out = qs.map((q) => Math.max(0.01, Math.min(1e9, q)));
  for (let j = 1; j < out.length; j++) if (out[j] < out[j - 1]) out[j] = out[j - 1];
  return out;
}

/** The five quantiles of one hour, from a log-mean and a log spread. */
function hourQuantiles(mu: number, s: number): number[] {
  const spread = Math.max(SPREAD_MIN, Math.min(SPREAD_MAX, s));
  return finalize(Z.map((z) => round2(Math.exp(mu + z * spread))));
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseInput;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${file} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);
  const x0 = Math.log(p1);
  const targetName = (c.game.name ?? '').toLowerCase();

  // Per-hour (x = log hour-1 viewers, y = log viewers at that hour) across launches.
  const points: Array<Array<[number, number]>> = Array.from({ length: 25 }, () => []);
  for (const cp of c.comparables) {
    if ((cp.name ?? '').toLowerCase() === targetName) continue;
    if ((cp.unlock ?? '').startsWith('Not a launch')) continue;
    const dayStart = Date.parse(cp.day_start);
    if (!Number.isFinite(dayStart)) continue;
    const byHour = new Map<number, number>();
    for (const p of cp.hourly ?? []) {
      const h = Math.round((Date.parse(p.from) - dayStart) / 3600000) + 1;
      if (h >= 1 && h <= 24 && typeof p.viewers === 'number' && p.viewers > 0) byHour.set(h, p.viewers);
    }
    const v1 = byHour.get(1);
    if (v1 === undefined) continue;
    const x = Math.log(v1);
    for (let h = 2; h <= 24; h++) {
      const v = byHour.get(h);
      if (v !== undefined) points[h].push([x, Math.log(v)]);
    }
  }

  const hours = c.forecast.map(({ hour }) => {
    const pts = points[hour];
    if (pts.length < 5) {
      // Too few launches cover this hour: fall back to widening persistence.
      return {
        hour,
        q: finalize(Z.map((z) => round2(p1 * Math.exp(z * FALLBACK_SIGMA * Math.sqrt(hour - 1))))),
      };
    }
    const { mu, s } = wols(pts, x0);
    return { hour, q: hourQuantiles(mu, s * SPREAD_SCALE) };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
